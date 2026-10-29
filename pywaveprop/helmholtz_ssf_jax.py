"""
JAX-based 2D split-step Fourier (SSF) propagator for the Helmholtz equation.

This is the spectral counterpart of :class:`pywaveprop.helmholtz_jax.RationalHelmholtzPropagator`
and exposes the same high-level interface (``create`` / ``compute`` / grid
accessors / pytree registration), so the two can be swapped in the RWP
front-end. The reduced field ``u = psi * exp(-i beta x)`` is advanced in range
with the symmetric (Strang) split

    u(x + dx) = P_half * F^{-1}[ exp(i dx (sqrt(beta^2 - p^2) - beta)) F[P_half * u] ],
    P_half    = exp(i dx (k(z) - beta) / 2),

where ``F`` is the transform matched to the lower boundary condition:

* ``'dirichlet'`` (PEC, horizontal polarization): sine transform (DST-I);
* ``'neumann'``   (PEC, vertical polarization):   cosine transform (DCT-I);
* ``'impedance'`` (finite-permittivity ground, Leontovich condition
  ``du/dz + alpha u = 0``): the discrete mixed Fourier transform (DMFT) of
  Dockery & Kuttler (1996), which propagates the sine spectrum of
  ``w_j = (u_{j+1} - u_{j-1}) / (2 dz) + alpha u_j`` plus the coefficients of the
  two discrete surface-wave modes ``r^j`` and ``(-1/r)^j``
  (``r^2 + 2 alpha dz r - 1 = 0``, ``|r| < 1``; the second mode is attached to
  the top of the domain and is negligible there thanks to the absorbing layer).
* ``'impedance_cf'`` (finite-permittivity ground, *angle-dependent* impedance):
  the consistent-field formulation of Sprouse & Awadallah (2012), see
  :class:`ConsistentFieldBoundary`. Every plane-wave component obeys the exact
  (Fresnel) impedance condition ``du/dz + i alpha(p) u = 0`` with
  ``alpha(p) = sqrt((eps - 1) k0^2 + p^2) [/ eps for V-pol]``, so there is no
  singularity at the Brewster angle and no grazing-angle impedance assumption.

All transforms are evaluated with complex FFTs of the odd / even extension of
the field, so only ``jnp.fft`` is required.

The upper boundary is handled with an absorbing layer (cos^2 taper) appended
above ``z_size_m``; the lower terrain and the optional lower slab
(:class:`~pywaveprop.helmholtz_jax.LowerLayerModel`) are treated exactly like
in the rational propagator, i.e. as a staircase mask and a range-dependent
heterogeneity applied at every range step.
"""
import cmath as cm
import math as fm

import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util

jax.config.update("jax_enable_x64", True)

from pywaveprop.helmholtz_common import HelmholtzMeshParams2D
from pywaveprop.helmholtz_jax import RegularGrid, LowerLayerModel


def impedance_alpha(k0: float, eps_complex: complex, polarization: str) -> complex:
    """Leontovich impedance coefficient ``alpha`` in ``du/dz + alpha u = 0`` at the
    ground for the ``exp(-i omega t)`` convention (``Im eps > 0`` is lossy).

    Horizontal polarization: ``alpha = i k0 sqrt(eps - 1)``;
    vertical polarization:   ``alpha = i k0 sqrt(eps - 1) / eps``.
    """
    s = cm.sqrt(eps_complex - 1.0)
    if s.imag < 0:
        s = -s
    if polarization.upper() == 'H':
        return 1j * k0 * s
    return 1j * k0 * s / eps_complex


class DMFTBoundary:
    """Discrete mixed Fourier transform (Dockery & Kuttler, 1996) for the
    Leontovich condition ``du/dz + alpha u = 0`` on a grid of ``n + 1`` points
    ``z_j = j dz`` (``j = 0`` on the ground).

    ``forward`` maps the field to the auxiliary variable
    ``w_j = (u_{j+1} - u_{j-1}) / (2 dz) + alpha u_j`` (``j = 1 .. n-1``), which
    obeys a Dirichlet condition, its sine-series coefficients ``c_m``, and the
    amplitudes ``a``, ``b`` of the two discrete surface-wave modes ``r^j`` and
    ``(-1/r)^j`` (``r^2 + 2 alpha dz r - 1 = 0``, ``|r| < 1``). ``inverse`` rebuilds
    the field from propagated sine coefficients and mode amplitudes; the mode
    amplitudes are propagated analytically by ``propagate_surface``. The pair is
    exactly invertible, so any Dirichlet-type propagation of ``w`` (spectral,
    wavelet, ...) can be used in between.
    """

    def __init__(self, alpha: complex, dz: float, n_points: int, beta: float, dx: float):
        self.alpha = complex(alpha)
        self.dz = float(dz)
        self.n = int(n_points) - 1
        n, dz = self.n, self.dz
        disc = cm.sqrt(self.alpha ** 2 * dz ** 2 + 1.0)
        r1, r2 = -self.alpha * dz + disc, -self.alpha * dz - disc
        self.r = r1 if abs(r1) < abs(r2) else r2
        r = self.r
        j = np.arange(n + 1)
        self.r_pow = jnp.asarray(r ** j)
        self.r_pow_top = jnp.asarray((-1.0) ** j * r ** (n - j))
        r_n = r ** n
        self._end_det = 1.0 / (1.0 - r_n ** 2)
        self._r_n = r_n
        self._r_n_signed = (-1.0) ** n * r_n

        def mode_mult(root):
            alpha_d = -cm.log(root) / dz  # decay constant of the discrete mode root^j
            kx_sw = cm.sqrt(beta ** 2 + alpha_d ** 2)
            if kx_sw.imag < 0:
                kx_sw = -kx_sw
            return jnp.asarray(cm.exp(1j * dx * (kx_sw - beta)), dtype=complex)

        self.surface_mult = mode_mult(r)
        self.surface_mult_top = mode_mult(-1.0 / r)
        th = np.pi * np.arange(1, n) / n
        denom = self.alpha ** 2 * dz ** 2 + np.sin(th) ** 2
        self._sin_coef = jnp.asarray(self.alpha * dz ** 2 / denom)
        self._cos_coef = jnp.asarray(-dz * np.sin(th) / denom)
        self._alt_sign = jnp.asarray((-1.0) ** np.arange(1, n))

    def w_of(self, u):
        """Auxiliary Dirichlet variable ``w_j`` (``j = 1 .. n-1``); operates along the last axis."""
        return (u[..., 2:] - u[..., :-2]) / (2 * self.dz) + self.alpha * u[..., 1:-1]

    def sine_coefs(self, w):
        """Sine-series coefficients ``c_m`` (``m = 1 .. n-1``) of the interior values ``w``
        (one FFT of the odd extension; batched along leading axes)."""
        n = self.n
        zero = jnp.zeros(w.shape[:-1] + (1,), dtype=complex)
        w_ext = jnp.concatenate((zero, w, zero, -w[..., ::-1]), axis=-1)
        return (1j * jnp.fft.fft(w_ext, axis=-1) / n)[..., 1:n]

    def inverse_sine_part(self, c):
        """Field on the ``n + 1`` grid points generated by the sine coefficients ``c``
        (the two kernels ``sin(th j)`` and ``cos(th j)`` are evaluated with a single
        FFT of the odd plus even extension, separated by their parity in frequency)."""
        n = self.n
        cs = c * self._sin_coef
        cc = c * self._cos_coef
        zero = jnp.zeros(c.shape[:-1] + (1,), dtype=complex)
        x_ext = jnp.concatenate((zero, cs + cc, zero, (cc - cs)[..., ::-1]), axis=-1)
        spec = jnp.fft.fft(x_ext, axis=-1)
        spec_neg = jnp.concatenate((spec[..., :1], spec[..., :0:-1]), axis=-1)  # spec[-k mod 2n]
        odd_part = (spec - spec_neg) / 2   # = fft of the odd extension of cs
        even_part = (spec + spec_neg) / 2  # = fft of the even extension of cc
        return (1j * odd_part[..., :n + 1] + even_part[..., :n + 1]) / 2

    def end_values(self, c):
        """Values at ``j = 0`` and ``j = n`` of :meth:`inverse_sine_part` (two dot products)."""
        cc = c * self._cos_coef
        return jnp.sum(cc, axis=-1), jnp.sum(cc * self._alt_sign, axis=-1)

    def forward(self, u):
        """Return ``(w, c, a, b)`` for the field ``u`` on the ``n + 1`` grid points
        (``u`` may carry leading batch axes)."""
        w = self.w_of(u)
        c = self.sine_coefs(w)
        s_0, s_n = self.end_values(c)
        e_0 = u[..., 0] - s_0
        e_n = u[..., -1] - s_n
        a = (e_0 - self._r_n_signed * e_n) * self._end_det
        b = (e_n - self._r_n * e_0) * self._end_det
        return w, c, a, b

    def propagate_surface(self, a, b):
        return a * self.surface_mult, b * self.surface_mult_top

    def inverse(self, c, a, b):
        a = jnp.asarray(a)[..., None]
        b = jnp.asarray(b)[..., None]
        return self.inverse_sine_part(c) + a * self.r_pow + b * self.r_pow_top


class ConsistentFieldBoundary:
    """Consistent-field angle-dependent impedance boundary condition of
    Sprouse & Awadallah, "An angle-dependent impedance boundary condition for
    the split-step parabolic equation method", IEEE TAP 60(2), 2012.

    The field is carried on the imaged grid ``z_k = k dz``, ``k = -(n-1) .. n``
    (FFT ordering: indices ``0 .. n`` are the physical points above the
    ground, ``n+1 .. 2n-1`` the points below it). With the transform pair
    ``u(z) = 1/(2 pi) int u~(p) exp(i p z) dp``, a field is consistent with the
    plane-wave impedance condition ``du/dz + i alpha(p) u = 0``,

        (p + alpha) u~(p) = (p - alpha) u~(-p),                          (eq. 33)
        alpha(p) = sqrt((eps - 1) k0^2 + p^2)  [/ eps for V-pol],  Re alpha >= 0,

    iff ``u~ = (1 - p / alpha) v~`` with ``v`` odd in ``z`` (eq. 35). The
    below-ground part of a consistent field is the continuation that carries
    the reflected waves, so free-space propagation of the whole line is exact
    above the ground. ``v~ = u~ / (1 - p / alpha)`` is singular at the V-pol
    Brewster root ``p0`` (``alpha(p0) = p0``), so a diffraction step
    (appendix, steps 3-8) splits ``u~ = u~_s + u~_f`` with a weight
    ``g = u~_f / u~`` concentrated at the Brewster root, represents the surface
    part by ``v~_s = u~_s / (1 - p / alpha)``, odd-extends ``v_s`` from
    ``z >= 0``, propagates ``v_s`` and ``u_f`` with the free-space multiplier
    and recombines them with eq. 63, ``u~ = (1 - p / alpha) v~_s + u~_f``.

    ``split`` selects ``g``:

    * ``'even'`` (default): ``g = exp(-|1 - |p| / alpha(p)|^2 / delta^2)``.
      Since ``g`` is even in ``p``, multiplication by ``g`` preserves eq. 33:
      both parts of a consistent field stay consistent, ``v_s`` is exactly odd
      (the odd extension changes nothing) and ``u_f`` needs no taper below the
      ground. The step is then exact for consistent fields for any ``delta``,
      which only controls how inconsistent content (e.g. a source radiating
      upwards at the Brewster angle, which cannot be represented by eq. 35)
      is handled. Measuring the distance to the root relative to ``alpha``
      keeps ``g`` negligible over a lossy ground (e.g. sea water), where
      ``1 - p / alpha`` has no real root and no splitting is needed.
    * ``'paper'``: eqs. 44-45 as published,
      ``g = -i k0 delta / (alpha - p - i k0 delta)`` (eq. 55 has a sign typo;
      with eq. 45 the parts sum to ``u~``), with ``u_f`` tapered to zero below
      the ground over ``taper_cells`` cells (eq. 60, the paper uses 50). This
      ``g`` is not even and has a slowly decaying Lorentzian tail, so the odd
      extension of ``v_s`` and the taper of ``u_f`` introduce an ``O(delta)``
      error in the reflected field.

    ``taper_cells=None`` gives 0 for ``'even'`` and 50 for ``'paper'``. The
    p-space window of step 9 (eq. 52 applied in ``p``) is applied at the
    recombination. ``k0`` is the wavenumber just above the ground and
    ``delta`` the splitting width (the paper uses 0.1; ``delta = 0`` is
    admissible for H-pol, where there is no Brewster root).
    """

    def __init__(self, eps_complex: complex, polarization: str, k0: float, dz: float, n_points: int,
                 delta: float = 0.1, split: str = 'even', taper_cells: int = None):
        if split not in ('even', 'paper'):
            raise ValueError(f"unknown split {split!r}; expected 'even' or 'paper'")
        self.n = int(n_points) - 1
        n = self.n
        n_ext = 2 * n
        eps = complex(eps_complex)
        p = 2 * np.pi * np.fft.fftfreq(n_ext, dz)
        alpha = np.sqrt((eps - 1.0) * k0 ** 2 + p.astype(complex) ** 2)  # principal branch, Re >= 0
        if polarization.upper() == 'V':
            alpha = alpha / eps
        diff = alpha - p
        if split == 'paper':
            den = diff - 1j * k0 * delta
            singular = np.abs(den) < 1e-9 * k0
            f_mult = -1j * k0 * delta / np.where(singular, 1.0, den)
            v_mult = alpha / np.where(singular, 1.0, den)
        else:
            f_mult = np.exp(-np.abs(1 - np.abs(p) / alpha) ** 2 / delta ** 2) if delta > 0 else np.zeros(n_ext)
            singular = (np.abs(diff) < 1e-9 * k0) & (f_mult < 1 - 1e-12)
            v_mult = alpha * (1 - f_mult) / np.where(np.abs(diff) < 1e-9 * k0, 1.0, diff)
        if np.any(singular):
            raise ValueError("consistent-field splitting is singular (Brewster root on the grid); use delta > 0")
        if taper_cells is None:
            taper_cells = 50 if split == 'paper' else 0
        self.v_mult = jnp.asarray(v_mult, dtype=complex)
        self.f_mult = jnp.asarray(f_mult, dtype=complex)
        pa = np.abs(p) / np.max(np.abs(p))
        p_window = np.where(pa <= 0.8, 1.0, 0.5 + 0.5 * np.cos(5 * np.pi * pa - 4 * np.pi))
        self.recomb = jnp.asarray((1.0 - p / alpha) * p_window)
        self.p_window = jnp.asarray(p_window, dtype=complex)
        z_ext = dz * np.concatenate((np.arange(n + 1), np.arange(-(n - 1), 0)))
        if taper_cells > 0:
            h = int(taper_cells) * dz
            self.taper = jnp.asarray(np.where(z_ext >= 0, 1.0,
                                              np.where(z_ext > -h, 0.5 + 0.5 * np.cos(np.pi * z_ext / h), 0.0)))
        else:
            self.taper = None

    def to_ext(self, u):
        """Physical field (``n + 1`` points) -> imaged field; zero below the ground."""
        return jnp.concatenate((u, jnp.zeros(self.n - 1, dtype=u.dtype)))

    def mirror(self, w):
        """Even extension of a physical-grid profile onto the imaged grid."""
        return jnp.concatenate((w, w[-2:0:-1]))

    def step(self, u_ext, spectral_mult):
        """Diffraction step over ``dx`` (``spectral_mult`` = free-space multiplier
        on the length ``2 n`` FFT grid)."""
        n = self.n
        u_hat = jnp.fft.fft(u_ext)
        vs = jnp.fft.ifft(u_hat * self.v_mult)[:n + 1]
        vs = jnp.concatenate((vs[:1] * 0, vs[1:n], -vs[n:0:-1]))  # odd extension, v(0) = 0
        if self.taper is None:
            vs_hat = jnp.fft.fft(vs)
            uf_hat = u_hat * self.f_mult
        else:
            uf = jnp.fft.ifft(u_hat * self.f_mult) * self.taper
            vs_hat, uf_hat = jnp.fft.fft(jnp.stack((vs, uf)), axis=-1)
        return jnp.fft.ifft(spectral_mult * (self.recomb * vs_hat + self.p_window * uf_hat))


class SSFHelmholtzPropagator:
    """Split-step Fourier propagator, see the module docstring."""

    @classmethod
    def create(cls, freq_hz: float, wave_speed, kz_max, k_bounds, precision, mesh_params: HelmholtzMeshParams2D,
               lower_terrain=None, lower_bc: str = 'dirichlet', lower_impedance_alpha=None,
               lower_layer: LowerLayerModel = None, abs_layer_scale: float = 0.5,
               dx_m: float = None, dz_m: float = None, spectral_margin: float = 2.0,
               absorber: str = 'window', abs_attenuation_db: float = 40.0,
               lower_eps=None, polarization: str = 'H', cf_delta: float = 0.1, cf_taper_cells: int = None,
               cf_split: str = 'even', **_ignored):
        """Build a propagator with the same conventions as
        :meth:`RationalHelmholtzPropagator.create`.

        ``kz_max`` fixes the vertical grid: ``dz <= pi / (spectral_margin * kz_max)``
        so that the sine/cosine spectrum covers ``spectral_margin`` times the
        maximum propagation angle. ``dx`` defaults to the paraxial phase scale
        ``2 pi / (beta sin^2(theta_max))`` (a plane wave at ``theta_max`` then
        accumulates one cycle of the spectral phase per step). Both can be
        overridden explicitly with ``dx_m`` / ``dz_m``; ``precision`` is accepted
        for interface compatibility only. The computational steps are always
        integer divisors of the requested output steps, exactly like in the
        rational propagator.

        ``absorber`` selects the upper absorbing layer: ``'window'`` (default)
        multiplies the field by a cos^2 window after every step, ``'profile'``
        adds a smooth imaginary part ``i s(z)``, ``s = s0 t^2``, to the
        heterogeneity so that the phase screen attenuates by ``abs_attenuation_db``
        (one way, at ``theta_max``) independently of the range step.

        ``lower_bc='impedance_cf'`` selects the consistent-field angle-dependent
        impedance condition (:class:`ConsistentFieldBoundary`); it needs the
        complex ground permittivity ``lower_eps`` and the ``polarization``
        (``'H'`` / ``'V'``); ``cf_split``, ``cf_delta`` and ``cf_taper_cells`` are
        its splitting weight (``'even'`` or the published ``'paper'`` one),
        splitting width and below-ground taper length. Only flat ground is
        supported.
        """
        kz_max = float(kz_max)
        k_bounds = (float(k_bounds[0]), float(k_bounds[1]))
        beta = 0.5 * (k_bounds[0] + k_bounds[1])

        dx_max = mesh_params.dx_output_m or mesh_params.x_size_m / (round(mesh_params.x_n_upper_bound / 2) - 1)
        dz_max = mesh_params.dz_output_m or mesh_params.z_size_m / (round(mesh_params.z_n_upper_bound / 2) - 1)

        if dz_m is None:
            dz_m = min(dz_max, fm.pi / (spectral_margin * kz_max))
        if dx_m is None:
            sin_theta = min(max(kz_max / beta, 1e-3), 1.0)
            dx_m = min(dx_max, 2 * fm.pi / (beta * sin_theta ** 2))

        if mesh_params.dx_output_m:
            x_output_step = fm.ceil(mesh_params.dx_output_m / dx_m)
            dx_m = mesh_params.dx_output_m / x_output_step
        else:
            x_output_step = fm.floor(dx_max / dx_m)
            dx_m = dx_max / x_output_step
        if mesh_params.dz_output_m:
            z_output_step = fm.ceil(mesh_params.dz_output_m / dz_m)
            dz_m = mesh_params.dz_output_m / z_output_step
        else:
            z_output_step = fm.floor(dz_max / dz_m)
            dz_m = dz_max / z_output_step

        x_n = fm.ceil(mesh_params.x_size_m / dx_m) + 1
        z_n_phys = fm.ceil(mesh_params.z_size_m / dz_m) + 1
        z_n_abs = fm.ceil(abs_layer_scale * mesh_params.z_size_m / dz_m)

        abs_strength = 0.0
        if absorber == 'profile' and z_n_abs > 0:
            sin_theta = min(max(kz_max / beta, 1e-3), 1.0)
            abs_strength = 6.0 * (abs_attenuation_db / 20.0) * fm.log(10.0) * sin_theta / (beta * z_n_abs * dz_m)

        print(f'ssf: beta: {beta}, dx: {dx_m}, dz: {dz_m}, z_n: {z_n_phys + z_n_abs} (abs layer: {z_n_abs})')

        return cls(
            beta=beta,
            dx_m=dx_m,
            dz_m=dz_m,
            x_n=x_n,
            z_n=z_n_phys + z_n_abs,
            z_n_phys=z_n_phys,
            x_grid_scale=x_output_step,
            z_grid_scale=z_output_step,
            freq_hz=freq_hz,
            wave_speed=wave_speed,
            lower_terrain=lower_terrain,
            lower_bc=lower_bc,
            lower_impedance_alpha=lower_impedance_alpha,
            lower_layer=lower_layer,
            absorber=absorber,
            abs_strength=abs_strength,
            lower_eps=lower_eps,
            polarization=polarization,
            cf_delta=cf_delta,
            cf_taper_cells=cf_taper_cells,
            cf_split=cf_split,
        )

    def __init__(self, beta: float, dx_m: float, dz_m: float, x_n: int, z_n: int, z_n_phys: int,
                 x_grid_scale: int, z_grid_scale: int, freq_hz: float, wave_speed,
                 lower_terrain=None, lower_bc: str = 'dirichlet', lower_impedance_alpha=None,
                 lower_layer: LowerLayerModel = None, absorber: str = 'window', abs_strength: float = 0.0,
                 lower_eps=None, polarization: str = 'H', cf_delta: float = 0.1, cf_taper_cells: int = None,
                 cf_split: str = 'even'):
        if absorber not in ('window', 'profile'):
            raise ValueError(f"unknown absorber {absorber!r}; expected 'window' or 'profile'")
        if lower_bc not in ('dirichlet', 'neumann', 'impedance', 'impedance_cf'):
            raise ValueError(f"unknown lower_bc {lower_bc!r}; "
                             f"expected 'dirichlet', 'neumann', 'impedance' or 'impedance_cf'")
        if lower_bc == 'impedance' and lower_impedance_alpha is None:
            raise ValueError("lower_bc='impedance' requires lower_impedance_alpha")
        if lower_bc == 'impedance_cf':
            if lower_eps is None:
                raise ValueError("lower_bc='impedance_cf' requires lower_eps")
            if lower_terrain is not None:
                raise ValueError("lower_bc='impedance_cf' supports flat ground only")
        self.beta = float(beta)
        self.dx_m = float(dx_m)
        self.dz_m = float(dz_m)
        self.x_n = int(x_n)
        self.z_n = int(z_n)
        self.z_n_phys = int(z_n_phys)
        self.x_grid_scale = int(x_grid_scale)
        self.z_grid_scale = int(z_grid_scale)
        self.freq_hz = freq_hz
        self.wave_speed = wave_speed
        self.lower_terrain = lower_terrain
        self.lower_bc = lower_bc
        self.lower_impedance_alpha = None if lower_impedance_alpha is None else complex(lower_impedance_alpha)
        self.lower_layer = lower_layer
        self.absorber = absorber
        self.abs_strength = float(abs_strength)
        self.lower_eps = None if lower_eps is None else complex(lower_eps)
        self.polarization = polarization
        self.cf_delta = float(cf_delta)
        self.cf_taper_cells = cf_taper_cells
        self.cf_split = cf_split
        self._prepare_arrays()

    # ------------------------------------------------------------------ setup
    def _prepare_arrays(self):
        n = self.z_n - 1  # number of intervals; the odd/even extension has length 2 n
        self._n_ext = 2 * n
        z_grid = RegularGrid(start=0, dx=self.dz_m, n=self.z_n)
        # A range-dependent medium exposes ``columns_on_regular_grid``: the wave
        # speed sampled on ``n_cols`` equally spaced ranges spanning the domain.
        # Every range step then interpolates linearly between the two nearest
        # columns, so the environment costs O(n_cols * z_n) instead of
        # O(x_n * z_n).
        columns = getattr(self.wave_speed, 'columns_on_regular_grid', None)
        try:
            c_cols = None if columns is None else jnp.asarray(columns(z_grid))
        except AttributeError:          # the medium is range-independent
            c_cols = None
        if c_cols is not None:
            k_cols = 2 * jnp.pi * self.freq_hz / c_cols
            self.het_cols = jnp.array(k_cols ** 2 / self.beta ** 2 - 1.0, dtype=complex)
            self.het = self.het_cols[0]
            n_cols = self.het_cols.shape[0]
            x_size = (self.x_n - 1) * self.dx_m
            self.env_dx_m = x_size / max(n_cols - 1, 1)
        else:
            k_arr = 2 * jnp.pi * self.freq_hz / self.wave_speed.on_regular_grid(z_grid)
            self.het = jnp.array(k_arr ** 2 / self.beta ** 2 - 1.0, dtype=complex)
            self.het_cols = None
            self.env_dx_m = None

        # Static (grid-only) arrays are built with numpy: eager jnp arithmetic
        # would trigger one XLA compilation per operation for every new grid.
        # spectral propagator on the extended (length 2n) FFT grid
        m = np.arange(self._n_ext)
        m_sym = np.minimum(m, self._n_ext - m)
        p = np.pi * m_sym / (n * self.dz_m)
        kx = np.sqrt(self.beta ** 2 - p.astype(complex) ** 2)
        kx = np.where(kx.imag < 0, -kx, kx)  # evanescent waves must decay
        self._spectral_mult = jnp.asarray(np.exp(1j * self.dx_m * (kx - self.beta)))

        # cos^2 absorbing layer above the physical domain
        z = np.arange(self.z_n) * self.dz_m
        z_phys = (self.z_n_phys - 1) * self.dz_m
        z_top = (self.z_n - 1) * self.dz_m
        if self.z_n > self.z_n_phys:
            t = np.clip((z - z_phys) / (z_top - z_phys), 0.0, 1.0)
        else:
            t = np.zeros(self.z_n)
        if self.absorber == 'profile':
            self._abs_window = jnp.ones(self.z_n)
            absorb = 1j * self.abs_strength * t ** 2
            self.het = self.het + absorb
            if self.het_cols is not None:
                self.het_cols = self.het_cols + absorb[None, :]
        else:
            self._abs_window = jnp.asarray(np.cos(np.pi / 2 * t) ** 2)

        # terrain / slab geometry stored as O(x_n) index arrays (see helmholtz_jax)
        x_grid = RegularGrid(start=0, dx=self.dx_m, n=self.x_n)
        self._z_indices = jnp.arange(self.z_n)
        if self.lower_terrain is not None:
            self.lower_terrain_idx = jnp.round(
                self.lower_terrain.on_regular_grid(x_grid) / self.dz_m).astype(jnp.int64)
        else:
            self.lower_terrain_idx = jnp.zeros(self.x_n, dtype=jnp.int64)
        if self.lower_layer is not None:
            self.layer_idx = jnp.round(
                self.lower_layer.thickness.on_regular_grid(x_grid) / self.dz_m).astype(jnp.int64)
            self.layer_delta_het = jnp.asarray(self.lower_layer.k2_contrast / self.beta ** 2, dtype=complex)
        else:
            self.layer_idx = jnp.zeros(self.x_n, dtype=jnp.int64)
            self.layer_delta_het = jnp.asarray(0.0, dtype=complex)

        if self.lower_bc == 'impedance':
            self._dmft = DMFTBoundary(self.lower_impedance_alpha, self.dz_m, self.z_n, self.beta, self.dx_m)
            self._r = self._dmft.r
        if self.lower_bc == 'impedance_cf':
            # k0 just above the ground is taken as the reference wavenumber beta
            self._cf = ConsistentFieldBoundary(self.lower_eps, self.polarization, self.beta, self.dz_m, self.z_n,
                                               delta=self.cf_delta, taper_cells=self.cf_taper_cells,
                                               split=self.cf_split)

    # ------------------------------------------------------------------ grids
    def x_computational_grid(self):
        return jnp.arange(0, self.x_n) * self.dx_m

    def x_output_grid(self):
        return self.x_computational_grid()[::self.x_grid_scale]

    def z_computational_grid(self):
        """Full vertical grid including the absorbing layer."""
        return jnp.arange(0, self.z_n) * self.dz_m

    def z_physical_grid(self):
        return jnp.arange(0, self.z_n_phys) * self.dz_m

    def z_output_grid(self):
        return self.z_physical_grid()[::self.z_grid_scale]

    def terrain_mask_row(self, i):
        """Field mask (1 in the medium, 0 inside the terrain) at range step ``i``."""
        return jnp.where(self._z_indices >= self.lower_terrain_idx[i], 1.0 + 0.0j, 0.0 + 0.0j)

    def het_row(self, i):
        """Heterogeneity profile at range step ``i`` (includes the lower slab).

        For a range-dependent medium the profile is interpolated linearly
        between the two environment columns bracketing the range ``i * dx``.
        """
        if self.het_cols is None:
            het = self.het
        else:
            t = (i * self.dx_m) / self.env_dx_m
            n_cols = self.het_cols.shape[0]
            k = jnp.clip(jnp.floor(t).astype(jnp.int32), 0, n_cols - 2)
            w = jnp.clip(t - k, 0.0, 1.0).astype(self.het_cols.dtype)
            c0 = jnp.take(self.het_cols, k, axis=0)
            c1 = jnp.take(self.het_cols, k + 1, axis=0)
            het = (1.0 - w) * c0 + w * c1
        if self.lower_layer is None:
            return het
        lo = self.lower_terrain_idx[i]
        inside = (self._z_indices >= lo) & (self._z_indices < lo + self.layer_idx[i])
        return het + self.layer_delta_het * inside

    # ----------------------------------------------------------------- pytree
    def _tree_flatten(self):
        dynamic = (self.wave_speed, self.lower_terrain, self.lower_layer)
        static = {
            'beta': self.beta,
            'dx_m': self.dx_m,
            'dz_m': self.dz_m,
            'x_n': self.x_n,
            'z_n': self.z_n,
            'z_n_phys': self.z_n_phys,
            'x_grid_scale': self.x_grid_scale,
            'z_grid_scale': self.z_grid_scale,
            'freq_hz': self.freq_hz,
            'lower_bc': self.lower_bc,
            'lower_impedance_alpha': self.lower_impedance_alpha,
            'absorber': self.absorber,
            'abs_strength': self.abs_strength,
            'lower_eps': self.lower_eps,
            'polarization': self.polarization,
            'cf_delta': self.cf_delta,
            'cf_taper_cells': self.cf_taper_cells,
            'cf_split': self.cf_split,
        }
        return dynamic, static

    @classmethod
    def _tree_unflatten(cls, static, dynamic):
        return cls(wave_speed=dynamic[0], lower_terrain=dynamic[1], lower_layer=dynamic[2], **static)

    # ------------------------------------------------------------- transforms
    def _extend_odd(self, u):
        return jnp.concatenate((u[:-1], -u[-1:0:-1]))

    def _extend_even(self, u):
        return jnp.concatenate((u[:-1], u[-1:0:-1]))

    def _spectral_step_dirichlet(self, u):
        return jnp.fft.ifft(self._spectral_mult * jnp.fft.fft(self._extend_odd(u)))[:self.z_n]

    def _spectral_step_neumann(self, u):
        return jnp.fft.ifft(self._spectral_mult * jnp.fft.fft(self._extend_even(u)))[:self.z_n]

    def _spectral_step_impedance(self, u):
        n = self.z_n - 1
        _, c, a_sw, b_sw = self._dmft.forward(u)
        c = c * self._spectral_mult[1:n]
        a_sw, b_sw = self._dmft.propagate_surface(a_sw, b_sw)
        return self._dmft.inverse(c, a_sw, b_sw)

    def _spectral_step(self, u):
        if self.lower_bc == 'dirichlet':
            return self._spectral_step_dirichlet(u)
        if self.lower_bc == 'neumann':
            return self._spectral_step_neumann(u)
        return self._spectral_step_impedance(u)

    # -------------------------------------------------------------- stepping
    def _step(self, u, het):
        phase_half = jnp.exp(0.5j * self.dx_m * self.beta * (jnp.sqrt(1.0 + het) - 1.0))
        if self.lower_bc == 'impedance_cf':
            # the state lives on the imaged grid; refraction acts above the ground only
            phase_half = jnp.concatenate((phase_half, jnp.ones(self.z_n - 2, dtype=complex)))
            u = self._cf.step(phase_half * u, self._spectral_mult)
            return phase_half * u * self._cf.mirror(self._abs_window)
        u = phase_half * u
        u = self._spectral_step(u)
        u = phase_half * u
        return u * self._abs_window

    def _mask(self, u, i):
        if self.lower_bc == 'impedance_cf':
            return u  # flat ground only, see __init__
        return u * self.terrain_mask_row(i)

    @jax.jit
    def compute(self, initial):
        """Propagate ``initial`` (given on :meth:`z_computational_grid` or on
        :meth:`z_physical_grid`; it is zero-padded into the absorbing layer)
        and return the field on the output grid, shape
        ``(len(x_output_grid), len(z_output_grid))``."""
        initial = jnp.asarray(initial, dtype=complex)
        u0 = jnp.zeros(self.z_n, dtype=complex).at[:initial.shape[0]].set(initial[:self.z_n])
        u0 = u0 * self.terrain_mask_row(0)
        if self.lower_bc == 'impedance_cf':
            u0 = self._cf.to_ext(u0)  # field below the ground initialised to zero (paper, Sec. II-C)

        n_x_out = (self.x_n - 1) // self.x_grid_scale + 1
        n_z_out = (self.z_n_phys - 1) // self.z_grid_scale + 1
        results = jnp.zeros((n_x_out, n_z_out), dtype=complex)
        results = results.at[0, :].set(u0[:self.z_n_phys:self.z_grid_scale])

        def body_fun(i, val):
            u, res = val
            u = self._step(u, self.het_row(i))
            u = self._mask(u, i)
            res = jax.lax.cond(i % self.x_grid_scale == 0,
                               lambda: res.at[i // self.x_grid_scale, :].set(u[:self.z_n_phys:self.z_grid_scale]),
                               lambda: res)
            return u, res

        _, results = jax.lax.fori_loop(1, self.x_n, body_fun, (u0, results))
        return results


tree_util.register_pytree_node(SSFHelmholtzPropagator,
                               SSFHelmholtzPropagator._tree_flatten,
                               SSFHelmholtzPropagator._tree_unflatten)
