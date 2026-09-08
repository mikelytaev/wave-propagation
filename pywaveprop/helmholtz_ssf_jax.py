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


class SSFHelmholtzPropagator:
    """Split-step Fourier propagator, see the module docstring."""

    @classmethod
    def create(cls, freq_hz: float, wave_speed, kz_max, k_bounds, precision, mesh_params: HelmholtzMeshParams2D,
               lower_terrain=None, lower_bc: str = 'dirichlet', lower_impedance_alpha=None,
               lower_layer: LowerLayerModel = None, abs_layer_scale: float = 0.5,
               dx_m: float = None, dz_m: float = None, spectral_margin: float = 2.0,
               **_ignored):
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
        )

    def __init__(self, beta: float, dx_m: float, dz_m: float, x_n: int, z_n: int, z_n_phys: int,
                 x_grid_scale: int, z_grid_scale: int, freq_hz: float, wave_speed,
                 lower_terrain=None, lower_bc: str = 'dirichlet', lower_impedance_alpha=None,
                 lower_layer: LowerLayerModel = None):
        if lower_bc not in ('dirichlet', 'neumann', 'impedance'):
            raise ValueError(f"unknown lower_bc {lower_bc!r}; expected 'dirichlet', 'neumann' or 'impedance'")
        if lower_bc == 'impedance' and lower_impedance_alpha is None:
            raise ValueError("lower_bc='impedance' requires lower_impedance_alpha")
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
        self._prepare_arrays()

    # ------------------------------------------------------------------ setup
    def _prepare_arrays(self):
        n = self.z_n - 1  # number of intervals; the odd/even extension has length 2 n
        self._n_ext = 2 * n
        z_grid = RegularGrid(start=0, dx=self.dz_m, n=self.z_n)
        k_arr = 2 * jnp.pi * self.freq_hz / self.wave_speed.on_regular_grid(z_grid)
        self.het = jnp.array(k_arr ** 2 / self.beta ** 2 - 1.0, dtype=complex)

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
            self._abs_window = jnp.asarray(np.cos(np.pi / 2 * t) ** 2)
        else:
            self._abs_window = jnp.ones(self.z_n)

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
            alpha = self.lower_impedance_alpha
            dz = self.dz_m
            # discrete surface-wave mode r^j: r^2 + 2 alpha dz r - 1 = 0, |r| < 1
            disc = cm.sqrt(alpha ** 2 * dz ** 2 + 1.0)
            r1, r2 = -alpha * dz + disc, -alpha * dz - disc
            r = r1 if abs(r1) < abs(r2) else r2
            self._r = r
            j = np.arange(self.z_n)
            self._r_pow = jnp.asarray(r ** j)
            # The second null vector of the difference operator, (-1/r)^j, is
            # the mirror image of r^j attached to the top of the domain; it is
            # kept (normalised to its value at j = n) so that the transform
            # pair is exactly invertible.  Both amplitudes follow from the
            # residual at the two end points.
            self._r_pow_top = jnp.asarray((-1.0) ** j * r ** (n - j))
            r_n = r ** n
            self._end_det = 1.0 / (1.0 - r_n ** 2)
            self._r_n = r_n
            self._r_n_signed = (-1.0) ** n * r_n

            def mode_mult(root):
                alpha_d = -cm.log(root) / dz  # decay constant of the discrete mode root^j
                kx_sw = cm.sqrt(self.beta ** 2 + alpha_d ** 2)
                if kx_sw.imag < 0:
                    kx_sw = -kx_sw
                return jnp.asarray(cm.exp(1j * self.dx_m * (kx_sw - self.beta)), dtype=complex)

            self._surface_mult = mode_mult(r)
            self._surface_mult_top = mode_mult(-1.0 / r)
            # inverse DMFT kernels: phi_m(j) = dz [alpha dz sin(th j) - sin(th) cos(th j)] / (alpha^2 dz^2 + sin^2 th)
            th = np.pi * np.arange(1, n) / n
            denom = alpha ** 2 * dz ** 2 + np.sin(th) ** 2
            self._dmft_sin_coef = jnp.asarray(alpha * dz ** 2 / denom)
            self._dmft_cos_coef = jnp.asarray(-dz * np.sin(th) / denom)

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
        """Heterogeneity profile at range step ``i`` (includes the lower slab)."""
        if self.lower_layer is None:
            return self.het
        lo = self.lower_terrain_idx[i]
        inside = (self._z_indices >= lo) & (self._z_indices < lo + self.layer_idx[i])
        return self.het + self.layer_delta_het * inside

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
        alpha = self.lower_impedance_alpha
        # forward DMFT
        w = (u[2:] - u[:-2]) / (2 * self.dz_m) + alpha * u[1:-1]
        w_ext = jnp.concatenate((jnp.zeros(1, dtype=complex), w, jnp.zeros(1, dtype=complex), -w[::-1]))
        c = 1j * jnp.fft.fft(w_ext) / n  # sine-series coefficients (index 1..n-1 meaningful)
        c_int = c[1:n]

        def inverse_sine_part(coefs):
            cs = coefs * self._dmft_sin_coef
            cc = coefs * self._dmft_cos_coef
            zero = jnp.zeros(1, dtype=complex)
            cs_ext = jnp.concatenate((zero, cs, zero, -cs[::-1]))
            cc_ext = jnp.concatenate((zero, cc, zero, cc[::-1]))
            return (1j * jnp.fft.fft(cs_ext)[:self.z_n] + jnp.fft.fft(cc_ext)[:self.z_n]) / 2

        # surface-wave amplitudes from the residual at the end points
        u_sine = inverse_sine_part(c_int)
        e_0 = u[0] - u_sine[0]
        e_n = u[-1] - u_sine[-1]
        a_sw = (e_0 - self._r_n_signed * e_n) * self._end_det
        b_sw = (e_n - self._r_n * e_0) * self._end_det
        # propagation
        c_int = c_int * self._spectral_mult[1:n]
        a_sw = a_sw * self._surface_mult
        b_sw = b_sw * self._surface_mult_top
        # inverse DMFT
        return inverse_sine_part(c_int) + a_sw * self._r_pow + b_sw * self._r_pow_top

    def _spectral_step(self, u):
        if self.lower_bc == 'dirichlet':
            return self._spectral_step_dirichlet(u)
        if self.lower_bc == 'neumann':
            return self._spectral_step_neumann(u)
        return self._spectral_step_impedance(u)

    # -------------------------------------------------------------- stepping
    def _step(self, u, het):
        phase_half = jnp.exp(0.5j * self.dx_m * self.beta * (jnp.sqrt(1.0 + het) - 1.0))
        u = phase_half * u
        u = self._spectral_step(u)
        u = phase_half * u
        return u * self._abs_window

    @jax.jit
    def compute(self, initial):
        """Propagate ``initial`` (given on :meth:`z_computational_grid` or on
        :meth:`z_physical_grid`; it is zero-padded into the absorbing layer)
        and return the field on the output grid, shape
        ``(len(x_output_grid), len(z_output_grid))``."""
        initial = jnp.asarray(initial, dtype=complex)
        u0 = jnp.zeros(self.z_n, dtype=complex).at[:initial.shape[0]].set(initial[:self.z_n])
        u0 = u0 * self.terrain_mask_row(0)

        n_x_out = (self.x_n - 1) // self.x_grid_scale + 1
        n_z_out = (self.z_n_phys - 1) // self.z_grid_scale + 1
        results = jnp.zeros((n_x_out, n_z_out), dtype=complex)
        results = results.at[0, :].set(u0[:self.z_n_phys:self.z_grid_scale])

        def body_fun(i, val):
            u, res = val
            u = self._step(u, self.het_row(i))
            u = u * self.terrain_mask_row(i)
            res = jax.lax.cond(i % self.x_grid_scale == 0,
                               lambda: res.at[i // self.x_grid_scale, :].set(u[:self.z_n_phys:self.z_grid_scale]),
                               lambda: res)
            return u, res

        _, results = jax.lax.fori_loop(1, self.x_n, body_fun, (u0, results))
        return results


tree_util.register_pytree_node(SSFHelmholtzPropagator,
                               SSFHelmholtzPropagator._tree_flatten,
                               SSFHelmholtzPropagator._tree_unflatten)
