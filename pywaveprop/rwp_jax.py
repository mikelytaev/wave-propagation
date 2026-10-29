"""
JAX-based tropospheric radio wave propagation module.

This is the primary implementation of the tropospheric radio wave propagation
solver using JAX for GPU-accelerated computation.
"""
import cmath as cm
from copy import deepcopy
from dataclasses import dataclass, field, replace

from pywaveprop.helmholtz_jax import RegularGrid, AbstractWaveSpeedModel, RationalHelmholtzPropagator, \
    AbstractTerrainModel, PiecewiseLinearTerrainModel, LowerLayerModel
from pywaveprop.helmholtz_common import HelmholtzMeshParams2D
from pywaveprop.helmholtz_ssf_jax import SSFHelmholtzPropagator, impedance_alpha
from pywaveprop.helmholtz_rassf_jax import RankAdaptiveSSFPropagator
from pywaveprop.helmholtz_ssw_jax import SSWHelmholtzPropagator
from pywaveprop.propagators._utils import reflection_coef
from pywaveprop.rwp_field import RWPField, TerrainFollowingRWPField
from pywaveprop.helmholtz_tf_jax import TerrainFollowingHelmholtzPropagator, TFMedium
from pywaveprop.terrain_transform import OrthogonalTerrainMap
import jax
import jax.numpy as jnp
import numpy as np
import math as fm
from jax import tree_util


@dataclass
class RWPComputationalParams:
    max_range_m: float
    max_height_m: float = None
    max_angle_deg: float = None
    rational_approx_order = None
    approx_method = 'pade'  # 'pade' / 'ratinterp' (rational FD), 'ssf' (split-step Fourier), 'rassf' (rank-adaptive SSF), 'ssw' (split-step wavelet)
    ssf_impedance_bc = 'dmft'  # SSF over a dielectric ground: 'dmft' (Leontovich, mixed Fourier) or 'cf' (angle-dependent, consistent field)
    terrain_method = 'staircase'  # rational FD over terrain: 'staircase' or 'transform' (orthogonal terrain-following coordinates)
    terrain_transition_m = None  # 'transform': height above the terrain where the coordinates become Cartesian
    terrain_smoothing_m = None  # 'transform': Gaussian smoothing of the terrain samples (default: their spacing)
    terrain_z_ref = 'mean'  # 'transform': level the coordinate lines relax to: 'mean', 'max', 'min' or metres
    terrain_slope_aperture = True  # 'transform': widen the aperture by twice the maximum terrain slope
    terrain_beta_mode = 'global'  # 'transform': 'global' or 'column' (one reference wavenumber per class of columns)
    terrain_beta_classes = 16  # 'transform', terrain_beta_mode='column': number of column classes
    terrain_absorb_m = None  # 'transform': absorbing layer on top with a Dirichlet condition instead of the NLBC
    terrain_absorb_stretch = True  # 'transform': stretched vertical grid in the absorbing layer
    terrain_upper_stretch = False  # 'transform': coarser vertical grid above the distorted layer (NLBC on top)
    dx_m: float = None
    dz_m: float = None
    x_output_points: int = None
    z_output_points: int = None
    precision: float = 0.01

    def __post_init__(self):
        if self.x_output_points is None and self.dx_m is None:
            raise ValueError("x output grid (x_output_points or dx_m) is not specified!")
        if self.x_output_points is not None and self.dx_m is not None:
            raise ValueError("only one x output grid parameter (x_output_points or dx_m) should be specified!")

        if self.z_output_points is None and self.dz_m is None:
            raise ValueError("z output grid (z_output_points or dz_m) is not specified!")
        if self.z_output_points is not None and self.dz_m is not None:
            raise ValueError("only one z output grid parameter (z_output_points or dz_m) should be specified!")


class RWPGaussSourceModel:

    def __init__(self, *, freq_hz, height_m, beam_width_deg, elevation_angle_deg=0, polarization='H', multiplier=1.0):
        self.freq_hz = freq_hz
        self.height_m = height_m
        self.beam_width_deg = beam_width_deg
        self.elevation_angle_deg = elevation_angle_deg
        self.polarization = polarization
        self.multiplier = multiplier

    def aperture(self, z):
        k0 = 2 * jnp.pi * self.freq_hz / 3E8
        elevation_angle_rad = jnp.radians(self.elevation_angle_deg)
        ww = jnp.sqrt(2 * jnp.log(2)) / (k0 * jnp.sin(jnp.radians(self.beam_width_deg) / 2))
        norm = self.multiplier / (jnp.sqrt(jnp.pi) * ww)
        # Method-of-images Gaussian: combine beam at +height_m with its mirror
        # at -height_m so the input automatically respects the BC at z=0.
        # For H-polarization the difference enforces u(0)=0 (Dirichlet, PEC);
        # for V-polarization the sum enforces du/dz(0)=0 (Neumann/PEC) and is
        # the natural starting field for finite-eps ground via the lower NLBC.
        pos = norm * jnp.exp(-1j * k0 * jnp.sin(elevation_angle_rad) * z) \
              * jnp.exp(-((z - self.height_m) / ww) ** 2)
        neg = norm * jnp.exp(-1j * k0 * jnp.sin(elevation_angle_rad) * (-z)) \
              * jnp.exp(-((-z - self.height_m) / ww) ** 2)
        sign = -1.0 if self.polarization.upper() == 'H' else 1.0
        return jnp.array(pos + sign * neg, dtype=complex)

    def max_angle_deg(self):
        return self.beam_width_deg + abs(self.elevation_angle_deg)

    def _tree_flatten(self):
        dynamic = (self.height_m, self.beam_width_deg, self.elevation_angle_deg, self.multiplier)
        static = {
            'freq_hz': self.freq_hz,
            'polarization': self.polarization,
        }
        return dynamic, static

    @classmethod
    def _tree_unflatten(cls, static, dynamic):
        return cls(height_m=dynamic[0], beam_width_deg=dynamic[1], elevation_angle_deg=dynamic[2], multiplier=dynamic[3], **static)


tree_util.register_pytree_node(RWPGaussSourceModel,
                               RWPGaussSourceModel._tree_flatten,
                               RWPGaussSourceModel._tree_unflatten)


class AbstractNProfileModel:

    def __call__(self, z):
        pass

    def M_Profile(self, z, M_0=320.0, r_e=6371000):
        return self(z) + M_0 + (1 / r_e) * 1E6 * z

    def max_height_m(self):
        pass

    def on_regular_grid(self, z_grid: RegularGrid):
        return self(z_grid.array_grid(0, 100000000))

    def __add__(self, other):
        return SumNProfileModel(self, other)

    def __mul__(self, other):
        return MultNProfileModel(self, other)


@dataclass
class SumNProfileModel(AbstractNProfileModel):
    left: AbstractNProfileModel
    right: AbstractNProfileModel

    def __call__(self, z):
        return self.left(z) + self.right(z)

    def max_height_m(self):
        return max(self.left.max_height_m(), self.right.max_height_m())

    def _tree_flatten(self):
        dynamic = (self.left, self.right)
        static = {}
        return dynamic, static

    @classmethod
    def _tree_unflatten(cls, static, dynamic):
        return cls(left=dynamic[0], right=dynamic[1])

tree_util.register_pytree_node(SumNProfileModel,
                               SumNProfileModel._tree_flatten,
                               SumNProfileModel._tree_unflatten)


@dataclass
class MultNProfileModel(AbstractNProfileModel):
    profile: AbstractNProfileModel
    scalar: float

    def __call__(self, z):
        return self.scalar*self.profile(z)

    def max_height_m(self):
        return self.profile.max_height_m()

    def _tree_flatten(self):
        dynamic = (self.profile, self.scalar)
        static = {}
        return dynamic, static

    @classmethod
    def _tree_unflatten(cls, static, dynamic):
        return cls(profile=dynamic[0], scalar=dynamic[1])

tree_util.register_pytree_node(MultNProfileModel,
                               MultNProfileModel._tree_flatten,
                               MultNProfileModel._tree_unflatten)


@dataclass
class EmptyNProfileModel(AbstractNProfileModel):

    def __call__(self, z):
        return z*0

    def max_height_m(self):
        return 0.0

    def _tree_flatten(self):
        dynamic = ()
        static = {}
        return dynamic, static

    @classmethod
    def _tree_unflatten(cls, static, dynamic):
        return cls()

tree_util.register_pytree_node(EmptyNProfileModel,
                               EmptyNProfileModel._tree_flatten,
                               EmptyNProfileModel._tree_unflatten)


def evaporation_duct(height, z_grid_m, m_0=320, z_0=1.5e-4):
    return m_0 + 0.125*(z_grid_m - height*jnp.log(1 + z_grid_m / z_0))


@dataclass
class EvaporationDuctModel(AbstractNProfileModel):
    height_m: float
    truncate_height_m: float = 100

    def __call__(self, z):
        shift = evaporation_duct(self.height_m, self.truncate_height_m) - evaporation_duct(0.0, self.truncate_height_m)
        return (evaporation_duct(self.height_m, z) - evaporation_duct(0.0, z) - shift) * (jnp.sign(self.truncate_height_m - z) + 1) / 2

    def max_height_m(self):
        return self.truncate_height_m

    def _tree_flatten(self):
        dynamic = (self.height_m, self.truncate_height_m)
        static = {}
        return dynamic, static

    @classmethod
    def _tree_unflatten(cls, static, dynamic):
        return cls(height_m=dynamic[0], truncate_height_m=dynamic[1])

tree_util.register_pytree_node(EvaporationDuctModel,
                               EvaporationDuctModel._tree_flatten,
                               EvaporationDuctModel._tree_unflatten)


class PiecewiseLinearNProfileModel(AbstractNProfileModel):

    def __init__(self, z_grid_m: jax.Array, N_vals: jax.Array):
        self.z_grid_m = z_grid_m
        self.N_vals = N_vals

    @staticmethod
    def create_from_M_profile(z_grid_m: jax.Array, M_vals: jax.Array):
        slope = (M_vals[-1] - M_vals[-2]) / (z_grid_m[-1] - z_grid_m[-2])
        N_vals = M_vals - z_grid_m*slope
        return PiecewiseLinearNProfileModel(z_grid_m, N_vals)

    def __call__(self, z):
        return jnp.interp(z, self.z_grid_m, self.N_vals,
                          left='extrapolate', right='extrapolate')

    def max_height_m(self):
        return max(self.z_grid_m) + 1

    def _tree_flatten(self):
        dynamic = (self.z_grid_m, self.N_vals)
        static = {}
        return dynamic, static

    @classmethod
    def _tree_unflatten(cls, static, dynamic):
        return cls(z_grid_m=dynamic[0], N_vals=dynamic[1])

tree_util.register_pytree_node(PiecewiseLinearNProfileModel,
                               PiecewiseLinearNProfileModel._tree_flatten,
                               PiecewiseLinearNProfileModel._tree_unflatten)


class RangeDependentNProfileModel(AbstractNProfileModel):
    """N(x, z) given on a set of ranges: piecewise-linear in z, linear in x.

    ``N_vals`` has shape ``(len(x_grid_m), len(z_grid_m))``. ``__call__`` returns
    the profile of the first column so that the model can be used wherever a
    single profile is expected (wavenumber bounds, mesh setup); propagators that
    support a range-dependent medium call :meth:`columns`.
    """

    def __init__(self, x_grid_m: jax.Array, z_grid_m: jax.Array, N_vals: jax.Array):
        self.x_grid_m = jnp.asarray(x_grid_m)
        self.z_grid_m = jnp.asarray(z_grid_m)
        self.N_vals = jnp.asarray(N_vals)

    @staticmethod
    def create_from_M_profile(x_grid_m: jax.Array, z_grid_m: jax.Array, M_vals: jax.Array):
        """``M_vals`` of shape (n_x, n_z); the curvature slope is taken from the
        top of the first column, as in :class:`PiecewiseLinearNProfileModel`."""
        M_vals = jnp.asarray(M_vals)
        z_grid_m = jnp.asarray(z_grid_m)
        slope = (M_vals[0, -1] - M_vals[0, -2]) / (z_grid_m[-1] - z_grid_m[-2])
        return RangeDependentNProfileModel(x_grid_m, z_grid_m, M_vals - z_grid_m[None, :] * slope)

    def __call__(self, z):
        return jnp.interp(z, self.z_grid_m, self.N_vals[0],
                          left='extrapolate', right='extrapolate')

    def columns(self, z):
        """Profiles on ``z`` for every range column, shape (n_x, len(z))."""
        return jax.vmap(lambda col: jnp.interp(z, self.z_grid_m, col,
                                               left='extrapolate', right='extrapolate'))(self.N_vals)

    def max_height_m(self):
        return max(self.z_grid_m) + 1

    def _tree_flatten(self):
        return (self.x_grid_m, self.z_grid_m, self.N_vals), {}

    @classmethod
    def _tree_unflatten(cls, static, dynamic):
        return cls(x_grid_m=dynamic[0], z_grid_m=dynamic[1], N_vals=dynamic[2])


tree_util.register_pytree_node(RangeDependentNProfileModel,
                               RangeDependentNProfileModel._tree_flatten,
                               RangeDependentNProfileModel._tree_unflatten)


@dataclass
class GroundMaterial:
    """Simple frequency-independent ground material described by relative
    permittivity and conductivity (S/m). Used for the lower nonlocal boundary
    condition of the JAX RWP propagator."""
    eps: float = 1.0
    sigma: float = 0.0

    def complex_permittivity(self, freq_hz: float) -> complex:
        wavelength = 3e8 / freq_hz
        return self.eps + 1j * 60 * self.sigma * wavelength


def _arnoldi_basis(z, degree):
    """Vandermonde with Arnoldi (Brubeck, Nakatsukasa & Trefethen 2021): the
    Hessenberg recurrence of a polynomial basis orthonormal on the points ``z``."""
    m = len(z)
    q = np.zeros((m, degree + 1), dtype=complex)
    hess = np.zeros((degree + 1, degree), dtype=complex)
    q[:, 0] = 1
    for k in range(degree):
        v = z * q[:, k]
        for j in range(k + 1):
            hess[j, k] = np.vdot(q[:, j], v) / m
            v = v - hess[j, k] * q[:, j]
        hess[k + 1, k] = np.linalg.norm(v) / np.sqrt(m)
        q[:, k + 1] = v / hess[k + 1, k]
    return hess


def _arnoldi_eval(hess, z):
    """Basis of :func:`_arnoldi_basis` evaluated at the points ``z``."""
    degree = hess.shape[1]
    w = np.zeros((len(z), degree + 1), dtype=complex)
    w[:, 0] = 1
    for k in range(degree):
        w[:, k + 1] = (z * w[:, k] - w[:, :k + 1] @ hess[:k + 1, k]) / hess[k + 1, k]
    return w


@dataclass
class RoughSeaSurface:
    """Rough sea surface for the lower nonlocal BC of the rational (Pade)
    propagator: the coherent reflection coefficient about the mean sea level is
    the smooth (Fresnel) one times a roughness factor ``rho(k_z)`` of the
    vertical wavenumber ``k_z = k sin(grazing angle)``.

    ``model`` selects the factor, with ``x = 2 (sigma k_z)^2``:

    * ``'miller_brown'`` -- Miller, Brown & Vegh (1984): ``rho = exp(-x) I0(x)``;
    * ``'ament'`` -- Ament (1953), Gaussian elevations: ``rho = exp(-x)``.

    The discrete NLBC (Lytaev, RadarConf 2022) samples ``R`` at complex ``k_z``
    along a closed loop in the first quadrant (see
    :meth:`RationalHelmholtzPropagator.lower_nlbc_contour_kz`), and needs ``R``
    analytic with ``1 + R`` zero-free inside it. Both factors are entire, but
    grow like ``exp(4 (sigma Im k_z)^2)`` off the real axis, and ``1 + R`` then
    has zeros inside the loop (unstable BC). No continuation bounded by 1 can
    reproduce the real factor near grazing (Kramers-Kronig). ``method``
    selects the remedy:

    * ``'fit'`` (default) -- ``1 + R~ = (1 + R_f) exp(K(k_z))``: zero-free by
      construction for any ``K``, which is fitted to the exact continuation of
      ``R`` on the propagating range and on the wedge below the real axis that
      creeping modes sample, and regularised on the rest of the contour;
      ``|R~| <= 1`` on the real stretch and in the trapping range of the
      atmosphere. Lifted by a small ``z0`` when needed (see
      :meth:`fitted_reflection_coef_func`).
    * ``'shift'`` -- the BC is placed at ``z0`` above the mean sea level
      (above the wave crests), where the reflection coefficient is
      ``R_f rho exp(2 i k_z z0)``: exact at every real grazing angle, and
      ``|exp(2 i k_z z0)| = exp(-2 z0 Im k_z)`` tames the growth. ``z0`` is the
      smallest height with ``|R| <= 1`` on the loop (maximum modulus principle),
      or ``reference_height_m``. The field is computed for ``z >= z0`` only,
      and ``z0`` grows like ``sigma^2`` times the loop size.

    Beyond the radio horizon the field is carried by creeping modes and by the
    lateral wave of the branch point at grazing incidence. Over a smooth sea
    (H-pol) the surface impedance ``k_z (R - 1) / (R + 1) = -q`` is constant and
    the lateral wave vanishes. With Miller-Brown it becomes
    ``-q / (1 + q sigma^2 k_z)`` near grazing, odd in ``k_z``: the model itself then
    predicts a lateral wave that decays slowly with range, raising the
    beyond-horizon field by tens of dB at 10-15 m/s wind speed (X band). This is
    a consequence of extrapolating Miller-Brown to grazing angles ~1e-5 rad, not
    of the boundary condition (the PE reproduces the independent modal +
    branch-cut solution), and it cannot be removed without switching roughness
    off at the low angles that matter for ducting. The fit follows the exact
    lateral wave to ~2 dB at 10 m/s; where the field is weaker still (~235 dB
    path loss at 5 m/s, X band) it differs by up to ~12 dB, as the singularity
    of the continued impedance at ``k_z = -1 / (q sigma^2)`` is out of reach of a
    polynomial ``K``.

    ``grazing_angle_deg`` enables a constant grazing impedance (Barrick, IEEE
    TAP 46(1), 1998: the roughness-modified impedance of any rough surface tends
    to a constant at grazing, unlike that of Miller-Brown / Ament). With
    ``kappa = k_z (R - 1) / (R + 1)`` the effective impedance (the Robin BC
    ``u_z = i kappa u`` for ``exp(-i k_z z) + R exp(i k_z z)``), the roughness
    admittance ``Y_r(k_z) = 1 / kappa - 1 / kappa_f`` (``kappa_f`` of
    the smooth sea; ``Y_r ~ -sigma^2 k_z`` for Miller-Brown near grazing) is
    replaced by ``Y_r(sqrt(k_z^2 + k_g^2))``, ``k_g = k sin(grazing angle)``: equal
    to the model at ``k_z >> k_g``, constant below ``k_g``, and even in ``k_z``,
    so there is no roughness-induced lateral wave. ``'rayleigh'`` places
    ``k_g = 1 / (2 sigma)``, where the Rayleigh roughness parameter
    ``2 k_z sigma`` reaches 1 (Kirchhoff/Miller-Brown above, effective
    impedance below). Only with ``method='fit'``: the blend has branch points
    at ``k_z = +-i k_g``.

    The value of the constant is not fixed by Barrick's theory, and the
    ``'rayleigh'`` choice (Miller-Brown frozen at the Rayleigh angle) is a
    heuristic awaiting calibration (method of moments, measurements). Beyond
    the horizon the field is carried by creeping modes at
    ``|k_z| ~ sqrt(|a_n|) (k^2 g)^(1/3)``; if ``k_g`` is below that range (strong
    wind at X band), they see the transition of the blend, and the far field
    depends on its shape (a warning is issued).
    """
    rms_height_m: float
    model: str = 'miller_brown'
    method: str = 'fit'
    grazing_angle_deg: object = None
    reference_height_m: float = None
    fit_degree: int = 40
    fit_tol: float = 5e-3
    fit_margin: float = 1.2
    fit_max_log: float = 15.0
    fit_harmonics: int = 6

    def __post_init__(self):
        if self.model not in ('miller_brown', 'ament'):
            raise ValueError(f"unknown roughness model {self.model!r}; expected 'miller_brown' or 'ament'")
        if self.method not in ('fit', 'shift'):
            raise ValueError(f"unknown rough-surface BC method {self.method!r}; expected 'fit' or 'shift'")
        if self.grazing_angle_deg is not None:
            if not (self.grazing_angle_deg == 'rayleigh' or
                    (isinstance(self.grazing_angle_deg, (int, float)) and 0 < self.grazing_angle_deg < 90)):
                raise ValueError(f"grazing_angle_deg must be None, 'rayleigh' or an angle in (0, 90) deg, "
                                 f"got {self.grazing_angle_deg!r}")
            if self.method != 'fit':
                raise ValueError("the constant grazing impedance (grazing_angle_deg) needs method='fit'")

    @classmethod
    def from_wind_speed(cls, wind_speed_m_s: float, **kwargs):
        """RMS wave height of a fully developed sea from the Phillips spectrum,
        ``sigma = 0.0051 U^2`` (U: wind speed in m/s, Levy 2000, sec. 9.4)."""
        return cls(rms_height_m=0.0051 * wind_speed_m_s ** 2, **kwargs)

    def log_factor(self, k_z):
        """``log rho(k_z)`` (analytic continuation, complex ``k_z``)."""
        from scipy.special import ive
        x = 2 * (self.rms_height_m * np.asarray(k_z, dtype=complex)) ** 2
        if self.model == 'ament':
            return -x
        # exp(-x) I0(x) = ive(0, x) * exp(|Re x| - x)
        return np.log(ive(0, x) + 0j) + np.abs(x.real) - x

    def factor(self, k_z):
        """Roughness factor ``rho(k_z)``."""
        return np.exp(self.log_factor(k_z))

    def grazing_k_z(self, k_surface):
        """Vertical wavenumber ``k_g`` below which the effective impedance is
        constant (0: no blend, see ``grazing_angle_deg``)."""
        if self.grazing_angle_deg is None:
            return 0.0
        if self.grazing_angle_deg == 'rayleigh':
            return min(1.0 / (2 * self.rms_height_m), float(k_surface))
        return float(k_surface) * fm.sin(fm.radians(self.grazing_angle_deg))

    def mean_level_reflection(self, k_z, smooth_refl_coef_func, k_surface):
        """Coherent reflection coefficient about the mean sea level at (complex)
        ``k_z``: ``R_f rho``, or the constant-grazing-impedance blend when
        ``grazing_angle_deg`` is set. ``smooth_refl_coef_func`` takes the grazing
        angle in degrees."""
        k_z = np.atleast_1d(np.asarray(k_z, dtype=complex))

        def r_f_of(kz):
            return np.array([complex(smooth_refl_coef_func(v)) for v in np.arcsin(kz / k_surface) * 180 / np.pi])

        r_f = r_f_of(k_z)
        k_g = self.grazing_k_z(k_surface)
        if k_g == 0.0:
            return r_f * np.exp(self.log_factor(k_z))
        # admittance 1/kappa = (R + 1) / (k_z (R - 1)); the roughness part is
        # evaluated at sqrt(k_z^2 + k_g^2) (principal branch: continuous on the
        # lower-half-plane sectors the fit samples, cut on the imaginary axis)
        k_p = np.sqrt(k_z ** 2 + k_g ** 2)
        r_f_p = r_f_of(k_p)
        if np.any(r_f_p == 1):
            raise NotImplementedError("constant grazing impedance over a Neumann (R_f = 1) surface")
        r_p = r_f_p * np.exp(self.log_factor(k_p))
        y_r = ((r_p + 1) / (r_p - 1) - (r_f_p + 1) / (r_f_p - 1)) / k_p
        w = (r_f + 1) / (r_f - 1) + k_z * y_r  # k_z / kappa
        return (w + 1) / (w - 1)

    def grazing_impedance(self, smooth_refl_coef_func, k_surface):
        """Normalized effective impedance ``kappa / k`` in the grazing limit."""
        k_z = float(k_surface) * 1e-7
        r = self.mean_level_reflection(k_z, smooth_refl_coef_func, k_surface)[0]
        return complex(k_z * (r - 1) / (r + 1) / k_surface)

    def auto_reference_height(self, contour_k_z, smooth_refl_coef_func, k_surface, tol=1e-9):
        """Smallest ``z0 >= 0`` with ``|R_f rho exp(2 i k_z z0)| <= 1`` on the NLBC
        contour loop ``contour_k_z`` (see
        :meth:`RationalHelmholtzPropagator.lower_nlbc_contour_kz`).

        ``log|R| = log|R_f rho| - 2 z0 Im k_z`` is linear in ``z0`` at every
        contour point, so ``z0 = max (log|R_f rho|) / (2 Im k_z)``."""
        k_z = np.asarray(contour_k_z, dtype=complex)
        theta = np.arcsin(k_z / k_surface) * 180 / np.pi
        r_f = np.array([complex(smooth_refl_coef_func(t)) for t in theta])
        log_abs = np.log(np.abs(r_f) + 1e-300) + self.log_factor(k_z).real
        bad = log_abs > tol
        if not bad.any():
            return 0.0
        if np.any(k_z.imag[bad] <= 0):
            raise ValueError("rough-surface reflection coefficient exceeds 1 on the real k_z axis")
        return float(np.max((log_abs[bad] - tol) / (2 * k_z.imag[bad])))

    def fitted_reflection_coef_func(self, smooth_refl_coef_func, k_surface, contour_k_z, k_z_max, z0=0.0,
                                    k_z_passive=0.0, k_z_creep=0.0):
        """Zero-free fitted reflection coefficient at the plane ``z0`` above the
        mean level, ``R~ = (1 + R_f) exp(K) - 1``, as a function of the (complex)
        grazing angle in degrees (``method='fit'``).

        ``K = sum_{n <= fit_harmonics} exp(2 i n k_z z0) P_n(k_z)``, ``P_n``
        polynomials of degree ``fit_degree`` (Vandermonde with Arnoldi basis),
        is fitted to ``log((1 + R) / (1 + R_f))`` with
        ``R = R_f rho exp(2 i k_z z0)``:

        * on the propagating range ``[0, fit_margin * k_z_max]``, and with a lower
          weight on the rest of the real k_z stretch of the contour loop;
        * below the real axis, where range-decaying modes see the boundary: the
          creeping (diffraction) modes beyond the horizon, ``|k_z| <= k_z_creep``,
          ``arg k_z`` down to -70 deg (they sit at -60 deg), and the slowly
          decaying trapped modes of a duct just below the real axis,
          ``|k_z| <= k_z_passive``, ``arg k_z`` down to -10 deg. The continuation of
          ``R`` is well defined there, and an unconstrained polynomial would
          produce spurious, slowly decaying modes.

        A Tikhonov term on the rest of the loop is relaxed until
        ``max |log((1 + R~) / (1 + R))| <= fit_tol`` on both (a relative error of
        ``1 + R``, which matters near grazing, where ``R -> -1``). A fit is
        admissible only if ``|Re K| <= fit_max_log`` on the loop, ``|R~| <= 1`` on
        the whole real stretch (an active reflection amplifies the wave at every
        bounce, which diverges in a duct) and in the quarter disc
        ``|k_z| <= k_z_passive`` above the real axis, which a trapping layer turns
        back to the surface (a growing mode would resonate with it). Returns
        ``(func, info)``, ``info['fit_error']`` being the achieved error (``inf``
        and ``func is None`` if no fit is admissible).
        """
        def r_f_of(kz):
            return np.array([complex(smooth_refl_coef_func(v)) for v in np.arcsin(kz / k_surface) * 180 / np.pi])

        def target_of(kz, r_f_kz):
            r = self.mean_level_reflection(kz, smooth_refl_coef_func, k_surface) * np.exp(2j * kz * z0)
            return np.log((1 + r) / (1 + r_f_kz))

        loop = np.asarray(contour_k_z, dtype=complex)
        scale = np.max(np.abs(loop))
        near_real = loop.imag < 1e-2 * scale
        kz_fit = self.fit_margin * k_z_max
        kz_real = max(kz_fit, float(np.max(loop.real[near_real])))
        t = np.concatenate([np.linspace(0, kz_fit, 600)[1:], kz_fit * np.logspace(-4, -1, 100)])
        n_fit = len(t)
        t = np.concatenate([t, np.linspace(kz_fit, kz_real, 300)[1:]])
        weights = np.where(np.arange(len(t)) < n_fit, 1.0, 0.2)
        r_f = r_f_of(t)
        target = target_of(t, r_f)
        # sectors below the real axis, target unwrapped along rays from the axis
        wedge, target_w = [], []
        for radius, max_deg in ((k_z_creep, 70.0), (k_z_passive, 10.0)):
            if radius <= 0:
                continue
            rays = radius * np.linspace(0.02, 1, 40)[:, None] * np.exp(-1j * np.radians(np.linspace(0, max_deg, 21)))[None, :]
            r_f_w = r_f_of(rays.ravel()).reshape(rays.shape)
            tw = target_of(rays.ravel(), r_f_w.ravel()).reshape(rays.shape)
            tw = tw.real + 1j * np.unwrap(tw.imag, axis=1)
            wedge.append(rays[:, 1:].ravel())
            target_w.append(tw[:, 1:].ravel())
        wedge = np.concatenate(wedge) if wedge else np.zeros(0, dtype=complex)
        target_w = np.concatenate(target_w) if target_w else np.zeros(0, dtype=complex)
        # regularise only where no accuracy is asked for: off the real stretch.
        # The contour has one sample per rational branch and quadrature node,
        # tens of thousands of mostly coincident points: thin them on a spatial
        # grid (a stride would alias with the ordering and skip whole arcs)
        loop = loop[~near_real]
        cell = 2e-3 * scale
        _, keep = np.unique(np.round(loop.real / cell) + 1j * np.round(loop.imag / cell), return_index=True)
        loop = loop[np.sort(keep)]
        hess = _arnoldi_basis(np.concatenate([t, wedge, loop]) / scale, self.fit_degree)
        # K = sum_n exp(2 i n k_z z0) P_n(k_z): log(1 + R) is a power series in
        # the lift phase exp(2 i k_z z0), too oscillatory for a polynomial alone;
        # each harmonic is bounded by 1 in the upper half-plane
        n_harm = self.fit_harmonics if z0 > 0 else 0

        def basis(kz):
            b = _arnoldi_eval(hess, kz / scale)
            return np.concatenate([b * np.exp(2j * n * kz * z0)[:, None] for n in range(n_harm + 1)], axis=1)
        basis_t = basis(t)
        basis_w = basis(wedge)
        basis_loop = basis(loop)
        if k_z_passive > 0:
            # boundary of the quarter disc |k_z| <= k_z_passive off the real axis
            eta = np.linspace(0, k_z_passive, 200)[1:]
            trap = np.concatenate([1j * eta, k_z_passive * np.exp(1j * np.linspace(0, np.pi / 2, 200))])
            r_f_trap = r_f_of(trap)
            basis_trap = basis(trap)
        best = None
        # match K to first order at grazing (value and slope, central differences
        # across k_z = 0): then R~ - R = O(k_z^3) and the effective impedance is
        # exact up to O(k_z^2), with no spurious odd (lateral-wave) term. Where
        # the impedance is small (constant grazing impedance) even a residual
        # slope of K ~ 1e-3 leaves a lateral wave far beyond the horizon
        h = 1e-4 * kz_fit
        pm = np.array([h, -h])
        t_pm = target_of(pm, r_f_of(pm))
        b_pm = basis(pm)
        w_g = 1e3
        a0 = np.concatenate([weights[:, None] * basis_t, basis_w,
                             w_g * np.array([(b_pm[0] + b_pm[1]) / 2, (b_pm[0] - b_pm[1]) / (2 * h) * kz_fit])])
        b0 = np.concatenate([weights * target, target_w,
                             w_g * np.array([(t_pm[0] + t_pm[1]) / 2, (t_pm[0] - t_pm[1]) / (2 * h) * kz_fit])])
        # below the real axis the lift harmonics grow like exp(2 n |Im k_z| z0):
        # for large lifts they overflow, and such a lift is not admissible
        if not (np.all(np.isfinite(a0)) and np.all(np.isfinite(b0)) and np.all(np.isfinite(basis_loop))):
            return None, {'fit_error': np.inf, 'reference_height_m': z0}
        for lam in 10.0 ** -np.arange(1, 13):
            a = np.concatenate([a0, np.sqrt(lam) * basis_loop])
            b = np.concatenate([b0, np.zeros(len(loop), dtype=complex)])
            try:
                c = np.linalg.lstsq(a, b, rcond=None)[0]
            except np.linalg.LinAlgError:
                continue
            k_fit = basis_t @ c
            r_fit = (1 + r_f) * np.exp(k_fit) - 1
            # relative error of 1 + R (= error of K): near grazing 1 + R -> 0 and
            # an absolute error in R would misplace the effective surface
            # impedance that governs diffraction beyond the horizon
            err = float(np.max(np.abs(k_fit - target)[:n_fit]))
            if len(wedge):
                err = max(err, float(np.max(np.abs(basis_w @ c - target_w))))
            max_abs_r = float(np.max(np.abs(r_fit)))
            max_log = float(np.max(np.abs((basis_loop @ c).real)))
            if max_log > self.fit_max_log:
                break
            if k_z_passive > 0:
                max_abs_r = max(max_abs_r, float(np.max(np.abs((1 + r_f_trap) * np.exp(basis_trap @ c) - 1))))
            if max_abs_r > 1 + 1e-6:
                continue
            if best is None or err < best[1]:
                best = (c, err, max_log, lam, max_abs_r)
            if err <= self.fit_tol:
                break
        if best is None:
            return None, {'fit_error': np.inf, 'reference_height_m': z0}
        c, err, max_log, lam, max_abs_r = best

        def func(theta_deg):
            k_z = k_surface * cm.sin(theta_deg * cm.pi / 180)
            k_poly = complex((basis(np.array([k_z])) @ c)[0])
            return (1 + complex(smooth_refl_coef_func(theta_deg))) * cm.exp(k_poly) - 1
        info = {'fit_error': err, 'max_abs_re_K_on_contour': max_log, 'regularization': float(lam),
                'max_abs_R_real_axis': max_abs_r, 'reference_height_m': z0}
        if self.grazing_angle_deg is not None:
            info['k_z_grazing'] = self.grazing_k_z(k_surface)
            info['grazing_impedance'] = self.grazing_impedance(smooth_refl_coef_func, k_surface)
        return func, info

    def reflection_coef_func(self, smooth_refl_coef_func, k_surface, z0=0.0):
        """Reflection coefficient at the plane ``z0`` above the mean sea level as
        a function of the (complex) grazing angle in degrees; ``k_surface`` is
        the wavenumber at the surface."""
        from scipy.special import ive
        sigma = self.rms_height_m
        ament = self.model == 'ament'

        def func(theta_deg):
            k_z = k_surface * cm.sin(theta_deg * cm.pi / 180)
            x = 2 * (sigma * k_z) ** 2
            if ament:
                log_rho = -x
            else:
                i0e = complex(ive(0, x))
                if i0e == 0:
                    return 0j
                log_rho = cm.log(i0e) + abs(x.real) - x
            # combined before exponentiating: the shift phase cancels the growth of rho
            return smooth_refl_coef_func(theta_deg) * cm.exp(log_rho + 2j * k_z * z0)
        return func


@dataclass
class VegetationLayerModel:
    """Vegetation (forest canopy) modelled as a lossy dielectric slab of
    range-dependent thickness lying on the terrain, following the classical
    slab model of Tamir. ``thickness`` gives the canopy height above ground as
    a function of range (0 where there is no vegetation); ``material`` holds
    the effective relative permittivity and conductivity of the canopy."""
    thickness: AbstractTerrainModel
    material: GroundMaterial = field(default_factory=lambda: GroundMaterial(eps=1.004, sigma=180e-6))

    def k2_contrast(self, freq_hz: float) -> complex:
        k0 = 2 * fm.pi * freq_hz / 3E8
        return complex(k0 ** 2 * (self.material.complex_permittivity(freq_hz) - 1.0))

    def _tree_flatten(self):
        return (self.thickness,), {'material': self.material}

    @classmethod
    def _tree_unflatten(cls, static, dynamic):
        return cls(thickness=dynamic[0], **static)


tree_util.register_pytree_node(VegetationLayerModel,
                               VegetationLayerModel._tree_flatten,
                               VegetationLayerModel._tree_unflatten)


@dataclass
class TroposphereModel:
    N_profile: AbstractNProfileModel = field(default_factory=EmptyNProfileModel)
    M0: float = 320
    slope: float = (1 / 6371000) * 1E6
    terrain: AbstractTerrainModel = None
    ground_material: GroundMaterial = None
    vegetation: VegetationLayerModel = None
    sea_roughness: RoughSeaSurface = None

    def M_profile(self, z):
        return self.N_profile(z) + self.M0 + self.slope*z

    def M_profile_columns(self, z):
        """M(z) for every range column of a range-dependent N profile."""
        return self.N_profile.columns(z) + self.M0 + self.slope*z

    def wave_speed_profile(self, z):
        return 3E8 / (1 + self.M_profile(z) * 1E-6)

    def max_height_m(self):
        return self.N_profile.max_height_m()

    def _tree_flatten(self):
        dynamic = (self.N_profile, self.M0, self.slope, self.terrain, self.vegetation)
        static = {
            'ground_material': self.ground_material,
            'sea_roughness': self.sea_roughness,
        }
        return dynamic, static

    @classmethod
    def _tree_unflatten(cls, static, dynamic):
        unf = cls(N_profile=dynamic[0], M0=dynamic[1], slope=dynamic[2],
                  terrain=dynamic[3], vegetation=dynamic[4], **static)
        return unf


tree_util.register_pytree_node(TroposphereModel,
                               TroposphereModel._tree_flatten,
                               TroposphereModel._tree_unflatten)


class ProxyWaveSpeedModel(AbstractWaveSpeedModel):

    def __init__(self, tm: TroposphereModel):
        self.tm = tm

    def __call__(self, z):
        return self.tm.wave_speed_profile(z)

    def on_regular_grid(self, z_grid: RegularGrid):
        return self(z_grid.array_grid(0, 100000000))

    def columns_on_regular_grid(self, z_grid: RegularGrid):
        """Wave speed for every range column; only for a range-dependent medium."""
        if not hasattr(self.tm.N_profile, 'columns'):
            raise AttributeError('the N profile is not range-dependent')
        z = z_grid.array_grid(0, 100000000)
        return 3E8 / (1 + self.tm.M_profile_columns(z) * 1E-6)

    def _tree_flatten(self):
        dynamic = (self.tm,)
        static = {}
        return dynamic, static

    @classmethod
    def _tree_unflatten(cls, static, dynamic):
        unf = cls(tm=dynamic[0])
        return unf


tree_util.register_pytree_node(ProxyWaveSpeedModel,
                               ProxyWaveSpeedModel._tree_flatten,
                               ProxyWaveSpeedModel._tree_unflatten)


def minmax_k(src: RWPGaussSourceModel, env: TroposphereModel, max_height_m: float = None):
    height = max_height_m or max(env.max_height_m(), 300.0)
    z_grid = jnp.linspace(0.0, height, 1000)
    if hasattr(env.N_profile, 'columns'):
        # range-dependent medium: bounds must cover every column
        k_func_arr = 2 * fm.pi * src.freq_hz / (3E8 / (1 + env.M_profile_columns(z_grid) * 1E-6))
    else:
        k_func_arr = 2 * fm.pi * src.freq_hz / env.wave_speed_profile(z_grid)
    k_min = jnp.min(k_func_arr)
    k_max = jnp.max(k_func_arr)
    print(f'k_min: {k_min}, k_max: {k_max}')
    return k_min, k_max


def _rwp_model_setup(src: RWPGaussSourceModel, env: TroposphereModel, params: RWPComputationalParams):
    """Grid-independent quantities shared by the rational-FD and SSF front-ends."""
    params = deepcopy(params)

    max_angle_deg = params.max_angle_deg or src.max_angle_deg()
    k0 = 2 * jnp.pi * src.freq_hz / 3E8
    kz_max = k0 * jnp.sin(jnp.radians(max_angle_deg))

    max_height_m = env.max_height_m()
    if params.max_height_m:
        params.max_height_m = max(params.max_height_m, max_height_m*1.1)
    else:
        params.max_height_m = max(max_height_m * 1.1, 300.0)

    k_min, k_max = minmax_k(src, env, max_height_m=params.max_height_m)
    k_min = float(k_min)
    k_max = float(k_max)
    # Ensure minimum spread in k_bounds for nearly homogeneous media
    k_center = (k_min + k_max) / 2
    if (k_max - k_min) / k_center < 1e-3:
        k_min = k_center * 0.999
        k_max = k_center * 1.001

    lower_refl_coef_func = None
    if env.ground_material is not None:
        eps_complex = env.ground_material.complex_permittivity(src.freq_hz)
        polarz = src.polarization
        # Closure: legacy reflection_coef takes the angle between the wave and
        # the surface normal, while the propagator passes a grazing angle, so
        # convert via 90 - theta.
        def lower_refl_coef_func(theta_deg):
            return reflection_coef(1, eps_complex, 90 - theta_deg, polarz)

    if getattr(env, 'sea_roughness', None) is not None and lower_refl_coef_func is None:
        # rough perfectly conducting surface
        pec_refl_coef = -1.0 if src.polarization.upper() == 'H' else 1.0
        lower_refl_coef_func = lambda theta_deg: pec_refl_coef

    lower_layer = None
    if getattr(env, 'vegetation', None) is not None:
        lower_layer = LowerLayerModel(thickness=env.vegetation.thickness,
                                      k2_contrast=env.vegetation.k2_contrast(src.freq_hz))

    mesh_params = HelmholtzMeshParams2D(
        x_size_m=params.max_range_m,
        z_size_m=params.max_height_m,
        dx_output_m=params.dx_m,
        x_n_upper_bound=params.x_output_points,
        dz_output_m=params.dz_m,
        z_n_upper_bound=params.z_output_points,
    )
    return params, float(kz_max), (k_min, k_max), lower_refl_coef_func, lower_layer, mesh_params


def _create_rwp_spectral_model(cls, src: RWPGaussSourceModel, env: TroposphereModel, params: RWPComputationalParams,
                               **kwargs):
    """Shared front-end of the SSF and SSW propagators: the lower boundary
    follows the source polarization (Dirichlet for H, Neumann for V over a
    perfectly conducting ground) or is the Leontovich impedance condition
    when a ground material is given."""
    if getattr(env, 'sea_roughness', None) is not None:
        raise NotImplementedError("sea_roughness needs the angle-dependent lower nonlocal BC; "
                                  "use approx_method='pade' or 'ratinterp'")
    params, kz_max, k_bounds, _, lower_layer, mesh_params = _rwp_model_setup(src, env, params)
    if env.ground_material is not None:
        k0 = 2 * fm.pi * src.freq_hz / 3E8
        lower_bc = 'impedance'
        alpha = impedance_alpha(k0, env.ground_material.complex_permittivity(src.freq_hz), src.polarization)
    else:
        lower_bc = 'dirichlet' if src.polarization.upper() == 'H' else 'neumann'
        alpha = None
    kwargs.setdefault('lower_bc', lower_bc)
    return cls.create(
        freq_hz=src.freq_hz,
        wave_speed=ProxyWaveSpeedModel(env),
        kz_max=kz_max,
        k_bounds=k_bounds,
        precision=params.precision,
        mesh_params=mesh_params,
        lower_terrain=env.terrain,
        lower_impedance_alpha=alpha,
        lower_layer=lower_layer,
        **kwargs,
    )


def create_rwp_ssf_model(src: RWPGaussSourceModel, env: TroposphereModel, params: RWPComputationalParams,
                         **ssf_kwargs) -> SSFHelmholtzPropagator:
    """Create a JAX-based split-step Fourier propagator configured for tropospheric RWP.

    Over a finite-permittivity ground ``params.ssf_impedance_bc`` selects the
    lower boundary: ``'dmft'`` (default) is the angle-independent Leontovich
    condition with the discrete mixed Fourier transform, ``'cf'`` the
    angle-dependent consistent-field condition of Sprouse & Awadallah (2012)
    (flat ground only). Extra keyword arguments (``dx_m``, ``dz_m``,
    ``abs_layer_scale``, ``spectral_margin``, ``cf_delta``, ``cf_split``, ...)
    are forwarded to :meth:`SSFHelmholtzPropagator.create`.
    """
    method = getattr(params, 'ssf_impedance_bc', 'dmft')
    if method not in ('dmft', 'cf'):
        raise ValueError(f"unknown ssf_impedance_bc {method!r}; expected 'dmft' or 'cf'")
    if method == 'cf' and env.ground_material is not None:
        ssf_kwargs = dict(lower_bc='impedance_cf', lower_eps=env.ground_material.complex_permittivity(src.freq_hz),
                          polarization=src.polarization, **ssf_kwargs)
    return _create_rwp_spectral_model(SSFHelmholtzPropagator, src, env, params, **ssf_kwargs)


def create_rwp_rassf_model(src: RWPGaussSourceModel, env: TroposphereModel, params: RWPComputationalParams,
                           **rassf_kwargs) -> RankAdaptiveSSFPropagator:
    """Create a JAX-based rank-adaptive split-step Fourier propagator configured for tropospheric RWP.

    ``params.precision`` is the target accumulated error of the low-rank symbol
    approximation. Extra keyword arguments (``dx_m``, ``dz_m``, ``abs_layer_scale``,
    ``spectral_margin``, ``composition``, ``absorber``, ``symbol_tol``, ``max_rank``)
    are forwarded to :meth:`RankAdaptiveSSFPropagator.create`.
    """
    return _create_rwp_spectral_model(RankAdaptiveSSFPropagator, src, env, params, **rassf_kwargs)


def create_rwp_ssw_model(src: RWPGaussSourceModel, env: TroposphereModel, params: RWPComputationalParams,
                         **ssw_kwargs) -> SSWHelmholtzPropagator:
    """Create a JAX-based split-step wavelet propagator configured for tropospheric RWP.

    ``params.precision`` is the target accumulated compression error. Extra
    keyword arguments (``wavelet``, ``levels``, ``n_image``, ``sparse``, ``dx_m``,
    ``dz_m``, ``abs_layer_scale``) are forwarded to
    :meth:`SSWHelmholtzPropagator.create`.
    """
    return _create_rwp_spectral_model(SSWHelmholtzPropagator, src, env, params, **ssw_kwargs)


def _create_rough_sea_rwp_model(src: RWPGaussSourceModel, env: TroposphereModel, params: RWPComputationalParams,
                                kz_max, k_bounds, smooth_refl_coef_func, mesh_params) -> RationalHelmholtzPropagator:
    """Rational propagator over a rough sea (see :class:`RoughSeaSurface`): the
    lower NLBC sits at the reference height ``z0`` (0 for ``method='fit'`` over
    a moderately rough sea), where the computational domain starts."""
    rough = env.sea_roughness
    if env.terrain is not None or getattr(env, 'vegetation', None) is not None:
        raise NotImplementedError("sea_roughness is only supported over a flat sea (no terrain / vegetation)")
    # Probe propagator (no lower NLBC): fixes the grid and the rational
    # coefficients, which define the NLBC contour.
    probe = RationalHelmholtzPropagator.create(
        rational_approx_order=params.rational_approx_order,
        freq_hz=src.freq_hz,
        wave_speed=ProxyWaveSpeedModel(env),
        kz_max=kz_max,
        k_bounds=k_bounds,
        precision=params.precision,
        mesh_params=mesh_params,
        approx_method=params.approx_method,
    )
    k_surface = 2 * fm.pi * src.freq_hz / float(env.wave_speed_profile(0.0))
    contour_k_z = probe.lower_nlbc_contour_kz()
    if rough.method == 'fit':
        z0_shift = rough.auto_reference_height(contour_k_z, smooth_refl_coef_func, k_surface)
        # smallest lift first; lifts are in units of sigma since the zeros of
        # the continued 1 + R scale with it, and a large lift makes the phase
        # exp(2 i k_z z0) too oscillatory for the polynomial
        lifts = [0.0] + [a * rough.rms_height_m for a in (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 7, 10)]
        # Vertical wavenumbers at the surface of the modes that carry the field:
        #  * a trapping layer returns grazing angles below sqrt(2 dM 1e-6), dM the
        #    M-deficit relative to the surface; R~ must be passive there above the
        #    real axis (growing modes would resonate with the layer);
        #  * any refractive gradient g = d(m^2)/dz carries creeping (diffraction)
        #    modes at |k_z| = sqrt(|a_n|) (k^2 g)^(1/3), arg k_z = -60 deg (a_n the
        #    Airy zeros); the first modes, sqrt(|a_n|) <= 3, dominate beyond the
        #    horizon. Range-decaying modes see R~ below the real axis, where it is
        #    fitted to the continued R (creeping sector and trapped-mode strip).
        z_env = np.linspace(0.0, params.max_height_m, 4001)
        m_env = np.asarray(env.M_profile(z_env))
        k_z_trap = k_surface * fm.sqrt(2e-6 * max(0.0, float(m_env[0] - m_env.min())))
        near = z_env <= min(100.0, params.max_height_m)
        grad = 2e-6 * abs(float(np.polyfit(z_env[near], m_env[near], 1)[0]))
        k_z_creep = 3.0 * (k_surface ** 2 * grad) ** (1 / 3)
        lifts = [z for z in lifts if z < z0_shift] + [z0_shift]
        k_g = rough.grazing_k_z(k_surface)
        if 0 < k_g < k_z_creep:
            import warnings
            warnings.warn(f"constant grazing impedance ends at k_z = {k_g:.3g} 1/m, inside the creeping-mode range "
                          f"|k_z| <= {k_z_creep:.3g} 1/m: the field far beyond the horizon depends on the blend "
                          f"shape there; a larger grazing_angle_deg extends the constant impedance over it")
        fit_info = None
        for degree in (rough.fit_degree, 2 * rough.fit_degree):
            fitter = replace(rough, fit_degree=degree)
            for z0 in lifts:
                lower_refl_coef_func, fit_info = fitter.fitted_reflection_coef_func(
                    smooth_refl_coef_func, k_surface, contour_k_z, kz_max, z0, k_z_trap, k_z_creep)
                if fit_info['fit_error'] <= rough.fit_tol:
                    break
            if fit_info['fit_error'] <= rough.fit_tol:
                fit_info['k_z_passive'] = k_z_trap
                fit_info['k_z_creep'] = k_z_creep
                break
        else:
            # at z0_shift the plain shifted coefficient is itself admissible
            import warnings
            warnings.warn(f"rough-surface reflection fit failed (error {fit_info['fit_error']:.2e}); "
                          f"falling back to method='shift' (z0 = {z0_shift:.2f} m)")
            z0 = z0_shift
            lower_refl_coef_func = rough.reflection_coef_func(
                smooth_refl_coef_func, 2 * fm.pi * src.freq_hz / float(env.wave_speed_profile(z0)), z0)
            fit_info = {'fit_error': 0.0, 'reference_height_m': z0, 'fallback': 'shift'}
    else:
        z0 = rough.reference_height_m
        if z0 is None:
            z0 = rough.auto_reference_height(contour_k_z, smooth_refl_coef_func, k_surface)
        fit_info = None
    if z0 >= src.height_m:
        raise ValueError(f"rough-sea reference height {z0:.2f} m is not below the source height {src.height_m} m")
    if rough.method == 'shift':
        # the NLBC converts k_z <-> grazing angle with the wavenumber at the boundary
        k_surface = 2 * fm.pi * src.freq_hz / float(env.wave_speed_profile(z0))
        lower_refl_coef_func = rough.reflection_coef_func(smooth_refl_coef_func, k_surface, z0)
    model = RationalHelmholtzPropagator(
        order=probe.order,
        beta=probe.beta,
        dx_m=probe.dx_m,
        dz_m=probe.dz_m,
        x_n=probe.x_n,
        z_n=probe.z_n,
        x_grid_scale=probe.x_grid_scale,
        z_grid_scale=probe.z_grid_scale,
        freq_hz=src.freq_hz,
        wave_speed=ProxyWaveSpeedModel(env),
        coefs=probe.coefs_t,
        # with the BC at the mean level the grid and profile are the probe's,
        # so is the upper NLBC; a lift shifts the profile and needs a new one
        upper_nlbc_coefs=probe.upper_nlbc_coefs if z0 == 0.0 else None,
        lower_refl_coef_func=lower_refl_coef_func,
        approx_method=probe.approx_method,
        xi_bounds=probe.xi_bounds,
        z_offset_m=z0,
    )
    model.rough_sea_fit_info = fit_info
    # reflection coefficient of the lower NLBC vs complex grazing angle (deg), for diagnostics
    model.rough_sea_refl_coef_func = lower_refl_coef_func
    return model


def _terrain_samples(terrain):
    if not isinstance(terrain, PiecewiseLinearTerrainModel):
        raise NotImplementedError("terrain_method='transform' needs a PiecewiseLinearTerrainModel terrain")
    return np.asarray(terrain.x_grid_m, dtype=float), np.asarray(terrain.height, dtype=float)


def create_rwp_tf_model(src: RWPGaussSourceModel, env: TroposphereModel, params: RWPComputationalParams):
    """Rational propagator in orthogonal terrain-following coordinates
    (``params.terrain_method = 'transform'``), see
    :class:`pywaveprop.helmholtz_tf_jax.TerrainFollowingHelmholtzPropagator`.

    The ground is perfectly conducting (Dirichlet (H) or Neumann (V) condition
    on the terrain) or, when ``env.ground_material`` is given, the Leontovich
    impedance condition ``d psi / dn + alpha psi = 0`` along the terrain normal.  ``params.max_height_m`` is the physical height of the
    upper boundary; the maximum propagation angle is raised by twice the
    maximum terrain slope (rays reflected by the slopes).
    """
    if params.approx_method not in ('pade', 'ratinterp'):
        raise NotImplementedError("terrain_method='transform' is implemented for the rational propagator")
    if getattr(env, 'vegetation', None) is not None or getattr(env, 'sea_roughness', None) is not None:
        raise NotImplementedError("terrain_method='transform' supports a perfectly conducting or an impedance "
                                  "ground only (set vegetation / sea_roughness to None)")
    if hasattr(env.N_profile, 'columns'):
        raise NotImplementedError("terrain_method='transform' needs a range-independent N profile")
    params, kz_max, (k_min, k_max), _, _, mesh_params = _rwp_model_setup(src, env, params)
    x_t, h_t = _terrain_samples(env.terrain)
    wavelength = 3e8 / src.freq_hz
    relief = float(h_t.max() - h_t.min())
    smoothing = params.terrain_smoothing_m
    if smoothing is None:
        smoothing = max(float(np.median(np.diff(x_t))), wavelength)
    transition = params.terrain_transition_m or max(2.0 * relief, 30.0 * wavelength, 4.0 * smoothing)
    z_ref_opts = {'mean': float(np.mean(h_t)), 'max': float(h_t.max()), 'min': float(h_t.min())}

    def build(z_ref):
        z_ref_m = z_ref_opts[z_ref] if isinstance(z_ref, str) else float(z_ref)
        tm = OrthogonalTerrainMap(x_t, h_t, x_max_m=params.max_range_m, kind='half_space',
                                  transition_m=transition, z_ref_m=z_ref_m, smoothing_m=smoothing)
        tm.metrics(np.linspace(0, params.max_range_m, 64), np.linspace(0, tm.eta_lab, 64))
        return tm

    try:
        tm = build(params.terrain_z_ref)
    except ValueError:
        # z_ref = max(terrain) keeps Z_eta >= 1 for any transition height
        tm = build('max')
    slope = tm.max_slope_deg()
    if params.terrain_slope_aperture:
        max_angle = float(np.degrees(np.arcsin(kz_max / (2 * np.pi * src.freq_hz / 3e8))))
        kz_max = 2 * np.pi * src.freq_hz / 3e8 * np.sin(np.radians(min(max_angle + 2 * slope, 85.0)))
    print(f'terrain: relief {relief:.1f} m, max slope {slope:.1f} deg, transition {transition:.1f} m, '
          f'smoothing {smoothing:.1f} m, z_ref {tm.z_off:.1f} m')

    mesh_params = HelmholtzMeshParams2D(
        x_size_m=mesh_params.x_size_m, z_size_m=params.max_height_m - tm.z_off,
        dx_output_m=mesh_params.dx_output_m, x_n_upper_bound=mesh_params.x_n_upper_bound,
        dz_output_m=mesh_params.dz_output_m, z_n_upper_bound=mesh_params.z_n_upper_bound)
    ws = ProxyWaveSpeedModel(env)
    k0 = 2 * np.pi * src.freq_hz
    if env.ground_material is not None:
        bc = dict(lower_bc='impedance', lower_impedance_alpha=impedance_alpha(
            2 * fm.pi * src.freq_hz / 3E8, env.ground_material.complex_permittivity(src.freq_hz),
            src.polarization))
    else:
        bc = dict(lower_bc='dirichlet' if src.polarization.upper() == 'H' else 'neumann')

    def k2_above(x, z):
        return (k0 / np.asarray(ws(jnp.asarray(z)))) ** 2

    return TerrainFollowingHelmholtzPropagator.create(
        freq_hz=src.freq_hz, terrain_map=tm, medium=TFMedium(k2_above=k2_above),
        kz_max=kz_max, k_bounds=(k_min, k_max), precision=params.precision, mesh_params=mesh_params,
        rational_approx_order=params.rational_approx_order, approx_method=params.approx_method,
        beta_mode=params.terrain_beta_mode, n_beta_classes=params.terrain_beta_classes,
        absorb_m=params.terrain_absorb_m, absorb_stretch=params.terrain_absorb_stretch,
        upper_stretch=params.terrain_upper_stretch,
        upper_stretch_start_m=getattr(params, 'terrain_upper_stretch_start_m', None),
        upper_stretch_ramp_m=getattr(params, 'terrain_upper_stretch_ramp_m', None),
        upper_stretch_s_top=getattr(params, 'terrain_upper_stretch_s_top', None),
        upper_bc='dirichlet' if params.terrain_absorb_m else 'nlbc', **bc)


def rwp_tf_forward_task(src: RWPGaussSourceModel, env: TroposphereModel,
                        params: RWPComputationalParams) -> TerrainFollowingRWPField:
    """Forward task in terrain-following coordinates.

    The source height ``src.height_m`` is absolute (above sea level, as the
    terrain); the aperture and its image are built relative to the ground at
    the transmitter, ``z - h(0)``.
    """
    model = create_rwp_tf_model(src, env, params)
    x0, z0 = model.initial_points()
    h0 = float(model.terrain_map.boundary(np.zeros(1))[0])
    src_local = deepcopy(src)
    src_local.height_m = src.height_m - h0
    init = src_local.aperture(jnp.asarray(z0 - h0))
    f = model.compute(init)
    x_phys, z_phys = model.output_metrics()
    return TerrainFollowingRWPField(
        x_grid=model.x_output_grid(), z_grid=model._z_identity(model.eta_computational_grid()[::model.z_grid_scale]),
        freq_hz=src.freq_hz, field=f, x_phys=x_phys, z_phys=z_phys,
        terrain_x=model.terrain_map.x_fine, terrain_z=model.terrain_map.boundary_fine)


def create_rwp_model(src: RWPGaussSourceModel, env: TroposphereModel, params: RWPComputationalParams):
    """Create a JAX-based Helmholtz propagator configured for tropospheric RWP.

    ``params.approx_method`` selects the solver: ``'pade'`` / ``'ratinterp'``
    give the rational finite-difference propagator
    (:class:`RationalHelmholtzPropagator`), ``'ssf'`` the split-step Fourier one
    (:class:`SSFHelmholtzPropagator`), ``'rassf'`` the rank-adaptive split-step
    Fourier one (:class:`RankAdaptiveSSFPropagator`) and ``'ssw'`` the
    split-step wavelet one (:class:`SSWHelmholtzPropagator`). All expose the
    same ``compute`` / grid interface.

    Parameters
    ----------
    src : RWPGaussSourceModel
        Source/antenna model.
    env : TroposphereModel
        Tropospheric environment model.
    params : RWPComputationalParams
        Computational parameters.

    Returns
    -------
    RationalHelmholtzPropagator, SSFHelmholtzPropagator or SSWHelmholtzPropagator
        Configured propagator ready for computation.
    """
    if params.approx_method == 'ssf':
        return create_rwp_ssf_model(src, env, params)
    if params.approx_method == 'rassf':
        return create_rwp_rassf_model(src, env, params)
    if params.approx_method == 'ssw':
        return create_rwp_ssw_model(src, env, params)
    if params.terrain_method == 'transform' and env.terrain is not None:
        return create_rwp_tf_model(src, env, params)

    params, kz_max, k_bounds, lower_refl_coef_func, lower_layer, mesh_params = _rwp_model_setup(src, env, params)
    rough = getattr(env, 'sea_roughness', None)
    if rough is not None:
        return _create_rough_sea_rwp_model(src, env, params, kz_max, k_bounds, lower_refl_coef_func, mesh_params)
    k_min, k_max = k_bounds
    return RationalHelmholtzPropagator.create(
        rational_approx_order=params.rational_approx_order,
        freq_hz=src.freq_hz,
        wave_speed=ProxyWaveSpeedModel(env),
        kz_max=kz_max,
        k_bounds=(k_min, k_max),
        precision=params.precision,
        mesh_params=mesh_params,
        lower_terrain=env.terrain,
        lower_refl_coef_func=lower_refl_coef_func,
        approx_method=params.approx_method,
        lower_layer=lower_layer,
    )


def rwp_forward_task(src: RWPGaussSourceModel, env: TroposphereModel, params: RWPComputationalParams) -> RWPField:
    """Compute tropospheric radio wave propagation using the JAX-based solver.

    This is the primary high-level function for tropospheric RWP computation.

    Parameters
    ----------
    src : RWPGaussSourceModel
        Source/antenna model with frequency, height, and beam parameters.
    env : TroposphereModel
        Tropospheric environment model with N-profile and optional terrain.
    params : RWPComputationalParams
        Computational parameters including grid size and precision.

    Returns
    -------
    RWPField
        Computed electromagnetic field with methods for path_loss, horizontal extraction, etc.
    Set ``params.approx_method`` to ``'ssf'`` / ``'rassf'`` / ``'ssw'`` for the split-step Fourier /
    rank-adaptive split-step Fourier / wavelet solvers.
    """
    if params.terrain_method == 'transform' and env.terrain is not None:
        return rwp_tf_forward_task(src, env, params)
    model = create_rwp_model(src, env, params)
    init = src.aperture(model.z_computational_grid())
    f = model.compute(init)
    return RWPField(
        x_grid=model.x_output_grid(),
        z_grid=model.z_output_grid(),
        freq_hz=src.freq_hz,
        field=f,
    )
