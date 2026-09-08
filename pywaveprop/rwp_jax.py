"""
JAX-based tropospheric radio wave propagation module.

This is the primary implementation of the tropospheric radio wave propagation
solver using JAX for GPU-accelerated computation.
"""
import cmath as cm
from copy import deepcopy
from dataclasses import dataclass, field

from pywaveprop.helmholtz_jax import RegularGrid, AbstractWaveSpeedModel, RationalHelmholtzPropagator, \
    AbstractTerrainModel, PiecewiseLinearTerrainModel, LowerLayerModel
from pywaveprop.helmholtz_common import HelmholtzMeshParams2D
from pywaveprop.helmholtz_ssf_jax import SSFHelmholtzPropagator, impedance_alpha
from pywaveprop.propagators._utils import reflection_coef
from pywaveprop.rwp_field import RWPField
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
    approx_method = 'pade'  # 'pade' / 'ratinterp' (rational FD) or 'ssf' (split-step Fourier)
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

    def M_profile(self, z):
        return self.N_profile(z) + self.M0 + self.slope*z

    def wave_speed_profile(self, z):
        return 3E8 / (1 + self.M_profile(z) * 1E-6)

    def max_height_m(self):
        return self.N_profile.max_height_m()

    def _tree_flatten(self):
        dynamic = (self.N_profile, self.M0, self.slope, self.terrain, self.vegetation)
        static = {
            'ground_material': self.ground_material,
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
    k_func_arr = 2 * fm.pi * src.freq_hz / env.wave_speed_profile(z_grid)
    k_min = min(k_func_arr)
    k_max = max(k_func_arr)
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


def create_rwp_ssf_model(src: RWPGaussSourceModel, env: TroposphereModel, params: RWPComputationalParams,
                         **ssf_kwargs) -> SSFHelmholtzPropagator:
    """Create a JAX-based split-step Fourier propagator configured for tropospheric RWP.

    The lower boundary follows the source polarization: a perfectly conducting
    ground (``env.ground_material is None``) gives a Dirichlet (H) or Neumann (V)
    condition, a finite-permittivity ground gives the Leontovich impedance
    condition handled by the discrete mixed Fourier transform. Extra keyword
    arguments (``dx_m``, ``dz_m``, ``abs_layer_scale``, ``spectral_margin``) are
    forwarded to :meth:`SSFHelmholtzPropagator.create`.
    """
    params, kz_max, k_bounds, _, lower_layer, mesh_params = _rwp_model_setup(src, env, params)
    if env.ground_material is not None:
        k0 = 2 * fm.pi * src.freq_hz / 3E8
        lower_bc = 'impedance'
        alpha = impedance_alpha(k0, env.ground_material.complex_permittivity(src.freq_hz), src.polarization)
    else:
        lower_bc = 'dirichlet' if src.polarization.upper() == 'H' else 'neumann'
        alpha = None
    return SSFHelmholtzPropagator.create(
        freq_hz=src.freq_hz,
        wave_speed=ProxyWaveSpeedModel(env),
        kz_max=kz_max,
        k_bounds=k_bounds,
        precision=params.precision,
        mesh_params=mesh_params,
        lower_terrain=env.terrain,
        lower_bc=lower_bc,
        lower_impedance_alpha=alpha,
        lower_layer=lower_layer,
        **ssf_kwargs,
    )


def create_rwp_model(src: RWPGaussSourceModel, env: TroposphereModel, params: RWPComputationalParams):
    """Create a JAX-based Helmholtz propagator configured for tropospheric RWP.

    ``params.approx_method`` selects the solver: ``'pade'`` / ``'ratinterp'``
    give the rational finite-difference propagator
    (:class:`RationalHelmholtzPropagator`), ``'ssf'`` the split-step Fourier one
    (:class:`SSFHelmholtzPropagator`). Both expose the same ``compute`` /
    grid interface.

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
    RationalHelmholtzPropagator or SSFHelmholtzPropagator
        Configured propagator ready for computation.
    """
    if params.approx_method == 'ssf':
        return create_rwp_ssf_model(src, env, params)

    params, kz_max, k_bounds, lower_refl_coef_func, lower_layer, mesh_params = _rwp_model_setup(src, env, params)
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
    Set ``params.approx_method = 'ssf'`` to use the split-step Fourier solver.
    """
    model = create_rwp_model(src, env, params)
    init = src.aperture(model.z_computational_grid())
    f = model.compute(init)
    return RWPField(
        x_grid=model.x_output_grid(),
        z_grid=model.z_output_grid(),
        freq_hz=src.freq_hz,
        field=f,
    )
