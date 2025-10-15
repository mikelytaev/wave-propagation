from copy import deepcopy
from dataclasses import dataclass, field

from pywaveprop.experimental.helmholtz_jax import RegularGrid, AbstractWaveSpeedModel, RationalHelmholtzPropagator, \
    HelmholtzMeshParams2D
import jax
import jax.numpy as jnp
import math as fm
from jax import tree_util


@dataclass
class RWPComputationalParams:
    max_range_m: float
    max_height_m: float = None
    rational_approx_order = (7, 8)
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

    def __init__(self, *, freq_hz, height_m, beam_width_deg, elevation_angle_deg=0, multiplier=1.0):
        self.freq_hz = freq_hz
        self.height_m = height_m
        self.beam_width_deg = beam_width_deg
        self.elevation_angle_deg = elevation_angle_deg
        self.multiplier = multiplier

    def aperture(self, z):
        k0 = 2 * jnp.pi * self.freq_hz / 3E8
        elevation_angle_rad = jnp.radians(self.elevation_angle_deg)
        ww = jnp.sqrt(2 * jnp.log(2)) / (k0 * jnp.sin(jnp.radians(self.beam_width_deg) / 2))
        return jnp.array(self.multiplier / (jnp.sqrt(jnp.pi) * ww) * jnp.exp(-1j * k0 * jnp.sin(elevation_angle_rad) * z)
                         * jnp.exp(-((z - self.height_m) / ww) ** 2), dtype=complex)

    def max_angle_deg(self):
        return self.beam_width_deg + abs(self.elevation_angle_deg)

    def _tree_flatten(self):
        dynamic = (self.height_m, self.beam_width_deg, self.elevation_angle_deg, self.multiplier)
        static = {
            'freq_hz': self.freq_hz
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


class AbstractTerrainModel:

    def __call__(self, z):
        pass


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
class TroposphereModel:
    N_profile: AbstractNProfileModel = field(default_factory=EmptyNProfileModel)
    M0: float = 320
    slope: float = (1 / 6371000) * 1E6
    terrain: AbstractTerrainModel = None

    def M_profile(self, z):
        return self.N_profile(z) + self.M0 + self.slope*z

    def wave_speed_profile(self, z):
        return 3E8 / (1 + self.M_profile(z) * 1E-6)

    def max_height_m(self):
        return self.N_profile.max_height_m()

    def _tree_flatten(self):
        dynamic = (self.N_profile, self.M0, self.slope)
        static = {}
        return dynamic, static

    @classmethod
    def _tree_unflatten(cls, static, dynamic):
        unf = cls(N_profile=dynamic[0], M0=dynamic[1], slope=dynamic[2])
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


def minmax_k(src: RWPGaussSourceModel, env: TroposphereModel):
    z_grid = jnp.linspace(0.0, env.max_height_m(), 1000)
    k_func_arr = 2 * fm.pi * src.freq_hz / env.wave_speed_profile(z_grid)
    k_min = min(k_func_arr)
    k_max = max(k_func_arr)
    print(f'k_min: {k_min}, k_max: {k_max}')
    return k_min, k_max


def create_rwp_model(src: RWPGaussSourceModel, env: TroposphereModel, params: RWPComputationalParams) -> RationalHelmholtzPropagator:
    params = deepcopy(params)

    max_angle_deg = src.max_angle_deg()
    k0 = 2 * jnp.pi * src.freq_hz / 3E8
    kz_max = k0 * jnp.sin(jnp.radians(max_angle_deg))

    max_height_m = env.max_height_m()
    if params.max_height_m:
        params.max_height_m = max(params.max_height_m, max_height_m*1.1)
    else:
        params.max_height_m = max_height_m * 1.1

    return RationalHelmholtzPropagator.create(
        freq_hz=src.freq_hz,
        wave_speed=ProxyWaveSpeedModel(env),
        kz_max=kz_max,
        k_bounds=minmax_k(src, env),
        precision=params.precision,
        mesh_params=HelmholtzMeshParams2D(
            x_size_m=params.max_range_m,
            z_size_m=params.max_height_m,
            dx_output_m=params.dx_m,
            x_n_upper_bound=params.x_output_points,
            dz_output_m=params.dz_m,
            z_n_upper_bound=params.z_output_points,
        ),
        lower_terrain=env.terrain
    )
