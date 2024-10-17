from copy import deepcopy
from dataclasses import dataclass

from experimental.helmholtz_jax import RegularGrid, AbstractWaveSpeedModel, RationalHelmholtzPropagator, \
    HelmholtzMeshParams2D
import jax
import jax.numpy as jnp
import math as fm


@dataclass
class ComputationalParams:
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


class GaussSourceModel:

    def __init__(self, *, freq_hz, depth_m, beam_width_deg, elevation_angle_deg=0, multiplier=1.0):
        self.freq_hz = freq_hz
        self.depth_m = depth_m
        self.beam_width_deg = beam_width_deg
        self.elevation_angle_deg = elevation_angle_deg
        self.multiplier = multiplier

    def aperture(self, z):
        k0 = 2 * jnp.pi * self.freq_hz / 3E8
        elevation_angle_rad = jnp.radians(self.elevation_angle_deg)
        ww = jnp.sqrt(2 * jnp.log(2)) / (k0 * jnp.sin(jnp.radians(self.beam_width_deg) / 2))
        return jnp.array(self.multiplier / (jnp.sqrt(jnp.pi) * ww) * jnp.exp(-1j * k0 * jnp.sin(elevation_angle_rad) * z)
                         * jnp.exp(-((z - self.depth_m) / ww) ** 2), dtype=complex)

    def max_angle_deg(self):
        return self.beam_width_deg + abs(self.elevation_angle_deg)

    def _tree_flatten(self):
        dynamic = (self.depth_m, self.beam_width_deg, self.elevation_angle_deg, self.multiplier)
        static = {
            'freq_hz': self.freq_hz
        }
        return dynamic, static

    @classmethod
    def _tree_unflatten(cls, static, dynamic):
        return cls(depth_m=dynamic[0], beam_width_deg=dynamic[1], elevation_angle_deg=dynamic[2], multiplier=dynamic[3], **static)


class AbstractNProfileModel:

    def __call__(self, z):
        pass

    def max_height_m(self):
        pass

    def on_regular_grid(self, z_grid: RegularGrid):
        return self(z_grid.array_grid(0, 100000000))


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


class PiecewiseLinearNProfileModel(AbstractNProfileModel):

    def __init__(self, z_grid_m: jax.Array, N_vals: jax.Array):
        self.z_grid_m = z_grid_m
        self.N_vals = N_vals

    def __call__(self, z):
        return jnp.interp(z, self.z_grid_m, self.N_vals,
                          left='extrapolate', right='extrapolate')

    def max_height_m(self):
        return max(self.z_grid_m) + 1


class TroposphereModel:
    N_profile: AbstractNProfileModel
    M0: float = 320
    slope: float = (2 / 6371000) * 1E6

    def M_profile(self, z):
        return self.N_profile(z) + self.M0 + self.slope*z

    def wave_speed_profile(self, z):
        return 3E8 / (1 + self.M_profile(z) * 1E-6)

    def max_height_m(self):
        self.N_profile.max_height_m()


class ProxyWaveSpeedModel(AbstractWaveSpeedModel):

    def __init__(self, tm: TroposphereModel):
        self.tm = tm

    def __call__(self, z):
        return self.tm.N_profile

    def on_regular_grid(self, z_grid: RegularGrid):
        return self(z_grid.array_grid(0, 100000000))


def minmax_k(src: GaussSourceModel, env: TroposphereModel):
    z_grid = jnp.linspace(0.0, env.max_height_m(), 1000)
    k_func_arr = 2 * fm.pi * src.freq_hz / env.wave_speed_profile(z_grid)
    k_min = min(k_func_arr)
    k_max = max(k_func_arr)
    print(f'k_min: {k_min}, k_max: {k_max}')
    return k_min, k_max


def create_rwp_model(src: GaussSourceModel, env: TroposphereModel, params: ComputationalParams) -> RationalHelmholtzPropagator:
    params = deepcopy(params)

    max_angle_deg = src.max_angle_deg()
    k0 = 2 * jnp.pi * src.freq_hz / 3E8
    kz_max = k0 * jnp.sin(jnp.radians(max_angle_deg))

    max_height_m = env.max_height_m()
    if params.max_depth_m:
        params.max_depth_m = max(params.max_height_m, max_height_m*1.1)
    else:
        params.max_depth_m = max_height_m * 1.1

    return RationalHelmholtzPropagator.create(
        freq_hz=src.freq_hz,
        wave_speed=ProxyWaveSpeedModel(env),
        kz_max=kz_max,
        k_bounds=minmax_k(src, env),
        precision=params.precision,
        mesh_params=HelmholtzMeshParams2D(
            x_size_m=params.max_range_m,
            z_size_m=params.max_depth_m,
            dx_output_m=params.dx_m,
            x_n_upper_bound=params.x_output_points,
            dz_output_m=params.dz_m,
            z_n_upper_bound=params.z_output_points,
        )
    )
