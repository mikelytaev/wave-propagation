import cmath as cm
import math as fm
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import List

import jax
import numpy as np
from jax import tree_util
from jax import numpy as jnp

from experimental.helmholtz_jax import AbstractWaveSpeedModel, LinearSlopeWaveSpeedModel, \
    RationalHelmholtzPropagator, RegularGrid, HelmholtzMeshParams2D
from uwa.field import AcousticPressureField


@dataclass
class UWAComputationalParams:
    max_range_m: float
    max_depth_m: float = None
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


class UWAGaussSourceModel:

    def __init__(self, *, freq_hz, depth_m, beam_width_deg, elevation_angle_deg=0, multiplier=1.0):
        self.freq_hz = freq_hz
        self.depth_m = depth_m
        self.beam_width_deg = beam_width_deg
        self.elevation_angle_deg = elevation_angle_deg
        self.multiplier = multiplier

    def aperture(self, k0, z):
        elevation_angle_rad = fm.radians(self.elevation_angle_deg)
        ww = cm.sqrt(2 * cm.log(2)) / (k0 * cm.sin(fm.radians(self.beam_width_deg) / 2))
        return jnp.array(self.multiplier / (cm.sqrt(cm.pi) * ww) * jnp.exp(-1j * k0 * jnp.sin(elevation_angle_rad) * z)
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


@dataclass
class UnderwaterLayerModel:
    height_m: float
    sound_speed_profile_m_s: AbstractWaveSpeedModel = LinearSlopeWaveSpeedModel(c0=1500, slope_degrees=0)
    density: float = 1.0
    attenuation_dm_lambda: float = 0.0

    def _tree_flatten(self):
        dynamic = (self.sound_speed_profile_m_s, self.density, self.attenuation_dm_lambda)
        static = {
            'height_m': self.height_m,
        }
        return dynamic, static

    @classmethod
    def _tree_unflatten(cls, static, dynamic):
        unf = cls(sound_speed_profile_m_s=dynamic[0], density=dynamic[1], attenuation_dm_lambda=dynamic[2], **static)
        return unf


tree_util.register_pytree_node(UnderwaterLayerModel,
                               UnderwaterLayerModel._tree_flatten,
                               UnderwaterLayerModel._tree_unflatten)


@dataclass
class UnderwaterEnvironmentModel:
    layers: List[UnderwaterLayerModel] = None

    def ssp(self, z_grid: jax.Array):
        cur_depth = 0.0
        res = jnp.empty(shape=z_grid.shape, dtype=float)
        for layer in self.layers:
            layer_height = layer.height_m if layer != self.layers[-1] else np.inf
            local_filter = (cur_depth <= z_grid) & (z_grid < cur_depth + layer_height)
            local_z = z_grid[local_filter] - cur_depth
            res = res.at[local_filter].set(layer.sound_speed_profile_m_s(local_z))
            cur_depth += layer_height
        return res

    @partial(jax.jit, static_argnums=(1,))
    def ssp_jit(self, z_grid: RegularGrid):
        cur_depth = 0.0
        res = jnp.empty(shape=z_grid.n, dtype=float)
        for layer in self.layers:
            layer_height = layer.height_m if layer != self.layers[-1] else 1E6
            a_i, b_i = z_grid.interval_indexes(cur_depth, cur_depth + layer_height)
            res = res.at[a_i:b_i].set(layer.sound_speed_profile_m_s(z_grid.array_grid(a_i, b_i) - cur_depth))
            cur_depth += layer_height
        return res

    def rho(self, z_grid: jax.Array):
        cur_depth = 0.0
        res = jnp.empty(shape=z_grid.shape, dtype=float)
        for layer in self.layers:
            layer_height = layer.height_m if layer != self.layers[-1] else np.inf
            local_filter = (cur_depth <= z_grid) & (z_grid < cur_depth + layer_height)
            res = res.at[local_filter].set(layer.density)
            cur_depth += layer_height
        return res

    @partial(jax.jit, static_argnums=(1,))
    def rho_jit(self, z_grid: RegularGrid):
        cur_depth = 0.0
        res = jnp.empty(shape=z_grid.n, dtype=float)
        for layer in self.layers:
            layer_height = layer.height_m if layer != self.layers[-1] else 1E6
            a_i, b_i = z_grid.interval_indexes(cur_depth, cur_depth + layer_height)
            res = res.at[a_i:b_i].set(layer.density)
            cur_depth += layer.height_m
        return res

    def max_depth_m(self):
        return sum([layer.height_m for layer in self.layers[:-1]])

    def _tree_flatten(self):
        dynamic = (self.layers,)
        static = {

        }
        return dynamic, static

    @classmethod
    def _tree_unflatten(cls, static, dynamic):
        unf = cls(dynamic[0])
        return unf


tree_util.register_pytree_node(UnderwaterEnvironmentModel,
                               UnderwaterEnvironmentModel._tree_flatten,
                               UnderwaterEnvironmentModel._tree_unflatten)


class ProxyWaveSpeedModel(AbstractWaveSpeedModel):

    def __init__(self, uwem: UnderwaterEnvironmentModel):
        self.uwem = uwem

    def __call__(self, z):
        return self.uwem.ssp(z)

    def on_regular_grid(self, z_grid: RegularGrid):
        return self.uwem.ssp_jit(z_grid)

    def _tree_flatten(self):
        dynamic = (self.uwem,)
        static = {}
        return dynamic, static

    @classmethod
    def _tree_unflatten(cls, static, dynamic):
        unf = cls(uwem=dynamic[0], **static)
        return unf


tree_util.register_pytree_node(ProxyWaveSpeedModel,
                               ProxyWaveSpeedModel._tree_flatten,
                               ProxyWaveSpeedModel._tree_unflatten)


class ProxyRhoModel(AbstractWaveSpeedModel):

    def __init__(self, uwem: UnderwaterEnvironmentModel):
        self.uwem = uwem

    def __call__(self, z):
        return self.uwem.rho(z)

    def on_regular_grid(self, z_grid: RegularGrid):
        return self.uwem.rho_jit(z_grid)

    def _tree_flatten(self):
        dynamic = (self.uwem,)
        static = {}
        return dynamic, static

    @classmethod
    def _tree_unflatten(cls, static, dynamic):
        unf = cls(uwem=dynamic[0], **static)
        return unf


tree_util.register_pytree_node(ProxyRhoModel,
                               ProxyRhoModel._tree_flatten,
                               ProxyRhoModel._tree_unflatten)


def minmax_k(src: UWAGaussSourceModel, env: UnderwaterEnvironmentModel):
    z_grid = jnp.linspace(0.0, env.max_depth_m(), 1000)
    k_func_arr = 2 * fm.pi * src.freq_hz / env.ssp(z_grid)
    k_min = min(k_func_arr)
    k_max = max(k_func_arr)
    print(f'k_min: {k_min}, k_max: {k_max}')
    return k_min, k_max


def uwa_get_model(src: UWAGaussSourceModel, env: UnderwaterEnvironmentModel, params: UWAComputationalParams) -> RationalHelmholtzPropagator:
    params = deepcopy(params)

    max_angle_deg = src.max_angle_deg()
    c0 = float(env.ssp(jnp.array([src.depth_m]))[0])
    k0 = 2 * fm.pi * src.freq_hz / c0
    kz_max = k0 * fm.sin(fm.radians(max_angle_deg))

    max_bottom_height = env.max_depth_m()
    if params.max_depth_m:
        params.max_depth_m = max(params.max_depth_m, max_bottom_height*1.1)
    else:
        params.max_depth_m = max_bottom_height * 1.1

    return RationalHelmholtzPropagator.create(
        freq_hz=src.freq_hz,
        wave_speed=ProxyWaveSpeedModel(env),
        rho=ProxyRhoModel(env),
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
        ),
    )


def uwa_forward_task(src: UWAGaussSourceModel, env: UnderwaterEnvironmentModel, params: UWAComputationalParams) -> AcousticPressureField:
    model = uwa_get_model(src, env, params)
    c0 = float(env.ssp(jnp.array([src.depth_m]))[0])
    k0 = 2 * fm.pi * src.freq_hz / c0
    init = src.aperture(k0, model.z_computational_grid())
    f = model.compute(init)
    return AcousticPressureField(freq_hz=src.freq_hz, x_grid=model.x_output_grid(), z_grid=model.z_output_grid(),
                                 field=f)


tree_util.register_pytree_node(UWAGaussSourceModel,
                               UWAGaussSourceModel._tree_flatten,
                               UWAGaussSourceModel._tree_unflatten)


