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
from scipy.optimize import differential_evolution

from experimental.grid_optimizer import get_optimal_grid
from experimental.helmholtz_jax import AbstractWaveSpeedModel, LinearSlopeWaveSpeedModel, \
    RationalHelmholtzPropagator, RegularGrid
from uwa.field import AcousticPressureField


@dataclass
class ComputationalParams:
    max_range_m: float
    max_depth_m: float = None
    rational_approx_order = (7, 8)
    dx_m: float = None
    dz_m: float = None
    x_output_points: int = None
    z_output_points: int = None
    precision: float = 0.01


class GaussSourceModel:

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


def check_computational_params(params: ComputationalParams):
    if params.x_output_points is None and params.dx_m is None:
        raise ValueError("x output grid (x_output_points or dx_m) is not specified!")
    if params.x_output_points is not None and params.dx_m is not None:
        raise ValueError("only one x output grid parameter (x_output_points or dx_m) should be specified!")

    if params.z_output_points is None and params.dz_m is None:
        raise ValueError("z output grid (z_output_points or dz_m) is not specified!")
    if params.z_output_points is not None and params.dz_m is not None:
        raise ValueError("only one z output grid parameter (z_output_points or dz_m) should be specified!")


def minmax_k(src: GaussSourceModel, env: UnderwaterEnvironmentModel):
    k_func = lambda z: 2 * fm.pi * src.freq_hz / env.ssp(z)
    result_ga = differential_evolution(
        func=k_func,
        bounds=[(0, 1000)],
        popsize=30,
        disp=False,
        recombination=1,
        strategy='randtobest1exp',
        tol=1e-5,
        maxiter=10000,
        polish=False
    )
    k_min = result_ga.fun

    k_func = lambda z: -2 * fm.pi * src.freq_hz / env.ssp(z)
    result_ga = differential_evolution(
        func=k_func,
        bounds=[(0, 1000)],
        popsize=30,
        disp=False,
        recombination=1,
        strategy='randtobest1exp',
        tol=1e-5,
        maxiter=10000,
        polish=False
    )
    k_max = -result_ga.fun

    print(f'k_min: {k_min}, k_max: {k_max}')
    return k_min, k_max


def uwa_get_model(src: GaussSourceModel, env: UnderwaterEnvironmentModel, params: ComputationalParams) -> RationalHelmholtzPropagator:
    check_computational_params(params)

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

    k_min, k_max = minmax_k(src, env)

    if params.x_output_points:
        params.dx_m = params.max_range_m / (params.x_output_points - 1)
    if params.z_output_points:
        params.dz_m = params.max_depth_m / (params.z_output_points - 1)
    beta, dx_computational, dz_computational = get_optimal_grid(
        kz_max, k_min, k_max, params.precision / params.max_range_m,
        dx_max=params.dx_m,
        dz_max=params.dz_m)
    if params.dx_m:
        dx_computational = params.dx_m / fm.ceil(params.dx_m / dx_computational)
    if params.dz_m:
        dz_computational = params.dz_m / fm.ceil(params.dz_m / dx_computational)

    params.max_range_m = fm.ceil(params.max_range_m / dx_computational) * dx_computational
    params.max_depth_m = fm.ceil(params.max_depth_m / dz_computational) * dz_computational

    if not params.x_output_points:
        params.x_output_points = round(params.max_range_m / params.dx_m) + 1
    if not params.z_output_points:
        params.z_output_points = round(params.max_depth_m / params.dz_m) + 1

    x_grid_scale = round(params.dx_m / dx_computational)
    z_grid_scale = round(params.dz_m / dz_computational)
    x_computational_points = params.x_output_points * x_grid_scale
    z_computational_points = params.z_output_points * z_grid_scale

    x_computational_grid = jnp.linspace(0, params.max_range_m, x_computational_points)
    z_computational_grid = jnp.linspace(0, params.max_depth_m, z_computational_points)

    x_output_grid = jnp.linspace(0, params.max_range_m, params.x_output_points)
    z_output_grid = jnp.linspace(0, params.max_depth_m, params.z_output_points)

    print(f'beta: {beta}, dx: {dx_computational}, dz: {dz_computational}')

    model = RationalHelmholtzPropagator(
        beta=beta,
        dx_m=dx_computational,
        dz_m=dz_computational,
        x_n=len(x_computational_grid),
        z_n=len(z_computational_grid),
        x_grid_scale=x_grid_scale,
        z_grid_scale=z_grid_scale,
        order=(7, 8),
        wave_speed=ProxyWaveSpeedModel(env),
        rho=ProxyRhoModel(env),
        freq_hz=src.freq_hz
    )

    return model


def uwa_forward_task(src: GaussSourceModel, env: UnderwaterEnvironmentModel, params: ComputationalParams) -> AcousticPressureField:
    model = uwa_get_model(src, env, params)
    c0 = float(env.ssp(jnp.array([src.depth_m]))[0])
    k0 = 2 * fm.pi * src.freq_hz / c0
    init = src.aperture(k0, model.z_computational_grid())
    f = model.compute(init)
    return AcousticPressureField(freq_hz=src.freq_hz, x_grid=model.x_output_grid(), z_grid=model.z_output_grid(),
                                 field=f)


tree_util.register_pytree_node(GaussSourceModel,
                               GaussSourceModel._tree_flatten,
                               GaussSourceModel._tree_unflatten)


