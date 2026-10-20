"""
JAX-based underwater acoustics propagation module.

This is the primary implementation of the underwater acoustics solver
using JAX for GPU-accelerated computation.
"""
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

from pywaveprop.helmholtz_jax import AbstractWaveSpeedModel, LinearSlopeWaveSpeedModel, \
    RationalHelmholtzPropagator, RegularGrid, PiecewiseLinearTerrainModel
from pywaveprop.helmholtz_common import HelmholtzMeshParams2D
from pywaveprop.uwa_utils import UWAComputationalParams
from pywaveprop.uwa.field import AcousticPressureField
from pywaveprop.helmholtz_tf_jax import TerrainFollowingHelmholtzPropagator, TFMedium
from pywaveprop.terrain_transform import OrthogonalTerrainMap


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
    bathymetry: PiecewiseLinearTerrainModel = None

    @staticmethod
    def _effective_speed(c_real, attenuation_dm_lambda):
        """Return complex effective sound speed incorporating attenuation.

        attenuation_dm_lambda is attenuation in dB per wavelength.
        The conversion follows: c_eff = c / (1 + j*eta*alpha) where
        eta = 1/(40*pi*log10(e)).  When alpha=0 the result is real-valued.
        """
        eta = 1.0 / (40.0 * np.pi * np.log10(np.e))
        return c_real / (1.0 + 1j * eta * attenuation_dm_lambda)

    def ssp(self, z_grid: jax.Array):
        cur_depth = 0.0
        res = jnp.empty(shape=z_grid.shape, dtype=complex)
        for layer in self.layers:
            layer_height = layer.height_m if layer != self.layers[-1] else np.inf
            local_filter = (cur_depth <= z_grid) & (z_grid < cur_depth + layer_height)
            local_z = z_grid[local_filter] - cur_depth
            c = layer.sound_speed_profile_m_s(local_z)
            c_eff = self._effective_speed(c, layer.attenuation_dm_lambda)
            res = res.at[local_filter].set(c_eff)
            cur_depth += layer_height
        return res

    @partial(jax.jit, static_argnums=(1,))
    def ssp_jit(self, z_grid: RegularGrid):
        cur_depth = 0.0
        res = jnp.empty(shape=z_grid.n, dtype=complex)
        for layer in self.layers:
            layer_height = layer.height_m if layer != self.layers[-1] else 1E6
            a_i, b_i = z_grid.interval_indexes(cur_depth, cur_depth + layer_height)
            c = layer.sound_speed_profile_m_s(z_grid.array_grid(a_i, b_i) - cur_depth)
            c_eff = self._effective_speed(c, layer.attenuation_dm_lambda)
            res = res.at[a_i:b_i].set(c_eff)
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
        dynamic = (self.layers, self.bathymetry)
        static = {}
        return dynamic, static

    @classmethod
    def _tree_unflatten(cls, static, dynamic):
        unf = cls(layers=dynamic[0], bathymetry=dynamic[1])
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
    k_func_arr = jnp.abs(2 * fm.pi * src.freq_hz / env.ssp(z_grid))
    k_min = float(min(k_func_arr))
    k_max = float(max(k_func_arr))
    print(f'k_min: {k_min}, k_max: {k_max}')
    return k_min, k_max


def uwa_get_model(src: UWAGaussSourceModel, env: UnderwaterEnvironmentModel, params: UWAComputationalParams) -> RationalHelmholtzPropagator:
    """Create a JAX-based rational Helmholtz propagator configured for underwater acoustics.

    Parameters
    ----------
    src : UWAGaussSourceModel
        Acoustic source model.
    env : UnderwaterEnvironmentModel
        Underwater environment with layered sound speed profile.
    params : UWAComputationalParams
        Computational parameters.

    Returns
    -------
    RationalHelmholtzPropagator
        Configured propagator ready for computation.
    """
    params = deepcopy(params)

    max_angle_deg = src.max_angle_deg()
    c0 = float(jnp.real(env.ssp(jnp.array([src.depth_m]))[0]))
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
        upper_terrain=env.bathymetry,
        kz_max=kz_max,
        k_bounds=minmax_k(src, env),
        precision=params.precision,
        rational_approx_order=params.rational_approx_order,
        mesh_params=HelmholtzMeshParams2D(
            x_size_m=params.max_range_m,
            z_size_m=params.max_depth_m,
            dx_output_m=params.dx_m,
            x_n_upper_bound=params.x_output_points,
            dz_output_m=params.dz_m,
            z_n_upper_bound=params.z_output_points,
        ),
    )


def uwa_tf_get_model(src: UWAGaussSourceModel, env: UnderwaterEnvironmentModel,
                     params: UWAComputationalParams) -> TerrainFollowingHelmholtzPropagator:
    """Rational propagator in orthogonal terrain-following coordinates
    (``params.terrain_method = 'transform'``).

    ``env.layers[0]`` is the water column (sound speed as a function of depth,
    density, attenuation) and ``env.layers[1]`` the bottom half-space, whose
    sound speed profile is a function of the depth *below the seabed*
    ``env.bathymetry``.  The seabed is the coordinate line ``eta = eta_b``, so the
    density / sound-speed jump follows the bathymetry exactly; the sea surface is
    pressure-release.  The bottom of the domain (``params.max_depth_m``) lies in
    the homogeneous part of the bottom, where the nonlocal boundary condition is
    imposed.
    """
    if env.bathymetry is None or len(env.layers) != 2:
        raise NotImplementedError("terrain_method='transform' needs bathymetry, a water layer and one bottom layer")
    params = deepcopy(params)
    water, bottom = env.layers
    x_b = np.asarray(env.bathymetry.x_grid_m, dtype=float)
    d_b = np.asarray(env.bathymetry.height, dtype=float)
    c_w0 = float(jnp.real(water.sound_speed_profile_m_s(jnp.array([src.depth_m]))[0]))
    wavelength = c_w0 / src.freq_hz
    relief = float(d_b.max() - d_b.min())
    smoothing = params.terrain_smoothing_m
    if smoothing is None:
        smoothing = max(float(np.median(np.diff(x_b))), wavelength)
    transition = params.terrain_transition_m or max(1.5 * relief, 20.0 * wavelength, 4.0 * smoothing)
    tm = OrthogonalTerrainMap(x_b, d_b, x_max_m=params.max_range_m, kind='strip',
                              transition_m=transition, eta_b_m=float(d_b.max()), smoothing_m=smoothing)
    slope = tm.max_slope_deg()
    min_depth = tm.eta_lab + max(10.0 * wavelength, 0.05 * tm.eta_lab)
    params.max_depth_m = max(params.max_depth_m or 0.0, min_depth)
    print(f'bathymetry: relief {relief:.1f} m, max slope {slope:.1f} deg, sediment transition {transition:.1f} m, '
          f'smoothing {smoothing:.1f} m, domain depth {params.max_depth_m:.1f} m')

    omega = 2 * fm.pi * src.freq_hz
    eff = UnderwaterEnvironmentModel._effective_speed

    def k2_above(x, z):
        c = np.asarray(water.sound_speed_profile_m_s(jnp.asarray(z)))
        return (omega / eff(c, water.attenuation_dm_lambda)) ** 2

    def k2_below(x, z, depth_below_seabed):
        c = np.asarray(bottom.sound_speed_profile_m_s(jnp.asarray(np.maximum(depth_below_seabed, 0.0))))
        return (omega / eff(c, bottom.attenuation_dm_lambda)) ** 2

    medium = TFMedium(k2_above=k2_above, k2_below=k2_below, rho_above=float(water.density),
                      rho_below=float(bottom.density))
    # The aperture is defined in the water: the longitudinal wavenumbers of the
    # field are bounded below by min_water(k) cos(aperture), whatever the (faster)
    # bottom, so the wavenumber interval comes from the water column only.
    z_probe = np.linspace(0.0, float(d_b.max()), 500)
    c_water = np.abs(np.asarray(water.sound_speed_profile_m_s(jnp.asarray(z_probe))))
    k_bounds = (omega / float(c_water.max()), omega / float(c_water.min()))
    k0 = omega / c_w0
    aperture = min(src.max_angle_deg() + 2 * slope, 85.0, 0.98 * fm.degrees(fm.asin(min(k_bounds[0] / k0, 1.0))))
    kz_max = k0 * fm.sin(fm.radians(aperture))
    return TerrainFollowingHelmholtzPropagator.create(
        freq_hz=src.freq_hz, terrain_map=tm, medium=medium, kz_max=kz_max, k_bounds=k_bounds,
        precision=params.precision, rational_approx_order=params.rational_approx_order,
        mesh_params=HelmholtzMeshParams2D(
            x_size_m=params.max_range_m, z_size_m=params.max_depth_m,
            dx_output_m=params.dx_m, x_n_upper_bound=params.x_output_points,
            dz_output_m=params.dz_m, z_n_upper_bound=params.z_output_points),
        lower_bc='dirichlet')


def uwa_tf_forward_task(src: UWAGaussSourceModel, env: UnderwaterEnvironmentModel,
                        params: UWAComputationalParams) -> AcousticPressureField:
    """Forward task in terrain-following coordinates; the result carries the
    physical coordinates ``x_phys`` / ``z_phys`` of the output nodes."""
    model = uwa_tf_get_model(src, env, params)
    c0 = float(jnp.real(env.layers[0].sound_speed_profile_m_s(jnp.array([src.depth_m]))[0]))
    k0 = 2 * fm.pi * src.freq_hz / c0
    _, z0 = model.initial_points()
    f = model.compute(src.aperture(k0, jnp.asarray(z0)))
    res = AcousticPressureField(freq_hz=src.freq_hz, x_grid=model.x_output_grid(),
                                z_grid=model.eta_computational_grid()[::model.z_grid_scale], field=f)
    res.x_phys, res.z_phys = model.output_metrics()
    res.seabed_x, res.seabed_z = model.terrain_map.x_fine, model.terrain_map.boundary_fine
    return res


def uwa_forward_task(src: UWAGaussSourceModel, env: UnderwaterEnvironmentModel, params: UWAComputationalParams) -> AcousticPressureField:
    """Compute underwater acoustic propagation using the JAX-based solver.

    This is the primary high-level function for underwater acoustics computation.

    Parameters
    ----------
    src : UWAGaussSourceModel
        Acoustic source model with frequency, depth, and beam parameters.
    env : UnderwaterEnvironmentModel
        Layered underwater environment with sound speed profiles and densities.
    params : UWAComputationalParams
        Computational parameters including grid size and precision.

    Returns
    -------
    AcousticPressureField
        Computed acoustic pressure field.
    """
    if getattr(params, 'terrain_method', 'staircase') == 'transform' and env.bathymetry is not None:
        return uwa_tf_forward_task(src, env, params)
    model = uwa_get_model(src, env, params)
    c0 = float(jnp.real(env.ssp(jnp.array([src.depth_m]))[0]))
    k0 = 2 * fm.pi * src.freq_hz / c0
    init = src.aperture(k0, model.z_computational_grid())
    f = model.compute(init)
    return AcousticPressureField(freq_hz=src.freq_hz, x_grid=model.x_output_grid(), z_grid=model.z_output_grid(),
                                 field=f)


tree_util.register_pytree_node(UWAGaussSourceModel,
                               UWAGaussSourceModel._tree_flatten,
                               UWAGaussSourceModel._tree_unflatten)


