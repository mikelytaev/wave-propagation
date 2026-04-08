"""
Tests for the JAX-based underwater acoustics propagation module.

Tests cover:
- Source models (UWAGaussSourceModel)
- Layer models (UnderwaterLayerModel)
- Environment models (UnderwaterEnvironmentModel) - SSP, density, multi-layer
- Full UWA forward propagation
- Shallow water scenario
- Deep water (Munk profile-like) scenario
- Multi-layer with density contrast
"""
import unittest
import math as fm

import jax
import jax.numpy as jnp
import numpy as np

from pywaveprop.uwa_jax import (
    UWAGaussSourceModel,
    UnderwaterLayerModel,
    UnderwaterEnvironmentModel,
    ProxyWaveSpeedModel,
    ProxyRhoModel,
    uwa_get_model,
    uwa_forward_task,
    minmax_k,
)
from pywaveprop.uwa_utils import UWAComputationalParams
from pywaveprop.helmholtz_jax import (
    ConstWaveSpeedModel,
    LinearSlopeWaveSpeedModel,
    PiecewiseLinearWaveSpeedModel,
    PiecewiseLinearTerrainModel,
    RegularGrid,
)
from pywaveprop.uwa.field import AcousticPressureField


class TestUWAGaussSourceModel(unittest.TestCase):

    def test_creation(self):
        src = UWAGaussSourceModel(
            freq_hz=500.0,
            depth_m=50.0,
            beam_width_deg=30.0,
        )
        self.assertEqual(src.freq_hz, 500.0)
        self.assertEqual(src.depth_m, 50.0)
        self.assertEqual(src.beam_width_deg, 30.0)

    def test_max_angle(self):
        src = UWAGaussSourceModel(
            freq_hz=500.0,
            depth_m=50.0,
            beam_width_deg=30.0,
            elevation_angle_deg=5.0,
        )
        self.assertAlmostEqual(src.max_angle_deg(), 35.0)

    def test_aperture_shape(self):
        src = UWAGaussSourceModel(
            freq_hz=500.0,
            depth_m=50.0,
            beam_width_deg=30.0,
        )
        k0 = 2 * fm.pi * 500.0 / 1500.0
        z = jnp.linspace(0, 200, 100)
        aperture = src.aperture(k0, z)
        self.assertEqual(aperture.shape, (100,))
        self.assertTrue(jnp.all(jnp.isfinite(aperture)))

    def test_aperture_peak_near_depth(self):
        src = UWAGaussSourceModel(
            freq_hz=500.0,
            depth_m=100.0,
            beam_width_deg=30.0,
        )
        k0 = 2 * fm.pi * 500.0 / 1500.0
        z = jnp.linspace(0, 200, 1000)
        aperture = src.aperture(k0, z)
        peak_idx = jnp.argmax(jnp.abs(aperture))
        peak_z = float(z[peak_idx])
        self.assertAlmostEqual(peak_z, 100.0, delta=1.0)

    def test_pytree_roundtrip(self):
        src = UWAGaussSourceModel(
            freq_hz=500.0,
            depth_m=50.0,
            beam_width_deg=30.0,
            elevation_angle_deg=2.0,
            multiplier=0.5,
        )
        leaves, treedef = jax.tree_util.tree_flatten(src)
        src2 = treedef.unflatten(leaves)
        k0 = 2 * fm.pi * 500.0 / 1500.0
        z = jnp.linspace(0, 200, 50)
        np.testing.assert_allclose(src.aperture(k0, z), src2.aperture(k0, z))


class TestUnderwaterLayerModel(unittest.TestCase):

    def test_default_layer(self):
        layer = UnderwaterLayerModel(height_m=200.0)
        self.assertEqual(layer.height_m, 200.0)
        self.assertEqual(layer.density, 1.0)
        self.assertEqual(layer.attenuation_dm_lambda, 0.0)

    def test_custom_layer(self):
        ssp = ConstWaveSpeedModel(c0=1600.0)
        layer = UnderwaterLayerModel(
            height_m=100.0,
            sound_speed_profile_m_s=ssp,
            density=1.8,
            attenuation_dm_lambda=0.5,
        )
        self.assertEqual(layer.density, 1.8)
        self.assertAlmostEqual(layer.attenuation_dm_lambda, 0.5)

    def test_pytree_roundtrip(self):
        layer = UnderwaterLayerModel(
            height_m=200.0,
            sound_speed_profile_m_s=ConstWaveSpeedModel(c0=1500.0),
            density=1.0,
        )
        leaves, treedef = jax.tree_util.tree_flatten(layer)
        layer2 = treedef.unflatten(leaves)
        self.assertEqual(layer.height_m, layer2.height_m)
        self.assertEqual(layer.density, layer2.density)


class TestUnderwaterEnvironmentModel(unittest.TestCase):

    def _make_simple_env(self):
        """Two-layer environment: water + sediment."""
        water = UnderwaterLayerModel(
            height_m=200.0,
            sound_speed_profile_m_s=ConstWaveSpeedModel(c0=1500.0),
            density=1.0,
        )
        sediment = UnderwaterLayerModel(
            height_m=100.0,
            sound_speed_profile_m_s=ConstWaveSpeedModel(c0=1700.0),
            density=1.8,
            attenuation_dm_lambda=0.5,
        )
        return UnderwaterEnvironmentModel(layers=[water, sediment])

    def test_creation(self):
        env = self._make_simple_env()
        self.assertEqual(len(env.layers), 2)

    def test_max_depth(self):
        env = self._make_simple_env()
        self.assertEqual(env.max_depth_m(), 200.0)

    def test_ssp_single_layer(self):
        water = UnderwaterLayerModel(
            height_m=200.0,
            sound_speed_profile_m_s=ConstWaveSpeedModel(c0=1500.0),
            density=1.0,
        )
        env = UnderwaterEnvironmentModel(layers=[water])
        z = jnp.linspace(0, 199, 50)
        ssp = env.ssp(z)
        np.testing.assert_allclose(jnp.real(ssp), 1500.0, atol=1e-5)

    def test_ssp_two_layers(self):
        env = self._make_simple_env()
        z_water = jnp.array([100.0])
        z_sediment = jnp.array([250.0])
        ssp_w = env.ssp(z_water)
        ssp_s = env.ssp(z_sediment)
        np.testing.assert_allclose(jnp.real(ssp_w), 1500.0, atol=1e-5)
        # Sediment has attenuation → complex, but real part ≈ 1700
        np.testing.assert_allclose(jnp.real(ssp_s), 1700.0, atol=1.0)

    def test_ssp_jit(self):
        env = self._make_simple_env()
        z_grid = RegularGrid(start=0.0, dx=1.0, n=300)
        ssp = env.ssp_jit(z_grid)
        self.assertEqual(ssp.shape, (300,))
        self.assertTrue(jnp.all(jnp.isfinite(ssp)))

    def test_rho_two_layers(self):
        env = self._make_simple_env()
        z_water = jnp.array([100.0])
        z_sediment = jnp.array([250.0])
        rho_w = env.rho(z_water)
        rho_s = env.rho(z_sediment)
        np.testing.assert_allclose(rho_w, 1.0, atol=1e-5)
        np.testing.assert_allclose(rho_s, 1.8, atol=1e-5)

    def test_rho_jit(self):
        env = self._make_simple_env()
        z_grid = RegularGrid(start=0.0, dx=1.0, n=300)
        rho = env.rho_jit(z_grid)
        self.assertEqual(rho.shape, (300,))
        self.assertTrue(jnp.all(jnp.isfinite(rho)))

    def test_piecewise_linear_ssp(self):
        """Test with a piecewise linear sound speed profile."""
        z_ssp = jnp.array([0.0, 100.0, 200.0])
        c_vals = jnp.array([1520.0, 1480.0, 1510.0])
        ssp_model = PiecewiseLinearWaveSpeedModel(z_grid_m=z_ssp, sound_speed=c_vals)

        water = UnderwaterLayerModel(
            height_m=200.0,
            sound_speed_profile_m_s=ssp_model,
            density=1.0,
        )
        env = UnderwaterEnvironmentModel(layers=[water])
        z = jnp.array([0.0, 50.0, 100.0, 150.0, 200.0])
        ssp = env.ssp(z)
        np.testing.assert_allclose(float(jnp.real(ssp[0])), 1520.0, atol=1e-3)
        np.testing.assert_allclose(float(jnp.real(ssp[1])), 1500.0, atol=1e-3)
        np.testing.assert_allclose(float(jnp.real(ssp[2])), 1480.0, atol=1e-3)

    def test_pytree_roundtrip(self):
        env = self._make_simple_env()
        leaves, treedef = jax.tree_util.tree_flatten(env)
        env2 = treedef.unflatten(leaves)
        z = jnp.linspace(0, 250, 50)
        np.testing.assert_allclose(env.ssp(z), env2.ssp(z), atol=1e-5)


class TestProxyModels(unittest.TestCase):

    def test_proxy_wave_speed(self):
        water = UnderwaterLayerModel(
            height_m=200.0,
            sound_speed_profile_m_s=ConstWaveSpeedModel(c0=1500.0),
        )
        env = UnderwaterEnvironmentModel(layers=[water])
        proxy = ProxyWaveSpeedModel(env)
        z = jnp.array([100.0])
        np.testing.assert_allclose(proxy(z), 1500.0, atol=1e-5)

    def test_proxy_rho(self):
        water = UnderwaterLayerModel(height_m=200.0, density=1.0)
        sediment = UnderwaterLayerModel(height_m=100.0, density=1.8)
        env = UnderwaterEnvironmentModel(layers=[water, sediment])
        proxy = ProxyRhoModel(env)
        z_w = jnp.array([100.0])
        z_s = jnp.array([250.0])
        np.testing.assert_allclose(proxy(z_w), 1.0, atol=1e-5)
        np.testing.assert_allclose(proxy(z_s), 1.8, atol=1e-5)


class TestMinmaxK(unittest.TestCase):

    def test_basic(self):
        src = UWAGaussSourceModel(freq_hz=500.0, depth_m=50.0, beam_width_deg=30.0)
        water = UnderwaterLayerModel(
            height_m=200.0,
            sound_speed_profile_m_s=ConstWaveSpeedModel(c0=1500.0),
        )
        sediment = UnderwaterLayerModel(
            height_m=100.0,
            sound_speed_profile_m_s=ConstWaveSpeedModel(c0=1700.0),
        )
        env = UnderwaterEnvironmentModel(layers=[water, sediment])
        k_min, k_max = minmax_k(src, env)
        self.assertGreater(float(k_min), 0)
        self.assertGreater(float(k_max), float(k_min))


class TestUWAForwardTask(unittest.TestCase):

    def test_shallow_water_const_ssp(self):
        """Test propagation in shallow water with constant sound speed."""
        src = UWAGaussSourceModel(
            freq_hz=500.0,
            depth_m=50.0,
            beam_width_deg=30.0,
        )
        water = UnderwaterLayerModel(
            height_m=200.0,
            sound_speed_profile_m_s=ConstWaveSpeedModel(c0=1500.0),
            density=1.0,
        )
        sediment = UnderwaterLayerModel(
            height_m=100.0,
            sound_speed_profile_m_s=ConstWaveSpeedModel(c0=1700.0),
            density=1.8,
            attenuation_dm_lambda=0.5,
        )
        env = UnderwaterEnvironmentModel(layers=[water, sediment])
        params = UWAComputationalParams(
            max_range_m=5000.0,
            dx_m=10.0,
            dz_m=1.0,
        )
        result = uwa_forward_task(src, env, params)
        self.assertIsInstance(result, AcousticPressureField)
        self.assertGreater(len(result.x_grid), 0)
        self.assertGreater(len(result.z_grid), 0)
        self.assertTrue(np.all(np.isfinite(np.asarray(result.field))))

    def test_shallow_water_linear_ssp(self):
        """Test with linear sound speed profile (positive gradient)."""
        src = UWAGaussSourceModel(
            freq_hz=500.0,
            depth_m=50.0,
            beam_width_deg=30.0,
        )
        water = UnderwaterLayerModel(
            height_m=200.0,
            sound_speed_profile_m_s=LinearSlopeWaveSpeedModel(c0=1500.0, slope_degrees=0.1),
            density=1.0,
        )
        sediment = UnderwaterLayerModel(
            height_m=100.0,
            sound_speed_profile_m_s=ConstWaveSpeedModel(c0=1700.0),
            density=1.8,
        )
        env = UnderwaterEnvironmentModel(layers=[water, sediment])
        params = UWAComputationalParams(
            max_range_m=5000.0,
            dx_m=10.0,
            dz_m=1.0,
        )
        result = uwa_forward_task(src, env, params)
        self.assertIsInstance(result, AcousticPressureField)
        self.assertTrue(np.all(np.isfinite(np.asarray(result.field))))

    def test_deep_water_piecewise_ssp(self):
        """Test with a Munk-like piecewise linear sound speed profile."""
        z_ssp = jnp.array([0.0, 200.0, 500.0, 1000.0, 1500.0])
        c_vals = jnp.array([1520.0, 1500.0, 1480.0, 1500.0, 1530.0])
        ssp_model = PiecewiseLinearWaveSpeedModel(z_grid_m=z_ssp, sound_speed=c_vals)

        src = UWAGaussSourceModel(
            freq_hz=100.0,
            depth_m=500.0,
            beam_width_deg=20.0,
        )
        water = UnderwaterLayerModel(
            height_m=1500.0,
            sound_speed_profile_m_s=ssp_model,
            density=1.0,
        )
        sediment = UnderwaterLayerModel(
            height_m=500.0,
            sound_speed_profile_m_s=ConstWaveSpeedModel(c0=1700.0),
            density=1.8,
        )
        env = UnderwaterEnvironmentModel(layers=[water, sediment])
        params = UWAComputationalParams(
            max_range_m=10000.0,
            dx_m=50.0,
            dz_m=5.0,
        )
        result = uwa_forward_task(src, env, params)
        self.assertIsInstance(result, AcousticPressureField)
        self.assertTrue(np.all(np.isfinite(np.asarray(result.field))))
        # Field should not be identically zero
        self.assertGreater(float(np.max(np.abs(np.asarray(result.field)))), 0)

    def test_output_points_more_efficient_than_dx_m(self):
        """x_n_upper_bound/z_n_upper_bound let the optimizer pick a grid that
        does not need to evenly divide a fixed output spacing.

        With dx_m=20, dx_computational must be 20/N for integer N — so the
        optimizer is forced to refine to 10 m even when ~20 m would suffice.
        With x_output_points giving the same dx_max, the optimizer can use
        the natural ~20 m step directly, halving the work.
        """
        src = UWAGaussSourceModel(
            freq_hz=100.0,
            depth_m=50.0,
            beam_width_deg=30.0,
        )
        water = UnderwaterLayerModel(
            height_m=200.0,
            sound_speed_profile_m_s=ConstWaveSpeedModel(c0=1500.0),
            density=1.0,
        )
        sediment = UnderwaterLayerModel(
            height_m=100.0,
            sound_speed_profile_m_s=ConstWaveSpeedModel(c0=1700.0),
            density=1.8,
        )
        env = UnderwaterEnvironmentModel(layers=[water, sediment])

        # Case A: strict dx_m=20 forces dx_comp = 20/N
        params_dx = UWAComputationalParams(
            max_range_m=5000.0,
            dx_m=20.0,
            dz_m=1.0,
        )
        params_dx.rational_approx_order = (7, 8)
        model_dx = uwa_get_model(src, env, params_dx)

        # Case B: x_output_points=500 gives dx_max=20 but lets optimizer
        # pick any dx_comp <= 20, not just divisors
        params_pts = UWAComputationalParams(
            max_range_m=5000.0,
            x_output_points=500,
            z_output_points=600,
        )
        params_pts.rational_approx_order = (7, 8)
        model_pts = uwa_get_model(src, env, params_pts)

        # Both should have similar dx_max ≈ 20 m, but the points-bounded
        # version should use a coarser computational grid (fewer x_n)
        self.assertLess(
            model_pts.x_n, model_dx.x_n,
            f"x_output_points should give fewer grid points "
            f"({model_pts.x_n}) than dx_m ({model_dx.x_n})"
        )
        # And the computational dx is NOT an integer divisor of any "round" output dx
        self.assertNotAlmostEqual(model_pts.dx_m, model_dx.dx_m, places=2)
        # Both should still produce finite, sensible results
        result_dx = uwa_forward_task(src, env, params_dx)
        result_pts = uwa_forward_task(src, env, params_pts)
        self.assertTrue(np.all(np.isfinite(np.asarray(result_dx.field))))
        self.assertTrue(np.all(np.isfinite(np.asarray(result_pts.field))))

    def test_nearest_value(self):
        """Test AcousticPressureField.nearest_value method."""
        src = UWAGaussSourceModel(
            freq_hz=500.0,
            depth_m=50.0,
            beam_width_deg=30.0,
        )
        water = UnderwaterLayerModel(
            height_m=200.0,
            sound_speed_profile_m_s=ConstWaveSpeedModel(c0=1500.0),
        )
        sediment = UnderwaterLayerModel(
            height_m=100.0,
            sound_speed_profile_m_s=ConstWaveSpeedModel(c0=1700.0),
            density=1.8,
        )
        env = UnderwaterEnvironmentModel(layers=[water, sediment])
        params = UWAComputationalParams(
            max_range_m=5000.0,
            dx_m=10.0,
            dz_m=1.0,
        )
        result = uwa_forward_task(src, env, params)
        x, z, val = result.nearest_value(2500.0, 50.0)
        self.assertIsNotNone(val)
        self.assertTrue(np.isfinite(val))


class TestUWAComputationalParams(unittest.TestCase):

    def test_valid_dx_dz(self):
        params = UWAComputationalParams(
            max_range_m=5000.0,
            dx_m=50.0,
            dz_m=1.0,
        )
        self.assertEqual(params.max_range_m, 5000.0)

    def test_valid_output_points(self):
        params = UWAComputationalParams(
            max_range_m=5000.0,
            x_output_points=100,
            z_output_points=200,
        )
        self.assertEqual(params.x_output_points, 100)

    def test_invalid_both_x(self):
        with self.assertRaises(ValueError):
            UWAComputationalParams(
                max_range_m=5000.0,
                dx_m=50.0,
                x_output_points=100,
                dz_m=1.0,
            )

    def test_invalid_neither_x(self):
        with self.assertRaises(ValueError):
            UWAComputationalParams(
                max_range_m=5000.0,
                dz_m=1.0,
            )


class TestUWABathymetry(unittest.TestCase):
    """Test irregular bathymetry (upper_terrain) support."""

    def test_flat_bottom_mask(self):
        """Flat bottom at 300m: water above, sediment below."""
        terrain = PiecewiseLinearTerrainModel(
            x_grid_m=jnp.array([0.0, 5000.0]),
            height=jnp.array([300.0, 300.0]),
        )
        env = UnderwaterEnvironmentModel(
            layers=[
                UnderwaterLayerModel(height_m=300, sound_speed_profile_m_s=ConstWaveSpeedModel(c0=1500)),
                UnderwaterLayerModel(height_m=200, sound_speed_profile_m_s=ConstWaveSpeedModel(c0=1700), density=1.8),
            ],
            bathymetry=terrain,
        )
        src = UWAGaussSourceModel(freq_hz=100, depth_m=50, beam_width_deg=10)
        params = UWAComputationalParams(max_range_m=5000, x_output_points=30, z_output_points=30)
        params.rational_approx_order = (1, 2)

        model = uwa_get_model(src, env, params)
        mask = np.asarray(model.lower_terrain_mask)
        z_grid = np.arange(model.z_n) * model.dz_m
        mid_x = model.x_n // 2

        # Water (z < 300) should be unmasked
        self.assertTrue(np.all(mask[mid_x, z_grid < 280] == 1 + 0j))
        # Sediment (z > 300) should be masked
        self.assertTrue(np.all(mask[mid_x, z_grid > 320] == 0 + 0j))

    def test_irregular_bottom_propagation(self):
        """Gaussian seamount: field should be nonzero in water, zero below bottom."""
        def munk_profile(z, ref_speed=1500, ref_depth=1300, eps=0.00737):
            z_ = 2 * (z - ref_depth) / ref_depth
            return ref_speed * (1 + eps * (z_ - 1 + np.exp(-z_)))

        z_ssp = jnp.linspace(0, 5000, 100)
        c_vals = jnp.array(munk_profile(np.asarray(z_ssp)))
        ssp = PiecewiseLinearWaveSpeedModel(z_grid_m=z_ssp, sound_speed=c_vals)

        max_range_m = 200e3
        x_bathy = jnp.linspace(0, max_range_m, 100)
        depth_bathy = 5000 - 3000 * jnp.exp(-(x_bathy - 100e3) ** 2 / 1e9)
        terrain = PiecewiseLinearTerrainModel(x_grid_m=x_bathy, height=depth_bathy)

        env = UnderwaterEnvironmentModel(
            layers=[
                UnderwaterLayerModel(height_m=5000, sound_speed_profile_m_s=ssp, density=1.0),
                UnderwaterLayerModel(height_m=500, sound_speed_profile_m_s=ConstWaveSpeedModel(c0=1700), density=1.8),
            ],
            bathymetry=terrain,
        )
        src = UWAGaussSourceModel(freq_hz=50, depth_m=100, beam_width_deg=3, multiplier=5)
        params = UWAComputationalParams(max_range_m=max_range_m, max_depth_m=5500, dx_m=100, dz_m=5)
        result = uwa_forward_task(src, env, params)

        self.assertTrue(np.all(np.isfinite(np.asarray(result.field))))

        # At 100km where bottom = 2000m: check field above/below
        x_mid = np.argmin(np.abs(result.x_grid - 100e3))
        z = np.asarray(result.z_grid)
        f_slice = np.abs(np.asarray(result.field[x_mid, :]))
        water_max = np.max(f_slice[z < 1800])
        sediment_max = np.max(f_slice[z > 2200])
        self.assertGreater(water_max, 0, "Field in water should be nonzero")
        self.assertLess(sediment_max, water_max * 0.1,
                        "Field below bottom should be much smaller than in water")


class TestUWAAttenuation(unittest.TestCase):
    """Test that attenuation_dm_lambda is correctly applied."""

    def test_attenuation_makes_ssp_complex(self):
        """SSP should be complex when attenuation is nonzero."""
        water = UnderwaterLayerModel(height_m=200,
                                     sound_speed_profile_m_s=ConstWaveSpeedModel(c0=1500), density=1.0)
        sediment = UnderwaterLayerModel(height_m=100,
                                        sound_speed_profile_m_s=ConstWaveSpeedModel(c0=1700),
                                        density=1.8, attenuation_dm_lambda=0.5)
        env = UnderwaterEnvironmentModel(layers=[water, sediment])

        z_water = jnp.array([100.0])
        z_sediment = jnp.array([250.0])

        c_water = env.ssp(z_water)
        c_sediment = env.ssp(z_sediment)

        # Water layer has no attenuation → real
        self.assertAlmostEqual(float(jnp.imag(c_water[0])), 0.0, places=10)
        # Sediment has attenuation → complex with negative imaginary part
        self.assertLess(float(jnp.imag(c_sediment[0])), 0.0,
                        "Attenuation should make sound speed have negative imaginary part")
        # Real part should still be close to c_bottom
        self.assertAlmostEqual(float(jnp.real(c_sediment[0])), 1700.0, delta=1.0)

    def test_zero_attenuation_real(self):
        """Zero attenuation should give real SSP."""
        layer = UnderwaterLayerModel(height_m=200,
                                     sound_speed_profile_m_s=ConstWaveSpeedModel(c0=1500),
                                     attenuation_dm_lambda=0.0)
        env = UnderwaterEnvironmentModel(layers=[layer])
        z = jnp.array([100.0])
        c = env.ssp(z)
        self.assertAlmostEqual(float(jnp.imag(c[0])), 0.0, places=10)

    def test_attenuation_increases_loss(self):
        """Field should decay faster with higher bottom attenuation."""
        src = UWAGaussSourceModel(freq_hz=50, depth_m=100, beam_width_deg=3, multiplier=5)

        def munk_profile(z):
            z_ = 2 * (z - 1300) / 1300
            return 1500 * (1 + 0.00737 * (z_ - 1 + np.exp(-z_)))
        z_ssp = jnp.linspace(0, 5000, 100)
        c_ssp = jnp.array(munk_profile(np.asarray(z_ssp)))
        ssp = PiecewiseLinearWaveSpeedModel(z_grid_m=z_ssp, sound_speed=c_ssp)

        fields = []
        for attn in [0.0, 0.5, 2.0]:
            env = UnderwaterEnvironmentModel(layers=[
                UnderwaterLayerModel(height_m=5000, sound_speed_profile_m_s=ssp, density=1.0),
                UnderwaterLayerModel(height_m=500, sound_speed_profile_m_s=ConstWaveSpeedModel(c0=1700),
                                     density=1.5, attenuation_dm_lambda=attn),
            ])
            params = UWAComputationalParams(max_range_m=200e3, max_depth_m=5500, dx_m=100, dz_m=5)
            result = uwa_forward_task(src, env, params)
            # Max field amplitude at far range
            x_far = np.argmin(np.abs(result.x_grid - 150e3))
            far_field = float(np.max(np.abs(np.asarray(result.field[x_far, :]))))
            fields.append(far_field)
            print(f"  attn={attn}: far field max = {far_field:.4e}")

        # Higher attenuation → weaker field at far range
        self.assertGreater(fields[0], fields[1],
                           "attn=0 should have stronger field than attn=0.5")
        self.assertGreater(fields[1], fields[2],
                           "attn=0.5 should have stronger field than attn=2.0")


if __name__ == '__main__':
    unittest.main()
