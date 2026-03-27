"""
Tests for the JAX-based tropospheric radio wave propagation module.

Tests cover:
- Source models (RWPGaussSourceModel)
- N-profile models (Empty, Evaporation duct, PiecewiseLinear, Sum, Mult)
- TroposphereModel
- Full RWP forward propagation
- Path loss computation
- Horizontal field extraction
- Terrain effects
"""
import unittest
import math as fm

import jax
import jax.numpy as jnp
import numpy as np

from pywaveprop.rwp_jax import (
    RWPGaussSourceModel,
    RWPComputationalParams,
    TroposphereModel,
    EmptyNProfileModel,
    EvaporationDuctModel,
    PiecewiseLinearNProfileModel,
    SumNProfileModel,
    MultNProfileModel,
    ProxyWaveSpeedModel,
    create_rwp_model,
    rwp_forward_task,
    minmax_k,
)
from pywaveprop.helmholtz_jax import PiecewiseLinearTerrainModel
from pywaveprop.rwp_field import RWPField, RWPRandomField


class TestRWPGaussSourceModel(unittest.TestCase):

    def test_creation(self):
        src = RWPGaussSourceModel(
            freq_hz=300e6,
            height_m=30.0,
            beam_width_deg=15.0,
            elevation_angle_deg=0.0,
        )
        self.assertEqual(src.freq_hz, 300e6)
        self.assertEqual(src.height_m, 30.0)
        self.assertEqual(src.beam_width_deg, 15.0)

    def test_max_angle(self):
        src = RWPGaussSourceModel(
            freq_hz=300e6,
            height_m=30.0,
            beam_width_deg=15.0,
            elevation_angle_deg=5.0,
        )
        self.assertAlmostEqual(src.max_angle_deg(), 20.0)

    def test_aperture_shape(self):
        src = RWPGaussSourceModel(
            freq_hz=300e6,
            height_m=30.0,
            beam_width_deg=15.0,
        )
        z = jnp.linspace(0, 200, 100)
        aperture = src.aperture(z)
        self.assertEqual(aperture.shape, (100,))
        self.assertTrue(jnp.all(jnp.isfinite(aperture)))

    def test_aperture_peak_near_height(self):
        src = RWPGaussSourceModel(
            freq_hz=300e6,
            height_m=100.0,
            beam_width_deg=15.0,
        )
        z = jnp.linspace(0, 200, 1000)
        aperture = src.aperture(z)
        peak_idx = jnp.argmax(jnp.abs(aperture))
        peak_z = float(z[peak_idx])
        self.assertAlmostEqual(peak_z, 100.0, delta=1.0)

    def test_pytree_roundtrip(self):
        src = RWPGaussSourceModel(
            freq_hz=300e6,
            height_m=30.0,
            beam_width_deg=15.0,
            elevation_angle_deg=2.0,
            multiplier=0.5,
        )
        leaves, treedef = jax.tree_util.tree_flatten(src)
        src2 = treedef.unflatten(leaves)
        z = jnp.linspace(0, 200, 50)
        np.testing.assert_allclose(src.aperture(z), src2.aperture(z))


class TestNProfileModels(unittest.TestCase):

    def test_empty_n_profile(self):
        model = EmptyNProfileModel()
        z = jnp.linspace(0, 200, 50)
        result = model(z)
        np.testing.assert_allclose(result, 0.0, atol=1e-15)

    def test_empty_max_height(self):
        model = EmptyNProfileModel()
        self.assertEqual(model.max_height_m(), 0.0)

    def test_evaporation_duct(self):
        model = EvaporationDuctModel(height_m=20.0)
        z = jnp.linspace(1, 100, 100)
        result = model(z)
        self.assertTrue(jnp.all(jnp.isfinite(result)))
        # Above truncation height the profile should be zero
        result_above = model(jnp.array([150.0]))
        self.assertAlmostEqual(float(result_above[0]), 0.0, delta=0.1)

    def test_evaporation_duct_max_height(self):
        model = EvaporationDuctModel(height_m=20.0, truncate_height_m=100)
        self.assertEqual(model.max_height_m(), 100)

    def test_evaporation_duct_pytree(self):
        model = EvaporationDuctModel(height_m=20.0)
        leaves, treedef = jax.tree_util.tree_flatten(model)
        model2 = treedef.unflatten(leaves)
        z = jnp.linspace(0, 100, 50)
        np.testing.assert_allclose(model(z), model2(z), atol=1e-10)

    def test_piecewise_linear_n_profile(self):
        z_grid = jnp.array([0.0, 100.0, 200.0])
        N_vals = jnp.array([0.0, -10.0, 0.0])
        model = PiecewiseLinearNProfileModel(z_grid_m=z_grid, N_vals=N_vals)
        z = jnp.array([50.0])
        result = model(z)
        self.assertAlmostEqual(float(result[0]), -5.0, places=3)

    def test_piecewise_linear_from_M_profile(self):
        z_grid = jnp.array([0.0, 100.0, 200.0])
        M_vals = jnp.array([320.0, 350.0, 380.0])
        model = PiecewiseLinearNProfileModel.create_from_M_profile(z_grid, M_vals)
        self.assertIsInstance(model, PiecewiseLinearNProfileModel)

    def test_sum_n_profile(self):
        m1 = EmptyNProfileModel()
        m2 = EvaporationDuctModel(height_m=10.0)
        model = m1 + m2
        self.assertIsInstance(model, SumNProfileModel)
        z = jnp.linspace(0, 50, 10)
        np.testing.assert_allclose(model(z), m2(z), atol=1e-10)

    def test_mult_n_profile(self):
        m1 = EvaporationDuctModel(height_m=10.0)
        model = m1 * 2.0
        self.assertIsInstance(model, MultNProfileModel)
        z = jnp.array([25.0])
        np.testing.assert_allclose(model(z), 2.0 * m1(z), atol=1e-10)


class TestTroposphereModel(unittest.TestCase):

    def test_default_creation(self):
        env = TroposphereModel()
        self.assertIsInstance(env.N_profile, EmptyNProfileModel)
        self.assertAlmostEqual(env.M0, 320.0)

    def test_M_profile_standard_atmosphere(self):
        """Standard atmosphere: M increases linearly with height."""
        env = TroposphereModel()
        z = jnp.array([0.0, 100.0])
        M = env.M_profile(z)
        self.assertAlmostEqual(float(M[0]), 320.0, delta=0.01)
        # M should increase with height (standard atmosphere)
        self.assertGreater(float(M[1]), float(M[0]))

    def test_wave_speed_profile(self):
        env = TroposphereModel()
        z = jnp.array([0.0])
        c = env.wave_speed_profile(z)
        # Should be close to speed of light
        self.assertAlmostEqual(float(c[0]), 3e8, delta=1e5)

    def test_with_evaporation_duct(self):
        env = TroposphereModel(N_profile=EvaporationDuctModel(height_m=20.0))
        z = jnp.linspace(0, 100, 50)
        M = env.M_profile(z)
        self.assertTrue(jnp.all(jnp.isfinite(M)))

    def test_pytree_roundtrip(self):
        env = TroposphereModel(N_profile=EvaporationDuctModel(height_m=20.0))
        leaves, treedef = jax.tree_util.tree_flatten(env)
        env2 = treedef.unflatten(leaves)
        z = jnp.linspace(0, 100, 20)
        np.testing.assert_allclose(env.M_profile(z), env2.M_profile(z), atol=1e-10)

    def test_flat_troposphere(self):
        env = TroposphereModel(slope=0)
        z = jnp.array([0.0, 100.0])
        M = env.M_profile(z)
        self.assertAlmostEqual(float(M[0]), 320.0, delta=0.01)
        self.assertAlmostEqual(float(M[1]), 320.0, delta=0.01)


class TestProxyWaveSpeedModel(unittest.TestCase):

    def test_proxy_matches_troposphere(self):
        env = TroposphereModel()
        proxy = ProxyWaveSpeedModel(env)
        z = jnp.linspace(0, 100, 50)
        np.testing.assert_allclose(proxy(z), env.wave_speed_profile(z), atol=1e-5)


class TestMinmaxK(unittest.TestCase):

    def test_minmax_k_standard_atmosphere(self):
        src = RWPGaussSourceModel(freq_hz=300e6, height_m=30.0, beam_width_deg=15.0)
        env = TroposphereModel(N_profile=EvaporationDuctModel(height_m=20.0))
        k_min, k_max = minmax_k(src, env)
        self.assertGreater(float(k_min), 0)
        self.assertGreater(float(k_max), float(k_min))


class TestCreateRWPModel(unittest.TestCase):

    def test_create_rwp_model_basic(self):
        src = RWPGaussSourceModel(freq_hz=300e6, height_m=30.0, beam_width_deg=15.0)
        env = TroposphereModel(N_profile=EvaporationDuctModel(height_m=20.0))
        params = RWPComputationalParams(
            max_range_m=1000.0,
            x_output_points=50,
            z_output_points=50,
        )
        params.rational_approx_order = (1, 2)
        model = create_rwp_model(src, env, params)
        self.assertGreater(model.x_n, 0)
        self.assertGreater(model.z_n, 0)


class TestRWPForwardTask(unittest.TestCase):

    def test_standard_atmosphere_propagation(self):
        """Test propagation in standard atmosphere (no duct)."""
        src = RWPGaussSourceModel(
            freq_hz=300e6,
            height_m=30.0,
            beam_width_deg=15.0,
        )
        env = TroposphereModel()
        params = RWPComputationalParams(
            max_range_m=1000.0,
            max_height_m=200.0,
            x_output_points=50,
            z_output_points=50,
        )
        params.rational_approx_order = (1, 2)
        field = rwp_forward_task(src, env, params)
        self.assertIsInstance(field, RWPField)
        self.assertGreater(len(field.x_grid), 0)
        self.assertGreater(len(field.z_grid), 0)
        self.assertTrue(np.all(np.isfinite(field.field)))

    def test_evaporation_duct_propagation(self):
        """Test propagation in an evaporation duct environment."""
        src = RWPGaussSourceModel(
            freq_hz=300e6,
            height_m=10.0,
            beam_width_deg=5.0,
        )
        env = TroposphereModel(N_profile=EvaporationDuctModel(height_m=20.0))
        params = RWPComputationalParams(
            max_range_m=5000.0,
            max_height_m=200.0,
            x_output_points=50,
            z_output_points=50,
        )
        params.rational_approx_order = (1, 2)
        field = rwp_forward_task(src, env, params)
        self.assertIsInstance(field, RWPField)
        self.assertTrue(np.all(np.isfinite(field.field)))
        # Field should not be identically zero at the end
        last_col = field.field[-1, :]
        self.assertGreater(float(np.max(np.abs(last_col))), 0)

    def test_piecewise_linear_N_profile(self):
        """Test propagation with a piecewise linear N-profile (surface duct)."""
        z_grid = jnp.array([0.0, 100.0, 200.0, 300.0])
        N_vals = jnp.array([0.0, -20.0, -20.0, 0.0])
        n_prof = PiecewiseLinearNProfileModel(z_grid_m=z_grid, N_vals=N_vals)

        src = RWPGaussSourceModel(
            freq_hz=300e6,
            height_m=50.0,
            beam_width_deg=10.0,
        )
        env = TroposphereModel(N_profile=n_prof)
        params = RWPComputationalParams(
            max_range_m=2000.0,
            max_height_m=400.0,
            x_output_points=50,
            z_output_points=50,
        )
        params.rational_approx_order = (1, 2)
        field = rwp_forward_task(src, env, params)
        self.assertTrue(np.all(np.isfinite(field.field)))

    def test_propagation_with_terrain(self):
        """Test propagation over terrain."""
        x_terrain = jnp.array([0.0, 500.0, 1000.0])
        h_terrain = jnp.array([0.0, 30.0, 0.0])
        terrain = PiecewiseLinearTerrainModel(x_grid_m=x_terrain, height=h_terrain)

        src = RWPGaussSourceModel(
            freq_hz=300e6,
            height_m=50.0,
            beam_width_deg=15.0,
        )
        env = TroposphereModel(terrain=terrain)
        params = RWPComputationalParams(
            max_range_m=1000.0,
            max_height_m=200.0,
            x_output_points=50,
            z_output_points=50,
        )
        params.rational_approx_order = (1, 2)
        field = rwp_forward_task(src, env, params)
        self.assertIsInstance(field, RWPField)
        self.assertTrue(np.all(np.isfinite(field.field)))

    def test_with_elevation_angle(self):
        """Test source with nonzero elevation angle."""
        src = RWPGaussSourceModel(
            freq_hz=300e6,
            height_m=30.0,
            beam_width_deg=15.0,
            elevation_angle_deg=5.0,
        )
        env = TroposphereModel()
        params = RWPComputationalParams(
            max_range_m=1000.0,
            max_height_m=300.0,
            x_output_points=50,
            z_output_points=50,
        )
        params.rational_approx_order = (1, 2)
        field = rwp_forward_task(src, env, params)
        self.assertTrue(np.all(np.isfinite(field.field)))


class TestRWPvsAnalytical(unittest.TestCase):
    """Compare JAX propagator results with the analytical TwoRayModel."""

    def test_flat_pec_h_pol_vs_two_ray(self):
        """Flat PEC ground, H-polarization, no duct.

        The TwoRayModel provides an exact analytical solution for this case.
        The JAX propagator (parabolic equation) should match at small angles
        (far from the source, away from the boundary).
        """
        freq_hz = 300e6
        height_m = 30.0
        beam_width_deg = 3.0
        max_range_m = 500.0
        max_height_m = 300.0

        # --- JAX propagator ---
        src_jax = RWPGaussSourceModel(
            freq_hz=freq_hz,
            height_m=height_m,
            beam_width_deg=beam_width_deg,
            elevation_angle_deg=0.0,
        )
        env_jax = TroposphereModel(slope=0)
        params = RWPComputationalParams(
            max_range_m=max_range_m,
            max_height_m=max_height_m,
            x_output_points=30,
            z_output_points=30,
        )
        params.rational_approx_order = (1, 2)
        jax_field = rwp_forward_task(src_jax, env_jax, params)

        # --- TwoRayModel (analytical reference) ---
        from pywaveprop.rwp.tworay import TwoRayModel
        from pywaveprop.rwp.antennas import GaussAntenna
        from pywaveprop.rwp.environment import Troposphere, Terrain, PerfectlyElectricConducting

        antenna = GaussAntenna(
            freq_hz=freq_hz,
            height=height_m,
            beam_width=beam_width_deg,
            elevation_angle=0.0,
            polarz='H',
        )
        env_old = Troposphere(flat=True)
        env_old.terrain = Terrain(ground_material=PerfectlyElectricConducting())
        env_old.z_max = max_height_m

        trm = TwoRayModel(src=antenna, env=env_old)
        trm_field = trm.calculate(np.asarray(jax_field.x_grid), np.asarray(jax_field.z_grid))

        # Compare at receiver height = source height, far from source
        # (skip near-field and boundary regions)
        z_idx = np.argmin(np.abs(jax_field.z_grid - height_m))
        x_start = len(jax_field.x_grid) // 3  # skip near-field

        jax_h = np.abs(np.asarray(jax_field.field[x_start:, z_idx]))
        trm_h = np.abs(trm_field[x_start:, z_idx])

        # Normalize both to their own max for shape comparison
        jax_norm = jax_h / (np.max(jax_h) + 1e-30)
        trm_norm = trm_h / (np.max(trm_h) + 1e-30)

        # Compare normalized patterns in dB (shape should match)
        jax_db = 20 * np.log10(jax_norm + 1e-10)
        trm_db = 20 * np.log10(trm_norm + 1e-10)

        db_diff = np.abs(jax_db - trm_db)
        median_diff = float(np.median(db_diff))
        self.assertLess(median_diff, 5.0,
                        f"Median normalized dB difference {median_diff:.2f} exceeds 5 dB threshold")

    def test_flat_pec_h_pol_path_loss_pattern(self):
        """Verify field at source height decays roughly as 1/r in free space.

        With PEC ground and H-pol, at source height the direct and reflected
        rays interfere. The field should not blow up and should be finite
        and non-zero along the propagation path.
        """
        freq_hz = 300e6
        height_m = 30.0
        beam_width_deg = 3.0
        max_range_m = 500.0
        max_height_m = 300.0

        src_jax = RWPGaussSourceModel(
            freq_hz=freq_hz,
            height_m=height_m,
            beam_width_deg=beam_width_deg,
        )
        env_jax = TroposphereModel(slope=0)
        params = RWPComputationalParams(
            max_range_m=max_range_m,
            max_height_m=max_height_m,
            x_output_points=30,
            z_output_points=30,
        )
        params.rational_approx_order = (1, 2)
        jax_field = rwp_forward_task(src_jax, env_jax, params)

        # Field at source height should be finite and non-zero
        z_idx = np.argmin(np.abs(jax_field.z_grid - height_m))
        horizontal = np.abs(np.asarray(jax_field.field[1:, z_idx]))
        self.assertTrue(np.all(np.isfinite(horizontal)))
        self.assertGreater(float(np.max(horizontal)), 0)

        # Field should generally decrease with range (not blow up)
        first_third = np.mean(horizontal[:len(horizontal)//3])
        last_third = np.mean(horizontal[-len(horizontal)//3:])
        self.assertGreater(first_third, last_third * 0.1,
                           "Field should not blow up at far range")


class TestRWPField(unittest.TestCase):

    def _make_field(self):
        x_grid = np.linspace(10, 1000, 100)
        z_grid = np.linspace(0, 200, 50)
        field = np.random.rand(100, 50) + 1j * np.random.rand(100, 50)
        return RWPField(x_grid=x_grid, z_grid=z_grid, freq_hz=300e6, field=field)

    def test_horizontal(self):
        f = self._make_field()
        h = f.horizontal(100.0)
        self.assertEqual(h.shape, (100,))

    def test_value(self):
        f = self._make_field()
        val = f.value(500.0, 100.0)
        self.assertIsInstance(val, (complex, np.complexfloating))

    def test_path_loss(self):
        f = self._make_field()
        pl = f.path_loss()
        self.assertIsInstance(pl, RWPField)
        self.assertEqual(pl.field.shape, f.field.shape)
        self.assertTrue(np.all(np.isfinite(pl.field)))

    def test_v_func(self):
        f = self._make_field()
        vf = f.v_func()
        self.assertIsInstance(vf, RWPField)
        self.assertEqual(vf.field.shape, f.field.shape)


class TestRWPRandomField(unittest.TestCase):

    def test_empty(self):
        rf = RWPRandomField()
        self.assertIsNone(rf.mean())
        self.assertIsNone(rf.sd())

    def test_single_sample(self):
        rf = RWPRandomField()
        x_grid = np.linspace(10, 100, 10)
        z_grid = np.linspace(0, 50, 5)
        field = np.ones((10, 5), dtype=complex)
        rf.add_sample(RWPField(x_grid=x_grid, z_grid=z_grid, freq_hz=300e6, field=field))
        mean = rf.mean()
        self.assertIsNotNone(mean)
        np.testing.assert_allclose(mean.field, 1.0)

    def test_multiple_samples_mean(self):
        rf = RWPRandomField()
        x_grid = np.linspace(10, 100, 10)
        z_grid = np.linspace(0, 50, 5)
        for val in [1.0, 3.0]:
            field = np.full((10, 5), val, dtype=complex)
            rf.add_sample(RWPField(x_grid=x_grid, z_grid=z_grid, freq_hz=300e6, field=field))
        mean = rf.mean()
        np.testing.assert_allclose(mean.field, 2.0)


if __name__ == '__main__':
    unittest.main()
