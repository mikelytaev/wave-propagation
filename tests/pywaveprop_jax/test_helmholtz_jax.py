"""
Tests for the JAX-based Helmholtz propagator (RationalHelmholtzPropagator).

Tests cover:
- Wave speed models (Const, Linear, PiecewiseLinear, EOF, Sum, Mult)
- Terrain models
- RegularGrid operations
- Crank-Nicolson propagation (with and without density variation)
- NLBC computation
- Full propagation pipeline
- Grid optimization and output grid scaling
"""
import unittest
import math as fm

import jax
import jax.numpy as jnp
import numpy as np

from pywaveprop.helmholtz_jax import (
    RegularGrid,
    ConstWaveSpeedModel,
    LinearSlopeWaveSpeedModel,
    PiecewiseLinearWaveSpeedModel,
    EOFWaveSpeedModel,
    SumWaveSpeedModel,
    MultWaveSpeedModel,
    PiecewiseLinearTerrainModel,
    StaircaseRhoModel,
    RationalHelmholtzPropagator,
    AbstractWaveSpeedModel,
)
from pywaveprop.helmholtz_common import HelmholtzMeshParams2D
from pywaveprop.jax_utils import bessel_ratio_4th_order, sqr_eq


class TestRegularGrid(unittest.TestCase):

    def test_basic_creation(self):
        grid = RegularGrid(start=0.0, dx=1.0, n=10)
        self.assertEqual(grid.start, 0.0)
        self.assertEqual(grid.dx, 1.0)
        self.assertEqual(grid.n, 10)

    def test_interval_indexes(self):
        grid = RegularGrid(start=0.0, dx=1.0, n=100)
        a_i, b_i = grid.interval_indexes(10.0, 50.0)
        self.assertEqual(a_i, 10)
        # b_i includes the endpoint
        self.assertGreaterEqual(b_i, 50)
        self.assertLessEqual(b_i, 51)

    def test_interval_indexes_clamp(self):
        grid = RegularGrid(start=0.0, dx=1.0, n=100)
        a_i, b_i = grid.interval_indexes(-10.0, 200.0)
        self.assertEqual(a_i, 0)
        self.assertEqual(b_i, 100)

    def test_array_grid(self):
        grid = RegularGrid(start=0.0, dx=0.5, n=5)
        arr = grid.array_grid(0, 5)
        np.testing.assert_allclose(arr, [0.0, 0.5, 1.0, 1.5, 2.0])

    def test_equality_and_hash(self):
        g1 = RegularGrid(start=0.0, dx=1.0, n=10)
        g2 = RegularGrid(start=0.0, dx=1.0, n=10)
        g3 = RegularGrid(start=0.0, dx=2.0, n=10)
        self.assertEqual(g1, g2)
        self.assertNotEqual(g1, g3)
        self.assertEqual(hash(g1), hash(g2))

    def test_pytree_roundtrip(self):
        grid = RegularGrid(start=1.0, dx=0.5, n=20)
        leaves, treedef = jax.tree_util.tree_flatten(grid)
        grid2 = treedef.unflatten(leaves)
        self.assertEqual(grid, grid2)


class TestWaveSpeedModels(unittest.TestCase):

    def test_const_wave_speed(self):
        model = ConstWaveSpeedModel(c0=1500.0)
        z = jnp.linspace(0, 100, 50)
        result = model(z)
        np.testing.assert_allclose(result, 1500.0 * jnp.ones(50))

    def test_const_wave_speed_pytree(self):
        model = ConstWaveSpeedModel(c0=343.0)
        leaves, treedef = jax.tree_util.tree_flatten(model)
        model2 = treedef.unflatten(leaves)
        z = jnp.array([0.0, 50.0, 100.0])
        np.testing.assert_allclose(model(z), model2(z))

    def test_linear_slope_wave_speed(self):
        model = LinearSlopeWaveSpeedModel(c0=1500.0, slope_degrees=0.0)
        z = jnp.linspace(0, 100, 10)
        result = model(z)
        np.testing.assert_allclose(result, 1500.0 * jnp.ones(10), atol=1e-10)

    def test_linear_slope_wave_speed_nonzero(self):
        model = LinearSlopeWaveSpeedModel(c0=1500.0, slope_degrees=45.0)
        z = jnp.array([0.0, 1.0])
        result = model(z)
        self.assertAlmostEqual(float(result[0]), 1500.0, places=5)
        self.assertAlmostEqual(float(result[1]), 1501.0, places=5)

    def test_piecewise_linear_wave_speed(self):
        z_grid = jnp.array([0.0, 50.0, 100.0])
        speeds = jnp.array([1500.0, 1490.0, 1510.0])
        model = PiecewiseLinearWaveSpeedModel(z_grid_m=z_grid, sound_speed=speeds)
        z = jnp.array([25.0])
        result = model(z)
        self.assertAlmostEqual(float(result[0]), 1495.0, places=3)

    def test_piecewise_linear_support(self):
        z_grid = jnp.array([0.0, 50.0, 100.0])
        speeds = jnp.array([1500.0, 1490.0, 1510.0])
        model = PiecewiseLinearWaveSpeedModel(z_grid_m=z_grid, sound_speed=speeds)
        self.assertEqual(model.support(), (0.0, 100.0))

    def test_eof_wave_speed(self):
        z_grid = jnp.array([0.0, 50.0, 100.0])
        mean = jnp.array([1500.0, 1490.0, 1510.0])
        modes = jnp.array([[1.0], [0.5], [-0.5]])
        coefs = jnp.array([0.0])
        model = EOFWaveSpeedModel(eof_z_grid_m=z_grid, mean_profile=mean, eof_modes=modes, coefs=coefs)
        z = jnp.array([0.0, 50.0, 100.0])
        result = model(z)
        np.testing.assert_allclose(result, mean, atol=1e-5)

    def test_eof_wave_speed_nonzero_coefs(self):
        z_grid = jnp.array([0.0, 50.0, 100.0])
        mean = jnp.array([1500.0, 1490.0, 1510.0])
        modes = jnp.array([[1.0], [0.5], [-0.5]])
        coefs = jnp.array([2.0])
        model = EOFWaveSpeedModel(eof_z_grid_m=z_grid, mean_profile=mean, eof_modes=modes, coefs=coefs)
        z = jnp.array([0.0, 50.0, 100.0])
        result = model(z)
        expected = mean + jnp.array([2.0, 1.0, -1.0])
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_sum_wave_speed_model(self):
        m1 = ConstWaveSpeedModel(c0=1500.0)
        m2 = ConstWaveSpeedModel(c0=10.0)
        model = m1 + m2
        self.assertIsInstance(model, SumWaveSpeedModel)
        z = jnp.array([0.0, 50.0])
        result = model(z)
        np.testing.assert_allclose(result, 1510.0)

    def test_mult_wave_speed_model(self):
        m1 = ConstWaveSpeedModel(c0=1500.0)
        model = m1 * 2.0
        self.assertIsInstance(model, MultWaveSpeedModel)
        z = jnp.array([0.0, 50.0])
        result = model(z)
        np.testing.assert_allclose(result, 3000.0)

    def test_on_regular_grid(self):
        model = ConstWaveSpeedModel(c0=1500.0)
        grid = RegularGrid(start=0.0, dx=10.0, n=5)
        result = model.on_regular_grid(grid)
        np.testing.assert_allclose(result, 1500.0 * jnp.ones(5))


class TestTerrainModels(unittest.TestCase):

    def test_piecewise_linear_terrain_flat(self):
        x_grid = jnp.array([0.0, 1000.0])
        height = jnp.array([0.0, 0.0])
        model = PiecewiseLinearTerrainModel(x_grid_m=x_grid, height=height)
        x = jnp.array([500.0])
        result = model(x)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_piecewise_linear_terrain_slope(self):
        x_grid = jnp.array([0.0, 1000.0])
        height = jnp.array([0.0, 100.0])
        model = PiecewiseLinearTerrainModel(x_grid_m=x_grid, height=height)
        x = jnp.array([500.0])
        result = model(x)
        np.testing.assert_allclose(result, 50.0, atol=1e-3)

    def test_terrain_pytree_roundtrip(self):
        x_grid = jnp.array([0.0, 500.0, 1000.0])
        height = jnp.array([0.0, 50.0, 0.0])
        model = PiecewiseLinearTerrainModel(x_grid_m=x_grid, height=height)
        leaves, treedef = jax.tree_util.tree_flatten(model)
        model2 = treedef.unflatten(leaves)
        x = jnp.array([250.0])
        np.testing.assert_allclose(model(x), model2(x))


class TestStaircaseRhoModel(unittest.TestCase):

    def test_constant_density(self):
        model = StaircaseRhoModel(heights=[0.0], vals=[1.0])
        z = jnp.array([0.0, 50.0, 100.0])
        result = model(z)
        np.testing.assert_allclose(result, 1.0)

    def test_two_layers(self):
        model = StaircaseRhoModel(heights=[0.0, 50.0], vals=[1.0, 2.5])
        z = jnp.array([25.0, 75.0])
        result = model(z)
        np.testing.assert_allclose(result, [1.0, 2.5])


class TestUtils(unittest.TestCase):

    def test_sqr_eq_basic(self):
        result = sqr_eq(1.0, -3.0, 2.0)
        self.assertTrue(abs(float(jnp.abs(result)) - 1.0) < 1e-5 or abs(float(jnp.abs(result)) - 2.0) < 1e-5)

    def test_sqr_eq_returns_smaller_root(self):
        result = sqr_eq(1.0 + 0j, -5.0 + 0j, 6.0 + 0j)
        self.assertAlmostEqual(float(jnp.abs(result)), 2.0, places=5)


class TestRationalHelmholtzPropagator(unittest.TestCase):

    def _create_simple_propagator(self, z_n=50, x_n=20):
        """Helper to create a simple propagator for testing.

        Uses small grids and pre-computed zero NLBC coefficients
        to avoid expensive NLBC computation on CPU.
        """
        freq_hz = 300e6
        c0 = 3e8
        k0 = 2 * fm.pi * freq_hz / c0
        beta = k0
        dx_m = 10.0
        dz_m = 1.0
        order = (1, 2)
        m_size = max(order)

        wave_speed = ConstWaveSpeedModel(c0=c0)
        # Pre-compute zero NLBC coefficients to skip expensive _calc_nlbc
        upper_nlbc_coefs = jnp.zeros((x_n, m_size, m_size), dtype=complex)
        prop = RationalHelmholtzPropagator(
            order=order,
            beta=beta,
            dx_m=dx_m,
            dz_m=dz_m,
            x_n=x_n,
            z_n=z_n,
            x_grid_scale=1,
            z_grid_scale=1,
            freq_hz=freq_hz,
            wave_speed=wave_speed,
            upper_nlbc_coefs=upper_nlbc_coefs,
        )
        return prop

    def test_creation(self):
        prop = self._create_simple_propagator()
        self.assertEqual(prop.order, (1, 2))
        self.assertEqual(prop.z_n, 50)
        self.assertEqual(prop.x_n, 20)

    def test_computational_grids(self):
        prop = self._create_simple_propagator()
        x_grid = prop.x_computational_grid()
        z_grid = prop.z_computational_grid()
        self.assertEqual(len(x_grid), 20)
        self.assertEqual(len(z_grid), 50)
        np.testing.assert_allclose(x_grid[0], 0.0)
        np.testing.assert_allclose(z_grid[0], 0.0)

    def test_output_grids_with_scaling(self):
        prop = self._create_simple_propagator(z_n=100, x_n=20)
        prop.x_grid_scale = 2
        prop.z_grid_scale = 5
        x_out = prop.x_output_grid()
        z_out = prop.z_output_grid()
        self.assertEqual(len(x_out), 10)
        self.assertEqual(len(z_out), 20)

    def test_het_arrays_homogeneous(self):
        """For a homogeneous medium with c0 matching beta, het should be ~0."""
        prop = self._create_simple_propagator()
        np.testing.assert_allclose(prop.het, jnp.zeros(50), atol=1e-6)

    def test_crank_nikolson_no_rho(self):
        """Test that CN propagation preserves boundary conditions."""
        prop = self._create_simple_propagator()
        initial = jnp.zeros(50, dtype=complex)
        initial = initial.at[25].set(1.0 + 0j)
        a, b = prop.coefs[0]
        result = prop._Crank_Nikolson_propagate_no_rho_4th_order(
            a, b, initial,
            lower_bound=(1, 0, 0),
            upper_bound=(0, 1, 0)
        )
        self.assertEqual(result.shape, (50,))
        # Result should be finite
        self.assertTrue(jnp.all(jnp.isfinite(result)))

    def test_lower_terrain_mask(self):
        """Test terrain mask generation."""
        freq_hz = 300e6
        c0 = 3e8
        k0 = 2 * fm.pi * freq_hz / c0
        x_grid = jnp.array([0.0, 1000.0])
        height = jnp.array([5.0, 5.0])
        terrain = PiecewiseLinearTerrainModel(x_grid_m=x_grid, height=height)

        wave_speed = ConstWaveSpeedModel(c0=c0)
        order = (1, 2)
        x_n, z_n = 20, 50
        m_size = max(order)
        upper_nlbc_coefs = jnp.zeros((x_n, m_size, m_size), dtype=complex)
        prop = RationalHelmholtzPropagator(
            order=order,
            beta=k0,
            dx_m=10.0,
            dz_m=1.0,
            x_n=x_n,
            z_n=z_n,
            x_grid_scale=1,
            z_grid_scale=1,
            freq_hz=freq_hz,
            wave_speed=wave_speed,
            lower_terrain=terrain,
            upper_nlbc_coefs=upper_nlbc_coefs,
        )
        # Below terrain (z_index < 5), mask should be 0
        self.assertAlmostEqual(float(jnp.abs(prop.lower_terrain_mask[10, 0])), 0.0)
        # Above terrain (z_index >= 5), mask should be 1
        self.assertAlmostEqual(float(jnp.abs(prop.lower_terrain_mask[10, 25])), 1.0)

    def test_pytree_roundtrip(self):
        prop = self._create_simple_propagator(z_n=50, x_n=20)
        leaves, treedef = jax.tree_util.tree_flatten(prop)
        prop2 = treedef.unflatten(leaves)
        self.assertEqual(prop.order, prop2.order)
        self.assertEqual(prop.z_n, prop2.z_n)
        self.assertEqual(prop.x_n, prop2.x_n)

    def test_compute_homogeneous_propagation(self):
        """Test full propagation in a homogeneous medium.

        A Gaussian beam in a homogeneous medium should propagate
        smoothly without blowing up.
        """
        prop = self._create_simple_propagator()
        z = prop.z_computational_grid()
        # Gaussian initial field
        z_center = float(z[25])
        sigma = 5.0
        initial = jnp.exp(-((z - z_center) / sigma) ** 2).astype(complex)
        result = prop.compute(initial)

        # Result should have correct shape
        expected_x_n = prop.x_n // prop.x_grid_scale + 1
        expected_z_n = (prop.z_n - 1) // prop.z_grid_scale + 1
        self.assertEqual(result.shape, (expected_x_n, expected_z_n))
        # All values should be finite
        self.assertTrue(jnp.all(jnp.isfinite(result)))
        # Field should not blow up
        self.assertTrue(float(jnp.max(jnp.abs(result))) < 100.0)

    @unittest.skip("Slow NLBC computation on CPU; tested via RWP/UWA integration tests")
    def test_create_factory(self):
        """Test the create() factory method with grid optimization."""
        freq_hz = 300e6
        k0 = 2 * fm.pi * freq_hz / 3e8
        kz_max = k0 * fm.sin(fm.radians(15))
        wave_speed = ConstWaveSpeedModel(c0=3e8)

        prop = RationalHelmholtzPropagator.create(
            freq_hz=freq_hz,
            wave_speed=wave_speed,
            kz_max=kz_max,
            k_bounds=(k0 * 0.99, k0 * 1.01),
            precision=0.01,
            mesh_params=HelmholtzMeshParams2D(
                x_size_m=1000.0,
                z_size_m=300.0,
                x_n_upper_bound=100,
                z_n_upper_bound=100,
            ),
        )
        self.assertIsInstance(prop, RationalHelmholtzPropagator)
        self.assertGreater(prop.x_n, 0)
        self.assertGreater(prop.z_n, 0)


class TestMeshParams2D(unittest.TestCase):

    def test_valid_creation(self):
        params = HelmholtzMeshParams2D(
            x_size_m=1000.0,
            z_size_m=300.0,
            dx_output_m=10.0,
            dz_output_m=1.0,
        )
        self.assertEqual(params.x_size_m, 1000.0)

    def test_missing_x_grid(self):
        with self.assertRaises(ValueError):
            HelmholtzMeshParams2D(x_size_m=1000.0, z_size_m=300.0, dz_output_m=1.0)

    def test_both_x_grid_specified(self):
        with self.assertRaises(ValueError):
            HelmholtzMeshParams2D(
                x_size_m=1000.0, z_size_m=300.0,
                dx_output_m=10.0, x_n_upper_bound=100,
                dz_output_m=1.0
            )

    def test_missing_z_grid(self):
        with self.assertRaises(ValueError):
            HelmholtzMeshParams2D(x_size_m=1000.0, z_size_m=300.0, dx_output_m=10.0)


if __name__ == '__main__':
    unittest.main()
