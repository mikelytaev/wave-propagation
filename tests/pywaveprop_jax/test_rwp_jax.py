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

Set the environment variable TEST_PLOTS to a directory path to generate
diagnostic plots for each test, e.g.::

    TEST_PLOTS=./test_plots python -m pytest tests/pywaveprop_jax/test_rwp_jax.py
"""
import os
import unittest
import math as fm

import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Optional plotting infrastructure
# ---------------------------------------------------------------------------
PLOT_DIR = os.environ.get("TEST_PLOTS")
if PLOT_DIR:
    os.makedirs(PLOT_DIR, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt


def _save_fig(name):
    """Save current figure to *PLOT_DIR/name.png* and close it."""
    if not PLOT_DIR:
        return
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{name}.png"), dpi=150)
    plt.close()


def _plot_field_2d(field, title="", filename="field", vmin=-60, vmax=0):
    """Plot a 2-D propagation field (dB re max) with range/height axes."""
    if not PLOT_DIR:
        return
    arr = np.asarray(field.field)
    x = np.asarray(field.x_grid)
    z = np.asarray(field.z_grid)
    n_x = min(arr.shape[0], x.shape[0])
    n_z = min(arr.shape[1], z.shape[0])
    arr = arr[:n_x, :n_z]
    amp = 20 * np.log10(np.abs(arr).T + 1e-30)
    amp_max = float(np.nanmax(amp))
    plt.figure(figsize=(8, 4))
    plt.pcolormesh(x[:n_x], z[:n_z], amp, shading="auto",
                   cmap="jet", vmin=amp_max + vmin, vmax=amp_max + vmax)
    plt.colorbar(label="dB")
    plt.xlabel("Range (m)")
    plt.ylabel("Height (m)")
    plt.title(title)
    _save_fig(filename)

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
    GroundMaterial,
    RoughSeaSurface,
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

        if PLOT_DIR:
            plt.figure()
            plt.plot(np.asarray(z), np.abs(np.asarray(aperture)))
            plt.xlabel("Height (m)")
            plt.ylabel("|Aperture|")
            plt.title("Gaussian aperture (h=30 m, bw=15°)")
            _save_fig("aperture_shape")

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

        if PLOT_DIR:
            plt.figure()
            plt.plot(np.asarray(z), np.abs(np.asarray(aperture)))
            plt.axvline(100.0, color="r", ls="--", label="expected peak")
            plt.axvline(peak_z, color="g", ls=":", label=f"actual peak ({peak_z:.1f} m)")
            plt.xlabel("Height (m)")
            plt.ylabel("|Aperture|")
            plt.title("Aperture peak location (h=100 m)")
            plt.legend()
            _save_fig("aperture_peak")

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

        if PLOT_DIR:
            plt.figure()
            plt.plot(np.asarray(result), np.asarray(z))
            plt.xlabel("N (M-units)")
            plt.ylabel("Height (m)")
            plt.title("Evaporation duct N-profile (h_d=20 m)")
            _save_fig("n_profile_evaporation_duct")

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

        if PLOT_DIR:
            z_plot = jnp.linspace(0, 200, 200)
            plt.figure()
            plt.plot(np.asarray(model(z_plot)), np.asarray(z_plot))
            plt.plot(float(result[0]), 50.0, "ro", label="tested point (−5)")
            plt.xlabel("N (M-units)")
            plt.ylabel("Height (m)")
            plt.title("Piecewise linear N-profile")
            plt.legend()
            _save_fig("n_profile_piecewise_linear")

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

        if PLOT_DIR:
            z_plot = jnp.linspace(1, 50, 100)
            plt.figure()
            plt.plot(np.asarray(m2(z_plot)), np.asarray(z_plot), label="EvapDuct")
            plt.plot(np.asarray(model(z_plot)), np.asarray(z_plot), "--", label="Empty + EvapDuct")
            plt.xlabel("N (M-units)")
            plt.ylabel("Height (m)")
            plt.title("Sum N-profile (Empty + EvapDuct)")
            plt.legend()
            _save_fig("n_profile_sum")

    def test_mult_n_profile(self):
        m1 = EvaporationDuctModel(height_m=10.0)
        model = m1 * 2.0
        self.assertIsInstance(model, MultNProfileModel)
        z = jnp.array([25.0])
        np.testing.assert_allclose(model(z), 2.0 * m1(z), atol=1e-10)

        if PLOT_DIR:
            z_plot = jnp.linspace(1, 50, 100)
            plt.figure()
            plt.plot(np.asarray(m1(z_plot)), np.asarray(z_plot), label="EvapDuct × 1")
            plt.plot(np.asarray(model(z_plot)), np.asarray(z_plot), "--", label="EvapDuct × 2")
            plt.xlabel("N (M-units)")
            plt.ylabel("Height (m)")
            plt.title("Mult N-profile (EvapDuct × 2)")
            plt.legend()
            _save_fig("n_profile_mult")


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

        if PLOT_DIR:
            z_plot = jnp.linspace(0, 300, 200)
            plt.figure()
            plt.plot(np.asarray(env.M_profile(z_plot)), np.asarray(z_plot))
            plt.xlabel("M (M-units)")
            plt.ylabel("Height (m)")
            plt.title("Standard atmosphere M-profile")
            _save_fig("m_profile_standard")

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

        if PLOT_DIR:
            z_plot = jnp.linspace(0, 200, 200)
            env_std = TroposphereModel()
            plt.figure()
            plt.plot(np.asarray(env_std.M_profile(z_plot)), np.asarray(z_plot),
                     label="Standard")
            plt.plot(np.asarray(env.M_profile(z_plot)), np.asarray(z_plot),
                     label="With evap. duct (h_d=20 m)")
            plt.xlabel("M (M-units)")
            plt.ylabel("Height (m)")
            plt.title("M-profile with evaporation duct")
            plt.legend()
            _save_fig("m_profile_evap_duct")

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

        if PLOT_DIR:
            z_plot = jnp.linspace(0, 200, 200)
            env_std = TroposphereModel()
            plt.figure()
            plt.plot(np.asarray(env_std.M_profile(z_plot)), np.asarray(z_plot),
                     label="Standard (slope=0.118)")
            plt.plot(np.asarray(env.M_profile(z_plot)), np.asarray(z_plot),
                     "--", label="Flat (slope=0)")
            plt.xlabel("M (M-units)")
            plt.ylabel("Height (m)")
            plt.title("Flat vs standard troposphere")
            plt.legend()
            _save_fig("m_profile_flat")


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

        _plot_field_2d(field, "Standard atmosphere propagation",
                       "forward_standard_atm")

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

        _plot_field_2d(field, "Evaporation duct propagation (h_d=20 m)",
                       "forward_evap_duct")

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

        _plot_field_2d(field, "Piecewise linear N-profile propagation",
                       "forward_piecewise_linear")

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

        if PLOT_DIR:
            _plot_field_2d(field, "Propagation over terrain", "forward_terrain")
            # overlay terrain profile on the last saved figure — reopen
            plt.figure(figsize=(8, 4))
            arr = np.asarray(field.field)
            x = np.asarray(field.x_grid)
            z = np.asarray(field.z_grid)
            n_x, n_z = min(arr.shape[0], x.shape[0]), min(arr.shape[1], z.shape[0])
            amp = 20 * np.log10(np.abs(arr[:n_x, :n_z]).T + 1e-30)
            amp_max = float(np.nanmax(amp))
            plt.pcolormesh(x[:n_x], z[:n_z], amp, shading="auto",
                           cmap="jet", vmin=amp_max - 60, vmax=amp_max)
            plt.colorbar(label="dB")
            plt.plot(np.asarray(x_terrain), np.asarray(h_terrain),
                     "k-", lw=2, label="Terrain")
            plt.xlabel("Range (m)")
            plt.ylabel("Height (m)")
            plt.title("Propagation over terrain (with terrain line)")
            plt.legend()
            _save_fig("forward_terrain_overlay")

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

        _plot_field_2d(field, "Propagation with elevation angle (5°)",
                       "forward_elevation_angle")


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

        if PLOT_DIR:
            fig, axes = plt.subplots(1, 3, figsize=(16, 4))
            # 2D JAX field
            jax_arr = np.asarray(jax_field.field)
            x = np.asarray(jax_field.x_grid)
            z = np.asarray(jax_field.z_grid)
            nx = min(jax_arr.shape[0], x.shape[0])
            nz = min(jax_arr.shape[1], z.shape[0])
            amp = 20 * np.log10(np.abs(jax_arr[:nx, :nz]).T + 1e-30)
            amax = float(np.nanmax(amp))
            axes[0].pcolormesh(x[:nx], z[:nz], amp, shading="auto",
                               cmap="jet", vmin=amax - 60, vmax=amax)
            axes[0].axhline(height_m, color="w", ls="--", lw=0.7)
            axes[0].set_title("JAX propagator")
            axes[0].set_xlabel("Range (m)")
            axes[0].set_ylabel("Height (m)")
            # 2D TwoRay field
            amp_trm = 20 * np.log10(np.abs(trm_field[:nx, :nz]).T + 1e-30)
            axes[1].pcolormesh(x[:nx], z[:nz], amp_trm, shading="auto",
                               cmap="jet", vmin=amax - 60, vmax=amax)
            axes[1].axhline(height_m, color="w", ls="--", lw=0.7)
            axes[1].set_title("TwoRay analytical")
            axes[1].set_xlabel("Range (m)")
            # 1D comparison at source height
            x_cmp = x[x_start:nx]
            axes[2].plot(x_cmp, jax_db, label="JAX (norm dB)")
            axes[2].plot(x_cmp, trm_db, "--", label="TwoRay (norm dB)")
            axes[2].set_xlabel("Range (m)")
            axes[2].set_ylabel("Normalized dB")
            axes[2].set_title(f"z={height_m} m | median Δ={median_diff:.2f} dB")
            axes[2].legend()
            _save_fig("vs_analytical_pec_h_pol")

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

        if PLOT_DIR:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            arr = np.asarray(jax_field.field)
            x = np.asarray(jax_field.x_grid)
            z = np.asarray(jax_field.z_grid)
            nx = min(arr.shape[0], x.shape[0])
            nz = min(arr.shape[1], z.shape[0])
            amp = 20 * np.log10(np.abs(arr[:nx, :nz]).T + 1e-30)
            amax = float(np.nanmax(amp))
            axes[0].pcolormesh(x[:nx], z[:nz], amp, shading="auto",
                               cmap="jet", vmin=amax - 60, vmax=amax)
            axes[0].axhline(height_m, color="w", ls="--", lw=0.7)
            axes[0].set_title("2D field (flat PEC, H-pol)")
            axes[0].set_xlabel("Range (m)")
            axes[0].set_ylabel("Height (m)")
            # 1D horizontal at source height
            axes[1].plot(x[1:nx], 20 * np.log10(horizontal + 1e-30))
            axes[1].set_xlabel("Range (m)")
            axes[1].set_ylabel("|Field| (dB)")
            axes[1].set_title(f"Horizontal slice at z={height_m} m")
            _save_fig("path_loss_pattern")


class TestBrewsterAngleNLBC(unittest.TestCase):
    """Lower NLBC validation: V-polarized beam at the Brewster angle.

    With a finite-permittivity ground material (eps_r=3) wired into
    TroposphereModel, the JAX propagator activates the angle-dependent
    lower nonlocal BC. For V-polarization at the Brewster angle (60 deg
    from the surface normal, i.e. 30 deg grazing) the Fresnel reflection
    coefficient vanishes. At 300 MHz the auto-grid optimizer picks
    Padé (3,4), keeping each forward task under ~3 s on CPU.
    """

    @staticmethod
    def _build_brewster_scenario(*, height_m=15.0, beam_width_deg=4.0, freq_hz=3e9):
        """Build a (src, env, params) triple for a V-polarized beam tilted
        at the Brewster angle of an eps_r=3 ground."""
        from pywaveprop.propagators._utils import brewster_angle

        ground = GroundMaterial(eps=3, sigma=0)
        b_angle = brewster_angle(1, ground.complex_permittivity(freq_hz)).real
        grazing = 90 - b_angle  # 30 deg

        src = RWPGaussSourceModel(
            freq_hz=freq_hz,
            height_m=height_m,
            beam_width_deg=beam_width_deg,
            elevation_angle_deg=grazing,  # positive = downward toward ground
            polarization='V',
        )
        env = TroposphereModel(M0=320, slope=0.0, ground_material=ground)
        a = abs(height_m / fm.tan(grazing * fm.pi / 180))
        params = RWPComputationalParams(
            max_range_m=2 * a + 100.0,
            max_height_m=max(4 * height_m, 30.0),
            max_angle_deg=beam_width_deg + grazing + 2.0,
            x_output_points=300,
            z_output_points=300,
            precision=0.01,
        )
        return src, env, params

    def test_brewster_angle_value(self):
        """Brewster angle for eps_r = 3 ground is exactly 60 deg from the
        surface normal (arctan(sqrt(3)))."""
        from pywaveprop.propagators._utils import brewster_angle
        ground = GroundMaterial(eps=3, sigma=0)
        b_angle = brewster_angle(1, ground.complex_permittivity(300e6)).real
        self.assertAlmostEqual(b_angle, 60.0, delta=1e-6)

    def test_v_pol_reflection_vanishes_at_brewster(self):
        """The Fresnel reflection coefficient must vanish for V-polarization
        at the Brewster angle and stay non-zero for H-polarization."""
        from pywaveprop.propagators._utils import reflection_coef, brewster_angle
        eps = 3.0 + 0j
        b_angle = brewster_angle(1, eps).real
        r_v = reflection_coef(1, eps, b_angle, 'V')
        r_h = reflection_coef(1, eps, b_angle, 'H')
        self.assertLess(abs(r_v), 1e-12)
        self.assertGreater(abs(r_h), 0.1)

    def test_lower_nlbc_activates_with_ground_material(self):
        """Setting ground_material on TroposphereModel must enable the lower
        NLBC inside the underlying RationalHelmholtzPropagator. Without it
        the propagator falls back to the default Dirichlet (PEC) BC."""
        src, env, params = self._build_brewster_scenario()
        model = create_rwp_model(src, env, params)
        self.assertTrue(getattr(model, 'has_lower_nlbc', False),
                        "ground_material should enable lower NLBC")

        env_pec = TroposphereModel(M0=320, slope=0.0, ground_material=None)
        model_pec = create_rwp_model(src, env_pec, params)
        self.assertFalse(getattr(model_pec, 'has_lower_nlbc', False))

    def test_lower_nlbc_coefs_depend_on_polarization(self):
        """The lower NLBC convolution coefficients must differ between
        V- and H-polarized sources sharing identical geometry."""
        src_v, env, params = self._build_brewster_scenario()
        src_h = RWPGaussSourceModel(
            freq_hz=src_v.freq_hz,
            height_m=src_v.height_m,
            beam_width_deg=src_v.beam_width_deg,
            elevation_angle_deg=src_v.elevation_angle_deg,
            polarization='H',
        )
        m_v = create_rwp_model(src_v, env, params)
        m_h = create_rwp_model(src_h, env, params)
        coefs_v = np.asarray(m_v.lower_nlbc_coefs)
        coefs_h = np.asarray(m_h.lower_nlbc_coefs)
        self.assertEqual(coefs_v.shape, coefs_h.shape)
        self.assertGreater(float(np.max(np.abs(coefs_v - coefs_h))), 1e-3,
                           "V- and H-pol NLBC coefs should be distinct")

    def test_brewster_forward_task_finite(self):
        """End-to-end smoke test: forward task with lower NLBC must
        produce a finite, non-zero field."""
        src, env, params = self._build_brewster_scenario()
        field = rwp_forward_task(src, env, params)
        arr = np.asarray(field.field)
        self.assertTrue(np.all(np.isfinite(arr)))
        self.assertGreater(float(np.max(np.abs(arr))), 0.0)
        n_x = arr.shape[0]
        z_idx = int(np.argmin(np.abs(np.asarray(field.z_grid) - src.height_m)))
        mid_slice = np.abs(arr[n_x // 4 : 3 * n_x // 4, z_idx])
        self.assertGreater(float(np.max(mid_slice)), 0.0)

        _plot_field_2d(field, "Brewster angle forward (V-pol, eps_r=3)",
                       "brewster_forward")

    def test_v_pol_differs_from_h_pol(self):
        """V- and H-polarized beams through the same eps_r=3 ground must
        produce measurably different fields because the Fresnel reflection
        coefficients differ (R_V ≈ 0, R_H ≈ −0.5 at 30 deg grazing)."""
        src_v, env, params = self._build_brewster_scenario(height_m=5.0)
        src_h = RWPGaussSourceModel(
            freq_hz=src_v.freq_hz,
            height_m=src_v.height_m,
            beam_width_deg=src_v.beam_width_deg,
            elevation_angle_deg=src_v.elevation_angle_deg,
            polarization='H',
        )
        f_v = rwp_forward_task(src_v, env, params)
        f_h = rwp_forward_task(src_h, env, params)
        arr_v = np.asarray(f_v.field)
        arr_h = np.asarray(f_h.field)
        # Truncate to common shape (grid optimizer may differ slightly)
        mn_x = min(arr_v.shape[0], arr_h.shape[0])
        mn_z = min(arr_v.shape[1], arr_h.shape[1])
        arr_v = arr_v[:mn_x, :mn_z]
        arr_h = arr_h[:mn_x, :mn_z]
        denom = max(float(np.mean(np.abs(arr_v))), float(np.mean(np.abs(arr_h))))
        rel_diff = float(np.mean(np.abs(np.abs(arr_v) - np.abs(arr_h)))) / (denom + 1e-30)
        self.assertGreater(rel_diff, 0.05,
                           f"V-pol/H-pol too similar: rel_diff={rel_diff:.4f}")

        if PLOT_DIR:
            fig, axes = plt.subplots(1, 3, figsize=(16, 4))
            px = min(mn_x, len(f_v.x_grid))
            pz = min(mn_z, len(f_v.z_grid))
            x_common = np.asarray(f_v.x_grid)[:px]
            z_common = np.asarray(f_v.z_grid)[:pz]
            for ax, arr_p, lbl in [(axes[0], arr_v[:px, :pz], "V-pol"),
                                    (axes[1], arr_h[:px, :pz], "H-pol")]:
                amp = 20 * np.log10(np.abs(arr_p).T + 1e-30)
                amax = float(np.nanmax(amp))
                ax.pcolormesh(x_common, z_common, amp, shading="auto",
                              cmap="jet", vmin=amax - 60, vmax=amax)
                ax.set_title(lbl)
                ax.set_xlabel("Range (m)")
                ax.set_ylabel("Height (m)")
            # difference plot
            amp_diff = 20 * np.log10(np.abs(arr_v[:px, :pz]).T + 1e-30) - \
                       20 * np.log10(np.abs(arr_h[:px, :pz]).T + 1e-30)
            axes[2].pcolormesh(x_common, z_common, amp_diff, shading="auto",
                               cmap="RdBu_r", vmin=-20, vmax=20)
            axes[2].set_title(f"|V|−|H| dB (rel_diff={rel_diff:.3f})")
            axes[2].set_xlabel("Range (m)")
            _save_fig("brewster_v_vs_h")

    def test_v_pol_brewster_matches_two_ray(self):
        """Compare JAX propagator with lower NLBC against the analytical
        TwoRayModel on a horizontal slice near the beam path (z ≈ h).
        For V-pol at the Brewster angle both models agree: reflection is
        suppressed so the field is dominated by the direct ray."""
        from pywaveprop.rwp.tworay import TwoRayModel
        from pywaveprop.rwp.antennas import GaussAntenna
        from pywaveprop.rwp.environment import Troposphere, Terrain, CustomMaterial

        src, env, params = self._build_brewster_scenario(height_m=15.0)
        params.x_output_points = 300
        params.z_output_points = 300
        jax_field = rwp_forward_task(src, env, params)

        jax_x = np.asarray(jax_field.x_grid)
        jax_z = np.asarray(jax_field.z_grid)
        n_x = min(jax_field.field.shape[0], jax_x.shape[0])
        n_z = min(jax_field.field.shape[1], jax_z.shape[0])
        jax_arr = np.asarray(jax_field.field[:n_x, :n_z])

        # --- TwoRayModel reference ---
        legacy_env = Troposphere(flat=True)
        legacy_env.z_max = params.max_height_m
        legacy_env.terrain = Terrain(ground_material=CustomMaterial(eps=3, sigma=0))
        legacy_antenna = GaussAntenna(
            freq_hz=src.freq_hz,
            height=src.height_m,
            beam_width=src.beam_width_deg,
            elevation_angle=src.elevation_angle_deg,  # same sign convention
            polarz='V',
        )
        trm = TwoRayModel(src=legacy_antenna, env=legacy_env)
        trm_x = np.maximum(jax_x[:n_x], 1.0)
        trm_z = np.maximum(jax_z[:n_z], 1e-3)
        trm_arr = trm.calculate(trm_x, trm_z)

        # Compare at z ≈ h/2 (on the downward beam path, above ground)
        z0 = src.height_m / 2
        z_idx = int(np.argmin(np.abs(jax_z[:n_z] - z0)))
        x_start = n_x // 4
        x_cmp = jax_x[x_start:n_x]

        # Calibrate PE reduced field → physical field.
        # The 2-D PE propagates the envelope u(x,z); the 3-D physical field is
        #   p(x,z) = u(x,z) · exp(iβx) · √(2π / (k₀ x))
        # TwoRay already returns the physical field (exp(ikr)/r convention).
        k0 = 2 * np.pi * src.freq_hz / 3e8
        jax_h = np.abs(jax_arr[x_start:, z_idx]) * np.sqrt(2 * np.pi / (k0 * x_cmp))
        trm_h = np.abs(trm_arr[x_start:, z_idx])

        self.assertGreater(float(np.max(jax_h)), 1e-20)
        self.assertGreater(float(np.max(trm_h)), 1e-20)

        corr = float(np.corrcoef(jax_h, trm_h)[0, 1])
        self.assertGreater(corr, 0.9,
                           f"Correlation with two-ray reference too low: {corr:.3f}")

        # After calibration the median dB difference should be small
        mid = slice(len(x_cmp) // 10, 8 * len(x_cmp) // 10)
        jax_db = 20 * np.log10(jax_h[mid] + 1e-30)
        trm_db = 20 * np.log10(trm_h[mid] + 1e-30)
        median_diff = float(np.median(np.abs(jax_db - trm_db)))
        self.assertLess(median_diff, 3.0,
                        f"Median calibrated dB difference {median_diff:.2f} exceeds 3 dB")

        if PLOT_DIR:
            fig, axes = plt.subplots(1, 3, figsize=(16, 4))
            # 2D JAX field
            amp = 20 * np.log10(np.abs(jax_arr[:n_x, :n_z]).T + 1e-30)
            amax = float(np.nanmax(amp))
            axes[0].pcolormesh(jax_x[:n_x], jax_z[:n_z], amp, shading="auto",
                               cmap="jet", vmin=amax - 60, vmax=amax)
            axes[0].set_title("JAX (V-pol, Brewster)")
            axes[0].set_xlabel("Range (m)")
            axes[0].set_ylabel("Height (m)")
            # 2D TwoRay field
            amp_trm = 20 * np.log10(np.abs(trm_arr[:n_x, :n_z]).T + 1e-30)
            axes[1].pcolormesh(jax_x[:n_x], jax_z[:n_z], amp_trm, shading="auto",
                               cmap="jet", vmin=amax - 60, vmax=amax)
            axes[1].set_title("TwoRay analytical")
            axes[1].set_xlabel("Range (m)")
            # 1D comparison (calibrated)
            axes[2].plot(x_cmp, 20 * np.log10(jax_h + 1e-30), label="JAX (calibrated)")
            axes[2].plot(x_cmp, 20 * np.log10(trm_h + 1e-30), "--", label="TwoRay")
            axes[2].set_xlabel("Range (m)")
            axes[2].set_ylabel("|Field| (dB)")
            axes[2].set_title(f"z≈{z0:.0f} m | corr={corr:.3f} Δ={median_diff:.1f}dB")
            axes[2].legend()
            _save_fig("brewster_vs_tworay")


def _flat_ground_spectral_reference(src, env, x_grid, z_grid):
    """Exact one-way field of a Gaussian beam over a flat reflecting surface in a
    homogeneous medium: plane-wave spectrum of the source, each downgoing wave
    reflected with R(|k_z|) (Fresnel times the roughness factor about the mean
    level z = 0, RoughSeaSurface.mean_level_reflection). Returns the PE envelope
    u = psi exp(-i k x)."""
    from pywaveprop.propagators._utils import reflection_coef
    k = 2 * np.pi * src.freq_hz / float(env.wave_speed_profile(0.0))
    k0 = 2 * np.pi * src.freq_hz / 3e8
    n = 2 ** 16
    zz = (np.arange(n) - n // 2) * 0.00625
    ww = np.sqrt(2 * np.log(2)) / (k0 * np.sin(np.radians(src.beam_width_deg) / 2))
    g = np.exp(-1j * k0 * np.sin(np.radians(src.elevation_angle_deg)) * zz) \
        * np.exp(-((zz - src.height_m) / ww) ** 2) / (np.sqrt(np.pi) * ww)
    kz = 2 * np.pi * np.fft.fftfreq(n, d=zz[1] - zz[0])
    spec = np.fft.fft(np.fft.ifftshift(g))
    theta = np.degrees(np.arcsin(np.minimum(np.abs(kz) / k, 1)))
    eps = env.ground_material.complex_permittivity(src.freq_hz)
    if env.sea_roughness is not None:
        smooth = lambda t: reflection_coef(1, eps, 90 - t, src.polarization)
        refl = env.sea_roughness.mean_level_reflection(np.minimum(np.abs(kz), k), smooth, k)
    else:
        refl = np.array([reflection_coef(1, eps, 90 - t, src.polarization) for t in theta])
    kx = np.sqrt(k ** 2 - kz ** 2 + 0j)
    out = np.empty((len(x_grid), len(z_grid)), complex)
    for i, x in enumerate(x_grid):
        ph = np.where(np.abs(kz) < k, np.exp(1j * (kx - k) * x), 0)
        inc = np.fft.fftshift(np.fft.ifft(spec * ph))
        ref = np.fft.fftshift(np.fft.ifft(spec * refl * ph))
        # the reflected wave at height z is the mirrored (image) spectrum at -z
        out[i] = np.interp(z_grid, zz, inc.real) + 1j * np.interp(z_grid, zz, inc.imag) \
            + np.interp(-z_grid, zz, ref.real) + 1j * np.interp(-z_grid, zz, ref.imag)
    return out


class TestRoughSeaNLBC(unittest.TestCase):
    """Lower NLBC over a rough sea surface (Miller-Brown / Ament).

    The analytic continuation of the roughness factor grows off the real k_z
    axis, so 1 + R has zeros inside the NLBC contour loop. Two remedies:
    'fit' approximates 1 + R by the zero-free (1 + R_f) exp(K), K a polynomial
    fitted on the propagating range and kept moderate on the loop (lifting the
    BC by a small z0 only if needed); 'shift' lifts the BC to the height z0
    where R = R_f rho exp(2 i k_z z0) is bounded by 1 on the loop. The field is
    validated against the exact plane-wave-spectrum solution for a Gaussian
    beam over a flat sea in a homogeneous atmosphere.
    """

    SEA = GroundMaterial(eps=70, sigma=4)

    @classmethod
    def _scenario(cls, pol, sigma_m, model='miller_brown', method='fit', reference_height_m=None,
                  height_m=30.0, elev_deg=5.0, grazing_angle_deg=None):
        src = RWPGaussSourceModel(freq_hz=3e9, height_m=height_m, beam_width_deg=3.0,
                                  elevation_angle_deg=elev_deg, polarization=pol)
        env = TroposphereModel(M0=320, slope=0.0, ground_material=cls.SEA,
                               sea_roughness=RoughSeaSurface(sigma_m, model=model, method=method,
                                                             reference_height_m=reference_height_m,
                                                             grazing_angle_deg=grazing_angle_deg))
        params = RWPComputationalParams(
            max_range_m=2 * height_m / fm.tan(fm.radians(elev_deg)) + 100, max_height_m=2.5 * height_m,
            max_angle_deg=elev_deg + 5, x_output_points=200, z_output_points=300, precision=0.01)
        return src, env, params

    def _error_vs_exact(self, src, env, params, name):
        field = rwp_forward_task(src, env, params)
        x = np.asarray(field.x_grid)
        z = np.asarray(field.z_grid)
        u = np.asarray(field.field)[:len(x), :len(z)]
        ref = _flat_ground_spectral_reference(src, env, x, z)
        after = slice(len(x) // 3, None)  # past the specular point
        err = np.linalg.norm(np.abs(u[after]) - np.abs(ref[after])) / np.linalg.norm(np.abs(ref[after]))
        if PLOT_DIR:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            for ax, arr, lbl in [(axes[0], u, "PE, rough-sea NLBC"), (axes[1], ref, "exact")]:
                ax.pcolormesh(x, z, 20 * np.log10(np.abs(arr).T + 1e-30), shading="auto",
                              cmap="jet", vmin=-60, vmax=0)
                ax.set_title(f"{lbl} ({name})")
                ax.set_xlabel("Range (m)")
            axes[0].set_ylabel("Height (m)")
            _save_fig(f"rough_sea_{name}")
        return err, field

    def test_factor_on_real_axis(self):
        from scipy.special import i0
        k_z = np.linspace(0, 10, 51)
        sigma = 0.2
        x = 2 * (sigma * k_z) ** 2
        mb = RoughSeaSurface(sigma).factor(k_z)
        ament = RoughSeaSurface(sigma, model='ament').factor(k_z)
        np.testing.assert_allclose(mb, np.exp(-x) * i0(x), rtol=1e-12)
        np.testing.assert_allclose(ament, np.exp(-x), rtol=1e-12)
        self.assertAlmostEqual(abs(mb[0] - 1), 0.0, places=14)

    def test_from_wind_speed(self):
        self.assertAlmostEqual(RoughSeaSurface.from_wind_speed(10).rms_height_m, 0.51)
        with self.assertRaises(ValueError):
            RoughSeaSurface(0.1, model='unknown')
        with self.assertRaises(ValueError):
            RoughSeaSurface(0.1, method='unknown')

    def test_shifted_reflection_is_mean_level_reflection_at_real_angles(self):
        """At real grazing angles the BC reflection coefficient is the rough-surface
        one about the mean level, re-referenced to the plane z0."""
        from pywaveprop.propagators._utils import reflection_coef
        rough = RoughSeaSurface(0.3)
        eps = self.SEA.complex_permittivity(3e9)
        k = 2 * np.pi * 3e9 / 3e8
        smooth = lambda th: reflection_coef(1, eps, 90 - th, 'V')
        z0 = 2.5
        func = rough.reflection_coef_func(smooth, k, z0)
        for th in [0.1, 1.0, 5.0, 20.0]:
            k_z = k * np.sin(np.radians(th))
            expected = smooth(th) * rough.factor(k_z) * np.exp(2j * k_z * z0)
            self.assertAlmostEqual(abs(func(th) - expected), 0.0, places=12)

    def test_auto_reference_height_bounds_reflection_on_contour(self):
        src, env, params = self._scenario('H', 0.3, method='shift')
        model = create_rwp_model(src, env, params)
        z0 = model.z_offset_m
        self.assertGreater(z0, 0.0)
        self.assertLess(z0, src.height_m)
        self.assertAlmostEqual(float(model.z_computational_grid()[0]), z0)
        # |R| <= 1 on the contour loop at z0 and > 1 somewhere below it
        from pywaveprop.propagators._utils import reflection_coef
        eps = self.SEA.complex_permittivity(src.freq_hz)
        k = 2 * np.pi * src.freq_hz / float(env.wave_speed_profile(z0))
        smooth = lambda th: reflection_coef(1, eps, 90 - th, 'H')
        k_z = model.lower_nlbc_contour_kz()
        theta = np.arcsin(k_z / k) * 180 / np.pi
        for h, bounded in [(z0, True), (0.9 * z0, False)]:
            func = env.sea_roughness.reflection_coef_func(smooth, k, h)
            r_max = max(abs(func(t)) for t in theta)
            if bounded:
                self.assertLessEqual(r_max, 1 + 1e-6)
            else:
                self.assertGreater(r_max, 1.0)

    def test_fit_keeps_boundary_at_mean_level(self):
        """For moderate roughness the fitted zero-free R~ needs (almost) no lift, matches
        R_f rho on the propagating range and is zero-free by construction."""
        src, env, params = self._scenario('H', 0.1)
        model = create_rwp_model(src, env, params)
        # no lift, or a negligible one (a fraction of the rms wave height)
        self.assertLessEqual(model.z_offset_m, 0.1 * env.sea_roughness.rms_height_m)
        info = model.rough_sea_fit_info
        self.assertLessEqual(info['fit_error'], env.sea_roughness.fit_tol)
        self.assertLessEqual(info['max_abs_re_K_on_contour'], env.sea_roughness.fit_max_log)
        # passive for all propagating waves the NLBC sees
        self.assertLessEqual(info['max_abs_R_real_axis'], 1 + 1e-6)

    def test_fit_passive_in_trapping_range(self):
        """Over a duct the fitted R~ is passive in the quarter disc of vertical
        wavenumbers the trapping layer returns to the surface, so the rough
        surface cannot resonate with the duct."""
        src = RWPGaussSourceModel(freq_hz=3e9, height_m=20.0, beam_width_deg=2.0, polarization='H')
        env = TroposphereModel(N_profile=EvaporationDuctModel(height_m=15.0), ground_material=self.SEA,
                               sea_roughness=RoughSeaSurface.from_wind_speed(15))
        params = RWPComputationalParams(max_range_m=3000, max_height_m=100, x_output_points=100, z_output_points=100)
        model = create_rwp_model(src, env, params)
        info = model.rough_sea_fit_info
        self.assertGreater(info['k_z_passive'], 0.0)
        self.assertLessEqual(info['fit_error'], env.sea_roughness.fit_tol)
        func = model.rough_sea_refl_coef_func
        k = 2 * np.pi * src.freq_hz / float(env.wave_speed_profile(0.0))
        kap = info['k_z_passive']
        pts = np.concatenate([1j * np.linspace(1e-3, kap, 50), kap * np.exp(1j * np.linspace(0, np.pi / 2, 50)),
                              np.linspace(1e-3, 2 * kap, 50)])
        r = [abs(func(np.arcsin(p / k) * 180 / np.pi)) for p in pts]
        self.assertLessEqual(max(r), 1 + 1e-6)

    def test_fit_matches_continuation_at_creeping_modes(self):
        """Beyond the horizon the field is carried by creeping modes, which see
        the boundary below the real k_z axis (arg ~ -60 deg). The fit must follow
        the continued Miller-Brown coefficient there, or spurious slowly decaying
        modes appear."""
        from pywaveprop.propagators._utils import reflection_coef
        src = RWPGaussSourceModel(freq_hz=3e9, height_m=20.0, beam_width_deg=2.0, polarization='H')
        env = TroposphereModel(ground_material=self.SEA, sea_roughness=RoughSeaSurface.from_wind_speed(10))
        params = RWPComputationalParams(max_range_m=3000, max_height_m=100, x_output_points=100, z_output_points=100)
        model = create_rwp_model(src, env, params)
        info = model.rough_sea_fit_info
        self.assertGreater(info['k_z_creep'], 0.0)
        k = 2 * np.pi * src.freq_hz / float(env.wave_speed_profile(0.0))
        eps = self.SEA.complex_permittivity(src.freq_hz)
        z0 = info['reference_height_m']
        for kz in info['k_z_creep'] * np.array([0.3, 0.6]) * np.exp(-1j * np.pi / 3):
            theta = np.arcsin(kz / k) * 180 / np.pi
            exact = reflection_coef(1, eps, 90 - theta, 'H') * env.sea_roughness.factor(kz) * np.exp(2j * kz * z0)
            fitted = model.rough_sea_refl_coef_func(theta)
            self.assertLess(abs(np.log((1 + fitted) / (1 + exact))), 1e-2)

    def test_fit_lifts_less_than_shift(self):
        src, env, params = self._scenario('H', 0.3)
        z0_fit = create_rwp_model(src, env, params).z_offset_m
        src, env, params = self._scenario('H', 0.3, method='shift')
        z0_shift = create_rwp_model(src, env, params).z_offset_m
        self.assertLess(z0_fit, 0.5 * z0_shift)

    def test_fit_miller_brown_h_pol_matches_exact(self):
        src, env, params = self._scenario('H', 0.1)
        err, _ = self._error_vs_exact(src, env, params, "fit_mb_h")
        self.assertLess(err, 0.03)

    def test_fit_miller_brown_v_pol_matches_exact(self):
        src, env, params = self._scenario('V', 0.3)
        err, _ = self._error_vs_exact(src, env, params, "fit_mb_v")
        self.assertLess(err, 0.03)

    def test_fit_ament_matches_exact(self):
        src, env, params = self._scenario('H', 0.3, model='ament')
        err, _ = self._error_vs_exact(src, env, params, "fit_ament_h")
        self.assertLess(err, 0.03)

    def test_shift_miller_brown_h_pol_matches_exact(self):
        src, env, params = self._scenario('H', 0.3, method='shift')
        err, _ = self._error_vs_exact(src, env, params, "shift_mb_h")
        self.assertLess(err, 0.03)

    def test_shift_miller_brown_v_pol_matches_exact(self):
        src, env, params = self._scenario('V', 0.1, method='shift')
        err, _ = self._error_vs_exact(src, env, params, "shift_mb_v")
        self.assertLess(err, 0.03)

    def test_roughness_reduces_specular_reflection(self):
        """The reflected beam over a rough sea is weaker than over a smooth one."""
        src, env, params = self._scenario('H', 0.3)
        rough = rwp_forward_task(src, env, params)
        env_smooth = TroposphereModel(M0=320, slope=0.0, ground_material=self.SEA)
        smooth = rwp_forward_task(src, env_smooth, params)
        # vertical cut through the reflected beam at z = h/2 (x = 1.5 h / tan(elev))
        x_r = 1.5 * src.height_m / fm.tan(fm.radians(src.elevation_angle_deg))
        amp_r = abs(rough.value(x_r, src.height_m / 2))
        amp_s = abs(smooth.value(x_r, src.height_m / 2))
        self.assertLess(amp_r, 0.5 * amp_s)

    def test_unshifted_boundary_is_wrong(self):
        """Regression for the motivation of z0: with the BC at the mean level the
        continued Miller-Brown factor makes 1 + R vanish inside the contour and
        the solution departs from the exact one."""
        src, env, params = self._scenario('H', 0.3, method='shift', reference_height_m=0.0)
        err, _ = self._error_vs_exact(src, env, params, "mb_h_unshifted")
        self.assertGreater(err, 0.2)

    def _smooth_sea(self, pol, freq_hz=3e9):
        from pywaveprop.propagators._utils import reflection_coef
        eps = self.SEA.complex_permittivity(freq_hz)
        return lambda th: reflection_coef(1, eps, 90 - th, pol)

    def test_grazing_blend_impedance_constant_and_even(self):
        """Barrick (1998): the roughness-modified impedance tends to a constant at
        grazing. The blend is constant below the blend angle, follows
        Miller-Brown well above it and is even in k_z (no lateral wave)."""
        k = 2 * np.pi * 3e9 / 3e8
        for pol in 'HV':
            smooth = self._smooth_sea(pol)
            mb = RoughSeaSurface.from_wind_speed(10)
            blend = RoughSeaSurface.from_wind_speed(10, grazing_angle_deg='rayleigh')
            k_g = blend.grazing_k_z(k)
            self.assertAlmostEqual(k_g, 1 / (2 * blend.rms_height_m))

            def kappa(rough, k_z):
                r = rough.mean_level_reflection(k_z, smooth, k)
                return k_z * (r - 1) / (r + 1)
            low = k_g * np.array([1e-4, 1e-3, 1e-2, 0.05])
            kap_low = kappa(blend, low)
            np.testing.assert_allclose(kap_low, blend.grazing_impedance(smooth, k) * k, rtol=5e-3)
            if pol == 'H':
                # Miller-Brown's H-pol impedance falls ~1/k_z over the same range
                kap_mb = kappa(mb, low)
                self.assertGreater(abs(kap_mb[0]) / abs(kap_mb[-1]), 5)
            # even in k_z
            kz = k_g * np.array([0.1, 0.5, 2.0])
            np.testing.assert_allclose(kappa(blend, -kz), kappa(blend, kz), rtol=1e-8)
            # the model is recovered well above the blend angle
            high = np.array([20.0, 40.0]) * k_g
            np.testing.assert_allclose(blend.mean_level_reflection(high, smooth, k),
                                       mb.mean_level_reflection(high, smooth, k), atol=5e-3)
            # passive on the real axis
            kz = k * np.sin(np.radians(np.linspace(1e-4, 89.9, 3000)))
            self.assertLessEqual(np.max(np.abs(blend.mean_level_reflection(kz, smooth, k))), 1.0)
            # R -> -1 at grazing, as for the smooth sea
            self.assertAlmostEqual(abs(blend.mean_level_reflection(1e-9 * k, smooth, k)[0] + 1), 0.0, places=6)

    def test_grazing_blend_without_option_is_model(self):
        k = 2 * np.pi * 3e9 / 3e8
        smooth = self._smooth_sea('H')
        rough = RoughSeaSurface(0.3)
        kz = np.linspace(0.01, 5, 20)
        r_f = np.array([smooth(t) for t in np.degrees(np.arcsin(kz / k))])
        np.testing.assert_allclose(rough.mean_level_reflection(kz, smooth, k), r_f * rough.factor(kz), rtol=1e-12)
        self.assertEqual(rough.grazing_k_z(k), 0.0)
        self.assertAlmostEqual(RoughSeaSurface(0.3, grazing_angle_deg=2.0).grazing_k_z(k), k * np.sin(np.radians(2.0)))

    def test_grazing_blend_validation(self):
        with self.assertRaises(ValueError):
            RoughSeaSurface(0.3, method='shift', grazing_angle_deg='rayleigh')
        for bad in ('auto', 0.0, 95.0):
            with self.assertRaises(ValueError):
                RoughSeaSurface(0.3, grazing_angle_deg=bad)

    def test_fit_grazing_blend_matches_exact(self):
        src, env, params = self._scenario('H', 0.3, grazing_angle_deg='rayleigh')
        err, field = self._error_vs_exact(src, env, params, "fit_mb_blend_h")
        self.assertLess(err, 0.03)
        model = create_rwp_model(src, env, params)
        info = model.rough_sea_fit_info
        self.assertLessEqual(info['fit_error'], env.sea_roughness.fit_tol)
        self.assertAlmostEqual(info['k_z_grazing'], 1 / (2 * 0.3))
        self.assertLess(info['grazing_impedance'].real, 0.0)
        # the fitted impedance keeps the constant grazing value and has no odd
        # (lateral-wave) term: K is matched to first order at k_z = 0
        k = 2 * np.pi * src.freq_hz / float(env.wave_speed_profile(0.0))
        z0 = model.z_offset_m

        def kappa(kz):
            r = model.rough_sea_refl_coef_func(np.arcsin(kz / k) * 180 / np.pi) * np.exp(-2j * kz * z0)
            return kz * (r - 1) / (r + 1)
        for h in (1e-4, 1e-3):
            self.assertLess(abs(kappa(h) - kappa(-h)) / abs(kappa(h)), 1e-6)
            self.assertLess(abs(kappa(h) / k - info['grazing_impedance']), 1e-3 * abs(info['grazing_impedance']))

    def test_grazing_blend_warns_inside_creeping_range(self):
        """At X band and 15 m/s the Rayleigh blend angle is below the creeping-mode
        range, where the far field depends on the blend shape."""
        src = RWPGaussSourceModel(freq_hz=10e9, height_m=20.0, beam_width_deg=2.0, polarization='H')
        env = TroposphereModel(ground_material=self.SEA,
                               sea_roughness=RoughSeaSurface.from_wind_speed(15, grazing_angle_deg='rayleigh'))
        params = RWPComputationalParams(max_range_m=2000, max_height_m=60, x_output_points=50, z_output_points=50)
        with self.assertWarnsRegex(UserWarning, 'creeping-mode range'):
            create_rwp_model(src, env, params)

    def test_split_step_methods_reject_roughness(self):
        src, env, params = self._scenario('H', 0.1)
        params.approx_method = 'ssf'
        with self.assertRaises(NotImplementedError):
            create_rwp_model(src, env, params)


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


class TestVegetationLayer(unittest.TestCase):
    """The vegetation slab must attenuate the field like a homogeneous lossy
    dielectric when it fills the whole computational domain."""

    def test_slab_attenuation_matches_theory(self):
        import cmath
        from pywaveprop.helmholtz_jax import PiecewiseLinearTerrainModel
        from pywaveprop.rwp_jax import (GroundMaterial, RWPComputationalParams,
                                        RWPGaussSourceModel, TroposphereModel,
                                        VegetationLayerModel, create_rwp_model)
        f = 300e6
        k0 = 2 * fm.pi * f / 3e8
        x = jnp.linspace(0, 1000, 11)
        terrain = PiecewiseLinearTerrainModel(x, jnp.zeros_like(x))
        material = GroundMaterial(eps=1.05, sigma=2e-4)
        src = RWPGaussSourceModel(freq_hz=f, height_m=100.0, beam_width_deg=5.0,
                                  polarization='V')
        params = RWPComputationalParams(max_range_m=1000, max_height_m=200,
                                        dx_m=20.0, dz_m=2.0, precision=1e-2)
        params.rational_approx_order = (7, 8)
        fields = {}
        for tag, veg in (("air", None),
                         ("slab", VegetationLayerModel(
                             thickness=PiecewiseLinearTerrainModel(x, jnp.full_like(x, 300.0)),
                             material=material))):
            env = TroposphereModel(terrain=terrain,
                                   ground_material=GroundMaterial(eps=15, sigma=0.005),
                                   vegetation=veg)
            model = create_rwp_model(src, env, params)
            fields[tag] = (np.asarray(model.compute(src.aperture(model.z_computational_grid()))),
                           np.asarray(model.x_output_grid()), np.asarray(model.z_output_grid()))
        u_air, xg, zg = fields["air"]
        u_slab = fields["slab"][0]
        iz = int(np.argmin(np.abs(zg - 100.0)))
        ix = int(np.argmin(np.abs(xg - 500.0)))
        ratio_db = 20 * np.log10(np.abs(u_slab[ix, iz]) / np.abs(u_air[ix, iz]))
        eps_c = material.complex_permittivity(f)
        theory_db = -8.686 * k0 * cmath.sqrt(eps_c).imag * xg[ix]
        self.assertLess(abs(ratio_db - theory_db), 0.02 * abs(theory_db))
