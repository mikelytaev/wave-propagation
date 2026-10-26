"""
Tests for the orthogonal terrain-following coordinates
(:mod:`pywaveprop.terrain_transform`) and the rational propagator built on them
(:mod:`pywaveprop.helmholtz_tf_jax`).

* geometry of the map: orthogonality, exact boundary, identity region, vertical
  starting trajectory;
* flat ground: the terrain-following propagator reproduces
  :class:`pywaveprop.helmholtz_jax.RationalHelmholtzPropagator` to round-off,
  including the upper nonlocal boundary condition;
* an exact solution (complex-source-point beam reflected by a perfectly
  conducting slope) is reproduced to well below the staircase error;
* the Leontovich impedance condition: discrete reflection coefficient of the
  compact closure, and an exact beam reflected by a sloping impedance plane;
* the troposphere and underwater front-ends run end to end.
"""
import unittest

import jax.numpy as jnp
import numpy as np
from scipy.special import hankel1

from pywaveprop.helmholtz_common import HelmholtzMeshParams2D
from pywaveprop.helmholtz_jax import (ConstWaveSpeedModel, LinearSlopeWaveSpeedModel, PiecewiseLinearTerrainModel,
                                      RationalHelmholtzPropagator)
from pywaveprop.helmholtz_tf_jax import TerrainFollowingHelmholtzPropagator, TFMedium, fourth_root_factors
from pywaveprop.helmholtz_ssf_jax import impedance_alpha
from pywaveprop.rwp_jax import (GroundMaterial, ProxyWaveSpeedModel, RWPComputationalParams, RWPGaussSourceModel, TroposphereModel,
                                rwp_forward_task)
from pywaveprop.rwp_field import TerrainFollowingRWPField
from pywaveprop.terrain_transform import OrthogonalTerrainMap
from pywaveprop.uwa_jax import UnderwaterEnvironmentModel, UnderwaterLayerModel, UWAGaussSourceModel, uwa_forward_task
from pywaveprop.uwa_utils import UWAComputationalParams


def _fd_metrics(m, xi, eta):
    X, Z = m.x, m.z
    tx = np.gradient(X, xi, axis=0), np.gradient(Z, xi, axis=0)
    te = np.gradient(X, eta, axis=1), np.gradient(Z, eta, axis=1)
    cos = (tx[0] * te[0] + tx[1] * te[1]) / np.hypot(*tx) / np.hypot(*te)
    return cos, np.hypot(*tx), np.hypot(*te)


class TestOrthogonalTerrainMap(unittest.TestCase):

    def test_half_space_map(self):
        xs = np.linspace(0, 5000, 501)
        hs = 150 * np.exp(-((xs - 2500) / 400) ** 2)
        tm = OrthogonalTerrainMap(xs, hs, x_max_m=5000, transition_m=400, smoothing_m=20)
        eta = np.arange(0, 700, 0.5)
        xi = np.arange(0, 5000, 5.0)
        m = tm.metrics(xi, eta)
        cos, hx, he = _fd_metrics(m, xi, eta)
        inner = (slice(2, -2), slice(2, int(tm.eta_lab / 0.5)))
        self.assertLess(np.abs(cos[inner]).max(), 1e-3)
        self.assertLess(np.abs(hx / m.h_xi - 1)[inner].max(), 1e-3)
        self.assertLess(np.abs(he / m.h_eta - 1)[inner].max(), 2e-3)
        np.testing.assert_allclose(m.z[:, 0], tm.boundary(m.x[:, 0]), atol=1e-9)
        np.testing.assert_allclose(m.x[:, -1], xi)
        np.testing.assert_allclose(m.z[:, -1], eta[-1] + tm.z_off)
        self.assertLess(np.abs(m.x[0]).max(), 1e-9)  # vertical starting trajectory

    def test_strip_map_interface(self):
        xs = np.linspace(0, 5000, 501)
        d = 200 - 120 * np.exp(-((xs - 2500) / 500) ** 2) + 0.02 * xs
        tm = OrthogonalTerrainMap(xs, d, x_max_m=5000, kind='strip', transition_m=300, smoothing_m=20,
                                  eta_b_m=float(np.ceil(d.max())))
        eta = np.arange(0, 700, 0.5)
        xi = np.arange(0, 5000, 10.0)
        m = tm.metrics(xi, eta)
        j_b = int(round(tm.eta_b / 0.5))
        np.testing.assert_allclose(m.z[:, j_b], tm.boundary(m.x[:, j_b]), atol=1e-9)
        np.testing.assert_allclose(m.z[:, 0], 0.0, atol=1e-9)
        # h_xi is continuous across the seabed level, h_eta is one-sided
        self.assertLess(np.abs(m.h_xi[:, j_b + 1] - m.h_xi[:, j_b - 1]).max(), 1e-2)
        self.assertTrue(np.all(m.h_eta_below > 0))

    def test_fourth_root_factors(self):
        """The flux-factor pairs use f(L1) / f(L2): what matters is the
        logarithmic derivative of the approximant, and that no pole or zero lies
        on the real axis (evanescent spectrum)."""
        L = np.linspace(-0.5, 0.2, 50)
        exact = 0.25 / (1 + L)
        for q, tol in ((2, 5e-2), (3, 5e-3)):
            f = fourth_root_factors(q)
            ld = sum(a / (1 + a * L) - b / (1 + b * L) for a, b in f)
            self.assertLess(np.max(np.abs(ld - exact) / exact), tol)
            for a, b in f:
                self.assertGreater(abs((-1 / a).imag), 1e-2)
                self.assertGreater(abs((-1 / b).imag), 1e-2)


class TestFlatGroundEquivalence(unittest.TestCase):

    def test_matches_rational_propagator(self):
        f = 3e8
        ws = ProxyWaveSpeedModel(TroposphereModel())
        order, beta, dx, dz, x_n, z_n = (6, 7), 2 * np.pi, 5.0, 0.25, 201, 801
        ref = RationalHelmholtzPropagator(order=order, beta=beta, dx_m=dx, dz_m=dz, x_n=x_n, z_n=z_n,
                                          x_grid_scale=10, z_grid_scale=4, freq_hz=f, wave_speed=ws)
        tm = OrthogonalTerrainMap([0, 3000], [0, 0], x_max_m=1000, transition_m=100.0, smoothing_m=10.0)
        med = TFMedium(k2_above=lambda x, z: (2 * np.pi * f / np.asarray(ws(jnp.asarray(z)))) ** 2)
        tables = TerrainFollowingHelmholtzPropagator.build_tables(tm, med, beta=beta, dxi=dx, deta=dz,
                                                                  x_n=x_n, z_n=z_n)
        tf = TerrainFollowingHelmholtzPropagator(order=order, beta=beta, dx_m=dx, dz_m=dz, x_n=x_n, z_n=z_n,
                                                 x_grid_scale=10, z_grid_scale=4, freq_hz=f, tables=tables,
                                                 terrain_map=tm)
        src = RWPGaussSourceModel(freq_hz=f, height_m=50, beam_width_deg=20, elevation_angle_deg=3)
        init = src.aperture(ref.z_computational_grid())
        a = np.asarray(ref.compute(init))
        b = np.asarray(tf.compute(init))
        self.assertLess(np.abs(a - b).max() / np.abs(a).max(), 1e-11)

    def test_output_rows_match_output_grid(self):
        """One output row per point of x_output_grid(), also when x_n is a multiple
        of the output stride."""
        f, beta, dx, dz, x_n, z_n = 3e8, 2 * np.pi, 5.0, 0.25, 200, 400
        tm = OrthogonalTerrainMap([0, 3000], [0, 0], x_max_m=1000, transition_m=30.0, smoothing_m=10.0)
        med = TFMedium(k2_above=lambda x, z: np.full(np.shape(z), beta ** 2, dtype=complex))
        tables = TerrainFollowingHelmholtzPropagator.build_tables(tm, med, beta=beta, dxi=dx, deta=dz,
                                                                  x_n=x_n, z_n=z_n)
        tf = TerrainFollowingHelmholtzPropagator(order=(3, 4), beta=beta, dx_m=dx, dz_m=dz, x_n=x_n, z_n=z_n,
                                                 x_grid_scale=10, z_grid_scale=4, freq_hz=f, tables=tables,
                                                 terrain_map=tm)
        out = tf.compute(jnp.zeros(z_n, dtype=complex))
        self.assertEqual(out.shape[0], len(tf.x_output_grid()))
        self.assertEqual(out.shape, tf.output_metrics()[0].shape)

    def test_dirichlet_row_after_jit(self):
        """Regression: after the pytree round trip of jax.jit the zero placeholder
        of the lower NLBC kernels must not switch the lower NLBC on."""
        ref = RationalHelmholtzPropagator(order=(3, 4), beta=2 * np.pi, dx_m=5.0, dz_m=0.25, x_n=60, z_n=400,
                                          x_grid_scale=1, z_grid_scale=1, freq_hz=3e8,
                                          wave_speed=ConstWaveSpeedModel(3e8))
        src = RWPGaussSourceModel(freq_hz=3e8, height_m=10, beam_width_deg=30, elevation_angle_deg=-10)
        f = np.asarray(ref.compute(src.aperture(ref.z_computational_grid())))
        self.assertLess(np.abs(f[:, 0]).max(), 1e-12 * np.abs(f).max())


class TestExactSlope(unittest.TestCase):
    """Complex-source-point beam reflected by a perfectly conducting 10 degree slope."""

    def test_reflected_beam(self):
        freq, k0 = 3e8, 2 * np.pi
        slope = np.tan(np.radians(10.0))
        xs, zs, b, alpha = -50.0, 30.0, 100.0, np.radians(-10.0)

        def csp(x, z, x0, z0, ang):
            r = np.sqrt((x - x0 - 1j * b * np.cos(ang)) ** 2 + (z - z0 - 1j * b * np.sin(ang)) ** 2)
            return 1j / 4 * hankel1(0, k0 * r)
        n = np.array([-slope, 1.0]) / np.hypot(slope, 1.0)
        refl = np.eye(2) - 2 * np.outer(n, n)
        ri = refl @ np.array([xs, zs])
        ei = refl @ np.array([np.cos(alpha), np.sin(alpha)])

        def exact(x, z):
            return csp(x, z, xs, zs, alpha) - csp(x, z, ri[0], ri[1], np.arctan2(ei[1], ei[0]))
        x_max = 250.0
        relief = slope * x_max
        xb = np.linspace(-2000, x_max + 2000, 4001)
        tm = OrthogonalTerrainMap(xb, slope * xb, x_max_m=x_max, transition_m=2.5 * relief, z_ref_m=relief,
                                  smoothing_m=0.0, mirror_start=False, dx_fine_m=0.5, pad_m=1500)
        med = TFMedium(k2_above=lambda x, z: np.full(np.shape(z), k0 ** 2, dtype=complex))
        z_top = relief + 2.5 * relief + 150
        kz_max = k0 * np.sin(np.radians(40.0))
        mp = HelmholtzMeshParams2D(x_size_m=x_max, z_size_m=z_top - relief, dx_output_m=5.0, dz_output_m=1.0)
        tf = TerrainFollowingHelmholtzPropagator.create(freq, tm, med, kz_max, (k0, k0), 0.1, mp,
                                                        rational_approx_order=(7, 8))
        x0, z0 = tf.initial_points()
        u = np.asarray(tf.compute(jnp.asarray(exact(x0, z0))))
        xo, zo = tf.output_metrics()
        ue = exact(xo, zo) * np.exp(-1j * tf.beta * np.asarray(tf.x_output_grid())[:, None])
        inside = zo < z_top - 50
        err_tf = np.abs(u - ue)[inside].max() / np.abs(ue[inside]).max()

        st = RationalHelmholtzPropagator.create(
            freq_hz=freq, wave_speed=ConstWaveSpeedModel(3e8), kz_max=kz_max, k_bounds=(k0, k0), precision=0.1,
            mesh_params=HelmholtzMeshParams2D(x_size_m=x_max, z_size_m=z_top, dx_output_m=5.0, dz_output_m=1.0),
            lower_terrain=PiecewiseLinearTerrainModel(np.array([0.0, x_max]), np.array([0.0, relief])),
            rational_approx_order=(7, 8))
        zg = np.asarray(st.z_computational_grid())
        us = np.asarray(st.compute(jnp.asarray(exact(np.zeros_like(zg), zg))))
        XX, ZZ = np.meshgrid(np.asarray(st.x_output_grid()), np.asarray(st.z_output_grid()), indexing='ij')
        ues = exact(XX, ZZ) * np.exp(-1j * st.beta * XX)
        ok = (ZZ > slope * XX + 2) & (ZZ < z_top - 50)
        err_st = np.abs(us - ues)[ok].max() / np.abs(ues[ok]).max()
        self.assertLess(err_tf, 5e-3)
        self.assertLess(10 * err_tf, err_st)


class TestImpedanceGround(unittest.TestCase):

    @staticmethod
    def _sloping_impedance_beam(slope_deg, eps, pol, k0, xs=-50.0, zs=30.0, inc_deg=-5.0, dphi=np.radians(2.0)):
        """Gaussian beam as a sum of plane waves, each reflected by the sloping
        plane with the Leontovich coefficient: exact for the impedance condition."""
        th = np.radians(slope_deg)
        c, s = np.cos(th), np.sin(th)
        xc, zc = xs * c + zs * s, -xs * s + zs * c
        phi0 = th - np.radians(inc_deg)
        phi = np.linspace(max(phi0 - 6 * dphi, 1e-3), phi0 + 6 * dphi, 3001)
        d = np.sqrt(complex(eps) - 1)
        d = d if pol == 'H' else d / eps
        refl = (np.sin(phi) - d) / (np.sin(phi) + d)
        w = np.exp(-((phi - phi0) / dphi) ** 2 - 1j * k0 * (xc * np.cos(phi) - zc * np.sin(phi))) * np.gradient(phi)

        def f(x, z):
            x, z = np.asarray(x, dtype=float), np.asarray(z, dtype=float)
            xp, zp = (x * c + z * s).ravel()[:, None], (-x * s + z * c).ravel()[:, None]
            e = np.exp(1j * k0 * np.cos(phi) * xp)
            v = (e * (np.exp(-1j * k0 * np.sin(phi) * zp) + refl * np.exp(1j * k0 * np.sin(phi) * zp))) @ w
            return v.reshape(x.shape)
        return f

    def test_compact_closure_reflection(self):
        """|R_h - R| <= 2 |1 - S/q| / (S/q), S = 3 sin q / (2 + cos q), for any passive ground."""
        theta = np.radians(np.linspace(0.5, 85, 200))
        for eps in (4 + 0.006j, 25 + 1.2j, 70 + 300j):
            for pol in ('H', 'V'):
                d = np.sqrt(eps - 1)
                d = d if pol == 'H' else d / eps
                r_exact = (np.sin(theta) - d) / (np.sin(theta) + d)
                for tau in (0.25, 0.5, 1.0):
                    q, g = tau * np.sin(theta), 1j * tau * d
                    p0, q0 = (4 * g / 3) / (1 - g / 3), (1 + g / 3) / (1 - g / 3)
                    r_h = -(np.exp(1j * q) - p0 - q0 * np.exp(-1j * q)) / (np.exp(-1j * q) - p0 - q0 * np.exp(1j * q))
                    sigma = 3 * np.sin(q) / (2 + np.cos(q)) / q
                    self.assertTrue(np.all(np.abs(r_h - r_exact) <= 2 * np.abs(1 - sigma) / sigma + 1e-12))

    def test_impedance_slope_exact(self):
        freq, k0 = 3e8, 2 * np.pi
        slope_deg, x_max = 10.0, 250.0
        slope = np.tan(np.radians(slope_deg))
        relief = slope * x_max
        xb = np.linspace(-2000, x_max + 2000, 4001)
        tm = OrthogonalTerrainMap(xb, slope * xb, x_max_m=x_max, transition_m=2.5 * relief, z_ref_m=relief,
                                  smoothing_m=0.0, mirror_start=False, dx_fine_m=0.5, pad_m=1500)
        med = TFMedium(k2_above=lambda x, z: np.full(np.shape(z), k0 ** 2, dtype=complex))
        z_top = relief + 2.5 * relief + 150
        mp = HelmholtzMeshParams2D(x_size_m=x_max, z_size_m=z_top - relief, dx_output_m=5.0, dz_output_m=1.0)
        eps = 25 + 1.2j
        for pol in ('H', 'V'):
            exact = self._sloping_impedance_beam(slope_deg, eps, pol, k0)
            tf = TerrainFollowingHelmholtzPropagator.create(
                freq, tm, med, k0 * np.sin(np.radians(35.0)), (k0, k0), 0.1, mp, rational_approx_order=(7, 8),
                lower_bc='impedance', lower_impedance_alpha=impedance_alpha(k0, eps, pol))
            x0, z0 = tf.initial_points()
            u = np.asarray(tf.compute(jnp.asarray(exact(x0, z0))))
            xo, zo = tf.output_metrics()
            ue = exact(xo, zo) * np.exp(-1j * tf.beta * np.asarray(tf.x_output_grid())[:, None])
            inside = zo < z_top - 50
            err = np.abs(u - ue)[inside].max() / np.abs(ue[inside]).max()
            self.assertLess(err, 5e-3, pol)


class TestFrontEnds(unittest.TestCase):

    def test_rwp_transform(self):
        xs = np.linspace(0, 3000, 61)
        hs = 60 * np.exp(-((xs - 1500) / 300) ** 2) + 20
        env = TroposphereModel(terrain=PiecewiseLinearTerrainModel(xs, hs))
        src = RWPGaussSourceModel(freq_hz=1e8, height_m=60, beam_width_deg=10)
        p = RWPComputationalParams(max_range_m=3000, max_height_m=400, dx_m=50, dz_m=2)
        p.terrain_method = 'transform'
        f = rwp_forward_task(src, env, p)
        self.assertIsInstance(f, TerrainFollowingRWPField)
        self.assertTrue(np.all(np.isfinite(f.field)))
        # the lowest row lies on the (smoothed) terrain, the top row is Cartesian
        np.testing.assert_allclose(f.z_phys[:, 0], np.interp(f.x_phys[:, 0], f.terrain_x, f.terrain_z), atol=1e-6)
        np.testing.assert_allclose(f.x_phys[:, -1], f.x_grid, atol=1e-9)
        pl = f.path_loss()
        self.assertEqual(pl.field.shape, f.field.shape)
        grid = f.on_regular_grid(np.array([1000.0, 2000.0]), np.array([100.0, 200.0, 300.0]))
        self.assertTrue(np.all(np.isfinite(grid)))

    def test_rwp_transform_impedance(self):
        xs = np.linspace(0, 3000, 61)
        hs = 60 * np.exp(-((xs - 1500) / 300) ** 2) + 20
        src = RWPGaussSourceModel(freq_hz=1e8, height_m=60, beam_width_deg=10, polarization='V')
        p = RWPComputationalParams(max_range_m=3000, max_height_m=400, dx_m=50, dz_m=2)
        p.terrain_method = 'transform'
        pec = rwp_forward_task(src, TroposphereModel(terrain=PiecewiseLinearTerrainModel(xs, hs)), p)
        f = rwp_forward_task(src, TroposphereModel(terrain=PiecewiseLinearTerrainModel(xs, hs),
                                                   ground_material=GroundMaterial(eps=15, sigma=0.005)), p)
        self.assertTrue(np.all(np.isfinite(f.field)))
        # a lossy ground absorbs: the field near the ground behind the hill is weaker than over PEC
        self.assertLess(np.abs(f.field[-1, :5]).max(), np.abs(pec.field[-1, :5]).max())

    def test_rwp_transform_beta_column(self):
        # one reference wavenumber per class of columns, absorbing top: same field as the global
        # beta with the NLBC, on a coarser range grid
        xs = np.linspace(0, 3000, 61)
        hs = 150 * np.exp(-((xs - 1500) / 250) ** 2) + 20
        env = TroposphereModel(terrain=PiecewiseLinearTerrainModel(xs, hs),
                               ground_material=GroundMaterial(eps=15, sigma=0.005))
        src = RWPGaussSourceModel(freq_hz=1e8, height_m=60, beam_width_deg=10, elevation_angle_deg=3,
                                  polarization='V')
        p = RWPComputationalParams(max_range_m=3000, max_height_m=600, max_angle_deg=30, dx_m=25, dz_m=1)
        p.terrain_method = 'transform'
        p.terrain_transition_m = 400
        p.terrain_smoothing_m = 50
        p.terrain_slope_aperture = False
        ref = rwp_forward_task(src, env, p)
        p.terrain_beta_mode = 'column'
        p.terrain_absorb_m = 200.0
        col = rwp_forward_task(src, env, p)
        n = ref.field.shape[1]
        np.testing.assert_allclose(col.z_phys[:, :n], ref.z_phys, atol=1e-6)
        pl_ref, pl_col = ref.path_loss().field, col.path_loss().field[:, :n]
        inside = (ref.z_phys < 500) & (ref.x_phys > 200) & (pl_ref < 130)
        self.assertLess(np.median(np.abs(pl_col - pl_ref)[inside]), 0.05)

    def test_uwa_transform(self):
        x_b = np.array([0.0, 1000.0, 3000.0, 4000.0])
        d_b = np.array([150.0, 150.0, 80.0, 80.0])
        env = UnderwaterEnvironmentModel(
            layers=[UnderwaterLayerModel(height_m=150.0, sound_speed_profile_m_s=LinearSlopeWaveSpeedModel(1500, 0)),
                    UnderwaterLayerModel(height_m=1e3, sound_speed_profile_m_s=LinearSlopeWaveSpeedModel(1700, 0),
                                         density=1.5, attenuation_dm_lambda=0.5)],
            bathymetry=PiecewiseLinearTerrainModel(x_b, d_b))
        src = UWAGaussSourceModel(freq_hz=50.0, depth_m=60.0, beam_width_deg=20)
        p = UWAComputationalParams(max_range_m=4000, dx_m=20, dz_m=1)
        p.terrain_method = 'transform'
        res = uwa_forward_task(src, env, p)
        self.assertTrue(np.all(np.isfinite(res.field)))
        self.assertEqual(res.x_phys.shape, res.field.shape)


if __name__ == '__main__':
    unittest.main()
