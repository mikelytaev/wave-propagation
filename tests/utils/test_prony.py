"""Tests for the Prony multipath extraction utility."""
import unittest

import numpy as np

from pywaveprop.utils.prony import (
    MultipathResult,
    prony_extract,
    prony_extract_field,
)
from pywaveprop.utils.prony import _cluster_paths


C0 = 299792458.0


def _synth_samples(amps, delays, f_start, delta_f, nf):
    """Build u_n = sum_p amps[p] * exp(-j 2*pi*n*delta_f*tau_p).

    ``amps`` are eq.-(5) amplitudes (the form recovered by
    :func:`prony_extract`); ``u_0 = sum(amps)``.  ``f_start`` is
    unused in the construction but kept in the signature for clarity
    of which reference frequency the amplitudes are tied to.
    """
    del f_start  # Reference frequency is implicit in ``amps``.
    amps = np.asarray(amps, dtype=complex)
    delays = np.asarray(delays, dtype=float)
    n = np.arange(nf)
    phase = -2j * np.pi * n[:, None] * delta_f * delays[None, :]
    return np.sum(amps[None, :] * np.exp(phase), axis=1)


class PronyExactRecoveryTest(unittest.TestCase):
    """When the data is exactly a sum of complex exponentials and
    ``nf == 2 * n_paths``, Prony should recover the parameters."""

    def test_two_paths_square_system(self):
        f0 = 5e9
        # 500 kHz. Principal-value tau is in (-1/(2df), 1/(2df)] = (-1, 1) us,
        # i.e., delays up to ~300 m can be recovered without phase wrapping.
        df = 0.5e6
        amps = np.array([0.7 + 0.1j, -0.25 + 0.05j])
        delays = np.array([100.0, 250.0]) / C0
        u = _synth_samples(amps, delays, f0, df, nf=4)

        res = prony_extract(u, f_start=f0, delta_f=df, n_paths=2)

        self.assertEqual(res.n_paths, 2)
        # Match on sorted delays (result returns sorted).
        np.testing.assert_allclose(res.delays, np.sort(delays), atol=1e-15)
        # Match amplitudes (sort by delay).
        order = np.argsort(delays)
        np.testing.assert_allclose(res.amplitudes, amps[order], atol=1e-10)

    def test_three_paths(self):
        f0 = 2e9
        df = 1.0e6  # principal-value range: |tau| < 0.5 us = 150 m of path
        amps = np.array([1.0 + 0j, 0.3 + 0.4j, 0.05 - 0.02j])
        delays = np.array([0.0, 50.0, 140.0]) / C0
        u = _synth_samples(amps, delays, f0, df, nf=6)

        res = prony_extract(u, f_start=f0, delta_f=df, n_paths=3)

        order = np.argsort(delays)
        np.testing.assert_allclose(res.delays, delays[order], atol=1e-14)
        np.testing.assert_allclose(res.amplitudes, amps[order], atol=1e-10)


class PronyOverdeterminedTest(unittest.TestCase):
    def test_two_paths_extra_samples(self):
        f0 = 5e9
        df = 0.5e6
        amps = np.array([0.7, -0.25])
        delays = np.array([0.0, 240.0]) / C0
        u = _synth_samples(amps, delays, f0, df, nf=8)  # 8 > 2*2

        res = prony_extract(u, f_start=f0, delta_f=df, n_paths=2)

        np.testing.assert_allclose(np.sort(res.delays), np.sort(delays), atol=1e-15)


class PronyReconstructionTest(unittest.TestCase):
    def test_reconstruct_at_samples(self):
        f0 = 1e9
        df = 2e6
        amps = np.array([1.0 + 0j, 0.4 + 0.2j])
        delays = np.array([10.0, 80.0]) / C0
        u = _synth_samples(amps, delays, f0, df, nf=4)
        res = prony_extract(u, f_start=f0, delta_f=df, n_paths=2)

        freqs = f0 + df * np.arange(4)
        u_hat = res.reconstruct(freqs)
        np.testing.assert_allclose(u_hat, u, atol=1e-10)

    def test_reconstruct_at_off_grid_frequency(self):
        """Eq. (21): reconstructed field at an unsampled frequency."""
        f0 = 5e9
        df = 0.5e6
        amps = np.array([1.0, -0.3])
        delays = np.array([20.0, 220.0]) / C0
        u = _synth_samples(amps, delays, f0, df, nf=4)
        res = prony_extract(u, f_start=f0, delta_f=df, n_paths=2)

        # True value at an off-grid frequency.  In the eq.-(5) form
        # used by the result, u(f) = sum_p A_p exp(-j 2 pi (f - f0) tau).
        f_query = 5e9 + 0.75e6
        u_true = np.sum(amps * np.exp(-2j * np.pi * (f_query - f0) * delays))
        u_hat = res.reconstruct(f_query)
        self.assertAlmostEqual(complex(u_hat), complex(u_true), places=10)

    def test_rms_error_below_floor_for_clean_signal(self):
        f0 = 5e9
        df = 0.5e6
        amps = np.array([1.0, -0.5])
        delays = np.array([0.0, 300.0]) / C0
        u = _synth_samples(amps, delays, f0, df, nf=4)
        res = prony_extract(u, f_start=f0, delta_f=df, n_paths=2)
        self.assertLess(res.rms_error_db(u, df), -150.0)


class PronySinglePathTest(unittest.TestCase):
    """Special case np=1, nf=2: z0 = u1/u0 (eq. 13)."""

    def test_single_path_closed_form(self):
        f0 = 5e9
        df = 0.5e6
        tau = 150.0 / C0
        amp = 0.8 + 0.2j
        u = _synth_samples([amp], [tau], f0, df, nf=2)

        res = prony_extract(u, f_start=f0, delta_f=df, n_paths=1)

        self.assertEqual(res.n_paths, 1)
        self.assertAlmostEqual(res.delays[0], tau, places=14)
        self.assertAlmostEqual(complex(res.amplitudes[0]), complex(amp), places=7)


class PronyPostProcessingTest(unittest.TestCase):
    def test_amplitude_threshold_drops_weak_path(self):
        f0 = 5e9
        df = 0.5e6
        amps = np.array([1.0, 1e-4])
        delays = np.array([0.0, 200.0]) / C0
        u = _synth_samples(amps, delays, f0, df, nf=4)

        res = prony_extract(
            u, f_start=f0, delta_f=df, n_paths=2, amp_threshold=1e-3
        )
        self.assertEqual(res.n_paths, 1)
        self.assertAlmostEqual(abs(res.amplitudes[0]), 1.0, places=8)

    def test_cluster_paths_helper_formula(self):
        """Direct test of eq. (16) on the clustering helper."""
        tau = np.array([10e-9, 10.4e-9, 50e-9])
        amp = np.array([1.0 + 0j, 0.5 + 0.5j, 0.2 + 0.0j])
        delay_cluster = 1e-9  # only the first two get merged
        tau_out, amp_out = _cluster_paths(tau, amp, delay_cluster)

        self.assertEqual(len(tau_out), 2)
        w0, w1 = abs(amp[0]) ** 2, abs(amp[1]) ** 2
        expected_tau01 = (w0 * tau[0] + w1 * tau[1]) / (w0 + w1)
        self.assertAlmostEqual(tau_out[0], expected_tau01, places=14)
        self.assertAlmostEqual(complex(amp_out[0]), complex(amp[0] + amp[1]), places=12)
        self.assertAlmostEqual(tau_out[1], tau[2], places=14)
        self.assertAlmostEqual(complex(amp_out[1]), complex(amp[2]), places=12)

    def test_delay_clustering_reduces_path_count(self):
        """End-to-end: Prony followed by clustering with close paths.

        Individual amplitudes of nearly-co-located paths cannot be
        recovered uniquely (the Vandermonde matrix is ill-conditioned),
        but their *combined* contribution reconstructs ``u`` well after
        clustering.
        """
        f0 = 5e9
        df = 0.5e6
        # Two paths 0.6 m apart (well under the 1 m clustering threshold).
        amps = np.array([0.6 + 0.1j, 0.4 - 0.2j])
        delays = np.array([100.0, 100.6]) / C0
        u = _synth_samples(amps, delays, f0, df, nf=4)

        res = prony_extract(
            u,
            f_start=f0,
            delta_f=df,
            n_paths=2,
            delay_cluster=1.0 / C0,
        )
        self.assertEqual(res.n_paths, 1)
        # Clustered delay lies between the two input delays.
        self.assertGreaterEqual(res.delays[0], delays.min())
        self.assertLessEqual(res.delays[0], delays.max())
        # The one-path reconstruction is close (but not exact) to u.
        # A 0.6 m delay spread at 5 GHz produces non-trivial residuals.
        self.assertLess(res.rms_error_db(u, df), -20.0)

    def test_clustering_leaves_distant_paths_alone(self):
        f0 = 5e9
        df = 0.5e6
        amps = np.array([1.0, 0.3])
        delays = np.array([0.0, 300.0]) / C0  # 300 m apart
        u = _synth_samples(amps, delays, f0, df, nf=4)

        res = prony_extract(
            u,
            f_start=f0,
            delta_f=df,
            n_paths=2,
            delay_cluster=1.0 / C0,
        )
        self.assertEqual(res.n_paths, 2)


class PronyValidationTest(unittest.TestCase):
    def test_too_few_samples_raises(self):
        with self.assertRaises(ValueError):
            prony_extract([1, 2, 3], f_start=1e9, delta_f=1e6, n_paths=2)

    def test_non_1d_input_raises(self):
        u = np.zeros((4, 4), dtype=complex)
        with self.assertRaises(ValueError):
            prony_extract(u, f_start=1e9, delta_f=1e6, n_paths=2)

    def test_non_positive_delta_f_raises(self):
        u = np.zeros(4, dtype=complex)
        with self.assertRaises(ValueError):
            prony_extract(u, f_start=1e9, delta_f=0.0, n_paths=2)


class PronyFieldExtractionTest(unittest.TestCase):
    def test_batched_matches_per_point(self):
        rng = np.random.default_rng(7)
        f0 = 5e9
        df = 0.5e6
        nf = 4
        nx, nz = 3, 5

        # Build per-point ground truth and compare batched extraction
        # against per-point extraction.
        amps_true = rng.standard_normal((2, nx, nz)) + 1j * rng.standard_normal((2, nx, nz))
        delays_true = rng.uniform(0.0, 250.0, size=(2, nx, nz)) / C0
        # Sort along path axis at every point.
        order = np.argsort(delays_true, axis=0)
        delays_true = np.take_along_axis(delays_true, order, axis=0)
        amps_true = np.take_along_axis(amps_true, order, axis=0)

        # Build u_n in the eq.-(5) form (matches what prony_extract recovers).
        n = np.arange(nf)
        phase = -2j * np.pi * df * n[:, None, None, None] * delays_true[None, :, :, :]
        u_freq = np.sum(amps_true[None, :, :, :] * np.exp(phase), axis=1)
        self.assertEqual(u_freq.shape, (nf, nx, nz))

        amp_b, tau_b = prony_extract_field(
            u_freq, f_start=f0, delta_f=df, n_paths=2
        )
        self.assertEqual(amp_b.shape, (2, nx, nz))
        self.assertEqual(tau_b.shape, (2, nx, nz))

        np.testing.assert_allclose(tau_b, delays_true, atol=1e-12)
        np.testing.assert_allclose(amp_b, amps_true, atol=1e-9)

    def test_field_extraction_with_1d_spatial_axis(self):
        f0 = 5e9
        df = 0.5e6
        nf = 4
        n_pts = 7
        rng = np.random.default_rng(11)
        amps_true = rng.standard_normal((2, n_pts)) + 1j * rng.standard_normal((2, n_pts))
        delays_true = rng.uniform(20.0, 300.0, size=(2, n_pts)) / C0
        order = np.argsort(delays_true, axis=0)
        delays_true = np.take_along_axis(delays_true, order, axis=0)
        amps_true = np.take_along_axis(amps_true, order, axis=0)

        n = np.arange(nf)
        phase = -2j * np.pi * df * n[:, None, None] * delays_true[None, :, :]
        u_freq = np.sum(amps_true[None, :, :] * np.exp(phase), axis=1)

        amp_b, tau_b = prony_extract_field(
            u_freq, f_start=f0, delta_f=df, n_paths=2
        )
        np.testing.assert_allclose(tau_b, delays_true, atol=1e-12)
        np.testing.assert_allclose(amp_b, amps_true, atol=1e-9)


class PronyTwoRayLikeTest(unittest.TestCase):
    """Simulation A in the paper: two paths over flat ground.

    Absolute propagation times in long-range tropospheric scenarios are
    huge (~50 us at 15 km) and far exceed the principal-value range of
    Prony's method.  What the method actually recovers in that case is
    the *excess* delay (modulo 1/df); we verify this matches the
    geometric path-length difference.
    """

    def test_two_ray_path_length_difference(self):
        f0 = 5e9
        df = 0.5e6
        zs, zr = 200.0, 30.0
        x = 15000.0
        r_direct = np.sqrt(x ** 2 + (zs - zr) ** 2)
        r_reflected = np.sqrt(x ** 2 + (zs + zr) ** 2)
        amps = np.array([1.0 / r_direct, -1.0 / r_reflected])
        delays = np.array([r_direct, r_reflected]) / C0
        u = _synth_samples(amps, delays, f0, df, nf=4)

        res = prony_extract(u, f_start=f0, delta_f=df, n_paths=2)
        # The path-length difference (in metres) should be recovered.
        # ~50 us of absolute delay produces ~25 phase wraps; double
        # precision limits the recoverable excess delay to ~10 um.
        excess = (res.delays[1] - res.delays[0]) * C0
        self.assertAlmostEqual(excess, r_reflected - r_direct, places=4)
        # Reconstruction at the sample frequencies should still be exact.
        self.assertLess(res.rms_error_db(u, df), -120.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
