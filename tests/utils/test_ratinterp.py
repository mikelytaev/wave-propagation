"""Tests for the pure-Python port of Chebfun's ``ratinterp``."""
import unittest

import numpy as np

from pywaveprop.utils.ratinterp import (
    RationalInterpolant,
    chebpts1,
    chebpts2,
    ratinterp,
)


class ChebTransformsTest(unittest.TestCase):
    """Sanity checks on the helper Chebyshev transforms."""

    def test_chebpts2_endpoints(self):
        x = chebpts2(5)
        self.assertAlmostEqual(x[0], -1.0)
        self.assertAlmostEqual(x[-1], 1.0)
        self.assertEqual(len(x), 5)

    def test_chebpts1_count(self):
        self.assertEqual(len(chebpts1(7)), 7)

    def test_cheb2_roundtrip(self):
        from pywaveprop.utils.ratinterp import (
            _cheb2_coeffs2vals,
            _cheb2_vals2coeffs,
        )

        rng = np.random.default_rng(0)
        for n in (2, 3, 5, 8, 16):
            c = rng.standard_normal(n)
            v = _cheb2_coeffs2vals(c)
            c2 = _cheb2_vals2coeffs(v)
            self.assertTrue(np.allclose(c, c2, atol=1e-13))

    def test_cheb1_roundtrip(self):
        from pywaveprop.utils.ratinterp import (
            _cheb1_coeffs2vals,
            _cheb1_vals2coeffs,
        )

        rng = np.random.default_rng(1)
        for n in (2, 3, 5, 8, 16):
            c = rng.standard_normal(n)
            v = _cheb1_coeffs2vals(c)
            c2 = _cheb1_vals2coeffs(v)
            self.assertTrue(np.allclose(c, c2, atol=1e-13))


class RatinterpExactTest(unittest.TestCase):
    """The interpolant should reproduce true rationals exactly."""

    def test_simple_pole_type2(self):
        f = lambda x: 1.0 / (x - 0.2)
        r = ratinterp(f, 0, 1)  # type2 default
        xs = np.linspace(-1.0, 1.0, 17)
        xs = xs[np.abs(xs - 0.2) > 1e-3]  # avoid pole
        err = np.max(np.abs(r(xs) - f(xs)))
        self.assertLess(err, 1e-12)
        self.assertEqual(r.mu, 0)
        self.assertEqual(r.nu, 1)

    def test_simple_pole_type1(self):
        f = lambda x: 1.0 / (x - 0.3)
        r = ratinterp(f, 0, 1, xi="type1")
        xs = np.linspace(-1.0, 1.0, 17)
        xs = xs[np.abs(xs - 0.3) > 1e-3]
        self.assertLess(np.max(np.abs(r(xs) - f(xs))), 1e-11)

    def test_double_pole_pair_type2(self):
        # f(x) = 1 / (x^2 - 0.04) = 1 / ((x - 0.2)(x + 0.2))
        f = lambda x: 1.0 / (x ** 2 - 0.04)
        r = ratinterp(f, 0, 2)
        xs = np.linspace(-1.0, 1.0, 33)
        mask = np.abs(np.abs(xs) - 0.2) > 1e-3
        xs = xs[mask]
        err = np.max(np.abs(r(xs) - f(xs)))
        self.assertLess(err, 1e-11)
        self.assertEqual(r.mu, 0)
        self.assertEqual(r.nu, 2)

    def test_quadratic_over_linear(self):
        # f(x) = (x^2 + 1) / (x - 0.4)
        f = lambda x: (x ** 2 + 1) / (x - 0.4)
        r = ratinterp(f, 2, 1)
        xs = np.linspace(-1.0, 1.0, 25)
        xs = xs[np.abs(xs - 0.4) > 1e-3]
        self.assertLess(np.max(np.abs(r(xs) - f(xs))), 1e-11)

    def test_exact_in_unitroots(self):
        # Rational on the unit disc: f(z) = 1 / (1 - 0.5 z).
        f = lambda z: 1.0 / (1.0 - 0.5 * z)
        r = ratinterp(f, 0, 1, xi="unitroots")
        # Test on a few points on the unit circle (avoiding the pole at z=2).
        thetas = np.linspace(0.0, 2 * np.pi, 19, endpoint=False)
        z = np.exp(1j * thetas)
        err = np.max(np.abs(r(z) - f(z)))
        self.assertLess(err, 1e-12)


class RatinterpRobustnessTest(unittest.TestCase):
    """Robust mode should reduce inflated degrees and clear Froissart doublets."""

    def test_overspecified_degree_reduces(self):
        # f is rational of true type (0, 1).  Asking for (4, 4) should
        # reduce to (0, 1) under robust mode.
        f = lambda x: 1.0 / (x - 0.25)
        r = ratinterp(f, 4, 4, tol=1e-13)
        self.assertEqual(r.mu, 0)
        self.assertEqual(r.nu, 1)
        xs = np.linspace(-1.0, 1.0, 21)
        xs = xs[np.abs(xs - 0.25) > 1e-3]
        self.assertLess(np.max(np.abs(r(xs) - f(xs))), 1e-11)

    def test_zero_function(self):
        f = lambda x: 0.0 * x
        r = ratinterp(f, 5, 5)
        # The convention from chebfun: zero function -> a = [0], b = [1]
        self.assertEqual(r.mu, 0)
        xs = np.linspace(-1.0, 1.0, 11)
        self.assertTrue(np.allclose(r(xs), 0.0, atol=1e-14))

    def test_constant_function(self):
        f = lambda x: 3.0 + 0.0 * x
        r = ratinterp(f, 4, 4)
        xs = np.linspace(-1.0, 1.0, 11)
        self.assertTrue(np.allclose(r(xs), 3.0, atol=1e-13))

    def test_least_squares_oversample(self):
        # Approximate exp(x) by a (5, 5) rational with 25 sample points.
        f = np.exp
        r = ratinterp(f, 5, 5, NN=25, tol=0.0)
        xs = np.linspace(-1.0, 1.0, 101)
        err = np.max(np.abs(r(xs) - f(xs)))
        # Best (5, 5) rational approximant to exp on [-1, 1] is far below
        # 1e-10 but the least-squares interpolant should easily reach 1e-10.
        self.assertLess(err, 1e-10)


class RatinterpDomainTest(unittest.TestCase):
    """Affine remapping to a non-default domain."""

    def test_nonstandard_domain(self):
        # f(x) = 1 / (x - 5) on [2, 6].  Asymmetric around the pole so
        # the symmetric branch is not triggered.
        f = lambda x: 1.0 / (x - 5.0)
        r = ratinterp(f, 0, 1, dom=(2.0, 6.0))
        xs = np.linspace(2.0, 6.0, 21)
        xs = xs[np.abs(xs - 5.0) > 1e-3]
        self.assertLess(np.max(np.abs(r(xs) - f(xs))), 1e-11)


class RatinterpPolesTest(unittest.TestCase):
    def test_recover_simple_pole(self):
        # Two simple poles inside [-1, 1].  Use irrational locations
        # that do not coincide with any Chebyshev sample point.
        pole_a, pole_b = 0.13, -0.41
        f = lambda x: 1.0 / (x - pole_a) + 2.0 / (x - pole_b)
        r = ratinterp(f, 3, 3)
        poles, residues = r.compute_poles_and_residues()
        real_poles = poles[np.abs(np.imag(poles)) < 1e-8].real
        real_residues = residues[np.abs(np.imag(poles)) < 1e-8].real
        expected_poles = np.sort([pole_a, pole_b])
        self.assertEqual(len(real_poles), 2)
        self.assertTrue(
            np.allclose(np.sort(real_poles), expected_poles, atol=1e-9)
        )
        idx_b = np.argmin(np.abs(real_poles - pole_b))
        idx_a = np.argmin(np.abs(real_poles - pole_a))
        self.assertAlmostEqual(real_residues[idx_b], 2.0, places=8)
        self.assertAlmostEqual(real_residues[idx_a], 1.0, places=8)


if __name__ == "__main__":
    unittest.main()
