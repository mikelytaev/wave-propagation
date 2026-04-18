"""Tests for grid and beta optimization with Pade and ratinterp.

Efficiency metric V = p / (dx_lam * dz_lam) where p is the max propagator
order and dx_lam, dz_lam are the optimal grid steps in wavelengths. Lower V
means more coarsened grid per order, i.e. cheaper propagation. In the
regimes checked below (high propagator order, moderate-to-wide angles),
ratinterp should deliver a strictly smaller V than Pade.
"""
import logging
import math
import os
import sys
import unittest

from pywaveprop.grid_optimizer import get_optimal_grid

PI2 = 2 * math.pi


# Verbose logging is enabled by setting the env var GRID_OPT_VERBOSE=1
# (any truthy value) or by passing --grid-opt-verbose on the command line.
_VERBOSE = (
    os.environ.get('GRID_OPT_VERBOSE', '').lower() in ('1', 'true', 'yes')
    or '--grid-opt-verbose' in sys.argv
)
if '--grid-opt-verbose' in sys.argv:
    sys.argv.remove('--grid-opt-verbose')

logger = logging.getLogger('grid_optimizer_test')
if _VERBOSE and not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def _V(dx, dz, lam, order):
    p = max(order)
    return p / ((dx / lam) * (dz / lam))


def _run(f, c0, c_min, c_max, theta_deg, x_max, eps,
         order=(7, 8), method='pade'):
    k0 = PI2 * f / c0
    k_min = PI2 * f / c_max   # max speed -> min wavenumber
    k_max = PI2 * f / c_min   # min speed -> max wavenumber
    lam = c0 / f
    kz_max = k0 * math.sin(math.radians(theta_deg))
    beta, dx, dz = get_optimal_grid(
        kz_max=kz_max, k_min=k_min, k_max=k_max,
        required_prec=eps / x_max,
        propagator_order=order, approx_method=method,
    )
    if _VERBOSE:
        p = max(order)
        if math.isnan(beta) or dx <= 0.0 or dz <= 0.0:
            result = 'infeasible'
        else:
            result = (
                f"beta/k0={beta/k0:.4f} "
                f"dx={dx:.4g} m ({dx/lam:.3f} lam) "
                f"dz={dz:.4g} m ({dz/lam:.4f} lam) "
                f"p={p} V={_V(dx, dz, lam, order):.4e}"
            )
        logger.info(
            "[%s] f=%.3g c0=%g c_min=%g c_max=%g theta=%g deg "
            "x_max=%g eps=%g order=%s | %s",
            method, f, c0, c_min, c_max, theta_deg, x_max, eps, order, result,
        )
    return beta, dx, dz, lam


class GridOptimizerPadeTest(unittest.TestCase):
    """Basic sanity checks for Pade grid optimization."""

    def test_rwp_pade_feasible(self):
        beta, dx, dz, lam = _run(
            f=3e9, c0=3e8, c_min=3e8, c_max=3e8,
            theta_deg=10, x_max=10000, eps=1e-3, method='pade',
        )
        self.assertFalse(math.isnan(beta))
        self.assertGreater(dx, 0.0)
        self.assertGreater(dz, 0.0)
        # beta should be close to k0 in the homogeneous case (within a
        # few percent; candidate grid in the optimizer is discrete)
        k0 = PI2 * 3e9 / 3e8
        self.assertAlmostEqual(beta / k0, 1.0, places=1)

    def test_uwa_pade_feasible(self):
        beta, dx, dz, lam = _run(
            f=500.0, c0=1500.0, c_min=1500.0, c_max=1700.0,
            theta_deg=10, x_max=10000, eps=1e-3, method='pade',
        )
        self.assertFalse(math.isnan(beta))
        self.assertGreater(dx, 0.0)
        self.assertGreater(dz, 0.0)
        # With k_min < k_max (sediment), the optimal beta must lie in
        # [sqrt(k_min^2 - kz_max^2), k_max]
        k0 = PI2 * 500.0 / 1500.0
        k_min = PI2 * 500.0 / 1700.0
        k_max = PI2 * 500.0 / 1500.0
        kz_max = k0 * math.sin(math.radians(10))
        self.assertGreaterEqual(beta, math.sqrt(k_min**2 - kz_max**2) - 1e-9)
        self.assertLessEqual(beta, k_max + 1e-9)


class GridOptimizerRatinterpTest(unittest.TestCase):
    """Basic sanity checks for ratinterp grid optimization."""

    def test_rwp_ratinterp_feasible(self):
        beta, dx, dz, lam = _run(
            f=3e9, c0=3e8, c_min=3e8, c_max=3e8,
            theta_deg=10, x_max=10000, eps=1e-3, method='ratinterp',
        )
        self.assertFalse(math.isnan(beta))
        self.assertGreater(dx, 0.0)
        self.assertGreater(dz, 0.0)

    def test_uwa_ratinterp_feasible(self):
        beta, dx, dz, lam = _run(
            f=500.0, c0=1500.0, c_min=1500.0, c_max=1700.0,
            theta_deg=10, x_max=10000, eps=1e-3, method='ratinterp',
        )
        self.assertFalse(math.isnan(beta))
        self.assertGreater(dx, 0.0)
        self.assertGreater(dz, 0.0)
        # Ratinterp restricts beta to [sqrt(k_min^2 - kz_max^2), k_max]
        # so that the interpolation interval [xi_a, xi_b] brackets zero.
        k0 = PI2 * 500.0 / 1500.0
        k_min = PI2 * 500.0 / 1700.0
        k_max = PI2 * 500.0 / 1500.0
        kz_max = k0 * math.sin(math.radians(10))
        self.assertGreaterEqual(beta, math.sqrt(k_min**2 - kz_max**2) - 1e-9)
        self.assertLessEqual(beta, k_max + 1e-9)


class RatinterpBeatsPadeTest(unittest.TestCase):
    """Ratinterp should yield a smaller V than Pade at high order."""

    ORDER = (7, 8)
    EPS = 1e-3

    def _assert_ri_better(self, f, c0, c_min, c_max, theta_deg, x_max,
                          label):
        _, dxp, dzp, lam = _run(
            f, c0, c_min, c_max, theta_deg, x_max, self.EPS,
            order=self.ORDER, method='pade',
        )
        _, dxr, dzr, _ = _run(
            f, c0, c_min, c_max, theta_deg, x_max, self.EPS,
            order=self.ORDER, method='ratinterp',
        )
        Vp = _V(dxp, dzp, lam, self.ORDER)
        Vr = _V(dxr, dzr, lam, self.ORDER)
        if _VERBOSE:
            logger.info(
                "  -> %s: V(pade)=%.4e V(ratinterp)=%.4e "
                "ratio V_ri/V_pade=%.3f (lower is better)",
                label, Vp, Vr, Vr / Vp,
            )
        self.assertLess(
            Vr, Vp,
            msg=f"{label}: expected V(ratinterp)={Vr:.3e} < V(pade)={Vp:.3e}",
        )

    def test_rwp_theta10(self):
        # Tropospheric RWP, homogeneous (n ~ 1), order [7/8]
        self._assert_ri_better(
            f=3e9, c0=3e8, c_min=3e8, c_max=3e8,
            theta_deg=10, x_max=10000, label='RWP theta=10',
        )

    def test_rwp_theta20(self):
        self._assert_ri_better(
            f=3e9, c0=3e8, c_min=3e8, c_max=3e8,
            theta_deg=20, x_max=1000, label='RWP theta=20',
        )

    def test_rwp_theta30(self):
        self._assert_ri_better(
            f=3e9, c0=3e8, c_min=3e8, c_max=3e8,
            theta_deg=30, x_max=1000, label='RWP theta=30',
        )

    def test_uwa_theta10(self):
        # UWA, inhomogeneous (c_water=1500, c_sed=1700), order [7/8]
        self._assert_ri_better(
            f=500.0, c0=1500.0, c_min=1500.0, c_max=1700.0,
            theta_deg=10, x_max=10000, label='UWA theta=10',
        )

    def test_uwa_theta20(self):
        self._assert_ri_better(
            f=500.0, c0=1500.0, c_min=1500.0, c_max=1700.0,
            theta_deg=20, x_max=10000, label='UWA theta=20',
        )

    def test_uwa_theta30(self):
        self._assert_ri_better(
            f=500.0, c0=1500.0, c_min=1500.0, c_max=1700.0,
            theta_deg=30, x_max=10000, label='UWA theta=30',
        )


if __name__ == "__main__":
    unittest.main()
