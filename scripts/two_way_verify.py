"""Verification of the two-way (Bremmer) terrain-following propagator.

Programme of ``papers/tap2026/two_way_helmholtz_tf.tex``:

1. Flat constant medium with a PEC wall: a plane wave at 30 deg to the wall.
   (a) invariance: marching the standing wave changes nothing (R = 0, u- = 0);
   (b) reflection generation: marching the incident wave produces the standing
       wave (the wall reflection the one-way march misses).

2. The 30-deg conducting slope with the beam of the exact slope test: the
   reflection happens inside the terrain-following domain and must be produced
   by the coupling; the exact CSP + image field is the reference.

Run::

    python scripts/two_way_verify.py flat
    python scripts/two_way_verify.py slope
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

FREQ = 3e8
K0 = 2 * np.pi * FREQ / 3e8


def flat_tests(angle_deg=30.0, x_size=200.0, z_size=100.0, deta=0.25, dxi=2.5):
    from pywaveprop.helmholtz_common import HelmholtzMeshParams2D
    from pywaveprop.helmholtz_two_way_jax import TwoWayTerrainFollowingPropagator
    from pywaveprop.terrain_transform import OrthogonalTerrainMap

    kx = K0 * np.cos(np.radians(angle_deg))
    kz = K0 * np.sin(np.radians(angle_deg))
    xb = np.linspace(-500.0, 700.0, 3001)
    tm = OrthogonalTerrainMap(xb, np.zeros_like(xb), x_max_m=x_size, transition_m=50.0,
                              z_ref_m=0.0, mirror_start=True, dx_fine_m=0.5, pad_m=500.0)
    med = __import__('pywaveprop.helmholtz_tf_jax', fromlist=['TFMedium']).TFMedium(
        k2_above=lambda x, z: np.full(np.shape(z), K0 ** 2, dtype=complex))
    mp = HelmholtzMeshParams2D(x_size_m=x_size, z_size_m=z_size, dx_output_m=dxi, dz_output_m=deta)
    kz_max = 1.2 * kz
    prop = TwoWayTerrainFollowingPropagator.create(FREQ, tm, med, kz_max, (K0, K0), 1e-2, mp,
                                                   rational_approx_order=(7, 8), upper_bc='dirichlet')
    eta = prop.initial_points()[1]                       # computational eta nodes
    standing = 2j * np.sin(kz * eta)                     # exact wall-satisfying field
    incident = np.exp(1j * kz * eta)                     # incident wave only
    # the exact standing wave in the solver's reduced form: e^{i (kx - beta) xi}
    exact_standing = (2j * np.sin(kz * eta)[None, :]
                      * np.exp(1j * (kx - prop.beta) * np.arange(prop.x_n) * float(prop.dx_m))[:, None])

    out = {}
    for name, init in (('invariance (standing wave input)', standing),
                       ('reflection generation (incident input)', incident)):
        psi_out, tp, tm_state, psi_full = prop.compute(init, n_bremmer=2)
        err = np.abs(np.asarray(psi_full)[:, 2:-2] - exact_standing[:, 2:-2]).max() / 2.0
        uminus_frac = float(np.abs(np.asarray(tm_state)[:, 2:-2]).max() / 2.0)
        out[name] = dict(err_rel=float(err), uminus_over_max=uminus_frac)
        print(f'{name}: err = {err:.3e},  max|u-|/max|psi| = {uminus_frac:.3e}', flush=True)
    return out


def slope_test(x_min=-60.0, x_max=150.0, layer=100.0, z_size=400.0, deta=0.25, dxi=5.0,
               n_bremmer=3, theta_s=30.0):
    """The exact slope test with the reflection inside the domain: the two-way
    coupling must produce the reflected beam that the one-way march misses."""
    from scipy.special import hankel1
    from pywaveprop.helmholtz_common import HelmholtzMeshParams2D
    from pywaveprop.helmholtz_two_way_jax import TwoWayTerrainFollowingPropagator
    from pywaveprop.helmholtz_tf_jax import TFMedium
    from pywaveprop.terrain_transform import OrthogonalTerrainMap

    s = np.tan(np.radians(theta_s))
    xs, zs, b = -50.0, 30.0, 100.0
    alpha = np.radians(-10.0)
    n = np.array([-s, 1.0]) / np.hypot(s, 1.0)
    refl = np.eye(2) - 2 * np.outer(n, n)
    ri = refl @ np.array([xs, zs])
    ei = refl @ np.array([np.cos(alpha), np.sin(alpha)])

    def csp(x, z, sx, sz, ang):
        r = np.sqrt((x - sx - 1j * b * np.cos(ang)) ** 2 + (z - sz - 1j * b * np.sin(ang)) ** 2)
        return 1j / 4 * hankel1(0, K0 * r)

    def exact(x, z):
        f = csp(x, z, xs, zs, alpha) - csp(x, z, ri[0], ri[1], np.arctan2(ei[1], ei[0]))
        norm = np.abs(csp(np.array([0.0]), np.array([zs]), xs, zs, alpha)
                      - csp(np.array([0.0]), np.array([zs]), ri[0], ri[1],
                            np.arctan2(ei[1], ei[0])))[0]
        return f / norm

    relief = s * 150.0                     # z_ref over the verification range
    z_ref = relief
    xb = np.linspace(-1500.0, 650.0, 3001)
    tmap = OrthogonalTerrainMap(xb, s * xb, x_max_m=x_max, transition_m=layer, z_ref_m=z_ref,
                                smoothing_m=0.0, mirror_start=False, dx_fine_m=0.5,
                                pad_m=500.0, x_min_m=x_min)
    med = __import__('pywaveprop.helmholtz_tf_jax', fromlist=['TFMedium']).TFMedium(
        k2_above=lambda x, z: np.full(np.shape(z), K0 ** 2, dtype=complex))
    mp = HelmholtzMeshParams2D(x_size_m=x_max - x_min, z_size_m=z_size, dx_output_m=dxi,
                               dz_output_m=deta)
    kz_max = min(1.15 * K0 * np.sin(np.radians(2 * theta_s + 10.0)), 0.97 * K0)
    prop = TwoWayTerrainFollowingPropagator.create(FREQ, tmap, med, kz_max, (K0, K0), 1e-2, mp,
                                                   rational_approx_order=(7, 8), upper_bc='dirichlet',
                                                   flux_order=2)
    x0, z0 = prop.initial_points()
    psi_inc = exact(x0, z0)                # incident beam only (pre-reflection)
    psi_out, tp, tm, psi_full = prop.compute(psi_inc, n_bremmer=n_bremmer)
    psi_out = np.asarray(psi_out)
    xo, zo = prop.output_metrics()

    # exact total field on the output nodes; compare above the ground
    xe, ze = xo, zo
    ue = exact(xe, ze)
    inside = ze > s * xe + 2.0
    err_twoway = np.abs(psi_out - ue)[inside].max() / np.abs(ue[inside]).max()
    print('two-way  (n_bremmer=%d): err = %.3f' % (n_bremmer, err_twoway), flush=True)

    # one-way reference on the identical domain
    import jax.numpy as jnp
    from pywaveprop.helmholtz_tf_jax import TerrainFollowingHelmholtzPropagator
    prop1 = TerrainFollowingHelmholtzPropagator.create(FREQ, tmap, med, kz_max, (K0, K0), 1e-2, mp,
                                                       rational_approx_order=(7, 8), upper_bc='dirichlet')
    x01, z01 = prop1.initial_points()
    u1 = np.asarray(prop1.compute(jnp.asarray(psi_inc)))[:psi_out.shape[0], :psi_out.shape[1]]
    xo1, zo1 = prop1.output_metrics()
    ue1 = exact(np.asarray(xo1)[:psi_out.shape[0], :psi_out.shape[1]],
                np.asarray(zo1)[:psi_out.shape[0], :psi_out.shape[1]])
    inside1 = zo1[:psi_out.shape[0], :psi_out.shape[1]] > s * xo1[:psi_out.shape[0], :psi_out.shape[1]] + 2.0
    err_oneway = np.abs(u1 - ue1)[inside1].max() / np.abs(ue1[inside1]).max()
    print('one-way  (same domain): err = %.3f' % err_oneway, flush=True)
    return dict(err_two_way=float(err_twoway), err_one_way=float(err_oneway),
                n_bremmer=n_bremmer, x_min=x_min, x_max=x_max, layer=layer)


if __name__ == '__main__':
    part = sys.argv[1] if len(sys.argv) > 1 else 'flat'
    if part == 'flat':
        res = flat_tests()
        with open(os.path.join(os.path.dirname(__file__), '..', 'papers', 'tap2026',
                               'two_way_flat_verification.json'), 'w') as f:
            json.dump(res, f, indent=1)
    if part == 'slope':
        res = slope_test()
        with open(os.path.join(os.path.dirname(__file__), '..', 'papers', 'tap2026',
                               'two_way_slope_verification.json'), 'w') as f:
            json.dump(res, f, indent=1)
        print(json.dumps(res, indent=1))
