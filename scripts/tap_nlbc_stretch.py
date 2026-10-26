"""Upper NLBC on a stretched vertical grid (``terrain_upper_stretch``), Glen
Etive case of ``tap_heavy_terrain.py`` (V, wet ground, standard gradient,
precision 1e-2).

Runs
* ``uniform``: uniform grid, NLBC (reference);
* ``tail``: grid coarsened above the distorted layer to ``s_top``; the top lies
  in the uniform tail, the NLBC is the uniform one with step ``s_top deta``;
* ``ramp``: a long ramp, so that the top lies inside it and the exterior is
  stretched: NLBC from the backward recursion over the exterior rows;
* ``ramp_tall``: the same grid on a domain about 650 m taller (the reference for
  ``ramp``: two transparent conditions give the same discrete solution);
* ``ramp_naive``: as ``ramp``, with the uniform-grid NLBC for the local step
  ``s_N deta`` at the top (ignores the stretching of the exterior).

Case ``beam``: a beam aimed at the top (100 MHz, 10 degrees wide, tilted
15 degrees up, over a 100 m hill), so that most of the energy leaves through
the upper boundary inside the ramp.

    python scripts/tap_nlbc_stretch.py [glen] [beam]
"""
import json
import os
import sys
import time
from copy import deepcopy

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import tap_heavy_terrain as ht  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from pywaveprop import helmholtz_tf_jax  # noqa: E402
from pywaveprop.helmholtz_jax import PiecewiseLinearTerrainModel  # noqa: E402
from pywaveprop.rwp_jax import create_rwp_tf_model  # noqa: E402
from pywaveprop.rwp_field import TerrainFollowingRWPField  # noqa: E402

RAMP_START, RAMP_LEN = 1500.0, 1500.0          # eta / physical metres


NAIVE = {'on': False}
_ORIG = helmholtz_tf_jax.TerrainFollowingHelmholtzPropagator._calc_upper_nlbc_stretched


def _maybe_naive(self, n_rows=None):
    """With ``NAIVE['on']``: the uniform-grid NLBC for the local step ``s_N deta``
    at the top node (ignores the stretching of the exterior)."""
    if not NAIVE['on']:
        return _ORIG(self, n_rows)
    _, s_n = helmholtz_tf_jax.layer_stretch((self.z_n - 1) * self.dz_m, *self.tables['stretch'])
    d = self.dz_m
    self.dz_m = d * float(s_n)
    het = self.het
    self.het = self.tables['het_top']
    gamma = (self.het[-1] - self.het[-2]) / self.dz_m
    try:
        return self._calc_upper_nlbc(self.het[-1] - gamma * (self.z_n - 1) * self.dz_m, gamma)
    finally:
        self.dz_m, self.het = d, het


helmholtz_tf_jax.TerrainFollowingHelmholtzPropagator._calc_upper_nlbc_stretched = _maybe_naive


def main():
    cases = sys.argv[1:] or ['glen', 'beam', 'beam_strong']
    res = {}
    if 'glen' in cases:
        res['glen'] = glen()
    if 'beam' in cases:
        res['beam'] = beam()
    if 'beam_strong' in cases:
        # strong stretching s_top = 4 (not resolved accuracy-wise: a test of the boundary condition only)
        res['beam_strong'] = beam(s_top=4.0, ramp_len=800.0)
    print(json.dumps(res, indent=1))
    with open(os.path.join(ht.FIG_DIR, 'nlbc_stretch.json'), 'w') as fh:
        json.dump(res, fh, indent=1)


def glen():
    x_t, h_t, tm = ht.setup()
    terrain = PiecewiseLinearTerrainModel(x_t, h_t)
    slope = tm.max_slope_deg()
    h0 = float(tm.boundary(np.zeros(1))[0])
    ap = ht.BEAM_DEG + abs(ht.TILT_DEG) + slope
    x_max = float(x_t[-1])
    env = ht.make_env(terrain, 'wet', ht.GRADS['std'])
    src = ht.make_src(h0, 'V')
    xs = np.arange(500.0, x_max - 100, 50.0)
    zs = np.arange(0.0, 2600.0, 2.0)
    return experiment(env, src, tm, x_max, ap, xs, zs, h0, ht.Z_TOP, RAMP_START, RAMP_LEN, 1300.0, 1e-2)


def beam(s_top=None, ramp_len=1000.0):
    from pywaveprop.terrain_transform import OrthogonalTerrainMap
    x = np.linspace(0, 4000.0, 801)
    h = 100.0 * np.exp(-0.5 * ((x - 2000.0) / 300.0) ** 2)
    tm = OrthogonalTerrainMap(x, h, x_max_m=x[-1], transition_m=300.0, smoothing_m=50.0)
    z = np.array([0.0, 900.0])          # 1.1 x profile height stays below the domain top
    from pywaveprop.rwp_jax import GroundMaterial, PiecewiseLinearNProfileModel, RWPGaussSourceModel, TroposphereModel
    env = TroposphereModel(N_profile=PiecewiseLinearNProfileModel.create_from_M_profile(z, 0.118 * z), M0=0.0,
                           slope=0.118, terrain=PiecewiseLinearTerrainModel(x, h),
                           ground_material=GroundMaterial(eps=25.0, sigma=0.02))
    src = RWPGaussSourceModel(freq_hz=ht.FREQ_HZ, height_m=30.0, beam_width_deg=10.0, elevation_angle_deg=15.0,
                              polarization='V')
    ap = 10.0 + 15.0 + tm.max_slope_deg()
    xs = np.arange(100.0, 3900.0, 25.0)
    zs = np.arange(0.0, 1000.0, 1.0)
    # the top (about 1000 m) lies inside a ramp from 450 m to 1450 m
    return experiment(env, src, tm, float(x[-1]), ap, xs, zs, 0.0, 1000.0, 450.0, ramp_len, 600.0, 1e-2,
                      transition=300.0, smoothing=50.0, s_top=s_top)


def experiment(env, src, tm, x_max, ap, xs, zs, h0, z_top, ramp_start, ramp_len, extra, prec,
               transition=None, smoothing=None, s_top=None):
    def run(name, stretch, start=None, ramp=None, extra=0.0, use_naive=False):
        p = ht.make_params(x_max, ap, prec, 'transform')
        p.max_height_m = z_top + extra
        if transition:
            p.terrain_transition_m = transition
            p.terrain_smoothing_m = smoothing
        p.terrain_upper_stretch = stretch
        p.terrain_upper_stretch_start_m = start
        p.terrain_upper_stretch_ramp_m = ramp
        p.terrain_upper_stretch_s_top = s_top if stretch else None
        NAIVE['on'] = use_naive
        t = time.time()
        m = create_rwp_tf_model(src, env, p)
        x0, z0 = m.initial_points()
        s = deepcopy(src)
        s.height_m -= h0
        u = m.compute(s.aperture(jnp.asarray(z0 - h0)))
        x_phys, z_phys = m.output_metrics()
        f = TerrainFollowingRWPField(x_grid=m.x_output_grid(), z_grid=z_phys[0], freq_hz=src.freq_hz, field=u,
                                     x_phys=x_phys, z_phys=z_phys, terrain_x=m.terrain_map.x_fine,
                                     terrain_z=m.terrain_map.boundary_fine)
        amp = f.on_regular_grid(xs, zs)
        X, _ = np.meshgrid(xs, zs, indexing='ij')
        pl = ht.path_loss(amp, X)
        dt = time.time() - t
        top = float(z_phys[0, -1])
        print(f'{name}: z_n {m.z_n}, physical top {top:.0f} m, {dt:.0f} s', flush=True)
        return dict(pl=pl, z_n=m.z_n, top=top, time_s=dt)

    R = {}
    R['uniform'] = run('uniform', False)
    R['tail'] = run('tail', True)
    R['ramp'] = run('ramp', True, ramp_start, ramp_len)
    R['ramp_tall'] = run('ramp_tall', True, ramp_start, ramp_len, extra=extra)
    R['ramp_naive'] = run('ramp_naive', True, ramp_start, ramp_len, use_naive=True)

    ground = np.asarray(tm.boundary(xs))[:, None]
    Z = np.broadcast_to(zs[None, :], R['uniform']['pl'].shape)

    def diff(a, b, z_hi):
        d = R[a]['pl'] - R[b]['pl']
        m = np.isfinite(d) & (Z > ground) & (Z < z_hi) & (R[b]['pl'] < 130)
        lo = m & (Z < ground + 100)
        return dict(median_db=float(np.median(np.abs(d[m]))), max_db=float(np.abs(d[m]).max()),
                    median_lowest_100m_db=float(np.median(np.abs(d[lo]))))

    top = min(R[k]['top'] for k in R) - 20
    res = {k: dict(z_n=R[k]['z_n'], top_m=R[k]['top'], time_s=R[k]['time_s']) for k in R}
    res['tail - uniform'] = diff('tail', 'uniform', top)
    res['ramp - ramp_tall'] = diff('ramp', 'ramp_tall', top)
    res['ramp_naive - ramp_tall'] = diff('ramp_naive', 'ramp_tall', top)
    res['ramp - uniform'] = diff('ramp', 'uniform', top)
    return res


if __name__ == '__main__':
    main()
