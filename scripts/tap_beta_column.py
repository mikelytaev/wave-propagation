"""Range-dependent reference wavenumber for the terrain-following propagator
(``terrain_beta_mode='column'``) on the Glen Etive case of
``tap_heavy_terrain.py`` (V, wet ground, standard gradient).

Runs: global beta with the absorbing upper layer on a uniform grid (isolates
the change of the upper condition), one beta per class of columns with the
uniform layer, and with the stretched layer (``terrain_absorb_stretch``), at
precision 1e-2 (and 1e-3 with ``fine``); compared with the stored NLBC runs
``tf_V_wet_std_1e-2`` / ``1e-3``.

    python scripts/tap_beta_column.py [fine]
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import tap_heavy_terrain as ht  # noqa: E402
from pywaveprop.helmholtz_jax import PiecewiseLinearTerrainModel  # noqa: E402
from pywaveprop.rwp_jax import rwp_tf_forward_task  # noqa: E402

ABSORB = 300.0


def main():
    fine = 'fine' in sys.argv[1:]
    x_t, h_t, tm = ht.setup()
    terrain = PiecewiseLinearTerrainModel(x_t, h_t)
    slope = tm.max_slope_deg()
    h0 = float(tm.boundary(np.zeros(1))[0])
    ap = ht.BEAM_DEG + abs(ht.TILT_DEG) + slope
    x_max = float(x_t[-1])
    x_cut = np.arange(250.0, x_max, ht.X_CUT_STEP)
    runs = ht.load_runs()

    def run(name, prec, mode, absorb, stretch=False):
        if name in runs:
            return runs[name]
        p = ht.make_params(x_max, ap, prec, 'transform')
        p.terrain_beta_mode = mode
        p.terrain_absorb_m = absorb
        p.terrain_absorb_stretch = stretch
        t = time.time()
        f = rwp_tf_forward_task(ht.make_src(h0, 'V'), ht.make_env(terrain, 'wet', ht.GRADS['std']), p)
        cuts = ht.tf_cuts(f, tm, x_cut)
        n = min(f.x_phys.shape[0], np.asarray(f.field).shape[0])
        obj = dict(cuts=cuts, time_s=time.time() - t, x_phys=np.asarray(f.x_phys)[:n, ::4],
                   z_phys=np.asarray(f.z_phys)[:n, ::4], pl=np.asarray(f.path_loss().field)[:n, ::4])
        ht.save_run(name, obj)
        runs[name] = obj
        print(f'{name}: {obj["time_s"]:.0f} s', flush=True)
        return obj

    precs = ['1e-2'] + (['1e-3'] if fine else [])
    for prec in precs:
        run(f'tfabs_V_wet_std_{prec}', float(prec), 'global', ABSORB)
        run(f'tfcol_V_wet_std_{prec}', float(prec), 'column', ABSORB)
        run(f'tfcols_V_wet_std_{prec}', float(prec), 'column', ABSORB, stretch=True)

    def compare(a, b):
        ra, rb = runs[a], runs[b]
        out = {f'cut {agl:.0f} m': ht.stats(ra['cuts'][agl], rb['cuts'][agl]) for agl in ht.AGLS}
        n = min(len(ra['pl']), len(rb['pl']))
        # rows below the absorbing layer (the same eta grid in all runs)
        k = int(min(np.sum(ra['z_phys'][-1] < ht.Z_TOP), np.sum(rb['z_phys'][-1] < ht.Z_TOP)))
        assert np.allclose(ra['z_phys'][:n, :k], rb['z_phys'][:n, :k], atol=1e-6)
        agl, m = ht.region(rb['x_phys'][:n, :k], rb['z_phys'][:n, :k], rb['pl'][:n, :k], tm)
        out['map'] = ht.band_stats(ra['pl'][:n, :k] - rb['pl'][:n, :k], agl, m)
        return out

    res = {}
    for prec in precs:
        ref = f'tf_V_wet_std_{prec}'
        res[f'abs - nlbc {prec}'] = compare(f'tfabs_V_wet_std_{prec}', ref)
        res[f'column - nlbc {prec}'] = compare(f'tfcol_V_wet_std_{prec}', ref)
        res[f'column - abs {prec}'] = compare(f'tfcol_V_wet_std_{prec}', f'tfabs_V_wet_std_{prec}')
        res[f'column stretched - nlbc {prec}'] = compare(f'tfcols_V_wet_std_{prec}', ref)
        res[f'column stretched - uniform {prec}'] = compare(f'tfcols_V_wet_std_{prec}', f'tfcol_V_wet_std_{prec}')
    res['nlbc 1e-2 - 1e-3'] = compare('tf_V_wet_std_1e-2', 'tf_V_wet_std_1e-3')
    res['times_s'] = {k: runs[k].get('time_s') for k in runs if k.startswith(('tf_V_wet_std', 'tfabs', 'tfcol', 'tfcols'))}
    print(json.dumps(res, indent=1))
    with open(os.path.join(ht.FIG_DIR, 'beta_column.json'), 'w') as fh:
        json.dump(res, fh, indent=1)


if __name__ == '__main__':
    main()
