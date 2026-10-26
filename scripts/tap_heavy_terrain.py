"""Steep real terrain over impedance ground: staircase versus terrain-following,
compared with the spread caused by environmental uncertainty.

Path: 5.3 km in Glen Etive, Scottish Highlands (56.6064N 4.9986W -> 56.645N 5.05W):
a transmitter on the glen floor (53 m) looks at the mountain flank rising to
750 m; terrain smoothed over 100 m, maximum slope 23.5 degrees.
100 MHz, 30 m mast, 10 degree beam tilted 3 degrees up, H and V polarization,
wet ground, standard atmosphere.

Solvers:

* terrain-following (``rwp_tf_forward_task``, Leontovich condition along the
  terrain normal) at precisions 1e-2 and 1e-3, aperture beam + tilt + slope
  (checked against beam + tilt + 2 slope, ``apcheck``);
* staircase: the package's rational scheme (its optimizer's Pade coefficients,
  beta, dx, dz for the same aperture and precision) marched column by column,
  the field zeroed below the ground node and the Leontovich condition imposed
  at that node with the mirror closure (vertical normal), absorbing layer on
  top.  ``pol=None`` gives the Dirichlet staircase, which is checked against
  the package staircase solver.

Environmental spread: the terrain-following solution with medium-dry ground
instead of wet, and with the refractivity gradient 0.118 +- 0.039 M/m.

Writes ``heavy_terrain.json``, ``heavy_terrain_runs/`` and figures into
``papers/terrain_transform/figures``.  Run::

    python scripts/tap_heavy_terrain.py [tf] [apcheck] [st] [check] [report]
"""
import json
import os
import sys
import time
from copy import deepcopy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.linalg import solve_banded  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jax.numpy as jnp  # noqa: E402

from pywaveprop.environment import get_troposphere_model  # noqa: E402
from pywaveprop.helmholtz_jax import PiecewiseLinearTerrainModel  # noqa: E402
from pywaveprop.helmholtz_ssf_jax import impedance_alpha  # noqa: E402
from pywaveprop.rwp_jax import (GroundMaterial, PiecewiseLinearNProfileModel, RWPComputationalParams,  # noqa: E402
                                RWPGaussSourceModel, TroposphereModel, create_rwp_model, rwp_tf_forward_task)
from pywaveprop.terrain_transform import OrthogonalTerrainMap  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "papers", "terrain_transform", "figures")
RUNS = os.path.join(FIG_DIR, "heavy_terrain_runs")
P1, P2 = (56.6064, -4.9986), (56.645, -5.05)
FREQ_HZ = 100e6
LAM = 3e8 / FREQ_HZ
K0 = 2 * np.pi / LAM
ANT_AGL = 30.0
BEAM_DEG, TILT_DEG = 10.0, 3.0
Z_TOP = 2000.0
SMOOTH = 100.0
TRANSITION = 1400.0
ABSORB_M = 300.0
GROUNDS = {'wet': GroundMaterial(eps=25.0, sigma=0.02), 'medium': GroundMaterial(eps=15.0, sigma=0.005)}
GRADS = {'std': 0.118, 'low': 0.079, 'high': 0.157}
X_CUT_STEP = 25.0
AGLS = (10.0, 50.0)


def setup():
    env0 = get_troposphere_model(P1[0], P1[1], P2[0], P2[1], n_points=800, resolution=0.0005, use_landcover=False)
    x_t = np.asarray(env0.terrain.x_grid_m)
    h_t = np.asarray(env0.terrain.height)
    tm = OrthogonalTerrainMap(x_t, h_t, x_max_m=x_t[-1], transition_m=TRANSITION, smoothing_m=SMOOTH)
    return x_t, h_t, tm


def make_env(terrain, ground, grad):
    z = np.array([0.0, Z_TOP + 400.0])
    N = PiecewiseLinearNProfileModel.create_from_M_profile(z, grad * z)
    return TroposphereModel(N_profile=N, M0=0.0, slope=grad, terrain=terrain,
                            ground_material=GROUNDS[ground] if ground else None)


def make_src(h0, pol):
    return RWPGaussSourceModel(freq_hz=FREQ_HZ, height_m=h0 + ANT_AGL, beam_width_deg=BEAM_DEG,
                               elevation_angle_deg=TILT_DEG, polarization=pol)


def make_params(x_max, aperture, precision, method):
    p = RWPComputationalParams(max_range_m=float(x_max), max_height_m=Z_TOP, max_angle_deg=aperture,
                               dx_m=X_CUT_STEP, dz_m=1.0)
    p.precision = precision
    p.terrain_method = method
    p.terrain_smoothing_m = SMOOTH
    p.terrain_transition_m = TRANSITION
    p.terrain_slope_aperture = False
    return p


def path_loss(amp, x):
    return -20 * np.log10(amp + 1e-30) + 20 * np.log10(4 * np.pi) + 10 * np.log10(x) - 30 * np.log10(LAM)


# ------------------------------------------------------------------ staircase
def staircase(src, env, params, tm, pol_bc, dz_out=0.5):
    """March the package's staircase scheme with the ground condition at the
    ground node of every column.  ``pol_bc``: 'impedance', 'dirichlet' or 'mask'
    (the package's scheme: no ground condition inside a step, field zeroed below
    the ground node after it)."""
    model = create_rwp_model(src, env, params)
    model._prepare_het_arrays()
    beta, dx, dz = float(model.beta), float(model.dx_m), float(model.dz_m)
    het_m = np.asarray(model.het)                     # on 0 .. Z_TOP
    n_m = het_m.size
    n = n_m + int(round(ABSORB_M / dz))
    z = np.arange(n) * dz
    het = np.concatenate([het_m, np.full(n - n_m, het_m[-1])])
    het = het + 0.02j * np.clip((z - Z_TOP) / ABSORB_M, 0, 1) ** 2
    tau2 = (beta * dz) ** 2
    al = float(model.alpha)
    coefs = [(complex(a), complex(b)) for a, b in np.asarray(model.coefs)]

    def tri(c):
        off = al * tau2 + c + al * c * tau2 * het
        diag = tau2 * (1 - 2 * al) - 2 * c + c * tau2 * (1 - 2 * al) * het
        return diag, off
    mats = [(tri(a), tri(b)) for a, b in coefs]
    gamma = None
    if pol_bc == 'impedance':
        gamma = impedance_alpha(K0, env.ground_material.complex_permittivity(FREQ_HZ), src.polarization)
    x_max = params.max_range_m
    nx = int(round(x_max / dx))
    xs = np.arange(nx + 1) * dx
    gidx = np.round(np.asarray(tm.boundary(xs)) / dz).astype(int)
    h0 = float(tm.boundary(np.zeros(1))[0])
    src_local = deepcopy(src)
    src_local.height_m = src.height_m - h0
    u = np.where(z >= h0, np.asarray(src_local.aperture(jnp.asarray(z - h0))), 0.0).astype(complex)
    u[:gidx[0]] = 0.0
    step_out = max(1, int(round(X_CUT_STEP / dx)))
    rows_out = np.arange(0, n_m, max(1, int(round(dz_out / dz))))
    out_x, out = [0.0], [u[rows_out].copy()]
    t0 = time.time()
    ab0 = [np.stack([np.concatenate([[0], ob[1:]]), db, np.concatenate([ob[:-1], [0]])]) for (_, _), (db, ob) in mats]
    for i in range(1, nx + 1):
        g = gidx[i]
        for (da, oa), ab_full in zip([m[0] for m in mats], ab0):
            ob = ab_full[0]
            rhs = da * u
            rhs[1:] += oa[:-1] * u[:-1]
            rhs[:-1] += oa[1:] * u[1:]
            if pol_bc == 'mask':
                # the package's scheme: no condition inside the step, field zeroed below the ground after it
                u = solve_banded((1, 1), ab_full, rhs, check_finite=False)
                continue
            ab = ab_full.copy()
            if gamma is not None:
                rhs[g] = (da[g] + 2 * dz * gamma * oa[g]) * u[g] + 2 * oa[g] * u[g + 1]
                ab[1, g] = ab_full[1, g] + 2 * dz * gamma * ab_full[2, g]   # off-diagonal value of row g
                ab[0, g + 1] = 2 * ab_full[2, g]
                lo = g
            else:
                lo = g + 1
                ab[1, g] = 1.0
                ab[0, g + 1] = 0.0
                rhs[g] = 0.0
            ab[1, :lo] = 1.0
            ab[0, 1:lo + 1] = 0.0
            ab[2, :lo] = 0.0
            rhs[:lo] = 0.0
            u = solve_banded((1, 1), ab, rhs, check_finite=False)
        u[:g] = 0.0
        if pol_bc == 'dirichlet':
            u[g] = 0.0
        if i % step_out == 0:
            out_x.append(xs[i])
            out.append(u[rows_out].copy())
    info = dict(beta=beta, dx=dx, dz=dz, order=[len(coefs)], z_n=n, x_n=nx, time_s=time.time() - t0)
    return np.array(out_x), z[rows_out], np.array(out), info


def staircase_cuts(xo, zo, u, tm, x_cut):
    amp = np.abs(u)
    ground = np.asarray(tm.boundary(x_cut))
    res = {}
    for agl in AGLS:
        vals = np.array([np.interp(g + agl, zo, amp[np.argmin(np.abs(xo - xc))]) for g, xc in zip(ground, x_cut)])
        res[agl] = path_loss(vals, x_cut)
    return res


def tf_cuts(field, tm, x_cut):
    z_fine = np.arange(0.0, Z_TOP, 0.25)
    grid = np.asarray(field.on_regular_grid(x_cut, z_fine))
    ground = np.asarray(tm.boundary(x_cut))
    return {agl: path_loss(np.array([np.interp(g + agl, z_fine, row) for g, row in zip(ground, grid)]), x_cut)
            for agl in AGLS}


def load_runs():
    import pickle
    runs = {}
    if os.path.isdir(RUNS):
        for f in sorted(os.listdir(RUNS)):
            with open(os.path.join(RUNS, f), 'rb') as fh:
                runs[f[:-4]] = pickle.load(fh)
    return runs


def save_run(name, obj):
    import pickle
    os.makedirs(RUNS, exist_ok=True)
    with open(os.path.join(RUNS, name + '.pkl'), 'wb') as fh:
        pickle.dump(obj, fh)


def main():
    parts = sys.argv[1:] or ['tf', 'st', 'check', 'report']
    os.makedirs(FIG_DIR, exist_ok=True)
    x_t, h_t, tm = setup()
    terrain = PiecewiseLinearTerrainModel(x_t, h_t)
    xs_fine = np.linspace(0, x_t[-1], 4000)
    terrain_s = PiecewiseLinearTerrainModel(xs_fine, np.asarray(tm.boundary(xs_fine)))
    slope = tm.max_slope_deg()
    h0 = float(tm.boundary(np.zeros(1))[0])
    # staircase: rays reflected by the slope leave at twice the slope from the horizontal;
    # terrain-following: near the ground they stay within one slope of the coordinate lines
    aperture = BEAM_DEG + abs(TILT_DEG) + 2 * slope
    aperture_tf = BEAM_DEG + abs(TILT_DEG) + slope
    x_max = float(x_t[-1])
    x_cut = np.arange(250.0, x_max, X_CUT_STEP)
    print(f'path {x_max / 1e3:.2f} km, terrain {h_t.min():.0f}..{h_t.max():.0f} m, smoothed slope {slope:.1f} deg, '
          f'aperture {aperture:.1f} deg (terrain-following {aperture_tf:.1f} deg)', flush=True)
    done = load_runs()

    def run_tf(name, pol, ground, grad, prec, ap=None):
        if name in done:
            return
        t = time.time()
        f = rwp_tf_forward_task(make_src(h0, pol), make_env(terrain, ground, grad),
                                make_params(x_max, ap or aperture_tf, prec, 'transform'))
        cuts = tf_cuts(f, tm, x_cut)
        n = min(f.x_phys.shape[0], np.asarray(f.field).shape[0])
        pl_map = np.asarray(f.path_loss().field)[:n]
        obj = dict(cuts=cuts, time_s=time.time() - t, x_phys=np.asarray(f.x_phys)[:n, ::4],
                   z_phys=np.asarray(f.z_phys)[:n, ::4], pl=pl_map[:, ::4])
        save_run(name, obj)
        print(f'{name}: {obj["time_s"]:.0f} s', flush=True)

    def run_st(name, pol, prec, bc='impedance', ground='wet'):
        if name in done:
            return
        env = make_env(terrain_s, ground if bc == 'impedance' else None, GRADS['std'])
        xo, zo, u, info = staircase(make_src(h0, pol), env, make_params(x_max, aperture, prec, 'staircase'), tm, bc)
        cuts = staircase_cuts(xo, zo, u, tm, x_cut)
        X, Z = np.meshgrid(xo, zo, indexing='ij')
        pl = path_loss(np.abs(u), np.maximum(X, 1.0))
        pl[Z < np.asarray(tm.boundary(xo))[:, None]] = np.nan
        save_run(name, dict(cuts=cuts, info=info, x=xo, z=zo[::4], pl=pl[:, ::4]))
        print(f'{name}: {json.dumps(info)}', flush=True)

    if 'tf' in parts:
        for pol in ('H', 'V'):
            run_tf(f'tf_{pol}_wet_std_1e-2', pol, 'wet', GRADS['std'], 1e-2)
            run_tf(f'tf_{pol}_wet_std_1e-3', pol, 'wet', GRADS['std'], 1e-3)
            run_tf(f'tf_{pol}_medium_std_1e-2', pol, 'medium', GRADS['std'], 1e-2)
            run_tf(f'tf_{pol}_wet_low_1e-2', pol, 'wet', GRADS['low'], 1e-2)
            run_tf(f'tf_{pol}_wet_high_1e-2', pol, 'wet', GRADS['high'], 1e-2)
    if 'apcheck' in parts:
        # the reduced terrain-following aperture against the staircase one
        run_tf('tf_V_wet_std_1e-2_apfull', 'V', 'wet', GRADS['std'], 1e-2, ap=aperture)
    if 'st' in parts:
        for pol in ('H', 'V'):
            run_st(f'st_{pol}_wet_1e-2', pol, 1e-2)
            run_st(f'st_{pol}_wet_1e-3', pol, 1e-3)
    if 'check' in parts:
        # Dirichlet staircase of this marcher against the package staircase solver (H, conducting)
        run_st('st_H_pec_1e-2', 'H', 1e-2, bc='dirichlet')
        run_st('st_H_mask_1e-2', 'H', 1e-2, bc='mask')
        run_tf('tf_H_pec_1e-2', 'H', None, GRADS['std'], 1e-2)
        if 'pkg_H_pec_1e-2' not in done:
            env = make_env(terrain_s, None, GRADS['std'])
            src = make_src(h0, 'H')
            model = create_rwp_model(src, env, make_params(x_max, aperture, 1e-2, 'staircase'))
            zg = np.asarray(model.z_computational_grid())
            sl = deepcopy(src)
            sl.height_m = src.height_m - h0
            init = np.where(zg >= h0, np.asarray(sl.aperture(jnp.asarray(zg - h0))), 0.0)
            f = np.asarray(model.compute(jnp.asarray(init)))
            cuts = staircase_cuts(np.asarray(model.x_output_grid()), np.asarray(model.z_output_grid()), f, tm, x_cut)
            save_run('pkg_H_pec_1e-2', dict(cuts=cuts))
    if 'report' in parts:
        report(x_cut, tm, slope, aperture, aperture_tf)


def stats(a, b):
    d = np.abs(a - b)
    d = d[np.isfinite(d)]
    return dict(median_db=float(np.median(d)), p90_db=float(np.percentile(d, 90)), max_db=float(d.max()))


def report(x_cut, tm, slope, aperture, aperture_tf):
    runs = load_runs()
    out = dict(slope_deg=slope, aperture_deg=aperture, aperture_tf_deg=aperture_tf, path=[P1, P2], freq_hz=FREQ_HZ)
    if 'tf_V_wet_std_1e-2_apfull' in runs:
        out['check_tf_aperture_reduced_vs_full'] = {
            f'agl{int(a)}': stats(runs['tf_V_wet_std_1e-2']['cuts'][a], runs['tf_V_wet_std_1e-2_apfull']['cuts'][a])
            for a in AGLS}
    for pol in ('H', 'V'):
        ref = runs[f'tf_{pol}_wet_std_1e-3']['cuts']
        for agl in AGLS:
            row = {}
            pairs = {'tf 1e-2 vs 1e-3': runs[f'tf_{pol}_wet_std_1e-2']['cuts'],
                     'staircase 1e-2 vs tf': runs.get(f'st_{pol}_wet_1e-2', {}).get('cuts'),
                     'staircase 1e-3 vs tf': runs.get(f'st_{pol}_wet_1e-3', {}).get('cuts'),
                     'ground medium vs wet': runs[f'tf_{pol}_medium_std_1e-2']['cuts'],
                     'dM/dz 0.079 vs 0.118': runs[f'tf_{pol}_wet_low_1e-2']['cuts'],
                     'dM/dz 0.157 vs 0.118': runs[f'tf_{pol}_wet_high_1e-2']['cuts']}
            for k, c in pairs.items():
                if c is not None:
                    row[k] = stats(c[agl], ref[agl])
            if 'st_1e-2' not in row and f'st_{pol}_wet_1e-3' in runs and f'st_{pol}_wet_1e-2' in runs:
                row['staircase 1e-2 vs 1e-3'] = stats(runs[f'st_{pol}_wet_1e-2']['cuts'][agl],
                                                      runs[f'st_{pol}_wet_1e-3']['cuts'][agl])
            out[f'{pol}_agl{int(agl)}'] = row
    if 'st_H_pec_1e-2' in runs and 'pkg_H_pec_1e-2' in runs:
        out['check_dirichlet_marcher_vs_package'] = {
            f'agl{int(a)}': stats(runs['st_H_pec_1e-2']['cuts'][a], runs['pkg_H_pec_1e-2']['cuts'][a]) for a in AGLS}
    if 'tf_H_pec_1e-2' in runs:
        out['pec_H_vs_tf'] = {
            f'{k} agl{int(a)}': stats(runs[n]['cuts'][a], runs['tf_H_pec_1e-2']['cuts'][a])
            for k, n in (('dirichlet staircase', 'st_H_pec_1e-2'), ('package staircase', 'pkg_H_pec_1e-2'))
            if n in runs for a in AGLS}
    if 'st_H_mask_1e-2' in runs and 'pkg_H_pec_1e-2' in runs:
        out['check_mask_marcher_vs_package'] = {
            f'agl{int(a)}': stats(runs['st_H_mask_1e-2']['cuts'][a], runs['pkg_H_pec_1e-2']['cuts'][a]) for a in AGLS}
    for k, v in runs.items():
        if 'time_s' in v:
            out[f'time_{k}'] = v['time_s']
        if 'info' in v:
            out[f'grid_{k}'] = v['info']
    out['map_bands'] = map_report(runs, tm)
    with open(os.path.join(FIG_DIR, 'heavy_terrain.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    plot(runs, x_cut, tm)


BANDS = ((0.0, 100.0), (100.0, 500.0), (500.0, 2000.0))


def tf_on_staircase_grid(tf, st):
    """Terrain-following path loss interpolated onto the staircase grid (x, z)."""
    from scipy.interpolate import griddata
    X, Z = np.meshgrid(st['x'], st['z'], indexing='ij')
    pts = np.c_[tf['x_phys'].ravel(), tf['z_phys'].ravel()]
    return X, Z, griddata(pts, tf['pl'].ravel(), (X, Z), method='linear')


def band_stats(d, agl, mask):
    """Median / p90 of |d| and mean of d in the height bands above ground."""
    out = {}
    for lo, hi in BANDS:
        m = mask & (agl >= lo) & (agl < hi) & np.isfinite(d)
        out[f'{lo:.0f}-{hi:.0f} m'] = dict(median_db=float(np.median(np.abs(d[m]))),
                                           p90_db=float(np.percentile(np.abs(d[m]), 90)),
                                           mean_db=float(d[m].mean()))
    return out


def region(x, z, pl_ref, tm):
    """Illuminated region (path loss below 130 dB) away from the ends of the path."""
    agl = z - np.asarray(tm.boundary(x.ravel())).reshape(x.shape)
    return agl, (x > 500.0) & (x < 5200.0) & (z < Z_TOP - 100.0) & (pl_ref < 130.0)


def map_report(runs, tm):
    """Staircase and environmental differences over the whole map, by height band."""
    out = {}
    for pol in ('H', 'V'):
        ref = runs[f'tf_{pol}_wet_std_1e-3']
        for prec in ('1e-2', '1e-3'):
            st = runs[f'st_{pol}_wet_{prec}']
            X, Z, pt = tf_on_staircase_grid(ref, st)
            agl, m = region(X, Z, pt, tm)
            out[f'{pol} staircase {prec} - tf'] = band_stats(st['pl'] - pt, agl, m)
        n = len(ref['pl'])
        agl, m = region(ref['x_phys'], ref['z_phys'], ref['pl'], tm)
        for k, name in (('medium_std', 'ground medium - wet'), ('wet_low', 'dM/dz 0.079 - 0.118'),
                        ('wet_high', 'dM/dz 0.157 - 0.118')):
            q = runs[f'tf_{pol}_{k}_1e-2']['pl']
            k_n = min(n, len(q))
            out[f'{pol} {name}'] = band_stats(q[:k_n] - ref['pl'][:k_n], agl[:k_n], m[:k_n])
    return out


def plot(runs, x_cut, tm):
    xf = np.linspace(0, x_cut[-1] + X_CUT_STEP, 2000)
    gf = np.asarray(tm.boundary(xf))
    fig = plt.figure(figsize=(11, 9.4), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.3, 1.3, 1])
    tf, st = runs['tf_V_wet_std_1e-3'], runs['st_V_wet_1e-3']
    X, Z, pt = tf_on_staircase_grid(tf, st)
    below = Z < np.asarray(tm.boundary(X[:, 0]))[:, None]
    # differences only where the reference path loss is below 130 dB (as in map_report)
    q = runs['tf_V_medium_std_1e-2']['pl']
    n = min(len(q), len(tf['pl']))
    panels = [(tf['x_phys'], tf['z_phys'], tf['pl'], 'terrain-following: path loss', 0),
              (X, Z, st['pl'], 'staircase: path loss', 0),
              (X, Z, np.where(below | ~(pt < 130.0), np.nan, st['pl'] - pt), 'staircase $-$ terrain-following', 1),
              (tf['x_phys'][:n], tf['z_phys'][:n], np.where(tf['pl'][:n] < 130.0, q[:n] - tf['pl'][:n], np.nan),
               'terrain-following: medium $-$ wet ground', 1)]
    map_axes = []
    for k, (x, z, f, title, kind) in enumerate(panels):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        map_axes.append(ax)
        if kind == 0:
            pc = ax.pcolormesh(x / 1e3, z, f, cmap='jet_r', vmin=60, vmax=140, shading='nearest', rasterized=True)
        else:
            pc = ax.pcolormesh(x / 1e3, z, f, cmap='RdBu', vmin=-20, vmax=20, shading='nearest', rasterized=True)
        ax.fill_between(xf / 1e3, gf, 0, color='saddlebrown')
        ax.set_ylim(0, 1800)
        ax.set_title(f'{title}, V polarization', fontsize=9)
        ax.set_ylabel('height ASL, m')
        if k == 1:
            fig.colorbar(pc, ax=map_axes[:2], label='path loss, dB', shrink=0.9)
        if k == 3:
            fig.colorbar(pc, ax=map_axes[2:4], label='difference, dB', shrink=0.9)
    for j, pol in enumerate(('H', 'V')):
        agl = 50.0
        ax = fig.add_subplot(gs[2, j])
        cases = ('medium_std', 'wet_low', 'wet_high', 'wet_std')
        lo = np.minimum.reduce([runs[f'tf_{pol}_{k}_1e-2']['cuts'][agl] for k in cases])
        hi = np.maximum.reduce([runs[f'tf_{pol}_{k}_1e-2']['cuts'][agl] for k in cases])
        ax.fill_between(x_cut / 1e3, lo, hi, color='0.75', label='environmental spread (TF)')
        ax.plot(x_cut / 1e3, runs[f'tf_{pol}_wet_std_1e-3']['cuts'][agl], 'C3', lw=1.3, label='terrain-following')
        ax.plot(x_cut / 1e3, runs[f'st_{pol}_wet_1e-3']['cuts'][agl], 'C0', lw=0.9, label='staircase')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{pol} polarization, {agl:.0f} m above ground', fontsize=9)
        ax.set_ylabel('path loss, dB')
        ax.set_xlabel('range, km')
        if j == 0:
            ax.legend(fontsize=7)
    fig.savefig(os.path.join(FIG_DIR, 'heavy_terrain.png'), dpi=140)
    plt.close(fig)


if __name__ == '__main__':
    main()
