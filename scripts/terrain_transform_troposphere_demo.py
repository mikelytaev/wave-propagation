"""Tropospheric propagation over real terrain in terrain-following coordinates.

A 28 km path across the Scottish Highlands (57.2163N 4.9214W -> 57.40N 4.60W),
starting on the top of a 666 m ridge,
SRTM-type elevation tiles (``pywaveprop.environment``, ~50 m resolution) and
the modified refractivity of the latest GFS analysis averaged along the path.
A 300 MHz horizontally polarised antenna on a 30 m mast (4 degree beam tilted 1 degree
down); the ground is perfectly conducting.

The terrain-following solver is compared with the staircase solver of the
package on the same aperture (source aperture plus twice the maximum terrain
slope) at two precisions: the terrain-following result does not change, the
staircase result depends on its vertical step.

All data are cached under ``~/.cache/pywaveprop``. Run::

    python scripts/terrain_transform_troposphere_demo.py
"""
import json
import os
import sys
import time
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jax.numpy as jnp  # noqa: E402

from pywaveprop.environment import BBox, fetch_refractivity_cube, get_troposphere_model  # noqa: E402
from pywaveprop.environment.nwp import refractivity_transect_from_cube  # noqa: E402
from pywaveprop.rwp_jax import (PiecewiseLinearNProfileModel, RWPComputationalParams,  # noqa: E402
                                RWPGaussSourceModel, TroposphereModel, create_rwp_model, rwp_tf_forward_task)
from pywaveprop.terrain_transform import OrthogonalTerrainMap  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "papers", "terrain_transform", "figures")
P1, P2 = (57.2163, -4.9214), (57.40, -4.60)
FREQ_HZ = 300e6
ANT_AGL = 30.0
Z_TOP = 1700.0
BEAM_DEG = 4.0
TILT_DEG = -1.0


def gfs_profile(length_m):
    box = BBox(min(P1[1], P2[1]) - 0.6, max(P1[1], P2[1]) + 0.6, min(P1[0], P2[0]) - 0.3, max(P1[0], P2[0]) + 0.3)
    cube = fetch_refractivity_cube(box, top_height_m=3000.0, dz_m=20.0, keep_pressure_levels=False)
    tr = refractivity_transect_from_cube(cube, P1, P2, z_top=Z_TOP + 200.0)
    z = np.asarray(tr.z_m, dtype=float)
    M = np.nanmean(np.asarray(tr.M, dtype=float), axis=0)
    ok = np.isfinite(M)
    # below the lowest valid GFS level (inside the ground on most of the path): standard gradient
    j0 = np.argmax(ok)
    M[:j0] = M[j0] - 0.118 * (z[j0] - z[:j0])
    return z, M, str(cube.attrs.get('cycle', ''))


def staircase_run(src, env, params, h0):
    model = create_rwp_model(src, env, params)
    z = np.asarray(model.z_computational_grid())
    src_local = deepcopy(src)
    src_local.height_m = src.height_m - h0
    init = np.where(z >= h0, np.asarray(src_local.aperture(jnp.asarray(z - h0))), 0.0)
    f = np.asarray(model.compute(jnp.asarray(init)))
    return np.asarray(model.x_output_grid()), np.asarray(model.z_output_grid()), f, model


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    env0 = get_troposphere_model(P1[0], P1[1], P2[0], P2[1], n_points=800, resolution=0.0005, use_landcover=False)
    x_t = np.asarray(env0.terrain.x_grid_m)
    h_t = np.asarray(env0.terrain.height)
    z_m, M, cycle = gfs_profile(x_t[-1])
    sel = z_m <= Z_TOP + 100
    slope_top = (M[sel][-1] - M[sel][-2]) / (z_m[sel][-1] - z_m[sel][-2])
    env = TroposphereModel(N_profile=PiecewiseLinearNProfileModel.create_from_M_profile(z_m[sel], M[sel]),
                           M0=0.0, slope=float(slope_top), terrain=env0.terrain, ground_material=None)
    smooth = 100.0
    tm_probe = OrthogonalTerrainMap(x_t, h_t, x_max_m=x_t[-1], transition_m=1000.0, smoothing_m=smooth)
    slope = tm_probe.max_slope_deg()
    h0 = float(tm_probe.boundary(np.zeros(1))[0])
    aperture = BEAM_DEG + abs(TILT_DEG) + 2 * slope
    print(f'path {x_t[-1] / 1e3:.1f} km, terrain {h_t.min():.0f}..{h_t.max():.0f} m, smoothed slope {slope:.1f} deg, '
          f'aperture {aperture:.1f} deg, GFS cycle {cycle}')
    src = RWPGaussSourceModel(freq_hz=FREQ_HZ, height_m=h0 + ANT_AGL, beam_width_deg=BEAM_DEG,
                              elevation_angle_deg=TILT_DEG, polarization='H')

    def params(precision, method):
        p = RWPComputationalParams(max_range_m=float(x_t[-1]), max_height_m=Z_TOP, max_angle_deg=aperture,
                                   dx_m=50.0, dz_m=2.0)
        p.precision = precision
        p.terrain_method = method
        p.terrain_smoothing_m = smooth
        p.terrain_slope_aperture = False
        return p

    out = {'cycle': cycle, 'aperture_deg': aperture, 'slope_deg': slope}
    runs = {}
    for prec in (1e-2, 1e-3):
        t = time.time()
        f = rwp_tf_forward_task(src, env, params(prec, 'transform'))
        runs[f'tf_{prec}'] = dict(field=f, time=time.time() - t)
        print(f'transform prec {prec}: {time.time() - t:.0f} s', flush=True)
    # the staircase solver sees the same (smoothed) terrain
    env_s = deepcopy(env)
    from pywaveprop.helmholtz_jax import PiecewiseLinearTerrainModel
    xs_fine = np.linspace(0, x_t[-1], 3000)
    env_s.terrain = PiecewiseLinearTerrainModel(xs_fine, tm_probe.boundary(xs_fine))
    for prec in (1e-2, 1e-3):
        t = time.time()
        xs, zs, fs, model = staircase_run(src, env_s, params(prec, 'staircase'), h0)
        runs[f'st_{prec}'] = dict(x=xs, z=zs, field=fs, time=time.time() - t, dz=model.dz_m, dx=model.dx_m)
        print(f'staircase prec {prec}: dz {model.dz_m:.3f}, {time.time() - t:.0f} s', flush=True)

    # ------------------------------------------------------ cuts above ground
    x_cut = np.arange(500.0, x_t[-1], 50.0)
    ground = tm_probe.boundary(x_cut)
    z_fine = np.arange(0.0, Z_TOP, 0.5)
    lam = 3e8 / FREQ_HZ

    def pl(amp, x):
        return -20 * np.log10(amp + 1e-30) + 20 * np.log10(4 * np.pi) + 10 * np.log10(x) - 30 * np.log10(lam)

    cuts = {}
    for name, r in runs.items():
        if name.startswith('tf'):
            grid = r['field'].on_regular_grid(x_cut, z_fine)
        else:
            amp = np.abs(r['field'])
            grid = np.array([np.interp(z_fine, r['z'], amp[np.argmin(np.abs(r['x'] - xc))]) for xc in x_cut])
        for agl in (10.0, 50.0):
            vals = np.array([np.interp(g + agl, z_fine, row) for g, row in zip(ground, grid)])
            cuts[(name, agl)] = pl(vals, x_cut)
    for agl in (10.0, 50.0):
        for a, b in (('tf_0.01', 'tf_0.001'), ('st_0.01', 'st_0.001'), ('tf_0.001', 'st_0.001')):
            d = np.abs(cuts[(a, agl)] - cuts[(b, agl)])
            ok = np.isfinite(d)
            out[f'{a}_vs_{b}_agl{int(agl)}'] = dict(median_db=float(np.median(d[ok])), p90_db=float(np.percentile(d[ok], 90)))
    for name, r in runs.items():
        out[f'time_{name}'] = r['time']

    np.savez_compressed(os.path.join(os.path.dirname(FIG_DIR), 'troposphere_highlands_results.npz'),
                        x_cut=x_cut, **{f'{k[0]}_agl{int(k[1])}': v for k, v in cuts.items()})
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), gridspec_kw={'height_ratios': [2.2, 1.2, 1.2]})
    f = runs['tf_0.001']['field']
    plmap = f.path_loss().field
    n = min(plmap.shape[0], f.x_phys.shape[0])
    pc = axes[0].pcolormesh(f.x_phys[:n] / 1e3, f.z_phys[:n], plmap[:n], cmap='jet_r', vmin=60, vmax=160,
                            shading='nearest')
    axes[0].fill_between(tm_probe.x_fine / 1e3, tm_probe.boundary_fine, 0, color='saddlebrown', alpha=0.8)
    axes[0].set_xlim(0, x_t[-1] / 1e3)
    axes[0].set_ylim(max(h_t.min() - 50, 0), Z_TOP)
    axes[0].set_ylabel('height ASL, m')
    axes[0].set_title(f'{FREQ_HZ / 1e6:.0f} MHz, path loss (terrain-following), GFS {cycle}')
    fig.colorbar(pc, ax=axes[0], label='dB')
    for ax, agl in zip(axes[1:], (10.0, 50.0)):
        ax.plot(x_cut / 1e3, cuts[('tf_0.001', agl)], 'C3', lw=1.6, label='terrain-following, prec $10^{-3}$')
        ax.plot(x_cut / 1e3, cuts[('tf_0.01', agl)], 'k:', lw=1.0, label='terrain-following, prec $10^{-2}$')
        ax.plot(x_cut / 1e3, cuts[('st_0.001', agl)], 'C0', lw=1.0, label='staircase, prec $10^{-3}$')
        ax.plot(x_cut / 1e3, cuts[('st_0.01', agl)], 'C0--', lw=0.8, label='staircase, prec $10^{-2}$')
        ax.set_ylabel(f'path loss, dB ({agl:.0f} m AGL)')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
    axes[1].legend(fontsize=7, ncol=2)
    axes[-1].set_xlabel('range, km')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'troposphere_highlands.png'), dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 4.5))
    ax.plot(M[sel], z_m[sel], 'k')
    ax.set_xlabel('M, M-units')
    ax.set_ylabel('height ASL, m')
    ax.set_title('GFS, path average')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'troposphere_highlands_M.png'), dpi=140)
    plt.close(fig)
    with open(os.path.join(FIG_DIR, 'troposphere_highlands.json'), 'w') as fjs:
        json.dump(out, fjs, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
