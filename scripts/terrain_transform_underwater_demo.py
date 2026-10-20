"""Underwater acoustics over real bathymetry in terrain-following coordinates.

A 31 km path down the slope of the Bermuda pedestal
(32.475N 64.598W, 100 m deep, -> 32.70N 64.40W, 3400 m deep), bathymetry from
the elevation tiles of ``pywaveprop.environment`` and the mean Argo sound speed
profile of the area (2020--2023), extended below 2000 m with the deep isothermal
gradient. A 100 Hz source 50 m deep on the platform edge; a fluid bottom
(1700 m/s, 1.5 g/cm^3, 0.5 dB/lambda) whose interface follows the bathymetry.

Compared solvers, all started from RAM's self-starter and fed the same smoothed
bathymetry:

* the terrain-following rational solver of ``pywaveprop``;
* the same rational scheme with a stair-step bottom;
* RAM 1.5 rebuilt in double precision (``$RAM_START_EXE``, a copy patched to
  write its starting field) at two grids, the finer one being the reference.

Run::

    RAM_START_EXE=/path/to/ram_start_dp python scripts/terrain_transform_underwater_demo.py
"""
import json
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import terrain_transform_wedge_uwa as W  # noqa: E402  (RAM driver and helpers)
from pywaveprop.environment.seafloor import SeafloorProvider  # noqa: E402
from pywaveprop.environment.svp import load_argo_svp  # noqa: E402
from pywaveprop.helmholtz_tf_jax import TFMedium  # noqa: E402
from pywaveprop.terrain_transform import OrthogonalTerrainMap  # noqa: E402

FIG_DIR = W.FIG_DIR
P1, P2 = (32.475, -64.598), (32.70, -64.40)
FREQ = 100.0
ZS = 50.0
RECEIVERS = (50.0, 500.0)
C_B, RHO_B, ATTN_B = 1700.0, 1.5, 0.5
SMOOTH = 500.0
APERTURE = float(os.environ.get('BERMUDA_APERTURE', 75.0))
PRECISION = float(os.environ.get('BERMUDA_PRECISION', 0.1))
TAG = f'_{int(APERTURE)}deg'


def environment():
    bathy = SeafloorProvider(resolution=0.002).profile(P1[0], P1[1], P2[0], P2[1], n_points=400)
    x = np.asarray(bathy.ranges_m, dtype=float)
    d = np.maximum(np.asarray(bathy.depths_m, dtype=float), 60.0)
    svp = load_argo_svp(31.65, -64.55, radius_deg=1.5, start_date="2020-01-01", end_date="2023-12-31",
                        max_depth_m=2000.0)
    zc = np.asarray(svp.depths_m, dtype=float)
    cc = np.asarray(svp.speeds_m_s, dtype=float)
    z_ext = np.arange(zc[-1] + 50.0, 12000.0, 50.0)
    zc = np.concatenate([zc, z_ext])
    cc = np.concatenate([cc, cc[-1] + 0.0165 * (z_ext - z_ext[0] + 50.0)])
    return x, d, zc, cc


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    x_b, d_b, zc, cc = environment()
    r_max = float(x_b[-1])
    W.R_MAX, W.FREQ, W.ZS, W.RECEIVERS = r_max, FREQ, ZS, RECEIVERS
    W.C_B, W.RHO_B, W.ATTN_B = C_B, RHO_B, ATTN_B
    W.C_W = float(np.interp(ZS, zc, cc))
    relief = float(d_b.max() - d_b.min())
    tm = OrthogonalTerrainMap(x_b, d_b, x_max_m=r_max, kind='strip', transition_m=1.5 * relief,
                              eta_b_m=float(d_b.max()), smoothing_m=SMOOTH)
    slope = tm.max_slope_deg()
    lam = W.C_W / FREQ
    z_size = tm.eta_lab + 10 * lam
    print(f'path {r_max / 1e3:.1f} km, depth {d_b.min():.0f}..{d_b.max():.0f} m, smoothed slope {slope:.1f} deg, '
          f'domain depth {z_size:.0f} m')
    omega = 2 * np.pi * FREQ
    kb2 = (omega / W.eff_speed(C_B, ATTN_B)) ** 2

    def k2_water(x, z):
        return (omega / np.interp(z, zc, cc)) ** 2 + 0j

    med_tf = TFMedium(k2_above=k2_water, k2_below=lambda x, z, dd: np.full(np.shape(z), kb2, dtype=complex),
                      rho_above=1.0, rho_below=RHO_B)

    def below(x, z):
        return z >= tm.boundary(x)
    med_st = TFMedium(k2_above=lambda x, z: np.where(below(x, z), kb2, k2_water(x, z)).astype(complex),
                      k2_below=lambda x, z, dd: np.full(np.shape(z), kb2, dtype=complex),
                      rho_above=lambda x, z: np.where(below(x, z), RHO_B, 1.0), rho_below=RHO_B)
    eta_flat = float(np.ceil(d_b.max() + 20.0))
    tm_flat = OrthogonalTerrainMap([0.0, r_max + 5000], [eta_flat, eta_flat], x_max_m=r_max, kind='strip',
                                   transition_m=20.0, eta_b_m=eta_flat, smoothing_m=0.0)

    # ---------------------------------------------------------------- RAM
    sx = np.linspace(0.0, r_max + 200, 900)
    sz = tm.boundary(sx)
    zmax_ram = float(d_b.max() + 2500.0)
    # RAM gets the profile at its own data points: the mean Argo profile has sharp
    # structure near the surface, and resampling it changes the field by ~1 dB
    ssp_pts = [(z, c) for z, c in zip(zc, cc) if z <= zmax_ram]
    if ssp_pts[0][0] > 0.0:
        ssp_pts = [(0.0, cc[0])] + ssp_pts

    def run_ram(dr, dz):
        # water and bottom profiles: RAM tabulates them in absolute depth
        import subprocess
        import tempfile
        work = tempfile.mkdtemp(prefix="ram_", dir=os.environ.get("RAM_WORK_DIR"))
        ndz = max(int(round(2.0 / dz)), 1)
        with open(os.path.join(work, "ram.in"), "w") as f:
            f.write("bermuda\n")
            f.write(f"{FREQ} {ZS} {RECEIVERS[0]}\n{r_max} {dr} 1\n{zmax_ram} {dz} {ndz} {zmax_ram}\n")
            f.write(f"{W.C_W} 8 1 0.0\n")
            for xb, zb in zip(sx, sz):
                f.write(f"{xb:.3f} {zb:.4f}\n")
            f.write("-1 -1\n")
            for z, c in ssp_pts:
                f.write(f"{z:.2f} {c:.4f}\n")
            f.write("-1 -1\n")
            f.write(f"0.0 {C_B}\n-1 -1\n0.0 {RHO_B}\n-1 -1\n")
            f.write(f"0.0 {ATTN_B}\n{zmax_ram - 600:.1f} {ATTN_B}\n{zmax_ram:.1f} 10.0\n-1 -1\n")
        t = time.time()
        subprocess.run([W.RAM_EXE], cwd=work, check=True, capture_output=True, timeout=7200)
        res = W.parse_ram_output(work, dz, ndz, zmax_ram)
        return res, time.time() - t

    ref, t_ref = run_ram(2.5, 0.125)
    coarse, t_coarse = run_ram(5.0, 0.25)
    start = (ref[0], ref[1])
    out = {'ram_ref_time_s': t_ref, 'ram_coarse_time_s': t_coarse, 'slope_deg': slope, 'domain_depth_m': z_size,
           'aperture_deg': APERTURE, 'precision': PRECISION}

    runs = {}
    # the aperture is defined in the water: the smallest longitudinal wavenumber is
    # min_water(k) cos(aperture) and does not depend on the (faster) bottom
    c_water = cc[zc <= d_b.max()]
    kw = dict(z_size=z_size, x0=2.5, start=start, angle_deg=APERTURE, dz_out=2.0,
              k_bounds=(omega / float(c_water.max()), omega / float(c_water.min())))
    runs['tf'] = W.run_pywaveprop(tm, med_tf, PRECISION, **kw)
    runs['stair'] = W.run_pywaveprop(tm_flat, med_st, PRECISION, flux_order=0, **kw)

    np.savez_compressed(os.path.join(os.path.dirname(FIG_DIR), f'underwater_bermuda{TAG}_results.npz'),
                        ref_r=ref[2], ref_z=ref[3], ref_amp=ref[4][:, ::4], coarse_r=coarse[2], coarse_z=coarse[3],
                        coarse_amp=coarse[4][:, ::2],
                        # every 2nd output column and 4th row, single precision (the full arrays are ~1 GB)
                        **{f'{k}_{n}': (v[::2, ::4].astype(np.complex64) if np.iscomplexobj(v)
                                        else v[::2, ::4].astype(np.float32))
                           for k, r in runs.items() for n, v in r.items() if isinstance(v, np.ndarray)})
    fig, axes = plt.subplots(len(RECEIVERS) + 1, 1, figsize=(8, 9.5),
                             gridspec_kw={'height_ratios': [2.2] + [1.2] * len(RECEIVERS)})
    res = runs['tf']
    sub = (slice(None, None, 2), slice(None, None, 4))
    db = 20 * np.log10(np.abs(res['field'][sub]) + 1e-12)
    pc = axes[0].pcolormesh(res['x'][sub] / 1e3, res['z'][sub], db, vmin=db.max() - 70, vmax=db.max(), cmap='jet',
                            shading='gouraud')
    axes[0].plot(sx / 1e3, sz, 'w', lw=1)
    axes[0].set_ylim(d_b.max() + 300, 0)
    axes[0].set_ylabel('depth, m')
    axes[0].set_title(f'Bermuda pedestal, {FREQ:.0f} Hz, source {ZS:.0f} m (terrain-following, aperture {APERTURE:.0f}°)')
    fig.colorbar(pc, ax=axes[0], label='dB')
    for ax, zr in zip(axes[1:], RECEIVERS):
        jr = int(np.argmin(np.abs(ref[3] - zr)))
        jc = int(np.argmin(np.abs(coarse[3] - zr)))
        ax.plot(ref[2] / 1e3, 20 * np.log10(ref[4][:, jr]), 'k', lw=2.0, label='RAM (double), dr=2.5 m, dz=0.125 m')
        ax.plot(coarse[2] / 1e3, 20 * np.log10(coarse[4][:, jc]), color='0.6', lw=1.0, label='RAM (double), dr=5 m, dz=0.25 m')
        for name, style, lab in (('tf', dict(color='C3', lw=1.3), 'terrain-following'),
                                 ('stair', dict(color='C0', lw=1.0, ls='--'), 'same scheme, stair-step bottom')):
            xr, ar = W.cut_tf(runs[name], zr)
            ax.plot(xr / 1e3, 20 * np.log10(ar), label=lab, **style)
            ok = (xr > 500) & (xr < r_max - 100)
            d = 20 * np.log10(ar[ok]) - 20 * np.log10(np.interp(xr[ok], ref[2], ref[4][:, jr]))
            out.setdefault(name, {})[f'z{int(zr)}'] = dict(median_abs_db=float(np.median(np.abs(d))),
                                                           p90_abs_db=float(np.percentile(np.abs(d), 90)))
        ok = coarse[2] > 500
        d = 20 * np.log10(coarse[4][ok, jc]) - 20 * np.log10(np.interp(coarse[2][ok], ref[2], ref[4][:, jr]))
        out.setdefault('ram_coarse', {})[f'z{int(zr)}'] = dict(median_abs_db=float(np.median(np.abs(d))),
                                                               p90_abs_db=float(np.percentile(np.abs(d), 90)))
        ax.set_ylabel(f'20 log|p|, z={zr:.0f} m')
        ax.grid(True, alpha=0.3)
    axes[1].legend(fontsize=7, loc='lower left')
    axes[-1].set_xlabel('range, km')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f'underwater_bermuda{TAG}.png'), dpi=140)
    plt.close(fig)
    for name, r in runs.items():
        out[name].update(time_s=r['time'], x_n=r['x_n'], z_n=r['z_n'], dx=r['dx'], dz=r['dz'])
    with open(os.path.join(FIG_DIR, f'underwater_bermuda{TAG}.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
