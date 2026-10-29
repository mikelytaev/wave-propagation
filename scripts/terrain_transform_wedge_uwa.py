"""Penetrable wedge (ASA-benchmark-type) in terrain-following coordinates.

Compares, for a sloping fluid bottom, the terrain-following rational solver
(:class:`pywaveprop.helmholtz_tf_jax.TerrainFollowingHelmholtzPropagator`) with

* RAM 1.5 (M. D. Collins, stair-step bathymetry), rebuilt in double precision, at a very fine grid, used as
  the reference, and at a typical grid;
* the same rational scheme with a stair-step bottom (flat map, range-dependent
  density and sound speed): it differs from the terrain-following solver only
  in the treatment of the geometry.

All solvers start from RAM's self-starter (dumped by a patched copy of RAM that
writes ``start.dat``) and use the same smoothed bathymetry.  The RAM executable
is taken from ``$RAM_START_EXE``.

Run::

    RAM_START_EXE=/path/to/ram_start python scripts/terrain_transform_wedge_uwa.py
"""
import json
import os
import struct
import subprocess
import sys
import tempfile
import time

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jax.numpy as jnp  # noqa: E402

from pywaveprop.helmholtz_common import HelmholtzMeshParams2D  # noqa: E402
from pywaveprop.helmholtz_tf_jax import TerrainFollowingHelmholtzPropagator, TFMedium  # noqa: E402
from pywaveprop.terrain_transform import OrthogonalTerrainMap  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "papers", "terrain_transform", "figures")
RAM_EXE = os.environ.get("RAM_START_EXE", "ram_start")

# ---------------------------------------------------------------- problem
FREQ = float(os.environ.get("WEDGE_FREQ", 25.0))
C_W, C_B, RHO_B, ATTN_B = 1500.0, 1700.0, 1.5, 0.5
SLOPE_DEG = float(os.environ.get("WEDGE_SLOPE", 2.86))
D0, R_FLAT = 200.0, 500.0           # flat lead-in, then upslope
R_MAX = float(os.environ.get("WEDGE_RMAX", 3500.0))
ZS = 100.0
RECEIVERS = (30.0, 150.0)
SMOOTH = 20.0


def bathymetry(x):
    return np.where(x < R_FLAT, D0, D0 - np.tan(np.radians(SLOPE_DEG)) * (x - R_FLAT))


def make_map(transition):
    xb = np.linspace(0.0, R_MAX + 3000.0, 3001)
    return OrthogonalTerrainMap(xb, np.maximum(bathymetry(xb), 10.0), x_max_m=R_MAX, kind='strip',
                                transition_m=transition, eta_b_m=D0, smoothing_m=SMOOTH)


def eff_speed(c, attn):
    eta = 1.0 / (40.0 * np.pi * np.log10(np.e))
    return c / (1.0 + 1j * eta * attn)


# ------------------------------------------------------------------- RAM
def run_ram(seabed_x, seabed_z, dr, dz, zmax, np_pade=8):
    """Run RAM; returns (z_start, p_start, ranges, depths, |p| grid)."""
    work = tempfile.mkdtemp(prefix="ram_", dir=os.environ.get("RAM_WORK_DIR"))
    ndz = max(int(round(0.5 / dz)), 1)
    with open(os.path.join(work, "ram.in"), "w") as f:
        f.write("wedge\n")
        f.write(f"{FREQ} {ZS} {RECEIVERS[0]}\n")
        f.write(f"{R_MAX} {dr} 1\n")
        f.write(f"{zmax} {dz} {ndz} {zmax}\n")
        f.write(f"{C_W} {np_pade} 1 0.0\n")
        for xb, zb in zip(seabed_x, seabed_z):
            f.write(f"{xb:.3f} {zb:.4f}\n")
        f.write("-1 -1\n")
        f.write(f"0.0 {C_W}\n-1 -1\n")
        f.write(f"0.0 {C_B}\n-1 -1\n")
        f.write(f"0.0 {RHO_B}\n-1 -1\n")
        # sediment attenuation plus an absorbing layer above zmax
        f.write(f"0.0 {ATTN_B}\n{zmax - 0.3 * zmax:.1f} {ATTN_B}\n{zmax:.1f} 10.0\n-1 -1\n")
    subprocess.run([RAM_EXE], cwd=work, check=True, capture_output=True, timeout=3600)
    return parse_ram_output(work, dz, ndz, zmax)


def parse_ram_output(work, dz, ndz, zmax):
    """Starting field and ``|p| sqrt(r)`` on the output grid of a RAM run in ``work``."""
    start = np.loadtxt(os.path.join(work, "start.dat"))
    p0 = start[:, 0] + 1j * start[:, 1]
    z0 = np.arange(p0.size) * dz
    with open(os.path.join(work, "tl.grid"), "rb") as f:
        data = f.read()
    recs, off = [], 0
    while off < len(data):
        n = struct.unpack("i", data[off:off + 4])[0]
        recs.append(data[off + 4:off + 4 + n])
        off += n + 8
    n_bytes = max(set(len(r) for r in recs), key=[len(r) for r in recs].count)
    lz = int(zmax / dz - 0.5) // ndz
    dtype = np.float64 if n_bytes == 8 * lz else np.float32  # double-precision build of RAM
    tl = np.array([np.frombuffer(r, dtype=dtype) for r in recs if len(r) == n_bytes])  # drop the header
    ranges = np.loadtxt(os.path.join(work, "tl.line"))[:, 0][-tl.shape[0]:]
    depths = (np.arange(tl.shape[1]) + 1) * dz * ndz - dz  # u(i) is at depth (i - 1) dz
    # reduced field |p| sqrt(r) (cylindrical spreading removed)
    amp = 10 ** (-(tl - 10 * np.log10(ranges)[:, None]) / 20)
    return z0, p0, ranges, depths, amp


# ------------------------------------------------------------- pywaveprop
def medium_tf():
    omega = 2 * np.pi * FREQ
    return TFMedium(
        k2_above=lambda x, z: np.full(np.shape(z), (omega / C_W) ** 2, dtype=complex),
        k2_below=lambda x, z, d: np.full(np.shape(z), (omega / eff_speed(C_B, ATTN_B)) ** 2, dtype=complex),
        rho_above=1.0, rho_below=RHO_B)


def medium_staircase(tm):
    omega = 2 * np.pi * FREQ
    kw2, kb2 = (omega / C_W) ** 2, (omega / eff_speed(C_B, ATTN_B)) ** 2

    def below(x, z):
        return z >= tm.boundary(x)
    return TFMedium(
        k2_above=lambda x, z: np.where(below(x, z), kb2, kw2).astype(complex),
        k2_below=lambda x, z, d: np.full(np.shape(z), kb2, dtype=complex),
        rho_above=lambda x, z: np.where(below(x, z), RHO_B, 1.0),
        rho_below=RHO_B)


def run_pywaveprop(tm, medium, precision, order=(7, 8), z_size=None, x0=0.0, start=None, dz_out=1.0,
                   k_bounds=None, **kw):
    omega = 2 * np.pi * FREQ
    k_bounds = k_bounds or (omega / C_B, omega / C_W)
    kz_max = omega / C_W * np.sin(np.radians(kw.pop('angle_deg', 50.0)))
    mp = HelmholtzMeshParams2D(x_size_m=R_MAX - x0, z_size_m=z_size, dx_output_m=10.0, dz_output_m=dz_out)
    t = time.time()
    model = TerrainFollowingHelmholtzPropagator.create(FREQ, tm, medium, kz_max, k_bounds, precision, mp,
                                                       rational_approx_order=order, **kw)
    _, z0 = model.initial_points()
    init = np.interp(z0, start[0], start[1].real) + 1j * np.interp(z0, start[0], start[1].imag)
    f = np.asarray(model.compute(jnp.asarray(init)))
    xs, zs = model.output_metrics()
    return dict(field=f, x=xs + x0, z=zs, time=time.time() - t, x_n=model.x_n, z_n=model.z_n,
                dx=model.dx_m, dz=model.dz_m)


def cut_tf(res, depth):
    """|p| along a receiver depth from a terrain-following result."""
    out = np.empty(res['x'].shape[0])
    for i in range(out.size):
        out[i] = np.interp(depth, res['z'][i], np.abs(res['field'][i]))
    return res['x'][:, 0], out


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    relief = D0 - float(bathymetry(np.array([R_MAX]))[0])
    tm = make_map(transition=max(1.5 * relief, 20 * C_W / FREQ))
    zmax_tf = tm.eta_lab + 10 * C_W / FREQ
    # bathymetry seen by every solver: the smoothed one
    sx = np.linspace(0, R_MAX + 100, 900)
    sz = tm.boundary(sx)

    results = {}
    t = time.time()
    ref = run_ram(sx, sz, dr=1.0, dz=0.025, zmax=1200.0)
    results['ram_ref_time'] = time.time() - t
    t = time.time()
    ram_c = run_ram(sx, sz, dr=10.0, dz=0.5, zmax=1200.0)
    results['ram_coarse_time'] = time.time() - t
    x0 = 1.0  # RAM starter range
    start = (ref[0], ref[1])
    runs = {}
    for prec in (1e-2, 1e-3):
        runs[f'tf_{prec}'] = run_pywaveprop(tm, medium_tf(), prec, z_size=zmax_tf, x0=x0, start=start)
    tm_flat = OrthogonalTerrainMap([0.0, R_MAX + 3000], [D0 + 5.0, D0 + 5.0], x_max_m=R_MAX, kind='strip',
                                   transition_m=10.0, eta_b_m=D0 + 5.0, smoothing_m=0.0)
    for prec in (1e-2, 1e-3):
        runs[f'stair_{prec}'] = run_pywaveprop(tm_flat, medium_staircase(tm), prec, z_size=zmax_tf, x0=x0,
                                               start=start, flux_order=0)

    # ------------------------------------------------------------ errors
    fig, axes = plt.subplots(len(RECEIVERS), 1, figsize=(7.5, 6.5), sharex=True)
    summary = {}
    for ax, zr in zip(axes, RECEIVERS):
        jr = int(np.argmin(np.abs(ref[3] - zr)))
        r_ref, a_ref = ref[2], ref[4][:, jr]
        ax.plot(r_ref / 1e3, 20 * np.log10(a_ref), 'k', lw=2.2, label='RAM (double precision), dr=1 m, dz=0.025 m (reference)')
        jc = int(np.argmin(np.abs(ram_c[3] - zr)))
        ax.plot(ram_c[2] / 1e3, 20 * np.log10(ram_c[4][:, jc]), color='0.6', lw=1.2, label='RAM, dr=10 m, dz=0.5 m')
        for name, style in (('tf_0.001', dict(color='C3', lw=1.4)), ('stair_0.001', dict(color='C0', lw=1.2, ls='--'))):
            xr, ar = cut_tf(runs[name], zr)
            lab = 'terrain-following' if name.startswith('tf') else 'same scheme, stair-step bottom'
            ax.plot(xr / 1e3, 20 * np.log10(ar), label=lab + f' (prec {name.split("_")[1]})', **style)
        for name, res in runs.items():
            xr, ar = cut_tf(res, zr)
            ok = (xr > 200) & (xr < R_MAX - 50)
            a_i = np.interp(xr[ok], r_ref, a_ref)
            d = 20 * np.log10(ar[ok]) - 20 * np.log10(a_i)
            summary.setdefault(name, {})[f'z{int(zr)}'] = dict(median_abs_db=float(np.median(np.abs(d))),
                                                               p90_abs_db=float(np.percentile(np.abs(d), 90)))
        ok = (ram_c[2] > 200)
        d = 20 * np.log10(ram_c[4][ok, jc]) - 20 * np.log10(np.interp(ram_c[2][ok], r_ref, a_ref))
        summary.setdefault('ram_coarse', {})[f'z{int(zr)}'] = dict(median_abs_db=float(np.median(np.abs(d))),
                                                                   p90_abs_db=float(np.percentile(np.abs(d), 90)))
        ax.set_ylabel(f'20 log|p|, z={zr:.0f} m')
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, loc='lower left')
    axes[-1].set_xlabel('range, km')
    fig.suptitle(f'Penetrable wedge, {FREQ:.0f} Hz, slope {SLOPE_DEG}°')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f'wedge_uwa_{int(FREQ)}hz_cuts.png'), dpi=150)
    plt.close(fig)

    # field map of the terrain-following solution
    res = runs['tf_0.001']
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    db = 20 * np.log10(np.abs(res['field']) + 1e-12)
    pc = ax.pcolormesh(res['x'] / 1e3, res['z'], db, vmin=db.max() - 60, vmax=db.max(), cmap='jet', shading='auto')
    ax.plot(sx / 1e3, sz, 'w', lw=1)
    ax.set_ylim(zmax_tf, 0)
    ax.set_xlabel('range, km')
    ax.set_ylabel('depth, m')
    fig.colorbar(pc, ax=ax, label='dB')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f'wedge_uwa_{int(FREQ)}hz_field.png'), dpi=150)
    plt.close(fig)

    for name, res in runs.items():
        summary[name].update(time_s=res['time'], x_n=res['x_n'], z_n=res['z_n'], dx=res['dx'], dz=res['dz'])
    # ------------------------------------------------- grid convergence
    conv = {'dz': [], 'tf': [], 'stair': []}
    for dz_out in (2.0, 1.0, 0.5, 0.25):
        conv['dz'].append(dz_out)
        for key, tmap, med, kw in (('tf', tm, medium_tf(), {}),
                                   ('stair', tm_flat, medium_staircase(tm), dict(flux_order=0))):
            res = run_pywaveprop(tmap, med, 1e-3, z_size=zmax_tf, x0=x0, start=start, dz_out=dz_out, **kw)
            errs = []
            for zr in RECEIVERS:
                jr = int(np.argmin(np.abs(ref[3] - zr)))
                xr, ar = cut_tf(res, zr)
                ok = (xr > 200) & (xr < R_MAX - 50)
                errs.append(np.median(np.abs(20 * np.log10(ar[ok]) - 20 * np.log10(np.interp(xr[ok], ref[2], ref[4][:, jr])))))
            conv[key].append(float(np.mean(errs)))
            print('convergence', key, dz_out, res['dz'], conv[key][-1], flush=True)
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.loglog(conv['dz'], conv['tf'], 'o-', color='C3', label='terrain-following')
    ax.loglog(conv['dz'], conv['stair'], 's--', color='C0', label='stair-step bottom')
    ax.set_xticks(conv['dz'])
    ax.set_xticklabels([f'{v:g}' for v in conv['dz']])
    ax.minorticks_off()
    ax.set_xlabel(r'$\Delta z$ (output step), m')
    ax.set_ylabel('median |TL error|, dB (both receivers)')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f'wedge_uwa_{int(FREQ)}hz_convergence.png'), dpi=150)
    plt.close(fig)
    summary['convergence'] = conv
    summary['ram_ref_time_s'] = results['ram_ref_time']
    summary['ram_coarse_time_s'] = results['ram_coarse_time']
    with open(os.path.join(FIG_DIR, f'wedge_uwa_{int(FREQ)}hz.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
