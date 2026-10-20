"""Rank-adaptive split-step Fourier (RA-SSF) vs classical SSF vs Padé FD on
tropospheric problems.

Produces, in ``--out`` (default: current directory):

* ``rassf_convergence_layer.png``  error vs range step against the exact
  one-way solution (wide-angle beam through a strong refractivity layer);
* ``rassf_convergence_evap10.png`` the same for a 10 GHz evaporation duct;
* ``rassf_fields_*.png``           2D fields and cuts of the Padé / SSF / RA-SSF
  solutions with kilometre steps (evaporation duct at 3 GHz over PEC and sea
  water, surface-based duct, hill);
* ``rassf_results.md``             all numbers (errors, ranks, FFT counts, times).

    python scripts/rassf_vs_ssf_demo.py --out ../rank-adaptive_ssf
"""
import argparse
import os
import time
from copy import deepcopy

import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pywaveprop.helmholtz_rassf_jax import exact_one_way_stratified
from pywaveprop.rwp_jax import (EvaporationDuctModel, GroundMaterial, PiecewiseLinearNProfileModel,
                                RWPComputationalParams, RWPGaussSourceModel, TroposphereModel,
                                create_rwp_rassf_model, create_rwp_ssf_model, rwp_forward_task)

parser = argparse.ArgumentParser()
parser.add_argument('--out', default='.')
parser.add_argument('--skip-pade', action='store_true', help='skip the (slow) Padé reference cases')
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)
lines = ["# Rank-adaptive SSF vs classical SSF vs Padé FD\n",
         "Errors are |20 log10 |u|| differences along horizontal cuts beyond 20% of the range, "
         "restricted to points within 30 dB of the cut maximum (median / 90th percentile / max, dB). "
         "Times are wall-clock seconds of `compute` after compilation (CPU).\n"]


def log(s=""):
    print(s)
    lines.append(s)


def db_stats(ref, other, x_grid, z_grid, z_m, x_skip=0.2, floor_db=-30.0):
    iz = int(np.argmin(np.abs(z_grid - z_m)))
    i0 = int(len(x_grid) * x_skip)
    a = 20 * np.log10(np.abs(ref[i0:, iz]) + 1e-30)
    b = 20 * np.log10(np.abs(other[i0:, iz]) + 1e-30)
    keep = a > a.max() + floor_db
    d = np.abs(a - b)[keep]
    return float(np.median(d)), float(np.percentile(d, 90)), float(d.max())


def timed_compute(model, init, repeat=3):
    f = np.asarray(model.compute(init))
    t0 = time.time()
    for _ in range(repeat):
        f = np.asarray(model.compute(init))
    return f, (time.time() - t0) / repeat


def describe(model):
    if hasattr(model, 'ranks'):
        return f"rank {max(model.ranks)}, {model.ffts_per_step} FFT/step, {model.x_n - 1} steps, " \
               f"{model.ffts_per_step * (model.x_n - 1)} FFTs"
    return f"2 FFT/step, {model.x_n - 1} steps, {2 * (model.x_n - 1)} FFTs"


# ----------------------------------------------------------------- convergence vs exact one-way solution
def convergence(name, title, src, env, params, heights, dxs, fig_name, **kw):
    log(f"\n## {title}\n")
    ref_model = create_rwp_rassf_model(src, env, params, **kw)
    init = src.aperture(ref_model.z_computational_grid())
    t0 = time.time()
    exact = exact_one_way_stratified(ref_model, init, ref_model.x_output_grid())
    log(f"Exact one-way reference: eigen-decomposition of a {ref_model.z_n - 1} x {ref_model.z_n - 1} matrix, "
        f"{time.time() - t0:.0f} s.\n")
    xg, zg = np.asarray(ref_model.x_output_grid()), np.asarray(ref_model.z_output_grid())
    ref = exact[:, :ref_model.z_n_phys:ref_model.z_grid_scale]
    log("| solver | dx (m) | rank | FFTs total | time (s) | " + " | ".join(f"z={z:g} m med/p90/max" for z in heights) + " |")
    log("|---|---|---|---|---|" + "---|" * len(heights))
    curves = {}
    for label, mk in (("SSF", lambda dx: create_rwp_ssf_model(src, env, params, dx_m=dx, absorber='profile', **kw)),
                      ("RA-SSF", lambda dx: create_rwp_rassf_model(src, env, params, dx_m=dx, **kw))):
        curves[label] = []
        for dx in dxs:
            model = mk(dx)
            f, t = timed_compute(model, src.aperture(model.z_computational_grid()))
            st = [db_stats(ref, f, xg, zg, z) for z in heights]
            rank = max(model.ranks) if hasattr(model, 'ranks') else 1
            n_fft = (model.ffts_per_step if hasattr(model, 'ffts_per_step') else 2) * (model.x_n - 1)
            curves[label].append((model.dx_m, max(s[1] for s in st), n_fft, t))
            log(f"| {label} | {model.dx_m:g} | {rank} | {n_fft} | {t:.3f} | " +
                " | ".join(f"{s[0]:.3f} / {s[1]:.3f} / {s[2]:.2f}" for s in st) + " |")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for label, c in curves.items():
        c = np.array(c)
        axes[0].loglog(c[:, 0], c[:, 1], 'o-', label=label)
        axes[1].loglog(c[:, 2], c[:, 1], 'o-', label=label)
    axes[0].set_xlabel('range step dx (m)')
    axes[0].set_ylabel('90th percentile |dB error| (worst height)')
    axes[1].set_xlabel('total number of FFTs')
    axes[1].set_ylabel('90th percentile |dB error| (worst height)')
    for ax in axes:
        ax.grid(True, which='both', alpha=0.3)
        ax.legend()
    fig.suptitle(f'{title}: error against the exact one-way solution')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, fig_name), dpi=120)
    plt.close()
    log(f"\n![{title}]({fig_name})\n")


z_grid = jnp.array([0.0, 100.0, 150.0, 400.0])
convergence('layer', 'Wide-angle beam (20 deg) through a -50 N-unit layer, 1 GHz, 10 km',
            RWPGaussSourceModel(freq_hz=1e9, height_m=50.0, beam_width_deg=20.0),
            TroposphereModel(N_profile=PiecewiseLinearNProfileModel(z_grid_m=z_grid,
                                                                    N_vals=jnp.array([0.0, 0.0, -50.0, -50.0]))),
            RWPComputationalParams(max_range_m=10000.0, max_height_m=300.0, dx_m=250.0, dz_m=1.0),
            [30.0, 120.0, 250.0], [5.0, 25.0, 125.0, 250.0], 'rassf_convergence_layer.png',
            abs_attenuation_db=0.0)

convergence('evap10', 'Evaporation duct (20 m) at 10 GHz, 5 deg beam, 2 km',
            RWPGaussSourceModel(freq_hz=10e9, height_m=10.0, beam_width_deg=5.0),
            TroposphereModel(N_profile=EvaporationDuctModel(height_m=20.0)),
            RWPComputationalParams(max_range_m=2000.0, max_height_m=350.0, dx_m=500.0, dz_m=0.5),
            [5.0, 10.0, 30.0], [10.0, 50.0, 100.0, 250.0, 500.0], 'rassf_convergence_evap10.png',
            abs_layer_scale=0.0)


# ------------------------------------------------------------------------ kilometre steps vs Padé FD
def pade_case(name, title, src, env, params, heights, big_dx):
    log(f"\n## {title}\n")
    p = deepcopy(params)
    p.approx_method = 'pade'
    p.rational_approx_order = (7, 8)
    t0 = time.time()
    fd = rwp_forward_task(src, env, p)
    t_fd = time.time() - t0
    fdf, xg, zg = np.asarray(fd.field), np.asarray(fd.x_grid), np.asarray(fd.z_grid)
    log(f"Padé (7, 8) reference: {t_fd:.0f} s including compilation.\n")
    log("| solver | dx (m) | absorber | cost | time (s) | " + " | ".join(f"z={z:g} m med/p90/max" for z in heights) + " |")
    log("|---|---|---|---|---|" + "---|" * len(heights))
    fields = {"Padé FD": fdf}
    runs = (("SSF, default step", lambda: create_rwp_ssf_model(src, env, params), 'window'),
            (f"SSF, {big_dx:g} m", lambda: create_rwp_ssf_model(src, env, params, dx_m=big_dx), 'window'),
            (f"SSF, {big_dx:g} m", lambda: create_rwp_ssf_model(src, env, params, dx_m=big_dx, absorber='profile'), 'profile'),
            (f"RA-SSF, {big_dx:g} m", lambda: create_rwp_rassf_model(src, env, params, dx_m=big_dx), 'profile'))
    for label, mk, absorber in runs:
        model = mk()
        f, t = timed_compute(model, src.aperture(model.z_computational_grid()))
        st = [db_stats(fdf, f, xg, zg, z) for z in heights]
        log(f"| {label} | {model.dx_m:g} | {absorber} | {describe(model)} | {t:.4f} | " +
            " | ".join(f"{s[0]:.2f} / {s[1]:.2f} / {s[2]:.2f}" for s in st) + " |")
        if label.startswith("RA-SSF") or (label.startswith("SSF") and absorber == 'window' and 'default' in label):
            fields[label] = f
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    for ax, (label, f) in zip(axes[0], fields.items()):
        amp = 20 * np.log10(np.abs(f).T + 1e-30)
        vmax = float(np.max(amp[:, len(xg) // 10:]))
        ax.pcolormesh(xg, zg, amp, shading='auto', cmap='jet', vmin=vmax - 60, vmax=vmax)
        ax.set_title(f'{label}')
        ax.set_xlabel('Range (m)')
        ax.set_ylabel('Height (m)')
    for ax, z in zip(axes[1], heights):
        for label, f in fields.items():
            iz = int(np.argmin(np.abs(zg - z)))
            ax.plot(xg, 20 * np.log10(np.abs(f[:, iz]) + 1e-30), label=label)
        ax.set_title(f'z = {z:g} m')
        ax.set_xlabel('Range (m)')
        ax.set_ylabel('|u| (dB)')
        ax.set_ylim(bottom=-80)
        ax.legend()
    fig.suptitle(title)
    plt.tight_layout()
    fig_name = f'rassf_fields_{name}.png'
    plt.savefig(os.path.join(args.out, fig_name), dpi=110)
    plt.close()
    log(f"\n![{title}]({fig_name})\n")


if not args.skip_pade:
    pade_case('evap3', 'Evaporation duct (20 m) at 3 GHz over PEC, 3 deg beam, 20 km',
              RWPGaussSourceModel(freq_hz=3e9, height_m=10.0, beam_width_deg=3.0),
              TroposphereModel(N_profile=EvaporationDuctModel(height_m=20.0)),
              RWPComputationalParams(max_range_m=20000.0, max_height_m=150.0, dx_m=1000.0, dz_m=0.5),
              [5.0, 10.0, 30.0], 1000.0)
    pade_case('evap3_sea', 'Evaporation duct (20 m) at 3 GHz over sea water, V-pol, 3 deg beam, 20 km',
              RWPGaussSourceModel(freq_hz=3e9, height_m=10.0, beam_width_deg=3.0, polarization='V'),
              TroposphereModel(N_profile=EvaporationDuctModel(height_m=20.0),
                               ground_material=GroundMaterial(eps=80.0, sigma=4.0)),
              RWPComputationalParams(max_range_m=20000.0, max_height_m=150.0, dx_m=1000.0, dz_m=0.5),
              [5.0, 10.0, 30.0], 1000.0)
    pade_case('sbd', 'Surface-based duct (-20 N-units, 100-200 m) at 1 GHz, 4 deg beam, 20 km',
              RWPGaussSourceModel(freq_hz=1e9, height_m=50.0, beam_width_deg=4.0),
              TroposphereModel(N_profile=PiecewiseLinearNProfileModel(
                  z_grid_m=jnp.array([0.0, 100.0, 200.0, 300.0]), N_vals=jnp.array([0.0, -20.0, -20.0, 0.0]))),
              RWPComputationalParams(max_range_m=20000.0, max_height_m=400.0, dx_m=1000.0, dz_m=1.0),
              [20.0, 50.0, 150.0], 1000.0)
    pade_case('strong_duct', 'Strong elevated duct (-40 N-units above 150 m) at 3 GHz, 8 deg beam, 20 km',
              RWPGaussSourceModel(freq_hz=3e9, height_m=50.0, beam_width_deg=8.0),
              TroposphereModel(N_profile=PiecewiseLinearNProfileModel(
                  z_grid_m=jnp.array([0.0, 100.0, 150.0, 300.0]), N_vals=jnp.array([0.0, 0.0, -40.0, -40.0]))),
              RWPComputationalParams(max_range_m=20000.0, max_height_m=400.0, dx_m=1000.0, dz_m=0.5),
              [20.0, 50.0, 120.0], 1000.0)

with open(os.path.join(args.out, 'rassf_results.md'), 'w') as fh:
    fh.write("\n".join(lines) + "\n")
print(f"\nwritten to {args.out}")
