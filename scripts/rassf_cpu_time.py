"""CPU time of the rank-adaptive SSF versus the classical SSF at matched accuracy.

Each solver is run once for compilation, then ``--repeat`` times; the median
process CPU time (all threads) and wall time of ``compute`` are reported
together with the setup time (grid + cross approximation), the number of FFTs
and the error against the reference (Padé FD, or the exact one-way solution
for the stratified wide-angle case). Writes ``rassf_cpu_time.md`` into ``--out``.

    python scripts/rassf_cpu_time.py --out ../rank-adaptive_ssf
"""
import argparse
import os
import time
from copy import deepcopy

import jax.numpy as jnp
import numpy as np

from pywaveprop.helmholtz_rassf_jax import exact_one_way_stratified
from pywaveprop.rwp_jax import (EvaporationDuctModel, GroundMaterial, PiecewiseLinearNProfileModel,
                                RWPComputationalParams, RWPGaussSourceModel, TroposphereModel,
                                create_rwp_rassf_model, create_rwp_ssf_model, rwp_forward_task)

parser = argparse.ArgumentParser()
parser.add_argument('--out', default='.')
parser.add_argument('--repeat', type=int, default=5)
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)
lines = ["# CPU time: rank-adaptive SSF vs classical SSF\n",
         f"Median of {args.repeat} runs after one warm-up run; CPU = process CPU time summed over all threads "
         f"(`time.process_time`), wall = `time.perf_counter`; {os.cpu_count()} logical CPUs, JAX CPU backend, "
         "float64. Setup = grid construction plus the cross approximation (numpy). Errors: |dB| difference "
         "statistics along horizontal cuts (median / 90th percentile / max over the listed heights, x > 20% "
         "of the range, points within 30 dB of the cut maximum).\n"]


def log(s=""):
    print(s, flush=True)
    lines.append(s)


def db_stats(ref, other, x_grid, z_grid, z_m, x_skip=0.2, floor_db=-30.0):
    iz = int(np.argmin(np.abs(z_grid - z_m)))
    i0 = int(len(x_grid) * x_skip)
    a = 20 * np.log10(np.abs(ref[i0:, iz]) + 1e-30)
    b = 20 * np.log10(np.abs(other[i0:, iz]) + 1e-30)
    keep = a > a.max() + floor_db
    d = np.abs(a - b)[keep]
    return float(np.median(d)), float(np.percentile(d, 90)), float(d.max())


def bench(mk, src):
    t0c, t0w = time.process_time(), time.perf_counter()
    model = mk()
    setup = time.perf_counter() - t0w
    init = src.aperture(model.z_computational_grid())
    f = np.asarray(model.compute(init))          # compilation
    cpu, wall = [], []
    for _ in range(args.repeat):
        c0, w0 = time.process_time(), time.perf_counter()
        f = np.asarray(model.compute(init))
        cpu.append(time.process_time() - c0)
        wall.append(time.perf_counter() - w0)
    return model, f, setup, float(np.median(cpu)), float(np.median(wall))


def cost(model):
    if hasattr(model, 'ranks'):
        return max(model.ranks), model.ffts_per_step * (model.x_n - 1)
    return 1, 2 * (model.x_n - 1)


def run_case(title, src, env, params, heights, runs, reference):
    log(f"\n## {title}\n")
    ref, xg, zg, ref_note = reference()
    log(ref_note + "\n")
    log("| solver | dx (m) | steps | rank | FFTs | setup (s) | CPU (s) | wall (s) | error med / p90 / max (dB) |")
    log("|---|---|---|---|---|---|---|---|---|")
    rows = []
    for label, mk in runs:
        model, f, setup, cpu, wall = bench(mk, src)
        rank, n_fft = cost(model)
        st = np.array([db_stats(ref, f, xg, zg, z) for z in heights])
        worst = st.max(axis=0)
        rows.append((label, cpu, worst[1]))
        log(f"| {label} | {model.dx_m:.4g} | {model.x_n - 1} | {rank} | {n_fft} | {setup:.2f} | {cpu:.3f} | {wall:.3f} | "
            f"{worst[0]:.2f} / {worst[1]:.2f} / {worst[2]:.2f} |")
    return rows


def pade_reference(src, env, params):
    def make():
        p = deepcopy(params)
        p.approx_method = 'pade'
        p.rational_approx_order = (7, 8)
        t0c, t0 = time.process_time(), time.perf_counter()
        fd = rwp_forward_task(src, env, p)
        note = (f"Reference: Padé (7, 8) FD, {time.perf_counter() - t0:.0f} s wall / "
                f"{time.process_time() - t0c:.0f} s CPU including compilation.")
        return np.asarray(fd.field), np.asarray(fd.x_grid), np.asarray(fd.z_grid), note
    return make


def exact_reference(src, env, params, **kw):
    def make():
        m = create_rwp_rassf_model(src, env, params, **kw)
        t0 = time.perf_counter()
        ex = exact_one_way_stratified(m, src.aperture(m.z_computational_grid()), m.x_output_grid())
        note = f"Reference: exact one-way solution (eigen-decomposition, {m.z_n - 1} points, {time.perf_counter() - t0:.0f} s)."
        return ex[:, :m.z_n_phys:m.z_grid_scale], np.asarray(m.x_output_grid()), np.asarray(m.z_output_grid()), note
    return make


summary = []

# ------------------------------------------------------------------ 1. wide-angle layer (exact reference)
src = RWPGaussSourceModel(freq_hz=1e9, height_m=50.0, beam_width_deg=20.0)
env = TroposphereModel(N_profile=PiecewiseLinearNProfileModel(
    z_grid_m=jnp.array([0.0, 100.0, 150.0, 400.0]), N_vals=jnp.array([0.0, 0.0, -50.0, -50.0])))
params = RWPComputationalParams(max_range_m=10000.0, max_height_m=300.0, dx_m=250.0, dz_m=1.0)
kw = dict(abs_attenuation_db=0.0, absorber='profile')
rows = run_case("Wide-angle beam (20 deg) through a -50 N-unit layer, 1 GHz, 10 km", src, env, params,
                [30.0, 120.0, 250.0],
                [("SSF, default step", lambda: create_rwp_ssf_model(src, env, params, **kw)),
                 ("SSF, 25 m", lambda: create_rwp_ssf_model(src, env, params, dx_m=25.0, **kw)),
                 ("SSF, 250 m", lambda: create_rwp_ssf_model(src, env, params, dx_m=250.0, **kw)),
                 ("RA-SSF, 25 m", lambda: create_rwp_rassf_model(src, env, params, dx_m=25.0, **kw)),
                 ("RA-SSF, 125 m", lambda: create_rwp_rassf_model(src, env, params, dx_m=125.0, **kw)),
                 ("RA-SSF, 250 m", lambda: create_rwp_rassf_model(src, env, params, dx_m=250.0, **kw))],
                exact_reference(src, env, params, **kw))
summary.append(("wide-angle layer 1 GHz", rows))

# ------------------------------------------------------------------ 2. evaporation duct 3 GHz, 100 km (Padé)
src = RWPGaussSourceModel(freq_hz=3e9, height_m=10.0, beam_width_deg=3.0)
env = TroposphereModel(N_profile=EvaporationDuctModel(height_m=20.0))
params = RWPComputationalParams(max_range_m=100000.0, max_height_m=150.0, dx_m=1000.0, dz_m=0.5)
rows = run_case("Evaporation duct (20 m) at 3 GHz over PEC, 3 deg beam, 100 km", src, env, params,
                [5.0, 10.0, 30.0],
                [("SSF, default step", lambda: create_rwp_ssf_model(src, env, params)),
                 ("SSF, 1000 m, profile absorber", lambda: create_rwp_ssf_model(src, env, params, dx_m=1000.0,
                                                                                absorber='profile')),
                 ("RA-SSF, 1000 m", lambda: create_rwp_rassf_model(src, env, params))],
                pade_reference(src, env, params))
summary.append(("evaporation duct 3 GHz, 100 km", rows))

# ------------------------------------------------------------------ 3. sea water, V-pol, DMFT, 100 km (Padé)
src = RWPGaussSourceModel(freq_hz=3e9, height_m=10.0, beam_width_deg=3.0, polarization='V')
env = TroposphereModel(N_profile=EvaporationDuctModel(height_m=20.0), ground_material=GroundMaterial(eps=80.0, sigma=4.0))
rows = run_case("Evaporation duct (20 m) at 3 GHz over sea water, V-pol (DMFT), 3 deg beam, 100 km", src, env, params,
                [5.0, 10.0, 30.0],
                [("SSF, default step", lambda: create_rwp_ssf_model(src, env, params)),
                 ("SSF, 1000 m, profile absorber", lambda: create_rwp_ssf_model(src, env, params, dx_m=1000.0,
                                                                                absorber='profile')),
                 ("RA-SSF, 1000 m", lambda: create_rwp_rassf_model(src, env, params))],
                pade_reference(src, env, params))
summary.append(("sea water V-pol 3 GHz, 100 km", rows))

# ------------------------------------------------------------------ 4. evaporation duct 10 GHz, 20 km (Padé)
src = RWPGaussSourceModel(freq_hz=10e9, height_m=10.0, beam_width_deg=5.0)
env = TroposphereModel(N_profile=EvaporationDuctModel(height_m=20.0))
params = RWPComputationalParams(max_range_m=20000.0, max_height_m=150.0, dx_m=500.0, dz_m=0.5)
rows = run_case("Evaporation duct (20 m) at 10 GHz over PEC, 5 deg beam, 20 km", src, env, params,
                [5.0, 10.0, 30.0],
                [("SSF, default step", lambda: create_rwp_ssf_model(src, env, params)),
                 ("SSF, 50 m, profile absorber", lambda: create_rwp_ssf_model(src, env, params, dx_m=50.0,
                                                                              absorber='profile')),
                 ("RA-SSF, 50 m", lambda: create_rwp_rassf_model(src, env, params, dx_m=50.0)),
                 ("RA-SSF, 500 m", lambda: create_rwp_rassf_model(src, env, params))],
                pade_reference(src, env, params))
summary.append(("evaporation duct 10 GHz, 20 km", rows))

log("\n## Summary: CPU time relative to the SSF at its default step\n")
log("| case | solver | CPU (s) | speed-up vs SSF default | worst p90 error (dB) |")
log("|---|---|---|---|---|")
for case, rows in summary:
    base = rows[0][1]
    for label, cpu, p90 in rows:
        log(f"| {case} | {label} | {cpu:.3f} | {base / cpu:.1f}x | {p90:.2f} |")

with open(os.path.join(args.out, 'rassf_cpu_time.md'), 'w') as fh:
    fh.write("\n".join(lines) + "\n")
print(f"\nwritten to {args.out}")
