"""Pointwise analysis of JAX vs RAM transmission loss differences.

Investigates sources of divergence:
1. Source model (Gaussian beam vs point source self-starter)
2. Null position alignment (interference pattern shift)
3. Breakdown by TL magnitude (are errors concentrated at nulls?)
4. Grid parameters (dx, Padé order, reference wavenumber)
"""
import os, sys, tempfile
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np
import math as fm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tests.pywaveprop_jax.test_ram_comparison import write_ram_input, run_ram

import jax
jax.config.update('jax_enable_x64', True)

# --- Parameters ---
freq_hz = 100.0
src_depth = 50.0
rcv_depth = 50.0
max_range_m = 5000.0
bottom_depth = 300.0
c_water = 1500.0
c_bottom = 1700.0
rho_bottom = 1.5
attn_bottom = 0.5
dr = 5.0
dz = 1.0
max_depth_m = bottom_depth + 500

# --- Run RAM ---
print("Running RAM...")
with tempfile.TemporaryDirectory() as tmpdir:
    inp = os.path.join(tmpdir, "ram.in")
    write_ram_input(
        inp, freq_hz, src_depth, rcv_depth,
        max_range_m, dr, max_depth_m, dz,
        c0=c_water, ssp_z=[0, bottom_depth], ssp_c=[c_water, c_water],
        bottom_c=c_bottom, bottom_rho=rho_bottom, bottom_attn=attn_bottom,
        bottom_depth=bottom_depth, np_pade=6,
    )
    ram_ranges, ram_tl = run_ram(inp, tmpdir)

# --- Run old code ---
print("Running old code...")
from pywaveprop.uwa.sspade import uwa_ss_pade, UWASSpadeComputationalParams
from pywaveprop.uwa.source import GaussSource
from pywaveprop.uwa.environment import UnderwaterEnvironment, Bathymetry

old_src = GaussSource(freq_hz=freq_hz, depth_m=src_depth,
                      beam_width_deg=30, elevation_angle_deg=0)
old_env = UnderwaterEnvironment(
    sound_speed_profile_m_s=lambda x, z: c_water,
    bottom_profile=Bathymetry(ranges_m=[0], depths_m=[bottom_depth]),
    bottom_sound_speed_m_s=c_bottom,
    bottom_density_g_cm=rho_bottom,
    bottom_attenuation_dm_lambda=attn_bottom,
)
old_params = UWASSpadeComputationalParams(
    max_range_m=max_range_m, max_depth_m=max_depth_m,
    dx_m=dr, dz_m=dz,
)
old_field = uwa_ss_pade(old_src, old_env, old_params)

old_z_idx = np.argmin(np.abs(old_field.z_grid - rcv_depth))
old_tl_full = -20 * np.log10(np.abs(old_field.field[:, old_z_idx]) + 1e-30)
old_ranges = old_field.x_grid

# --- Run JAX ---
print("Running JAX...")
import jax.numpy as jnp
from pywaveprop.uwa_jax import (
    UWAGaussSourceModel, UnderwaterLayerModel,
    UnderwaterEnvironmentModel, uwa_forward_task,
)
from pywaveprop.uwa_utils import UWAComputationalParams
from pywaveprop.helmholtz_jax import ConstWaveSpeedModel

jax_src = UWAGaussSourceModel(
    freq_hz=freq_hz, depth_m=src_depth,
    beam_width_deg=30, elevation_angle_deg=0, multiplier=1,
)
jax_env = UnderwaterEnvironmentModel(layers=[
    UnderwaterLayerModel(height_m=bottom_depth,
                         sound_speed_profile_m_s=ConstWaveSpeedModel(c0=c_water), density=1.0),
    UnderwaterLayerModel(height_m=800,
                         sound_speed_profile_m_s=ConstWaveSpeedModel(c0=c_bottom),
                         density=rho_bottom, attenuation_dm_lambda=attn_bottom),
])
jax_params = UWAComputationalParams(
    max_range_m=max_range_m, max_depth_m=max_depth_m,
    dx_m=dr, dz_m=dz,
)
jax_params.rational_approx_order = (7, 8)
jax_result = uwa_forward_task(jax_src, jax_env, jax_params)

jax_z_idx = np.argmin(np.abs(np.asarray(jax_result.z_grid) - rcv_depth))
jax_tl = -20 * np.log10(np.abs(np.asarray(jax_result.field)[:, jax_z_idx]) + 1e-30)
jax_ranges = np.asarray(jax_result.x_grid)

# --- Interpolate all onto RAM range grid ---
jax_on_ram = np.interp(ram_ranges, jax_ranges, jax_tl)
old_on_ram = np.interp(ram_ranges, old_ranges, old_tl_full)

# Normalise to TL at 1000 m
ref_idx = np.argmin(np.abs(ram_ranges - 1000))
ram_norm = ram_tl - ram_tl[ref_idx]
jax_norm = jax_on_ram - jax_on_ram[ref_idx]
old_norm = old_on_ram - old_on_ram[ref_idx]

# Analysis range: skip near-source
mask = ram_ranges >= 500
r = ram_ranges[mask]
jn = jax_norm[mask]
rn = ram_norm[mask]
on = old_norm[mask]
nd = jn - rn  # normalised diff JAX-RAM

print()
print("=" * 70)
print("DIAGNOSTIC ANALYSIS: JAX vs RAM")
print("=" * 70)

# ==========================================================================
# 1. Grid & algorithm parameters
# ==========================================================================
print("\n--- 1. Grid & algorithm parameters ---")
omega = 2 * fm.pi * freq_hz
k0_ram = omega / c_water
k0_jax = k0_ram  # same c0
print(f"  freq: {freq_hz} Hz,  k0 = {k0_ram:.6f} rad/m,  lambda = {c_water/freq_hz:.1f} m")
print(f"  RAM:  dr={dr} m, dz={dz} m, np_pade=6 (12 terms), point-source self-starter")
print(f"  JAX:  dx_output={dr} m, dx_computational=2.5 m (grid optimiser), dz={dz} m")
print(f"        rational_approx_order=(7,8), Gaussian beam (30 deg, src@{src_depth} m)")
print(f"  Old:  dx={dr} m, dz={dz} m, auto-optimised Padé order, Gaussian beam")

# ==========================================================================
# 2. Null detection & classification
# ==========================================================================
print("\n--- 2. Interference null analysis ---")

# Find nulls: local maxima in TL (i.e. minima in field amplitude)
def find_nulls(tl, ranges, threshold_above_neighbours=3.0):
    """Find indices where TL has local maxima (field nulls)."""
    nulls = []
    for i in range(2, len(tl) - 2):
        if (tl[i] > tl[i-1] and tl[i] > tl[i+1] and
            tl[i] - min(tl[i-2:i+3]) > threshold_above_neighbours):
            nulls.append(i)
    return np.array(nulls)

ram_nulls_g = find_nulls(ram_tl[mask], r)
jax_nulls_g = find_nulls(jax_on_ram[mask], r)

print(f"  RAM nulls found: {len(ram_nulls_g)}")
print(f"  JAX nulls found: {len(jax_nulls_g)}")

# Match nulls between RAM and JAX (nearest-neighbour)
if len(ram_nulls_g) > 0 and len(jax_nulls_g) > 0:
    matched = []
    for ri in ram_nulls_g:
        dists = np.abs(r[jax_nulls_g] - r[ri])
        ji = jax_nulls_g[np.argmin(dists)]
        shift = r[ji] - r[ri]
        if abs(shift) < 200:  # only match within 200 m
            matched.append((ri, ji, shift))

    shifts = [m[2] for m in matched]
    print(f"  Matched null pairs: {len(matched)}")
    if shifts:
        print(f"  Null position shifts (JAX - RAM):")
        print(f"    mean: {np.mean(shifts):+.1f} m")
        print(f"    std:  {np.std(shifts):.1f} m")
        print(f"    range: {min(shifts):+.1f} to {max(shifts):+.1f} m")
        print(f"  Detailed null matches:")
        print(f"    {'RAM null (m)':>12s}  {'JAX null (m)':>12s}  {'shift (m)':>10s}")
        for ri, ji, sh in matched[:15]:
            print(f"    {r[ri]:12.0f}  {r[ji]:12.0f}  {sh:+10.1f}")

# ==========================================================================
# 3. Error breakdown: at nulls vs away from nulls
# ==========================================================================
print("\n--- 3. Errors at nulls vs smooth regions ---")

# Define "near null" as within 30 m of any RAM null
near_null = np.zeros(len(r), dtype=bool)
for ni in ram_nulls_g:
    near_null |= (np.abs(np.arange(len(r)) - ni) <= 6)  # 6 points * 5 m = 30 m

abs_nd = np.abs(nd)
print(f"  Points near RAM nulls: {near_null.sum()} ({100*near_null.sum()/len(r):.1f}%)")
print(f"  Points in smooth regions: {(~near_null).sum()} ({100*(~near_null).sum()/len(r):.1f}%)")
if near_null.any():
    print(f"  |diff| near nulls:   median {np.median(abs_nd[near_null]):.2f}, "
          f"mean {np.mean(abs_nd[near_null]):.2f}, max {np.max(abs_nd[near_null]):.2f} dB")
if (~near_null).any():
    print(f"  |diff| smooth:       median {np.median(abs_nd[~near_null]):.2f}, "
          f"mean {np.mean(abs_nd[~near_null]):.2f}, max {np.max(abs_nd[~near_null]):.2f} dB")

# ==========================================================================
# 4. TL-magnitude-binned errors
# ==========================================================================
print("\n--- 4. Errors binned by RAM TL magnitude ---")
print(f"  {'RAM TL range':>20s}  {'N':>5s}  {'median |diff|':>13s}  {'mean |diff|':>11s}  {'max |diff|':>10s}")

ram_tl_m = ram_tl[mask]
tl_bins = [(20, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 100)]
for lo, hi in tl_bins:
    bmask = (ram_tl_m >= lo) & (ram_tl_m < hi)
    if bmask.sum() == 0:
        continue
    bd = abs_nd[bmask]
    print(f"  {lo:>8.0f} – {hi:<8.0f}  {bmask.sum():5d}  {np.median(bd):13.2f}  "
          f"{np.mean(bd):11.2f}  {np.max(bd):10.2f}")

# ==========================================================================
# 5. Old code as intermediary
# ==========================================================================
print("\n--- 5. Triangle inequality: JAX→Old→RAM ---")
od = on - rn  # old - ram (normalised)
jod = jn - on  # jax - old (normalised)

print(f"  Old vs RAM (normalised): median |diff| = {np.median(np.abs(od)):.2f} dB")
print(f"  JAX vs Old (normalised): median |diff| = {np.median(np.abs(jod)):.2f} dB")
print(f"  JAX vs RAM (normalised): median |diff| = {np.median(abs_nd):.2f} dB")

# Correlation: does jax-old error correlate with old-ram error?
corr = np.corrcoef(np.abs(jod), np.abs(od))[0, 1]
print(f"  Correlation(|JAX-Old|, |Old-RAM|): {corr:.3f}")

# Same-sign analysis
same_sign = np.sign(jod) == np.sign(od)
print(f"  JAX-Old and Old-RAM same sign: {same_sign.sum()}/{len(same_sign)} "
      f"({100*same_sign.mean():.0f}%) — {'errors compound' if same_sign.mean() > 0.6 else 'errors partially cancel'}")

# ==========================================================================
# 6. Smooth trend (low-pass filtered difference)
# ==========================================================================
print("\n--- 6. Smooth trend (moving average, window=100 m) ---")
win = max(1, int(100 / dr))  # 100 m window
if len(nd) > win:
    smooth = np.convolve(nd, np.ones(win)/win, mode='valid')
    sr = r[win//2:win//2+len(smooth)]
    print(f"  Smoothed diff range: {np.min(smooth):+.2f} to {np.max(smooth):+.2f} dB")
    print(f"  Smoothed diff std: {np.std(smooth):.2f} dB")
    # Sample at a few ranges
    print(f"  Smoothed diff at key ranges:")
    for target in [500, 1000, 2000, 3000, 4000, 5000]:
        idx = np.argmin(np.abs(sr - target))
        if idx < len(smooth):
            print(f"    r={sr[idx]:5.0f} m: {smooth[idx]:+.2f} dB")

# ==========================================================================
# 7. Summary diagnosis
# ==========================================================================
print("\n" + "=" * 70)
print("DIAGNOSIS SUMMARY")
print("=" * 70)
smooth_median = np.median(abs_nd[~near_null]) if (~near_null).any() else np.nan
null_median = np.median(abs_nd[near_null]) if near_null.any() else np.nan
print(f"""
  Raw TL offset (JAX - RAM):  ~{np.median(jax_on_ram - ram_tl):.0f} dB
    → Due to source model difference (Gaussian beam vs point source)
    → This is expected and removed by normalisation

  Normalised errors:
    Smooth regions: {smooth_median:.1f} dB median — {'GOOD' if smooth_median < 3 else 'MODERATE' if smooth_median < 6 else 'POOR'}
    Near nulls:     {null_median:.1f} dB median — {'expected' if null_median > 5 else 'good'}
    Overall:        {np.median(abs_nd):.1f} dB median

  Null position shifts: mean {np.mean(shifts):+.1f} m, std {np.std(shifts):.1f} m
    → {'Significant' if np.std(shifts) > 20 else 'Small'} — interference patterns {'diverge' if np.std(shifts) > 20 else 'track well'}

  Error compounding: JAX-Old and Old-RAM errors {'compound' if same_sign.mean() > 0.6 else 'partially cancel'} ({100*same_sign.mean():.0f}% same sign)
""")
