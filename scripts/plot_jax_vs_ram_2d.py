"""Plot 2D TL fields from JAX and RAM, plus their pointwise difference."""
import os, sys, struct, tempfile, subprocess
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tests.pywaveprop_jax.test_ram_comparison import (
    write_ram_input, RAM_EXE,
)


def parse_tl_grid_v15(filename):
    """Parse RAM 1.5 tl.grid: header record then lz-float TL columns."""
    with open(filename, 'rb') as f:
        data = f.read()
    # Header record: 15 mixed int/real values
    rec_len = struct.unpack('i', data[0:4])[0]
    header = struct.unpack('5f i 2f i 2f 2i f i', data[4:4+rec_len])
    names = 'freq zs zr rmax dr ndr zmax dz ndz zmplt c0 np ns rs lz'.split()
    hdr = dict(zip(names, header))
    lz = hdr['lz']
    dr_out = hdr['dr'] * hdr['ndr']
    offset = 4 + rec_len + 4  # skip header + trailing length

    # Each subsequent record: lz floats (TL column at one range)
    tl_cols = []
    while offset < len(data):
        rl = struct.unpack('i', data[offset:offset+4])[0]
        offset += 4
        vals = struct.unpack(f'{rl//4}f', data[offset:offset+rl])
        offset += rl + 4
        tl_cols.append(vals)

    tl_grid = np.array(tl_cols)  # (n_range, lz)
    ranges = np.arange(1, len(tl_cols)+1) * dr_out
    z_grid = np.arange(1, lz+1) * hdr['dz'] * hdr['ndz']
    return ranges, z_grid, tl_grid, hdr

import jax
jax.config.update('jax_enable_x64', True)

# --- Parameters ---
freq_hz = 100.0
src_depth = 50.0
max_range_m = 5000.0
bottom_depth = 300.0
c_water = 1500.0
c_bottom = 1700.0
rho_bottom = 1.5
attn_bottom = 0.5
dr = 5.0
dz = 1.0
max_depth_m = bottom_depth + 500

# ======================================================================
# Run RAM — keep tl.grid for 2D field
# ======================================================================
print("Running RAM...")
tmpdir = tempfile.mkdtemp()
inp = os.path.join(tmpdir, "ram.in")
write_ram_input(
    inp, freq_hz, src_depth, src_depth,
    max_range_m, dr, max_depth_m, dz,
    c0=c_water, ssp_z=[0, bottom_depth], ssp_c=[c_water, c_water],
    bottom_c=c_bottom, bottom_rho=rho_bottom, bottom_attn=attn_bottom,
    bottom_depth=bottom_depth, np_pade=6,
)
result = subprocess.run([RAM_EXE], cwd=tmpdir, capture_output=True, timeout=120)
if result.returncode != 0:
    raise RuntimeError(f"RAM failed: {result.stderr.decode()}")

ram_ranges_2d, ram_z, ram_tl_grid, ram_hdr = parse_tl_grid_v15(
    os.path.join(tmpdir, "tl.grid"))

print(f"  RAM grid: {ram_tl_grid.shape[0]} ranges x {ram_tl_grid.shape[1]} depths")
print(f"  RAM range: {ram_ranges_2d[0]:.0f} – {ram_ranges_2d[-1]:.0f} m")
print(f"  RAM depth: {ram_z[0]:.0f} – {ram_z[-1]:.0f} m")
print(f"  RAM header: lz={ram_hdr['lz']}, np={ram_hdr['np']}")

# ======================================================================
# Run JAX
# ======================================================================
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

jax_field = np.asarray(jax_result.field)          # (n_range, n_depth)
jax_tl_2d = -20 * np.log10(np.abs(jax_field) + 1e-30)
jax_r = np.asarray(jax_result.x_grid)
jax_z = np.asarray(jax_result.z_grid)

print(f"  JAX grid: {jax_tl_2d.shape[0]} ranges x {jax_tl_2d.shape[1]} depths")
print(f"  JAX range: {jax_r[0]:.0f} – {jax_r[-1]:.0f} m")
print(f"  JAX depth: {jax_z[0]:.0f} – {jax_z[-1]:.0f} m")

# ======================================================================
# Interpolate JAX onto RAM grid for pointwise comparison
# ======================================================================
print("Interpolating JAX onto RAM grid...")

from scipy.interpolate import RegularGridInterpolator

# Clip to common domain
z_max_common = min(ram_z[-1], jax_z[-1])
r_max_common = min(ram_ranges_2d[-1], jax_r[-1])
r_min_common = max(ram_ranges_2d[0], jax_r[0])

ram_z_mask = ram_z <= z_max_common
ram_r_mask = (ram_ranges_2d >= r_min_common) & (ram_ranges_2d <= r_max_common)

common_r = ram_ranges_2d[ram_r_mask]
common_z = ram_z[ram_z_mask]

# RAM on common grid
ram_common = ram_tl_grid[np.ix_(ram_r_mask, ram_z_mask)]

# Interpolate JAX TL onto common grid
jax_interp = RegularGridInterpolator(
    (jax_r, jax_z), jax_tl_2d, method='linear', bounds_error=False, fill_value=np.nan,
)
rr, zz = np.meshgrid(common_r, common_z, indexing='ij')
jax_common = jax_interp((rr, zz))

# Normalise: subtract TL at reference range (1000 m) for each depth
ref_r_idx = np.argmin(np.abs(common_r - 1000))
ram_norm = ram_common - ram_common[ref_r_idx, :][np.newaxis, :]
jax_norm = jax_common - jax_common[ref_r_idx, :][np.newaxis, :]
diff_norm = jax_norm - ram_norm

# Raw difference (without normalisation)
diff_raw = jax_common - ram_common

# Restrict depth to water column for plots
z_water_mask = common_z <= bottom_depth
z_plot = common_z[z_water_mask]
r_plot = common_r

ram_plot = ram_common[:, z_water_mask]
jax_plot = jax_common[:, z_water_mask]
diff_raw_plot = diff_raw[:, z_water_mask]
diff_norm_plot = diff_norm[:, z_water_mask]

print(f"  Common grid: {len(r_plot)} ranges x {len(z_plot)} depths (water column)")

# ======================================================================
# Plot
# ======================================================================
tl_vmin, tl_vmax = 30, 90

fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey=True)

# RAM TL
ax = axes[0, 0]
im = ax.pcolormesh(r_plot / 1000, z_plot, ram_plot.T,
                   cmap='viridis_r', vmin=tl_vmin, vmax=tl_vmax, shading='auto')
ax.set_title(f'RAM TL (np=6, point source)')
ax.set_ylabel('Depth (m)')
ax.invert_yaxis()
plt.colorbar(im, ax=ax, label='TL (dB)')

# JAX TL
ax = axes[0, 1]
im = ax.pcolormesh(r_plot / 1000, z_plot, jax_plot.T,
                   cmap='viridis_r', vmin=tl_vmin, vmax=tl_vmax, shading='auto')
ax.set_title(f'JAX TL (Padé (7,8), Gaussian beam)')
ax.invert_yaxis()
plt.colorbar(im, ax=ax, label='TL (dB)')

# Raw difference
ax = axes[1, 0]
vlim = 20
norm_div = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)
im = ax.pcolormesh(r_plot / 1000, z_plot, diff_raw_plot.T,
                   cmap='RdBu_r', norm=norm_div, shading='auto')
ax.set_title('Raw difference (JAX - RAM)')
ax.set_xlabel('Range (km)')
ax.set_ylabel('Depth (m)')
ax.invert_yaxis()
plt.colorbar(im, ax=ax, label='dB')

# Normalised difference
ax = axes[1, 1]
im = ax.pcolormesh(r_plot / 1000, z_plot, diff_norm_plot.T,
                   cmap='RdBu_r', norm=norm_div, shading='auto')
ax.set_title('Normalised difference (ref @ 1 km)')
ax.set_xlabel('Range (km)')
ax.invert_yaxis()
plt.colorbar(im, ax=ax, label='dB')

for ax in axes.flat:
    ax.axhline(bottom_depth, color='brown', linewidth=1.5, linestyle='--', alpha=0.7)

fig.suptitle(f'JAX vs RAM comparison — {freq_hz:.0f} Hz, isovelocity, bottom @ {bottom_depth:.0f} m',
             fontsize=14, fontweight='bold')
fig.tight_layout()

out_path = os.path.join(os.path.dirname(__file__), 'jax_vs_ram_2d.png')
fig.savefig(out_path, dpi=150)
print(f"\nSaved: {out_path}")

# ======================================================================
# Statistics on 2D difference
# ======================================================================
print("\n--- 2D pointwise statistics (water column, r >= 500 m) ---")
r_mask_2d = r_plot >= 500
d = diff_norm_plot[r_mask_2d, :]
dr_raw = diff_raw_plot[r_mask_2d, :]

print(f"  Raw diff (JAX - RAM):")
print(f"    mean: {np.nanmean(dr_raw):+.2f} dB")
print(f"    median: {np.nanmedian(dr_raw):+.2f} dB")
print(f"    std: {np.nanstd(dr_raw):.2f} dB")

print(f"  Normalised diff:")
print(f"    mean |diff|: {np.nanmean(np.abs(d)):.2f} dB")
print(f"    median |diff|: {np.nanmedian(np.abs(d)):.2f} dB")
print(f"    P90 |diff|: {np.nanpercentile(np.abs(d), 90):.2f} dB")
print(f"    P99 |diff|: {np.nanpercentile(np.abs(d), 99):.2f} dB")
print(f"    max |diff|: {np.nanmax(np.abs(d)):.2f} dB")

# Depth profile of mean absolute error
print(f"\n--- Mean |normalised diff| by depth (r >= 500 m) ---")
print(f"  {'Depth (m)':>10s}  {'mean |diff|':>11s}  {'median |diff|':>13s}")
depth_bins = np.arange(0, bottom_depth + 1, 50)
for i in range(len(depth_bins) - 1):
    z_lo, z_hi = depth_bins[i], depth_bins[i + 1]
    zmask = (z_plot >= z_lo) & (z_plot < z_hi)
    if zmask.sum() == 0:
        continue
    vals = np.abs(d[:, zmask])
    print(f"  {z_lo:>5.0f}-{z_hi:<5.0f}  {np.nanmean(vals):11.2f}  {np.nanmedian(vals):13.2f}")
