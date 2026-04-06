"""Compare JAX vs RAM in deep water with Munk sound speed profile."""
import os, sys, struct, tempfile, subprocess
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np
import math as fm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tests.pywaveprop_jax.test_ram_comparison import RAM_EXE

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp


# ======================================================================
# Munk profile
# ======================================================================
def munk_ssp(z):
    """Munk canonical deep-ocean SSP (m/s)."""
    z_axis = 1300.0   # channel axis depth (m)
    B = 1300.0         # scale depth (m)
    eps = 0.00737      # perturbation
    eta = 2.0 * (z - z_axis) / B
    return 1500.0 * (1.0 + eps * (eta + np.exp(-eta) - 1.0))


# ======================================================================
# Parameters
# ======================================================================
freq_hz = 50.0
src_depth = 1000.0      # near channel axis
rcv_depth = 1000.0
max_range_m = 100_000.0  # 100 km
bottom_depth = 5000.0
c_bottom = 1600.0
rho_bottom = 1.8
attn_bottom = 0.2        # dB/lambda
dr = 50.0                # range step
dz = 5.0                 # depth step
max_depth_m = bottom_depth + 1000  # absorbing layer below

# SSP discretised for RAM input
ssp_z = np.arange(0, bottom_depth + 1, 50.0)
ssp_c = munk_ssp(ssp_z)
c0 = float(ssp_c.min())  # reference speed at channel axis

print(f"Munk SSP: c_min={ssp_c.min():.1f} m/s @ z={ssp_z[np.argmin(ssp_c)]:.0f} m, "
      f"c_max={ssp_c.max():.1f} m/s")
print(f"c0 (reference) = {c0:.1f} m/s")

# ======================================================================
# Write RAM input & run
# ======================================================================
def write_ram_munk(filename, freq_hz, src_depth, rcv_depth,
                   max_range_m, dr, max_depth_m, dz, c0,
                   ssp_z, ssp_c, bottom_depth,
                   bottom_c, bottom_rho, bottom_attn, np_pade=8):
    ndr = 1
    ndz = 1
    with open(filename, 'w') as f:
        f.write("Munk deep water comparison\n")
        f.write(f"{freq_hz} {src_depth} {rcv_depth}\n")
        f.write(f"{max_range_m} {dr} {ndr}\n")
        f.write(f"{max_depth_m} {dz} {ndz} {max_depth_m}\n")
        f.write(f"{c0} {np_pade} 1 0.0\n")
        # Bathymetry (flat)
        f.write(f"0.0 {bottom_depth}\n")
        f.write(f"{max_range_m} {bottom_depth}\n")
        f.write("-1 -1\n")
        # Sound speed in water
        for z, c in zip(ssp_z, ssp_c):
            f.write(f"{z} {c:.4f}\n")
        f.write("-1 -1\n")
        # Sediment sound speed
        f.write(f"0.0 {bottom_c}\n")
        f.write("-1 -1\n")
        # Sediment density
        f.write(f"0.0 {bottom_rho}\n")
        f.write("-1 -1\n")
        # Attenuation
        f.write(f"0.0 {bottom_attn}\n")
        f.write("-1 -1\n")


def parse_tl_line(filename):
    data = np.loadtxt(filename)
    return data[:, 0], data[:, 1]


def parse_tl_grid_v15(filename):
    with open(filename, 'rb') as f:
        data = f.read()
    rec_len = struct.unpack('i', data[0:4])[0]
    header = struct.unpack('5f i 2f i 2f 2i f i', data[4:4+rec_len])
    names = 'freq zs zr rmax dr ndr zmax dz ndz zmplt c0 np ns rs lz'.split()
    hdr = dict(zip(names, header))
    lz = hdr['lz']
    dr_out = hdr['dr'] * hdr['ndr']
    offset = 4 + rec_len + 4
    tl_cols = []
    while offset < len(data):
        rl = struct.unpack('i', data[offset:offset+4])[0]
        offset += 4
        vals = struct.unpack(f'{rl//4}f', data[offset:offset+rl])
        offset += rl + 4
        tl_cols.append(vals)
    tl_grid = np.array(tl_cols)
    ranges = np.arange(1, len(tl_cols)+1) * dr_out
    z_grid = np.arange(1, lz+1) * hdr['dz'] * hdr['ndz']
    return ranges, z_grid, tl_grid, hdr


print("\nRunning RAM...")
tmpdir = tempfile.mkdtemp()
inp = os.path.join(tmpdir, "ram.in")
write_ram_munk(inp, freq_hz, src_depth, rcv_depth,
               max_range_m, dr, max_depth_m, dz, c0,
               ssp_z, ssp_c, bottom_depth,
               c_bottom, rho_bottom, attn_bottom, np_pade=8)
result = subprocess.run([RAM_EXE], cwd=tmpdir, capture_output=True, timeout=600)
if result.returncode != 0:
    print(f"RAM stderr: {result.stderr.decode()}")
    raise RuntimeError("RAM failed")

ram_ranges, ram_tl = parse_tl_line(os.path.join(tmpdir, "tl.line"))
ram_r2d, ram_z2d, ram_tl_grid, ram_hdr = parse_tl_grid_v15(
    os.path.join(tmpdir, "tl.grid"))
print(f"  RAM: {len(ram_ranges)} range points, tl.grid {ram_tl_grid.shape}")

# ======================================================================
# Run old code
# ======================================================================
print("Running old code...")
from pywaveprop.uwa.sspade import uwa_ss_pade, UWASSpadeComputationalParams
from pywaveprop.uwa.source import GaussSource
from pywaveprop.uwa.environment import UnderwaterEnvironment, Bathymetry

old_src = GaussSource(freq_hz=freq_hz, depth_m=src_depth,
                      beam_width_deg=30, elevation_angle_deg=0)
old_env = UnderwaterEnvironment(
    sound_speed_profile_m_s=lambda x, z: munk_ssp(np.atleast_1d(z)).squeeze(),
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
old_tl_line = -20 * np.log10(np.abs(old_field.field[:, old_z_idx]) + 1e-30)
old_ranges = old_field.x_grid
old_tl_2d = -20 * np.log10(np.abs(old_field.field) + 1e-30)
old_z = old_field.z_grid
print(f"  Old: {old_tl_2d.shape[0]} ranges x {old_tl_2d.shape[1]} depths")

# ======================================================================
# Run JAX
# ======================================================================
print("Running JAX...")
from pywaveprop.uwa_jax import (
    UWAGaussSourceModel, UnderwaterLayerModel,
    UnderwaterEnvironmentModel, uwa_forward_task,
)
from pywaveprop.uwa_utils import UWAComputationalParams
from pywaveprop.helmholtz_jax import PiecewiseLinearWaveSpeedModel, ConstWaveSpeedModel

# Munk SSP as piecewise-linear model
ssp_z_jax = jnp.array(ssp_z)
ssp_c_jax = jnp.array(ssp_c)
water_ssp = PiecewiseLinearWaveSpeedModel(z_grid_m=ssp_z_jax, sound_speed=ssp_c_jax)

jax_src = UWAGaussSourceModel(
    freq_hz=freq_hz, depth_m=src_depth,
    beam_width_deg=30, elevation_angle_deg=0, multiplier=1,
)
jax_env = UnderwaterEnvironmentModel(layers=[
    UnderwaterLayerModel(height_m=bottom_depth,
                         sound_speed_profile_m_s=water_ssp, density=1.0),
    UnderwaterLayerModel(height_m=1000,
                         sound_speed_profile_m_s=ConstWaveSpeedModel(c0=c_bottom),
                         density=rho_bottom, attenuation_dm_lambda=attn_bottom),
])
jax_params = UWAComputationalParams(
    max_range_m=max_range_m, max_depth_m=max_depth_m,
    dx_m=dr, dz_m=dz,
)
jax_params.rational_approx_order = (7, 8)
jax_result = uwa_forward_task(jax_src, jax_env, jax_params)

jax_field = np.asarray(jax_result.field)
jax_r = np.asarray(jax_result.x_grid)
jax_z = np.asarray(jax_result.z_grid)

jax_z_idx = np.argmin(np.abs(jax_z - rcv_depth))
jax_tl_line = -20 * np.log10(np.abs(jax_field[:, jax_z_idx]) + 1e-30)
jax_tl_2d = -20 * np.log10(np.abs(jax_field) + 1e-30)

print(f"  JAX: {jax_tl_2d.shape[0]} ranges x {jax_tl_2d.shape[1]} depths")

# ======================================================================
# Interpolate JAX & old code onto RAM grid for 2D comparison
# ======================================================================
from scipy.interpolate import RegularGridInterpolator

z_max_common = min(ram_z2d[-1], jax_z[-1], old_z[-1], bottom_depth)
r_max_common = min(ram_r2d[-1], jax_r[-1], old_ranges[-1])
r_min_common = max(ram_r2d[0], jax_r[1], old_ranges[0])

ram_r_mask = (ram_r2d >= r_min_common) & (ram_r2d <= r_max_common)
ram_z_mask = ram_z2d <= z_max_common
common_r = ram_r2d[ram_r_mask]
common_z = ram_z2d[ram_z_mask]

ram_common = ram_tl_grid[np.ix_(ram_r_mask, ram_z_mask)]

rr, zz = np.meshgrid(common_r, common_z, indexing='ij')

jax_interp = RegularGridInterpolator(
    (jax_r, jax_z), jax_tl_2d, method='linear', bounds_error=False, fill_value=np.nan)
jax_common = jax_interp((rr, zz))

old_interp = RegularGridInterpolator(
    (old_ranges, old_z), old_tl_2d, method='linear', bounds_error=False, fill_value=np.nan)
old_common = old_interp((rr, zz))

# Normalise at 10 km reference
ref_r_idx = np.argmin(np.abs(common_r - 10000))
ram_norm_2d = ram_common - ram_common[ref_r_idx, :][np.newaxis, :]
jax_norm_2d = jax_common - jax_common[ref_r_idx, :][np.newaxis, :]
old_norm_2d = old_common - old_common[ref_r_idx, :][np.newaxis, :]
diff_jax_ram_2d = jax_norm_2d - ram_norm_2d
diff_old_ram_2d = old_norm_2d - ram_norm_2d
diff_jax_old_2d = jax_norm_2d - old_norm_2d

# ======================================================================
# Figure 1: TL vs range at receiver depth
# ======================================================================
fig1, axes = plt.subplots(2, 1, figsize=(14, 7),
                          gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

ax = axes[0]
ax.plot(ram_ranges / 1000, ram_tl, linewidth=0.6, alpha=0.8, label='RAM (point src, np=8)')
ax.plot(old_ranges / 1000, old_tl_line, linewidth=0.6, alpha=0.8, label='Old code (Gauss beam)')
ax.plot(jax_r / 1000, jax_tl_line, linewidth=0.6, alpha=0.8, label='JAX (Gauss beam, Padé (7,8))')
ax.set_ylabel('Transmission Loss (dB)')
ax.set_title(f'Munk deep water — {freq_hz:.0f} Hz, src/rcv @ {rcv_depth:.0f} m, bottom @ {bottom_depth:.0f} m')
ax.legend(fontsize=9)
ax.invert_yaxis()
ax.grid(True, alpha=0.3)

# Normalised differences
ax = axes[1]
ref_idx_line = np.argmin(np.abs(ram_ranges - 10000))
ram_norm_line = ram_tl - ram_tl[ref_idx_line]

jax_on_ram_line = np.interp(ram_ranges, jax_r, jax_tl_line)
jax_norm_line = jax_on_ram_line - jax_on_ram_line[ref_idx_line]

old_on_ram_line = np.interp(ram_ranges, old_ranges, old_tl_line)
old_norm_line = old_on_ram_line - old_on_ram_line[ref_idx_line]

nd_jax = jax_norm_line - ram_norm_line
nd_old = old_norm_line - ram_norm_line

mask_line = ram_ranges >= 2000
ax.plot(ram_ranges[mask_line] / 1000, nd_old[mask_line],
        linewidth=0.5, alpha=0.8, color='C1', label='Old - RAM')
ax.plot(ram_ranges[mask_line] / 1000, nd_jax[mask_line],
        linewidth=0.5, alpha=0.8, color='C2', label='JAX - RAM')
ax.axhline(0, color='k', linewidth=0.5)
ax.set_ylabel('Norm. diff (dB)')
ax.set_xlabel('Range (km)')
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_ylim(-30, 30)

fig1.tight_layout()
out1 = os.path.join(os.path.dirname(__file__), 'munk_tl_vs_range.png')
fig1.savefig(out1, dpi=150)
print(f"Saved: {out1}")

# ======================================================================
# Figure 2: 2D TL fields (3 solvers) + normalised differences
# ======================================================================
fig2, axes2 = plt.subplots(2, 3, figsize=(20, 10), sharex=True, sharey=True)
tl_vmin, tl_vmax = 50, 110
vlim = 20
norm_div = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)

# Top row: TL fields
ax = axes2[0, 0]
im = ax.pcolormesh(common_r / 1000, common_z, ram_common.T,
                   cmap='viridis_r', vmin=tl_vmin, vmax=tl_vmax, shading='auto')
ax.set_title('RAM (point source, np=8)')
ax.set_ylabel('Depth (m)')
ax.invert_yaxis()
plt.colorbar(im, ax=ax, label='TL (dB)')

ax = axes2[0, 1]
im = ax.pcolormesh(common_r / 1000, common_z, old_common.T,
                   cmap='viridis_r', vmin=tl_vmin, vmax=tl_vmax, shading='auto')
ax.set_title('Old code (Gauss beam)')
ax.invert_yaxis()
plt.colorbar(im, ax=ax, label='TL (dB)')

ax = axes2[0, 2]
im = ax.pcolormesh(common_r / 1000, common_z, jax_common.T,
                   cmap='viridis_r', vmin=tl_vmin, vmax=tl_vmax, shading='auto')
ax.set_title('JAX (Gauss beam, Padé (7,8))')
ax.invert_yaxis()
plt.colorbar(im, ax=ax, label='TL (dB)')

# Bottom row: normalised differences
ax = axes2[1, 0]
im = ax.pcolormesh(common_r / 1000, common_z, diff_old_ram_2d.T,
                   cmap='RdBu_r', norm=norm_div, shading='auto')
ax.set_title('Old - RAM (normalised)')
ax.set_xlabel('Range (km)')
ax.set_ylabel('Depth (m)')
ax.invert_yaxis()
plt.colorbar(im, ax=ax, label='dB')

ax = axes2[1, 1]
im = ax.pcolormesh(common_r / 1000, common_z, diff_jax_ram_2d.T,
                   cmap='RdBu_r', norm=norm_div, shading='auto')
ax.set_title('JAX - RAM (normalised)')
ax.set_xlabel('Range (km)')
ax.invert_yaxis()
plt.colorbar(im, ax=ax, label='dB')

ax = axes2[1, 2]
im = ax.pcolormesh(common_r / 1000, common_z, diff_jax_old_2d.T,
                   cmap='RdBu_r', norm=norm_div, shading='auto')
ax.set_title('JAX - Old (normalised)')
ax.set_xlabel('Range (km)')
ax.invert_yaxis()
plt.colorbar(im, ax=ax, label='dB')

for ax in axes2.flat:
    ax.axhline(bottom_depth, color='brown', linewidth=1.5, linestyle='--', alpha=0.7)

fig2.suptitle(f'Munk deep water — {freq_hz:.0f} Hz, bottom @ {bottom_depth:.0f} m',
              fontsize=14, fontweight='bold')
fig2.tight_layout()

out2 = os.path.join(os.path.dirname(__file__), 'munk_2d_comparison.png')
fig2.savefig(out2, dpi=150)
print(f"Saved: {out2}")

# ======================================================================
# Statistics
# ======================================================================
stat_r_mask = common_r >= 5000
line_mask = ram_ranges >= 5000

print(f"\n{'='*70}")
print(f"Statistics (water column, r >= 5 km)")
print(f"{'='*70}")

for label, diff_2d, diff_1d in [
    ("Old - RAM",  diff_old_ram_2d, nd_old),
    ("JAX - RAM",  diff_jax_ram_2d, nd_jax),
    ("JAX - Old",  diff_jax_old_2d, jax_norm_line - old_norm_line),
]:
    d = diff_2d[stat_r_mask, :]
    d1 = diff_1d[line_mask]
    print(f"\n  {label} (normalised):")
    print(f"    2D field:  median |diff| = {np.nanmedian(np.abs(d)):.2f} dB, "
          f"mean = {np.nanmean(np.abs(d)):.2f} dB, "
          f"P90 = {np.nanpercentile(np.abs(d), 90):.2f} dB, "
          f"max = {np.nanmax(np.abs(d)):.2f} dB")
    print(f"    At {rcv_depth:.0f} m: median |diff| = {np.median(np.abs(d1)):.2f} dB, "
          f"mean = {np.mean(np.abs(d1)):.2f} dB, "
          f"max = {np.max(np.abs(d1)):.2f} dB")
