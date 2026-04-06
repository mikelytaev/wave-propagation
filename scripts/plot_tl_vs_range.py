"""Plot TL vs range at source depth for JAX, old code, and RAM."""
import os, sys, struct, tempfile, subprocess
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
old_tl = -20 * np.log10(np.abs(old_field.field[:, old_z_idx]) + 1e-30)
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

# ======================================================================
# Plot
# ======================================================================
fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]},
                         sharex=True)

# --- TL vs range ---
ax = axes[0]
ax.plot(ram_ranges / 1000, ram_tl, linewidth=0.8, alpha=0.85, label='RAM (point source, np=6)')
ax.plot(old_ranges / 1000, old_tl, linewidth=0.8, alpha=0.85, label='Old code (Gauss beam)')
ax.plot(jax_ranges / 1000, jax_tl, linewidth=0.8, alpha=0.85, label='JAX (Gauss beam, Padé (7,8))')
ax.set_ylabel('Transmission Loss (dB)')
ax.set_title(f'TL at source/receiver depth = {rcv_depth:.0f} m  —  '
             f'{freq_hz:.0f} Hz, isovelocity, bottom @ {bottom_depth:.0f} m')
ax.legend(loc='upper left', fontsize=9)
ax.invert_yaxis()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, max_range_m / 1000)

# --- Normalised difference vs range ---
ax = axes[1]

# Interpolate onto common grid (RAM ranges)
jax_on_ram = np.interp(ram_ranges, jax_ranges, jax_tl)
old_on_ram = np.interp(ram_ranges, old_ranges, old_tl)

ref_idx = np.argmin(np.abs(ram_ranges - 1000))
ram_norm = ram_tl - ram_tl[ref_idx]
jax_norm = jax_on_ram - jax_on_ram[ref_idx]
old_norm = old_on_ram - old_on_ram[ref_idx]

mask = ram_ranges >= 200
ax.plot(ram_ranges[mask] / 1000, (old_norm - ram_norm)[mask],
        linewidth=0.7, alpha=0.7, label='Old - RAM')
ax.plot(ram_ranges[mask] / 1000, (jax_norm - ram_norm)[mask],
        linewidth=0.7, alpha=0.7, label='JAX - RAM')
ax.axhline(0, color='k', linewidth=0.5)
ax.set_ylabel('Normalised diff (dB)')
ax.set_xlabel('Range (km)')
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(-30, 30)

fig.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), 'tl_vs_range.png')
fig.savefig(out_path, dpi=150)
print(f"\nSaved: {out_path}")
