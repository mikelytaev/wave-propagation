"""Compare JAX output at dx=25 vs dx=50 to diagnose refractivity bug."""
import os, sys
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

from pywaveprop.uwa_jax import (
    UWAGaussSourceModel, UnderwaterLayerModel,
    UnderwaterEnvironmentModel, uwa_forward_task, uwa_get_model,
)
from pywaveprop.uwa_utils import UWAComputationalParams
from pywaveprop.helmholtz_jax import PiecewiseLinearWaveSpeedModel, ConstWaveSpeedModel

# Munk profile
def munk_ssp(z):
    z_axis = 1300.0
    B = 1300.0
    eps = 0.00737
    eta = 2.0 * (z - z_axis) / B
    return 1500.0 * (1.0 + eps * (eta + np.exp(-eta) - 1.0))

freq_hz = 50.0
src_depth = 1000.0
rcv_depth = 1000.0
max_range_m = 100_000.0
bottom_depth = 5000.0
c_bottom = 1600.0
rho_bottom = 1.8
attn_bottom = 0.2
dz = 5.0
max_depth_m = bottom_depth + 1000

ssp_z = np.arange(0, bottom_depth + 1, 50.0)
ssp_c = munk_ssp(ssp_z)
water_ssp = PiecewiseLinearWaveSpeedModel(z_grid_m=jnp.array(ssp_z), sound_speed=jnp.array(ssp_c))

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

results = {}
for dx in [50.0, 25.0, 10.0]:
    print(f"\n--- dx = {dx} ---")
    params = UWAComputationalParams(
        max_range_m=max_range_m, max_depth_m=max_depth_m,
        dx_m=dx, dz_m=dz,
    )
    params.rational_approx_order = (7, 8)

    model = uwa_get_model(jax_src, jax_env, params)
    print(f"  beta={model.beta:.6f}, dx_comp={model.dx_m:.2f}, dz={model.dz_m:.2f}")
    print(f"  x_grid_scale={model.x_grid_scale}, grid: {model.x_n} x {model.z_n}")
    print(f"  het[0]={float(model.het[0].real):.6f}, het[-1]={float(model.het[-1].real):.6f}")
    print(f"  het range: [{float(jnp.min(model.het.real)):.4f}, {float(jnp.max(model.het.real)):.4f}]")
    print(f"  beta*dx = {model.beta * model.dx_m:.6f}")

    c0 = float(jnp.real(jax_env.ssp(jnp.array([src_depth]))[0]))
    k0 = 2 * np.pi * freq_hz / c0
    init = jax_src.aperture(k0, model.z_computational_grid())
    f = model.compute(init)

    r = np.asarray(model.x_output_grid())
    z = np.asarray(model.z_output_grid())
    z_idx = np.argmin(np.abs(z - rcv_depth))
    tl = -20 * np.log10(np.abs(np.asarray(f)[:, z_idx]) + 1e-30)

    results[dx] = (r, tl, z, np.asarray(f))

# Compare
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# TL vs range
ax = axes[0]
for dx in [50.0, 25.0, 10.0]:
    r, tl, _, _ = results[dx]
    ax.plot(r / 1000, tl, linewidth=0.6, alpha=0.8, label=f'dx={dx} m')
ax.set_ylabel('TL (dB)')
ax.set_title(f'JAX TL at {rcv_depth:.0f} m — Munk {freq_hz:.0f} Hz — varying dx')
ax.legend(fontsize=9)
ax.invert_yaxis()
ax.grid(True, alpha=0.3)

# Normalised difference vs dx=10 (finest)
ax = axes[1]
r_ref, tl_ref, _, _ = results[10.0]
ref_idx = np.argmin(np.abs(r_ref - 10000))
tl_ref_norm = tl_ref - tl_ref[ref_idx]
for dx in [50.0, 25.0]:
    r, tl, _, _ = results[dx]
    tl_interp = np.interp(r_ref, r, tl)
    tl_norm = tl_interp - tl_interp[ref_idx]
    diff = tl_norm - tl_ref_norm
    mask = r_ref >= 2000
    ax.plot(r_ref[mask] / 1000, diff[mask], linewidth=0.5, alpha=0.8,
            label=f'dx={dx} - dx=10')
ax.axhline(0, color='k', linewidth=0.5)
ax.set_ylabel('Norm. diff vs dx=10 (dB)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(-30, 30)

# 2D TL at dx=50 vs dx=10 difference
ax = axes[2]
r50, tl50, z50, f50 = results[50.0]
r10, tl10, z10, f10 = results[10.0]
tl50_2d = -20 * np.log10(np.abs(f50) + 1e-30)
tl10_2d = -20 * np.log10(np.abs(f10) + 1e-30)

from scipy.interpolate import RegularGridInterpolator
interp10 = RegularGridInterpolator((r10, z10), tl10_2d, method='linear',
                                    bounds_error=False, fill_value=np.nan)
z_water = z50[z50 <= bottom_depth]
rr, zz = np.meshgrid(r50, z_water, indexing='ij')
tl10_on_50 = interp10((rr, zz))
tl50_water = tl50_2d[:, z50 <= bottom_depth]

# Normalise both at r=10km
ref_r = np.argmin(np.abs(r50 - 10000))
diff_2d = (tl50_water - tl50_water[ref_r, :]) - (tl10_on_50 - tl10_on_50[ref_r, :])

from matplotlib.colors import TwoSlopeNorm
im = ax.pcolormesh(r50 / 1000, z_water, diff_2d.T,
                   cmap='RdBu_r', norm=TwoSlopeNorm(vmin=-20, vcenter=0, vmax=20),
                   shading='auto')
ax.set_title('2D norm. diff: dx=50 vs dx=10')
ax.set_xlabel('Range (km)')
ax.set_ylabel('Depth (m)')
ax.invert_yaxis()
plt.colorbar(im, ax=ax, label='dB')

fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), 'debug_dx_dependence.png')
fig.savefig(out, dpi=150)
print(f"\nSaved: {out}")

# Print convergence stats
print(f"\n--- Convergence statistics (r >= 5 km, at rcv depth) ---")
for dx in [50.0, 25.0]:
    r, tl, _, _ = results[dx]
    tl_interp = np.interp(r_ref, r, tl)
    ref_i = np.argmin(np.abs(r_ref - 10000))
    tl_n = tl_interp - tl_interp[ref_i]
    tl_ref_n = tl_ref - tl_ref[ref_i]
    d = np.abs(tl_n - tl_ref_n)[r_ref >= 5000]
    print(f"  dx={dx:4.0f} vs dx=10: median={np.median(d):.2f}, mean={np.mean(d):.2f}, max={np.max(d):.2f} dB")
