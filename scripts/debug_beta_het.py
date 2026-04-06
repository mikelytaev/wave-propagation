"""Check if different beta values with same dx produce different results.
This isolates whether beta is correctly a gauge parameter."""
import os, sys
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np
import math as fm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

from pywaveprop.uwa_jax import (
    UWAGaussSourceModel, UnderwaterLayerModel,
    UnderwaterEnvironmentModel, uwa_get_model,
)
from pywaveprop.uwa_utils import UWAComputationalParams
from pywaveprop.helmholtz_jax import (
    PiecewiseLinearWaveSpeedModel, ConstWaveSpeedModel,
    RationalHelmholtzPropagator, HelmholtzMeshParams2D,
)
from pywaveprop.propagators._utils import pade_propagator_coefs

# Munk profile
def munk_ssp(z):
    z_axis = 1300.0; B = 1300.0; eps = 0.00737
    eta = 2.0 * (z - z_axis) / B
    return 1500.0 * (1.0 + eps * (eta + np.exp(-eta) - 1.0))

# Shallow water isovelocity for simpler test first
freq_hz = 100.0
src_depth = 50.0
bottom_depth = 300.0
c_water = 1500.0
c_bottom = 1700.0
rho_bottom = 1.5
attn_bottom = 0.5
max_range_m = 5000.0
max_depth_m = 800.0

jax_src = UWAGaussSourceModel(
    freq_hz=freq_hz, depth_m=src_depth,
    beam_width_deg=30, elevation_angle_deg=0, multiplier=1,
)
jax_env = UnderwaterEnvironmentModel(layers=[
    UnderwaterLayerModel(height_m=bottom_depth,
                         sound_speed_profile_m_s=ConstWaveSpeedModel(c0=c_water), density=1.0),
    UnderwaterLayerModel(height_m=500,
                         sound_speed_profile_m_s=ConstWaveSpeedModel(c0=c_bottom),
                         density=rho_bottom, attenuation_dm_lambda=attn_bottom),
])

from pywaveprop.uwa_jax import ProxyWaveSpeedModel, ProxyRhoModel, minmax_k

k_bounds = minmax_k(jax_src, jax_env)
k0 = 2 * fm.pi * freq_hz / c_water
kz_max = k0 * fm.sin(fm.radians(jax_src.max_angle_deg()))

print(f"k_bounds: {k_bounds}")
print(f"k0: {k0:.6f}, kz_max: {kz_max:.6f}")

# Run with different beta values but same dx and dz
dx = 5.0
dz = 1.0
results = {}

for beta_val in [kz_max, k0, (k_bounds[0]+k_bounds[1])/2, 0.15, 0.25, 0.35]:
    print(f"\n--- beta = {beta_val:.6f} ---")

    # Compute Padé coefficients for this beta
    coefs = pade_propagator_coefs(pade_order=(7, 8), beta=beta_val, dx=dx)[0]
    coefs_list = [list(v) for v in coefs]

    x_n = int(max_range_m / dx) + 1
    z_n = int(max_depth_m / dz) + 1

    model = RationalHelmholtzPropagator(
        order=(7, 8),
        beta=beta_val,
        dx_m=dx,
        dz_m=dz,
        x_n=x_n,
        z_n=z_n,
        x_grid_scale=1,
        z_grid_scale=1,
        freq_hz=freq_hz,
        wave_speed=ProxyWaveSpeedModel(jax_env),
        rho=ProxyRhoModel(jax_env),
        coefs=coefs_list,
    )

    print(f"  beta*dx = {beta_val * dx:.4f}")
    print(f"  het range: [{float(jnp.min(model.het.real)):.6f}, {float(jnp.max(model.het.real)):.6f}]")

    init = jax_src.aperture(k0, model.z_computational_grid())
    f = model.compute(init)

    z = np.asarray(model.z_output_grid())
    z_idx = np.argmin(np.abs(z - src_depth))
    tl = -20 * np.log10(np.abs(np.asarray(f)[:, z_idx]) + 1e-30)
    r = np.asarray(model.x_output_grid())

    results[beta_val] = (r, tl)
    print(f"  TL at 1km: {tl[np.argmin(np.abs(r-1000))]:.2f} dB")
    print(f"  TL at 3km: {tl[np.argmin(np.abs(r-3000))]:.2f} dB")
    print(f"  TL at 5km: {tl[np.argmin(np.abs(r-5000))]:.2f} dB")

# Plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                         gridspec_kw={'height_ratios': [3, 1]})

ax = axes[0]
for beta_val, (r, tl) in results.items():
    ax.plot(r / 1000, tl, linewidth=0.7, alpha=0.8, label=f'beta={beta_val:.4f}')
ax.set_ylabel('TL (dB)')
ax.set_title(f'Same dx={dx}, dz={dz} — varying beta only (shallow water {freq_hz} Hz)')
ax.legend(fontsize=8)
ax.invert_yaxis()
ax.grid(True, alpha=0.3)

# Normalised diff vs first beta
ax = axes[1]
betas = list(results.keys())
r_ref, tl_ref = results[betas[0]]
ref_idx = np.argmin(np.abs(r_ref - 1000))
tl_ref_norm = tl_ref - tl_ref[ref_idx]
for beta_val in betas[1:]:
    r, tl = results[beta_val]
    tl_interp = np.interp(r_ref, r, tl)
    tl_norm = tl_interp - tl_interp[ref_idx]
    diff = tl_norm - tl_ref_norm
    ax.plot(r_ref / 1000, diff, linewidth=0.5, alpha=0.8,
            label=f'beta={beta_val:.4f} - ref')
ax.axhline(0, color='k', linewidth=0.5)
ax.set_ylabel('Norm. diff (dB)')
ax.set_xlabel('Range (km)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_ylim(-15, 15)

fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), 'debug_beta_invariance.png')
fig.savefig(out, dpi=150)
print(f"\nSaved: {out}")
