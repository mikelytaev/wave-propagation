"""Compare a single propagation step with different beta values.
If the solution diverges after 1 step, the bug is in the CN discretization.
If it's fine after 1 step but diverges over many steps, it's accumulation."""
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
    UnderwaterEnvironmentModel, ProxyWaveSpeedModel, ProxyRhoModel, minmax_k,
)
from pywaveprop.helmholtz_jax import ConstWaveSpeedModel, RationalHelmholtzPropagator
from pywaveprop.propagators._utils import pade_propagator_coefs

# Simple isovelocity environment
freq_hz = 100.0
src_depth = 50.0
c_water = 1500.0
c_bottom = 1700.0
bottom_depth = 300.0
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
                         density=1.5, attenuation_dm_lambda=0.5),
])

k0 = 2 * fm.pi * freq_hz / c_water
dx = 5.0
dz = 1.0
max_range_m = 500.0  # short range

betas = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, k0]
results = {}

for beta_val in betas:
    coefs = pade_propagator_coefs(pade_order=(7, 8), beta=beta_val, dx=dx)[0]
    coefs_list = [list(v) for v in coefs]
    x_n = int(max_range_m / dx) + 1
    z_n = int(max_depth_m / dz) + 1

    model = RationalHelmholtzPropagator(
        order=(7, 8), beta=beta_val, dx_m=dx, dz_m=dz,
        x_n=x_n, z_n=z_n, x_grid_scale=1, z_grid_scale=1,
        freq_hz=freq_hz,
        wave_speed=ProxyWaveSpeedModel(jax_env),
        rho=ProxyRhoModel(jax_env),
        coefs=coefs_list,
    )

    init = jax_src.aperture(k0, model.z_computational_grid())
    f = model.compute(init)
    z = np.asarray(model.z_output_grid())
    z_idx = np.argmin(np.abs(z - src_depth))
    tl = -20 * np.log10(np.abs(np.asarray(f)[:, z_idx]) + 1e-30)
    r = np.asarray(model.x_output_grid())
    results[beta_val] = (r, tl)

# Compare at key ranges
print("TL at source depth vs beta (short range, isovelocity):")
print(f"{'beta':>8s}", end="")
for rng in [50, 100, 200, 300, 500]:
    print(f"  {'r='+str(rng)+'m':>10s}", end="")
print()
for beta_val in betas:
    r, tl = results[beta_val]
    print(f"{beta_val:8.4f}", end="")
    for rng in [50, 100, 200, 300, 500]:
        idx = np.argmin(np.abs(r - rng))
        print(f"  {tl[idx]:10.2f}", end="")
    print()

# Show max pairwise difference across all betas
print(f"\nMax TL spread across betas at each range:")
for rng in [50, 100, 200, 300, 500]:
    vals = []
    for beta_val in betas:
        r, tl = results[beta_val]
        idx = np.argmin(np.abs(r - rng))
        vals.append(tl[idx])
    print(f"  r={rng:4d} m: spread = {max(vals)-min(vals):.2f} dB  (min={min(vals):.2f}, max={max(vals):.2f})")
