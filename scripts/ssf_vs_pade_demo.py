"""Split-step Fourier vs rational (Padé) finite-difference JAX propagators.

Runs both solvers on a few tropospheric problems (standard atmosphere over
PEC, evaporation duct, hill terrain, V-polarization over a dielectric near
the Brewster angle) and plots the 2D fields together with horizontal cuts.

    python scripts/ssf_vs_pade_demo.py
"""
import time
from copy import deepcopy

import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pywaveprop.helmholtz_jax import PiecewiseLinearTerrainModel
from pywaveprop.rwp_jax import (EvaporationDuctModel, GroundMaterial, RWPComputationalParams,
                                RWPGaussSourceModel, TroposphereModel, rwp_forward_task)


def run(src, env, params, method):
    p = deepcopy(params)
    p.approx_method = method
    p.rational_approx_order = (7, 8)
    t = time.time()
    f = rwp_forward_task(src, env, p)
    print(f'{method}: {time.time() - t:.1f} s')
    return f


cases = {
    'standard_pec': (
        RWPGaussSourceModel(freq_hz=300e6, height_m=30.0, beam_width_deg=5.0),
        TroposphereModel(),
        RWPComputationalParams(max_range_m=20000.0, max_height_m=300.0, dx_m=50.0, dz_m=1.0),
        [30, 100]),
    'evaporation_duct': (
        RWPGaussSourceModel(freq_hz=3e9, height_m=10.0, beam_width_deg=3.0),
        TroposphereModel(N_profile=EvaporationDuctModel(height_m=20.0)),
        RWPComputationalParams(max_range_m=30000.0, max_height_m=150.0, dx_m=100.0, dz_m=0.5),
        [5, 30]),
    'hill': (
        RWPGaussSourceModel(freq_hz=300e6, height_m=50.0, beam_width_deg=8.0),
        TroposphereModel(terrain=PiecewiseLinearTerrainModel(
            x_grid_m=jnp.array([0.0, 3000.0, 5000.0, 7000.0, 10000.0]),
            height=jnp.array([0.0, 0.0, 80.0, 0.0, 0.0]))),
        RWPComputationalParams(max_range_m=10000.0, max_height_m=400.0, dx_m=50.0, dz_m=1.0),
        [30, 150]),
    'brewster_v_pol': (
        RWPGaussSourceModel(freq_hz=300e6, height_m=30.0, beam_width_deg=5.0, polarization='V'),
        TroposphereModel(ground_material=GroundMaterial(eps=3.0, sigma=0.0)),
        RWPComputationalParams(max_range_m=10000.0, max_height_m=300.0, dx_m=50.0, dz_m=1.0),
        [5, 100]),
}

for name, (src, env, params, heights) in cases.items():
    print(f'=== {name}')
    fd = run(src, env, params, 'pade')
    ssf = run(src, env, params, 'ssf')
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, f, label in zip(axes[0], (fd, ssf), ('Padé FD', 'SSF')):
        amp = 20 * np.log10(np.abs(f.field).T + 1e-30)
        vmax = float(np.max(amp[:, len(f.x_grid) // 10:]))
        ax.pcolormesh(f.x_grid, f.z_grid, amp, shading='auto', cmap='jet', vmin=vmax - 60, vmax=vmax)
        ax.set_title(f'{label}: {name}')
        ax.set_xlabel('Range (m)')
        ax.set_ylabel('Height (m)')
    for ax, z in zip(axes[1], heights):
        for f, label in zip((fd, ssf), ('Padé FD', 'SSF')):
            iz = int(np.argmin(np.abs(f.z_grid - z)))
            ax.plot(f.x_grid, 20 * np.log10(np.abs(f.field[:, iz]) + 1e-30), label=label)
        ax.set_title(f'z = {z} m')
        ax.set_xlabel('Range (m)')
        ax.set_ylabel('|u| (dB)')
        ax.set_ylim(bottom=-80)
        ax.legend()
    plt.tight_layout()
    plt.savefig(f'ssf_vs_pade_{name}.png', dpi=120)
    plt.close()
