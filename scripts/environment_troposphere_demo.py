"""Tropospheric propagation over a real-world terrain profile.

Downloads SRTM-style elevations and ESA WorldCover land-cover tiles for a
short path (Caucasus foothills, ~40 km), builds a JAX-based
``TroposphereModel`` via :func:`pywaveprop.environment.get_troposphere_model`,
and runs an RWP forward task. All data is cached under
``~/.cache/pywaveprop/`` so a second invocation is fully offline.

Run::

    python scripts/environment_troposphere_demo.py
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pywaveprop.environment import get_troposphere_model
from pywaveprop.rwp_jax import (RWPComputationalParams, RWPGaussSourceModel,
                                rwp_forward_task)


# ~15 km northbound path through the foothills of the North Caucasus.
LAT1, LON1 = 43.55, 42.70
LAT2, LON2 = 43.68, 42.75


def main():
    env = get_troposphere_model(
        LAT1, LON1, LAT2, LON2,
        n_points=300,
        resolution=0.002,
    )
    print("Ground material: eps=%.2f, sigma=%.4f"
          % (env.ground_material.eps, env.ground_material.sigma))
    print("Terrain samples: min=%.0f m, max=%.0f m"
          % (float(np.min(env.terrain.height)),
             float(np.max(env.terrain.height))))

    # Place antenna 30 m above local terrain at the start of the path;
    # terrain heights in env.terrain are absolute metres above sea level.
    src_terrain_elev = float(np.asarray(env.terrain.height)[0])
    src_height_absolute = src_terrain_elev + 30.0
    max_terrain = float(np.max(np.asarray(env.terrain.height)))
    max_height = max_terrain + 1000.0
    src = RWPGaussSourceModel(
        freq_hz=3000e6,
        height_m=src_height_absolute,
        beam_width_deg=10.0,
    )
    max_range = float(env.terrain.x_grid_m[-1])
    params = RWPComputationalParams(
        max_range_m=max_range,
        max_height_m=max_height,
        x_output_points=500,
        z_output_points=500,
    )
    #params.approx_method = 'ratinterp'
    field = rwp_forward_task(src, env, params)

    arr = np.asarray(field.field)
    db = 20 * np.log10(np.abs(arr) + 1e-12)
    db = db - db.max()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7),
                                   gridspec_kw={"height_ratios": [1, 3]})
    ax1.plot(np.asarray(env.terrain.x_grid_m) / 1000.0,
             np.asarray(env.terrain.height), color="saddlebrown")
    ax1.fill_between(np.asarray(env.terrain.x_grid_m) / 1000.0,
                     np.asarray(env.terrain.height),
                     alpha=0.3, color="saddlebrown")
    ax1.set_ylabel("Elevation, m")
    ax1.set_title("Path elevation (ESA WorldCover ground + SRTM elev)")
    ax1.grid(True, alpha=0.3)

    im = ax2.imshow(
        db.T,
        origin="lower", aspect="auto",
        extent=[0, max_range / 1000.0, 0, max_height],
        vmin=-80, vmax=0, cmap="jet",
    )
    ax2.plot(np.asarray(env.terrain.x_grid_m) / 1000.0,
             np.asarray(env.terrain.height), color="white", lw=1.0)
    ax2.set_xlabel("Range, km")
    ax2.set_ylabel("Height, m")
    ax2.set_title("RWP field, dB re max (30 MHz, 30 m antenna)")
    plt.colorbar(im, ax=ax2, label="dB")

    out = os.path.join(os.path.dirname(__file__),
                       "environment_troposphere_demo.png")
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    print("Saved plot to", out)


if __name__ == "__main__":
    main()
