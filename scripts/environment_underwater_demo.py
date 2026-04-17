"""Underwater acoustic propagation for a real-world ocean path.

Downloads a bathymetry profile (Mapzen Terrarium) and a mean sound-velocity
profile (Argo) for a ~50 km transect near Bermuda, builds a JAX
``UnderwaterEnvironmentModel`` via
:func:`pywaveprop.environment.get_underwater_environment_model`, and runs a
UWA forward task. Cached data under ``~/.cache/pywaveprop/`` makes reruns
offline.

Run::

    python scripts/environment_underwater_demo.py
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pywaveprop.environment import get_underwater_environment_model
from pywaveprop.uwa_jax import UWAGaussSourceModel, uwa_forward_task
from pywaveprop.uwa_utils import UWAComputationalParams


LAT1, LON1 = 31.50, -64.80
LAT2, LON2 = 31.80, -64.30


def main():
    env = get_underwater_environment_model(
        LAT1, LON1, LAT2, LON2,
        n_range_points=200,
        resolution=0.005,
        max_svp_depth_m=2000.0,
    )
    print("Water layer depth: %.0f m" % env.layers[0].height_m)
    print("Bathymetry: max depth %.0f m"
          % float(np.max(env.bathymetry.height)))

    src = UWAGaussSourceModel(
        freq_hz=100.0,
        depth_m=50.0,
        beam_width_deg=25.0,
    )
    max_range = float(env.bathymetry.x_grid_m[-1])
    params = UWAComputationalParams(
        max_range_m=max_range,
        dx_m=50.0,
        dz_m=2.0,
    )
    result = uwa_forward_task(src, env, params)

    arr = np.asarray(result.field)
    db = 20 * np.log10(np.abs(arr) + 1e-12)
    db = db - db.max()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7),
                                   gridspec_kw={"height_ratios": [1, 3]})
    ax1.plot(np.asarray(env.bathymetry.x_grid_m) / 1000.0,
             -np.asarray(env.bathymetry.height), color="navy")
    ax1.fill_between(np.asarray(env.bathymetry.x_grid_m) / 1000.0,
                     -np.asarray(env.bathymetry.height),
                     np.min(-np.asarray(env.bathymetry.height)) - 100,
                     alpha=0.3, color="navy")
    ax1.set_ylabel("Sea floor, m")
    ax1.set_title("Bathymetry (Mapzen Terrarium) along ~50 km path")
    ax1.grid(True, alpha=0.3)

    depth_max = float(np.max(np.asarray(result.z_grid)))
    im = ax2.imshow(
        db.T,
        origin="upper", aspect="auto",
        extent=[0, max_range / 1000.0, depth_max, 0],
        vmin=-80, vmax=0, cmap="jet",
    )
    ax2.set_xlabel("Range, km")
    ax2.set_ylabel("Depth, m")
    ax2.set_title("UWA field, dB re max (100 Hz source at 50 m)")
    plt.colorbar(im, ax=ax2, label="dB")

    out = os.path.join(os.path.dirname(__file__),
                       "environment_underwater_demo.png")
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    print("Saved plot to", out)


if __name__ == "__main__":
    main()
