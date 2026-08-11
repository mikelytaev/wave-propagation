"""Propagation through a real NWP refractivity field.

Fetches a GFS analysis over the Persian Gulf, builds the range-dependent
modified refractivity ``M(x, z)`` along a maritime path (with the Monin-Obukhov
evaporation duct spliced below 40 m, which GFS itself cannot resolve), and runs
the split-step Pade parabolic equation on it -- once on the true
range-dependent field, once on the launch-point profile held constant with
range, which is the classical horizontally-homogeneous assumption.

The difference between the two panels is the error that assumption costs.
GFS downloads are cached under ``~/.cache/pywaveprop/gfs``, so a second
invocation is fully offline.

Run::

    python scripts/environment_nwp_demo.py
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pywaveprop.environment import BBox, fetch_refractivity_cube, fetch_surface_bulk
from pywaveprop.environment.nwp import refractivity_transect_from_cube

# Persian Gulf: a shallow, hot basin where surface ducts are routine and vary
# strongly along a path of only ~150 km.
P1 = (26.40, 51.90)
P2 = (25.60, 53.10)
BOX = BBox(50.9, 54.1, 24.6, 27.4)

FREQ_HZ = 3e9
TX_HEIGHT_M = 20.0
Z_TOP_M = 600.0


def run_pe(M_profile, z_max, max_range_m):
    from pywaveprop.rwp.antennas import GaussAntenna
    from pywaveprop.rwp.environment import Troposphere
    from pywaveprop.rwp.sspade import RWPSSpadeComputationalParams, rwp_ss_pade

    env = Troposphere(flat=False)  # M-units already carry the Earth curvature
    env.z_max = float(z_max)
    env.M_profile = M_profile

    ant = GaussAntenna(freq_hz=FREQ_HZ, height=TX_HEIGHT_M, beam_width=1.0,
                       elevation_angle=0.0, polarz="H")
    params = RWPSSpadeComputationalParams(max_range_m=max_range_m,
                                          max_height_m=float(z_max))
    field = rwp_ss_pade(ant, env, params)
    return field.x_grid, field.z_grid, np.asarray(field.path_loss().field)


def main():
    cube = fetch_refractivity_cube(BOX, top_height_m=3000.0, dz_m=25.0,
                                   keep_pressure_levels=False)
    bulk = fetch_surface_bulk(BOX)
    tr = refractivity_transect_from_cube(cube, P1, P2, bulk=bulk, z_top=Z_TOP_M)

    print("GFS cycle %s, path %.0f km, %d columns"
          % (cube.attrs["cycle"], tr.length_m / 1000.0, tr.x_m.size))
    print("Evaporation duct height: %.1f..%.1f m"
          % (np.nanmin(tr.edh_pts), np.nanmax(tr.edh_pts)))
    ducted = np.isfinite(tr.duct_strength)
    if ducted.any():
        print("Trapping layers on %d/%d columns, up to %.0f M-units deep"
              % (ducted.sum(), ducted.size, np.nanmax(tr.duct_strength)))

    x, z, tl_rd = run_pe(tr.M_profile(), Z_TOP_M, tr.length_m)
    _, _, tl_uni = run_pe(tr.uniform_M_profile(), Z_TOP_M, tr.length_m)

    vmin = float(np.percentile(tl_rd[np.isfinite(tl_rd)], 3))
    extent = [x[0] / 1e3, x[-1] / 1e3, z[0], z[-1]]
    fig, axs = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    for ax, tl, title in (
        (axs[0], tl_rd, "Range-dependent M(x,z) from GFS"),
        (axs[1], tl_uni, "Range-independent (launch-point profile)"),
    ):
        im = ax.imshow(tl.T, origin="lower", aspect="auto", extent=extent,
                       vmin=vmin, vmax=vmin + 90.0, cmap="jet_r")
        ax.set_title(title)
        ax.set_ylabel("Height, m")
        plt.colorbar(im, ax=ax, label="TL, dB")

    im = axs[2].imshow((tl_rd - tl_uni).T, origin="lower", aspect="auto",
                       extent=extent, vmin=-40, vmax=40, cmap="RdBu_r")
    axs[2].set_title("Difference (range-dependent $-$ uniform)")
    axs[2].set_ylabel("Height, m")
    axs[2].set_xlabel("Range, km")
    plt.colorbar(im, ax=axs[2], label="dB")
    fig.suptitle("%.0f GHz, %.0f m transmitter, GFS %s"
                 % (FREQ_HZ / 1e9, TX_HEIGHT_M, cube.attrs["cycle"]))

    out = os.path.join(os.path.dirname(__file__), "environment_nwp_demo.png")
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    print("Saved plot to", out)


if __name__ == "__main__":
    main()
