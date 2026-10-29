"""Coordinate grids for the radio examples only (Gaussian hill, 20 degree slope,
Scottish Highlands), laid out for a two-column journal page.

Writes ``papers/terrain_transform/figures/coordinate_grids_rwp.png``. Run::

    python scripts/terrain_transform_grid_figure_rwp.py
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from terrain_transform_grid_figure import FIG_DIR, draw  # noqa: E402
from pywaveprop.terrain_transform import OrthogonalTerrainMap  # noqa: E402


def main():
    plt.rcParams.update({'font.size': 9})
    fig = plt.figure(figsize=(7.2, 5.6), layout='constrained')
    gs = fig.add_gridspec(2, 2, width_ratios=[2.2, 1.0], height_ratios=[1.0, 1.25])
    ax_a, ax_b, ax_c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])

    xs = np.linspace(-1000, 4000, 1001)
    hs = 150 * np.exp(-((xs - 1500) / 300) ** 2)
    tm = OrthogonalTerrainMap(xs, hs, x_max_m=3000, transition_m=400, smoothing_m=20, z_ref_m=float(hs.mean()))
    draw(ax_a, tm, 3000, 550, title='(a) Gaussian hill (vertical exaggeration 2)')
    ax_a.set_aspect(2.0)

    s = np.tan(np.radians(20))
    xb = np.linspace(-2500, 3000, 2001)
    tm = OrthogonalTerrainMap(xb, s * xb, x_max_m=600, transition_m=2.2 * s * 600, z_ref_m=s * 600,
                              smoothing_m=0.0, mirror_start=False, dx_fine_m=0.5, pad_m=2000)
    draw(ax_b, tm, 600, 2.2 * s * 600 + 100, n_xi_lines=16, n_eta_lines=12, title='(b) 20° slope')
    ax_b.set_aspect('equal')

    from pywaveprop.environment import get_troposphere_model
    env = get_troposphere_model(57.2163, -4.9214, 57.40, -4.60, n_points=800, resolution=0.0005,
                                use_landcover=False)
    x_t, h_t = np.asarray(env.terrain.x_grid_m), np.asarray(env.terrain.height)
    tm = OrthogonalTerrainMap(x_t, h_t, x_max_m=x_t[-1], transition_m=1000.0, smoothing_m=100.0,
                              z_ref_m=float(h_t.mean()))
    pc = draw(ax_c, tm, x_t[-1], 1250, n_xi_lines=48, scale=1e3,
              title='(c) Scottish Highlands (vertical exaggeration ≈ 6)')
    fig.colorbar(pc, ax=[ax_a, ax_b, ax_c], orientation='horizontal', fraction=0.04, pad=0.01,
                 label=r'$\log_2 (h_\eta/h_\xi)$')
    fig.savefig(os.path.join(FIG_DIR, 'coordinate_grids_rwp.png'), dpi=200)
    print('written')


if __name__ == '__main__':
    main()
