"""Schematic of the orthogonal terrain-following coordinates on examples.

Draws the coordinate lines ``eta = const`` (level curves) and ``xi = const``
(orthogonal trajectories) over the anisotropy ``r = h_eta / h_xi`` (colour):

(a) Gaussian hill, (b) 20 degree slope (boundary not mirrored at x = 0),
(c) Scottish Highlands terrain (elevation tiles), (d) penetrable wedge with a
flat sea surface, (e) Bermuda pedestal (bathymetry tiles).

Writes ``papers/terrain_transform/figures/coordinate_grids.png``. Run::

    python scripts/terrain_transform_grid_figure.py
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pywaveprop.terrain_transform import OrthogonalTerrainMap  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "papers", "terrain_transform", "figures")


def draw(ax, tm, x_max, eta_max, n_xi_lines=24, n_eta_lines=16, scale=1.0, depth=False, title=''):
    eta = np.linspace(0.0, eta_max, 400)
    xi = np.linspace(0.0, x_max, 400)
    m = tm.metrics(xi, eta)
    X, Z = m.x / scale, m.z
    r = np.log2(m.h_eta / m.h_xi)
    pc = ax.pcolormesh(X, Z, r, cmap='RdBu_r', norm=TwoSlopeNorm(0.0, -2.0, 2.0), shading='gouraud', alpha=0.55)
    for j in np.linspace(0, eta.size - 1, n_eta_lines).astype(int):
        ax.plot(X[:, j], Z[:, j], color='k', lw=0.6)
    for i in np.linspace(0, xi.size - 1, n_xi_lines).astype(int):
        ax.plot(X[i], Z[i], color='0.25', lw=0.6)
    if tm.kind == 'strip':
        jb = int(np.argmin(np.abs(eta - tm.eta_b)))
        ax.plot(X[:, jb], Z[:, jb], color='saddlebrown', lw=2.0)
        ax.invert_yaxis()
    else:
        ax.fill_between(X[:, 0], Z[:, 0], Z[:, 0].min() - 0.05 * (Z.max() - Z.min()), color='saddlebrown')
    eta_l = tm.eta_lab
    if eta_l < eta_max:
        jl = int(np.argmin(np.abs(eta - eta_l)))
        ax.plot(X[:, jl], Z[:, jl], 'g--', lw=1.2)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('x, km' if scale == 1e3 else 'x, m')
    ax.set_ylabel('depth, m' if depth else 'z, m')
    return pc


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(11, 12.5), layout='constrained')
    ax = axes.ravel()

    # (a) Gaussian hill
    xs = np.linspace(-1000, 4000, 1001)
    hs = 150 * np.exp(-((xs - 1500) / 300) ** 2)
    tm = OrthogonalTerrainMap(xs, hs, x_max_m=3000, transition_m=400, smoothing_m=20, z_ref_m=float(hs.mean()))
    draw(ax[0], tm, 3000, 550, title='(a) Gaussian hill, $z_{\\rm ref}$ = mean, $L$ = 400 m (exaggeration 2)')
    ax[0].set_aspect(2.0)

    # (b) steep slope, start trajectory curved
    s = np.tan(np.radians(20))
    xb = np.linspace(-2500, 3000, 2001)
    tm = OrthogonalTerrainMap(xb, s * xb, x_max_m=600, transition_m=2.2 * s * 600, z_ref_m=s * 600,
                              smoothing_m=0.0, mirror_start=False, dx_fine_m=0.5, pad_m=2000)
    draw(ax[1], tm, 600, 2.2 * s * 600 + 100, title='(b) 20° slope, boundary not mirrored at $x$ = 0')
    ax[1].set_aspect('equal')

    # (c) Highlands
    from pywaveprop.environment import get_troposphere_model
    env = get_troposphere_model(57.2163, -4.9214, 57.40, -4.60, n_points=800, resolution=0.0005,
                                use_landcover=False)
    x_t, h_t = np.asarray(env.terrain.x_grid_m), np.asarray(env.terrain.height)
    tm = OrthogonalTerrainMap(x_t, h_t, x_max_m=x_t[-1], transition_m=1000.0, smoothing_m=100.0,
                              z_ref_m=float(h_t.mean()))
    draw(ax[2], tm, x_t[-1], 1250, n_xi_lines=40, scale=1e3,
         title='(c) Scottish Highlands, $L$ = 1000 m (vertical exaggeration ≈ 6)')

    # (d) penetrable wedge
    xw = np.linspace(0, 6500, 1301)
    dw = np.maximum(np.where(xw < 500, 200.0, 200.0 - np.tan(np.radians(2.86)) * (xw - 500)), 10.0)
    tm = OrthogonalTerrainMap(xw, dw, x_max_m=3500, kind='strip', transition_m=270, eta_b_m=200, smoothing_m=20)
    draw(ax[3], tm, 3500, 520, depth=True, scale=1e3, title='(d) wedge: water ($\\sigma$) + sediment taper')

    # (e) Bermuda
    from pywaveprop.environment.seafloor import SeafloorProvider
    b = SeafloorProvider(resolution=0.002).profile(32.475, -64.598, 32.70, -64.40, n_points=400)
    xbm, dbm = np.asarray(b.ranges_m), np.maximum(np.asarray(b.depths_m), 60.0)
    rel = dbm.max() - dbm.min()
    tm = OrthogonalTerrainMap(xbm, dbm, x_max_m=xbm[-1], kind='strip', transition_m=1.5 * rel,
                              eta_b_m=float(dbm.max()), smoothing_m=500.0)
    pc = draw(ax[4], tm, xbm[-1], tm.eta_lab + 300, n_xi_lines=40, depth=True, scale=1e3,
              title='(e) Bermuda pedestal, $\\eta_b$ = max depth')

    # (f) Bermuda with a smaller eta_b
    tm = OrthogonalTerrainMap(xbm, dbm, x_max_m=xbm[-1], kind='strip', transition_m=1.5 * rel,
                              eta_b_m=1200.0, smoothing_m=500.0)
    draw(ax[5], tm, xbm[-1], tm.eta_lab + 300, n_xi_lines=40, depth=True, scale=1e3,
         title='(f) Bermuda pedestal, $\\eta_b$ = 1200 m')

    fig.colorbar(pc, ax=axes, orientation='horizontal', fraction=0.03, pad=0.02,
                 label=r'$\log_2 r$,  $r=h_\eta/h_\xi$  (red: stretched, blue: compressed)')
    fig.savefig(os.path.join(FIG_DIR, 'coordinate_grids.png'), dpi=140, bbox_inches='tight')
    print('written')


if __name__ == '__main__':
    main()
