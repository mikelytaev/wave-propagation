"""Exact wide-angle test: Gaussian beam reflected by a perfectly conducting slope.

The incident field is a complex-source-point (CSP) beam, an exact solution of
the 2-D Helmholtz equation; reflecting the complex source in the line
``z = x tan(theta_s)`` gives the exact field above a Dirichlet slope.  The beam
leaves at ``-10`` degrees and is reflected to ``2 theta_s + 10`` degrees, so the
solvers have to resolve 30--70 degrees.  Both the terrain-following solver and
the staircase solver of the package start from the exact field (on the curved
starting trajectory and on the vertical line respectively) and use the same
aperture and precision; the error is the maximum of ``|u - u_exact|`` relative
to the maximum of ``|u_exact|`` over the domain above the ground.

Writes ``slope_exact.json`` and figures into ``papers/terrain_transform/figures``.

Run::

    python scripts/terrain_transform_slope_exact.py
"""
import json
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import hankel1

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jax.numpy as jnp  # noqa: E402

from pywaveprop.helmholtz_common import HelmholtzMeshParams2D  # noqa: E402
from pywaveprop.helmholtz_jax import (ConstWaveSpeedModel, PiecewiseLinearTerrainModel,  # noqa: E402
                                      RationalHelmholtzPropagator)
from pywaveprop.helmholtz_tf_jax import TerrainFollowingHelmholtzPropagator, TFMedium  # noqa: E402
from pywaveprop.terrain_transform import OrthogonalTerrainMap  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "papers", "terrain_transform", "figures")
FREQ = 3e8
K0 = 2 * np.pi * FREQ / 3e8
XS, ZS, B = -50.0, 30.0, 100.0      # beam waist and Rayleigh parameter
ALPHA_DEG = -10.0


def csp(x, z, xs, zs, ang):
    R = np.sqrt((x - xs - 1j * B * np.cos(ang)) ** 2 + (z - zs - 1j * B * np.sin(ang)) ** 2)
    return 1j / 4 * hankel1(0, K0 * R)


def exact_field(slope_deg):
    s = np.tan(np.radians(slope_deg))
    n = np.array([-s, 1.0]) / np.hypot(s, 1.0)
    refl = np.eye(2) - 2 * np.outer(n, n)
    ri = refl @ np.array([XS, ZS])
    ei = refl @ np.array([np.cos(np.radians(ALPHA_DEG)), np.sin(np.radians(ALPHA_DEG))])

    def f(x, z):
        return csp(x, z, XS, ZS, np.radians(ALPHA_DEG)) - csp(x, z, ri[0], ri[1], np.arctan2(ei[1], ei[0]))
    norm = np.abs(f(np.array([0.0]), np.array([ZS])))[0]
    return lambda x, z: f(x, z) / norm


def run_case(slope_deg, precision, order=(7, 8), x_max=600.0):
    s = np.tan(np.radians(slope_deg))
    exact = exact_field(slope_deg)
    z_top = s * x_max + max(400.0, 2.2 * s * x_max + 100.0)
    theta = abs(ALPHA_DEG) + 2 * slope_deg + 5.0
    kz_max = K0 * np.sin(np.radians(min(theta, 85.0)))
    out = dict(slope_deg=slope_deg, precision=precision, theta_max_deg=theta)

    # terrain-following
    t = time.time()
    xb = np.linspace(-2500.0, x_max + 2500.0, 6001)
    relief = s * x_max
    tm = OrthogonalTerrainMap(xb, s * xb, x_max_m=x_max, transition_m=2.2 * relief, z_ref_m=relief,
                              smoothing_m=0.0, mirror_start=False, dx_fine_m=0.5, pad_m=2000.0)
    med = TFMedium(k2_above=lambda x, z: np.full(np.shape(z), K0 ** 2, dtype=complex))
    mp = HelmholtzMeshParams2D(x_size_m=x_max, z_size_m=z_top - relief, dx_output_m=5.0, dz_output_m=1.0)
    tf = TerrainFollowingHelmholtzPropagator.create(FREQ, tm, med, kz_max, (K0, K0), precision, mp,
                                                    rational_approx_order=order)
    x0, z0 = tf.initial_points()
    u = np.asarray(tf.compute(jnp.asarray(exact(x0, z0))))
    xo, zo = tf.output_metrics()
    ue = exact(xo, zo) * np.exp(-1j * tf.beta * np.asarray(tf.x_output_grid())[:, None])
    inside = zo < z_top - 50.0
    out['tf'] = dict(err=float(np.abs(u - ue)[inside].max() / np.abs(ue[inside]).max()), time_s=time.time() - t,
                     x_n=tf.x_n, z_n=tf.z_n, dx=tf.dx_m, dz=tf.dz_m, order=list(tf.order))
    tf_maps = (xo, zo, u, ue)

    # staircase
    t = time.time()
    st = RationalHelmholtzPropagator.create(
        freq_hz=FREQ, wave_speed=ConstWaveSpeedModel(3e8), kz_max=kz_max, k_bounds=(K0, K0), precision=precision,
        mesh_params=HelmholtzMeshParams2D(x_size_m=x_max, z_size_m=z_top, dx_output_m=5.0, dz_output_m=1.0),
        lower_terrain=PiecewiseLinearTerrainModel(np.array([0.0, x_max]), np.array([0.0, relief])),
        rational_approx_order=order)
    zg = np.asarray(st.z_computational_grid())
    init = np.where(zg >= 0.0, exact(np.zeros_like(zg), zg), 0.0)
    us = np.asarray(st.compute(jnp.asarray(init)))
    xg, zgo = np.asarray(st.x_output_grid()), np.asarray(st.z_output_grid())
    XX, ZZ = np.meshgrid(xg, zgo, indexing='ij')
    ues = exact(XX, ZZ) * np.exp(-1j * st.beta * XX)
    inside_s = (ZZ > s * XX + 2.0) & (ZZ < z_top - 50.0)
    out['staircase'] = dict(err=float(np.abs(us - ues)[inside_s].max() / np.abs(ues[inside_s]).max()),
                            time_s=time.time() - t, x_n=st.x_n, z_n=st.z_n, dx=st.dx_m, dz=st.dz_m,
                            order=list(st.order))
    return out, tf_maps, (XX, ZZ, us, ues, inside_s)


def plot_maps(slope_deg, tf_maps, st_maps):
    xo, zo, u, ue = tf_maps
    XX, ZZ, us, ues, inside = st_maps
    vmax = 20 * np.log10(np.abs(ue).max())
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, (X, Z, f, title) in zip(axes, [(xo, zo, ue, 'exact'), (xo, zo, u, 'terrain-following'),
                                         (XX, ZZ, np.where(inside, us, np.nan), 'staircase')]):
        pc = ax.pcolormesh(X, Z, 20 * np.log10(np.abs(f) + 1e-12), vmin=vmax - 40, vmax=vmax, cmap='jet',
                           shading='auto')
        ax.plot([0, X.max()], [0, X.max() * np.tan(np.radians(slope_deg))], 'k', lw=1.5)
        ax.set_title(title)
        ax.set_xlabel('x, m')
    axes[0].set_ylabel('z, m')
    fig.colorbar(pc, ax=axes, label='dB')
    fig.savefig(os.path.join(FIG_DIR, f'slope_exact_{int(slope_deg)}deg_maps.png'), dpi=130, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    e_tf = np.abs(u - ue).max(axis=1) / np.abs(ue).max()
    e_st = np.nanmax(np.where(inside, np.abs(us - ues), np.nan), axis=1) / np.abs(ues).max()
    ax.semilogy(xo[:, -1], e_tf, 'C3', label='terrain-following')
    ax.semilogy(XX[:, 0], e_st, 'C0', label='staircase')
    ax.set_ylim(1e-6, 10)
    ax.set_xlabel('x, m')
    ax.set_ylabel(r'$\max_z |u-u_{\rm exact}| / \max |u_{\rm exact}|$')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    ax.set_title(f'slope {slope_deg}°, reflected beam at {2 * slope_deg + 10:.0f}°')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f'slope_exact_{int(slope_deg)}deg_error.png'), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    table = []
    for slope in (10.0, 20.0, 30.0):
        for prec in (1e-1, 1e-2):
            x_max = 600.0 if slope < 25 else 400.0
            res, tf_maps, st_maps = run_case(slope, prec, x_max=x_max)
            print(json.dumps(res), flush=True)
            table.append(res)
            if prec == 1e-2:
                plot_maps(slope, tf_maps, st_maps)
    with open(os.path.join(FIG_DIR, 'slope_exact.json'), 'w') as f:
        json.dump(table, f, indent=1)


if __name__ == '__main__':
    main()
