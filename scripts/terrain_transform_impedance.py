"""Exact wide-angle test over an impedance ground: beam reflected by a sloping
plane with the Leontovich condition ``d psi / dn + alpha psi = 0``.

The incident field is a Gaussian beam written as a superposition of plane
waves with a smooth angular spectrum; every plane wave is reflected by the
slope with the Leontovich coefficient

    R(phi) = (sin phi - D) / (sin phi + D),
    D = sqrt(eps - 1) (H) or sqrt(eps - 1) / eps (V),

``phi`` the grazing angle to the slope, so the sum is an exact solution of the
Helmholtz equation with the impedance condition on the slope.  The beam leaves
at ``-5`` degrees and is reflected to ``2 theta_s + 5`` degrees.  The
terrain-following solver starts from the exact field on its curved starting
trajectory; the error is the maximum of ``|u - u_exact|`` relative to the
maximum of ``|u_exact|`` over the domain above the ground.

Also tabulates the discrete reflection coefficient of the ghost-node closures
(compact fourth-order closure of the propagator and the second-order mirror closure of
the DMFT / classical FD PE) against the exact Leontovich coefficient.

Writes ``impedance_exact.json``, ``impedance_grid.json``, ``impedance_refl_error.json`` and figures into
``papers/terrain_transform/figures``.  Run::

    python scripts/terrain_transform_impedance.py [refl] [grid] [exact]
"""
import json
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jax.numpy as jnp  # noqa: E402

from pywaveprop.helmholtz_common import HelmholtzMeshParams2D  # noqa: E402
from pywaveprop.helmholtz_ssf_jax import impedance_alpha  # noqa: E402
from pywaveprop.helmholtz_tf_jax import TerrainFollowingHelmholtzPropagator, TFMedium  # noqa: E402
from pywaveprop.terrain_transform import OrthogonalTerrainMap  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "papers", "terrain_transform", "figures")
FREQ = 3e8
K0 = 2 * np.pi * FREQ / 3e8
XS, ZS = -50.0, 30.0           # beam waist
INC_DEG = -5.0                 # beam elevation
DPHI = np.radians(2.0)         # angular half-width: waist 2 / (k0 DPHI) ~ 9 m
GROUNDS = {
    'wet ground': 25 + 1.2j,   # sigma = 0.02 S/m at 300 MHz
    'dry ground': 4 + 0.006j,  # sigma = 1e-4 S/m
    'sea water': 70 + 300j,    # sigma = 5 S/m
}


def leontovich_d(eps, pol):
    d = np.sqrt(complex(eps) - 1.0)
    if d.imag < 0:
        d = -d
    return d if pol == 'H' else d / eps


def exact_field(slope_deg, eps, pol, n_phi=4001):
    th = np.radians(slope_deg)
    c, s = np.cos(th), np.sin(th)
    xc, zc = XS * c + ZS * s, -XS * s + ZS * c          # waist in slope coordinates
    phi0 = th - np.radians(INC_DEG)
    phi = np.linspace(max(phi0 - 6 * DPHI, 1e-3), min(phi0 + 6 * DPHI, np.pi / 2 - 1e-3), n_phi)
    D = leontovich_d(eps, pol)
    R = (np.sin(phi) - D) / (np.sin(phi) + D)
    A = np.exp(-((phi - phi0) / DPHI) ** 2) * np.exp(-1j * K0 * (xc * np.cos(phi) - zc * np.sin(phi)))
    w = A * np.gradient(phi)
    kx, kz = K0 * np.cos(phi), K0 * np.sin(phi)

    def f(x, z):
        x = np.asarray(x, dtype=float)
        z = np.asarray(z, dtype=float)
        xp, zp = (x * c + z * s).ravel(), (-x * s + z * c).ravel()
        out = np.empty(xp.size, dtype=complex)
        for i in range(0, xp.size, 2000):
            X, Z = xp[i:i + 2000, None], zp[i:i + 2000, None]
            e = np.exp(1j * kx * X)
            out[i:i + 2000] = (e * (np.exp(-1j * kz * Z) + R * np.exp(1j * kz * Z))) @ w
        return out.reshape(x.shape)
    norm = np.abs(f(np.array([0.0]), np.array([ZS])))[0]
    return lambda x, z: f(x, z) / norm


def run_case(slope_deg, eps, pol, precision=1e-2, order=(7, 8), x_max=600.0, closure='compact'):
    s = np.tan(np.radians(slope_deg))
    exact = exact_field(slope_deg, eps, pol)
    relief = s * x_max
    z_top = relief + max(400.0, 2.2 * relief + 100.0)
    theta = abs(INC_DEG) + 2 * slope_deg + 8.0
    kz_max = K0 * np.sin(np.radians(min(theta, 85.0)))
    t = time.time()
    xb = np.linspace(-2500.0, x_max + 2500.0, 6001)
    tm = OrthogonalTerrainMap(xb, s * xb, x_max_m=x_max, transition_m=2.2 * relief, z_ref_m=relief,
                              smoothing_m=0.0, mirror_start=False, dx_fine_m=0.5, pad_m=2000.0)
    med = TFMedium(k2_above=lambda x, z: np.full(np.shape(z), K0 ** 2, dtype=complex))
    mp = HelmholtzMeshParams2D(x_size_m=x_max, z_size_m=z_top - relief, dx_output_m=5.0, dz_output_m=1.0)
    tf = TerrainFollowingHelmholtzPropagator.create(FREQ, tm, med, kz_max, (K0, K0), precision, mp,
                                                    rational_approx_order=order, lower_bc='impedance',
                                                    lower_impedance_alpha=impedance_alpha(K0, eps, pol),
                                                    impedance_closure=closure)
    x0, z0 = tf.initial_points()
    u = np.asarray(tf.compute(jnp.asarray(exact(x0, z0))))
    t_run = time.time() - t
    xo, zo = tf.output_metrics()
    ue = exact(xo, zo) * np.exp(-1j * tf.beta * np.asarray(tf.x_output_grid())[:, None])
    inside = zo < z_top - 50.0
    err = np.abs(u - ue)
    res = dict(slope_deg=slope_deg, eps=[eps.real, eps.imag], pol=pol, theta_max_deg=theta,
               precision=precision, closure=closure, k_deta=float(K0 * tf.dz_m),
               err=float(err[inside].max() / np.abs(ue[inside]).max()),
               err_ground=float(err[:, 0].max() / np.abs(ue[inside]).max()),
               time_s=t_run, x_n=tf.x_n, z_n=tf.z_n, dx=tf.dx_m, dz=tf.dz_m, order=list(tf.order))
    return res, (xo, zo, u, ue)


# ----------------------------------------------------------- discrete reflection
def discrete_refl(q, g, closure):
    """Reflection coefficient of the discrete wave ``e^{-iqj} + R e^{iqj}`` under
    the ghost-node closure ``u_-1 = P u_0 + Q u_1 + S u_2`` (``g = gamma deta``)."""
    if closure == 'compact':   # (u_1 - u_-1) + g (u_-1 + 4 u_0 + u_1) / 3 = 0
        P, Q, S = (4 * g / 3) / (1 - g / 3), (1 + g / 3) / (1 - g / 3), 0.0
    else:   # mirror: (u_1 - u_-1) / (2 deta) + gamma u_0 = 0
        P, Q, S = 2 * g, 1.0, 0.0
    num = np.exp(1j * q) - P - Q * np.exp(-1j * q) - S * np.exp(-2j * q)
    den = np.exp(-1j * q) - P - Q * np.exp(1j * q) - S * np.exp(2j * q)
    return -num / den


def refl_table():
    """``|R_h - R|`` against the grazing angle for ``k0 deta`` = ``tau``."""
    res = []
    theta = np.radians(np.linspace(0.5, 85, 341))
    for name, eps in GROUNDS.items():
        for pol in ('H', 'V'):
            D = leontovich_d(eps, pol)
            R = (np.sin(theta) - D) / (np.sin(theta) + D)
            for tau in (0.25, 0.5, 1.0, 1.5):
                q = tau * np.sin(theta)
                row = dict(ground=name, pol=pol, tau=tau, theta_deg=np.degrees(theta).tolist())
                for cl in ('compact', 'mirror'):
                    row[cl] = np.abs(discrete_refl(q, 1j * tau * D, cl) - R).tolist()
                res.append(row)
    with open(os.path.join(FIG_DIR, 'impedance_refl_error.json'), 'w') as f:
        json.dump(res, f)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, pol in zip(axes, ('H', 'V')):
        for row in res:
            if row['ground'] != 'wet ground' or row['pol'] != pol or row['tau'] not in (0.5, 1.5):
                continue
            col = 'C0' if row['tau'] == 0.5 else 'C3'
            ax.semilogy(row['theta_deg'], row['compact'], '-', color=col,
                        label=rf"compact (fourth order), $k\Delta\eta={row['tau']:g}$")
            ax.semilogy(row['theta_deg'], row['mirror'], '--', color=col,
                        label=rf"mirror (second order), $k\Delta\eta={row['tau']:g}$")
        ax.set_title(f'wet ground, {pol} polarization', fontsize=9)
        ax.set_xlabel('grazing angle, deg')
        ax.grid(True, which='both', alpha=0.3)
    axes[0].set_ylabel(r'$|R_h-R|$')
    axes[0].set_ylim(1e-8, 1)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'impedance_refl_error.png'), dpi=160)
    plt.close(fig)
    for row in res:
        if row['ground'] == 'wet ground':
            th = np.array(row['theta_deg'])
            sel = th <= 60
            print(row['pol'], row['tau'], 'max |dR| (<=60 deg): compact %.2e mirror %.2e'
                  % (np.max(np.array(row['compact'])[sel]), np.max(np.array(row['mirror'])[sel])))


def run_case_grid(slope_deg, eps, pol, deta, closure, order=(7, 8), dx=2.5, x_max=600.0):
    """Same problem on a prescribed grid: exposes the ground closure."""
    s = np.tan(np.radians(slope_deg))
    exact = exact_field(slope_deg, eps, pol)
    relief = s * x_max
    z_top = relief + max(400.0, 2.2 * relief + 100.0)
    xb = np.linspace(-2500.0, x_max + 2500.0, 6001)
    tm = OrthogonalTerrainMap(xb, s * xb, x_max_m=x_max, transition_m=2.2 * relief, z_ref_m=relief,
                              smoothing_m=0.0, mirror_start=False, dx_fine_m=0.5, pad_m=2000.0)
    med = TFMedium(k2_above=lambda x, z: np.full(np.shape(z), K0 ** 2, dtype=complex))
    x_n = int(round(x_max / dx)) + 1
    z_n = int(round((z_top - relief) / deta)) + 1
    beta = K0 * np.cos(np.radians(20.0))
    tables = TerrainFollowingHelmholtzPropagator.build_tables(tm, med, beta=beta, dxi=dx, deta=deta, x_n=x_n, z_n=z_n)
    tf = TerrainFollowingHelmholtzPropagator(
        order=order, beta=beta, dx_m=dx, dz_m=deta, x_n=x_n, z_n=z_n, x_grid_scale=2,
        z_grid_scale=int(round(1.0 / deta)), freq_hz=FREQ,
        tables=tables, terrain_map=tm, lower_bc='impedance', lower_impedance_alpha=impedance_alpha(K0, eps, pol),
        impedance_closure=closure)
    x0, z0 = tf.initial_points()
    u = np.asarray(tf.compute(jnp.asarray(exact(x0, z0))))
    xo, zo = tf.output_metrics()
    ue = exact(xo, zo) * np.exp(-1j * tf.beta * np.asarray(tf.x_output_grid())[:, None])
    inside = zo < z_top - 50.0
    return float(np.abs(u - ue)[inside].max() / np.abs(ue[inside]).max())


def main_grid():
    table = []
    for pol in ('H', 'V'):
        for frac in (16, 10, 6, 4):
            row = dict(slope_deg=10.0, ground='wet ground', pol=pol, deta=1.0 / frac, k_deta=float(K0 / frac))
            for closure in ('compact', 'mirror'):
                row[closure] = run_case_grid(10.0, GROUNDS['wet ground'], pol, 1.0 / frac, closure)
            print(json.dumps(row), flush=True)
            table.append(row)
            with open(os.path.join(FIG_DIR, 'impedance_grid.json'), 'w') as f:
                json.dump(table, f, indent=1)


def main_exact():
    table = []
    for slope in (5.0, 10.0, 20.0, 30.0):
        for name in ('wet ground', 'sea water', 'dry ground'):
            if name != 'wet ground' and slope != 10.0:
                continue
            for pol in ('H', 'V'):
                x_max = 600.0 if slope < 25 else 400.0
                res, maps = run_case(slope, GROUNDS[name], pol, precision=1e-2, x_max=x_max)
                res['ground'] = name
                print(json.dumps({k: v for k, v in res.items() if k != 'eps'}), flush=True)
                table.append(res)
                if slope == 20.0 and name == 'wet ground':
                    plot_maps(slope, pol, maps)
                with open(os.path.join(FIG_DIR, 'impedance_exact.json'), 'w') as f:
                    json.dump(table, f, indent=1)


def plot_maps(slope_deg, pol, maps):
    xo, zo, u, ue = maps
    vmax = 20 * np.log10(np.abs(ue).max())
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, (f, title, rng) in zip(axes, [(ue, 'exact', 40), (u, 'terrain-following', 40),
                                         (u - ue, 'error', 40)]):
        pc = ax.pcolormesh(xo, zo, 20 * np.log10(np.abs(f) + 1e-12), vmin=vmax - 80 if title == 'error' else vmax - 40,
                           vmax=vmax - 40 if title == 'error' else vmax, cmap='jet', shading='auto')
        ax.plot([0, xo.max()], [0, xo.max() * np.tan(np.radians(slope_deg))], 'k', lw=1.5)
        ax.set_title(title)
        ax.set_xlabel('x, m')
        fig.colorbar(pc, ax=ax, label='dB')
    axes[0].set_ylabel('z, m')
    fig.savefig(os.path.join(FIG_DIR, f'impedance_{int(slope_deg)}deg_{pol}_maps.png'), dpi=130,
                bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    os.makedirs(FIG_DIR, exist_ok=True)
    parts = sys.argv[1:] or ['refl', 'grid', 'exact']
    if 'refl' in parts:
        refl_table()
    if 'grid' in parts:
        main_grid()
    if 'exact' in parts:
        main_exact()
