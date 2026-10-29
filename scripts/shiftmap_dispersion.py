"""Discrete dispersion comparison: shift-map finite-difference PEs versus the
higher-order rational (Numerov--Padé) one-way propagator.

Methods (flat ground, homogeneous medium, k = 2 pi, lengths in wavelengths):

* finite-difference PEs ``u_x = i k (f(X) - 1) u``, Crank--Nicolson in range, with
  the square-root approximations f of the literature
  - NA: ``1 + X/2`` (3-point second-order differences in height),
  - WA Claerbout: ``(1 + 3X/4) / (1 + X/4)`` (3-point),
  - WA Greene: ``(0.99987 + 0.79624 X) / (1 + 0.30102 X)`` (5-point fourth order),
  - WA Padé [2/2] of the square root (5-point fourth order);
* rational: Padé ``[m/m+1]`` approximation of ``exp(i beta dx (sqrt(1+L) - 1))``,
  Numerov differences in height, reference wavenumber ``beta`` optimised.

For each method, accuracy budget (phase error per wavelength; 0.1 rad over 100,
1000 or 10000 wavelengths) and maximum propagation angle, the cheapest grid is
found by brute force under the same criterion: the error stays below the budget
for every angle up to the maximum.  A phase drift common to all angles is a
global phase and is removed for every method.  The cost is the number of
tridiagonal solves per unit area, ``p / (dx dz)``.

A second figure gives the error floor of the shift map on a constant slope (the
part of the error that does not vanish when the grid is refined).

Writes ``shiftmap_efficiency.png``, ``shiftmap_error_vs_angle.png``,
``shiftmap_slope_floor.png`` and ``shiftmap_dispersion.json`` into
``papers/terrain_transform/figures``.  ``$SHIFTMAP_CACHE`` (an .npz path) caches
the rational part.  Run::

    python scripts/shiftmap_dispersion.py
"""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pywaveprop.propagators._utils import pade_propagator_coefs  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "papers", "terrain_transform", "figures")
K = 2 * np.pi
BUDGETS = [1e-3, 1e-4, 1e-5]   # rad per wavelength: 0.1 rad over 1e2, 1e3, 1e4 wavelengths
PLOT_BUDGET = 1e-4
THETA = np.radians(np.arange(0.25, 89.01, 0.25))
DX = np.logspace(-2.5, 2.5, 81)
DZ = np.logspace(-3.0, 0.0, 61)
ORDERS = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9)]  # (10, 11): library Padé coefficients unreliable for some beta*dx
BETA_FACTORS = [1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5]
SQRT_APPROX = {
    'na': lambda X: 1 + X / 2,
    'wa': lambda X: (1 + 3 * X / 4) / (1 + X / 4),
    'greene': lambda X: (0.99987 + 0.79624 * X) / (1 + 0.30102 * X),
    'pade22': lambda X: (1 + 5 * X / 4 + 5 * X ** 2 / 16) / (1 + 3 * X / 4 + X ** 2 / 16),
}
FD_METHODS = [('na', 3, 'C0', 'NA-FD, 3-point $z$'),
              ('wa', 3, 'C1', 'WA-FD Claerbout, 3-point $z$'),
              ('greene', 5, 'C2', 'WA-FD Greene, 5-point $z$'),
              ('pade22', 5, 'C3', r'WA-FD Padé $[2/2]$ sqrt, 5-point $z$')]


def vertical_symbol(theta, dz, alpha, stencil=3):
    """Discrete ``d_zz`` symbol for ``exp(i k sin(theta) z)``, shape (n_theta, n_dz)."""
    t = K * np.sin(theta)[:, None] * dz[None, :]
    if stencil == 5:
        return (-2 * np.cos(2 * t) + 32 * np.cos(t) - 30) / (12 * dz[None, :] ** 2)
    s = np.sin(t / 2) ** 2
    return -4 * s / dz[None, :] ** 2 / (1 - 4 * alpha * s)


def error_rate(R, theta, dx, beta, R0):
    """``|k_x^h - k_x - Re(k_x^h - k_x)|_{theta=0}|`` (rad per wavelength)."""
    exact = np.exp(1j * dx * (K * np.cos(theta)[:, None] - beta))
    with np.errstate(divide='ignore', invalid='ignore'):
        d = np.log(R / exact) / (1j * dx)
        d0 = np.log(R0 / np.exp(1j * dx * (K - beta))) / (1j * dx)
        e = np.abs(d - np.real(d0))
    return np.where(np.isfinite(e), e, np.inf)


def min_costs(cube, cost):
    """Minimal cost per maximum angle for every budget, shape (n_budgets, n_theta)."""
    cummax = np.maximum.accumulate(cube, axis=0).reshape(THETA.size, -1)
    flat = cost.ravel()
    out = np.empty((len(BUDGETS), THETA.size))
    for ib, b in enumerate(BUDGETS):
        out[ib] = np.where(cummax <= b, flat[None, :], np.inf).min(axis=1)
    return out


def fd_method(kind, stencil):
    X = vertical_symbol(THETA, DZ, 0.0, stencil) / K ** 2
    g = SQRT_APPROX[kind](X) - 1
    g0 = SQRT_APPROX[kind](np.zeros(1)) - 1
    cube = np.empty((THETA.size, DX.size, DZ.size))
    for i, dx in enumerate(DX):
        c = 1j * K * dx / 2
        cube[:, i, :] = error_rate((1 + c * g) / (1 - c * g), THETA, dx, K, (1 + c * g0) / (1 - c * g0))
    return min_costs(cube, 1.0 / (DX[:, None] * DZ[None, :]))


def _rational_one(args):
    order, bf = args
    beta = K * bf
    L = vertical_symbol(THETA, DZ, 1 / 12) / beta ** 2 + (K / beta) ** 2 - 1
    L0 = (K / beta) ** 2 - 1
    cube = np.empty((THETA.size, DX.size, DZ.size))
    for i, dx in enumerate(DX):
        coefs = pade_propagator_coefs(pade_order=order, beta=beta, dx=dx)[0]
        R = np.ones_like(L, dtype=complex)
        R0 = 1.0 + 0j
        for a, b in coefs:
            R *= (1 + a * L) / (1 + b * L)
            R0 *= (1 + a * L0) / (1 + b * L0)
        cube[:, i, :] = error_rate(R, THETA, dx, beta, R0)
    return order, min_costs(cube, order[1] / (DX[:, None] * DZ[None, :]))


def rational_all():
    cache = os.environ.get('SHIFTMAP_CACHE')
    if cache and os.path.exists(cache):
        d = np.load(cache)
        return {tuple(int(v) for v in k.split('_')): d[k] for k in d.files}
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(o, bf) for o in ORDERS for bf in BETA_FACTORS]
    with ProcessPoolExecutor(max_workers=min(len(jobs), os.cpu_count() or 4)) as ex:
        res = list(ex.map(_rational_one, jobs))
    out = {}
    for order, c in res:
        out[order] = c if order not in out else np.minimum(out[order], c)
    if cache:
        np.savez(cache, **{f'{o[0]}_{o[1]}': c for o, c in out.items()})
    return out


def max_angle(c):
    ok = np.isfinite(c)
    return float(np.degrees(THETA[ok]).max()) if ok.any() else 0.0


def slope_floor(f, alpha, s):
    """Grid-independent error of the shift map with square-root approximation f
    (rad per wavelength) for a wave at angle alpha over a slope s = tan(theta_s)."""
    X = -(np.sin(alpha) - s) ** 2
    return K * np.abs(f(X) - f(0.0) + 1 + s ** 2 / 2 - s * np.sin(alpha) - np.cos(alpha))


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    deg = np.degrees(THETA)
    ib = BUDGETS.index(PLOT_BUDGET)
    out = {'budgets_rad_per_wavelength': BUDGETS, 'max_angle_deg': {}}
    fd = {kind: fd_method(kind, st) for kind, st, _, _ in FD_METHODS}
    rat = rational_all()
    env = np.minimum.reduce([rat[o] for o in ORDERS])
    for kind, _, _, _ in FD_METHODS:
        out['max_angle_deg'][kind] = [max_angle(fd[kind][j]) for j in range(len(BUDGETS))]
    for o in ORDERS:
        out['max_angle_deg'][f'pade{o}'] = [max_angle(rat[o][j]) for j in range(len(BUDGETS))]
    out['max_angle_deg']['rational_best'] = [max_angle(env[j]) for j in range(len(BUDGETS))]
    out['cost_at_angle'] = {}
    for a in (2, 5, 10, 15, 20, 30, 45, 60, 75, 85):
        j = int(np.argmin(np.abs(deg - a)))
        row = {}
        for k in fd:
            row[k] = [float(fd[k][jb, j]) if np.isfinite(fd[k][jb, j]) else None for jb in range(len(BUDGETS))]
        row['rational'] = [float(env[jb, j]) if np.isfinite(env[jb, j]) else None for jb in range(len(BUDGETS))]
        out['cost_at_angle'][a] = row
    print(json.dumps(out['max_angle_deg'], indent=0), flush=True)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for kind, _, col, lab in FD_METHODS:
        c = fd[kind][ib]
        ok = np.isfinite(c)
        ax.semilogy(deg[ok], c[ok], color=col, lw=2.0, label=lab + ', CN')
    for o, col in zip(ORDERS, plt.cm.viridis(np.linspace(0.0, 0.9, len(ORDERS)))):
        c = rat[o][ib]
        ok = np.isfinite(c)
        ax.semilogy(deg[ok], c[ok], color=col, lw=0.8, alpha=0.7)
    ok = np.isfinite(env[ib])
    ax.semilogy(deg[ok], env[ib, ok], 'k', lw=2.2, label='rational Numerov–Padé (best order; thin: orders)')
    ax.set_xlabel('maximum propagation angle, deg')
    ax.set_ylabel(r'cost: tridiagonal solves per $\lambda^2$')
    ax.set_title(f'phase error < {PLOT_BUDGET:g} rad/$\\lambda$ for all angles up to the maximum', fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=7, loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'shiftmap_efficiency.png'), dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    th = np.radians(np.linspace(0.1, 89, 600))
    for kind, _, col, lab in FD_METHODS:
        f = SQRT_APPROX[kind]
        ax.semilogy(np.degrees(th), K * np.abs(f(-np.sin(th) ** 2) - f(0.0) + 1 - np.cos(th)), color=col, lw=2,
                    label=lab.split(',')[0] + r': limit $\Delta x,\Delta z\to0$')
    for order, col in zip([(2, 3), (4, 5), (6, 7), (8, 9)], ['C4', 'C5', 'C6', 'C7']):
        beta, dx, dz = K, 0.5, 0.02
        L = vertical_symbol(th, np.array([dz]), 1 / 12)[:, 0] / beta ** 2 + (K / beta) ** 2 - 1
        coefs = pade_propagator_coefs(pade_order=order, beta=beta, dx=dx)[0]
        R = np.ones_like(L, dtype=complex)
        R0 = 1.0 + 0j
        for a, b in coefs:
            R *= (1 + a * L) / (1 + b * L)
            R0 *= (1 + a * ((K / beta) ** 2 - 1)) / (1 + b * ((K / beta) ** 2 - 1))
        ax.semilogy(np.degrees(th), error_rate(R[:, None], th, dx, beta, R0)[:, 0], color=col, lw=1.2, ls='--',
                    label=f'Padé {order}, $\\Delta x=\\lambda/2$, $\\Delta z=\\lambda/50$')
    for b in BUDGETS:
        ax.axhline(b, color='k', ls=':', lw=0.8)
    ax.set_ylim(1e-9, 10)
    ax.set_xlabel('propagation angle, deg')
    ax.set_ylabel(r'$|k_x^h-k_x|$, rad/$\lambda$')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=6.5, loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'shiftmap_error_vs_angle.png'), dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=True)
    rel = np.radians(np.linspace(-40, 60, 500))
    f_ex = lambda X: np.sqrt(1 + X + 0j).real  # noqa: E731
    out['slope_floor_feasible_range_deg'] = {}
    for ax, (f, name) in zip(axes, ((SQRT_APPROX['na'], 'NA shift map'),
                                    (SQRT_APPROX['wa'], 'WA (Claerbout) shift map'),
                                    (f_ex, 'exact square root + shift map'))):
        for sd, col in zip((0, 5, 10, 20, 30), plt.cm.plasma(np.linspace(0, 0.85, 5))):
            ts = np.radians(sd)
            e = slope_floor(f, rel + ts, np.tan(ts))
            ax.semilogy(np.degrees(rel), e, color=col, label=f'slope {sd}°')
            for b in BUDGETS:
                ok = e <= b
                out['slope_floor_feasible_range_deg'][f'{name}, slope {sd}, budget {b:g}'] = \
                    [float(np.degrees(rel[ok]).min()), float(np.degrees(rel[ok]).max())] if ok.any() else None
        for b in BUDGETS:
            ax.axhline(b, color='k', ls=':', lw=0.8)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel('angle to the slope, deg')
        ax.grid(True, which='both', alpha=0.3)
        ax.set_ylim(1e-7, 10)
    axes[0].set_ylabel(r'error floor, rad/$\lambda$')
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'shiftmap_slope_floor.png'), dpi=160)
    plt.close(fig)
    with open(os.path.join(FIG_DIR, 'shiftmap_dispersion.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print('written')


if __name__ == '__main__':
    main()
