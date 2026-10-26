"""Staircase versus terrain-following over an impedance slope (exact solution).

The exact field is the plane-wave superposition of
``terrain_transform_impedance.py``: a Gaussian beam at ``-5`` degrees reflected
by a plane slope with the Leontovich coefficient of the local grazing angle.

The staircase solver is the classical one: Pade ``[8/8]`` rational steps of the
wide-angle PE with ``beta = k0``, a fourth-order Numerov discretization in
``z``, the field set to zero below the ground node ``g_i = round(T(x_i) / dz)``
of every column, and the Leontovich condition ``u_z + alpha u = 0`` imposed at
that node with the mirror (ghost-node) closure, i.e. with the vertical normal of
the step instead of the normal of the slope.  An absorbing layer is placed
above the evaluated region.  On flat ground the same code reproduces the exact
field (validation case ``theta_s = 0``).

Writes ``impedance_staircase.json`` and ``impedance_staircase_*.png`` into
``papers/terrain_transform/figures``.  Run::

    python scripts/tap_impedance_staircase.py [validate] [table] [tf] [refine] [plot]
"""
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.linalg import solve_banded  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pywaveprop.propagators._utils import pade_propagator_coefs  # noqa: E402
from terrain_transform_impedance import GROUNDS, K0, exact_field, leontovich_d  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "papers", "terrain_transform", "figures")
OUT = os.path.join(FIG_DIR, "impedance_staircase.json")
OUT_TF = os.path.join(FIG_DIR, "impedance_staircase_tf.json")
ABSORB_M = 150.0


def domain(slope_deg, x_max):
    relief = np.tan(np.radians(slope_deg)) * x_max
    z_top = relief + max(400.0, 2.2 * relief + 100.0)       # as in the terrain-following test
    return relief, z_top


def staircase(slope_deg, eps, pol, dz=0.025, dx=1.0, x_max=600.0, order=8, dx_out=5.0, dz_out=0.5):
    """March the staircase PE; ``pol=None`` is the perfectly conducting (Dirichlet) staircase.
    Returns the output grid, the envelope ``u exp(-i k0 x)`` and the run time."""
    s = np.tan(np.radians(slope_deg))
    exact = exact_field(slope_deg, eps if pol else 1e12 + 0j, pol or 'H')
    relief, z_top = domain(slope_deg, x_max)
    n = int(round((z_top + ABSORB_M) / dz)) + 1
    z = np.arange(n) * dz
    t_abs = np.clip((z - z_top) / ABSORB_M, 0.0, 1.0)
    het = 0.03j * t_abs ** 2
    tau2 = (K0 * dz) ** 2
    al = 1.0 / 12.0
    gamma = 1j * K0 * leontovich_d(eps, pol) if pol else None
    coefs = pade_propagator_coefs(pade_order=(order, order), beta=K0, dx=dx)[0]

    def tri(c):
        off = al * tau2 + c + al * c * tau2 * het
        diag = tau2 * (1 - 2 * al) - 2 * c + c * tau2 * (1 - 2 * al) * het
        return diag, off

    mats = [(tri(complex(a)), tri(complex(b))) for a, b in coefs]
    nx = int(round(x_max / dx))
    step_out = int(round(dx_out / dx))
    rows_out = np.arange(0, int(round(z_top / dz)) + 1, int(round(dz_out / dz)))
    u = np.where(z >= 0.0, exact(np.zeros_like(z), z), 0.0).astype(complex)
    out = [u[rows_out].copy()]
    t0 = time.time()
    for i in range(1, nx + 1):
        g = int(round(s * i * dx / dz))
        for (da, oa), (db, ob) in mats:
            rhs = da * u
            rhs[1:] += oa[:-1] * u[:-1]
            rhs[:-1] += oa[1:] * u[1:]
            ab = np.zeros((3, n), dtype=complex)
            ab[0, 1:] = ob[1:]          # upper: coefficient of u_{j+1} in row j
            ab[1] = db
            ab[2, :-1] = ob[:-1]        # lower: coefficient of u_{j-1} in row j
            if pol:
                # ghost u_{g-1} = u_{g+1} + 2 dz gamma u_g in the rows g of both sides
                rhs[g] = (da[g] + 2 * dz * gamma * oa[g]) * u[g] + 2 * oa[g] * u[g + 1]
                ab[1, g] = db[g] + 2 * dz * gamma * ob[g]
                ab[0, g + 1] = 2 * ob[g]
                lo = g
            else:
                lo = g + 1              # Dirichlet: u_g = 0
                ab[1, g] = 1.0
                ab[0, g + 1] = 0.0
                rhs[g] = 0.0
            ab[1, :lo] = 1.0
            ab[0, 1:lo + 1] = 0.0
            ab[2, :lo] = 0.0
            rhs[:lo] = 0.0
            u = solve_banded((1, 1), ab, rhs)
        u[:g] = 0.0
        if not pol:
            u[g] = 0.0
        if i % step_out == 0:
            out.append(u[rows_out].copy())
    t_run = time.time() - t0
    xo = np.arange(0, nx + 1, step_out) * dx
    return xo, z[rows_out], np.array(out), t_run, exact


def errors(slope_deg, xo, zo, u, exact, margin=2.0):
    s = np.tan(np.radians(slope_deg))
    _, z_top = domain(slope_deg, xo[-1])
    X, Z = np.meshgrid(xo, zo, indexing='ij')
    ue = exact(X, Z) * np.exp(-1j * K0 * X)
    inside = (Z >= s * X + margin) & (Z < z_top - 50.0)
    d = np.abs(u - ue)
    return dict(err=float(d[inside].max() / np.abs(ue[inside]).max()),
                err_l2=float(np.linalg.norm(d[inside]) / np.linalg.norm(ue[inside]))), (X, Z, ue, inside)


def load():
    if os.path.exists(OUT):
        with open(OUT) as f:
            return json.load(f)
    return dict(staircase=[])


def save(res):
    with open(OUT, 'w') as f:
        json.dump(res, f, indent=1)


def run_staircase_cases(cases, **kw):
    res = load()
    for slope, ground, pol in cases:
        eps = GROUNDS.get(ground, 1 + 0j)
        xo, zo, u, t_run, exact = staircase(slope, eps, pol, **kw)
        e, _ = errors(slope, xo, zo, u, exact)
        row = dict(slope_deg=slope, ground=ground, pol=pol, time_s=t_run, **{k: v for k, v in kw.items()}, **e)
        print(json.dumps(row), flush=True)
        res['staircase'] = [r for r in res['staircase']
                            if not (r['slope_deg'] == slope and r['ground'] == ground and r['pol'] == pol
                                    and r.get('dz', 0.025) == kw.get('dz', 0.025)
                                    and r.get('dx', 1.0) == kw.get('dx', 1.0))] + [row]
        save(res)


def main():
    parts = sys.argv[1:] or ['validate', 'table', 'tf', 'refine', 'plot']
    os.makedirs(FIG_DIR, exist_ok=True)
    if 'validate' in parts:
        run_staircase_cases([(0.0, 'wet ground', 'H'), (0.0, 'wet ground', 'V')])
    if 'table' in parts:
        cases = []
        for slope in (2.0, 5.0, 10.0, 20.0):
            cases += [(slope, 'wet ground', 'H'), (slope, 'wet ground', 'V'), (slope, 'sea water', 'V'),
                      (slope, 'pec', None)]
        run_staircase_cases(cases)
    if 'refine' in parts:
        for dz, dx in ((0.0125, 1.0), (0.0125, 0.5)):
            run_staircase_cases([(5.0, 'wet ground', 'V'), (5.0, 'wet ground', 'H')], dz=dz, dx=dx)
    if 'tf' in parts:
        from terrain_transform_impedance import run_case
        res = json.load(open(OUT_TF)) if os.path.exists(OUT_TF) else []
        for slope in (2.0, 5.0, 10.0, 20.0):
            for ground, pol in (('wet ground', 'H'), ('wet ground', 'V'), ('sea water', 'V')):
                r, _ = run_case(slope, GROUNDS[ground], pol, precision=1e-2, x_max=600.0)
                r['ground'] = ground
                r.pop('eps')
                print(json.dumps(r), flush=True)
                res = [q for q in res if not (q['slope_deg'] == slope and q['ground'] == ground
                                              and q['pol'] == pol)] + [r]
                with open(OUT_TF, 'w') as f:
                    json.dump(res, f, indent=1)
    if 'plot' in parts:
        plot(5.0, 'wet ground', 'V')


def plot(slope, ground, pol):
    from terrain_transform_impedance import run_case
    xo, zo, u, _, exact = staircase(slope, GROUNDS[ground], pol)
    _, (X, Z, ue, inside) = errors(slope, xo, zo, u, exact, margin=0.0)
    _, maps = run_case(slope, GROUNDS[ground], pol, precision=1e-2, x_max=600.0)
    xt, zt, ut, uet = maps
    vmax = 20 * np.log10(np.abs(ue[inside]).max())
    above = Z >= np.tan(np.radians(slope)) * X
    zlim = 120.0
    fig, axes = plt.subplots(2, 2, figsize=(11, 5.6), sharex=True, sharey=True, constrained_layout=True)
    panels = [(X, Z, np.where(above, ue, np.nan), 'exact', 0),
              (X, Z, np.where(above, u, np.nan), 'staircase', 0),
              (xt, zt, ut - uet, 'terrain-following: error', 1),
              (X, Z, np.where(above, u - ue, np.nan), 'staircase: error', 1)]
    for ax, (x, z, f, title, kind) in zip(axes.ravel(), panels):
        lo, hi = (vmax - 40, vmax) if kind == 0 else (vmax - 80, vmax - 20)
        pc = ax.pcolormesh(x, z, 20 * np.log10(np.abs(f) + 1e-12), vmin=lo, vmax=hi,
                           cmap='jet' if kind == 0 else 'magma', shading='auto', rasterized=True)
        ax.fill_between([0, x.max()], [-5, -5], [0, x.max() * np.tan(np.radians(slope))], color='0.35')
        ax.set_title(title, fontsize=9)
        ax.set_ylim(-5, zlim)
        if kind == 0 and title == 'staircase':
            fig.colorbar(pc, ax=axes[0], label='dB', shrink=0.9)
        if kind == 1 and title.startswith('staircase'):
            fig.colorbar(pc, ax=axes[1], label='error, dB', shrink=0.9)
    for ax in axes[1]:
        ax.set_xlabel('x, m')
    for ax in axes[:, 0]:
        ax.set_ylabel('z, m')
    fig.suptitle(f'beam at $-5^\\circ$ over a {slope:g}$^\\circ$ {ground} slope, {pol} polarization', fontsize=10)
    fig.savefig(os.path.join(FIG_DIR, f'impedance_staircase_{int(slope)}deg_{pol}.png'), dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.2), constrained_layout=True)
    e_st = np.nanmax(np.where(inside, np.abs(u - ue), np.nan), axis=1) / np.abs(ue[inside]).max()
    e_tf = np.abs(ut - uet).max(axis=1) / np.abs(uet).max()
    ax.semilogy(xt[:, -1], e_tf, 'C3', label='terrain-following')
    ax.semilogy(X[:, 0], e_st, 'C0', label='staircase (impedance on the steps)')
    ax.set_xlabel('x, m')
    ax.set_ylabel(r'$\max_z|u-u_{\rm exact}|/\max|u_{\rm exact}|$')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(FIG_DIR, f'impedance_staircase_{int(slope)}deg_{pol}_error.png'), dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
