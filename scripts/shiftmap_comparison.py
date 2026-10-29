"""Numerical comparison with shift-map finite-difference PEs and with absorbing layers.

All fields are compared with exact solutions built from complex-source-point
(CSP) beams at 300 MHz (lambda = 1 m) over a perfectly conducting ground
(horizontal polarisation), so that every error is measured, not estimated.

E1 (``angles``): beams at 5--70 degrees over flat ground.  NA-FD and WA-FD on a
    typical grid (dx = lambda/2, dz = lambda/10) and on a fine grid
    (dx = lambda/8, dz = lambda/40) versus the rational Numerov--Padé propagator
    on the grid chosen by its optimizer.  All three use the discrete NLBC on top,
    so the comparison isolates the approximation of the propagator.
E2 (``slopes``): the exact sloping-ground problem of the paper (beam at -10 degrees
    reflected by a slope of 5--30 degrees): NA and WA shift maps on the fine grid
    versus the terrain-following rational propagator.
E3 (``absorber``): beams at 1--30 degrees leaving the domain through the top; the
    same rational propagator with the NLBC and with absorbing layers of thickness
    D = H/4, H/2, H and strength A = 1e-3 ... 1 (complex index i A t^2, zero field
    on top), error measured below the height H of interest; Padé (7,8),
    dx = 2.5 m, dz = 0.1 m for all.

Writes ``shiftmap_comparison_<part>.json`` and figures into
``papers/terrain_transform/figures``. Run::

    python scripts/shiftmap_comparison.py angles slopes absorber
"""
import json
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import hankel1

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pywaveprop.grid_optimizer import get_optimal_grid  # noqa: E402
from pywaveprop.helmholtz_shiftmap_jax import ShiftMapPropagator, ShiftMapTerrain  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "papers", "terrain_transform", "figures")
FREQ = 3e8
K0 = 2 * np.pi
B = 50.0            # CSP Rayleigh parameter: waist ~ 4 m, divergence ~ 2.3 degrees
XS, ZS = -60.0, 40.0   # |XS| > B keeps the CSP branch cut out of x >= 0


def csp(x, z, xs, zs, ang):
    r = np.sqrt((x - xs - 1j * B * np.cos(ang)) ** 2 + (z - zs - 1j * B * np.sin(ang)) ** 2)
    return 1j / 4 * hankel1(0, K0 * r)


def beam_flat(el_deg):
    a = np.radians(el_deg)
    norm = abs(csp(0.0, ZS, XS, ZS, a) - csp(0.0, ZS, XS, -ZS, -a))

    def f(x, z):
        return (csp(x, z, XS, ZS, a) - csp(x, z, XS, -ZS, -a)) / norm
    return f


def homog(z):
    return np.full(np.shape(z), K0 ** 2, dtype=complex)


def rel_err(u, ue, mask):
    return float(np.abs(u - ue)[mask].max() / np.abs(ue[mask]).max())


def run_sm(exact, scheme, x_size, z_size, dx, dz, terrain=None, top='nlbc', absorber_m=0.0, strength=0.0,
           beta=None, h_interest=None):
    terrain = terrain or ShiftMapTerrain.flat()
    xs = max(int(round(5.0 / dx)), 1)
    zs = max(int(round(1.0 / dz)), 1)
    t = time.time()
    sm = ShiftMapPropagator(FREQ, K0, homog, terrain, x_size, z_size, dx, dz, scheme=scheme, top=top,
                            absorber_m=absorber_m, absorber_strength=strength, x_grid_scale=xs,
                            z_grid_scale=zs, beta=beta)
    zeta = sm.zeta_grid()
    x, Z, u = sm.compute(exact(np.zeros_like(zeta), zeta + terrain.T(np.zeros(1))[0]))
    X = np.repeat(x[:, None], Z.shape[1], axis=1)
    ue = exact(X, Z) * np.exp(-1j * sm.beta * X)
    zeta_o = Z - terrain.T(x)[:, None]
    h = h_interest if h_interest is not None else z_size - 30.0
    mask = (zeta_o <= h) & (zeta_o > 0.5)
    return dict(err=rel_err(u, ue, mask), time_s=time.time() - t, x_n=sm.prop.x_n, z_n=sm.prop.z_n), (X, Z, u, ue)


def optimal_pade(theta_max_deg, precision, x_size):
    best = None
    for order in [(3, 4), (5, 6), (7, 8)]:
        beta, dx, dz = get_optimal_grid(K0 * np.sin(np.radians(theta_max_deg)), K0 * 0.9999, K0 * 1.0001,
                                        precision / x_size, dx_max=50.0, dz_max=1.0, propagator_order=order)
        if not np.isfinite(beta):
            continue
        cost = order[1] / (dx * dz)
        if best is None or cost < best[0]:
            best = (cost, order, beta, dx, dz)
    return best[1:]


# --------------------------------------------------------------------------- E1
def part_angles():
    res = []
    for el in (5, 15, 30, 50, 70):
        x_size = 300.0
        z_size = min(x_size * np.tan(np.radians(el)) + 150.0, 1100.0)
        exact = beam_flat(el)
        row = dict(elevation_deg=el)
        for name, scheme, dx, dz in (('na_typ', 'na', 0.5, 0.1), ('wa_typ', 'wa', 0.5, 0.1),
                                     ('na_fine', 'na', 0.125, 0.025), ('wa_fine', 'wa', 0.125, 0.025)):
            row[name], _ = run_sm(exact, scheme, x_size, z_size, dx, dz)
            print(el, name, row[name], flush=True)
        order, beta, dx, dz = optimal_pade(min(el + 8, 85), 1e-2, x_size)
        dx = 5.0 / max(round(5.0 / dx), 1) if dx < 5 else 5.0
        dz = 1.0 / max(round(1.0 / dz), 1)
        row['pade'], _ = run_sm(exact, ('pade', order), x_size, z_size, dx, dz, beta=beta)
        row['pade'].update(order=list(order), dx=dx, dz=dz)
        print(el, 'pade', row['pade'], flush=True)
        res.append(row)
    with open(os.path.join(FIG_DIR, 'shiftmap_comparison_angles.json'), 'w') as f:
        json.dump(res, f, indent=1)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    els = [r['elevation_deg'] for r in res]
    for key, style, lab in (('na_typ', 'C0--o', r'NA-FD, $\Delta x=\lambda/2$, $\Delta z=\lambda/10$'),
                            ('na_fine', 'C0-o', r'NA-FD, $\Delta x=\lambda/8$, $\Delta z=\lambda/40$'),
                            ('wa_typ', 'C1--s', r'WA-FD, $\Delta x=\lambda/2$, $\Delta z=\lambda/10$'),
                            ('wa_fine', 'C1-s', r'WA-FD, $\Delta x=\lambda/8$, $\Delta z=\lambda/40$'),
                            ('pade', 'k-^', 'rational Numerov–Padé, optimized grid')):
        ax.semilogy(els, [r[key]['err'] for r in res], style, label=lab, ms=5)
    ax.set_xlabel('beam elevation, deg')
    ax.set_ylabel('max relative error')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'shiftmap_comparison_angles.png'), dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- E2
def part_slopes():
    import terrain_transform_slope_exact as S
    res = []
    for slope_deg in (5.0, 10.0, 20.0, 30.0):
        s = np.tan(np.radians(slope_deg))
        exact = S.exact_field(slope_deg)
        x_max = 600.0 if slope_deg < 25 else 400.0
        relief = s * x_max
        z_zeta = max(400.0, 2.2 * relief + 100.0) + 0.0   # height above the ground, as in the TF runs
        row = dict(slope_deg=slope_deg)
        for name, scheme in (('na', 'na'), ('wa', 'wa')):
            r, _ = run_sm(lambda x, z: exact(x, z), scheme, x_max, z_zeta, 0.125, 0.025,
                          terrain=ShiftMapTerrain.linear(s), h_interest=z_zeta - 50.0)
            row[name] = r
            print(slope_deg, name, r, flush=True)
        out, _, _ = S.run_case(slope_deg, 1e-2, x_max=x_max)
        row['tf'] = out['tf']
        print(slope_deg, 'tf', out['tf'], flush=True)
        res.append(row)
    with open(os.path.join(FIG_DIR, 'shiftmap_comparison_slopes.json'), 'w') as f:
        json.dump(res, f, indent=1)


# --------------------------------------------------------------------------- E3
def part_absorber():
    H = 200.0
    res = []
    for el in (1.0, 3.0, 10.0, 30.0):
        exact = beam_flat(el)
        x_size = min(1.5 * H / np.tan(np.radians(el)), 20000.0)
        # a fixed fine grid: the propagation error (~1e-4) is far below the
        # boundary errors studied here; a taller domain with the NLBC gives the same field
        order, beta, dx, dz = (7, 8), K0, 2.5, 0.1
        row = dict(elevation_deg=el, order=list(order), dx=dx, dz=dz, x_size=x_size)
        row['nlbc'], _ = run_sm(exact, ('pade', order), x_size, H + 20.0, dx, dz, beta=beta, h_interest=H)
        print(el, 'nlbc', row['nlbc'], flush=True)
        row['absorber'] = []
        for dfrac in (0.25, 0.5, 1.0):
            for A in (1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0):
                r, _ = run_sm(exact, ('pade', order), x_size, H + dfrac * H, dx, dz, top='absorber',
                              absorber_m=dfrac * H, strength=A, beta=beta, h_interest=H)
                r.update(thickness_frac=dfrac, strength=A)
                row['absorber'].append(r)
                print(el, 'absorber', dfrac, A, r['err'], flush=True)
        res.append(row)
    with open(os.path.join(FIG_DIR, 'shiftmap_comparison_absorber.json'), 'w') as f:
        json.dump(res, f, indent=1)
    fig, axes = plt.subplots(1, len(res), figsize=(12, 3.4), sharey=True)
    for ax, row in zip(axes, res):
        for dfrac, col in zip((0.25, 0.5, 1.0), ('C0', 'C1', 'C2')):
            pts = [r for r in row['absorber'] if r['thickness_frac'] == dfrac]
            ax.loglog([r['strength'] for r in pts], [r['err'] for r in pts], 'o-', color=col,
                      label=f'absorber, $D={dfrac:g}H$')
        ax.axhline(row['nlbc']['err'], color='k', lw=2, label='discrete NLBC')
        ax.set_title(f"beam at {row['elevation_deg']:g}°", fontsize=9)
        ax.set_xlabel('absorber strength $A$')
        ax.grid(True, which='both', alpha=0.3)
    axes[0].set_ylabel('max relative error, $z<H$')
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'shiftmap_comparison_absorber.png'), dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    os.makedirs(FIG_DIR, exist_ok=True)
    parts = sys.argv[1:] or ['angles', 'slopes', 'absorber']
    for p in parts:
        {'angles': part_angles, 'slopes': part_slopes, 'absorber': part_absorber}[p]()
