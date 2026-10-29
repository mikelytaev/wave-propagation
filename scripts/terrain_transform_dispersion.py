"""Discrete dispersion analysis of the terrain-following rational scheme.

Produces the figures of Section "Discrete dispersion analysis" of
``papers/terrain_transform/main.tex``:

* ``disp_local_equivalence.png`` -- error of the discrete longitudinal
  wavenumber versus the local propagation angle for several metric values
  ``(h_xi, h_eta)`` on one computational grid, overlaid with the untransformed
  scheme on the local effective grid ``(beta / h_xi, h_xi dxi, h_eta deta)``;
* ``disp_orders.png`` -- the same error for Padé orders (3,4) ... (10,11) on the
  grids selected by the optimizer for a 60 degree aperture (any order works
  unchanged in transformed coordinates);
* ``disp_shear_asymmetry.png`` -- the residual asymmetry of the incident /
  reflected vertical wavenumbers in sheared (Beilis--Tappert) coordinates, which
  no function of the vertical operator can represent.

Run::

    python scripts/terrain_transform_dispersion.py
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pywaveprop.propagators._utils import pade_propagator_coefs  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "papers", "terrain_transform", "figures")
ALPHA = 1.0 / 12


def propagator_symbol(theta, *, k, beta, dxi, deta, h_xi, h_eta, coefs):
    """One-step amplification ``R(L_h)`` of the transformed scheme for a plane
    wave with local physical angle ``theta`` to the marching direction."""
    r = h_eta / h_xi
    q = k * np.sin(theta) * h_eta
    s2 = np.sin(q * deta / 2) ** 2
    het = (k * h_xi) ** 2 / beta ** 2 - 1
    L = het - 4.0 / ((beta * deta * r) ** 2) * s2 / (1 - 4 * ALPHA * s2)
    R = np.ones_like(L, dtype=complex)
    for a, b in coefs:
        R *= (1 + a * L) / (1 + b * L)
    return R


def dispersion_error(theta, *, k, beta, dxi, deta, h_xi, h_eta, coefs):
    """``|k_x^h - k_x|`` per unit physical length (phase error per step / step),
    measured without the 2 pi ambiguity of the discrete logarithm."""
    R = propagator_symbol(theta, k=k, beta=beta, dxi=dxi, deta=deta, h_xi=h_xi, h_eta=h_eta, coefs=coefs)
    exact = np.exp(1j * dxi * (k * h_xi * np.cos(theta) - beta))
    return np.abs(np.log(R / exact)) / (dxi * h_xi)


def fig_local_equivalence():
    k = 2 * np.pi
    dxi, deta = 1.0, 0.1
    beta = k
    coefs = pade_propagator_coefs(pade_order=(7, 8), beta=beta, dx=dxi)[0]
    theta = np.radians(np.linspace(0, 80, 400))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for (h_xi, h_eta), col in zip([(1.0, 1.0), (1.15, 0.9), (0.85, 1.4), (1.0, 2.0)],
                                  ['k', 'C0', 'C1', 'C2']):
        err = dispersion_error(theta, k=k, beta=beta, dxi=dxi, deta=deta, h_xi=h_xi, h_eta=h_eta, coefs=coefs)
        ax.semilogy(np.degrees(theta), err, color=col, lw=2,
                    label=rf'$h_\xi={h_xi}$, $h_\eta={h_eta}$')
        # untransformed scheme on the local effective grid
        b_eff, dx_eff, dz_eff = beta / h_xi, h_xi * dxi, h_eta * deta
        coefs_eff = pade_propagator_coefs(pade_order=(7, 8), beta=b_eff, dx=dx_eff)[0]
        err0 = dispersion_error(theta, k=k, beta=b_eff, dxi=dx_eff, deta=dz_eff, h_xi=1.0, h_eta=1.0,
                                coefs=coefs_eff)
        print('local equivalence: max abs difference / max error =', np.max(np.abs(err0 - err)) / np.max(err))
        ax.semilogy(np.degrees(theta), err0, color=col, ls='--', lw=1)
    ax.set_xlabel(r'local propagation angle $\theta$, deg')
    ax.set_ylabel(r'$|k_x^{h} - k_x|$ per metre of path, rad/m')
    ax.set_title(r'Padé (7,8), $\Delta\xi=\lambda$, $\Delta\eta=\lambda/10$; dashed: flat scheme on local grid')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'disp_local_equivalence.png'), dpi=150)
    plt.close(fig)


def fig_orders():
    from pywaveprop.grid_optimizer import get_optimal_grid
    k = 2 * np.pi
    theta_max = np.radians(60)
    theta = np.radians(np.linspace(0, 89, 500))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    h_xi, h_eta = 1.1, 1.3  # a strongly distorted cell
    for order, col in zip([(3, 4), (4, 5), (6, 7), (8, 9), (10, 11)], plt.cm.viridis(np.linspace(0, 0.9, 5))):
        # grid chosen for 1e-2 rad of phase error over 1000 wavelengths, in transformed units
        beta, dx, dz = get_optimal_grid(k * h_xi * np.sin(theta_max), k * h_xi, k * h_xi, 1e-2 / 1000,
                                        dx_max=50.0, dz_max=1.0, propagator_order=order)
        if not np.isfinite(beta):
            continue
        deta = dz / (h_eta / h_xi)
        coefs = pade_propagator_coefs(pade_order=order, beta=beta, dx=dx)[0]
        err = dispersion_error(theta, k=k, beta=beta, dxi=dx, deta=deta, h_xi=h_xi, h_eta=h_eta, coefs=coefs)
        ax.semilogy(np.degrees(theta), err, color=col, lw=1.8,
                    label=f'{order}: $\\Delta\\xi={dx:.2f}\\lambda$, $\\Delta\\eta={deta:.3f}\\lambda$')
    ax.axvline(60, color="gray", ls=":")
    ax.axhline(1e-5, color="gray", ls=":")
    ax.set_ylim(1e-10, 1e0)
    ax.set_xlabel(r'local propagation angle $\theta$, deg')
    ax.set_ylabel(r'$|k_x^{h} - k_x|$, rad/$\lambda$')
    ax.set_title(r'grids for $\theta_{\max}=60^\circ$, $h_\xi=1.1$, $h_\eta=1.3$')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=7, loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'disp_orders.png'), dpi=150)
    plt.close(fig)


def fig_shear_asymmetry():
    alpha = np.radians(np.linspace(-30, 60, 400))
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for slope_deg, col in zip([2, 5, 10, 20, 30], plt.cm.plasma(np.linspace(0, 0.85, 5))):
        ts = np.radians(slope_deg)
        s = np.tan(ts)
        # (q+ + q-)/k after the Beilis-Tappert gauge exp(i k s zeta)
        asym = 2 * s * (np.cos(alpha - ts) * np.cos(ts) - 1)
        ax.semilogy(np.degrees(alpha - ts), np.abs(asym), color=col, label=f'slope {slope_deg}°')
    ax.set_xlabel(r'angle to the boundary $\alpha-\theta_s$, deg')
    ax.set_ylabel(r'$|q_+ + q_- - 2ks|\,/\,k$')
    ax.set_ylim(1e-6, 1)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title('sheared coordinates: incident/reflected pair is not symmetric')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'disp_shear_asymmetry.png'), dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    os.makedirs(FIG_DIR, exist_ok=True)
    fig_local_equivalence()
    fig_orders()
    fig_shear_asymmetry()
    print('figures written to', os.path.abspath(FIG_DIR))
