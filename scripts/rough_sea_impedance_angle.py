"""
Angle dependence of the effective surface impedance of a rough sea.

For a coherent reflection coefficient R(psi) the equivalent Robin (impedance)
boundary condition u_z = i kappa u has kappa = k_z (R - 1) / (R + 1). A smooth
sea gives a constant kappa; Barrick (IEEE TAP 46(1), 1998) shows that the
roughness-modified impedance must also tend to a constant at grazing. Plotted:
smooth, Ament exp(-x), Miller-Brown exp(-x) I0(x) and Ament with Smith
shadowing (illuminated-height characteristic function, about the illuminated
mean), for H and V polarization at several wind speeds.

Usage:
    python scripts/rough_sea_impedance_angle.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid
from scipy.special import erfc, i0e

FREQ_HZ = 3e9
EPS, SIGMA = 70.0, 4.0
WINDS = [5.0, 10.0, 15.0]
PSI_DEG = np.logspace(-3, np.log10(30), 300)


def fresnel(psi, eps, pol):
    s, root = np.sin(psi), np.sqrt(eps - np.cos(psi) ** 2)
    return (s - root) / (s + root) if pol == 'H' else (eps * s - root) / (eps * s + root)


def shadowed_factor(psi, k, sigma_h, slope):
    """Ament factor over the illuminated heights (Smith monostatic shadowing)."""
    h = np.linspace(-8 * sigma_h, 8 * sigma_h, 8001)
    p = np.exp(-h ** 2 / 2 / sigma_h ** 2) / np.sqrt(2 * np.pi) / sigma_h
    log_cdf = np.log(np.maximum(1 - 0.5 * erfc(h / np.sqrt(2) / sigma_h), 1e-300))
    out = np.empty(len(psi), complex)
    for i, ps in enumerate(psi):
        v = np.tan(ps) / np.sqrt(2) / slope
        lam = 0.5 * (np.exp(-v ** 2) / (v * np.sqrt(np.pi)) - erfc(v))
        p_ill = (1 + lam) * p * np.exp(lam * log_cdf)
        mean = trapezoid(h * p_ill, h)
        out[i] = trapezoid(p_ill * np.exp(-2j * k * np.sin(ps) * (h - mean)), h)
    return out


def main():
    k = 2 * np.pi * FREQ_HZ / 3e8
    eps = EPS + 1j * SIGMA * 60 * 3e8 / FREQ_HZ
    psi = np.radians(PSI_DEG)
    kz = k * np.sin(psi)
    colors = {"Ament": "#c2571a", "Miller-Brown": "#4a6fa5", "Ament + Smith shadowing": "#2a9d6f"}
    styles = {5.0: ":", 10.0: "-", 15.0: "--"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, layout="constrained")
    for col, pol in enumerate("HV"):
        rf = fresnel(psi, eps, pol)
        curves = [("smooth sea", None, np.ones_like(psi))]
        for u in WINDS:
            sigma_h, slope = 0.0051 * u ** 2, np.sqrt(0.00316 * u)
            x = 2 * (sigma_h * kz) ** 2
            curves += [("Ament", u, np.exp(-x)), ("Miller-Brown", u, i0e(x)),
                       ("Ament + Smith shadowing", u, shadowed_factor(psi, k, sigma_h, slope))]
        for name, u, rho in curves:
            r = rf * rho
            kappa = kz * (r - 1) / (r + 1) / k
            kw = dict(color="black", lw=2.5) if u is None else \
                dict(color=colors[name], ls=styles[u], lw=1.8 if u == 10 else 1.2)
            label = name if u is None else f"{name}, U={u:g} m/s"
            axes[0, col].loglog(PSI_DEG, np.abs(kappa), label=label, **kw)
            axes[1, col].semilogx(PSI_DEG, np.degrees(np.angle(-kappa)), **kw)
        axes[0, col].set_title(f"{pol} polarization, {FREQ_HZ / 1e9:g} GHz")
        axes[0, col].set_ylabel(r"$|\kappa| / k$")
        axes[1, col].set_ylabel(r"arg$(-\kappa)$ (deg)")
        axes[1, col].set_xlabel("Grazing angle (deg)")
        for ax in axes[:, col]:
            ax.grid(alpha=0.3, which="both")
    axes[0, 0].legend(fontsize=7, ncol=2, loc="lower left")
    fig.suptitle(r"Effective impedance $\kappa = k_z(R-1)/(R+1)$: Barrick (1998) predicts a constant near grazing")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rough_sea_impedance_angle.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
