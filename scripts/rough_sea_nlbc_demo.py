"""
Rough sea surface in the rational (Pade) propagator via the lower nonlocal BC.

A 3 GHz H-polarized Gaussian beam hits a flat sea (eps=70, sigma=4 S/m) at 5 deg
grazing. The sea is either smooth (Fresnel reflection) or rough with the
Miller-Brown coherent reflection factor rho = exp(-x) I0(x), x = 2 (sigma_h k_z)^2.

The NLBC evaluates R at complex k_z on a contour loop. The analytic
continuation of rho grows off the real axis, so with the plain continuation
1 + R vanishes inside the loop and the result is wrong (fourth panel). Two
remedies, both validated against the plane-wave-spectrum solution (last panel):

* method='fit' (default): 1 + R~ = (1 + R_f) exp(K), zero-free by construction,
  K a polynomial fitted on the propagating range and kept moderate on the loop;
  the BC stays (almost) at the mean sea level;
* method='shift': the BC is lifted to the height z0 where
  R_f rho exp(2 i k_z z0) is bounded by 1 on the loop.

Usage:
    python scripts/rough_sea_nlbc_demo.py
"""
import math as fm
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pywaveprop.propagators._utils import reflection_coef
from pywaveprop.rwp_jax import (GroundMaterial, RWPComputationalParams, RWPGaussSourceModel, RoughSeaSurface,
                                TroposphereModel, rwp_forward_task)

FREQ_HZ = 3e9
HEIGHT_M = 30.0
ELEV_DEG = 5.0
SIGMA_H_M = 0.3
SEA = GroundMaterial(eps=70, sigma=4)


def exact_field(src, env, x_grid, z_grid):
    """Plane-wave spectrum solution over a flat surface (reflection about z = 0)."""
    k = 2 * np.pi * src.freq_hz / float(env.wave_speed_profile(0.0))
    k0 = 2 * np.pi * src.freq_hz / 3e8
    n = 2 ** 16
    zz = (np.arange(n) - n // 2) * 0.00625
    ww = np.sqrt(2 * np.log(2)) / (k0 * np.sin(np.radians(src.beam_width_deg) / 2))
    g = np.exp(-1j * k0 * np.sin(np.radians(src.elevation_angle_deg)) * zz) \
        * np.exp(-((zz - src.height_m) / ww) ** 2) / (np.sqrt(np.pi) * ww)
    kz = 2 * np.pi * np.fft.fftfreq(n, d=zz[1] - zz[0])
    spec = np.fft.fft(np.fft.ifftshift(g))
    theta = np.degrees(np.arcsin(np.minimum(np.abs(kz) / k, 1)))
    eps = env.ground_material.complex_permittivity(src.freq_hz)
    refl = np.array([reflection_coef(1, eps, 90 - t, src.polarization) for t in theta])
    if env.sea_roughness is not None:
        refl = refl * env.sea_roughness.factor(np.abs(kz))
    kx = np.sqrt(k ** 2 - kz ** 2 + 0j)
    out = np.empty((len(x_grid), len(z_grid)), complex)
    for i, x in enumerate(x_grid):
        ph = np.where(np.abs(kz) < k, np.exp(1j * (kx - k) * x), 0)
        inc = np.fft.fftshift(np.fft.ifft(spec * ph))
        ref = np.fft.fftshift(np.fft.ifft(spec * refl * ph))
        out[i] = np.interp(z_grid, zz, inc.real) + 1j * np.interp(z_grid, zz, inc.imag) \
            + np.interp(-z_grid, zz, ref.real) + 1j * np.interp(-z_grid, zz, ref.imag)
    return out


def main():
    src = RWPGaussSourceModel(freq_hz=FREQ_HZ, height_m=HEIGHT_M, beam_width_deg=3.0,
                              elevation_angle_deg=ELEV_DEG, polarization='H')
    params = RWPComputationalParams(max_range_m=2 * HEIGHT_M / fm.tan(fm.radians(ELEV_DEG)) + 100,
                                    max_height_m=2.5 * HEIGHT_M, max_angle_deg=ELEV_DEG + 5,
                                    x_output_points=200, z_output_points=300, precision=0.01)
    cases = {
        "smooth sea": None,
        "Miller-Brown, zero-free fit": RoughSeaSurface(SIGMA_H_M),
        "Miller-Brown, shifted BC": RoughSeaSurface(SIGMA_H_M, method='shift'),
        "Miller-Brown, plain continuation": RoughSeaSurface(SIGMA_H_M, method='shift', reference_height_m=0.0),
    }
    results = {}
    for name, rough in cases.items():
        env = TroposphereModel(M0=320, slope=0.0, ground_material=SEA, sea_roughness=rough)
        field = rwp_forward_task(src, env, params)
        x, z = np.asarray(field.x_grid), np.asarray(field.z_grid)
        u = np.asarray(field.field)[:len(x), :len(z)]
        ref = exact_field(src, env, x, z)
        after = slice(len(x) // 3, None)
        err = np.linalg.norm(np.abs(u[after]) - np.abs(ref[after])) / np.linalg.norm(np.abs(ref[after]))
        results[name] = (x, z, u, ref, err)
        print(f"{name}: BC height z0 = {z[0]:.2f} m, |u| rel. error vs exact {err:.2e}")

    fig, axes = plt.subplots(1, 5, figsize=(25, 4.5), layout="constrained")
    for ax, (name, (x, z, u, _, err)) in zip(axes, results.items()):
        m = ax.pcolormesh(x, z, 20 * np.log10(np.abs(u).T + 1e-30), shading="auto", cmap="viridis", vmin=-50, vmax=0)
        ax.set_title(f"{name}\nz0 = {z[0]:.2f} m, error vs exact {100 * err:.1f}%", fontsize=10)
        ax.set_xlabel("Range (m)")
        ax.set_ylim(0, 75)
    axes[0].set_ylabel("Height (m)")
    fig.colorbar(m, ax=axes[3], label="20 log|u| (dB)")

    # vertical cut through the reflected beam, x = 1.5 h / tan(elev)
    ax = axes[4]
    x_cut = 1.5 * HEIGHT_M / fm.tan(fm.radians(ELEV_DEG))
    styles = [("#4a6fa5", "smooth"), ("#c2571a", "rough, fit"), ("#2a9d6f", "rough, shift"),
              ("#8a8a8a", "rough, plain")]
    for (color, label), (x, z, u, ref, _) in zip(styles, results.values()):
        i = int(np.argmin(np.abs(x - x_cut)))
        ax.plot(20 * np.log10(np.abs(u[i]) + 1e-30), z, color=color, lw=2, label=f"PE, {label}")
        ax.plot(20 * np.log10(np.abs(ref[i]) + 1e-30), z, color=color, lw=1, ls="--")
    ax.plot([], [], color="black", lw=1, ls="--", label="exact (dashed)")
    ax.set_xlim(-60, 0)
    ax.set_ylim(0, 75)
    ax.set_xlabel("20 log|u| (dB)")
    ax.set_title(f"Cut at x = {x_cut:.0f} m", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rough_sea_nlbc_demo.png")
    plt.savefig(out, dpi=110, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
