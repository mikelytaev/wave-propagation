"""Demonstrate Prony multipath extraction on the two-ray scenario.

This reproduces the validation of Section IV-A in:

    H. Zhou and A. Chabory, "An Extraction Method for the Multipath
    Characteristics of Simulated Tropospheric Propagation Channels",
    IEEE TAES, 2025.

A Gaussian beam from a complex source point propagates over flat PEC
ground in a homogeneous atmosphere.  The analytic two-ray solution is
evaluated at four frequencies spaced by ``delta_f``, and Prony's method
recovers the direct and ground-reflected paths from these few frequency
samples.  Recovery is validated against the geometric path-length
difference at the maximum-range vertical slice (the paper's Fig. 4).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Allow running directly from a checkout without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pywaveprop.rwp.antennas import GaussAntenna
from pywaveprop.rwp.environment import (
    PerfectlyElectricConducting,
    Terrain,
    Troposphere,
)
from pywaveprop.rwp.tworay import TwoRayModel
from pywaveprop.utils.prony import prony_extract_field

C0 = 299792458.0


def simulate_two_ray(*, freq_hz, x_grid_m, z_grid_m, src_height_m, beam_width_deg):
    """Compute the analytic two-ray field at a single frequency."""
    src = GaussAntenna(
        freq_hz=freq_hz,
        height=src_height_m,
        beam_width=beam_width_deg,
        elevation_angle=0.0,
        polarz="H",
    )
    env = Troposphere(flat=True)
    env.terrain = Terrain(ground_material=PerfectlyElectricConducting())
    return TwoRayModel(src, env).calculate(x_grid_m, z_grid_m)


def main():
    # --- Parameters (Table I/II in the paper, scaled down for runtime).
    f0 = 5e9
    delta_f = 0.5e6  # ambiguity-free up to 1/df = 2 us  =>  ~600 m of path
    nf = 4
    n_paths = 2
    src_height_m = 200.0
    beam_width_deg = 5.0
    max_range_m = 15000.0

    x_grid_m = np.linspace(1000.0, max_range_m, 150)
    z_grid_m = np.linspace(1.0, 500.0, 200)

    # --- Frequency-domain "simulations".
    freqs = f0 + delta_f * np.arange(nf)
    print(f"Computing two-ray field at {nf} frequencies: "
          f"{', '.join(f'{f * 1e-9:.4f} GHz' for f in freqs)}")
    u_freq = np.empty((nf, x_grid_m.size, z_grid_m.size), dtype=complex)
    for i, f in enumerate(freqs):
        u_freq[i] = simulate_two_ray(
            freq_hz=float(f),
            x_grid_m=x_grid_m,
            z_grid_m=z_grid_m,
            src_height_m=src_height_m,
            beam_width_deg=beam_width_deg,
        )

    # --- Prony extraction over the full (x, z) grid.
    print(f"Running Prony extraction (n_paths={n_paths}) over "
          f"{x_grid_m.size}x{z_grid_m.size} spatial points...")
    amps, delays = prony_extract_field(
        u_freq, f_start=float(freqs[0]), delta_f=delta_f, n_paths=n_paths
    )

    # --- Reconstruction error at the sampled frequencies (eq. 20).
    n_idx = np.arange(nf)[:, None, None, None]
    phase = -2j * np.pi * n_idx * delta_f * delays[None, :, :, :]
    u_hat = np.sum(amps[None, :, :, :] * np.exp(phase), axis=1)
    num = np.sum(np.abs(u_freq - u_hat) ** 2, axis=0)
    den = np.sum(np.abs(u_freq) ** 2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rms_dB = 10.0 * np.log10(
            np.where(den > 0, num / np.maximum(den, 1e-300), 1e-30)
        )

    # --- Geometric reference: path-length difference between rays.
    XG, ZG = np.meshgrid(x_grid_m, z_grid_m, indexing="ij")
    r_direct = np.sqrt(XG ** 2 + (src_height_m - ZG) ** 2)
    r_reflected = np.sqrt(XG ** 2 + (src_height_m + ZG) ** 2)
    excess_geom_m = r_reflected - r_direct
    excess_extracted_m = (delays[1] - delays[0]) * C0

    # --- Validation at the maximum-range vertical slice (paper Fig 4a, 4b).
    x_slice_idx = -1  # x = max_range_m
    z_slice = z_grid_m
    excess_geom_slice = excess_geom_m[x_slice_idx]
    excess_extracted_slice = excess_extracted_m[x_slice_idx]
    amp_reflected_slice_dB = 20 * np.log10(np.abs(amps[1, x_slice_idx]) + 1e-30)
    field_mag_slice_dB = 20 * np.log10(np.abs(u_freq[0, x_slice_idx]) + 1e-30)
    # Restrict the validation to altitudes where the second path is well
    # above the numerical floor (the paper's Fig. 4 covers the upper half
    # of the domain where direct and reflected contributions are
    # comparable).
    valid = amp_reflected_slice_dB > (amp_reflected_slice_dB.max() - 40)
    slice_err_m = np.max(np.abs(
        excess_extracted_slice[valid] - excess_geom_slice[valid]
    ))

    # --- Plots.
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0])

    # Top row: 2D maps.
    ax_field = fig.add_subplot(gs[0, 0])
    im_field = ax_field.pcolormesh(
        x_grid_m * 1e-3,
        z_grid_m,
        20 * np.log10(np.abs(u_freq[0]).T + 1e-30),
        cmap="jet",
        shading="auto",
        vmin=-90,
        vmax=-30,
    )
    ax_field.axvline(max_range_m * 1e-3, color="w", linestyle="--", linewidth=1)
    ax_field.set_title(f"|u| (dB) at {freqs[0] * 1e-9:.4f} GHz")
    ax_field.set_xlabel("range, km")
    ax_field.set_ylabel("height, m")
    fig.colorbar(im_field, ax=ax_field)

    ax_excess = fig.add_subplot(gs[0, 1])
    im_excess = ax_excess.pcolormesh(
        x_grid_m * 1e-3,
        z_grid_m,
        excess_extracted_m.T,
        cmap="viridis",
        shading="auto",
        vmin=0,
        vmax=10,
    )
    ax_excess.set_title(r"Extracted $(\tau_1 - \tau_0)\,c_0$, m")
    ax_excess.set_xlabel("range, km")
    ax_excess.set_ylabel("height, m")
    fig.colorbar(im_excess, ax=ax_excess)

    ax_rms = fig.add_subplot(gs[0, 2])
    im_rms = ax_rms.pcolormesh(
        x_grid_m * 1e-3,
        z_grid_m,
        rms_dB.T,
        cmap="magma",
        shading="auto",
        vmin=-200,
        vmax=-50,
    )
    ax_rms.set_title(r"$\varepsilon_{RMS}$, dB (eq. 20)")
    ax_rms.set_xlabel("range, km")
    ax_rms.set_ylabel("height, m")
    fig.colorbar(im_rms, ax=ax_rms)

    # Bottom row: validation at max-range slice (paper Fig 4).
    ax_d = fig.add_subplot(gs[1, 0])
    ax_d.plot(excess_geom_slice, z_slice, "k-", linewidth=2, label="geometric")
    ax_d.plot(excess_extracted_slice, z_slice, "r--", linewidth=1.5, label="Prony")
    ax_d.set_xlim(0, 10)
    ax_d.set_xlabel(r"$(\tau_1 - \tau_0)\,c_0$, m")
    ax_d.set_ylabel("height, m")
    ax_d.set_title(f"Reflected-path delay at x = {max_range_m * 1e-3:.0f} km")
    ax_d.legend()
    ax_d.grid(True, alpha=0.3)

    ax_a = fig.add_subplot(gs[1, 1])
    # Analytic reflected-ray amplitude over PEC ground: -1 / r_reflected
    # (Gaussian pattern factor included implicitly through the simulation).
    ax_a.plot(field_mag_slice_dB, z_slice, "k-", linewidth=2, label="|u| (total)")
    ax_a.plot(amp_reflected_slice_dB, z_slice, "r--", linewidth=1.5, label="|a_1| (reflected)")
    ax_a.set_xlabel("amplitude, dB")
    ax_a.set_ylabel("height, m")
    ax_a.set_title(f"Path 2 amplitude at x = {max_range_m * 1e-3:.0f} km")
    ax_a.legend()
    ax_a.grid(True, alpha=0.3)

    ax_e = fig.add_subplot(gs[1, 2])
    ax_e.plot(np.abs(excess_extracted_slice - excess_geom_slice), z_slice, "b-")
    ax_e.set_xscale("log")
    ax_e.set_xlabel(r"$|extracted - geometric|$, m")
    ax_e.set_ylabel("height, m")
    ax_e.set_title("Path-length recovery error vs height")
    ax_e.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        "Two-ray scenario: Prony extraction over PEC ground "
        f"(f0={f0 * 1e-9:.0f} GHz, df={delta_f * 1e-6:.2f} MHz, "
        f"nf={nf}, n_paths={n_paths})"
    )
    fig.tight_layout()

    out_path = os.path.join(os.path.dirname(__file__), "prony_two_ray_demo.png")
    fig.savefig(out_path, dpi=120)
    print(f"Saved figure to {out_path}")

    # --- Summary stats.
    field_mag_db = 20 * np.log10(np.abs(u_freq[0]) + 1e-30)
    illuminated = field_mag_db > field_mag_db.max() - 40.0
    median_err_dB = float(np.median(rms_dB[illuminated]))
    print(f"Median RMS reconstruction error (within 40 dB of peak): {median_err_dB:.2f} dB")
    print(
        f"Max |extracted - geometric| path-length difference at x={max_range_m*1e-3:.0f} km "
        f"(over heights where |a_1| within 40 dB of peak): {slice_err_m * 1000:.3f} mm"
    )


if __name__ == "__main__":
    main()
