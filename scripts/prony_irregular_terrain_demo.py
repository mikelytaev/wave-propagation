"""Prony multipath extraction over irregular dielectric terrain.

Configuration follows Section IV-B of:

    H. Zhou and A. Chabory, "An Extraction Method for the Multipath
    Characteristics of Simulated Tropospheric Propagation Channels",
    IEEE TAES, 2025.

A Gaussian source illuminates a smooth random terrain profile over a
dielectric ground (eps_r = 15, sigma = 0.5 S/m).  The RWP forward task
is run at four frequencies spaced by ``delta_f = 2 MHz``, and Prony's
method extracts two dominant propagation paths.  Path 0 captures the
direct (line-of-sight) component; path 1 captures the bundled
terrain-reflected / diffracted contribution.

Increase ``n_paths`` (and ``nf``) for richer scenarios — fitting more
paths than the channel actually supports causes Prony to split the
dominant signal across spurious extra paths.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pywaveprop.helmholtz_jax import PiecewiseLinearTerrainModel
from pywaveprop.rwp_jax import (
    GroundMaterial,
    RWPComputationalParams,
    RWPGaussSourceModel,
    TroposphereModel,
    rwp_forward_task,
)
from pywaveprop.utils.prony import prony_extract_field


C0 = 299792458.0


def random_terrain(*, x_grid_m, max_height_m=80.0, seed=2026):
    """Smooth multi-scale random terrain (paper Section IV-B).

    Superposes 1/N-weighted layers of cosine-interpolated random
    anchor heights at progressively finer scales.  Result is shifted
    to be non-negative and normalised so the peak equals
    ``max_height_m``.
    """
    rng = np.random.default_rng(seed)
    layers = []
    weights = []
    # Only smooth scales: avoid sub-resolution roughness that creates
    # alias-like speckle when sampled at the output grid.
    for n_anchors in (3, 5, 9):
        anchors_x = np.linspace(x_grid_m[0], x_grid_m[-1], n_anchors)
        anchors_h = rng.uniform(-1.0, 1.0, size=n_anchors)
        idx = np.searchsorted(anchors_x, x_grid_m, side="right") - 1
        idx = np.clip(idx, 0, n_anchors - 2)
        x0 = anchors_x[idx]
        x1 = anchors_x[idx + 1]
        t = (x_grid_m - x0) / np.maximum(x1 - x0, 1e-9)
        t = 0.5 * (1.0 - np.cos(np.pi * t))
        layer = anchors_h[idx] * (1.0 - t) + anchors_h[idx + 1] * t
        layers.append(layer)
        weights.append(1.0 / n_anchors)
    profile = sum(w * layer for w, layer in zip(weights, layers))
    profile = profile - profile.min()
    profile = profile * (max_height_m / max(profile.max(), 1e-12))
    return profile.astype(float)


def main():
    # --- Parameters.
    f0 = 1000e6
    delta_f = 2e6  # 1/df = 500 ns => up to 150 m of unambiguous excess path
    nf = 4
    n_paths = 2

    max_range_m = 4000.0
    max_height_m = 400.0
    src_height_m = 200.0
    beam_width_deg = 8.0

    # --- Synthetic dielectric terrain.
    x_terrain = np.linspace(0.0, max_range_m, 201)
    h_terrain = random_terrain(
        x_grid_m=x_terrain, max_height_m=70.0, seed=2026
    )
    terrain = PiecewiseLinearTerrainModel(x_grid_m=x_terrain, height=h_terrain)
    ground = GroundMaterial(eps=15.0, sigma=0.5)

    print(
        f"Terrain: {x_terrain.size} samples over {max_range_m / 1000:.1f} km, "
        f"max height = {h_terrain.max():.1f} m"
    )
    print(f"Ground: eps_r = {ground.eps}, sigma = {ground.sigma} S/m")

    # --- Run the JAX RWP forward task at every frequency.
    freqs = f0 + delta_f * np.arange(nf)
    print(f"Running {nf} forward simulations at "
          f"{freqs[0] * 1e-9:.4f} ... {freqs[-1] * 1e-9:.4f} GHz "
          f"(delta_f = {delta_f * 1e-6:g} MHz)")

    # Output spacing kept constant across frequencies.  dz must be a
    # small fraction of a wavelength (lambda ~= 0.3 m at 1 GHz) for the
    # downsampled output to faithfully represent the field; otherwise
    # the cross-range phase pattern aliases into the Prony fit.
    dx_out_m = 20.0
    dz_out_m = 0.5
    u_freq = None
    x_out = z_out = None
    for i, f in enumerate(freqs):
        src = RWPGaussSourceModel(
            freq_hz=float(f),
            height_m=src_height_m,
            beam_width_deg=beam_width_deg,
        )
        env = TroposphereModel(terrain=terrain, ground_material=ground)
        params = RWPComputationalParams(
            max_range_m=max_range_m,
            max_height_m=max_height_m,
            dx_m=dx_out_m,
            dz_m=dz_out_m,
        )
        print(f"  [{i + 1}/{nf}] {f * 1e-9:.4f} GHz", flush=True)
        field = rwp_forward_task(src, env, params)
        u = np.asarray(field.field)
        if u_freq is None:
            nx_out, nz_out = u.shape
            x_out = np.arange(nx_out) * dx_out_m
            z_out = np.arange(nz_out) * dz_out_m
            u_freq = np.empty((nf, nx_out, nz_out), dtype=complex)
        elif u.shape != u_freq.shape[1:]:
            raise RuntimeError(
                f"output grid drift across frequencies: expected "
                f"{u_freq.shape[1:]}, got {u.shape} at {f * 1e-9:.4f} GHz"
            )
        u_freq[i] = u
    n_x_out, n_z_out = u_freq.shape[1:]

    # --- Prony extraction.
    print(
        f"Running Prony extraction (n_paths={n_paths}) over "
        f"{n_x_out}x{n_z_out} spatial points..."
    )
    amps, delays = prony_extract_field(
        u_freq, f_start=float(freqs[0]), delta_f=delta_f, n_paths=n_paths
    )

    # --- Post-process: paper eq. (16) clustering applied per pixel.
    # When two recovered paths share nearly the same delay (e.g., very
    # close to the source where the field has not yet developed
    # frequency-dependent structure, or at grazing reflection points),
    # the Vandermonde fit is rank-deficient and the individual
    # amplitudes split into near-equal-and-opposite huge values that
    # sum back to u(f_start).  Merge such pairs into a single effective
    # path with summed amplitude and amplitude-weighted mean delay.
    if n_paths == 2:
        excess_m = (delays[1] - delays[0]) * C0
        cluster_threshold_m = 1.0  # paper's tau_min = 1/c0
        too_close = np.abs(excess_m) < cluster_threshold_m
        if np.any(too_close):
            w0 = np.abs(amps[0]) ** 2
            w1 = np.abs(amps[1]) ** 2
            wsum = w0 + w1
            with np.errstate(divide="ignore", invalid="ignore"):
                merged_tau = np.where(
                    wsum > 0,
                    (w0 * delays[0] + w1 * delays[1]) / wsum,
                    0.5 * (delays[0] + delays[1]),
                )
            merged_amp = amps[0] + amps[1]
            amps[0] = np.where(too_close, merged_amp, amps[0])
            amps[1] = np.where(too_close, 0.0, amps[1])
            delays[0] = np.where(too_close, merged_tau, delays[0])
            delays[1] = np.where(too_close, merged_tau, delays[1])
            frac = np.mean(too_close)
            print(f"  clustered {frac:.1%} of pixels (close-delay paths merged)")

    # --- Reconstruction error (eq. 20).
    n_idx = np.arange(nf)[:, None, None, None]
    phase = -2j * np.pi * n_idx * delta_f * delays[None, :, :, :]
    u_hat = np.sum(amps[None, :, :, :] * np.exp(phase), axis=1)
    num = np.sum(np.abs(u_freq - u_hat) ** 2, axis=0)
    den = np.sum(np.abs(u_freq) ** 2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rms_dB = 10.0 * np.log10(
            np.where(den > 0, num / np.maximum(den, 1e-300), 1e-30)
        )

    field_db = 20 * np.log10(np.abs(u_freq[0]) + 1e-30)
    field_max_db = float(field_db.max())
    # Mask any pixel where the total |u| at f_0 is more than 60 dB below
    # the peak — Prony cannot meaningfully fit there and the recovered
    # parameters are dominated by numerical noise.
    illuminated = field_db > field_max_db - 60.0

    def masked(arr_2d):
        return np.where(illuminated, arr_2d, np.nan)

    # --- Plot grid sized to ``n_paths``.
    # Row 0: |u|, RMS error, terrain profile (spans remaining columns)
    # Row 1: amplitude (dB) of each extracted path
    # Row 2: excess delay (rel. to path 0) of each extracted path
    ncols = max(n_paths, 3)
    fig = plt.figure(figsize=(4.5 * ncols, 11.5))
    gs = fig.add_gridspec(3, ncols)

    def overlay_terrain(ax):
        ax.fill_between(
            x_terrain / 1000.0, 0, h_terrain,
            color="k", alpha=1.0, zorder=5,
        )

    # Row 0.
    ax_u = fig.add_subplot(gs[0, 0])
    im = ax_u.pcolormesh(
        x_out / 1000.0, z_out, field_db.T,
        cmap="jet", shading="auto",
        vmin=field_max_db - 60, vmax=field_max_db,
    )
    ax_u.set_title(f"|u| (dB) at {freqs[0] * 1e-9:.4f} GHz")
    ax_u.set_xlabel("range, km")
    ax_u.set_ylabel("height, m")
    overlay_terrain(ax_u)
    fig.colorbar(im, ax=ax_u)

    ax_rms = fig.add_subplot(gs[0, 1])
    im = ax_rms.pcolormesh(
        x_out / 1000.0, z_out, masked(rms_dB).T,
        cmap="magma", shading="auto",
        vmin=-150, vmax=-20,
    )
    ax_rms.set_title(r"$\varepsilon_{RMS}$, dB (eq. 20)")
    ax_rms.set_xlabel("range, km")
    ax_rms.set_ylabel("height, m")
    overlay_terrain(ax_rms)
    fig.colorbar(im, ax=ax_rms)

    ax_terr = fig.add_subplot(gs[0, 2:ncols])
    ax_terr.fill_between(x_terrain / 1000.0, 0, h_terrain,
                         color="saddlebrown", alpha=0.6)
    ax_terr.plot(x_terrain / 1000.0, h_terrain, "k-", lw=1.0)
    ax_terr.set_xlim(0, max_range_m / 1000.0)
    ax_terr.set_xlabel("range, km")
    ax_terr.set_ylabel("elevation, m")
    ax_terr.set_title(
        f"Terrain (eps_r={ground.eps}, sigma={ground.sigma} S/m) — "
        f"source at z={src_height_m:.0f} m"
    )
    ax_terr.grid(True, alpha=0.3)

    # Use the field's own dB range for amplitude colormaps so the plots
    # show physically meaningful levels (Prony amplitudes can be slightly
    # larger than the total field at points where paths interfere
    # destructively, but should not exceed it by more than a few dB).
    amp_dB = 20 * np.log10(np.abs(amps) + 1e-30)
    amp_peak_dB = field_max_db
    excess_m = (delays - delays[0:1]) * C0  # excess relative to direct path
    # Clamp wildly inflated delays (near-singular fits at the unambiguity
    # boundary) for visualisation only.
    excess_m = np.clip(excess_m, -1.0 / delta_f * C0, 1.0 / delta_f * C0)
    delay_max_m = 1.0 / delta_f * C0 * 0.5  # half the unambiguous range

    # Row 1 & 2: amplitude and excess delay for each path.
    for p in range(n_paths):
        ax_a = fig.add_subplot(gs[1, p])
        amp_p_dB = amp_dB[p]
        im = ax_a.pcolormesh(
            x_out / 1000.0, z_out, masked(amp_p_dB).T,
            cmap="jet", shading="auto",
            vmin=amp_peak_dB - 60, vmax=amp_peak_dB,
        )
        ax_a.set_title(f"|a_{p}| (dB)")
        ax_a.set_xlabel("range, km")
        ax_a.set_ylabel("height, m")
        overlay_terrain(ax_a)
        fig.colorbar(im, ax=ax_a)

        ax_d = fig.add_subplot(gs[2, p])
        # Suppress delay where the path itself is weak — those values
        # are not meaningful.
        path_strong = amp_p_dB > amp_peak_dB - 40.0
        delay_mask = illuminated & path_strong
        im = ax_d.pcolormesh(
            x_out / 1000.0, z_out,
            np.where(delay_mask, excess_m[p], np.nan).T,
            cmap="viridis", shading="auto",
            vmin=0, vmax=delay_max_m,
        )
        label = r"$(\tau_0 - \tau_0)c_0$ = 0 m" if p == 0 else \
                rf"$(\tau_{p}-\tau_0)c_0$, m"
        ax_d.set_title(label)
        ax_d.set_xlabel("range, km")
        ax_d.set_ylabel("height, m")
        overlay_terrain(ax_d)
        fig.colorbar(im, ax=ax_d)

    fig.suptitle(
        "Prony multipath extraction over irregular dielectric terrain "
        f"(f0={f0 * 1e-9:.2f} GHz, delta_f={delta_f * 1e-6:g} MHz, "
        f"nf={nf}, n_paths={n_paths})"
    )
    fig.tight_layout()
    out_path = os.path.join(
        os.path.dirname(__file__), "prony_irregular_terrain_demo.png"
    )
    fig.savefig(out_path, dpi=120)
    print(f"Saved figure to {out_path}")

    # --- Summary stats over the illuminated region.
    median_err_dB = float(np.median(rms_dB[illuminated]))
    print(
        f"Median RMS reconstruction error (within 60 dB of peak): "
        f"{median_err_dB:.2f} dB"
    )
    for p in range(1, n_paths):
        path_strong = amp_dB[p] > amp_peak_dB - 40.0
        mask = illuminated & path_strong
        if not np.any(mask):
            continue
        print(
            f"Path {p} excess delay (illuminated, within 40 dB of peak): "
            f"min={excess_m[p, mask].min():.2f} m, "
            f"max={excess_m[p, mask].max():.2f} m, "
            f"median={np.median(excess_m[p, mask]):.2f} m"
        )


if __name__ == "__main__":
    main()
