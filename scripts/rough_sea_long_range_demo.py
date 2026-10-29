"""
Long-range propagation over a rough sea with the rough-surface lower NLBC.

Sweeps frequency (L, S, X band), wind speed (sea state) and refractivity
(standard atmosphere, evaporation duct, surface-based duct) for a 20 m high
H-polarized shipborne antenna over 60 km of open sea. Roughness uses the
Miller-Brown coherent reflection factor with the rms wave height of a fully
developed sea, sigma = 0.0051 U^2 (RoughSeaSurface.from_wind_speed), inside the
rational (Pade) JAX propagator; the lower NLBC uses the zero-free fitted
reflection coefficient (RoughSeaSurface method='fit').

Wind speed is varied independently of the refractivity profile here, to
isolate the effect of the surface; in reality it also shapes the evaporation
duct.

Beyond the radio horizon in the standard atmosphere the field is carried by
creeping modes and by the lateral wave of the branch point at grazing
incidence. Miller-Brown applied down to grazing makes the surface impedance
odd in k_z there, so the model itself predicts a slowly decaying lateral wave:
far beyond the horizon the rough-sea field exceeds the smooth one by tens of dB
at 10-15 m/s (X band). The PE reproduces the independent modal + branch-cut
solution, so this is the model's behaviour, shaded in the excess-loss figure.

Outputs (next to this script):
    rough_sea_long_range_maps_{standard,evaporation,surface}.png
                                      range-height path-loss maps, frequency x wind
    rough_sea_long_range_cuts.png     path loss vs range at the receiver height
    rough_sea_long_range_excess.png   roughness-induced excess loss vs range
    rough_sea_long_range_summary.png  roughness-induced excess loss vs wind speed

Usage:
    python scripts/rough_sea_long_range_demo.py [--workers 4] [--cache-dir DIR]

Each case is cached as .npz in the cache directory, so re-plotting is cheap.
"""
import argparse
import math as fm
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

FREQS_HZ = {'L band, 1 GHz': 1e9, 'S band, 3 GHz': 3e9, 'X band, 10 GHz': 10e9}
WINDS_M_S = [0.0, 5.0, 10.0, 15.0]
PROFILES = ['standard atmosphere', 'evaporation duct 15 m', 'surface-based duct']
ANTENNA_HEIGHT_M = 20.0
RECEIVER_HEIGHT_M = 10.0
MAX_RANGE_M = 60e3
MAX_HEIGHT_M = 300.0
PLOT_HEIGHT_M = 150.0
# base and top of the trapping layer (M decreasing with height), marked on the maps
TRAPPING_LAYERS_M = {'surface-based duct': (100.0, 130.0)}
SEA = dict(eps=70.0, sigma=4.0)

# colours (dataviz reference palette): wind speed is ordinal -> blue ramp, smooth sea
# is a neutral dashed reference; frequency bands are categorical slots 1-3
WIND_COLORS = {0.0: '#52514e', 5.0: '#86b6ef', 10.0: '#2a78d6', 15.0: '#0d366b'}
FREQ_COLORS = ['#2a78d6', '#eb6834', '#1baf7a']
LOSS_RAMP = ['#0d366b', '#184f95', '#256abf', '#3987e5', '#6da7ec', '#9ec5f4', '#cde2fb', '#fcfcfb']


def n_profile(name):
    from pywaveprop.rwp_jax import EmptyNProfileModel, EvaporationDuctModel, PiecewiseLinearNProfileModel
    import jax.numpy as jnp
    if name == 'standard atmosphere':
        return EmptyNProfileModel()
    if name == 'evaporation duct 15 m':
        return EvaporationDuctModel(height_m=15.0)
    # surface-based duct: the model's standard gradient (0.157 M/m, as for the
    # standard atmosphere here) up to a 30 m trapping layer at 100 m losing
    # 30 M units (M at its top below the surface value), standard above
    z = jnp.array([0.0, 100.0, 130.0, 301.0])
    m = jnp.array([0.0, 15.7, -14.3, -14.3 + 0.157 * 171.0])
    return PiecewiseLinearNProfileModel.create_from_M_profile(z, m)


def case_key(freq_hz, wind, profile):
    return f"f{freq_hz / 1e9:g}GHz_U{wind:g}_{profile.replace(' ', '_')}"


def run_case(args):
    freq_hz, wind, profile, cache_dir = args
    path = os.path.join(cache_dir, case_key(freq_hz, wind, profile) + '.npz')
    if os.path.exists(path):
        return path
    from pywaveprop.rwp_jax import (GroundMaterial, RWPComputationalParams, RWPGaussSourceModel, RoughSeaSurface,
                                    TroposphereModel, create_rwp_model)
    from pywaveprop.rwp_field import RWPField
    src = RWPGaussSourceModel(freq_hz=freq_hz, height_m=ANTENNA_HEIGHT_M, beam_width_deg=2.0,
                              elevation_angle_deg=0.0, polarization='H')
    rough = RoughSeaSurface.from_wind_speed(wind) if wind > 0 else None
    env = TroposphereModel(N_profile=n_profile(profile), ground_material=GroundMaterial(**SEA), sea_roughness=rough)
    params = RWPComputationalParams(max_range_m=MAX_RANGE_M, max_height_m=MAX_HEIGHT_M,
                                    x_output_points=600, z_output_points=300)
    t = time.time()
    model = create_rwp_model(src, env, params)
    t_build = time.time() - t
    t = time.time()
    u = np.asarray(model.compute(src.aperture(model.z_computational_grid())))
    t_compute = time.time() - t
    x, z = np.asarray(model.x_output_grid()), np.asarray(model.z_output_grid())
    field = RWPField(x_grid=x, z_grid=z, freq_hz=freq_hz, field=u[:len(x), :len(z)])
    keep = z <= PLOT_HEIGHT_M
    info = getattr(model, 'rough_sea_fit_info', None) or {}
    np.savez(path, x=x, z=z[keep], loss=field.path_loss().field[:, keep].astype(np.float32),
             sigma=rough.rms_height_m if rough else 0.0, z0=model.z_offset_m,
             fit_error=info.get('fit_error', np.nan), t_build=t_build, t_compute=t_compute,
             finite=np.isfinite(u).all())
    return path


def load(cache_dir):
    res = {}
    for f in FREQS_HZ.values():
        for w in WINDS_M_S:
            for p in PROFILES:
                d = np.load(os.path.join(cache_dir, case_key(f, w, p) + '.npz'))
                res[f, w, p] = {k: d[k] for k in d.files}
    return res


def loss_at(r, height):
    i = int(np.argmin(np.abs(r['z'] - height)))
    return r['loss'][:, i]


def plot(res, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    plt.rcParams.update({'axes.edgecolor': '#8a8a8a', 'axes.labelcolor': '#0b0b0b', 'xtick.color': '#52514e',
                         'ytick.color': '#52514e', 'axes.titlesize': 10, 'axes.labelsize': 9})
    cmap = LinearSegmentedColormap.from_list('loss', LOSS_RAMP)
    km = 1e-3

    # 1. range-height path-loss maps, one figure per refractivity profile
    for prof in PROFILES:
        ymax = PLOT_HEIGHT_M if prof == 'surface-based duct' else 100.0
        fig, axes = plt.subplots(len(FREQS_HZ), len(WINDS_M_S), figsize=(18, 9), sharex=True, sharey=True,
                                 layout='constrained')
        for i, (fname, f) in enumerate(FREQS_HZ.items()):
            for j, w in enumerate(WINDS_M_S):
                r = res[f, w, prof]
                ax = axes[i, j]
                m = ax.pcolormesh(r['x'] * km, r['z'], r['loss'].T, shading='auto', cmap=cmap, vmin=110, vmax=190,
                                  rasterized=True)
                if prof in TRAPPING_LAYERS_M:
                    for zl in TRAPPING_LAYERS_M[prof]:
                        ax.axhline(zl, color='#eb6834', lw=1, ls='--')
                sea = 'smooth sea' if w == 0 else f"U = {w:g} m/s, σ = {float(r['sigma']):.2f} m"
                ax.set_title(f"{fname} · {sea}")
                ax.set_ylim(0, ymax)
                if i == len(FREQS_HZ) - 1:
                    ax.set_xlabel('Range (km)')
                if j == 0:
                    ax.set_ylabel('Height (m)')
        fig.colorbar(m, ax=axes, label='Path loss (dB)', shrink=0.6)
        note = ' (dashed: trapping layer)' if prof in TRAPPING_LAYERS_M else ''
        fig.suptitle(f'Path loss over a rough sea, {prof}{note}, H-pol antenna at {ANTENNA_HEIGHT_M:g} m')
        slug = prof.split(' ')[0].replace('-based', '')
        fig.savefig(os.path.join(out_dir, f'rough_sea_long_range_maps_{slug}.png'), dpi=110)
        plt.close(fig)

    # 2. path loss vs range at the receiver height
    fig, axes = plt.subplots(len(FREQS_HZ), len(PROFILES), figsize=(16, 10), sharex=True, layout='constrained')
    for i, (fname, f) in enumerate(FREQS_HZ.items()):
        for j, p in enumerate(PROFILES):
            ax = axes[i, j]
            for w in WINDS_M_S:
                r = res[f, w, p]
                label = 'smooth sea' if w == 0 else f"U = {w:g} m/s (σ = {float(r['sigma']):.2f} m)"
                ax.plot(r['x'][1:] * km, loss_at(r, RECEIVER_HEIGHT_M)[1:], color=WIND_COLORS[w],
                        lw=1.5 if w == 0 else 2, ls='--' if w == 0 else '-', label=label)
            ax.set_title(f'{fname} · {p}')
            ax.set_ylim(100, 220)
            ax.invert_yaxis()
            ax.grid(alpha=0.25)
            if i == len(FREQS_HZ) - 1:
                ax.set_xlabel('Range (km)')
            if j == 0:
                ax.set_ylabel(f'Path loss at {RECEIVER_HEIGHT_M:g} m (dB)')
    axes[0, 0].legend(fontsize=8, loc='lower left')
    fig.suptitle(f'Path loss at {RECEIVER_HEIGHT_M:g} m height, antenna at {ANTENNA_HEIGHT_M:g} m (H-pol)')
    fig.savefig(os.path.join(out_dir, 'rough_sea_long_range_cuts.png'), dpi=110)
    plt.close(fig)

    # 3a. roughness-induced excess loss vs range at the receiver height
    fig, axes = plt.subplots(len(PROFILES), len(FREQS_HZ), figsize=(16, 10), sharex=True, sharey='row',
                             layout='constrained')
    horizon_km = 4.12 * (fm.sqrt(ANTENNA_HEIGHT_M) + fm.sqrt(RECEIVER_HEIGHT_M))  # 4/3 earth radio horizon
    for i, p in enumerate(PROFILES):
        for j, (fname, f) in enumerate(FREQS_HZ.items()):
            ax = axes[i, j]
            ref = res[f, 0.0, p]
            x = ref['x'] * km
            for w in WINDS_M_S[1:]:
                ex = loss_at(res[f, w, p], RECEIVER_HEIGHT_M) - loss_at(ref, RECEIVER_HEIGHT_M)
                # running median over 2 km: interference nulls would otherwise dominate
                n = max(1, int(2.0 / (x[1] - x[0])))
                smooth = np.array([np.median(ex[max(0, k - n):k + n + 1]) for k in range(len(ex))])
                ax.plot(x[1:], smooth[1:], color=WIND_COLORS[w], lw=2,
                        label=f"U = {w:g} m/s (σ = {float(res[f, w, p]['sigma']):.2f} m)")
            ax.axhline(0, color='#8a8a8a', lw=1)
            if p == 'standard atmosphere':
                ax.axvspan(horizon_km, x[-1], color='#8a8a8a', alpha=0.12, lw=0)
                ax.text(horizon_km + 1, 0.95, 'beyond horizon:\nMiller-Brown lateral wave', transform=ax.get_xaxis_transform(),
                        va='top', fontsize=8, color='#52514e')
            ax.set_title(f'{fname} · {p}')
            ax.grid(alpha=0.25)
            if i == len(PROFILES) - 1:
                ax.set_xlabel('Range (km)')
            if j == 0:
                ax.set_ylabel(f'Excess loss at {RECEIVER_HEIGHT_M:g} m (dB)')
    axes[0, 0].legend(fontsize=8, loc='lower left')
    fig.suptitle('Roughness-induced excess path loss vs smooth sea (2 km running median)')
    fig.savefig(os.path.join(out_dir, 'rough_sea_long_range_excess.png'), dpi=110)
    plt.close(fig)

    # 3. roughness-induced excess loss, averaged over 20-60 km at the receiver height
    fig, axes = plt.subplots(1, len(PROFILES), figsize=(15, 4.2), sharey=True, layout='constrained')
    for j, p in enumerate(PROFILES):
        ax = axes[j]
        ends = []
        for (fname, f), color in zip(FREQS_HZ.items(), FREQ_COLORS):
            ref = res[f, 0.0, p]
            sel = ref['x'] >= 20e3
            excess = [float(np.mean(loss_at(res[f, w, p], RECEIVER_HEIGHT_M)[sel] - loss_at(ref, RECEIVER_HEIGHT_M)[sel]))
                      for w in WINDS_M_S]
            ax.plot(WINDS_M_S, excess, color=color, lw=2, marker='o', ms=8, mec='#fcfcfb', mew=2, label=fname)
            ends.append(excess[-1])
        # end-of-line values, staggered so that close ones do not overlap
        for rank, i in enumerate(np.argsort(ends)):
            ax.annotate(f'{ends[i]:+.1f} dB', (WINDS_M_S[-1], ends[i]), textcoords='offset points',
                        xytext=(8, 11 * (rank - (len(ends) - 1) / 2)), va='center', fontsize=8, color='#52514e')
        ax.axhline(0, color='#8a8a8a', lw=1)
        ax.set_title(p)
        ax.set_xlabel('Wind speed (m/s)')
        ax.set_xticks(WINDS_M_S)
        ax.set_xlim(-0.5, 18)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(f'Excess loss vs smooth sea (dB)\nmean over 20-60 km at {RECEIVER_HEIGHT_M:g} m')
    axes[0].legend(fontsize=8)
    fig.savefig(os.path.join(out_dir, 'rough_sea_long_range_summary.png'), dpi=110)
    plt.close(fig)


def _pin_worker(counter, n_workers):
    """Give each worker its own block of cores: JAX creates its thread pool
    lazily and inherits the affinity (BLAS is limited through the environment,
    see main)."""
    with counter.get_lock():
        i = counter.value % n_workers
        counter.value += 1
    cores = sorted(os.sched_getaffinity(0))
    per = max(1, len(cores) // n_workers)
    os.sched_setaffinity(0, cores[i * per:(i + 1) * per] or cores)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--cache-dir', default=os.path.join(tempfile.gettempdir(), 'rough_sea_long_range_cache'))
    args = ap.parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)
    # heaviest (X band) first for better load balance
    jobs = [(f, w, p, args.cache_dir) for f in sorted(FREQS_HZ.values(), reverse=True) for w in WINDS_M_S
            for p in PROFILES]
    import multiprocessing as mp
    # Parallel workers must share the cores: OpenBLAS threads busy-wait, and
    # 4 workers x all-core thread pools made the least-squares fits ~10x slower.
    # Set before the pool starts, so that spawned workers load BLAS limited.
    threads = str(max(1, os.cpu_count() // args.workers))
    for var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(var, threads)
    ctx = mp.get_context('spawn')
    counter = ctx.Value('i', 0)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx, initializer=_pin_worker,
                             initargs=(counter, args.workers)) as pool:
        for path in pool.map(run_case, jobs):
            print('done', os.path.basename(path), flush=True)
    res = load(args.cache_dir)
    print(f"{'case':<48} {'sigma':>6} {'z0':>6} {'fit err':>8} {'build':>6} {'compute':>8}")
    for (f, w, p), r in res.items():
        print(f"{case_key(f, w, p):<48} {float(r['sigma']):6.2f} {float(r['z0']):6.2f} {float(r['fit_error']):8.1e} "
              f"{float(r['t_build']):5.0f}s {float(r['t_compute']):7.0f}s" + ('' if r['finite'] else '  NON-FINITE'))
    plot(res, os.path.dirname(os.path.abspath(__file__)))
    print('figures saved to', os.path.dirname(os.path.abspath(__file__)))


if __name__ == '__main__':
    main()
