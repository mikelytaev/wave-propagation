"""
Miller-Brown vs Miller-Brown with a constant grazing impedance (Barrick 1998)
over 60 km of sea.

Miller-Brown applied down to grazing gives an effective surface impedance that
falls like 1/k_z near grazing (odd in k_z), and the model predicts a lateral
wave that raises the field far beyond the horizon above the smooth-sea one.
RoughSeaSurface(grazing_angle_deg='rayleigh') keeps the impedance constant
below the angle where the Rayleigh parameter 2 k_z sigma reaches 1 (see
scripts/rough_sea_impedance_angle.py). A 20 m H-polarized antenna; path loss
at a 10 m receiver for S band in a standard atmosphere and a 15 m evaporation
duct, and X band in a standard atmosphere.

Far beyond the horizon the blend follows the creeping modes of the constant
impedance (checked against the Airy-mode attenuation), except at X band and
15 m/s, where the blend angle lies inside the creeping-mode range and the field
there depends on the blend shape (RoughSeaSurface warns).

Output: rough_sea_grazing_blend_demo.png (next to this script).

Usage:
    python scripts/rough_sea_grazing_blend_demo.py [--workers 5] [--cache-dir DIR]
"""
import argparse
import multiprocessing as mp
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np

SETUPS = [(3e9, 'standard'), (3e9, 'evaporation'), (10e9, 'standard')]
WINDS_M_S = [10.0, 15.0]
RECEIVER_HEIGHT_M = 10.0


def cases():
    out = []
    for f, prof in SETUPS:
        out.append((f, prof, 0.0, None))
        out += [(f, prof, u, g) for u in WINDS_M_S for g in (None, 'rayleigh')]
    return out


def key(case):
    f, prof, u, g = case
    return f"f{f / 1e9:g}_{prof}_U{u:g}_{'blend' if g else 'mb'}"


def run(args):
    case, cache_dir = args
    path = os.path.join(cache_dir, key(case) + '.npz')
    if os.path.exists(path):
        return path
    from pywaveprop.rwp_field import RWPField
    from pywaveprop.rwp_jax import (EvaporationDuctModel, GroundMaterial, RWPComputationalParams, RWPGaussSourceModel,
                                    RoughSeaSurface, TroposphereModel, create_rwp_model)
    f, prof, u, g = case
    src = RWPGaussSourceModel(freq_hz=f, height_m=20.0, beam_width_deg=2.0, elevation_angle_deg=0.0, polarization='H')
    rough = RoughSeaSurface.from_wind_speed(u, grazing_angle_deg=g) if u > 0 else None
    kw = dict(N_profile=EvaporationDuctModel(height_m=15.0)) if prof == 'evaporation' else {}
    env = TroposphereModel(ground_material=GroundMaterial(70.0, 4.0), sea_roughness=rough, **kw)
    params = RWPComputationalParams(max_range_m=60e3, max_height_m=300.0, x_output_points=600, z_output_points=300)
    model = create_rwp_model(src, env, params)
    u_field = np.asarray(model.compute(src.aperture(model.z_computational_grid())))
    x, z = np.asarray(model.x_output_grid()), np.asarray(model.z_output_grid())
    field = RWPField(x_grid=x, z_grid=z, freq_hz=f, field=u_field[:len(x), :len(z)])
    keep = z <= 150
    info = getattr(model, 'rough_sea_fit_info', None) or {}
    np.savez(path, x=x, z=z[keep], loss=field.path_loss().field[:, keep].astype(np.float32), z0=model.z_offset_m,
             fit_error=info.get('fit_error', np.nan), kappa_g=info.get('grazing_impedance', np.nan))
    return path


def _pin(counter, n):
    # share the cores between workers (busy-waiting BLAS threads oversubscribe)
    with counter.get_lock():
        i = counter.value % n
        counter.value += 1
    cores = sorted(os.sched_getaffinity(0))
    per = max(1, len(cores) // n)
    os.sched_setaffinity(0, cores[i * per:(i + 1) * per] or cores)


def loss_at(d, height):
    return d['loss'][:, int(np.argmin(np.abs(d['z'] - height)))]


def plot(cache_dir, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    res = {c: np.load(os.path.join(cache_dir, key(c) + '.npz')) for c in cases()}
    colors = {10.0: '#2a78d6', 15.0: '#0d366b'}
    fig, axes = plt.subplots(len(SETUPS), 2, figsize=(13, 11), sharex=True, layout='constrained')
    for i, (f, prof) in enumerate(SETUPS):
        smooth = res[f, prof, 0.0, None]
        x = smooth['x'] * 1e-3
        l0 = loss_at(smooth, RECEIVER_HEIGHT_M)
        axes[i, 0].plot(x, l0, color='#52514e', ls='--', lw=1.5, label='smooth sea')
        for u in WINDS_M_S:
            for g, ls, name in ((None, ':', 'Miller-Brown'), ('rayleigh', '-', 'MB + constant grazing impedance')):
                lu = loss_at(res[f, prof, u, g], RECEIVER_HEIGHT_M)
                kw = dict(color=colors[u], ls=ls, lw=1.6, label=f'{name}, U = {u:g} m/s')
                axes[i, 0].plot(x, lu, **kw)
                axes[i, 1].plot(x, lu - l0, **kw)
        title = f"{f / 1e9:g} GHz, {'standard atmosphere' if prof == 'standard' else '15 m evaporation duct'}"
        axes[i, 0].set_title(f'{title}: path loss at {RECEIVER_HEIGHT_M:g} m')
        axes[i, 1].set_title(f'{title}: excess loss over the smooth sea')
        axes[i, 0].set_ylabel('Path loss (dB)')
        axes[i, 1].set_ylabel('Rough - smooth (dB)')
        axes[i, 0].invert_yaxis()
        axes[i, 1].axhline(0, color='#8a8a8a', lw=0.8)
        for ax in axes[i]:
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel('Range (km)')
    fig.suptitle('Rough sea, H-pol antenna at 20 m: Miller-Brown with and without a constant grazing impedance')
    fig.savefig(out, dpi=110)
    print(f'saved {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=5)
    ap.add_argument('--cache-dir', default=os.path.join(tempfile.gettempdir(), 'rough_sea_grazing_blend_cache'))
    a = ap.parse_args()
    os.makedirs(a.cache_dir, exist_ok=True)
    threads = str(max(1, os.cpu_count() // a.workers))
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(v, threads)
    ctx = mp.get_context('spawn')
    counter = ctx.Value('i', 0)
    with ProcessPoolExecutor(a.workers, mp_context=ctx, initializer=_pin, initargs=(counter, a.workers)) as ex:
        list(ex.map(run, [(c, a.cache_dir) for c in cases()]))
    plot(a.cache_dir, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rough_sea_grazing_blend_demo.png'))


if __name__ == '__main__':
    main()
