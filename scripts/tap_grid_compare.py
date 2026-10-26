"""Optimal grids chosen by the package optimizer for the staircase and the
terrain-following propagators, over terrain steepness, frequency, precision,
transition height and aperture.  No march is run: the propagator constructors
are intercepted after grid selection.

Both solvers get the same physical aperture (beam + tilt + slope, as in the
Glen Etive study).  Cost = x_n * z_n * (denominator degree).

    python scripts/tap_grid_compare.py
"""
import contextlib
import io
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pywaveprop import helmholtz_jax, helmholtz_tf_jax  # noqa: E402
from pywaveprop.helmholtz_jax import PiecewiseLinearTerrainModel  # noqa: E402
from pywaveprop.rwp_jax import (GroundMaterial, PiecewiseLinearNProfileModel, RWPComputationalParams,  # noqa: E402
                                RWPGaussSourceModel, TroposphereModel, create_rwp_model)
from pywaveprop.terrain_transform import OrthogonalTerrainMap  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "papers", "terrain_transform", "figures", "grid_compare.json")
GLEN = os.path.join(os.path.dirname(__file__), "..", "papers", "terrain_transform", "figures", "glen_etive_profile.npz")


class _Grid(Exception):
    pass


def _capture(self, **kw):
    raise _Grid(kw)


helmholtz_jax.RationalHelmholtzPropagator.__init__ = _capture
_RAW = {}


def _logged(fn):
    def wrap(*a, **k):
        res = fn(*a, **k)
        _RAW[tuple(k['propagator_order'])] = tuple(float(v) for v in res)
        return res
    return wrap


helmholtz_jax.get_optimal_grid = _logged(helmholtz_jax.get_optimal_grid)
helmholtz_tf_jax.get_optimal_grid = _logged(helmholtz_tf_jax.get_optimal_grid)
helmholtz_tf_jax.TerrainFollowingHelmholtzPropagator.build_tables = staticmethod(lambda *a, **k: None)


def grid(src, env, params):
    buf = io.StringIO()
    _RAW.clear()
    try:
        with contextlib.redirect_stdout(buf):
            create_rwp_model(src, env, params)
    except _Grid as e:
        kw = e.args[0]
    except ValueError as e:
        return dict(fail=str(e)[:300])
    log = buf.getvalue()
    out = dict(order=list(kw['order']), beta=float(kw['beta']), dx=float(kw['dx_m']), dz=float(kw['dz_m']),
               x_n=int(kw['x_n']), z_n=int(kw['z_n']))
    out['cost'] = out['x_n'] * out['z_n'] * out['order'][1]
    raw = _RAW[tuple(kw['order'])]
    out['raw'] = dict(beta=raw[0], dx=raw[1], dz=raw[2])  # before snapping; TF: dz before the metric rescaling
    m = re.search(r'h_xi in \[([\d.]+), ([\d.]+)\], r in \[([\d.]+), ([\d.]+)\]', log)
    if m:
        out['hxi'] = [float(m.group(1)), float(m.group(2))]
        out['r'] = [float(m.group(3)), float(m.group(4))]
    m = re.search(r'max slope ([\d.]+) deg', log)
    if m:
        out['slope'] = float(m.group(1))
    return out


def hill(height, sigma, x_max=10e3, x0=5e3):
    x = np.linspace(0, x_max, 2001)
    return x, height * np.exp(-0.5 * ((x - x0) / sigma) ** 2)


def case(x, h, *, freq=100e6, prec=1e-2, beam=10.0, tilt=3.0, ant=30.0, smooth=100.0, transition=None,
         z_top=None, dx_out=25.0, dz_out=1.0, ground=GroundMaterial(eps=25.0, sigma=0.02)):
    relief = float(h.max() - h.min())
    transition = transition or max(2.0 * relief, 30.0 * 3e8 / freq, 4.0 * smooth)
    z_top = z_top or float(h.max()) + 1000.0
    tm = OrthogonalTerrainMap(x, h, x_max_m=x[-1], transition_m=transition, smoothing_m=smooth)
    slope = tm.max_slope_deg()
    aperture = min(beam + abs(tilt) + slope, 85.0)
    z = np.array([0.0, z_top + 400.0])
    env = TroposphereModel(N_profile=PiecewiseLinearNProfileModel.create_from_M_profile(z, 0.118 * z), M0=0.0,
                           slope=0.118, terrain=PiecewiseLinearTerrainModel(x, h), ground_material=ground)
    src = RWPGaussSourceModel(freq_hz=freq, height_m=float(h[0]) + ant, beam_width_deg=beam,
                              elevation_angle_deg=tilt, polarization='V')
    res = dict(freq=freq, prec=prec, beam=beam, relief=relief, slope=slope, aperture=aperture,
               transition=transition, smooth=smooth)
    for method in ('staircase', 'transform'):
        p = RWPComputationalParams(max_range_m=float(x[-1]), max_height_m=z_top, max_angle_deg=aperture,
                                   dx_m=dx_out, dz_m=dz_out)
        p.precision = prec
        p.terrain_method = method
        p.terrain_smoothing_m = smooth
        p.terrain_transition_m = transition
        p.terrain_slope_aperture = False
        res[method] = grid(src, env, p)
    st, tf = res['staircase'], res['transform']
    if 'cost' in st and 'cost' in tf:
        res['cost_ratio'] = tf['cost'] / st['cost']
        # the same ratio without snapping to the output grid
        tf['raw']['deta'] = tf['raw']['dz'] * tf['hxi'][0] / tf['hxi'][1] / tf['r'][1]
        z_st, z_tf = st['z_n'] * st['dz'], tf['z_n'] * tf['dz']
        c_st = z_st / (st['raw']['dx'] * st['raw']['dz']) * st['order'][1]
        c_tf = z_tf / (tf['raw']['dx'] * tf['raw']['deta']) * tf['order'][1]
        res['cost_ratio_raw'] = c_tf / c_st
    return res


def glen():
    if not os.path.exists(GLEN):
        from pywaveprop.environment import get_troposphere_model
        env0 = get_troposphere_model(56.6064, -4.9986, 56.645, -5.05, n_points=800, resolution=0.0005,
                                     use_landcover=False)
        np.savez(GLEN, x=np.asarray(env0.terrain.x_grid_m), h=np.asarray(env0.terrain.height))
    d = np.load(GLEN)
    return d['x'], d['h']


def main():
    gx, gh = glen()
    S = {}
    S['glen_etive'] = [case(gx, gh, transition=1400.0, z_top=2000.0)]
    # steepness: Gaussian hill, sigma 500 m, 100 MHz
    S['steepness'] = [case(*hill(H, 500.0)) for H in (25, 50, 100, 200, 400, 800)]
    S['frequency'] = [case(gx, gh, freq=f, transition=1400.0, z_top=2000.0, dz_out=0.25 if f > 5e8 else 1.0)
                      for f in (30e6, 100e6, 300e6, 1e9, 3e9)]
    S['precision'] = [case(gx, gh, prec=e, transition=1400.0, z_top=2000.0) for e in (1e-1, 1e-2, 1e-3, 1e-4)]
    S['transition'] = [case(gx, gh, transition=L, z_top=max(2000.0, 750 + L + 200)) for L in (700, 1400, 2800, 5600)]
    S['beam'] = [case(gx, gh, beam=b, tilt=0.0, transition=1400.0, z_top=2000.0) for b in (2.0, 5.0, 10.0, 20.0)]
    with open(OUT, 'w') as f:
        json.dump(S, f, indent=1)
    for name, rows in S.items():
        print(f'\n== {name}')
        print(f'{"f MHz":>6} {"eps":>6} {"slope":>5} {"ap":>5} {"L":>6} | {"order":>5} {"dx":>7} {"dz":>6} '
              f'{"Mcell":>7} | {"order":>5} {"dxi":>7} {"deta":>6} {"Mcell":>7} {"hxi":>11} {"rmax":>5} | ratio  raw | raw st dx/dz, tf dxi/dz/deta')
        for r in rows:
            st, tf = r['staircase'], r['transform']
            def fmt(g):
                if 'fail' in g:
                    return f'{"fail":>36}'
                return (f'{g["order"][0]},{g["order"][1]:<3} {g["dx"]:7.2f} {g["dz"]:6.3f} '
                        f'{g["cost"] / 1e6:7.1f}')
            hx = f'{tf["hxi"][0]:.2f}-{tf["hxi"][1]:.2f}' if 'hxi' in tf else ''
            rm = f'{tf["r"][1]:.2f}' if 'r' in tf else ''
            print(f'{r["freq"] / 1e6:6.0f} {r["prec"]:6.0e} {r["slope"]:5.1f} {r["aperture"]:5.1f} '
                  f'{r["transition"]:6.0f} | {fmt(st)} | {fmt(tf)} {hx:>11} {rm:>5} | '
                  f'{r.get("cost_ratio", float("nan")):5.2f} {r.get("cost_ratio_raw", float("nan")):5.2f} | ' +
                  (f'{st["raw"]["dx"]:.2f}/{st["raw"]["dz"]:.3f}, {tf["raw"]["dx"]:.2f}/{tf["raw"]["dz"]:.3f}/'
                   f'{tf["raw"]["deta"]:.3f}' if 'cost_ratio' in r else tf.get('fail', '')))


if __name__ == '__main__':
    main()
