"""Roll the learned steps out over the test scenarios and time them.

Run::

    python evaluate.py lss_k2_s2 fno_s2 ...      # trained models (models/<name>.json)
    python evaluate.py --rational                # rational solver at coarser precisions

Writes ``results/eval_<name>.json`` (or ``results/rational.json``).
"""
import argparse
import json
import os
import time
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from common import DATA_DIR, DX_OUT, N_EVAL, reference, sample_scenario
from models import FreeStep, LearnedSplitStep, SymbolStep, symbol_basis
from train import HERE, feat_index, load, make_model

RES = os.path.join(HERE, "results")


def load_model(name):
    with open(os.path.join(HERE, "models", name + ".json")) as f:
        meta = json.load(f)
    args = SimpleNamespace(**meta["args"])
    skel = make_model(args, jax.random.PRNGKey(0))
    return eqx.tree_deserialise_leaves(os.path.join(HERE, "models", name + ".eqx"), skel), args.step


def errors(pred, ref):
    """Max-norm error relative to max |ref| and relative L2, over all marched columns (rows below 200 m)."""
    pred, ref = pred[..., :N_EVAL], ref[..., :N_EVAL]
    d = np.abs(pred - ref)
    return dict(max_rel=float(d.max() / np.abs(ref).max()),
                l2_rel=float(np.linalg.norm(pred - ref) / np.linalg.norm(ref)),
                l2_last=float(np.linalg.norm(pred[-1] - ref[-1]) / np.linalg.norm(ref[-1])))


def rollout_fn(model):
    """Jitted march over a whole scenario; screens of a split-step model are
    computed in a separate function (they depend on the terrain only)."""
    if isinstance(model, (LearnedSplitStep, SymbolStep)):
        screens = eqx.filter_jit(jax.vmap(model.screens))

        @eqx.filter_jit
        def march(v0, scr):
            def body(v, s):
                v1 = model.apply(v, s)
                return v1, v1
            return jax.lax.scan(body, v0, scr)[1]
        return screens, march

    @eqx.filter_jit
    def march(v0, feats):
        def body(v, f):
            v1 = model(v, f)
            return v1, v1
        return jax.lax.scan(body, v0, feats)[1]
    return None, march


def timed(fn, *a, reps=5):
    jax.block_until_ready(fn(*a))
    t = time.perf_counter()
    for _ in range(reps):
        out = jax.block_until_ready(fn(*a))
    return out, (time.perf_counter() - t) / reps


def eval_model(name, model, s, V, F):
    screens, march = rollout_fn(model)
    rows = []
    for i in range(V.shape[0]):
        n = np.arange(0, V.shape[1] - s, s)
        feats = jnp.asarray(F[i][feat_index(n, s)])
        v0 = jnp.asarray(V[i, 0])
        if screens is not None:
            scr, t_scr = timed(screens, feats)
            pred, t_march = timed(march, v0, scr)
        else:
            t_scr = 0.0
            pred, t_march = timed(march, v0, feats)
        ref = V[i, s::s][:len(n)]
        rows.append(dict(errors(np.asarray(pred), ref), t_screens=t_scr, t_march=t_march))
    out = dict(name=name, step_m=s * DX_OUT, n_scen=len(rows), rows=rows,
               median={k: float(np.median([r[k] for r in rows])) for k in rows[0]})
    print(name, json.dumps(out["median"]), flush=True)
    return out


def eval_rational(precisions=(1e-1, 3e-2, 1e-2), n_scen=6):
    files = sorted(f for f in os.listdir(DATA_DIR) if int(f[4:8]) >= 200)[:n_scen]
    rows = []
    for f in files:
        seed = int(f[4:8])
        ref = np.load(os.path.join(DATA_DIR, f))["v"].astype(complex)
        sc = sample_scenario(seed)
        for p in precisions:
            v, tf, _ = reference(sc, precision=p)
            # time the march alone (reference() already compiled it)
            init = jnp.asarray(sc.initial(tf.eta_computational_grid()))
            t = time.perf_counter()
            jax.block_until_ready(tf.compute(init))
            wall = time.perf_counter() - t
            e = errors(v[1:], ref[1:])
            rows.append(dict(seed=seed, precision=p, order=list(tf.order), dxi=tf.dx_m, deta=tf.dz_m,
                             z_n=tf.z_n, x_n=tf.x_n, t_march=wall, **e))
            print(rows[-1], flush=True)
    return dict(rows=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--rational", action="store_true")
    ap.add_argument("--baselines", action="store_true")
    args = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    if args.rational:
        with open(os.path.join(RES, "rational.json"), "w") as f:
            json.dump(eval_rational(), f, indent=1)
        return
    V, F = load("test")
    if args.baselines:
        out = []
        for s in (1, 2):
            out.append(eval_model(f"free_s{s}", FreeStep(s * DX_OUT), s, V, F))
            for k in (1,):
                m = LearnedSplitStep(s * DX_OUT, k, 6, jax.random.PRNGKey(0), use_net=False)
                out.append(eval_model(f"prior_k{k}_s{s}", m, s, V, F))
        Vtr, Ftr = load("train")
        for s, k, rank in ((1, 1, 64), (2, 1, 100)):
            basis = symbol_basis(Ftr[::10], s * DX_OUT / k, rank, cols_per_stage=5 * s // k)
            m = SymbolStep(s * DX_OUT, k, basis, jax.random.PRNGKey(0))
            out.append(eval_model(f"symbol_k{k}_s{s}_L{rank}", m, s, V, F))
        with open(os.path.join(RES, "baselines.json"), "w") as f:
            json.dump(out, f, indent=1)
    for name in args.names:
        model, s = load_model(name)
        res = eval_model(name, model, s, V, F)
        with open(os.path.join(RES, f"eval_{name}.json"), "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
