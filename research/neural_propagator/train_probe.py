"""Operator (Sobolev-type) training of a learned 25 m step against the JAX oracle.

The loss combines
* trajectory pairs ``(v_j, v_{j+1})`` from the reference fields (``--w-traj``),
* forward probes ``(p, P_j p)``: random band-limited inputs (``--w-fwd``),
* adjoint probes ``(w, P_j^H w)``, compared with the model adjoint obtained by
  ``jax.vjp`` (``--w-adj``).
For a linear step the Jacobian with respect to the field is the operator
itself, so the forward and adjoint probes match it by columns and by rows.

Run::

    python train_probe.py --model nr --w-fwd 1 --w-adj 1 --name nr_sobolev
"""
import argparse
import glob
import json
import os
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from common import DX_OUT, N_EVAL
from models import n_params
from neural_rational import NeuralRationalStep
from oracle import PROBE_DIR
from train import HERE, feat_index, load, make_model, N_TRAIN


def load_probes(split, kind="random"):
    P = {}
    d_ = PROBE_DIR if kind == "random" else PROBE_DIR + "_" + kind
    for f in sorted(glob.glob(os.path.join(d_, "prb_*.npz"))):
        seed = int(os.path.basename(f)[4:8])
        if (seed < N_TRAIN) != (split == "train"):
            continue
        d = np.load(f)
        P[seed] = {k: d[k] for k in d.files}
    return P


def rel(a, b):
    return jnp.sum(jnp.abs(a[:N_EVAL] - b[:N_EVAL]) ** 2) / (jnp.sum(jnp.abs(b) ** 2) + 1e-20)


def adjoint(model, w, f):
    _, vjp = jax.vjp(lambda v: model(v, f), jnp.zeros_like(w))
    return jnp.conj(vjp(jnp.conj(w))[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nr")
    ap.add_argument("--order", type=int, default=8)
    ap.add_argument("--stages", type=int, default=1)
    ap.add_argument("--refine", type=int, default=1)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--n-res", type=int, default=8)
    ap.add_argument("--learn", type=int, default=1)
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--probes", default="random", choices=["random", "pod"])
    ap.add_argument("--w-traj", type=float, default=1.0)
    ap.add_argument("--w-fwd", type=float, default=0.0)
    ap.add_argument("--w-adj", type=float, default=0.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    args.step = 1

    V, F = load("train")
    from common import DATA_DIR
    all_seeds = sorted(int(os.path.basename(f)[4:8]) for f in glob.glob(os.path.join(DATA_DIR, "scn_*.npz")))
    train_seeds = [s for s in all_seeds if s < N_TRAIN]          # order of load("train")
    test_seeds = [s for s in all_seeds if s >= N_TRAIN]          # order of load("test")
    PR = load_probes("train", args.probes)
    PT = load_probes("test", args.probes)
    Vt, Ft = load("test")
    print(f"trajectories {V.shape[0]}, probe scenarios {len(PR)} train / {len(PT)} test", flush=True)

    key = jax.random.PRNGKey(args.seed)
    if args.model == "nr":
        model = NeuralRationalStep(DX_OUT, key, order=args.order, n_stages=args.stages, refine=args.refine,
                                   learn=bool(args.learn), width=args.width)
    else:
        args.model = "sym"
        model = make_model(args, key, feats=F[::10])
    print(f"{args.model}: {n_params(model)} parameters", flush=True)
    sched = optax.warmup_cosine_decay_schedule(0.0, args.lr, min(200, args.iters // 10), args.iters, args.lr * 1e-2)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))
    params, static = eqx.partition(model, eqx.is_inexact_array)
    opt_state = opt.init(params)

    def parts(params, tv, tf_, tt, pv, pf, pt, aw, af, at, with_adj=True):
        m = eqx.combine(params, static)
        lt = jnp.mean(jax.vmap(lambda v, f, t: rel(m(v, f), t))(tv, tf_, tt))
        lf = jnp.mean(jax.vmap(lambda v, f, t: rel(m(v, f), t))(pv, pf, pt))
        la = jnp.mean(jax.vmap(lambda w, f, t: rel(adjoint(m, w, f), t))(aw, af, at)) if with_adj else 0.0
        return lt, lf, la

    def loss_fn(params, *b):
        lt, lf, la = parts(params, *b, with_adj=bool(args.w_adj))
        return args.w_traj * lt + args.w_fwd * lf + args.w_adj * la

    @eqx.filter_jit
    def train_step(params, opt_state, *b):
        loss, g = jax.value_and_grad(loss_fn)(params, *b)
        upd, opt_state = opt.update(g, opt_state, params)
        return optax.apply_updates(params, upd), opt_state, loss

    eval_parts = eqx.filter_jit(lambda p, *b: parts(p, *b, with_adj=True))

    def draw(Vs, Fs, probes, rng, size):
        i = rng.integers(0, Vs.shape[0], size)
        n = rng.integers(0, Vs.shape[1] - 1, size)
        tv, tt = Vs[i, n], Vs[i, n + 1]
        tf_ = np.stack([Fs[a][feat_index(np.array(b), 1)] for a, b in zip(i, n)])
        keys = list(probes)
        pv, pf, pt, aw, af, at = [], [], [], [], [], []
        for _ in range(size):
            s = keys[rng.integers(len(keys))]
            d = probes[s]
            k = rng.integers(len(d["fwd_in"]))
            pv.append(d["fwd_in"][k]); pt.append(d["fwd_out"][k])
            pf.append((s, d["fwd_col"][k]))
            k = rng.integers(len(d["adj_in"]))
            aw.append(d["adj_in"][k]); at.append(d["adj_out"][k])
            af.append((s, d["adj_col"][k]))
        return tv, tf_, tt, pv, pf, pt, aw, af, at

    def feats_for(Fs, seed_list, pairs):
        return np.stack([Fs[seed_list.index(s)][feat_index(np.array(j), 1)] for s, j in pairs])


    def make_batch(Vs, Fs, probes, seed_list, rng, size):
        tv, tf_, tt, pv, pf, pt, aw, af, at = draw(Vs, Fs, probes, rng, size)
        c = lambda x: jnp.asarray(np.array(x), dtype=complex)  # noqa: E731
        return (c(tv), jnp.asarray(tf_), c(tt), c(pv), jnp.asarray(feats_for(Fs, seed_list, pf)), c(pt),
                c(aw), jnp.asarray(feats_for(Fs, seed_list, af)), c(at))

    rng = np.random.default_rng(args.seed)
    val = make_batch(Vt, Ft, PT, test_seeds, np.random.default_rng(7), 64)
    log, t0 = [], time.time()
    for it in range(args.iters):
        params, opt_state, loss = train_step(params, opt_state,
                                             *make_batch(V, F, PR, train_seeds, rng, args.batch))
        if it % 250 == 0 or it == args.iters - 1:
            lt, lf, la = (float(x) for x in eval_parts(params, *val))
            log.append(dict(it=it, loss=float(loss), val_traj=lt, val_fwd=lf, val_adj=la, time=time.time() - t0))
            print(f"it {it:5d}  train {float(loss):.3e}  val rms: traj {np.sqrt(lt):.2e}  fwd probes {np.sqrt(lf):.2e}"
                  f"  adj probes {np.sqrt(la):.2e}  {time.time() - t0:.0f} s", flush=True)
    model = eqx.combine(params, static)
    os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
    eqx.tree_serialise_leaves(os.path.join(HERE, "models", args.name + ".eqx"), model)
    with open(os.path.join(HERE, "models", args.name + ".json"), "w") as f:
        json.dump(dict(args=vars(args), log=log, n_params=n_params(model)), f, indent=1)


if __name__ == "__main__":
    main()
