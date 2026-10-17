"""Train a learned range step on the reference scenarios.

Run::

    python train.py --model lss --stages 2 --step 2 --iters 6000 --name lss_k2_s2
    python train.py --model fno --step 2 --iters 6000 --name fno_s2

``--step s`` is the range step in units of ``DX_OUT`` (25 m).  ``--rollout R``
trains on ``R`` consecutive steps (loss on every step) after
``--rollout-from`` iterations.
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

from common import DATA_DIR, DX_OUT, N_EVAL
from models import N_HALF, FNOStep, LearnedSplitStep, SymbolStep, n_params, symbol_basis

HERE = os.path.dirname(os.path.abspath(__file__))
N_COLS = 6
N_TRAIN = 200


def load(split):
    files = sorted(glob.glob(os.path.join(DATA_DIR, "scn_*.npz")))
    seeds = [int(os.path.basename(f)[4:8]) for f in files]
    sel = [f for f, s in zip(files, seeds) if (s < N_TRAIN) == (split == "train")]
    vs, fs = [], []
    for f in sel:
        d = np.load(f)
        vs.append(d["v"].astype(np.complex128))
        fs.append(d["feats"].astype(np.float64))
    return np.stack(vs), np.stack(fs)


def feat_index(n, s):
    """Indices of the N_COLS metric columns across the step from column n (5 m spacing)."""
    return 5 * n[..., None] + s * np.arange(N_COLS)


def make_model(args, key, feats=None):
    """``feats`` (training metric features) are needed to build the POD basis of
    a new symbol model; without them a zero basis of the right shape is made
    (skeleton for deserialization)."""
    dx = args.step * DX_OUT
    if args.model == "nr":
        from neural_rational import NeuralRationalStep
        return NeuralRationalStep(dx, key, order=args.order, n_stages=args.stages, refine=args.refine,
                                  learn=bool(args.learn), width=args.width)
    if args.model == "sym":
        ds = dx / args.stages
        basis = (symbol_basis(feats, ds, args.rank, cols_per_stage=5 * args.step // args.stages)
                 if feats is not None
                 else np.zeros((args.rank, N_HALF), dtype=complex))
        return SymbolStep(dx, args.stages, basis, key, learn=bool(args.learn), width=args.width, n_res=args.n_res)
    if args.model == "lss":
        return LearnedSplitStep(dx, args.stages, N_COLS, key, width=args.width)
    if args.model == "fno":
        return FNOStep(dx, N_COLS, key, width=args.width, modes=args.modes)
    raise ValueError(args.model)


def rel_err(pred, tgt):
    """Squared error below 200 m relative to the energy of the target in the whole
    window (columns whose field has mostly left the evaluation rows do not dominate)."""
    return jnp.sum(jnp.abs(pred[..., :N_EVAL] - tgt[..., :N_EVAL]) ** 2) / (jnp.sum(jnp.abs(tgt) ** 2) + 1e-20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lss")
    ap.add_argument("--stages", type=int, default=2)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--modes", type=int, default=200)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--learn", type=int, default=1)
    ap.add_argument("--n-res", type=int, default=0)
    ap.add_argument("--order", type=int, default=8)
    ap.add_argument("--refine", type=int, default=1)
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--rollout", type=int, default=1)
    ap.add_argument("--rollout-from", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    V, F = load("train")
    Vt, Ft = load("test")
    print(f"train scenarios {V.shape[0]}, test {Vt.shape[0]}", flush=True)
    s = args.step
    n_cols = V.shape[1]
    key = jax.random.PRNGKey(args.seed)
    model = make_model(args, key, feats=F[::10])
    print(f"{args.model}: {n_params(model)} parameters", flush=True)

    sched = optax.warmup_cosine_decay_schedule(0.0, args.lr, 200, args.iters, args.lr * 1e-2)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))
    params, static = eqx.partition(model, eqx.is_inexact_array)
    opt_state = opt.init(params)

    def loss_fn(params, v0, feats, targets):
        m = eqx.combine(params, static)

        def one(v0, feats, targets):
            def body(v, x):
                f, t = x
                v1 = m(v, f)
                return v1, rel_err(v1, t)
            _, errs = jax.lax.scan(body, v0, (feats, targets))
            return jnp.mean(errs)
        return jnp.mean(jax.vmap(one)(v0, feats, targets))

    def per_sample(params, v0, feats, targets):
        m = eqx.combine(params, static)
        return jax.vmap(lambda v, f, t: rel_err(m(v, f[0]), t[0]))(v0, feats, targets)

    @eqx.filter_jit
    def train_step(params, opt_state, v0, feats, targets):
        loss, g = jax.value_and_grad(loss_fn)(params, v0, feats, targets)
        upd, opt_state = opt.update(g, opt_state, params)
        return optax.apply_updates(params, upd), opt_state, loss

    eval_loss = eqx.filter_jit(loss_fn)
    eval_ps = eqx.filter_jit(per_sample)

    def batch(Vs, Fs, rng, R, size):
        i = rng.integers(0, Vs.shape[0], size)
        n = rng.integers(0, n_cols - R * s, size)
        steps = n[:, None] + s * np.arange(R + 1)                  # (b, R+1)
        v = Vs[i[:, None], steps]                                  # (b, R+1, N)
        fi = feat_index(steps[:, :-1], s)                          # (b, R, N_COLS)
        f = Fs[i[:, None, None], fi]                               # (b, R, N_COLS, N, 2)
        return jnp.asarray(v[:, 0]), jnp.asarray(f), jnp.asarray(v[:, 1:])

    rng = np.random.default_rng(args.seed)
    rng_val = np.random.default_rng(12345)
    val = batch(Vt, Ft, rng_val, 1, 256)
    log = []
    t0 = time.time()
    for it in range(args.iters):
        R = args.rollout if it >= args.rollout_from else 1
        params, opt_state, loss = train_step(params, opt_state, *batch(V, F, rng, R, args.batch))
        if it % 250 == 0 or it == args.iters - 1:
            ps = np.sqrt(np.asarray(eval_ps(params, *val)))
            vl, vmed = float(np.mean(ps ** 2)), float(np.median(ps))
            log.append(dict(it=it, loss=float(loss), val=vl, val_median=vmed, time=time.time() - t0))
            print(f"it {it:5d}  train {float(loss):.3e}  val(1 step) {vl:.3e}  median rel.L2 {vmed:.2e}  "
                  f"{time.time() - t0:.0f} s", flush=True)
    model = eqx.combine(params, static)
    os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
    eqx.tree_serialise_leaves(os.path.join(HERE, "models", args.name + ".eqx"), model)
    with open(os.path.join(HERE, "models", args.name + ".json"), "w") as f:
        json.dump(dict(args=vars(args), log=log, n_params=n_params(model)), f, indent=1)


if __name__ == "__main__":
    main()
