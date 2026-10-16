"""Latent-space range propagators: encoder u_j -> s_j, propagator s_j -> s_{j+1}
conditioned on the step dx and the metrics h_xi, h_eta across the step, decoder
s -> u.

* ``LinearLatent`` (reduced-order / Koopman model): tied linear encoder and
  decoder ``s = Phi^H u``, ``u = Phi s`` (Phi: N x m complex, initialized with
  POD modes of the training columns) and
  ``s_{j+1} = expm(dx L) s_j``,  ``L = L0 + sum_r a_r(metrics, dx) B_r``.
  ``L0`` is the Galerkin projection of the free-space generator; ``a_r`` comes
  from a CNN of the metric columns (zero for identity metrics).  For fixed
  metrics the propagator is a semigroup in dx.
* ``NonlinearLatent``: CNN encoder / decoder to a real latent vector and a
  residual MLP propagator ``s' = s + f(s, dx, code(metrics))``.  The field is
  normalized by its norm, and the log-norm is propagated as an extra latent
  variable.

Run::

    python latent.py train --model linear --m 128 --name lat_lin_m128
    python latent.py train --model nonlinear --m 256 --name lat_nl_m256
    python latent.py eval lat_lin_m128 lat_nl_m256
"""
import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from common import DX_OUT, K0, N_EVAL, N_WIN
from models import ConvNet, KZ, N_PAD, n_params
from train import HERE, feat_index, load

DX_SET = (1, 2, 4)            # steps in units of DX_OUT (25, 50, 100 m)
N_COLS = 6


def pod_basis(V, m, n_snap=8000, seed=0):
    X = V[:, 1:].reshape(-1, V.shape[-1])
    X = X[np.random.default_rng(seed).choice(len(X), min(n_snap, len(X)), replace=False)]
    _, _, wh = np.linalg.svd(X, full_matrices=False)
    return wh[:m].T                                         # (N, m), orthonormal columns: u ~ Phi (Phi^H u)


def free_generator_galerkin(phi):
    """Phi^H G0 Phi, G0 = i (sqrt(K0^2 + d_eta^2) - K0) on the zero-padded odd extension."""
    kx = np.sqrt((K0 ** 2 - KZ ** 2).astype(complex))
    kx = np.where(kx.imag < 0, -kx, kx)
    sym = 1j * (kx - K0)
    cols = []
    for l in range(phi.shape[1]):
        w = np.zeros(N_PAD, dtype=complex)
        w[:N_WIN] = phi[:, l]
        ext = np.concatenate([w, [0.0], -w[:0:-1]])
        cols.append(np.fft.ifft(sym * np.fft.fft(ext))[:N_WIN])
    G = np.stack(cols, axis=1)
    return phi.conj().T @ G


class MetricEncoder(eqx.Module):
    """Metric columns (N_COLS, N, 2) and dx -> code (n_out,), zero for identity metrics.

    Strided convolutions down to 32 rows keep the vertical position of the
    metric deviations (1024 rows at 0.25 m -> 32 rows of 8 m); the flattened
    features and dx go through an MLP."""
    convs: list
    head: eqx.nn.MLP

    def __init__(self, n_out, key, width=32):
        ks = jax.random.split(key, 6)
        ch = [2 * N_COLS + 1, width, width, width, width, width]
        self.convs = [eqx.nn.Conv1d(ch[i], ch[i + 1], 5, stride=2, padding=2, key=ks[i]) for i in range(5)]
        self.head = eqx.nn.MLP(width * 32 + 1, n_out, 256, 2, activation=jax.nn.gelu, key=ks[5])

    def features(self, f):
        x = jnp.concatenate([f.transpose(0, 2, 1).reshape(-1, N_WIN),
                             (jnp.arange(N_WIN)[None] / N_WIN).astype(float)], axis=0)
        for c in self.convs:
            x = jax.nn.gelu(c(x))
        return x.reshape(-1)

    def __call__(self, f, dx):
        d = jnp.array([dx / 100.0])
        h = self.features(f) - self.features(jnp.zeros_like(f))
        return self.head(jnp.concatenate([h, d])) - self.head(jnp.concatenate([jnp.zeros_like(h), d]))


class LinearLatent(eqx.Module):
    phi: jnp.ndarray          # (N, m, 2)
    l0: jnp.ndarray           # (m, m, 2)
    b: jnp.ndarray            # (R, m, m, 2)
    menc: MetricEncoder
    m: int = eqx.field(static=True)
    freeze_basis: bool = eqx.field(static=True)
    rank: int = eqx.field(static=True)

    def __init__(self, phi, key, n_terms=16, freeze_basis=False, rank=0):
        """``rank > 0``: the metric encoder also outputs U, V (m x rank complex) and
        L gets the metric-dependent low-rank term U V^H (not restricted to the
        span of the learned matrices B_r)."""
        k1, k2 = jax.random.split(key)
        self.m = phi.shape[1]
        self.freeze_basis = freeze_basis
        self.rank = rank
        self.phi = jnp.stack([phi.real, phi.imag], axis=-1)
        l0 = free_generator_galerkin(phi)
        self.l0 = jnp.stack([l0.real, l0.imag], axis=-1)
        self.b = 1e-3 * jax.random.normal(k1, (n_terms, self.m, self.m, 2))
        self.menc = MetricEncoder(n_terms + 4 * self.m * rank, k2)

    def _c(self, x):
        return x[..., 0] + 1j * x[..., 1]

    def _phi(self):
        p = self._c(self.phi)
        return jax.lax.stop_gradient(p) if self.freeze_basis else p

    def encode(self, u):
        return jnp.conj(self._phi()).T @ u

    def decode(self, s):
        return self._phi() @ s

    def step_matrix(self, f, dx):
        code = self.menc(f, dx)
        nt = self.b.shape[0]
        a = code[:nt]
        L = self._c(self.l0) + jnp.tensordot(a, self._c(self.b), axes=1)
        if self.rank:
            uv = 1e-2 * code[nt:].reshape(2, 2, self.m, self.rank)
            U, V = uv[0, 0] + 1j * uv[0, 1], uv[1, 0] + 1j * uv[1, 1]
            L = L + U @ jnp.conj(V).T
        return jax.scipy.linalg.expm(dx * L)

    def step(self, s, f, dx):
        return self.step_matrix(f, dx) @ s


class NonlinearLatent(eqx.Module):
    enc: list
    enc_head: eqx.nn.Linear
    dec_head: eqx.nn.Linear
    dec: list
    prop: eqx.nn.MLP
    menc: MetricEncoder
    m: int = eqx.field(static=True)

    def __init__(self, m, key, width=48, code=32):
        ks = jax.random.split(key, 12)
        self.m = m
        ch = [2, width, width, width, width, width]                  # 1024 -> 32 rows
        self.enc = [eqx.nn.Conv1d(ch[i], ch[i + 1], 5, stride=2, padding=2, key=ks[i]) for i in range(5)]
        self.enc_head = eqx.nn.Linear(width * 32, m, key=ks[5])
        self.dec_head = eqx.nn.Linear(m, width * 32, key=ks[6])
        self.dec = [eqx.nn.ConvTranspose1d(ch[5 - i], ch[4 - i] if i < 4 else width, 4, stride=2, padding=1,
                                           key=ks[7 + i]) for i in range(5)]
        self.dec = self.dec + [eqx.nn.Conv1d(width, 2, 3, padding=1, key=ks[11])]
        self.prop = eqx.nn.MLP(m + 1 + code + 1, m + 1, 512, 3, activation=jax.nn.gelu, key=ks[0])
        self.prop = eqx.tree_at(lambda p: p.layers[-1].weight, self.prop, self.prop.layers[-1].weight * 1e-2)
        self.menc = MetricEncoder(code, ks[1])

    def encode(self, u):
        nrm = jnp.linalg.norm(u) + 1e-12
        x = jnp.stack([u.real, u.imag]) / nrm * jnp.sqrt(N_WIN)
        for c in self.enc:
            x = jax.nn.gelu(c(x))
        return jnp.concatenate([self.enc_head(x.reshape(-1)), jnp.log(nrm)[None]])

    def decode(self, s):
        x = self.dec_head(s[:-1]).reshape(-1, 32)
        for c in self.dec[:-1]:
            x = jax.nn.gelu(c(x))
        x = self.dec[-1](x)
        return (x[0] + 1j * x[1]).at[0].set(0.0) * jnp.exp(s[-1]) / jnp.sqrt(N_WIN)

    def step(self, s, f, dx):
        code = self.menc(f, dx)
        return s + self.prop(jnp.concatenate([s, jnp.array([dx / 100.0]), code]))


def rel(a, b, rows=N_WIN):
    return jnp.sum(jnp.abs(a[..., :rows] - b[..., :rows]) ** 2) / (jnp.sum(jnp.abs(b) ** 2) + 1e-20)


def make(args, V=None, key=None):
    key = jax.random.PRNGKey(args.seed) if key is None else key
    if args.model == "linear":
        phi = pod_basis(V, args.m) if V is not None else np.zeros((N_WIN, args.m), dtype=complex)
        return LinearLatent(phi, key, n_terms=args.terms, freeze_basis=bool(getattr(args, "freeze_basis", 0)),
                            rank=getattr(args, "rank", 0))
    return NonlinearLatent(args.m, key)


def batch(V, F, rng, size, rollout=1):
    """Random (u_j, metrics of R steps, targets u_{j+k}, ..., dx) with one dx per sample."""
    k = rng.choice(DX_SET, size)
    n_cols = V.shape[1]
    i = rng.integers(0, V.shape[0], size)
    n = np.array([rng.integers(0, n_cols - rollout * kk) for kk in k])
    steps = n[:, None] + k[:, None] * np.arange(rollout + 1)
    v = V[i[:, None], steps]
    f = np.stack([F[i[b]][feat_index(steps[b, :-1], k[b])] for b in range(size)])
    return jnp.asarray(v[:, 0]), jnp.asarray(f), jnp.asarray(v[:, 1:]), jnp.asarray(k * DX_OUT, dtype=float)


def train(args):
    V, F = load("train")
    Vt, Ft = load("test")
    model = make(args, V)
    print(f"{args.model}: m={args.m}, {n_params(model)} parameters", flush=True)
    warm = min(200, max(args.iters // 10, 1))
    sched = optax.warmup_cosine_decay_schedule(0.0, args.lr, warm, max(args.iters, warm + 1), args.lr * 1e-2)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))
    params, static = eqx.partition(model, eqx.is_inexact_array)
    opt_state = opt.init(params)

    def losses(params, u0, f, tgt, dx):
        mdl = eqx.combine(params, static)

        def one(u0, f, tgt, dx):
            s = mdl.encode(u0)
            l_rec = rel(mdl.decode(s), u0)
            l_pred, l_lat = 0.0, 0.0
            for r in range(tgt.shape[0]):
                s = mdl.step(s, f[r], dx)
                l_pred = l_pred + rel(mdl.decode(s), tgt[r])
                s_t = mdl.encode(tgt[r])
                l_lat = l_lat + jnp.sum(jnp.abs(s - s_t) ** 2) / (jnp.sum(jnp.abs(s_t) ** 2) + 1e-20)
            return l_rec, l_pred / tgt.shape[0], l_lat / tgt.shape[0]
        lr_, lp, ll = jax.vmap(one)(u0, f, tgt, dx)
        return jnp.mean(lr_), jnp.mean(lp), jnp.mean(ll)

    def loss_fn(params, *b):
        lr_, lp, ll = losses(params, *b)
        return lr_ + lp + 0.1 * ll

    @eqx.filter_jit
    def train_step(params, opt_state, *b):
        loss, g = jax.value_and_grad(loss_fn)(params, *b)
        upd, opt_state = opt.update(g, opt_state, params)
        return optax.apply_updates(params, upd), opt_state, loss

    eval_l = eqx.filter_jit(losses)
    rng = np.random.default_rng(args.seed)
    val = batch(Vt, Ft, np.random.default_rng(1), 128)
    log, t0 = [], time.time()
    for it in range(args.iters):
        R = args.rollout if it >= args.rollout_from else 1
        params, opt_state, loss = train_step(params, opt_state, *batch(V, F, rng, args.batch, R))
        if it % 250 == 0 or it == args.iters - 1:
            lr_, lp, ll = (float(x) for x in eval_l(params, *val))
            log.append(dict(it=it, loss=float(loss), val_rec=lr_, val_pred=lp, val_lat=ll, time=time.time() - t0))
            print(f"it {it:5d}  train {float(loss):.3e}  val: rec {np.sqrt(lr_):.2e}  pred {np.sqrt(lp):.2e}  "
                  f"latent {np.sqrt(ll):.2e}  {time.time() - t0:.0f} s", flush=True)
    model = eqx.combine(params, static)
    os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
    eqx.tree_serialise_leaves(os.path.join(HERE, "models", args.name + ".eqx"), model)
    with open(os.path.join(HERE, "models", args.name + ".json"), "w") as f:
        json.dump(dict(args=vars(args), log=log, n_params=n_params(model)), f, indent=1)


def evaluate(names):
    Vt, Ft = load("test")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    for name in names:
        with open(os.path.join(HERE, "models", name + ".json")) as f:
            args = SimpleNamespace(**json.load(f)["args"])
        model = eqx.tree_deserialise_leaves(os.path.join(HERE, "models", name + ".eqx"), make(args))
        out = dict(name=name, by_dx={})
        for k in DX_SET:
            dx = float(k * DX_OUT)
            if isinstance(model, LinearLatent):
                mats = eqx.filter_jit(jax.vmap(lambda f: model.step_matrix(f, dx)))

                @eqx.filter_jit
                def march(u0, A):
                    s0 = model.encode(u0)
                    s = jax.lax.scan(lambda s, a: (a @ s, a @ s), s0, A)[1]
                    return jax.vmap(model.decode)(s)
            else:
                mats = None

                @eqx.filter_jit
                def march(u0, f):
                    s0 = model.encode(u0)
                    s = jax.lax.scan(lambda s, ff: (model.step(s, ff, dx),) * 2, s0, f)[1]
                    return jax.vmap(model.decode)(s)
            rows = []
            for i in range(Vt.shape[0]):
                n = np.arange(0, Vt.shape[1] - k, k)
                f = jnp.asarray(Ft[i][feat_index(n, k)])
                u0 = jnp.asarray(Vt[i, 0])
                t_prep = 0.0
                if mats is not None:
                    jax.block_until_ready(mats(f))
                    t = time.perf_counter()
                    f = jax.block_until_ready(mats(f))
                    t_prep = time.perf_counter() - t
                jax.block_until_ready(march(u0, f))
                t = time.perf_counter()
                pred = np.asarray(jax.block_until_ready(march(u0, f)))
                t_march = time.perf_counter() - t
                ref = Vt[i, k::k][:len(n)]
                d = np.abs(pred - ref)[:, :N_EVAL]
                rows.append(dict(max_rel=float(d.max() / np.abs(ref[:, :N_EVAL]).max()),
                                 l2_rel=float(np.linalg.norm(d) / np.linalg.norm(ref[:, :N_EVAL])),
                                 l2_last=float(np.linalg.norm(d[-1]) / np.linalg.norm(ref[-1, :N_EVAL])),
                                 t_prep=t_prep, t_march=t_march))
            med = {kk: float(np.median([r[kk] for r in rows])) for kk in rows[0]}
            out["by_dx"][str(dx)] = dict(median=med, rows=rows)
            print(name, f"dx={dx:g} m", json.dumps(med), flush=True)
        with open(os.path.join(HERE, "results", f"eval_{name}.json"), "w") as f:
            json.dump(out, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval"])
    ap.add_argument("names", nargs="*")
    ap.add_argument("--model", default="linear")
    ap.add_argument("--m", type=int, default=128)
    ap.add_argument("--terms", type=int, default=16)
    ap.add_argument("--freeze-basis", type=int, default=0)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--rollout", type=int, default=1)
    ap.add_argument("--rollout-from", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    if args.cmd == "train":
        train(args)
    else:
        evaluate(args.names)


if __name__ == "__main__":
    main()
