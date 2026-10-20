"""2D field maps of the learned steps on test scenarios.

For each chosen test scenario, marches the reference start column with every
model (25 m steps) and plots 20 log10 |u| in physical coordinates (x, z) below
200 m above the ground, with the error 20 log10 |u - u_ref| / max |u_ref|.

Run::

    python plot_fields.py [test_index ...]      # default: the median and worst cases
                                                # of the untrained coarse scheme

Writes ``figs/fields_<seed>.png``.
"""
import json
import os
import sys

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import DETA, DX_OUT, N_EVAL, sample_scenario, terrain_map
from evaluate import load_model, rollout_fn
from latent import make as make_latent
from neural_rational import NeuralRationalStep
from train import HERE, feat_index, load

import equinox as eqx
from types import SimpleNamespace

MODELS = [("coarse scheme (untrained)", None), ("NR, trajectories", "nr_traj"),
          ("NR, + POD probes", "nr_podfwd"), ("NR, + random probes", "nr_fwd"),
          ("symbol step + residual", "symres_s1"), ("latent linear, m=128", "lat_lin_m128_r128")]


def march_step_model(model, v0, F):
    n = np.arange(0, 120)
    feats = jnp.asarray(F[feat_index(n, 1)])
    screens, march = rollout_fn(model)
    if screens is not None:
        return np.asarray(march(jnp.asarray(v0), screens(feats)))
    return np.asarray(march(jnp.asarray(v0), feats))


def march_latent(name, v0, F):
    with open(os.path.join(HERE, "models", name + ".json")) as f:
        args = SimpleNamespace(**json.load(f)["args"])
    model = eqx.tree_deserialise_leaves(os.path.join(HERE, "models", name + ".eqx"), make_latent(args))
    feats = jnp.asarray(F[feat_index(np.arange(0, 120), 1)])
    A = eqx.filter_jit(jax.vmap(lambda f: model.step_matrix(f, DX_OUT)))(feats)
    s = model.encode(jnp.asarray(v0))
    out = []
    for a in A:
        s = a @ s
        out.append(model.decode(s))
    return np.asarray(jnp.stack(out))


def main():
    V, F = load("test")
    with open(os.path.join(HERE, "results", "coarse_rational.json")) as f:
        rows = json.load(f)[0]["rows"]
    err = np.array([r["l2_rel"] for r in rows])
    idx = [int(a) for a in sys.argv[1:]] or [int(np.argsort(err)[len(err) // 2]), int(np.argmax(err))]
    seeds = sorted(int(f[4:8]) for f in os.listdir(os.path.join(HERE, "data_d025")) if int(f[4:8]) >= 200)
    os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
    coarse = NeuralRationalStep(DX_OUT, jax.random.PRNGKey(0), order=8, refine=1, learn=False)

    for i in idx:
        seed = seeds[i]
        sc = sample_scenario(seed)
        ref = V[i, 1:121]
        preds = {}
        for label, name in MODELS:
            if name is None:
                preds[label] = march_step_model(coarse, V[i, 0], F[i])
            elif name.startswith("lat_"):
                preds[label] = march_latent(name, V[i, 0], F[i])
            else:
                preds[label] = march_step_model(load_model(name)[0], V[i, 0], F[i])
            p = preds[label][:, :N_EVAL]
            print(f"seed {seed} {label}: rel L2 {np.linalg.norm(p - ref[:, :N_EVAL]) / np.linalg.norm(ref[:, :N_EVAL]):.3f}",
                  flush=True)

        xi = DX_OUT * np.arange(1, 121)
        eta = DETA * np.arange(N_EVAL)
        mt = terrain_map(sc).metrics(xi, eta)
        X, Z = mt.x, mt.z
        xg = np.linspace(0, 3000, 1000)
        ground = sc.terrain(xg)
        amax = np.abs(ref[:, :N_EVAL]).max()
        db = lambda u: 20 * np.log10(np.abs(u) / amax + 1e-12)  # noqa: E731

        n = len(MODELS) + 1
        fig, ax = plt.subplots(n, 2, figsize=(13, 2.1 * n), sharex=True, sharey=True, constrained_layout=True)
        panels = [("reference", ref)] + list(preds.items())
        for r, (label, u) in enumerate(panels):
            u = u[:, :N_EVAL]
            im0 = ax[r, 0].pcolormesh(X, Z, db(u), vmin=-40, vmax=0, cmap="jet", shading="gouraud",
                                      rasterized=True)
            e = np.linalg.norm(u - ref[:, :N_EVAL]) / np.linalg.norm(ref[:, :N_EVAL])
            ax[r, 0].set_title(f"{label}" + ("" if r == 0 else f"   (rel. L2 error {e:.3f})"), fontsize=9)
            if r == 0:
                ax[r, 1].axis("off")
            else:
                im1 = ax[r, 1].pcolormesh(X, Z, db(u - ref[:, :N_EVAL]), vmin=-60, vmax=0, cmap="magma",
                                          shading="gouraud", rasterized=True)
                ax[r, 1].set_title(f"error |u - u_ref| / max|u_ref|, {label}", fontsize=9)
            for a in (ax[r] if r else ax[r, :1]):
                a.fill_between(xg, ground.min() - 20, ground, color="0.35")
        for a in ax[:, 0]:
            a.set_ylabel("z, m")
        for a in ax[-1]:
            a.set_xlabel("x, m")
        ax[0, 0].set_ylim(ground.min() - 5, Z.max())
        ax[0, 0].set_xlim(0, 3000)
        fig.colorbar(im0, ax=ax[:, 0], label="20 log10 |u| / max|u_ref|, dB", shrink=0.4)
        fig.colorbar(im1, ax=ax[1:, 1], label="error, dB", shrink=0.4)
        fig.suptitle(f"test scenario {seed}: 300 MHz, 25 m range steps, 0.25 m grid")
        path = os.path.join(HERE, "figs", f"fields_{seed}.png")
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
