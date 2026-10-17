"""The reference propagator as a differentiable operator oracle.

For a scenario, ``Oracle`` builds the terrain-following rational propagator
(Dirichlet top on the tall domain: within one 25 m step nothing reflected at
420 m reaches the 256 m window) and exposes the one-step map
``P_j: v -> v'`` from the window column at range ``j * DX_OUT`` to the column
``DX_OUT`` further, on the 0.25 m grid of the learned models.  The coarse
column is interpolated to the fine solver grid spectrally (the fields are
band-limited); the result is sampled back.  ``P_j`` is linear, so ``jax.vjp``
gives the adjoint ``P_j^H w`` exactly.

``gen`` writes, per scenario, forward probes ``(v, P v)`` and adjoint probes
``(w, P^H w)`` with random band-limited inputs, for Sobolev / operator
matching::

    python oracle.py gen <first_seed> <last_seed_exclusive>
"""
import math
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from common import (DETA, DX_OUT, FREQ, H_COMP, K0, N_WIN, THETA_MAX_DEG, X_MAX, HelmholtzMeshParams2D,
                    TerrainFollowingHelmholtzPropagator, TFMedium, sample_scenario, terrain_map)

PROBE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"probes_d{int(round(DETA * 100)):03d}")


class Oracle:
    def __init__(self, sc, precision=1e-2):
        tm = terrain_map(sc)
        med = TFMedium(k2_above=lambda x, z: np.full(np.shape(z), K0 ** 2, dtype=complex))
        mp = HelmholtzMeshParams2D(x_size_m=X_MAX, z_size_m=H_COMP, dx_output_m=DX_OUT, dz_output_m=DETA)
        out = sys.stdout
        sys.stdout = open(os.devnull, "w")
        try:
            self.tf = TerrainFollowingHelmholtzPropagator.create(
                FREQ, tm, med, K0 * math.sin(math.radians(THETA_MAX_DEG)), (K0, K0), precision, mp,
                upper_bc='dirichlet')
        finally:
            sys.stdout.close()
            sys.stdout = out
        tf = self.tf
        self.nsub = tf.x_grid_scale
        self.zs = tf.z_grid_scale
        self.n_fine = N_WIN * self.zs
        n_ext = 2 * tf.z_n                                     # odd extension of the fine column
        self.n_ext = n_ext
        self.step = jax.jit(self._step)

    def _upsample(self, v):
        """Band-limited interpolation of the coarse window column to the fine grid (sine series)."""
        n = N_WIN
        ext = jnp.concatenate([v, jnp.zeros(1, dtype=v.dtype), -v[:0:-1]])          # 2n, odd
        spec = jnp.fft.fft(ext)
        nf = 2 * n * self.zs
        full = jnp.zeros(nf, dtype=complex)
        full = full.at[:n].set(spec[:n]).at[nf - n + 1:].set(spec[n + 1:])
        fine = jnp.fft.ifft(full)[:n * self.zs] * self.zs
        return jnp.zeros(self.tf.z_n, dtype=complex).at[:self.n_fine].set(fine)

    def _step(self, v, j):
        tf = self.tf
        xi0 = j * DX_OUT
        y = self._upsample(v * jnp.exp(-1j * (tf.beta - K0) * xi0))
        y = y.at[0].set(0.0)
        conv = jnp.zeros(len(tf.coefs), dtype=complex)

        def body(i, y):
            cp = tf._columns_at(xi0 + (i - 1) * tf.dx_m)
            cm = tf._columns_at(xi0 + (i - 0.5) * tf.dx_m)
            cn = tf._columns_at(xi0 + i * tf.dx_m)
            return tf._tf_step(y, conv, cp, cm, cn)[0]
        y = jax.lax.fori_loop(1, self.nsub + 1, body, y)
        return y[:self.n_fine:self.zs] * jnp.exp(1j * (tf.beta - K0) * (xi0 + DX_OUT))

    def adjoint(self, w, j):
        """P_j^H w (P is complex-linear: vjp gives P^T c)."""
        _, vjp = jax.vjp(lambda v: self.step(v, j), jnp.zeros(N_WIN, dtype=complex))
        return jnp.conj(vjp(jnp.conj(w))[0])


def random_probes(rng, n, kappa_max=K0 * math.sin(math.radians(40.0))):
    """Band-limited random columns: random complex sine coefficients for kappa <= kappa_max
    (default K0 sin 40 deg, the aperture of the physical fields; steeper probes measure the
    operator where no field lives and where the coarse rational scheme is not accurate)
    (half of them additionally localized by a Gaussian envelope), tapered to zero above 200 m,
    normalized to unit norm."""
    eta = np.arange(N_WIN) * DETA
    kap = np.pi * np.arange(N_WIN) / (N_WIN * DETA)
    out = np.zeros((n, N_WIN), dtype=complex)
    taper = np.clip((200.0 - eta) / 40.0, 0.0, 1.0) ** 2
    for p in range(n):
        c = (rng.normal(size=N_WIN) + 1j * rng.normal(size=N_WIN)) * (kap <= rng.uniform(0.3, 1.0) * kappa_max)
        u = np.fft.ifft(np.concatenate([c, [0.0], -c[:0:-1]]) * 1j)[:N_WIN]          # odd -> sine series
        if p % 2:
            u *= np.exp(-((eta - rng.uniform(0, 180)) / rng.uniform(5, 60)) ** 2)
        u *= taper
        u[0] = 0.0
        out[p] = u / np.linalg.norm(u)
    return out


def pod_probes(rng, n, basis, sv):
    """Random combinations of the POD modes of the reference fields, with coefficient
    standard deviations proportional to the singular values (the physical subspace)."""
    m = basis.shape[1]
    c = (rng.normal(size=(n, m)) + 1j * rng.normal(size=(n, m))) * sv[None, :m]
    u = c @ basis.T
    u[:, 0] = 0.0
    return u / np.linalg.norm(u, axis=1, keepdims=True)


def gen(s0, s1, n_cols=24, n_fwd=4, n_adj=2, kind="random"):
    out_dir = PROBE_DIR if kind == "random" else PROBE_DIR + "_pod"
    os.makedirs(out_dir, exist_ok=True)
    if kind == "pod":
        pb = np.load(os.path.join(os.path.dirname(PROBE_DIR), "pod_basis_d025.npz"))
        basis, sv = pb["basis"], pb["sv"]
        draw = lambda rng, n: pod_probes(rng, n, basis, sv)  # noqa: E731
    else:
        draw = random_probes
    for seed in range(s0, s1):
        path = os.path.join(out_dir, f"prb_{seed:04d}.npz")
        if os.path.exists(path):
            continue
        t = time.time()
        rng = np.random.default_rng(1000 + seed)
        orc = Oracle(sample_scenario(seed))
        cols = np.sort(rng.choice(int(X_MAX / DX_OUT), n_cols, replace=False))
        fv, fo, aw, ao, fj, aj = [], [], [], [], [], []
        for j in cols:
            v = draw(rng, n_fwd)
            for p in range(n_fwd):
                fv.append(v[p])
                fo.append(np.asarray(orc.step(jnp.asarray(v[p]), j)))
                fj.append(j)
            w = draw(rng, n_adj)
            for p in range(n_adj):
                aw.append(w[p])
                ao.append(np.asarray(orc.adjoint(jnp.asarray(w[p]), j)))
                aj.append(j)
        np.savez_compressed(path, fwd_in=np.array(fv, np.complex64), fwd_out=np.array(fo, np.complex64),
                            fwd_col=np.array(fj), adj_in=np.array(aw, np.complex64),
                            adj_out=np.array(ao, np.complex64), adj_col=np.array(aj))
        print(f"seed {seed}: {len(fv)} forward + {len(aw)} adjoint probes, {time.time() - t:.1f} s", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "gen":
        gen(int(sys.argv[2]), int(sys.argv[3]), kind=sys.argv[4] if len(sys.argv) > 4 else "random")
