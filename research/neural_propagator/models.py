"""Learned range steps ``v_n -> v_{n+s}`` on the window ``0 <= eta < WINDOW_M``.

All models act on the envelope ``v`` (relative to ``K0``) on ``N_WIN`` rows with
``v_0 = 0`` (Dirichlet ground) and are conditioned on the metric features
``f`` of shape ``(S, N_WIN, 2)`` sampled at ``S`` columns across the step.

* :class:`FreeStep` -- exact free-space propagator (no terrain), the baseline
  that ignores the coordinates.
* :class:`LearnedSplitStep` -- linear in ``v``: ``K`` stages
  ``v <- F^-1[P_k F[D_k v]]`` with phase screens ``D_k = exp(a_k + i b_k)``
  produced by a convolutional network from ``f`` (a hypernetwork), and
  spectral multipliers ``P_k`` initialized to the exact free-space phase of a
  sub-step and corrected by learned, terrain-independent factors.  The screens
  are ``net(f) - net(0)``, so over flat ground and in the identity region the
  model is exactly the free-space step.
* :class:`FNOStep` -- nonlinear Fourier neural operator (Li et al. 2021) on
  ``[Re v, Im v, Re Fv, Im Fv, f]`` that predicts a correction to the
  free-space step ``Fv``; inputs are scaled by ``max|v|`` and the output
  rescaled, so the model is positively homogeneous of degree one.

Spectral operations use the odd extension of the zero-padded column (sine
series), which keeps ``v_0 = 0``; the padding absorbs what leaves the window
during one step.
"""
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from common import ABSORB_PER_25M, DETA, K0, N_EVAL, N_WIN

N_PAD = 2 * N_WIN                       # padded column (window + absorbing pad)
N_EXT = 2 * N_PAD                       # odd extension
KZ = 2 * np.pi * np.fft.fftfreq(N_EXT, d=DETA)
N_HALF = N_PAD + 1                      # independent |kz| values


def absorber(dx):
    """Damping screen exp(-A t^2) over the rows above N_EVAL (t = 0..1), A per 25 m of range."""
    t = np.clip((np.arange(N_WIN) - N_EVAL) / (N_WIN - N_EVAL), 0.0, 1.0)
    return jnp.asarray(np.exp(-ABSORB_PER_25M * dx / 25.0 * t ** 2))


def free_phase(dx):
    kx = np.sqrt((K0 ** 2 - KZ ** 2).astype(complex))
    kx = np.where(kx.imag < 0, -kx, kx)
    return np.exp(1j * dx * (kx - K0))


def half_to_full(p_half):
    """Even extension of a multiplier given on |kz| index 0..N_PAD."""
    return jnp.concatenate([p_half, p_half[-2:0:-1]])


def odd_fft_apply(v, mult):
    """Apply an even spectral multiplier to the column ``v`` (length N_WIN or N_PAD)."""
    w = jnp.zeros(N_PAD, dtype=v.dtype).at[:v.shape[0]].set(v)
    ext = jnp.concatenate([w, jnp.zeros(1, dtype=v.dtype), -w[:0:-1]])[:N_EXT]
    out = jnp.fft.ifft(mult * jnp.fft.fft(ext))
    return out[:N_PAD]


class FreeStep(eqx.Module):
    phase: jnp.ndarray
    dx: float = eqx.field(static=True)

    def __init__(self, dx):
        self.dx = float(dx)
        self.phase = jnp.asarray(free_phase(dx))

    def __call__(self, v, f):
        return odd_fft_apply(v, jax.lax.stop_gradient(self.phase))[:N_WIN] * absorber(self.dx)


class ConvNet(eqx.Module):
    """Dilated residual 1-D CNN over eta."""
    lift: eqx.nn.Conv1d
    blocks: list
    proj: eqx.nn.Conv1d

    def __init__(self, c_in, c_out, width, dilations, kernel, key):
        keys = jax.random.split(key, len(dilations) + 2)
        self.lift = eqx.nn.Conv1d(c_in, width, 1, key=keys[0])
        self.blocks = [eqx.nn.Conv1d(width, width, kernel, dilation=d, padding=d * (kernel // 2), key=k)
                       for d, k in zip(dilations, keys[1:-1])]
        self.proj = eqx.nn.Conv1d(width, c_out, 1, key=keys[-1])
        self.proj = eqx.tree_at(lambda m: m.weight, self.proj, self.proj.weight * 1e-2)

    def __call__(self, x):
        h = self.lift(x)
        for b in self.blocks:
            h = h + jax.nn.gelu(b(h))
        return self.proj(h)


class LearnedSplitStep(eqx.Module):
    net: ConvNet
    p_corr: jnp.ndarray          # (K, N_HALF, 2): log-amplitude and phase corrections
    p0: jnp.ndarray              # exact free-space phase of a sub-step (not trained)
    n_stages: int = eqx.field(static=True)

    dx: float = eqx.field(static=True)
    use_net: bool = eqx.field(static=True)

    stride: int = eqx.field(static=True)

    def __init__(self, dx, n_stages, n_cols, key, width=64, dilations=(1, 2, 4, 8, 16, 1), kernel=7,
                 use_net=True, stride=4):
        self.n_stages = n_stages
        self.dx = float(dx)
        self.use_net = use_net
        self.stride = stride
        self.net = ConvNet(2 * n_cols + 1, 2 * (n_stages + 1), width, dilations, kernel, key)
        self.p_corr = jnp.zeros((n_stages, N_HALF, 2))
        self.p0 = jnp.asarray(free_phase(dx / n_stages))

    def screens(self, f):
        """(K + 1, N_PAD) complex screens from features f (S, N_WIN, 2)."""
        k_st = self.n_stages
        if self.use_net:
            # the network runs on a grid coarser by `stride` (the metrics are smooth in eta);
            # its output is interpolated back linearly
            nc = N_WIN // self.stride
            eta_c = (jnp.arange(nc) + 0.5) * self.stride * DETA
            fc = f.reshape(f.shape[0], nc, self.stride, 2).mean(axis=2)
            x = jnp.concatenate([fc.transpose(0, 2, 1).reshape(-1, nc), eta_c[None] / 256.0], axis=0)
            x0 = jnp.concatenate([jnp.zeros_like(x[:-1]), eta_c[None] / 256.0], axis=0)
            sc = self.net(x) - self.net(x0)
            eta = jnp.arange(N_WIN) * DETA
            s = jax.vmap(lambda c: jnp.interp(eta, eta_c, c))(sc)
            a, b = s[0::2], s[1::2]
        else:
            a = b = jnp.zeros((k_st + 1, N_WIN))
        # physical prior: phase K0 (h_xi - 1) dx / K of each sub-step, h_xi at its midpoint
        n_c = f.shape[0]
        hxi = jnp.exp(f[:, :, 0])
        for k in range(k_st):
            p = (n_c - 1) * (k + 0.5) / k_st
            j0 = min(int(math.floor(p)), n_c - 2)
            w = p - j0
            h = (1 - w) * hxi[j0] + w * hxi[j0 + 1]
            b = b.at[k].add(K0 * (h - 1.0) * self.dx / k_st)
        scr = jnp.exp(a + 1j * b)
        return jnp.concatenate([scr, jnp.ones((scr.shape[0], N_PAD - N_WIN))], axis=1)

    def apply(self, v, scr):
        w = jnp.zeros(N_PAD, dtype=complex).at[:N_WIN].set(v)
        for k in range(self.n_stages):
            corr = half_to_full(self.p_corr[k, :, 0] + 1j * self.p_corr[k, :, 1])
            w = odd_fft_apply(scr[k] * w, jax.lax.stop_gradient(self.p0) * jnp.exp(corr))
        return (scr[-1] * w)[:N_WIN] * absorber(self.dx)

    def __call__(self, v, f):
        return self.apply(v, self.screens(f))


class SpectralConv(eqx.Module):
    w: jnp.ndarray     # (modes, c_in, c_out) complex as real pair
    modes: int = eqx.field(static=True)

    def __init__(self, c_in, c_out, modes, key):
        self.modes = modes
        scale = 1.0 / (c_in * c_out)
        self.w = scale * jax.random.normal(key, (modes, c_in, c_out, 2))

    def __call__(self, x):            # x: (c, N)
        n = x.shape[-1]
        xf = jnp.fft.rfft(x, axis=-1)[:, :self.modes]                 # (c, m)
        w = self.w[..., 0] + 1j * self.w[..., 1]
        yf = jnp.einsum('cm,mcd->dm', xf, w)
        out = jnp.zeros((w.shape[2], n // 2 + 1), dtype=complex).at[:, :self.modes].set(yf)
        return jnp.fft.irfft(out, n=n, axis=-1)


class FNOStep(eqx.Module):
    lift: eqx.nn.Conv1d
    spec: list
    local: list
    proj1: eqx.nn.Conv1d
    proj2: eqx.nn.Conv1d
    free: FreeStep

    def __init__(self, dx, n_cols, key, width=48, modes=200, depth=4):
        keys = jax.random.split(key, 2 * depth + 3)
        c_in = 4 + 2 * n_cols + 1
        self.lift = eqx.nn.Conv1d(c_in, width, 1, key=keys[0])
        self.spec = [SpectralConv(width, width, modes, keys[1 + i]) for i in range(depth)]
        self.local = [eqx.nn.Conv1d(width, width, 1, key=keys[1 + depth + i]) for i in range(depth)]
        self.proj1 = eqx.nn.Conv1d(width, 128, 1, key=keys[-2])
        self.proj2 = eqx.nn.Conv1d(128, 2, 1, key=keys[-1])
        self.proj2 = eqx.tree_at(lambda m: m.weight, self.proj2, self.proj2.weight * 1e-2)
        self.free = FreeStep(dx)

    def __call__(self, v, f):
        scale = jnp.max(jnp.abs(v)) + 1e-12
        fv = self.free(v, f)
        vs, fs = v / scale, fv / scale
        eta = jnp.arange(N_WIN) * DETA / 256.0
        x = jnp.concatenate([jnp.stack([vs.real, vs.imag, fs.real, fs.imag]),
                             f.transpose(0, 2, 1).reshape(-1, N_WIN), eta[None]], axis=0)
        h = self.lift(x)
        for i, (s, l) in enumerate(zip(self.spec, self.local)):
            h = s(h) + l(h)
            if i < len(self.spec) - 1:
                h = jax.nn.gelu(h)
        out = self.proj2(jax.nn.gelu(self.proj1(h)))
        corr = (out[0] + 1j * out[1]).at[0].set(0.0)
        return (fs + corr * absorber(self.free.dx)) * scale


def n_params(model):
    return sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array)))


def symbol_targets(a, c, ds):
    """Frozen-coefficient symbol exp(i ds a (sqrt(K0^2 - c^2 kappa^2) - K0)) on |kappa| = 0..N_PAD."""
    kap = jnp.abs(jnp.asarray(KZ[:N_HALF]))
    q = jnp.sqrt((K0 ** 2 - (c[..., None] * kap) ** 2).astype(complex))
    q = jnp.where(q.imag < 0, -q, q)
    return jnp.exp(1j * ds * a[..., None] * (q - K0))


def kappa_xi(lf):
    """Local longitudinal wavenumber h_xi sqrt(K0^2 - kappa^2 / h_eta^2) from log-metrics (..., 2)."""
    kap = jnp.abs(jnp.asarray(KZ[:N_HALF]))
    c = jnp.exp(-lf[..., 1])
    q = jnp.sqrt((K0 ** 2 - (c[..., None] * kap) ** 2).astype(complex))
    q = jnp.where(q.imag < 0, -q, q)
    return jnp.exp(lf[..., 0])[..., None] * q


def flux_symbol(fa, fm, fb, ds):
    """Frozen-coefficient symbol of the flux-conserving step
    B_b^(-1/4) exp(i ds sqrt(B_m)) B_a^(1/4) (envelope relative to K0):
    (q_a / q_b)^(1/2) exp(i ds (q_m - K0)), q = kappa_xi; the bulk phase
    K0 (h_xi,m - 1) ds is taken out (applied as a screen)."""
    qa, qm, qb = kappa_xi(fa), kappa_xi(fm), kappa_xi(fb)
    eps = 1e-3 * K0
    ratio = jnp.sqrt((qa + eps) / (qb + eps))
    bulk = K0 * (jnp.exp(fm[..., 0]) - 1.0)
    return ratio * jnp.exp(1j * ds * (qm - K0 - bulk[..., None]))


def symbol_basis(feats, ds, rank, n_samples=6000, seed=0, cols_per_stage=None):
    """POD basis (rank, N_HALF) of the step symbols over metric samples.

    ``feats`` (n_scen, n_cols, N, 2).  Without ``cols_per_stage`` the plain
    symbol of a single column is sampled; with it, the flux symbol of the
    triple of columns (j, j + n/2, j + n)."""
    rng = np.random.default_rng(seed)
    ns, nc, nr, _ = feats.shape
    i, r = rng.integers(0, ns, n_samples), rng.integers(0, nr, n_samples)
    if cols_per_stage is None:
        j = rng.integers(0, nc, n_samples)
        f = feats[i, j, r]
        T = np.asarray(flux_symbol(f, f, f, ds))
    else:
        n = cols_per_stage
        j = rng.integers(0, nc - n, n_samples)
        T = np.asarray(flux_symbol(feats[i, j, r], 0.5 * (feats[i, j, r] + feats[i, j + n, r]), feats[i, j + n, r], ds))
    _, s, vh = np.linalg.svd(T, full_matrices=False)
    return vh[:rank]


class SymbolStep(eqx.Module):
    """Pseudo-differential step: in each of ``n_stages`` sub-steps the field is
    multiplied by the local frozen-coefficient symbol
    ``exp(i ds h_xi (sqrt(K0^2 - kappa^2 / h_eta^2) - K0))`` of its output row,
    expanded in a fixed POD basis ``P_l(kappa)``: ``sum_l w_l(eta) F^-1[P_l F v]``,
    with the bulk phase ``K0 (h_xi - 1) ds`` and the Liouville factors
    ``r^(+-1/2)`` as screens.

    ``learn=True``: a CNN of the metric features predicts, per stage and row,
    bounded corrections of the *effective* metrics (``log h_xi``, ``log h_eta``,
    at most 10 %) and a log-amplitude / phase screen (at most 0.05 / 0.5 rad).
    The step therefore stays a frozen-coefficient one-way propagator; it is
    exactly the free-space step where the metrics are the identity."""
    basis: jnp.ndarray            # (L, N_HALF) complex, fixed
    net: ConvNet
    dx: float = eqx.field(static=True)
    n_stages: int = eqx.field(static=True)
    learn: bool = eqx.field(static=True)
    stride: int = eqx.field(static=True)
    quant: str = eqx.field(static=True)
    flux: bool = eqx.field(static=True)
    q_res: jnp.ndarray            # (K, M, N_EXT, 2) learned multipliers of the residual operator
    res_net: ConvNet
    n_res: int = eqx.field(static=True)

    def __init__(self, dx, n_stages, basis, key, n_cols=6, learn=False, width=48, stride=4, quant='left',
                 flux=True, n_res=0):
        """``n_res = M > 0`` adds the learned low-rank linear residual
        ``sum_m g_m(eta) F^-1[Q_m(kappa) F[h_m(eta) u]]`` to every stage, with
        screens ``g_m, h_m`` from a CNN of the metrics (zero where the metrics
        are the identity) and free multipliers ``Q_m`` over signed ``kappa``
        (odd symbols such as first derivatives are representable)."""
        """``quant``: 'left' evaluates the symbol at the output row (L inverse
        FFTs per stage), 'sym' averages it with the input-row evaluation (L
        forward and L inverse FFTs), which removes the first-order error of the
        eta dependence."""
        self.dx, self.n_stages, self.learn, self.stride = float(dx), n_stages, learn, stride
        self.quant = quant
        self.flux = flux
        self.basis = jnp.asarray(basis)
        k1, k2, k3 = jax.random.split(key, 3)
        self.net = ConvNet(2 * n_cols + 1, 4 * n_stages, width, (1, 2, 4, 8, 16, 1), 7, k1)
        self.n_res = n_res
        m = max(n_res, 1)
        self.res_net = ConvNet(2 * n_cols + 1, 4 * m * n_stages, width, (1, 2, 4, 8, 16, 1), 7, k2)
        self.q_res = 0.1 * jax.random.normal(k3, (n_stages, m, N_EXT, 2))

    def res_screens(self, f):
        """(K, M, 2, N_WIN) complex screens g, h of the residual operator."""
        rows = jnp.arange(0, N_WIN, self.stride)
        rows_f = rows.astype(float)
        eta_r = jnp.arange(N_WIN, dtype=float)
        x = jnp.concatenate([f[:, rows].transpose(0, 2, 1).reshape(-1, rows.size),
                             rows_f[None] * DETA / 256.0], axis=0)
        x0 = jnp.concatenate([jnp.zeros_like(x[:-1]), x[-1:]], axis=0)
        d = (self.res_net(x) - self.res_net(x0)).reshape(self.n_stages, self.n_res, 2, 2, rows.size)
        d = jax.vmap(lambda c: jnp.interp(eta_r, rows_f, c))(d.reshape(-1, rows.size)).reshape(d.shape[:-1] + (N_WIN,))
        return d[..., 0, :] + 1j * d[..., 1, :]

    def _stage_mean(self, f, k):
        """Range average of the log-metric columns over stage k: Newton-Cotes rule
        on the columns of the stage (6-point rule for one stage over 6 columns)."""
        n_c, K = f.shape[0], self.n_stages
        per = (n_c - 1) // K
        cols = f[k * per:(k + 1) * per + 1]
        nc_w = {2: [1, 1], 3: [1, 4, 1], 4: [1, 3, 3, 1], 5: [7, 32, 12, 32, 7], 6: [19, 75, 50, 50, 75, 19]}
        w = jnp.asarray(nc_w[cols.shape[0]], dtype=float)
        return jnp.tensordot(w / w.sum(), cols, axes=1)

    def _interp_cols(self, f, p):
        n_c = f.shape[0]
        p = p * (n_c - 1)
        j0 = min(int(math.floor(p)), n_c - 2)
        w = p - j0
        return (1 - w) * f[j0] + w * f[j0 + 1]

    def weights(self, f):
        """(K, N_PAD, L) weights and (K, 2, N_PAD) in/out screens."""
        K = self.n_stages
        ds = self.dx / K
        P = jax.lax.stop_gradient(self.basis)
        pad = N_PAD - N_WIN
        # the weights are projected on every `stride`-th row and interpolated
        # linearly (the metrics are smooth in eta); the pad is free space
        rows = jnp.arange(0, N_WIN, self.stride)
        rows_f = rows.astype(float)
        eta_r = jnp.arange(N_WIN, dtype=float)

        def up(col):
            return jnp.interp(eta_r, rows_f, col)
        if self.learn:
            x = jnp.concatenate([f[:, rows].transpose(0, 2, 1).reshape(-1, rows.size),
                                 rows_f[None] * DETA / 256.0], axis=0)
            x0 = jnp.concatenate([jnp.zeros_like(x[:-1]), x[-1:]], axis=0)
            d = (self.net(x) - self.net(x0)).reshape(K, 4, rows.size)
            d_hx, d_he = 0.1 * jnp.tanh(d[:, 0]), 0.1 * jnp.tanh(d[:, 1])
            d_amp, d_ph = 0.05 * jnp.tanh(d[:, 2]), 0.5 * jnp.tanh(d[:, 3])
        z = jnp.zeros((1, 2))
        w_free = flux_symbol(z, z, z, ds) @ jnp.conj(P).T                          # (1, L)
        ws, scr = [], []
        for k in range(K):
            fa, fb = (self._interp_cols(f, p) for p in (k / K, (k + 1) / K))
            if (f.shape[0] - 1) % K == 0:
                fm_ = self._stage_mean(f, k)
            else:
                fm_ = self._interp_cols(f, (k + 0.5) / K)
            gm = fm_[rows]
            if self.learn:
                gm = gm + jnp.stack([d_hx[k], d_he[k]], axis=-1)
            T = flux_symbol(fa[rows], gm, fb[rows], ds) if self.flux else \
                flux_symbol(gm, gm, gm, ds)                                         # (n_r, N_HALF)
            wr = T @ jnp.conj(P).T                                                  # (n_r, L)
            w = jax.vmap(lambda col: up(col.real) + 1j * up(col.imag), in_axes=1, out_axes=1)(wr)
            ws.append(jnp.concatenate([w, jnp.repeat(w_free, pad, axis=0)]))
            phase = K0 * (jnp.exp(fm_[:, 0]) - 1.0) * ds
            if self.learn:
                phase = phase + up(d_ph[k]) - 1j * up(d_amp[k])
            ra = jnp.exp(fa[:, 1] - fa[:, 0])
            rb = jnp.exp(fb[:, 1] - fb[:, 0])
            s_in = jnp.concatenate([jnp.sqrt(ra), jnp.ones(pad)])
            s_out = jnp.concatenate([jnp.exp(1j * phase) / jnp.sqrt(rb), jnp.ones(pad)])
            scr.append(jnp.stack([s_in, s_out]))
        return jnp.stack(ws), jnp.stack(scr)

    def apply(self, v, wscr):
        if self.n_res:
            ws, scr, gh = wscr
        else:
            ws, scr = wscr
        w = jnp.zeros(N_PAD, dtype=complex).at[:N_WIN].set(v)
        Pf = jax.vmap(half_to_full)(jax.lax.stop_gradient(self.basis))     # (L, N_EXT)
        def odd(u):
            return jnp.concatenate([u, jnp.zeros(u.shape[:-1] + (1,), dtype=u.dtype), -u[..., :0:-1]], axis=-1)
        for k in range(self.n_stages):
            u = scr[k, 0] * w
            spec = jnp.fft.fft(odd(u))
            y = jnp.fft.ifft(Pf * spec[None], axis=-1)[:, :N_PAD]          # (L, N_PAD)
            left = jnp.sum(ws[k].T * y, axis=0)
            if self.quant == 'sym':
                specs = jnp.fft.fft(odd(ws[k].T * u[None]), axis=-1)       # (L, N_EXT)
                right = jnp.fft.ifft(jnp.sum(Pf * specs, axis=0))[:N_PAD]
                left = 0.5 * (left + right)
            if self.n_res:
                g, h = gh[k, :, 0], gh[k, :, 1]                              # (M, N_WIN)
                hu = jnp.zeros((self.n_res, N_PAD), dtype=complex).at[:, :N_WIN].set(h * u[None, :N_WIN])
                q = self.q_res[k, ..., 0] + 1j * self.q_res[k, ..., 1]
                r = jnp.fft.ifft(q * jnp.fft.fft(odd(hu), axis=-1), axis=-1)[:, :N_WIN]
                res = jnp.sum(g * r, axis=0).at[0].set(0.0)
                left = left.at[:N_WIN].add(1e-2 * res)
            w = scr[k, 1] * left
        return w[:N_WIN] * absorber(self.dx)

    def screens(self, f):
        if self.n_res:
            return self.weights(f) + (self.res_screens(f),)
        return self.weights(f)

    def __call__(self, v, f):
        return self.apply(v, self.screens(f))
