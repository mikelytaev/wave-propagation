"""Neural rational step: the reference scheme on a coarser grid with a longer step,
with learned coefficients.

One step of ``dx`` = ``n_stages`` sub-steps; each sub-step applies the rational
factors ``prod_l (1 + a_l L) / (1 + b_l L)`` of the exponential propagator in
the psi form of ``helmholtz_tf_jax`` (density-weighted Numerov scheme with the
weight ``m = r^2`` and ``het = k^2 h_xi^2 / beta^2 - 1``) on an internal grid
``DETA / refine``, plus the Liouville rescaling ``(m_prev / m_next)^(1/4)``.
Without learning this is the reference scheme on a coarse grid.

Learned parts (all initialized so that the step equals that coarse scheme):

* per-row complex corrections of ``het`` and ``log m`` for every sub-step,
  from a CNN of the metric columns (zero where the metrics are the identity);
* the Pade coefficients ``(a_l, b_l)`` (global);
* the Numerov weight ``alpha`` (global);
* the strength of the absorbing layer (complex ``het`` above ``N_EVAL``).

Linear in the field, tridiagonal (lineax), differentiable.
"""
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax
import numpy as np

from common import ABSORB_PER_25M, DETA, K0, N_EVAL, N_WIN
from models import ConvNet


def pade_coefs(order, dx, beta=K0):
    from pywaveprop.propagators._utils import pade_propagator_coefs
    t = pade_propagator_coefs(pade_order=(order, order), beta=beta, dx=dx)[0]
    return np.array([[complex(a), complex(b)] for a, b in t])


def upsample(v, q):
    """Band-limited (sine-series) interpolation of a window column by an integer factor q."""
    if q == 1:
        return v
    n = v.shape[0]
    ext = jnp.concatenate([v, jnp.zeros(1, dtype=v.dtype), -v[:0:-1]])
    spec = jnp.fft.fft(ext)
    nf = 2 * n * q
    full = jnp.zeros(nf, dtype=complex).at[:n].set(spec[:n]).at[nf - n + 1:].set(spec[n + 1:])
    return jnp.fft.ifft(full)[:n * q] * q


class NeuralRationalStep(eqx.Module):
    coefs: jnp.ndarray            # (p, 2, 2): (a_l, b_l) as real pairs
    log_alpha: jnp.ndarray        # Numerov weight
    absorb: jnp.ndarray           # absorbing-layer strength (log)
    net: ConvNet
    dx: float = eqx.field(static=True)
    n_stages: int = eqx.field(static=True)
    refine: int = eqx.field(static=True)
    learn: bool = eqx.field(static=True)
    train_coefs: bool = eqx.field(static=True)

    def __init__(self, dx, key, order=8, n_stages=1, refine=2, learn=True, width=48, n_cols=6, train_coefs=False):
        """``train_coefs``: also train the Pade coefficients (off by default: independent
        changes of a_l and b_l break the near-unit modulus of each factor pair)."""
        self.dx, self.n_stages, self.refine, self.learn = float(dx), n_stages, refine, learn
        self.train_coefs = train_coefs
        c = pade_coefs(order, dx / n_stages)
        self.coefs = jnp.asarray(np.stack([c.real, c.imag], axis=-1))
        self.log_alpha = jnp.asarray(math.log(1 / 12))
        self.absorb = jnp.asarray(0.0)
        self.net = ConvNet(2 * n_cols + 1, 4 * n_stages, width, (1, 2, 4, 8, 16, 1), 7, key)

    # ------------------------------------------------------------- metrics
    def _interp_cols(self, f, p):
        n_c = f.shape[0]
        p = p * (n_c - 1)
        j0 = min(int(math.floor(p)), n_c - 2)
        w = p - j0
        return (1 - w) * f[j0] + w * f[j0 + 1]

    def stage_tables(self, f):
        """Per sub-step (het, m, rt, rt_half) on the internal grid and the Liouville factors."""
        K, q = self.n_stages, self.refine
        n = N_WIN * q
        eta_c = jnp.arange(N_WIN) * DETA
        eta_f = jnp.arange(n) * (DETA / q)
        eta_h = eta_f[:-1] + DETA / (2 * q)

        def fine(col, at):
            return jnp.interp(at, eta_c, col)
        if self.learn:
            rows = jnp.arange(0, N_WIN, 4)
            x = jnp.concatenate([f[:, rows].transpose(0, 2, 1).reshape(-1, rows.size),
                                 rows[None].astype(float) * DETA / 256.0], axis=0)
            x0 = jnp.concatenate([jnp.zeros_like(x[:-1]), x[-1:]], axis=0)
            d = (self.net(x) - self.net(x0)).reshape(K, 4, rows.size)
            d = jax.vmap(lambda c: jnp.interp(eta_c, rows.astype(float) * DETA, c))(d.reshape(-1, rows.size))
            d = d.reshape(K, 4, N_WIN)
        t = jnp.clip((eta_f / DETA - N_EVAL) / (N_WIN - N_EVAL), 0.0, 1.0)
        sigma = jnp.exp(self.absorb) * 0.05 * t ** 2
        tabs = []
        for k in range(K):
            fa, fb = self._interp_cols(f, k / K), self._interp_cols(f, (k + 1) / K)
            fm_ = self._interp_cols(f, (k + 0.5) / K)
            lr = fm_[:, 1] - fm_[:, 0]                                   # log r at the coarse nodes
            hx2 = jnp.exp(2 * fm_[:, 0])
            het = fine(hx2, eta_f) * (K0 ** 2) / K0 ** 2 - 1.0 + 0j
            logm = 2 * fine(lr, eta_f) + 0j
            if self.learn:
                het = het + 1e-2 * fine(d[k, 0] + 1j * d[k, 1], eta_f)
                logm = logm + 1e-2 * fine(d[k, 2] + 1j * d[k, 3], eta_f)
            het = het + 1j * sigma
            rt = jnp.exp(fine(lr, eta_f))
            rt_half = jnp.exp(fine(lr, eta_h))
            ma = jnp.exp(2 * fine(fa[:, 1] - fa[:, 0], eta_f))
            mb = jnp.exp(2 * fine(fb[:, 1] - fb[:, 0], eta_f))
            tabs.append((het, jnp.exp(logm), rt, rt_half, (ma / mb) ** 0.25))
        return tabs

    # ---------------------------------------------------------- the scheme
    def _factor(self, a, b, u, het, m, rt, rt_half):
        """Solve (1 + b L) u_new = (1 + a L) u (psi form, Numerov, Dirichlet at both ends)."""
        alpha = jnp.exp(self.log_alpha)
        tau2 = (K0 * DETA / self.refine) ** 2
        wl = jnp.concatenate((jnp.zeros(1), rt[1:] / rt_half))
        wr = jnp.concatenate((rt[:-1] / rt_half, jnp.zeros(1)))

        def coeffs(c):
            F = tau2 * m * (1 + c * het)
            off = alpha * F + c
            diag = F - (wl + wr) * off
            return diag, off[:-1] * wl[1:], off[1:] * wr[:-1]
        d_a, l_a, u_a = coeffs(a)
        rhs = d_a * u
        rhs = rhs.at[1:].add(l_a * u[:-1]).at[:-1].add(u_a * u[1:])
        d_b, l_b, u_b = coeffs(b)
        # rows 1 .. n-2 (psi = 0 at both ends)
        op = lineax.TridiagonalLinearOperator(d_b[1:-1], l_b[1:-1], u_b[1:-1])
        sol = lineax.linear_solve(op, rhs[1:-1]).value
        return jnp.concatenate([jnp.zeros(1, dtype=complex), sol, jnp.zeros(1, dtype=complex)])

    def apply(self, v, tabs):
        q = self.refine
        u = upsample(v, q)
        coefs = self.coefs if self.train_coefs else jax.lax.stop_gradient(self.coefs)
        c = coefs[..., 0] + 1j * coefs[..., 1]
        for het, m, rt, rt_half, liou in tabs:
            u = u * liou
            for l in range(c.shape[0]):
                u = self._factor(c[l, 0], c[l, 1], u, het, m, rt, rt_half)
        return u[::q]

    def screens(self, f):
        return self.stage_tables(f)

    def __call__(self, v, f):
        return self.apply(v, self.stage_tables(f))
