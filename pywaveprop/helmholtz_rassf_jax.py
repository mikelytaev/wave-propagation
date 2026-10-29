"""
Rank-adaptive split-step Fourier (RA-SSF) propagator for the 2D Helmholtz
equation (JAX).

The classical split-step Fourier method advances the reduced field
``u = psi exp(-i beta x)`` with the Strang split

    u(x + dx) = P_half F^{-1} S F P_half u,
    P_half = exp(i dx (k(z) - beta) / 2),   S = exp(i dx (sqrt(beta^2 - p^2) - beta)),

which is a *rank-one separable* approximation of the exact one-way symbol of
the half step,

    b(z, p) = exp(i dx/2 (sqrt(k(z)^2 - p^2) - beta))  ~=  P_half(z) S^{1/2}(p).

The rank-one factorisation is exact only at ``p = 0`` (paraxial waves) or for a
homogeneous medium, which is the origin of the wide-angle / refractivity
splitting error of the SSF and of its range-step limitation.

This module replaces the rank-one factorisation by a rank-``r`` cross
approximation of the same symbol matrix,

    b(z_j, p_m) ~= sum_{l=1}^{r} U_l(z_j) V_l(p_m),      (adaptive rank r)

computed by adaptive cross approximation (ACA) with partial pivoting, verified
on random samples and recompressed by a QR/SVD step, so that the entrywise
error is below a prescribed tolerance. The rank is selected automatically:
``r = 1`` recovers the classical SSF, larger ``r`` is needed for larger range
steps, stronger refractivity contrasts and wider propagation angles. One range
step is the symmetric composition of the "right" and "left" quantisations of
the half-step symbol,

    u_hat = sum_l V_l(p) F[ U_l(z) u ],          (r forward transforms)
    u'    = sum_l U_l(z) F^{-1}[ V_l(p) u_hat ], (r inverse transforms)

which for ``r = 1`` is exactly the Strang SSF step above. The leading
pseudo-differential correction of the exact propagator is reproduced by this
composition, so the scheme is consistent for any range step and converges to
the one-way solution with the *exact* wide-angle symbol, without the
``(k - beta)(1 / cos(theta) - 1)`` phase error of the phase-screen SSF. The
cost per step is ``2 r`` FFTs (or ``r + 1`` with ``composition='one_sided'``).

As in :class:`~pywaveprop.helmholtz_ssf_jax.SSFHelmholtzPropagator`, ``F`` is
the transform matched to the lower boundary: DST-I (``'dirichlet'``), DCT-I
(``'neumann'``) or the discrete mixed Fourier transform (``'impedance'``,
Dockery-Kuttler). For the DMFT the two discrete surface-wave modes are
additional "spectral points" ``p^2 = -alpha_d^2`` that are appended as columns
of the symbol matrix, so the low-rank factors propagate them consistently.

Because the range step is no longer limited by the splitting error, the
absorbing layer is embedded in the symbol as a complex heterogeneity
``k^2 -> k^2 (1 + i s(z))`` with a smooth ``s(z) ~ t^2`` profile
(``absorber='profile'``, default): the attenuation is then part of the exact
one-way propagation within the step and does not depend on the number of
steps, in contrast to the per-step cosine window of the SSF, which fails once
a wave crosses the layer within a single step.

Terrain, the lower slab (:class:`~pywaveprop.helmholtz_jax.LowerLayerModel`)
and the interface (``create`` / ``compute`` / grids / pytree) are identical to
the SSF propagator; the low-rank factors are built once per distinct vertical
profile (the slab thickness is range dependent), all with numpy, and the
range loop is a jitted ``jax.lax.fori_loop``.
"""
import cmath as cm
import math as fm

import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util

jax.config.update("jax_enable_x64", True)

from pywaveprop.helmholtz_common import HelmholtzMeshParams2D
from pywaveprop.helmholtz_jax import RegularGrid, LowerLayerModel
from pywaveprop.helmholtz_ssf_jax import DMFTBoundary


# ------------------------------------------------------------------------- ACA
def cross_approximation(row_fn, col_fn, shape, tol, max_rank=128, n_check=512, seed=0):
    """Rank-adaptive cross approximation ``A ~= U @ V`` of a matrix given by
    its row / column generators.

    Adaptive cross approximation with partial pivoting (Bebendorf) draws only
    ``O(r)`` rows and columns of ``A`` (``row_fn(i)`` and ``col_fn(j)`` return
    them as complex 1D arrays). The iteration stops when the estimated relative
    Frobenius norm of the last rank-one update falls below ``tol``; the result
    is then verified on ``n_check`` random entries and refined (the row of the
    worst entry becomes the next pivot) until the maximum sampled entrywise
    error, relative to the maximum sampled magnitude, is below ``tol``. Finally
    the factors are orthogonalised and recompressed by a QR / SVD step.

    Returns ``U`` (``m x r``), ``V`` (``r x n``) and an info dict with the
    rank, the sampled maximum error and the number of generated rows/columns.
    """
    m, n = shape
    rng = np.random.default_rng(seed)
    rows, cols = [], []       # rank-one factors
    used_rows = np.zeros(m, dtype=bool)
    used_cols = np.zeros(n, dtype=bool)
    n_rows_eval = 0
    n_cols_eval = 0
    fro2 = 0.0
    pending_row = m // 2
    tiny = 1e-300

    def residual_row(i):
        v = np.asarray(row_fn(i), dtype=complex)
        if rows:
            v = v - np.asarray([r[i] for r in rows]) @ np.asarray(cols)
        return v

    def residual_col(j):
        u = np.asarray(col_fn(j), dtype=complex)
        if rows:
            u = u - np.asarray(rows).T @ np.asarray([c[j] for c in cols])
        return u

    def add_pivot_from_row(i):
        nonlocal fro2, n_rows_eval, n_cols_eval
        v = residual_row(i)
        n_rows_eval += 1
        used_rows[i] = True
        v_abs = np.where(used_cols, 0.0, np.abs(v))
        j = int(np.argmax(v_abs))
        if v_abs[j] <= tiny:
            return None
        piv = v[j]
        u = residual_col(j) / piv
        n_cols_eval += 1
        used_cols[j] = True
        if rows:
            cross = 2.0 * np.real(np.sum((np.asarray(rows).conj() @ u) * (np.asarray(cols) @ v.conj())))
        else:
            cross = 0.0
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        fro2 = fro2 + cross + (nu * nv) ** 2
        rows.append(u)
        cols.append(v)
        u_abs = np.where(used_rows, 0.0, np.abs(u))
        return nu * nv, int(np.argmax(u_abs))

    # --- ACA with partial pivoting
    while len(rows) < max_rank:
        res = add_pivot_from_row(pending_row)
        if res is None:
            free = np.flatnonzero(~used_rows)
            if free.size == 0:
                break
            pending_row = int(rng.choice(free))
            continue
        step_norm, pending_row = res
        if step_norm <= tol * fm.sqrt(max(fro2, tiny)) and len(rows) > 0:
            break

    # --- verification on random samples, refinement if necessary
    max_err = 0.0
    for _ in range(max_rank):
        ii = rng.integers(0, m, size=n_check)
        jj = rng.integers(0, n, size=n_check)
        # exact entries from a handful of rows would be cheaper, but the
        # column generator is what we have: evaluate distinct sampled rows
        uniq = np.unique(ii)
        exact = {i: np.asarray(row_fn(i), dtype=complex) for i in uniq}
        n_rows_eval += uniq.size
        a_exact = np.asarray([exact[i][j] for i, j in zip(ii, jj)])
        U = np.asarray(rows).T if rows else np.zeros((m, 0), dtype=complex)
        V = np.asarray(cols) if cols else np.zeros((0, n), dtype=complex)
        a_approx = np.einsum('kr,rk->k', U[ii, :], V[:, jj]) if rows else np.zeros(n_check, dtype=complex)
        err = np.abs(a_exact - a_approx)
        scale = max(float(np.max(np.abs(a_exact))), tiny)
        max_err = float(np.max(err)) / scale
        if max_err <= tol or len(rows) >= max_rank:
            break
        worst = int(np.argmax(err))
        i_worst = int(ii[worst])
        if used_rows[i_worst]:
            free = np.flatnonzero(~used_rows)
            if free.size == 0:
                break
            i_worst = int(rng.choice(free))
        res = add_pivot_from_row(i_worst)
        if res is None:
            continue
        _, pending_row = res
        # continue the ACA sweep from the new pivot until its own criterion holds
        while len(rows) < max_rank:
            res = add_pivot_from_row(pending_row)
            if res is None:
                break
            step_norm, pending_row = res
            if step_norm <= tol * fm.sqrt(max(fro2, tiny)):
                break

    U = np.asarray(rows).T if rows else np.zeros((m, 0), dtype=complex)
    V = np.asarray(cols) if cols else np.zeros((0, n), dtype=complex)

    U, V = _recompress(U, V, tol)

    info = dict(rank=U.shape[1], max_sampled_error=max_err, rows_evaluated=n_rows_eval,
                cols_evaluated=n_cols_eval)
    return U, V, info


def _recompress(U, V, tol):
    """Orthogonalise the factors ``U @ V`` and drop numerically null directions
    (singular values below ``1e-3 tol`` relative to the largest)."""
    if U.shape[1] == 0:
        return U, V
    q_u, r_u = np.linalg.qr(U)
    q_v, r_v = np.linalg.qr(V.T)
    w_u, sig, w_vh = np.linalg.svd(r_u @ r_v.T)
    keep = sig > 1e-3 * tol * sig[0]
    keep[0] = True
    w_u, sig, w_vh = w_u[:, keep], sig[keep], w_vh[keep, :]
    sq = np.sqrt(sig)
    return q_u @ (w_u * sq[None, :]), (sq[:, None] * w_vh) @ q_v.T


def _kx_branch(kx2):
    """``sqrt(kx2)`` with ``Im >= 0`` (evanescent / lossy waves must decay in range)."""
    kx = np.sqrt(np.asarray(kx2, dtype=complex))
    return np.where(kx.imag < 0, -kx, kx)


def symbol_factors(k2, p2, beta, dx, tol, max_rank=128):
    """Low-rank factors of the symbol ``exp(i dx (sqrt(k2(z) - p2) - beta))``.

    ``k2`` (complex, per grid point ``z_j``) and ``p2`` (complex, per spectral
    point; may include ``-alpha_d^2`` of surface-wave modes) define the matrix
    ``B[j, m] = exp(i dx (sqrt(k2[j] - p2[m]) - beta))``; returns ``U`` (``z x r``),
    ``V`` (``r x p``) and the ACA info.
    """
    k2 = np.asarray(k2, dtype=complex)
    p2 = np.asarray(p2, dtype=complex)

    def row_fn(j):
        return np.exp(1j * dx * (_kx_branch(k2[j] - p2) - beta))

    def col_fn(m):
        return np.exp(1j * dx * (_kx_branch(k2 - p2[m]) - beta))

    return cross_approximation(row_fn, col_fn, (k2.size, p2.size), tol, max_rank=max_rank)


def exact_one_way_stratified(model, initial, x_out):
    """Reference solution of the semi-discrete one-way equation
    ``du/dx = i (sqrt(D2 + k^2(z)) - beta) u`` for a stratified medium.

    ``D2`` is the DST-I Laplacian on the vertical grid of ``model`` (Dirichlet
    at the ground and at the top of the computational domain) and ``k^2(z)`` the
    *real* part of the model's heterogeneity (no absorbing layer), so the
    operator is Hermitian and is diagonalised exactly by ``numpy.linalg.eigh``
    (cost ``O(z_n^3)``, fine up to a few thousand points). The result is the
    exact solution of the same discrete one-way problem the SSF / RA-SSF
    approximate in range, which isolates their splitting / symbol errors.
    Returns the field on the ranges ``x_out``, shape ``(len(x_out), z_n)``.
    """
    n = model.z_n - 1
    dz = model.dz_m
    k2 = model.beta ** 2 * (1.0 + np.asarray(model.het).real)
    j = np.arange(1, n)
    m = np.arange(1, n)
    basis = np.sin(np.pi * np.outer(j, m) / n) * np.sqrt(2.0 / n)
    p2 = (np.pi * m / (n * dz)) ** 2
    a_mat = (basis * (-p2)[None, :]) @ basis.T + np.diag(k2[1:n])
    lam, w = np.linalg.eigh(a_mat)
    kx = np.sqrt(lam.astype(complex))
    kx = np.where(kx.imag < 0, -kx, kx)
    u0 = np.asarray(initial, dtype=complex)[:model.z_n]
    c0 = w.T @ u0[1:n]
    out = np.zeros((len(x_out), model.z_n), dtype=complex)
    for i, x in enumerate(np.asarray(x_out)):
        out[i, 1:n] = w @ (np.exp(1j * x * (kx - model.beta)) * c0)
    return out


# ------------------------------------------------------------------ propagator
class RankAdaptiveSSFPropagator:
    """Rank-adaptive split-step Fourier propagator, see the module docstring."""

    @classmethod
    def create(cls, freq_hz: float, wave_speed, kz_max, k_bounds, precision, mesh_params: HelmholtzMeshParams2D,
               lower_terrain=None, lower_bc: str = 'dirichlet', lower_impedance_alpha=None,
               lower_layer: LowerLayerModel = None, abs_layer_scale: float = 0.5,
               dx_m: float = None, dz_m: float = None, spectral_margin: float = 2.0,
               composition: str = 'symmetric', absorber: str = 'profile', abs_attenuation_db: float = 40.0,
               symbol_tol: float = None, max_rank: int = 128, symbol_band: bool = True, **_ignored):
        """Build a propagator with the same conventions as
        :meth:`~pywaveprop.helmholtz_ssf_jax.SSFHelmholtzPropagator.create`.

        ``kz_max`` fixes the vertical grid (``dz <= pi / (spectral_margin kz_max)``).
        The range step defaults to the requested *output* step, since the
        low-rank symbol step has no splitting error; with a terrain the SSF
        paraxial rule ``2 pi / (beta sin^2 theta_max)`` is used as a cap so the
        staircase is resolved as in the SSF. ``precision`` is the target
        accumulated symbol error: the entrywise tolerance of the cross
        approximation is ``precision / n_steps`` unless ``symbol_tol`` is given.
        ``absorber`` is ``'profile'`` (complex heterogeneity giving
        ``abs_attenuation_db`` one-way attenuation at ``theta_max``) or ``'window'``
        (the per-step cos^2 window of the SSF). With ``symbol_band`` (default) the
        cross approximation covers only the spectral band
        ``p <= spectral_margin kz_max`` that can carry energy on this grid, and the
        classical rank-one (phase screen) factorisation is used above it, so the
        rank does not depend on ``dz``.
        """
        kz_max = float(kz_max)
        k_bounds = (float(k_bounds[0]), float(k_bounds[1]))
        beta = 0.5 * (k_bounds[0] + k_bounds[1])

        dx_max = mesh_params.dx_output_m or mesh_params.x_size_m / (round(mesh_params.x_n_upper_bound / 2) - 1)
        dz_max = mesh_params.dz_output_m or mesh_params.z_size_m / (round(mesh_params.z_n_upper_bound / 2) - 1)

        sin_theta = min(max(kz_max / beta, 1e-3), 1.0)
        if dz_m is None:
            dz_m = min(dz_max, fm.pi / (spectral_margin * kz_max))
        if dx_m is None:
            dx_m = dx_max
            if lower_terrain is not None:
                dx_m = min(dx_max, 2 * fm.pi / (beta * sin_theta ** 2))

        if mesh_params.dx_output_m:
            x_output_step = fm.ceil(mesh_params.dx_output_m / dx_m)
            dx_m = mesh_params.dx_output_m / x_output_step
        else:
            x_output_step = fm.floor(dx_max / dx_m)
            dx_m = dx_max / x_output_step
        if mesh_params.dz_output_m:
            z_output_step = fm.ceil(mesh_params.dz_output_m / dz_m)
            dz_m = mesh_params.dz_output_m / z_output_step
        else:
            z_output_step = fm.floor(dz_max / dz_m)
            dz_m = dz_max / z_output_step

        x_n = fm.ceil(mesh_params.x_size_m / dx_m) + 1
        z_n_phys = fm.ceil(mesh_params.z_size_m / dz_m) + 1
        z_n_abs = fm.ceil(abs_layer_scale * mesh_params.z_size_m / dz_m)

        if symbol_tol is None:
            symbol_tol = min(max(float(precision) / max(x_n - 1, 1), 1e-10), 1e-2)

        # absorbing profile s(z) = s0 t^2: one-way attenuation exp(-beta s0 L / (6 sin theta))
        abs_strength = 0.0
        if absorber == 'profile' and z_n_abs > 0:
            layer_m = z_n_abs * dz_m
            abs_strength = 6.0 * (abs_attenuation_db / 20.0) * fm.log(10.0) * sin_theta / (beta * layer_m)

        print(f'rassf: beta: {beta}, dx: {dx_m}, dz: {dz_m}, z_n: {z_n_phys + z_n_abs} (abs layer: {z_n_abs}), '
              f'symbol tol: {symbol_tol:.2e}')

        return cls(
            beta=beta,
            dx_m=dx_m,
            dz_m=dz_m,
            x_n=x_n,
            z_n=z_n_phys + z_n_abs,
            z_n_phys=z_n_phys,
            x_grid_scale=x_output_step,
            z_grid_scale=z_output_step,
            freq_hz=freq_hz,
            wave_speed=wave_speed,
            lower_terrain=lower_terrain,
            lower_bc=lower_bc,
            lower_impedance_alpha=lower_impedance_alpha,
            lower_layer=lower_layer,
            composition=composition,
            absorber=absorber,
            abs_strength=abs_strength,
            symbol_tol=symbol_tol,
            max_rank=max_rank,
            symbol_p_max=spectral_margin * kz_max if symbol_band else None,
        )

    def __init__(self, beta: float, dx_m: float, dz_m: float, x_n: int, z_n: int, z_n_phys: int,
                 x_grid_scale: int, z_grid_scale: int, freq_hz: float, wave_speed,
                 lower_terrain=None, lower_bc: str = 'dirichlet', lower_impedance_alpha=None,
                 lower_layer: LowerLayerModel = None, composition: str = 'symmetric',
                 absorber: str = 'profile', abs_strength: float = 0.0, symbol_tol: float = 1e-4,
                 max_rank: int = 128, symbol_p_max: float = None):
        if lower_bc not in ('dirichlet', 'neumann', 'impedance'):
            raise ValueError(f"unknown lower_bc {lower_bc!r}; expected 'dirichlet', 'neumann' or 'impedance'")
        if lower_bc == 'impedance' and lower_impedance_alpha is None:
            raise ValueError("lower_bc='impedance' requires lower_impedance_alpha")
        if composition not in ('symmetric', 'one_sided'):
            raise ValueError(f"unknown composition {composition!r}; expected 'symmetric' or 'one_sided'")
        if absorber not in ('profile', 'window'):
            raise ValueError(f"unknown absorber {absorber!r}; expected 'profile' or 'window'")
        self.beta = float(beta)
        self.dx_m = float(dx_m)
        self.dz_m = float(dz_m)
        self.x_n = int(x_n)
        self.z_n = int(z_n)
        self.z_n_phys = int(z_n_phys)
        self.x_grid_scale = int(x_grid_scale)
        self.z_grid_scale = int(z_grid_scale)
        self.freq_hz = freq_hz
        self.wave_speed = wave_speed
        self.lower_terrain = lower_terrain
        self.lower_bc = lower_bc
        self.lower_impedance_alpha = None if lower_impedance_alpha is None else complex(lower_impedance_alpha)
        self.lower_layer = lower_layer
        self.composition = composition
        self.absorber = absorber
        self.abs_strength = float(abs_strength)
        self.symbol_tol = float(symbol_tol)
        self.max_rank = int(max_rank)
        self.symbol_p_max = None if symbol_p_max is None else float(symbol_p_max)
        self._prepare_arrays()

    # ------------------------------------------------------------------ setup
    def _static(self):
        return {
            'beta': self.beta, 'dx_m': self.dx_m, 'dz_m': self.dz_m, 'x_n': self.x_n, 'z_n': self.z_n,
            'z_n_phys': self.z_n_phys, 'x_grid_scale': self.x_grid_scale, 'z_grid_scale': self.z_grid_scale,
            'freq_hz': self.freq_hz, 'lower_bc': self.lower_bc, 'lower_impedance_alpha': self.lower_impedance_alpha,
            'composition': self.composition, 'absorber': self.absorber, 'abs_strength': self.abs_strength,
            'symbol_tol': self.symbol_tol, 'max_rank': self.max_rank, 'symbol_p_max': self.symbol_p_max,
        }

    def _prepare_arrays(self):
        n = self.z_n - 1
        self._n_ext = 2 * n
        z_grid = RegularGrid(start=0, dx=self.dz_m, n=self.z_n)
        k_arr = np.asarray(2 * jnp.pi * self.freq_hz / self.wave_speed.on_regular_grid(z_grid))
        het = np.asarray(k_arr ** 2 / self.beta ** 2 - 1.0, dtype=complex)

        # absorbing layer: complex heterogeneity profile or per-step window
        z = np.arange(self.z_n) * self.dz_m
        z_phys = (self.z_n_phys - 1) * self.dz_m
        z_top = (self.z_n - 1) * self.dz_m
        t = np.clip((z - z_phys) / max(z_top - z_phys, 1e-300), 0.0, 1.0) if self.z_n > self.z_n_phys \
            else np.zeros(self.z_n)
        if self.absorber == 'profile':
            self._abs_profile = self.abs_strength * t ** 2
            abs_window = np.ones(self.z_n)
        else:
            self._abs_profile = np.zeros(self.z_n)
            abs_window = np.cos(np.pi / 2 * t) ** 2
        self._abs_window = jnp.asarray(abs_window)
        het = het + 1j * self._abs_profile
        self.het = jnp.asarray(het)

        # terrain / slab geometry stored as O(x_n) index arrays (see helmholtz_jax)
        x_grid = RegularGrid(start=0, dx=self.dx_m, n=self.x_n)
        self._z_indices = jnp.arange(self.z_n)
        if self.lower_terrain is not None:
            terrain_idx = np.asarray(jnp.round(
                self.lower_terrain.on_regular_grid(x_grid) / self.dz_m).astype(jnp.int64))
        else:
            terrain_idx = np.zeros(self.x_n, dtype=np.int64)
        if self.lower_layer is not None:
            layer_idx = np.asarray(jnp.round(
                self.lower_layer.thickness.on_regular_grid(x_grid) / self.dz_m).astype(jnp.int64))
            layer_delta_het = complex(self.lower_layer.k2_contrast) / self.beta ** 2
        else:
            layer_idx = np.zeros(self.x_n, dtype=np.int64)
            layer_delta_het = 0.0
        self.lower_terrain_idx = jnp.asarray(terrain_idx)
        self.layer_idx = jnp.asarray(layer_idx)
        self.layer_delta_het = jnp.asarray(layer_delta_het, dtype=complex)

        # spectral points of the transform matched to the boundary condition
        m = np.arange(n + 1)
        p = np.pi * m / (n * self.dz_m)
        p2 = p.astype(complex) ** 2
        if self.lower_bc == 'impedance':
            self._dmft = DMFTBoundary(self.lower_impedance_alpha, self.dz_m, self.z_n, self.beta, self.dx_m)
            self._r = self._dmft.r
            alpha_d = [-cm.log(self._r) / self.dz_m, -cm.log(-1.0 / self._r) / self.dz_m]
            p2 = np.concatenate((p2, [-alpha_d[0] ** 2, -alpha_d[1] ** 2]))
        self._p2 = p2

        # distinct vertical profiles (the slab thickness / terrain height change with range)
        if self.lower_layer is not None:
            keys = np.stack((terrain_idx, layer_idx), axis=1)
            uniq, profile_id = np.unique(keys, axis=0, return_inverse=True)
            profile_id = np.asarray(profile_id).reshape(-1)
        else:
            uniq = np.zeros((1, 2), dtype=np.int64)
            profile_id = np.zeros(self.x_n, dtype=np.int64)
        self.profile_id = jnp.asarray(profile_id)

        half = 0.5 if self.composition == 'symmetric' else 1.0
        # columns handled by the cross approximation: the physical band p <= symbol_p_max
        # (always including the surface-wave points); above it the symbol is factorised
        # like in the SSF, exp(i dx/2 (k(z) - beta)) * exp(i dx/2 (sqrt(beta^2 - p^2) - beta))
        n_p = p2.size
        if self.symbol_p_max is None:
            in_band = np.ones(n_p, dtype=bool)
        else:
            in_band = np.ones(n_p, dtype=bool)
            in_band[:n + 1] = p <= self.symbol_p_max
        band_idx = np.flatnonzero(in_band)
        rest_idx = np.flatnonzero(~in_band)
        s_rest = np.exp(1j * half * self.dx_m * (_kx_branch(self.beta ** 2 - p2[rest_idx]) - self.beta))
        factors = []
        self.ranks = []
        self.aca_info = []
        j_idx = np.arange(self.z_n)
        for lo, thick in uniq:
            prof = het.copy()
            if self.lower_layer is not None:
                inside = (j_idx >= lo) & (j_idx < lo + thick)
                prof = prof + layer_delta_het * inside
            k2 = self.beta ** 2 * (1.0 + prof)
            u_band, v_band, info = symbol_factors(k2, p2[band_idx], self.beta, half * self.dx_m, self.symbol_tol,
                                                  max_rank=self.max_rank)
            u_fac = np.zeros((self.z_n, u_band.shape[1] + (1 if rest_idx.size else 0)), dtype=complex)
            v_fac = np.zeros((u_fac.shape[1], n_p), dtype=complex)
            u_fac[:, :u_band.shape[1]] = u_band
            v_fac[:u_band.shape[1], band_idx] = v_band
            if rest_idx.size:
                u_fac[:, -1] = np.exp(1j * half * self.dx_m * (_kx_branch(k2) - self.beta))
                v_fac[-1, rest_idx] = s_rest
                u_fac, v_fac = _recompress(u_fac, v_fac, self.symbol_tol)
            factors.append((u_fac, v_fac))
            self.ranks.append(int(u_fac.shape[1]))
            self.aca_info.append(info)
        self.rank_max = max(self.ranks)
        r_max = self.rank_max
        n_prof = len(factors)
        u_all = np.zeros((n_prof, r_max, self.z_n), dtype=complex)
        if self.lower_bc == 'impedance':
            v_all = np.zeros((n_prof, r_max, n - 1), dtype=complex)
            v_sw = np.zeros((n_prof, r_max, 2), dtype=complex)
            for q, (u_fac, v_fac) in enumerate(factors):
                r = u_fac.shape[1]
                u_all[q, :r] = u_fac.T
                v_all[q, :r] = v_fac[:, 1:n]
                v_sw[q, :r] = v_fac[:, n + 1:n + 3]
            self._v_sw_all = jnp.asarray(v_sw)
        else:
            # V on the extended (length 2n) FFT grid: the symbol is even in p
            m_ext = np.arange(self._n_ext)
            m_sym = np.minimum(m_ext, self._n_ext - m_ext)
            v_all = np.zeros((n_prof, r_max, self._n_ext), dtype=complex)
            for q, (u_fac, v_fac) in enumerate(factors):
                r = u_fac.shape[1]
                u_all[q, :r] = u_fac.T
                v_all[q, :r] = v_fac[:, m_sym]
            self._v_sw_all = jnp.zeros((n_prof, r_max, 2), dtype=complex)
        self._u_all = jnp.asarray(u_all)
        self._v_all = jnp.asarray(v_all)

    @property
    def ffts_per_step(self) -> int:
        """Number of length-``2n`` FFTs per range step (``2 r`` symmetric, ``r + 1`` one-sided)."""
        return 2 * self.rank_max if self.composition == 'symmetric' else self.rank_max + 1

    # ------------------------------------------------------------------ grids
    def x_computational_grid(self):
        return jnp.arange(0, self.x_n) * self.dx_m

    def x_output_grid(self):
        return self.x_computational_grid()[::self.x_grid_scale]

    def z_computational_grid(self):
        """Full vertical grid including the absorbing layer."""
        return jnp.arange(0, self.z_n) * self.dz_m

    def z_physical_grid(self):
        return jnp.arange(0, self.z_n_phys) * self.dz_m

    def z_output_grid(self):
        return self.z_physical_grid()[::self.z_grid_scale]

    def terrain_mask_row(self, i):
        """Field mask (1 in the medium, 0 inside the terrain) at range step ``i``."""
        return jnp.where(self._z_indices >= self.lower_terrain_idx[i], 1.0 + 0.0j, 0.0 + 0.0j)

    # ----------------------------------------------------------------- pytree
    def _tree_flatten(self):
        dynamic = (self.wave_speed, self.lower_terrain, self.lower_layer,
                   self.het, self._abs_window, self.lower_terrain_idx, self.layer_idx, self.layer_delta_het,
                   self.profile_id, self._u_all, self._v_all, self._v_sw_all)
        static = dict(self._static(), ranks=tuple(self.ranks), rank_max=self.rank_max)
        return dynamic, static

    @classmethod
    def _tree_unflatten(cls, static, dynamic):
        obj = cls.__new__(cls)
        static = dict(static)
        obj.ranks = list(static.pop('ranks'))
        obj.rank_max = static.pop('rank_max')
        for key, val in static.items():
            setattr(obj, key, val)
        (obj.wave_speed, obj.lower_terrain, obj.lower_layer,
         obj.het, obj._abs_window, obj.lower_terrain_idx, obj.layer_idx, obj.layer_delta_het,
         obj.profile_id, obj._u_all, obj._v_all, obj._v_sw_all) = dynamic
        obj._n_ext = 2 * (obj.z_n - 1)
        obj._z_indices = jnp.arange(obj.z_n)
        obj.aca_info = []
        if obj.lower_bc == 'impedance':
            obj._dmft = DMFTBoundary(obj.lower_impedance_alpha, obj.dz_m, obj.z_n, obj.beta, obj.dx_m)
            obj._r = obj._dmft.r
        return obj

    # ------------------------------------------------------------- transforms
    def _extend(self, u):
        if self.lower_bc == 'dirichlet':
            return jnp.concatenate((u[..., :-1], -u[..., -1:0:-1]), axis=-1)
        return jnp.concatenate((u[..., :-1], u[..., -1:0:-1]), axis=-1)

    def _step_fourier(self, u, u_fac, v_fac):
        """One range step with the sine / cosine transform: ``u_fac`` is
        ``(r, z_n)``, ``v_fac`` is ``(r, 2n)`` on the extended FFT grid."""
        if self.composition == 'symmetric':
            u_hat = jnp.sum(v_fac * jnp.fft.fft(self._extend(u_fac * u[None, :]), axis=-1), axis=0)
        else:
            u_hat = jnp.fft.fft(self._extend(u))
        back = jnp.fft.ifft(v_fac * u_hat[None, :], axis=-1)[:, :self.z_n]
        return jnp.sum(u_fac * back, axis=0)

    def _step_impedance(self, u, u_fac, v_fac, v_sw):
        """One range step with the DMFT: ``v_fac`` is ``(r, n-1)`` on the sine
        spectrum, ``v_sw`` is ``(r, 2)`` at the two surface-wave modes."""
        d = self._dmft
        if self.composition == 'symmetric':
            _, c, a, b = d.forward(u_fac * u[None, :])
            c = jnp.sum(v_fac * c, axis=0)
            a = jnp.sum(v_sw[:, 0] * a)
            b = jnp.sum(v_sw[:, 1] * b)
        else:
            _, c, a, b = d.forward(u)
        back = d.inverse(v_fac * c[None, :], v_sw[:, 0] * a, v_sw[:, 1] * b)
        return jnp.sum(u_fac * back, axis=0)

    def _step(self, u, i):
        q = self.profile_id[i]
        u_fac = self._u_all[q]
        v_fac = self._v_all[q]
        if self.lower_bc == 'impedance':
            u = self._step_impedance(u, u_fac, v_fac, self._v_sw_all[q])
        else:
            u = self._step_fourier(u, u_fac, v_fac)
        return u * self._abs_window

    def step(self, u, i=0):
        """Advance the field ``u`` (on :meth:`z_computational_grid`) by one range
        step using the profile of range index ``i`` (no terrain mask)."""
        return self._step(jnp.asarray(u, dtype=complex), i)

    @jax.jit
    def compute(self, initial):
        """Propagate ``initial`` (given on :meth:`z_computational_grid` or on
        :meth:`z_physical_grid`; it is zero-padded into the absorbing layer)
        and return the field on the output grid, shape
        ``(len(x_output_grid), len(z_output_grid))``."""
        initial = jnp.asarray(initial, dtype=complex)
        u0 = jnp.zeros(self.z_n, dtype=complex).at[:initial.shape[0]].set(initial[:self.z_n])
        u0 = u0 * self.terrain_mask_row(0)

        n_x_out = (self.x_n - 1) // self.x_grid_scale + 1
        n_z_out = (self.z_n_phys - 1) // self.z_grid_scale + 1
        results = jnp.zeros((n_x_out, n_z_out), dtype=complex)
        results = results.at[0, :].set(u0[:self.z_n_phys:self.z_grid_scale])

        def body_fun(i, val):
            u, res = val
            u = self._step(u, i)
            u = u * self.terrain_mask_row(i)
            res = jax.lax.cond(i % self.x_grid_scale == 0,
                               lambda: res.at[i // self.x_grid_scale, :].set(u[:self.z_n_phys:self.z_grid_scale]),
                               lambda: res)
            return u, res

        _, results = jax.lax.fori_loop(1, self.x_n, body_fun, (u0, results))
        return results


tree_util.register_pytree_node(RankAdaptiveSSFPropagator,
                               RankAdaptiveSSFPropagator._tree_flatten,
                               RankAdaptiveSSFPropagator._tree_unflatten)
