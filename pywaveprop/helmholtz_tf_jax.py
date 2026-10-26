"""
Rational (Padé / ratinterp) one-way Helmholtz propagator in orthogonal
terrain-following coordinates.

The irregular boundary (terrain, seabed) is removed by the orthogonal map of
:mod:`pywaveprop.terrain_transform`.  With ``r = h_eta / h_xi`` and the
generalised density ``rt = rho * r`` the transformed equation for the
Liouville-scaled field ``phi = r**(1/2) psi`` is

    phi_xixi + B phi = 0,
    B phi = rho r^(-1/2) d_eta[(rho r)^(-1) d_eta (r^(-1/2) phi)] + (k^2 h_xi^2 - V) phi,
    V = (r^(1/2))_xixi / r^(1/2),

and every factor ``(1 + a L) / (1 + b L)``, ``L = B / beta**2 - 1``, of the
rational propagator reduces, in terms of ``psi``, to

    b rt d(rt^-1 d psi_new) + beta^2 m (1 + b het) psi_new = (same with a) psi_old,
    m = r^2,   het = (k^2 h_xi^2 - V) / beta^2 - 1,

i.e. to the density-weighted Numerov scheme of
:class:`pywaveprop.helmholtz_jax.RationalHelmholtzPropagator` with the extra
weight ``m``.  No term of the rational approximation is touched, so any order
and any propagation angle the approximation resolves are supported.  The map
is the identity at the top of the domain, where the upper discrete nonlocal
boundary condition of the parent class is used unchanged.  See
``papers/terrain_transform/main.tex``.
"""
import functools
import math as fm
from dataclasses import dataclass, replace
from typing import Callable

import jax
import jax.numpy as jnp
import lineax
import numpy as np
from jax import tree_util

from pywaveprop.grid_optimizer import get_optimal_grid, get_last_ratinterp_xi_bounds
from pywaveprop.helmholtz_common import HelmholtzMeshParams2D
from pywaveprop.helmholtz_jax import RationalHelmholtzPropagator
from pywaveprop.propagators._utils import pade_propagator_coefs
from pywaveprop.terrain_transform import OrthogonalTerrainMap


def fourth_root_factors(order: int, rotation_deg: float = 45.0):
    """Factors ``(a_l, b_l)`` of a rational approximation
    ``(1 + L)**(1/4) ~ C prod_l (1 + a_l L) / (1 + b_l L)``; swapping ``a`` and
    ``b`` approximates ``(1 + L)**(-1/4) / C``.

    The diagonal ``[order/order]`` Padé approximant is taken in
    ``Y = (1 + L) exp(-i alpha) - 1`` (branch cut rotated by ``alpha``,
    Milinazzo, Zala and Brooke 1997), so that its poles leave the real axis:
    unrotated, they lie on the cut ``L < -1``, i.e. inside the evanescent
    spectrum of the discrete operator, where a pair of flux factors at slightly
    different columns can amplify a spurious mode.  The constant ``C`` cancels in
    the flux-factor pairs and is not returned.
    """
    import cmath
    import mpmath
    mpmath.mp.dps = 50
    t = mpmath.taylor(lambda y: (1 + y) ** mpmath.mpf(0.25), 0, 2 * order)
    p, q = mpmath.pade(t, order, order)
    p_roots = mpmath.polyroots([c / p[0] for c in p], maxsteps=500, asc=True)
    q_roots = mpmath.polyroots([c / q[0] for c in q], maxsteps=500, asc=True)
    rot = cmath.exp(-1j * fm.radians(rotation_deg))

    def in_l(y_root):
        # 1 + c Y = (1 - c + c rot) (1 + c' L),  c' = c rot / (1 - c + c rot)
        c = -1 / complex(y_root)
        return c * rot / (1 - c + c * rot)
    return [(in_l(a), in_l(b)) for a, b in zip(p_roots, q_roots)]


def column_classes(h_min, h_max, r_max, n_classes):
    """Group columns by the range of ``h_xi`` on them (bins of
    ``log h_min + log h_max``).  Returns the class of every column and, per
    class, ``(h_min, h_max, r_max)`` over its members."""
    key = np.log(h_min) + np.log(h_max)
    edges = np.linspace(key.min(), key.max() + 1e-12, n_classes + 1)
    cls = np.clip(np.searchsorted(edges, key, side='right') - 1, 0, n_classes - 1)
    used = np.unique(cls)
    cls = np.searchsorted(used, cls)
    bounds = [(float(h_min[cls == k].min()), float(h_max[cls == k].max()), float(r_max[cls == k].max()))
              for k in range(used.size)]
    return cls, bounds


@functools.lru_cache(maxsize=None)
def pade_coefs_cached(order, beta, dx):
    return tuple(tuple(v) for v in pade_propagator_coefs(pade_order=order, beta=beta, dx=dx)[0])


def class_grid(order, classes, kz_max, k_bounds, required_prec, dx_out, dz_out, z_size, x_size):
    """Common ``dxi`` and ``deta`` with one reference wavenumber per class.

    ``classes``: ``(h_min, h_max, r_max)`` per class.  A candidate
    ``dxi = dx_out / n`` is admissible if each class has an entry
    ``(t_j, delta_j)`` of the Padé error table and a ``beta`` in that entry's
    admissible interval ``[l_j, r_j]`` with ``beta dxi <= t_j`` (the error at a
    smaller ``t`` is assumed not larger) and ``delta_j beta / t_j <=
    required_prec`` (accumulation over the entry's own step, as in the flat
    optimizer before snapping).  The class takes ``beta = l_j`` of the entry
    with the largest vertical step; ``deta`` is the smallest class step
    ``dz (h_min / h_max) / r_max``.  Returns ``(cost, dxi, deta, betas,
    deltas)`` or ``None``."""
    from pywaveprop import grid_optimizer as go
    if order not in go._tables_cache:
        go._get_optimal_grid_pade(kz_max, *k_bounds, required_prec, propagator_order=order)
    tab = go._tables_cache[order]
    T = tab.t_betas[:, None]
    D = tab.precs[None, :]
    pre = []
    for h_min, h_max, r_max in classes:
        kz = kz_max * h_min
        k_min, k_max = k_bounds[0] * h_min, k_bounds[1] * h_max
        a, b = k_min ** 2 - kz ** 2, k_max ** 2
        if a <= 0:
            return None
        lo = np.sqrt(b / (tab.xi_b + 1))
        hi = np.minimum(np.sqrt(a / (tab.xi_a + 1)), required_prec * T / D)
        dz_max = min(dz_out * r_max, 2.0 / (0.5 * (k_min + k_max)))
        pre.append((kz, a, lo, hi, dz_max, (h_min / h_max) / r_max))
    best = None
    dz_memo = {}            # the vertical step depends on the class and the table entry only
    for n in range(1, max(int(fm.ceil(dx_out / 0.01)), 1) + 1):
        dx = dx_out / n
        if best is not None and (x_size / dx) * order[1] > best[0]:
            break
        betas, deltas, deta = [], [], fm.inf
        for c, (kz, a, lo, hi, dz_max, factor) in enumerate(pre):
            ok = lo <= np.minimum(hi, T / dx)
            if not ok.any():
                betas = None
                break
            # vertical tolerance t delta 2 sqrt(a) / beta (flat optimizer): largest at beta = l
            score = np.where(ok, T * D / lo, -1.0)
            i, j = np.unravel_index(int(np.argmax(score)), score.shape)
            t, delta, beta = float(tab.t_betas[i]), float(tab.precs[j]), float(lo[i, j])
            if (c, i, j) not in dz_memo:
                dz_prec = t ** 2 * delta / (t / (2 * fm.sqrt(a) / beta))
                if go.fourth_order_error_kz(kz, dz_max) < dz_prec:
                    dz_memo[c, i, j] = dz_max
                else:
                    dz_memo[c, i, j] = go.func_binary_search(lambda z: go.fourth_order_error_kz(kz, z), 0.0001,
                                                             dz_max, dz_prec, 1e-3)
            dz = dz_memo[c, i, j]
            if fm.isnan(dz):
                betas = None
                break
            betas.append(beta)
            deltas.append(delta)
            deta = min(deta, dz * factor)
        if betas is None:
            continue
        deta = dz_out / fm.ceil(dz_out / deta)
        cost = (fm.ceil(x_size / dx) + 1) * (fm.ceil(z_size / deta) + 1) * order[1]
        if best is None or cost < best[0]:
            best = (cost, dx, deta, betas, deltas)
    return best


@dataclass
class TFMedium:
    """Medium seen by the terrain-following propagator.

    ``k2_above(x, z)`` is the squared (complex) wavenumber between ``eta = 0`` and
    the boundary level ``eta_b`` of a strip map, or in the whole domain of a
    half-space map; ``k2_below(x, z, depth_below_boundary)`` is used under the
    boundary of a strip map (sediment).  ``rho_above`` / ``rho_below`` are the
    densities of the two regions: constants, or functions ``rho(x, z)`` (used
    e.g. for a stair-step reference with a flat map; the density then has to be
    constant in the identity region).
    """
    k2_above: Callable
    k2_below: Callable = None
    rho_above: float = 1.0
    rho_below: float = 1.0


class TerrainFollowingHelmholtzPropagator(RationalHelmholtzPropagator):

    @classmethod
    def create(cls, freq_hz: float, terrain_map: OrthogonalTerrainMap, medium: TFMedium,
               kz_max, k_bounds, precision, mesh_params: HelmholtzMeshParams2D,
               rational_approx_order=(7, 8), approx_method='pade', lower_bc='dirichlet',
               upper_bc='nlbc', coarse_dxi_m=None, max_table_size=40_000_000, flux_order=2,
               lower_impedance_alpha=None, impedance_closure='compact', beta_mode='global', n_beta_classes=16,
               absorb_m=None):
        """Choose the grid, build the coordinate tables and the propagator.

        ``beta_mode='column'``: one reference wavenumber per class of columns
        (grouped by the range of ``h_xi`` on the column, ``n_beta_classes``
        bins), so that the Padé approximant covers the local spectrum of each
        column only instead of the global one (:func:`class_grid`).  The
        nonlocal upper condition assumes the same step in every column, so this
        mode needs ``upper_bc='dirichlet'`` with an absorbing layer
        (``absorb_m`` metres added on top of the domain, damping ``0.02 i k^2``
        ramped quadratically).

        ``lower_bc='impedance'`` imposes the Leontovich condition
        ``d psi / dn + lower_impedance_alpha * psi = 0`` on the ground (see
        :func:`pywaveprop.helmholtz_ssf_jax.impedance_alpha`).

        ``k_bounds`` and ``kz_max`` are the physical ones; they are rescaled by the
        range metric ``h_xi`` (local equivalence, Section 7 of the note) before
        the grid optimizer of the parent class is called.  ``mesh_params``
        refers to computational coordinates: ``z_size_m`` is the height of the
        computational domain in ``eta``.
        """
        kz_max = float(kz_max)
        k_bounds = (float(k_bounds[0]), float(k_bounds[1]))
        precision = float(precision)

        # metric statistics on a provisional grid
        deta_p = max(min(terrain_map.L / 200.0, mesh_params.z_size_m / 400), 1e-3)
        if terrain_map.kind == 'strip':
            deta_p = terrain_map.eta_b / max(round(terrain_map.eta_b / deta_p), 1)
        lev = np.arange(0.0, min(terrain_map.eta_lab + 4 * deta_p, mesh_params.z_size_m) + deta_p / 4, deta_p / 2)
        # the metric extremes sit on the ground, at the scale of the terrain smoothing
        dxi_p = max(min(terrain_map.L, terrain_map.smoothing_m or terrain_map.L) / 8, 1.0)
        xi_p = np.linspace(0.0, mesh_params.x_size_m, min(max(int(mesh_params.x_size_m / dxi_p) + 1, 16), 4000))
        mp = terrain_map.metrics(xi_p, lev)
        r_all = np.concatenate([(mp.h_eta / mp.h_xi).ravel(), (mp.h_eta_below / mp.h_xi).ravel()])
        hxi_min, hxi_max = float(mp.h_xi.min()), float(mp.h_xi.max())
        r_max = float(r_all.max())
        print(f'terrain-following map: h_xi in [{hxi_min:.3f}, {hxi_max:.3f}], '
              f'r in [{r_all.min():.3f}, {r_max:.3f}]')

        # local equivalence: k -> k h_xi, k_z -> k_z h_xi.  The optimizer needs the
        # smallest longitudinal wavenumber h_xi^2 (k_min^2 - k_z^2) (k_z scaled by
        # min h_xi) and resolves k_z h_xi with its vertical step: the step it returns
        # is rescaled by min h_xi / max h_xi below.
        kb = (k_bounds[0] * hxi_min, k_bounds[1] * hxi_max)
        kzb = kz_max * hxi_min

        dx_max = mesh_params.dx_output_m or mesh_params.x_size_m / (round(mesh_params.x_n_upper_bound / 2) - 1)
        dz_max = mesh_params.dz_output_m or mesh_params.z_size_m / (round(mesh_params.z_n_upper_bound / 2) - 1)
        # beta * dz < 2 for the NLBC, in the effective vertical step r * deta
        dz_max = min(dz_max * r_max, 2.0 / (0.5 * (kb[0] + kb[1])))

        orders = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)] \
            if rational_approx_order is None else [tuple(rational_approx_order)]
        if absorb_m:
            z_top = terrain_map.z_off + mesh_params.z_size_m
            k2_phys = medium.k2_above

            def k2_absorbing(x, z):
                ramp = np.clip((np.asarray(z) - z_top) / absorb_m, 0.0, 1.0) ** 2
                return np.asarray(k2_phys(x, z), dtype=complex) * (1 + 0.02j * ramp)
            medium = replace(medium, k2_above=k2_absorbing)
            mesh_params = replace(mesh_params, z_size_m=mesh_params.z_size_m + absorb_m)
        beta_classes = None
        if beta_mode == 'column':
            if upper_bc == 'nlbc':
                raise ValueError("beta_mode='column' needs upper_bc='dirichlet' (with absorb_m)")
            if not mesh_params.dx_output_m or not mesh_params.dz_output_m:
                raise ValueError("beta_mode='column' needs dx_output_m and dz_output_m")
            from scipy.ndimage import maximum_filter1d, minimum_filter1d
            r_col = np.maximum((mp.h_eta / mp.h_xi).max(axis=1), (mp.h_eta_below / mp.h_xi).max(axis=1))
            # a step reads the columns within dxi / 2 of its midpoint
            w = 2 * (fm.ceil(dx_max / (xi_p[1] - xi_p[0])) + 1) + 1
            h_lo = minimum_filter1d(mp.h_xi.min(axis=1), w, mode='nearest')
            h_hi = maximum_filter1d(mp.h_xi.max(axis=1), w, mode='nearest')
            cls_p, bounds = column_classes(h_lo, h_hi, maximum_filter1d(r_col, w, mode='nearest'), n_beta_classes)
            best_c = None
            for order in orders:
                g = class_grid(order, bounds, kz_max, k_bounds, precision / mesh_params.x_size_m,
                               mesh_params.dx_output_m, mesh_params.dz_output_m, mesh_params.z_size_m,
                               mesh_params.x_size_m)
                if g is not None and (best_c is None or g[0] < best_c[0]):
                    best_c = (g[0], order) + g[1:]
            if best_c is None:
                raise ValueError("Grid optimization failed for beta_mode='column'")
            _, order, dx, deta, betas, deltas = best_c
            x_step = fm.ceil(mesh_params.dx_output_m / dx - 1e-9)
            z_step = fm.ceil(mesh_params.dz_output_m / deta - 1e-9)
            x_n = fm.ceil(mesh_params.x_size_m / dx) + 1
            z_n = fm.ceil(mesh_params.z_size_m / deta) + 1
            # tables and flux factors keep one global beta: the one of the class of flattest columns
            beta = betas[int(np.argmin([abs(np.log(b[0] * b[1])) for b in bounds]))]
            beta_classes = dict(betas=np.array(betas), cls_p=cls_p, dxi_p=float(xi_p[1] - xi_p[0]),
                                coefs=np.array([pade_coefs_cached(order, bk, dx) for bk in betas], dtype=complex))
            xi_bounds = None
            # actual error of each class approximant over its spectrum (tabulated bound: delta)
            from pywaveprop.grid_optimizer import rational_approx_error
            kzc = kz_max
            errs = []
            for (h_lo_k, h_hi_k, _), bk, ck in zip(bounds, betas, beta_classes['coefs']):
                xi = np.linspace(((k_bounds[0] * h_lo_k) ** 2 - (kzc * h_lo_k) ** 2) / bk ** 2 - 1,
                                 (k_bounds[1] * h_hi_k) ** 2 / bk ** 2 - 1, 400)
                errs.append(float(np.max(rational_approx_error(bk * dx, xi, ck))))
            beta_classes['errors'] = errs
            print('beta classes: ' + ', '.join(f'[{b[0]:.3f}, {b[1]:.3f}] -> beta {bk:.4f} err {e:.1e} (<= {d:.0e})'
                                              for b, bk, e, d in zip(bounds, betas, errs, deltas)))
            orders = []
        best = None
        for order in orders:
            beta_t, dx_t, dz_t = get_optimal_grid(kzb, kb[0], kb[1], precision / mesh_params.x_size_m,
                                                  dx_max=dx_max, dz_max=dz_max,
                                                  propagator_order=order, approx_method=approx_method)
            if fm.isnan(beta_t):
                continue
            xi_bounds_t = get_last_ratinterp_xi_bounds() if approx_method == 'ratinterp' else None
            deta_t = dz_t * (hxi_min / hxi_max) / r_max
            if mesh_params.dx_output_m:
                x_step = fm.ceil(mesh_params.dx_output_m / dx_t)
                dx_t = mesh_params.dx_output_m / x_step
            else:
                x_step = max(fm.floor(dx_max / dx_t), 1)
                dx_t = dx_max / x_step
            if mesh_params.dz_output_m:
                z_step = fm.ceil(mesh_params.dz_output_m / deta_t)
                deta_t = mesh_params.dz_output_m / z_step
            else:
                dz_out = mesh_params.z_size_m / (round(mesh_params.z_n_upper_bound / 2) - 1)
                z_step = max(fm.ceil(dz_out / deta_t), 1)
                deta_t = dz_out / z_step
            x_n_t = fm.ceil(mesh_params.x_size_m / dx_t) + 1
            z_n_t = fm.ceil(mesh_params.z_size_m / deta_t) + 1
            cost = x_n_t * z_n_t * order[1]
            if best is None or cost < best[0]:
                best = (cost, order, beta_t, dx_t, deta_t, x_n_t, z_n_t, x_step, z_step, xi_bounds_t)
        if best is None and beta_classes is None:
            raise ValueError(
                "Grid optimization failed for the terrain-following propagator: the rational approximation "
                f"cannot cover k in [{kb[0]:.4g}, {kb[1]:.4g}] (metric h_xi in [{hxi_min:.3f}, {hxi_max:.3f}]) "
                f"with k_z up to {kz_max:.4g} at this precision; reduce the aperture, relax the precision, "
                "or smooth the boundary / enlarge the transition height")
        if beta_classes is None:
            _, order, beta, dx, deta, x_n, z_n, x_step, z_step, xi_bounds = best
        print(f'rational_approx_order: {order}, beta: {beta}, dxi: {dx}, deta: {deta}')

        if terrain_map.kind == 'strip':
            # the boundary level has to be a grid node
            n_b = fm.ceil(terrain_map.eta_b / deta - 1e-9)
            terrain_map = terrain_map.with_eta_b(n_b * deta)

        tables = cls.build_tables(terrain_map, medium, beta=beta, dxi=dx, deta=deta, x_n=x_n, z_n=z_n,
                                  coarse_dxi_m=coarse_dxi_m, max_table_size=max_table_size)
        if beta_classes is not None:
            tables.update(beta_cls=jnp.asarray(beta_classes['betas']), coefs_cls=jnp.asarray(beta_classes['coefs']),
                          cls_p=jnp.asarray(beta_classes['cls_p']), dxi_p=beta_classes['dxi_p'])

        return cls(order=order, beta=beta, dx_m=dx, dz_m=deta, x_n=x_n, z_n=z_n,
                   x_grid_scale=x_step, z_grid_scale=z_step, freq_hz=freq_hz,
                   approx_method=approx_method, xi_bounds=xi_bounds,
                   tables=tables, lower_bc=lower_bc, upper_bc=upper_bc,
                   terrain_map=terrain_map, flux_order=flux_order, lower_impedance_alpha=lower_impedance_alpha,
                   impedance_closure=impedance_closure)

    # ------------------------------------------------------------------ tables
    @staticmethod
    def build_tables(terrain_map: OrthogonalTerrainMap, medium: TFMedium, *, beta, dxi, deta, x_n, z_n,
                     coarse_dxi_m=None, max_table_size=40_000_000):
        """Coefficient tables on coarse ``xi`` columns (rows below the identity
        region only) plus the identity-region profiles."""
        n_lab = fm.ceil(terrain_map.eta_lab / deta) + 2
        if n_lab > z_n - 4:
            raise ValueError(f"the computational domain ({(z_n - 1) * deta:.1f} m) has to extend at least a few "
                             f"grid nodes above the transition height ({terrain_map.eta_lab:.1f} m)")
        xi_max = (x_n - 1) * dxi
        if coarse_dxi_m is None:
            coarse_dxi_m = max(dxi, min(terrain_map.L, max(terrain_map.smoothing_m, 4 * deta)) / 4)
        n_c = fm.ceil(xi_max / coarse_dxi_m) + 3
        if n_c * n_lab > max_table_size:
            n_c = max(max_table_size // n_lab, 8)
        coarse_dxi_m = xi_max / (n_c - 3)
        xi_c = np.arange(n_c) * coarse_dxi_m

        levels = np.arange(2 * n_lab + 1) * (deta / 2)
        mt = terrain_map.metrics(xi_c, levels)
        node, half = slice(0, 2 * n_lab, 2), slice(1, 2 * n_lab + 1, 2)
        X, Z, A = mt.x[:, node], mt.z[:, node], mt.h_xi[:, node] ** 2
        r_up = mt.h_eta[:, node] / mt.h_xi[:, node]
        r_dn = mt.h_eta_below[:, node] / mt.h_xi[:, node]
        r_half = mt.h_eta[:, half] / mt.h_xi[:, half]

        # V = (r^1/2)_xixi / r^1/2 (second difference along the coarse columns)
        s = np.sqrt(0.5 * (r_up + r_dn))
        V = np.zeros_like(s)
        V[1:-1] = (s[2:] - 2 * s[1:-1] + s[:-2]) / coarse_dxi_m ** 2 / s[1:-1]
        V[0], V[-1] = V[1], V[-2]

        eta_n = np.arange(n_lab) * deta
        strip = terrain_map.kind == 'strip'
        eta_b = terrain_map.eta_b

        def k2(region, Xa, Za):
            if region == 'above':
                return np.asarray(medium.k2_above(Xa, Za), dtype=complex)
            return np.asarray(medium.k2_below(Xa, Za, Za - terrain_map.boundary(Xa)), dtype=complex)

        def rho(region, Xa, Za):
            val = medium.rho_above if region == 'above' else medium.rho_below
            if callable(val):
                return np.asarray(val(Xa, Za), dtype=float)
            return np.full(np.shape(Xa), float(val))

        Xh, Zh = mt.x[:, half], mt.z[:, half]
        het_up = (k2('above', X, Z) * A - V) / beta ** 2 - 1.0
        het_t, m_t, rt_t = het_up, r_up ** 2, rho('above', X, Z) * r_up
        if strip:
            het_dn = (k2('below', X, Z) * A - V) / beta ** 2 - 1.0
            below = eta_n[None, :] > eta_b + 1e-9 * deta
            het_t = np.where(below, het_dn, het_t)
            m_t = np.where(below, r_dn ** 2, m_t)
            rt_t = np.where(below, rho('below', X, Z) * r_dn, rt_t)
            # interface node: cell averages of m / rt and m het / rt (flux form)
            j = int(round(eta_b / deta))
            mu, ru = r_up[:, j] ** 2, rho('above', X[:, j], Z[:, j]) * r_up[:, j]
            md, rd = r_dn[:, j] ** 2, rho('below', X[:, j], Z[:, j]) * r_dn[:, j]
            inv = 0.5 * (1 / ru + 1 / rd)
            mr = 0.5 * (mu / ru + md / rd)
            rt_t[:, j] = 1.0 / inv
            m_t[:, j] = mr / inv
            het_t[:, j] = 0.5 * (mu * het_up[:, j] / ru + md * het_dn[:, j] / rd) / mr
            rho_half = np.where((eta_n + deta / 2 < eta_b)[None, :], rho('above', Xh, Zh), rho('below', Xh, Zh))
        else:
            rho_half = rho('above', Xh, Zh)
        rt_half_t = rho_half * r_half

        # identity region (range independent): x = xi, z = eta + z_off
        eta_all = np.arange(z_n) * deta
        z_all = eta_all + terrain_map.z_off
        if strip:
            k2_top = np.asarray(medium.k2_below(np.zeros(z_n), z_all, z_all - terrain_map.boundary(np.zeros(1))[0]),
                                dtype=complex)
            rho_top = rho('below', np.zeros(1), z_all[-1:])[0]
        else:
            k2_top = np.asarray(medium.k2_above(np.zeros(z_n), z_all), dtype=complex)
            rho_top = rho('above', np.zeros(1), z_all[-1:])[0]
        het_top = k2_top / beta ** 2 - 1.0
        return dict(
            het=jnp.asarray(het_t, dtype=complex), m=jnp.asarray(m_t), rt=jnp.asarray(rt_t),
            h_eta0=jnp.asarray(mt.h_eta[:, 0]),
            rt_half=jnp.asarray(rt_half_t), het_top=jnp.asarray(het_top, dtype=complex),
            rho_top=float(rho_top), n_lab=int(n_lab), coarse_dxi=float(coarse_dxi_m), n_c=int(n_c),
        )

    # ----------------------------------------------------------------- object
    def __init__(self, order, beta, dx_m, dz_m, x_n, z_n, x_grid_scale, z_grid_scale, freq_hz,
                 tables=None, lower_bc='dirichlet', upper_bc='nlbc', coefs=None, upper_nlbc_coefs=None,
                 approx_method='pade', xi_bounds=None, terrain_map=None, flux_order=2, alpha=1 / 12,
                 lower_impedance_alpha=None, impedance_closure='compact', **_ignored):
        """``flux_order`` is the order of the diagonal Padé approximants of
        ``(1 + L)**(+-1/4)`` used by the flux-conserving step (Section 5 of the
        note); ``0`` keeps the Liouville rescaling only.  ``alpha`` is the weight
        of the compact vertical scheme: ``1/12`` (Numerov, fourth order) or ``0``
        (second-order central differences, as in the classical FD PE).

        ``lower_bc='impedance'``: Leontovich condition
        ``d psi / dn + lower_impedance_alpha * psi = 0`` on the ground, i.e.
        ``psi_eta + alpha h_eta psi = 0`` in computational coordinates, closed
        by a ghost node: ``impedance_closure='compact'`` (fourth order, default)
        or ``'mirror'`` (second order, as in the DMFT), see :meth:`_tf_substep`."""
        if lower_bc not in ('dirichlet', 'neumann', 'impedance'):
            raise ValueError("lower_bc must be 'dirichlet', 'neumann' or 'impedance'")
        if lower_bc == 'impedance' and lower_impedance_alpha is None:
            raise ValueError("lower_bc='impedance' needs lower_impedance_alpha")
        if impedance_closure not in ('compact', 'mirror'):
            raise ValueError("impedance_closure must be 'compact' or 'mirror'")
        self.impedance_closure = impedance_closure
        self.lower_impedance_alpha = None if lower_impedance_alpha is None else complex(lower_impedance_alpha)
        if upper_bc not in ('nlbc', 'dirichlet'):
            raise ValueError("upper_bc must be 'nlbc' or 'dirichlet'")
        self.tables = tables
        self.lower_bc = lower_bc
        self.upper_bc = upper_bc
        self.terrain_map = terrain_map
        self.flux_order = int(flux_order)
        self.flux_coefs = jnp.array(fourth_root_factors(self.flux_order) if self.flux_order > 0
                                    else np.zeros((0, 2)), dtype=complex)
        if upper_bc == 'dirichlet' and upper_nlbc_coefs is None:
            upper_nlbc_coefs = jnp.zeros((x_n, max(order), max(order)), dtype=complex)
        super().__init__(order=order, beta=beta, dx_m=dx_m, dz_m=dz_m, x_n=x_n, z_n=z_n,
                         x_grid_scale=x_grid_scale, z_grid_scale=z_grid_scale, freq_hz=freq_hz,
                         wave_speed=None, coefs=coefs, upper_nlbc_coefs=upper_nlbc_coefs,
                         approx_method=approx_method, xi_bounds=xi_bounds)
        nlbc_given = upper_nlbc_coefs is not None
        if tables is not None and 'beta_cls' not in tables:
            # one reference wavenumber for all columns
            tables.update(beta_cls=jnp.array([self.beta]), coefs_cls=self.coefs[None], cls_p=jnp.zeros(1, dtype=int),
                          dxi_p=1e30)
        self.alpha = float(alpha)
        if not nlbc_given and upper_bc == 'nlbc' and self.alpha != 1 / 12:
            # the parent computed the kernels for the Numerov exterior scheme
            gamma = (self.het[-1] - self.het[-2]) / self.dz_m
            self.upper_nlbc_coefs = self._calc_upper_nlbc(self.het[-1] - gamma * (self.z_n - 1) * self.dz_m, gamma)
        # rows on which the flux factors are solved: transition layer plus a few
        # wavelengths, above which consecutive columns coincide and the factor
        # pairs reduce to the identity (pass-through top row)
        n_margin = fm.ceil(4 * 2 * fm.pi / beta / dz_m) + 4
        self.n_flux = int(min(self.tables['n_lab'] + n_margin, z_n)) if tables is not None else z_n

    def _prepare_het_arrays(self):
        # the parent uses het at the two upper nodes for the NLBC: identity region
        self.het = self.tables['het_top']

    def _tree_flatten(self):
        t = self.tables
        dynamic = (t['het'], t['m'], t['rt'], t['rt_half'], t['het_top'], self.upper_nlbc_coefs, t['h_eta0'],
                   t['beta_cls'], t['coefs_cls'], t['cls_p'])
        static = {
            'order': self.order, 'beta': self.beta, 'dx_m': self.dx_m, 'dz_m': self.dz_m,
            'x_n': self.x_n, 'z_n': self.z_n, 'coefs': self.coefs_t, 'x_grid_scale': self.x_grid_scale,
            'z_grid_scale': self.z_grid_scale, 'freq_hz': self.freq_hz, 'approx_method': self.approx_method,
            'xi_bounds': self.xi_bounds, 'lower_bc': self.lower_bc, 'upper_bc': self.upper_bc,
            'flux_order': self.flux_order, 'alpha': self.alpha,
            'lower_impedance_alpha': self.lower_impedance_alpha, 'impedance_closure': self.impedance_closure,
            'tables_static': (t['rho_top'], t['n_lab'], t['coarse_dxi'], t['n_c'], t['dxi_p']),
        }
        return dynamic, static

    @classmethod
    def _tree_unflatten(cls, static, dynamic):
        static = dict(static)
        rho_top, n_lab, coarse_dxi, n_c, dxi_p = static.pop('tables_static')
        tables = dict(het=dynamic[0], m=dynamic[1], rt=dynamic[2], rt_half=dynamic[3], het_top=dynamic[4],
                      h_eta0=dynamic[6], beta_cls=dynamic[7], coefs_cls=dynamic[8], cls_p=dynamic[9],
                      rho_top=rho_top, n_lab=n_lab, coarse_dxi=coarse_dxi, n_c=n_c, dxi_p=dxi_p)
        return cls(tables=tables, upper_nlbc_coefs=dynamic[5], **static)

    # ---------------------------------------------------------- coefficients
    def _columns_at(self, xi):
        """Interpolated (het, m, rt, rt_half) over the whole column at range ``xi``
        and ``h_eta`` on the ground."""
        t = self.tables
        c = xi / t['coarse_dxi']
        i0 = jnp.clip(jnp.floor(c).astype(jnp.int64), 0, t['n_c'] - 2)
        w = c - i0

        def lerp(tab):
            return (1 - w) * tab[i0] + w * tab[i0 + 1]

        n_lab = t['n_lab']
        het = t['het_top'].at[:n_lab].set(lerp(t['het']))
        m = jnp.ones(self.z_n).at[:n_lab].set(lerp(t['m']))
        rt = jnp.full(self.z_n, t['rho_top']).at[:n_lab].set(lerp(t['rt']))
        rt_half = jnp.full(self.z_n - 1, t['rho_top']).at[:n_lab].set(lerp(t['rt_half']))
        return het, m, rt, rt_half, lerp(t['h_eta0'])

    # ------------------------------------------------------------ propagation
    def _tf_substep(self, a, b, u, het, m, rt, rt_half, h_eta0, upper_bound, beta=None):
        alpha = self.alpha
        tau2 = ((self.beta if beta is None else beta) * self.dz_m) ** 2
        wl = jnp.concatenate((jnp.zeros(1), rt[1:] / rt_half))      # rt_j / rt_{j-1/2}
        wr = jnp.concatenate((rt[:-1] / rt_half, jnp.zeros(1)))     # rt_j / rt_{j+1/2}

        def coeffs(c):
            F = tau2 * m * (1 + c * het)
            off = alpha * F + c                                 # multiplies wl_j (from j-1) / wr_j (from j+1)
            diag = F - (wl + wr) * (alpha * F + c)
            lower = off[:-1] * wl[1:]                           # row j, column j-1
            upper = off[1:] * wr[:-1]                           # row j, column j+1
            if self.lower_bc == 'neumann':
                # mirror node u_-1 = u_1 with even coefficients
                diag = diag.at[0].set(F[0] - 2 * wr[0] * (alpha * F[0] + c))
                upper = upper.at[0].set(2 * wr[0] * off[1])
            return diag, lower, upper

        d_a, l_a, u_a = coeffs(a)
        rhs = d_a * u
        rhs = rhs.at[1:].add(l_a * u[:-1])
        rhs = rhs.at[:-1].add(u_a * u[1:])
        d_b, l_b, u_b = coeffs(b)
        if self.lower_bc == 'impedance':
            # compact closure (u_1 - u_-1) + g (u_-1 + 4 u_0 + u_1) / 3 = 0, g = alpha h_eta deta:
            # the discrete vertical wavenumber of the reflection is 3 sin q / (2 + cos q) = q + O(q^5)
            # for every impedance; 'mirror' is (u_1 - u_-1) + 2 g u_0 = 0 (sin q, second order).
            # The coefficients of row 0 at the ghost node are extrapolated quadratically.
            g = self.lower_impedance_alpha * h_eta0 * self.dz_m
            if self.impedance_closure == 'mirror':
                p0, q0 = 2 * g, 1.0
            else:
                p0, q0 = (4 * g / 3) / (1 - g / 3), (1 + g / 3) / (1 - g / 3)
            m_g = 3 * m[0] - 3 * m[1] + m[2]
            het_g = 3 * het[0] - 3 * het[1] + het[2]
            rth_g = 3 * rt_half[0] - 3 * rt_half[1] + rt_half[2]
            wl0 = rt[0] / rth_g

            def ghost(c):
                return (alpha * tau2 * m_g * (1 + c * het_g) + c) * wl0     # row 0, column -1

            def diag0(c):
                # the term -wl_0 (alpha F_0 + c) of the diagonal (wl_0 = 0 in coeffs)
                return -wl0 * (alpha * tau2 * m[0] * (1 + c * het[0]) + c)

            g_a = ghost(a)
            rhs = rhs.at[0].add(g_a * (p0 * u[0] + q0 * u[1]) + diag0(a) * u[0])
            g_b = ghost(b)
            d_b = d_b.at[0].add(g_b * p0 + diag0(b))
            u_b = u_b.at[0].add(q0 * g_b)
        d_b = d_b.at[-1].set(upper_bound[1])
        l_b = l_b.at[-1].set(upper_bound[0])
        rhs = rhs.at[-1].set(upper_bound[2])
        if self.lower_bc == 'dirichlet':
            # psi_0 = 0 is eliminated rather than imposed by a unit row, which a
            # pivoting solver satisfies only up to round-off amplified by |c / 1|
            op = lineax.TridiagonalLinearOperator(d_b[1:].astype(complex), l_b[1:].astype(complex),
                                                  u_b[1:].astype(complex))
            sol = lineax.linear_solve(op, rhs[1:].astype(complex)).value
            return jnp.concatenate((jnp.zeros(1, dtype=complex), sol))
        op = lineax.TridiagonalLinearOperator(d_b.astype(complex), l_b.astype(complex), u_b.astype(complex))
        return lineax.linear_solve(op, rhs.astype(complex)).value

    def _flux_pair(self, y, cols_from, cols_to):
        """``B_to^(-1/4) B_from^(1/4) y`` (Padé factors, Liouville rescaling in
        between) on the rows ``[0, n_flux)``; the top row is passed through."""
        n = self.n_flux
        cf = tuple(c[:n] for c in cols_from[:3]) + (cols_from[3][:n - 1], cols_from[4])
        ct = tuple(c[:n] for c in cols_to[:3]) + (cols_to[3][:n - 1], cols_to[4])
        u = y[:n]
        for l in range(self.flux_coefs.shape[0]):
            a, b = self.flux_coefs[l]
            u = self._tf_substep(a, b, u, *cf, (0.0, 1.0, u[-1]))
        u = u * (cf[1] / ct[1]) ** 0.25
        for l in range(self.flux_coefs.shape[0]):
            a, b = self.flux_coefs[l]
            u = self._tf_substep(b, a, u, *ct, (0.0, 1.0, u[-1]))
        return y.at[:n].set(u)

    def _tf_step(self, initial, upper_convolution, cols_prev, cols_mid, cols_next, k=0):
        """One range step ``B_(n+1)^(-1/4) R(L_(n+1/2)) B_n^(1/4)`` written as
        ``[B_(n+1)^(-1/4) B_mid^(1/4)] R [B_mid^(-1/4) B_n^(1/4)]``: the bracketed
        pairs are the identity in the exterior, so the upper NLBC of the plain
        rational chain applies unchanged.

        ``R`` uses the reference wavenumber ``beta_k`` of class ``k``:
        ``exp(i dxi sqrt(B)) = exp(i beta_k dxi) R_k(L_k)``, with
        ``L_k = (beta / beta_k)^2 (1 + L) - 1``; the envelope keeps the global
        ``beta`` (factor ``exp(i (beta_k - beta) dxi)``).  The flux factors are
        exact powers of ``1 + L`` and keep the global ``beta``."""
        if self.flux_order > 0:
            y = self._flux_pair(initial, cols_prev, cols_mid)
        else:
            y = initial * (cols_prev[1] / cols_mid[1]) ** 0.25
        het, m, rt, rt_half, h_eta0 = cols_mid
        beta_k = self.tables['beta_cls'][k]
        coefs_k = self.tables['coefs_cls'][k]
        het = (het + 1) * (self.beta / beta_k) ** 2 - 1
        upper_field = jnp.zeros(len(self.coefs), dtype=complex)
        use_nlbc = self.upper_bc == 'nlbc'

        def substep(i, val):
            y0, upper_field = val
            if use_nlbc:
                upper_bound = (1, -self.upper_nlbc_coefs[0, i, i],
                               upper_convolution[i] + self.upper_nlbc_coefs[0, i] @ upper_field)
            else:
                upper_bound = (0.0, 1.0, 0.0)
            y1 = self._tf_substep(coefs_k[i, 0], coefs_k[i, 1], y0, het, m, rt, rt_half, h_eta0, upper_bound,
                                  beta=beta_k)
            return y1, upper_field.at[i].set(y1[-1])

        y, upper_field = jax.lax.fori_loop(0, len(self.coefs), substep, (y, upper_field))
        y = y * jnp.exp(1j * (beta_k - self.beta) * self.dx_m)
        if self.flux_order > 0:
            y = self._flux_pair(y, cols_mid, cols_next)
        else:
            y = y * (cols_mid[1] / cols_next[1]) ** 0.25
        return y, upper_field

    @jax.jit
    def compute(self, initial):
        """March ``initial`` (the field on the trajectory ``xi = 0``) through the
        domain; returns the field on the computational output grid."""
        # one row per point of x_output_grid(), i.e. per i = 0, s, 2s, ... <= x_n - 1
        results = jnp.zeros(shape=((self.x_n - 1) // self.x_grid_scale + 1, (self.z_n - 1) // self.z_grid_scale + 1),
                            dtype=complex)
        if self.lower_bc == 'dirichlet':
            initial = initial.at[0].set(0.0)
        results = results.at[0, :].set(initial[::self.z_grid_scale])

        def body_fun(i, val):
            y0, res, upper_field = val
            cols_prev = self._columns_at((i - 1) * self.dx_m)
            cols_mid = self._columns_at((i - 0.5) * self.dx_m)
            cols_next = self._columns_at(i * self.dx_m)
            upper_convolution = self._convolution(self.upper_nlbc_coefs, upper_field, i)
            cls_p = self.tables['cls_p']
            k = cls_p[jnp.clip(jnp.round((i - 0.5) * self.dx_m / self.tables['dxi_p']).astype(jnp.int64),
                               0, cls_p.shape[0] - 1)]
            y1, upper_field_i = self._tf_step(y0, upper_convolution, cols_prev, cols_mid, cols_next, k)
            res = jax.lax.cond(i % self.x_grid_scale == 0,
                               lambda: res.at[i // self.x_grid_scale, :].set(y1[::self.z_grid_scale]),
                               lambda: res)
            upper_field = upper_field.at[i].set(upper_field_i)
            return y1, res, upper_field

        zeros_field = jnp.zeros(shape=(self.x_n, max(self.order)), dtype=complex)
        _, results, _ = jax.lax.fori_loop(1, self.x_n, body_fun, (initial, results, zeros_field))
        return results

    # ------------------------------------------------------------- geometry
    def eta_computational_grid(self):
        return np.arange(self.z_n) * self.dz_m

    def output_metrics(self):
        """Physical coordinates ``(x, z)`` of the output nodes, each ``(n_x_out, n_z_out)``."""
        xi = np.asarray(self.x_output_grid())
        eta = self.eta_computational_grid()[::self.z_grid_scale]
        tm = self.terrain_map
        n_lab = min(fm.ceil(tm.eta_lab / self.dz_m) + 2, self.z_n)
        lev = np.arange(2 * n_lab + 1) * (self.dz_m / 2)
        mt = tm.metrics(xi, lev)
        X = np.repeat(xi[:, None], eta.size, axis=1)
        Z = np.repeat((eta + tm.z_off)[None, :], xi.size, axis=0)
        idx = np.arange(0, n_lab, self.z_grid_scale)
        X[:, :idx.size] = mt.x[:, 2 * idx]
        Z[:, :idx.size] = mt.z[:, 2 * idx]
        return X, Z

    def initial_points(self):
        """Physical ``(x, z)`` of the computational nodes of the starting trajectory ``xi = 0``."""
        tm = self.terrain_map
        n_lab = min(fm.ceil(tm.eta_lab / self.dz_m) + 2, self.z_n)
        lev = np.arange(2 * n_lab + 1) * (self.dz_m / 2)
        mt = tm.metrics(np.zeros(1), lev)
        eta = self.eta_computational_grid()
        X = np.zeros(self.z_n)
        Z = eta + tm.z_off
        X[:n_lab] = mt.x[0, 0:2 * n_lab:2]
        Z[:n_lab] = mt.z[0, 0:2 * n_lab:2]
        return X, Z


tree_util.register_pytree_node(TerrainFollowingHelmholtzPropagator,
                               TerrainFollowingHelmholtzPropagator._tree_flatten,
                               TerrainFollowingHelmholtzPropagator._tree_unflatten)
