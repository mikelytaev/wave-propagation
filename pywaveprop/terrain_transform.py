"""
Orthogonal terrain-following coordinates for the rational one-way Helmholtz solver.

The physical domain bounded by an irregular boundary (terrain in tropospheric
propagation, the seabed in underwater acoustics) is mapped onto a rectangle
``(xi, eta)``. The map is built in two stages (see
``papers/terrain_transform/main.tex`` for the derivation):

1. a family of *level curves* ``z = Z(x, eta)``, ``eta = const``, that starts at
   the boundary and relaxes to horizontal lines.  It is the harmonic extension of
   the boundary (Poisson kernel above it, ``sinh`` kernel in the water strip under
   a flat sea surface) multiplied by a C2 taper, so that ``Z(x, eta) = eta + z_off``
   exactly for ``eta >= eta_lab``.  Small-scale features of the boundary therefore
   decay within their own horizontal scale, as in a conformal map;
2. the *orthogonal trajectories* of this family, labelled by the abscissa
   ``xi`` at which they leave the identity region ``eta >= eta_lab``.

The coordinates ``(xi, eta)`` are orthogonal, so the Laplacian has no mixed
derivative,

    Delta = 1/(h_xi h_eta) [d_xi (r d_xi) + d_eta (r^-1 d_eta)],   r = h_eta / h_xi,

and they coincide with the physical ones (up to the constant vertical offset
``z_off``) in the identity region, where the upper nonlocal boundary condition
is imposed.  This module is NumPy only; the propagator that uses it lives in
:mod:`pywaveprop.helmholtz_tf_jax`.
"""
from dataclasses import dataclass

import numpy as np


def _taper(t):
    """C2 taper: 1 at t <= 0, 0 at t >= 1 (complement of the quintic smoothstep)."""
    t = np.clip(t, 0.0, 1.0)
    return 1.0 - t ** 3 * (10.0 - 15.0 * t + 6.0 * t ** 2)


def _taper_d(t):
    inside = (t > 0.0) & (t < 1.0)
    t = np.clip(t, 0.0, 1.0)
    return np.where(inside, -30.0 * t ** 2 * (1.0 - t) ** 2, 0.0)


@dataclass
class TerrainMapMetrics:
    """Metric quantities of the orthogonal map on a set of ``(xi, eta)`` nodes.

    All arrays have shape ``(n_xi, n_eta)``.  ``h_eta`` and ``r`` are one-sided at
    an interface level where the level-curve family is only continuous (the
    seabed of the strip map): ``*_below`` refers to the side ``eta < eta_b``.
    """
    xi: np.ndarray
    eta: np.ndarray
    x: np.ndarray
    z: np.ndarray
    h_xi: np.ndarray
    h_eta: np.ndarray
    h_eta_below: np.ndarray


class OrthogonalTerrainMap:
    """Orthogonal terrain-following map ``(xi, eta) -> (x, z)``.

    Parameters
    ----------
    x_m, boundary_m :
        Samples of the irregular boundary ``z = B(x)`` (piecewise-linear).
    x_max_m :
        Range extent of the computational domain ``xi in [0, x_max_m]``.
    kind :
        ``'half_space'`` -- the domain is ``z > B(x)``, ``eta = 0`` is the boundary
        (terrain in tropospheric propagation, ``z`` up);
        ``'strip'`` -- the domain is ``z > 0`` with a flat surface ``z = 0`` at
        ``eta = 0`` and the boundary at the level ``eta = eta_b`` (seabed in
        underwater acoustics, ``z`` down).
    transition_m :
        Height ``L`` of the taper above the boundary level; for
        ``eta >= eta_b + L`` the map is the identity ``x = xi, z = eta + z_off``.
    z_ref_m :
        ``'half_space'`` only: reference height ``z_ref`` (``z_off = z_ref``); the
        level curves relax to ``z = eta + z_ref``.  ``z_ref = max(B)`` keeps the
        map regular for any transition height.
    eta_b_m :
        ``'strip'`` only: computational level of the boundary (``z_off = 0``).
        ``eta_b = max(B)`` keeps the water column regular for any depth.
    strip_profile :
        ``'strip'`` only: level curves in the water column.  ``'linear'``
        (sigma coordinates, ``Z = eta B / eta_b``) is regular for any bathymetry;
        ``'harmonic'`` (``sinh`` kernel) confines small-scale relief near the seabed
        but its ``Z_eta`` can become negative next to steep, high relief.
    mirror_start :
        Extend the boundary evenly about ``x = 0``: the starting trajectory
        ``xi = 0`` is then the vertical line ``x = 0`` and the source aperture can
        be given as a function of height.  Otherwise the boundary samples have to
        cover ``[-pad_m, 0]`` and the starting field is given on a curved line
        (see :meth:`TerrainFollowingHelmholtzPropagator.initial_points`).
    smoothing_m :
        Width of the Gaussian filter applied to the boundary (removes the corners
        of the piecewise-linear samples; the map needs a C2 boundary).
    dx_fine_m :
        Spacing of the uniform grid the boundary is resampled on (FFT grid).
    """

    def __init__(self, x_m, boundary_m, *, x_max_m: float, kind: str = 'half_space',
                 transition_m: float, z_ref_m: float = None, eta_b_m: float = None,
                 smoothing_m: float = 0.0, dx_fine_m: float = None, pad_m: float = None,
                 mirror_start: bool = True, strip_profile: str = 'linear'):
        if kind not in ('half_space', 'strip'):
            raise ValueError("kind must be 'half_space' or 'strip'")
        if strip_profile not in ('linear', 'harmonic'):
            raise ValueError("strip_profile must be 'linear' or 'harmonic'")
        self.strip_profile = strip_profile
        x_m = np.asarray(x_m, dtype=float)
        boundary_m = np.asarray(boundary_m, dtype=float)
        self._args = dict(x_m=x_m, boundary_m=boundary_m, x_max_m=x_max_m, kind=kind,
                          transition_m=transition_m, z_ref_m=z_ref_m, eta_b_m=eta_b_m,
                          smoothing_m=smoothing_m, dx_fine_m=dx_fine_m, pad_m=pad_m,
                          mirror_start=mirror_start, strip_profile=strip_profile)
        self.kind = kind
        self.x_max_m = float(x_max_m)
        self.L = float(transition_m)
        self.smoothing_m = float(smoothing_m)
        if self.L <= 0:
            raise ValueError("transition_m must be positive")

        if kind == 'half_space':
            self.eta_b = 0.0
            self.z_off = float(np.max(boundary_m) if z_ref_m is None else z_ref_m)
        else:
            self.eta_b = float(np.max(boundary_m) if eta_b_m is None else eta_b_m)
            self.z_off = 0.0
        self.eta_lab = self.eta_b + self.L

        # Uniform FFT grid on [0, x_hi]; the boundary is extended evenly at x = 0
        # (``mirror_start``: the trajectory xi = 0 is then the vertical line x = 0) and at x_hi.
        if dx_fine_m is None:
            dx_fine_m = max(min(self.L, self.smoothing_m or self.L) / 8.0, self.x_max_m / 2 ** 16)
        if pad_m is None:
            pad_m = 2.0 * self.L + 4.0 * self.smoothing_m
        x_lo = 0.0 if mirror_start else -pad_m
        n = int(np.ceil((self.x_max_m + pad_m - x_lo) / dx_fine_m)) + 1
        self.dx_fine = dx_fine_m
        self.x_fine = x_lo + np.arange(n) * dx_fine_m
        b = np.interp(self.x_fine, x_m, boundary_m) - (self.z_off + self.eta_b)
        ext = np.concatenate([b, b[-2:0:-1]])
        self._kappa = 2 * np.pi * np.fft.fftfreq(ext.size, d=dx_fine_m)
        self._b_hat = np.fft.fft(ext) * np.exp(-0.5 * (self._kappa * self.smoothing_m) ** 2)
        self._n = n
        # smoothed boundary, i.e. the boundary actually used by the solver
        self.boundary_fine = np.fft.ifft(self._b_hat).real[:n] + self.z_off + self.eta_b

    def with_eta_b(self, eta_b_m: float) -> 'OrthogonalTerrainMap':
        """Copy of a strip map with the boundary placed at the level ``eta_b_m``."""
        return OrthogonalTerrainMap(**{**self._args, 'eta_b_m': eta_b_m})

    def max_slope_deg(self) -> float:
        """Maximum slope of the (smoothed) boundary inside ``[0, x_max_m]``."""
        inside = (self.x_fine >= 0) & (self.x_fine <= self.x_max_m)
        slope = np.gradient(self.boundary_fine, self.dx_fine)[inside]
        return float(np.degrees(np.arctan(np.max(np.abs(slope)))))

    # ------------------------------------------------------------------ family
    def _kernel(self, eta, side):
        """Fourier multiplier of ``u(., eta)`` and of ``du/deta``."""
        k = self._kappa
        ak = np.abs(k)
        if self.kind == 'strip' and (eta < self.eta_b or (eta == self.eta_b and side == 'below')):
            eb = self.eta_b
            if self.strip_profile == 'linear':
                # sigma coordinates: Z = eta B / eta_b, Z_eta = B / eta_b > 0 for any bathymetry
                return np.full_like(ak, eta / eb), np.full_like(ak, 1.0 / eb)
            # sinh(k eta) / sinh(k eta_b), written without overflow
            with np.errstate(divide='ignore', invalid='ignore'):
                den = 1.0 - np.exp(-2 * ak * eb)
                K = np.exp(ak * (eta - eb)) * (1.0 - np.exp(-2 * ak * eta)) / den
                K_eta = ak * np.exp(ak * (eta - eb)) * (1.0 + np.exp(-2 * ak * eta)) / den
            K = np.where(ak == 0, eta / eb, K)
            K_eta = np.where(ak == 0, 1.0 / eb, K_eta)
            return K, K_eta
        t = eta - self.eta_b
        w = _taper(t / self.L)
        w_d = _taper_d(t / self.L) / self.L
        e = np.exp(-ak * t)
        return w * e, (w_d - ak * w) * e

    def family(self, eta: float, side: str = 'above'):
        """``Z - eta - z_off``, ``Z_x``, ``Z_xx``, ``Z_eta``, ``Z_x eta`` on ``x_fine``."""
        K, K_eta = self._kernel(eta, side)
        ik = 1j * self._kappa
        n = self._n
        stack = np.fft.ifft(np.stack([self._b_hat * K, ik * self._b_hat * K,
                                      ik ** 2 * self._b_hat * K, self._b_hat * K_eta,
                                      ik * self._b_hat * K_eta]), axis=1).real[:, :n]
        u, u_x, u_xx, u_eta, u_xeta = stack
        return u, u_x, u_xx, 1.0 + u_eta, u_xeta

    def boundary(self, x):
        """Smoothed boundary ``B(x)`` used by the solver."""
        return np.interp(x, self.x_fine, self.boundary_fine)

    # ------------------------------------------------------------ trajectories
    def _rhs(self, fam, X, P):
        u, u_x, u_xx, z_eta, u_xeta = (np.interp(X, self.x_fine, f) for f in fam)
        q = 1.0 + u_x ** 2
        g = -u_x * z_eta / q
        g_x = -(u_xx * z_eta + u_x * u_xeta) / q + 2.0 * u_x ** 2 * u_xx * z_eta / q ** 2
        return g, g_x * P

    def _coarse_levels(self, step):
        """Levels at which the trajectories are integrated: uniform pieces on
        ``[0, eta_b]`` (strip map) and ``[eta_b, eta_lab]``."""
        pieces = []
        bounds = [0.0, self.eta_b, self.eta_lab] if self.kind == 'strip' else [0.0, self.eta_lab]
        for a, b in zip(bounds[:-1], bounds[1:]):
            n = max(int(np.ceil((b - a) / step)), 8)
            pieces.append(np.linspace(a, b, n + 1))
        return pieces

    def _integrate(self, xi, pieces):
        """RK4 integration of the trajectories and of ``P = dX/dxi`` downwards from
        ``eta_lab`` through the coarse levels; metrics on every coarse level.

        Returns one dict per piece with arrays of shape ``(n_xi, n_levels)``."""
        x_cur, p_cur = xi.copy(), np.ones_like(xi)
        out = [None] * len(pieces)
        for ip in range(len(pieces) - 1, -1, -1):
            lev = pieces[ip]
            side = 'below' if (self.kind == 'strip' and ip == 0) else 'above'
            X = np.empty((xi.size, lev.size))
            P = np.empty((xi.size, lev.size))
            X[:, -1], P[:, -1] = x_cur, p_cur
            f_hi = self.family(lev[-1], side)
            for k in range(lev.size - 1, 0, -1):
                h = lev[k] - lev[k - 1]
                f_mid, f_lo = self.family(0.5 * (lev[k] + lev[k - 1]), side), self.family(lev[k - 1], side)
                k1x, k1p = self._rhs(f_hi, x_cur, p_cur)
                k2x, k2p = self._rhs(f_mid, x_cur - 0.5 * h * k1x, p_cur - 0.5 * h * k1p)
                k3x, k3p = self._rhs(f_mid, x_cur - 0.5 * h * k2x, p_cur - 0.5 * h * k2p)
                k4x, k4p = self._rhs(f_lo, x_cur - h * k3x, p_cur - h * k3p)
                x_cur = x_cur - h / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
                p_cur = p_cur - h / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
                X[:, k - 1], P[:, k - 1] = x_cur, p_cur
                f_hi = f_lo
            Z = np.empty_like(X)
            h_xi = np.empty_like(X)
            h_eta = np.empty_like(X)
            for k, eta in enumerate(lev):
                u, u_x, _, z_eta, _ = (np.interp(X[:, k], self.x_fine, f) for f in self.family(eta, side))
                sq = np.sqrt(1.0 + u_x ** 2)
                Z[:, k] = eta + self.z_off + u
                h_xi[:, k] = P[:, k] * sq
                h_eta[:, k] = z_eta / sq
            out[ip] = dict(eta=lev, x=X, z=Z, h_xi=h_xi, h_eta=h_eta)
        return out

    def metrics(self, xi, eta_levels, coarse_step=None) -> TerrainMapMetrics:
        """Metrics of the map at the nodes ``xi x eta_levels``.

        The orthogonal trajectories are integrated downwards from the identity
        region (RK4) on coarse levels of spacing ``coarse_step`` (default: an
        eighth of the smallest horizontal scale of the boundary) and the metric
        quantities, which are smooth in ``eta`` on each side of ``eta_b``, are
        interpolated to ``eta_levels`` with cubic splines.
        """
        from scipy.interpolate import CubicSpline
        xi = np.asarray(xi, dtype=float)
        eta = np.asarray(eta_levels, dtype=float)
        if coarse_step is None:
            scale = min(max(self.smoothing_m, 4.0 * self.dx_fine), self.L)
            coarse_step = scale / 8.0
        pieces = self._coarse_levels(coarse_step)
        res = self._integrate(xi, pieces)

        n = (xi.size, eta.size)
        X = np.repeat(xi[:, None], eta.size, axis=1)
        Z = np.repeat((eta + self.z_off)[None, :], xi.size, axis=0)
        h_xi, h_eta, h_eta_below = np.ones(n), np.ones(n), np.ones(n)
        tol = 1e-9 * max(self.eta_lab, 1.0)
        for ip, r in enumerate(res):
            lo, hi = r['eta'][0], r['eta'][-1]
            if ip == len(res) - 1:
                sel = (eta >= lo - tol) & (eta < hi)
            else:
                sel = (eta >= lo - tol) & (eta < hi - tol)
            if not np.any(sel):
                continue
            for name, arr in (('x', X), ('z', Z), ('h_xi', h_xi), ('h_eta', h_eta)):
                arr[:, sel] = CubicSpline(r['eta'], r[name], axis=1)(eta[sel])
            h_eta_below[:, sel] = h_eta[:, sel]
        if self.kind == 'strip':
            at_b = np.abs(eta - self.eta_b) <= tol
            # eta_b itself belongs to the upper piece; its lower-side h_eta:
            h_eta_below[:, at_b] = res[0]['h_eta'][:, -1:]
        if np.min(h_eta_below) <= 0 or np.min(h_eta) <= 0 or np.min(h_xi) <= 0:
            raise ValueError("terrain-following map is singular (Z_eta <= 0 or converging "
                             "trajectories): increase transition_m / smoothing_m or use "
                             "z_ref_m = max(terrain)")
        return TerrainMapMetrics(xi=xi, eta=eta, x=X, z=Z, h_xi=h_xi, h_eta=h_eta,
                                 h_eta_below=h_eta_below)
