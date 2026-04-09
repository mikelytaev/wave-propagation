"""
Robust rational interpolation / least-squares (Chebfun-style ``ratinterp``).

Pure-Python port of Chebfun's ``ratinterp.m`` based on:

    P. Gonnet, R. Pachon, and L. N. Trefethen,
    "Robust Rational Interpolation and Least-Squares",
    Electronic Transactions on Numerical Analysis, 38:146-167, 2011.

    R. Pachon, P. Gonnet, and J. van Deun,
    "Fast and Stable Rational Interpolation in Roots of Unity and
    Chebyshev Points", SIAM Journal on Numerical Analysis, 2012.

The original MATLAB implementation is part of Chebfun and is distributed
under the BSD-3-Clause license, Copyright (c) 2017 by The University of
Oxford and The Chebfun Developers.  See http://www.chebfun.org/.

Public API
----------
``ratinterp(f, m, n, NN=None, xi='type2', tol=1e-14, dom=(-1.0, 1.0))``
    Compute a robust rational interpolant or linear least-squares
    approximant.

``RationalInterpolant``
    Dataclass that holds the resulting numerator/denominator coefficients
    (in the appropriate basis), exposes ``__call__`` for evaluation, and
    provides :py:meth:`compute_poles_and_residues` for partial-fraction
    decomposition.

The four supported node families are

* ``'type1'``       — first-kind Chebyshev points on ``[-1, 1]``
* ``'type2'``       — second-kind Chebyshev points on ``[-1, 1]`` (default)
* ``'unitroots'``   — N-th roots of unity (synonym: ``'type0'``)
* ``'equi'``        — equispaced points on ``[-1, 1]``

An explicit array of arbitrary nodes (already mapped to ``[-1, 1]`` for the
real-axis cases, or to the unit circle for the rational-on-the-circle case)
may also be supplied.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.linalg


__all__ = ["ratinterp", "RationalInterpolant", "chebpts1", "chebpts2"]


# ---------------------------------------------------------------------------
# Chebyshev nodes and discrete transforms
# ---------------------------------------------------------------------------


def chebpts1(n: int) -> np.ndarray:
    """First-kind Chebyshev points on ``[-1, 1]`` in increasing order."""
    if n <= 0:
        return np.zeros(0)
    return -np.cos((2 * np.arange(n) + 1) * np.pi / (2 * n))


def chebpts2(n: int) -> np.ndarray:
    """Second-kind Chebyshev points on ``[-1, 1]`` in increasing order."""
    if n <= 0:
        return np.zeros(0)
    if n == 1:
        return np.zeros(1)
    return -np.cos(np.arange(n) * np.pi / (n - 1))


def _eval_T_matrix(x: np.ndarray, n: int) -> np.ndarray:
    """Build the matrix ``T[k, j] = T_j(x_k)`` for ``j = 0..n-1``."""
    m = x.shape[0]
    if n == 0:
        return np.zeros((m, 0), dtype=x.dtype)
    T = np.empty((m, n), dtype=x.dtype)
    T[:, 0] = 1.0
    if n == 1:
        return T
    T[:, 1] = x
    for j in range(2, n):
        T[:, j] = 2.0 * x * T[:, j - 1] - T[:, j - 2]
    return T


def _cheb1_coeffs2vals(c: np.ndarray) -> np.ndarray:
    """Evaluate a Chebyshev T-series at the first-kind Chebyshev points."""
    c = np.asarray(c)
    n = c.shape[0]
    if n <= 1:
        return c.copy()
    return _eval_T_matrix(chebpts1(n), n).astype(c.dtype, copy=False) @ c


def _cheb1_vals2coeffs(v: np.ndarray) -> np.ndarray:
    """Inverse Chebyshev transform on first-kind Chebyshev points.

    Works for 1D vectors and 2D matrices (transform applied along axis 0).
    """
    v = np.asarray(v)
    n = v.shape[0]
    if n <= 1:
        return v.copy()
    T = _eval_T_matrix(chebpts1(n), n).astype(v.dtype, copy=False)
    c = (T.T @ v) * (2.0 / n)
    c[0] = c[0] * 0.5
    return c


def _cheb2_coeffs2vals(c: np.ndarray) -> np.ndarray:
    """Evaluate a Chebyshev T-series at the second-kind Chebyshev points."""
    c = np.asarray(c)
    n = c.shape[0]
    if n <= 1:
        return c.copy()
    return _eval_T_matrix(chebpts2(n), n).astype(c.dtype, copy=False) @ c


def _cheb2_vals2coeffs(v: np.ndarray) -> np.ndarray:
    """Inverse Chebyshev transform on second-kind Chebyshev points.

    Works for 1D vectors and 2D matrices (transform applied along axis 0).
    """
    v = np.asarray(v)
    n = v.shape[0]
    if n <= 1:
        return v.copy()
    N = n - 1
    T = _eval_T_matrix(chebpts2(n), n).astype(v.dtype, copy=False)
    w = np.ones(n)
    w[0] = 0.5
    w[-1] = 0.5
    if v.ndim == 1:
        wv = w * v
    else:
        wv = w[:, None] * v
    c = (T.T @ wv) * (2.0 / N)
    c[0] = c[0] * 0.5
    c[-1] = c[-1] * 0.5
    return c


# ---------------------------------------------------------------------------
# Symmetry detection, matrix assembly, denominator/numerator coefficients
# ---------------------------------------------------------------------------


def _check_symmetries(
    f: np.ndarray, xi: np.ndarray, xi_type: str, ts: float
) -> Tuple[bool, bool]:
    """Detect even/odd symmetry in the sampled values (Chebfun-style)."""
    N1 = len(f)
    N = N1 - 1
    fEven = False
    fOdd = False

    if xi_type == "type0":
        # Roots of unity: only check when N is odd (so N1 is even) and
        # the test slices are non-empty (M >= 1).  Without this guard,
        # ``np.linalg.norm`` of an empty array returns 0 and we would
        # spuriously detect both fEven and fOdd.
        if N % 2 == 1 and N // 2 >= 1:
            M = N // 2
            fl = f[1 : M + 1]
            fr = f[N + 1 - M : N1]
            fEven = np.linalg.norm(fl - fr, np.inf) < ts
            fOdd = np.linalg.norm(fl + fr, np.inf) < ts
    elif xi_type in ("type1", "type2"):
        M = N1 // 2  # = ceil(N/2)
        fl = f[:M]
        fr = f[::-1][:M]
        fEven = np.linalg.norm(fl - fr, np.inf) < ts
        fOdd = np.linalg.norm(fl + fr, np.inf) < ts
    else:
        # Arbitrary nodes: only meaningful when nodes are real and
        # symmetric about 0.  Replicates the behaviour of Chebfun's
        # ``checkSymmetries`` for the "ARBITRARY" branch.
        if not np.iscomplexobj(xi) or np.max(np.abs(xi.imag)) < ts:
            x = xi.real if np.iscomplexobj(xi) else xi
            ord_idx = np.argsort(x)
            xs = x[ord_idx]
            M = N // 2
            xl = xs[: M + 1]
            xr = xs[::-1][: M + 1]
            if np.linalg.norm(xl + xr, np.inf) < ts:
                fs = f[ord_idx]
                M2 = N1 // 2
                fl = fs[:M2]
                fr = fs[::-1][:M2]
                fEven = np.linalg.norm(fl - fr, np.inf) < ts
                fOdd = np.linalg.norm(fl + fr, np.inf) < ts

    return fEven, fOdd


def _assemble_matrices(
    f: np.ndarray, n: int, xi: np.ndarray, xi_type: str, N1: int
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Build the ``(N1, n+1)`` coefficient matrix ``Z``.

    For the ``arbitrary`` node branch, the upper-triangular factor ``R``
    of the QR decomposition of the Chebyshev Vandermonde matrix is also
    returned (it is needed to convert the result back to the standard
    Chebyshev basis).
    """
    if xi_type == "type0":
        col = np.fft.fft(f) / N1
        row = np.conj(np.fft.fft(np.conj(f))) / N1
        col = col.copy()
        col[0] = row[0]
        Z = scipy.linalg.toeplitz(col, row[: n + 1])
        return Z, None

    if xi_type == "type1":
        D = _eval_T_matrix(chebpts1(N1), N1)
        if np.iscomplexobj(f):
            D = D.astype(complex)
        Df = D[:, : n + 1] * f[:, None]
        Z = _cheb1_vals2coeffs(Df)
        return Z, None

    if xi_type == "type2":
        D = _eval_T_matrix(chebpts2(N1), N1)
        if np.iscomplexobj(f):
            D = D.astype(complex)
        Df = D[:, : n + 1] * f[:, None]
        Z = _cheb2_vals2coeffs(Df)
        return Z, None

    # Arbitrary nodes: build the Chebyshev Vandermonde matrix and orthogonalise.
    dtype = complex if (np.iscomplexobj(xi) or np.iscomplexobj(f)) else float
    C = np.empty((N1, N1), dtype=dtype)
    C[:, 0] = 1.0
    if N1 >= 2:
        C[:, 1] = xi
    for k in range(2, N1):
        C[:, k] = 2.0 * xi * C[:, k - 1] - C[:, k - 2]
    Q, R = np.linalg.qr(C)
    Z = Q.conj().T @ (f[:, None] * Q[:, : n + 1])
    return Z, R


def _compute_denominator_coeffs(
    Z: np.ndarray,
    m: int,
    n: int,
    fEven: bool,
    fOdd: bool,
    N1: int,
    ts: float,
) -> Tuple[np.ndarray, int]:
    """Compute the denominator coefficient vector ``b`` and the (possibly
    reduced) effective denominator degree ``n``.
    """
    shift = bool(fEven) ^ ((m % 2) == 1)
    is_complex = np.iscomplexobj(Z)
    dtype = complex if is_complex else float

    if n > 0 and (not (fOdd or fEven) or n > 1):
        while True:
            if not (fOdd or fEven):
                Zsub = Z[m + 1 : N1, : n + 1]
                if Zsub.shape[0] == 0:
                    b = np.zeros(n + 1, dtype=dtype)
                    b[0] = 1.0
                    break
                # NOTE: full_matrices=True matches MATLAB's ``svd(A, 0)``
                # behaviour when ``A`` has more columns than rows: it
                # exposes the null-space rows of Vh that
                # full_matrices=False would discard.
                _, S, Vh = np.linalg.svd(Zsub, full_matrices=True)
                ns = min(n, len(S))
                b = np.conj(Vh[-1])
            else:
                start_row = m + 1 + (1 if shift else 0)
                Zsub = Z[start_row:N1:2, 0 : n + 1 : 2]
                if Zsub.shape[0] == 0 or Zsub.shape[1] == 0:
                    b = np.zeros(n + 1, dtype=dtype)
                    if fOdd and n + 1 > 1:
                        b[1] = 1.0
                    else:
                        b[0] = 1.0
                    break
                _, S, Vh = np.linalg.svd(Zsub, full_matrices=True)
                ns = min(n // 2, len(S))
                b = np.zeros(n + 1, dtype=dtype)
                b[0::2] = np.conj(Vh[-1])

            if ns == 0:
                # Nothing to test against — accept the trivial null vector.
                break

            ssv = S[ns - 1]
            if ssv > ts:
                break

            s = S[:ns]
            count = int(np.sum((s - ssv) <= ts))
            if fEven or fOdd:
                n = n - 2 * count
            else:
                n = n - count

            if n <= 0:
                b = np.array([1.0], dtype=dtype)
                n = 0
                break
            if n == 1:
                if fEven:
                    b = np.array([1.0, 0.0], dtype=dtype)
                    break
                if fOdd:
                    b = np.array([0.0, 1.0], dtype=dtype)
                    break
    elif n > 0:
        if fEven:
            b = np.array([1.0, 0.0], dtype=dtype)
        elif fOdd:
            b = np.array([0.0, 1.0], dtype=dtype)
        else:
            b = np.array([1.0], dtype=dtype)
    else:
        b = np.array([1.0], dtype=dtype)

    return b, n


def _compute_numerator_coeffs(
    f: np.ndarray,
    m: int,
    n: int,
    xi_type: str,
    Z: np.ndarray,
    b: np.ndarray,
    fEven: bool,
    fOdd: bool,
    N: int,
    N1: int,
) -> np.ndarray:
    """Compute the numerator coefficient vector ``a``."""
    is_complex = np.iscomplexobj(b) or np.iscomplexobj(f)

    if xi_type == "type0":
        b_pad = np.zeros(N1, dtype=complex)
        b_pad[: len(b)] = b
        a = np.fft.fft(np.fft.ifft(b_pad) * f)
        a = a[: m + 1]
    elif xi_type == "type1":
        b_pad = np.zeros(N1, dtype=complex if is_complex else float)
        b_pad[: len(b)] = b
        a = _cheb1_vals2coeffs(_cheb1_coeffs2vals(b_pad) * f)
        a = a[: m + 1]
    elif xi_type == "type2":
        b_pad = np.zeros(N1, dtype=complex if is_complex else float)
        b_pad[: len(b)] = b
        a = _cheb2_vals2coeffs(_cheb2_coeffs2vals(b_pad) * f)
        a = a[: m + 1]
    else:
        a = Z[: m + 1, : n + 1] @ b

    a = np.asarray(a).copy()
    if fEven:
        a[1::2] = 0
    elif fOdd:
        a[0::2] = 0
    return a


def _trim_coeffs(
    a: np.ndarray, b: np.ndarray, tol: float, ts: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Discard negligible leading and trailing coefficients."""
    a = np.asarray(a).copy()
    b = np.asarray(b).copy()

    if tol > 0:
        nna_idx = np.where(np.abs(a) > ts)[0]
        nnb_idx = np.where(np.abs(b) > tol)[0]
        a = a[: nna_idx[-1] + 1] if len(nna_idx) > 0 else a[:0]
        b = b[: nnb_idx[-1] + 1] if len(nnb_idx) > 0 else b[:0]
        while (
            len(a) > 0
            and len(b) > 0
            and abs(a[0]) < ts
            and abs(b[0]) < ts
        ):
            a = a[1:]
            b = b[1:]

    if len(a) == 0:
        a = np.array([0.0], dtype=a.dtype if a.dtype != object else float)
        b = np.array([1.0], dtype=b.dtype if b.dtype != object else float)
    return a, b


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


def _polyval_horner(coefs: np.ndarray, t):
    """Horner evaluation of ``sum_k coefs[k] * t**k`` (handles complex ``t``)."""
    coefs = np.asarray(coefs)
    t_arr = np.asarray(t)
    out = np.full(t_arr.shape, coefs[-1], dtype=np.result_type(coefs.dtype, t_arr.dtype))
    for c in coefs[-2::-1]:
        out = c + t_arr * out
    return out


def _polyder_monomial(coefs: np.ndarray) -> np.ndarray:
    """Derivative of a polynomial in monomial basis (length max(1, n))."""
    if len(coefs) <= 1:
        return np.zeros(1, dtype=coefs.dtype)
    k = np.arange(1, len(coefs))
    return coefs[1:] * k


@dataclass
class RationalInterpolant:
    """Result of :func:`ratinterp`.

    The interpolant is :math:`r(x) = p(x)/q(x)` where ``p`` and ``q`` are
    polynomials whose coefficients are stored in :attr:`p` and :attr:`q`.
    The basis depends on the sampling node family used by ``ratinterp``:

    * ``basis == 'chebyshev_t1'`` or ``'chebyshev_t2'``:
      ``p`` and ``q`` are stored as Chebyshev T-series coefficients on
      :attr:`domain`.  Calling the instance maps the input affinely from
      :attr:`domain` to :math:`[-1, 1]` and evaluates with Clenshaw's
      recurrence.

    * ``basis == 'monomial'``:
      ``p`` and ``q`` are stored as ordinary monomial coefficients of the
      polynomials in the variable ``z``.  This basis is used by the
      ``unitroots`` (``type0``) branch.  Inputs to ``__call__`` are
      forwarded directly to the polynomials, so the user can pass complex
      arguments living anywhere on or off the unit circle.

    Attributes
    ----------
    p, q : np.ndarray
        Numerator and denominator coefficients in the appropriate basis.
    mu, nu : int
        ``len(p) - 1`` and ``len(q) - 1``.
    basis : str
        ``'chebyshev_t1'``, ``'chebyshev_t2'``, or ``'monomial'``.
    domain : (float, float)
        Real interval on which the Chebyshev bases are defined.  Inputs
        to ``__call__`` are mapped affinely from this domain to
        ``[-1, 1]``.  Ignored for the ``'monomial'`` basis.
    poles, residues : np.ndarray, optional
        Populated by :py:meth:`compute_poles_and_residues`.
    """

    p: np.ndarray
    q: np.ndarray
    mu: int
    nu: int
    basis: str
    domain: Tuple[float, float]
    poles: Optional[np.ndarray] = None
    residues: Optional[np.ndarray] = None

    def __call__(self, x):
        x_arr = np.asarray(x)
        scalar = x_arr.ndim == 0
        if self.basis == "monomial":
            num = _polyval_horner(self.p, x_arr)
            den = _polyval_horner(self.q, x_arr)
        else:
            a, b = self.domain
            t = (2.0 * x_arr - (a + b)) / (b - a)
            num = np.polynomial.chebyshev.chebval(t, self.p)
            den = np.polynomial.chebyshev.chebval(t, self.q)
        out = num / den
        return out.item() if scalar else out

    def compute_poles_and_residues(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute poles and residues of the rational approximation.

        Returns
        -------
        poles : np.ndarray
            Locations of the poles in the user's domain (or in the
            polynomial's natural variable for the ``'monomial'`` basis).
        residues : np.ndarray
            Residues at simple poles.  For repeated/complex poles the
            simple-pole formula is still applied; consider it a heuristic
            for non-simple cases.
        """
        if self.basis == "monomial":
            poles = np.polynomial.polynomial.polyroots(self.q)
            num_at = _polyval_horner(self.p, poles)
            qprime = _polyder_monomial(self.q)
            den_at = _polyval_horner(qprime, poles)
            residues = num_at / den_at
            mapped_poles = poles
        else:
            cheb_poles = np.polynomial.chebyshev.Chebyshev(self.q).roots()
            num_at = np.polynomial.chebyshev.chebval(cheb_poles, self.p)
            qprime_coeffs = np.polynomial.chebyshev.chebder(self.q)
            den_at = np.polynomial.chebyshev.chebval(cheb_poles, qprime_coeffs)
            a, b = self.domain
            mapped_poles = 0.5 * (a + b) + 0.5 * (b - a) * cheb_poles
            # If r(x) ~ R / (x - x0) and t = (2x - (a+b))/(b-a),
            # then r(t) = R * (b-a)/2 / (t - t0), so the t-residue
            # equals R * (b-a)/2 and we recover R = 2/(b-a) * t-residue.
            residues_t = num_at / den_at
            residues = residues_t * 2.0 / (b - a)

        order = np.argsort(np.real(mapped_poles))
        self.poles = mapped_poles[order]
        self.residues = residues[order]
        return self.poles, self.residues


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


_NodeSpec = Union[str, np.ndarray, Sequence[float], Sequence[complex]]


def _resolve_nodes(
    xi: _NodeSpec, NN: Optional[int], m: int, n: int
) -> Tuple[str, np.ndarray, int]:
    """Determine the canonical type string and nodes from the user input."""
    if isinstance(xi, str):
        s = xi.strip().lower()
        if s in ("type0", "unitroots", "unit-roots", "rou"):
            xi_type = "type0"
            if NN is None:
                NN = m + n + 1
            xi_nodes = np.exp(2j * np.pi * np.arange(NN) / NN)
        elif s == "type1":
            xi_type = "type1"
            if NN is None:
                NN = m + n + 1
            xi_nodes = chebpts1(NN)
        elif s == "type2":
            xi_type = "type2"
            if NN is None:
                NN = m + n + 1
            xi_nodes = chebpts2(NN)
        elif s.startswith("equi"):
            xi_type = "arbitrary"
            if NN is None:
                NN = m + n + 1
            xi_nodes = np.linspace(-1.0, 1.0, NN)
        else:
            raise ValueError(f"Unrecognized xi type: {xi!r}")
    else:
        xi_nodes = np.asarray(xi)
        xi_type = "arbitrary"
        if NN is None:
            NN = len(xi_nodes)
        elif len(xi_nodes) != NN:
            raise ValueError(
                f"len(xi)={len(xi_nodes)} does not match NN={NN}."
            )
    return xi_type, xi_nodes, NN


def ratinterp(
    f,
    m: int,
    n: int,
    NN: Optional[int] = None,
    xi: _NodeSpec = "type2",
    tol: float = 1e-14,
    dom: Tuple[float, float] = (-1.0, 1.0),
) -> RationalInterpolant:
    """Robust rational interpolation or linear least-squares approximation.

    This is a pure-Python port of Chebfun's ``ratinterp``.  The function
    computes a numerator polynomial of degree at most ``m`` and a
    denominator polynomial of degree at most ``n`` such that
    :math:`r(x) = p(x)/q(x)` interpolates ``f`` at the chosen nodes (or
    is a least-squares approximant when ``NN > m + n + 1``).

    Robustness, in the sense of Gonnet–Pachón–Trefethen (2011),
    automatically reduces ``m`` and ``n`` whenever the SVD reveals
    rank-deficiency below the tolerance ``tol``.

    Parameters
    ----------
    f : callable or array_like
        Either a function returning ``f(x)`` for an array of nodes, or a
        column vector of length ``NN`` containing the sampled values.
    m : int
        Numerator degree.
    n : int
        Denominator degree.
    NN : int, optional
        Number of sampling nodes.  Defaults to ``m + n + 1``
        (interpolation).  Larger values trigger linear least-squares
        approximation.
    xi : str or array_like, optional
        Sampling node family or an explicit array of nodes.  Recognised
        strings are ``'type1'``, ``'type2'``, ``'unitroots'`` (alias
        ``'type0'``), and ``'equispaced'``.  Default: ``'type2'``.
    tol : float, optional
        Robustness tolerance.  Set to ``0`` to disable rank reduction.
        Default: ``1e-14``.
    dom : (float, float), optional
        Domain ``[a, b]`` of ``f``.  When ``f`` is a callable, it is
        sampled at ``a + (b - a) * (xi + 1) / 2`` for the real-axis
        cases.  Ignored when an explicit ``xi`` array is supplied.
        Default: ``(-1.0, 1.0)``.

    Returns
    -------
    RationalInterpolant
        Object containing the numerator/denominator coefficients in the
        appropriate basis along with helpers for evaluation and the
        partial-fraction decomposition.

    Examples
    --------
    Recover a degree-(0, 1) rational exactly:

    >>> import numpy as np
    >>> from pywaveprop.utils.ratinterp import ratinterp
    >>> r = ratinterp(lambda x: 1.0 / (x - 0.2), 0, 1)
    >>> abs(r(0.5) - 1.0 / (0.5 - 0.2)) < 1e-12
    True
    """
    if not isinstance(m, (int, np.integer)) or m < 0:
        raise ValueError("m must be a non-negative integer.")
    if not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError("n must be a non-negative integer.")
    if dom[1] <= dom[0]:
        raise ValueError("dom must be a strictly increasing pair (a, b).")

    xi_type, xi_nodes, NN = _resolve_nodes(xi, NN, m, n)

    if NN < m + n + 1:
        raise ValueError(
            f"NN={NN} must be at least m + n + 1 = {m + n + 1}."
        )

    # Sample f if it is a callable.
    if callable(f):
        a_d, b_d = dom
        if xi_type == "type0":
            # Evaluate directly on the unit circle.
            f_vals = np.asarray(f(xi_nodes))
        else:
            x_phys = 0.5 * (a_d + b_d) + 0.5 * (b_d - a_d) * xi_nodes
            f_vals = np.asarray(f(x_phys))
    else:
        f_vals = np.asarray(f)
        if len(f_vals) != NN:
            raise ValueError(
                f"len(f)={len(f_vals)} does not match NN={NN}."
            )

    f_vals = np.atleast_1d(f_vals)

    N = NN - 1
    N1 = NN
    ts = float(tol) * float(np.linalg.norm(f_vals, np.inf))

    fEven, fOdd = _check_symmetries(f_vals, xi_nodes, xi_type, ts)
    Z, R = _assemble_matrices(f_vals, n, xi_nodes, xi_type, N1)
    b, n_eff = _compute_denominator_coeffs(Z, m, n, fEven, fOdd, N1, ts)
    a = _compute_numerator_coeffs(
        f_vals, m, n_eff, xi_type, Z, b, fEven, fOdd, N, N1
    )
    a, b = _trim_coeffs(a, b, tol, ts)

    mu = len(a) - 1
    nu = len(b) - 1

    if xi_type == "type0":
        return RationalInterpolant(
            p=a, q=b, mu=mu, nu=nu, basis="monomial", domain=tuple(dom)
        )
    if xi_type == "type1":
        return RationalInterpolant(
            p=a, q=b, mu=mu, nu=nu, basis="chebyshev_t1", domain=tuple(dom)
        )
    if xi_type == "type2":
        return RationalInterpolant(
            p=a, q=b, mu=mu, nu=nu, basis="chebyshev_t2", domain=tuple(dom)
        )

    # Arbitrary nodes: convert from the QR-orthogonal basis to standard
    # Chebyshev T-coefficients via the leading principal submatrix of R.
    R_p = R[: mu + 1, : mu + 1]
    R_q = R[: nu + 1, : nu + 1]
    p_cheb = scipy.linalg.solve_triangular(R_p, a, lower=False)
    q_cheb = scipy.linalg.solve_triangular(R_q, b, lower=False)
    return RationalInterpolant(
        p=p_cheb,
        q=q_cheb,
        mu=mu,
        nu=nu,
        basis="chebyshev_t2",
        domain=tuple(dom),
    )
