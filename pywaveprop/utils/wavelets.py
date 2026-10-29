"""
Orthogonal wavelet filter banks (Daubechies ``dbN`` and least-asymmetric
symlets ``symN``) computed from Daubechies' spectral factorisation, so that no
external wavelet package is required.

The scaling filter ``h`` (length ``2N``) satisfies the double-shift
orthogonality ``sum_m h[m] h[m + 2k] = delta_k`` and ``sum_m h[m] = sqrt(2)``;
the wavelet filter is ``g[m] = (-1)^m h[2N - 1 - m]``. For ``symN`` the roots
of the spectral factor are selected to minimise the phase non-linearity,
which reproduces the standard (PyWavelets) symlet coefficients up to the
usual reversal / sign conventions.
"""
import itertools
from functools import lru_cache
from math import comb

import numpy as np


def _daubechies_roots(p):
    """Roots (in z) of the Daubechies polynomial ``z^(p-1) P(y(z))`` with
    ``P(y) = sum_{k<p} C(p-1+k, k) y^k`` and ``y = -(z - 1)^2 / (4 z)``."""
    poly = np.zeros(2 * p - 1)
    for k in range(p):
        term = comb(p - 1 + k, k) * (-0.25) ** k * np.polynomial.polynomial.polypow([1.0, -2.0, 1.0], k)
        term = np.convolve(term, np.array([0.0] * (p - 1 - k) + [1.0]))
        poly[:len(term)] += term
    return np.roots(poly[::-1])


def _filter_from_roots(p, roots):
    q = np.array([1.0 + 0j])
    for r in roots:
        q = np.convolve(q, [1.0, -r])
    h = np.convolve(q.real, np.polynomial.polynomial.polypow([1.0, 1.0], p))
    h = h / np.sqrt(np.sum(h ** 2))
    if h.sum() < 0:
        h = -h
    return h


@lru_cache(maxsize=None)
def scaling_filter(name: str) -> np.ndarray:
    """Return the orthonormal scaling (low-pass) filter of ``'dbN'`` or ``'symN'``."""
    kind = name[:-1] if name[:-1] in ("db", "sym") else name.rstrip("0123456789")
    p = int(name[len(kind):])
    if kind not in ("db", "sym") or p < 1:
        raise ValueError(f"unsupported wavelet {name!r}; use 'dbN' or 'symN'")
    if p == 1:
        return np.array([1.0, 1.0]) / np.sqrt(2.0)
    roots = _daubechies_roots(p)
    inside = [r for r in roots if abs(r) < 1]
    groups, used = [], set()
    for i, r in enumerate(inside):
        if i in used:
            continue
        if abs(r.imag) < 1e-10:
            groups.append([r])
            used.add(i)
        else:
            j = next(k for k in range(len(inside)) if k not in used and k != i
                     and abs(inside[k] - np.conj(r)) < 1e-8)
            groups.append([r, inside[j]])
            used.update((i, j))
    if kind == "db":
        return _filter_from_roots(p, [r for g in groups for r in g])
    # symlet: least-asymmetric selection among the 2^(#groups) root choices
    w = np.linspace(0.01, np.pi - 0.01, 400)
    best = None
    for flags in itertools.product((0, 1), repeat=len(groups)):
        sel = [r if f == 0 else 1.0 / r for g, f in zip(groups, flags) for r in g]
        h = _filter_from_roots(p, sel)
        spectrum = np.polyval(h[::-1], np.exp(-1j * w))
        phase = np.unwrap(np.angle(spectrum))
        score = float(np.max(np.abs(phase - np.polyval(np.polyfit(w, phase, 1), w))))
        if best is None or score < best[0]:
            best = (score, h)
    return best[1]


def wavelet_filter(h: np.ndarray) -> np.ndarray:
    """Quadrature-mirror wavelet (high-pass) filter of scaling filter ``h``."""
    m = np.arange(len(h))
    return (-1.0) ** m * h[::-1]


def filter_bank(name: str):
    h = scaling_filter(name)
    return h, wavelet_filter(h)
