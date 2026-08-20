"""
Multipath extraction by Prony's method.

Based on:

    H. Zhou and A. Chabory, "An Extraction Method for the Multipath
    Characteristics of Simulated Tropospheric Propagation Channels",
    IEEE Transactions on Aerospace and Electronic Systems, 2025.

The frequency-domain channel is modelled as a sum of complex exponentials

    u(f) = sum_p  a_p * exp(-j*2*pi*f*tau_p),                    (eq. 4)

where ``a_p`` is the complex amplitude and ``tau_p`` is the delay of the
``p``-th propagation path.  Given samples ``u_n = u(f_start + n*delta_f)``
for ``n = 0, ..., nf-1``, the method recovers ``a_p`` and ``tau_p`` for
``n_paths`` paths via three steps:

1. Solve the homogeneous-difference equation (eq. 11) for the FIR
   coefficients ``h_1, ..., h_{n_paths}`` (least squares if
   ``nf > 2 * n_paths``).
2. Take the roots ``z_p`` of the characteristic polynomial (eq. 10);
   ``tau_p = -angle(z_p) / (2*pi*delta_f)`` (eq. 12).
3. Solve the Vandermonde least-squares system for the amplitudes
   (eq. 8).

Optional post-processing (subsection III-D):

* drop paths whose amplitude falls below ``amp_threshold``;
* merge paths whose delays differ by less than ``delay_cluster``
  using the amplitude-weighted formulas in eq. (16).

Two entry points are provided:

* :func:`prony_extract` operates on a single 1-D vector of samples and
  returns a :class:`MultipathResult`.
* :func:`prony_extract_field` runs Prony's method in parallel over an
  arbitrary spatial grid.  It accepts an array of shape
  ``(nf, *spatial)`` and returns ``(amplitudes, delays)`` of shape
  ``(n_paths, *spatial)``.

Phase ambiguity (eq. 14) requires ``delta_f <= 1 / tau_max``; the caller
is responsible for choosing ``delta_f`` accordingly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


__all__ = [
    "MultipathResult",
    "prony_extract",
    "prony_extract_field",
]


@dataclass
class MultipathResult:
    """Result of :func:`prony_extract`.

    Amplitudes are stored in the form of eq. (5): they reference the
    starting frequency ``f_start``, so

        u(f_start + n*delta_f) = sum_p amplitudes[p] * exp(-j*2*pi*n*delta_f*delays[p]).

    In particular ``sum(amplitudes) == u(f_start)``.  Reconstruction at
    an arbitrary absolute frequency ``f`` uses

        u(f) = sum_p amplitudes[p] * exp(-j*2*pi*(f - f_start)*delays[p]).

    Working at frequencies relative to ``f_start`` keeps clustering of
    close-delay paths well-conditioned: the merged amplitude exactly
    reproduces ``u`` at ``f_start`` and is a small-``delta_f * delta_tau``
    approximation at neighbouring sample frequencies.

    Paths are sorted by increasing delay.
    """

    amplitudes: np.ndarray
    delays: np.ndarray
    f_start: float

    @property
    def n_paths(self) -> int:
        return int(self.delays.shape[0])

    def reconstruct(self, freq) -> np.ndarray:
        """Reconstruct the channel at one or more absolute frequencies."""
        f = np.asarray(freq, dtype=float)
        phase = -2j * np.pi * (f[..., None] - self.f_start) * self.delays
        return np.sum(self.amplitudes * np.exp(phase), axis=-1)

    def rms_error_db(self, u_samples, delta_f: float) -> float:
        """RMS error (in dB) of the reconstruction at the sample grid.

        Implements eq. (20).  ``u_samples`` is the original ``nf`` length
        vector of complex samples spaced by ``delta_f``.
        """
        u_samples = np.asarray(u_samples, dtype=complex)
        nf = u_samples.shape[0]
        freqs = self.f_start + delta_f * np.arange(nf)
        u_hat = self.reconstruct(freqs)
        num = float(np.sum(np.abs(u_samples - u_hat) ** 2))
        den = float(np.sum(np.abs(u_samples) ** 2))
        if den == 0.0:
            return float("-inf")
        return 10.0 * np.log10(num / den)


def prony_extract(
    u,
    *,
    f_start: float,
    delta_f: float,
    n_paths: int,
    amp_threshold: Optional[float] = None,
    delay_cluster: Optional[float] = None,
) -> MultipathResult:
    """Extract multipath components from frequency samples by Prony's method.

    Parameters
    ----------
    u : array_like
        Complex samples ``u_n = u(f_start + n*delta_f)`` for
        ``n = 0, ..., nf-1``.  Requires ``len(u) >= 2 * n_paths``.
    f_start : float
        Reference frequency ``f_0`` (Hz).  Used to express amplitudes in
        the absolute-frequency form of eq. (4).
    delta_f : float
        Frequency increment between samples (Hz).
    n_paths : int
        Number of propagation paths to extract.
    amp_threshold : float, optional
        Discard paths with ``|a_p| <= amp_threshold`` (eq. 15).
    delay_cluster : float, optional
        Merge paths whose delays differ by less than this value (in
        seconds) using eq. (16).  Common choice: ``1.0 / c0``, so that
        path lengths within one metre are merged.

    Returns
    -------
    MultipathResult
        Recovered amplitudes and delays.
    """
    u = np.asarray(u, dtype=complex)
    if u.ndim != 1:
        raise ValueError("u must be 1-D; use prony_extract_field for arrays")
    nf = u.shape[0]
    if n_paths <= 0:
        raise ValueError("n_paths must be a positive integer")
    if nf < 2 * n_paths:
        raise ValueError(
            f"need at least 2*n_paths = {2 * n_paths} samples, got nf={nf}"
        )
    if delta_f <= 0:
        raise ValueError("delta_f must be positive")

    A_p, tau = _prony_core(u[None, :], n_paths=n_paths, delta_f=delta_f)
    A_p = A_p[0]
    tau = tau[0]

    order = np.argsort(tau)
    tau = tau[order]
    A_p = A_p[order]

    if amp_threshold is not None and amp_threshold > 0:
        keep = np.abs(A_p) > amp_threshold
        A_p = A_p[keep]
        tau = tau[keep]

    if delay_cluster is not None and delay_cluster > 0 and tau.size > 1:
        tau, A_p = _cluster_paths(tau, A_p, delay_cluster)

    return MultipathResult(amplitudes=A_p, delays=tau, f_start=float(f_start))


def prony_extract_field(
    u_freq,
    *,
    f_start: float,
    delta_f: float,
    n_paths: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised Prony extraction over a spatial grid.

    Parameters
    ----------
    u_freq : array_like
        Complex array of shape ``(nf, *spatial)`` containing the
        propagated field at ``nf`` frequencies ``f_start + n*delta_f``.
    f_start, delta_f, n_paths
        Same meaning as in :func:`prony_extract`.

    Returns
    -------
    amplitudes : np.ndarray, complex
        Shape ``(n_paths, *spatial)``.  Amplitudes in the eq.-(5)
        reference-frequency form; see :class:`MultipathResult`.
    delays : np.ndarray, float
        Shape ``(n_paths, *spatial)``.  Sorted in increasing order
        along the first axis at every spatial point.

    Notes
    -----
    Post-processing (thresholding / clustering) is not applied here
    because the number of surviving paths would vary across the grid.
    Apply it per-point on the returned arrays if needed.
    """
    u_freq = np.asarray(u_freq, dtype=complex)
    if u_freq.ndim < 1:
        raise ValueError("u_freq must have at least one axis (frequency)")
    nf = u_freq.shape[0]
    spatial_shape = u_freq.shape[1:]
    if n_paths <= 0:
        raise ValueError("n_paths must be a positive integer")
    if nf < 2 * n_paths:
        raise ValueError(
            f"need at least 2*n_paths = {2 * n_paths} samples, got nf={nf}"
        )
    if delta_f <= 0:
        raise ValueError("delta_f must be positive")

    n_points = int(np.prod(spatial_shape)) if spatial_shape else 1
    u_flat = u_freq.reshape(nf, n_points).T  # (n_points, nf)

    A_p, tau = _prony_core(u_flat, n_paths=n_paths, delta_f=delta_f)

    # ``f_start`` is recorded by the caller but not needed in the
    # reference-frequency-form amplitudes returned here.
    _ = f_start

    order = np.argsort(tau, axis=1)
    tau = np.take_along_axis(tau, order, axis=1)
    A_p = np.take_along_axis(A_p, order, axis=1)

    amp_out = A_p.T.reshape((n_paths,) + spatial_shape)
    tau_out = tau.T.reshape((n_paths,) + spatial_shape)
    return amp_out, tau_out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _prony_core(
    u_batched: np.ndarray, *, n_paths: int, delta_f: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Batched Prony fit.

    Parameters
    ----------
    u_batched : (N, nf) complex array
        Each row is an independent set of samples.

    Returns
    -------
    a : (N, n_paths) complex array
        Amplitudes such that ``u_n = sum_p a[p] * z_p**n``.
    tau : (N, n_paths) real array
        Delays in seconds.
    """
    n_points, nf = u_batched.shape
    rows = nf - n_paths

    # Build the Hankel-like prediction matrix (eq. 11) per point.
    # H[i, r, k] = u_batched[i, n_paths - 1 - k + r],  r = 0..rows-1,
    #                                                   k = 0..n_paths-1.
    H = np.empty((n_points, rows, n_paths), dtype=complex)
    for k in range(n_paths):
        H[:, :, k] = u_batched[:, n_paths - 1 - k : nf - 1 - k]
    rhs = u_batched[:, n_paths:nf]  # (n_points, rows)

    # Solve the per-point least-squares system via the Moore-Penrose
    # pseudo-inverse (batched SVD).  Slightly more expensive than the
    # normal equations but unconditionally robust to rank deficiency,
    # which occurs naturally at points where the field is essentially
    # zero (e.g., on a PEC ground) or where two paths nearly collide.
    # The ``rcond`` cutoff suppresses singular values below ``1e-10``
    # times the largest, which prevents the pinv from amplifying noise
    # into amplitudes many orders of magnitude larger than the input.
    rcond = 1e-10
    with np.errstate(divide="ignore", invalid="ignore"):
        H_pinv = np.linalg.pinv(H, rcond=rcond)  # (n_points, n_paths, rows)
    h = (H_pinv @ rhs[..., None])[..., 0]  # (n_points, n_paths)
    # At points where the input field is exactly zero, pinv produces NaNs;
    # treat those as "no signal" by zeroing the linear-prediction
    # coefficients (yields zero roots and therefore zero delays).
    h = np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)

    # Roots of the characteristic polynomial via the companion matrix
    # P(z) = z^{n_p} - h_1 z^{n_p-1} - ... - h_{n_p}.
    if n_paths == 1:
        z = h.astype(complex)  # (n_points, 1) — single root is h_1.
    else:
        comp = np.zeros((n_points, n_paths, n_paths), dtype=complex)
        # Subdiagonal of ones (n_paths-1 of them).
        idx = np.arange(n_paths - 1)
        comp[:, idx + 1, idx] = 1.0
        # Last column = h reversed: [h_{n_p}, h_{n_p-1}, ..., h_1].
        comp[:, :, -1] = h[:, ::-1]
        z = np.linalg.eigvals(comp)  # (n_points, n_paths)

    tau = -np.angle(z) / (2.0 * np.pi * delta_f)

    # Vandermonde least squares (eq. 8): u = M a per point.
    n_idx = np.arange(nf)
    M = z[:, None, :] ** n_idx[None, :, None]  # (n_points, nf, n_paths)
    # Vandermonde least-squares for the amplitudes (eq. 8), batched via
    # the same pseudo-inverse approach.  Same rcond cutoff prevents
    # explosive amplitudes when two recovered roots collide.
    with np.errstate(divide="ignore", invalid="ignore"):
        M_pinv = np.linalg.pinv(M, rcond=rcond)  # (n_points, n_paths, nf)
    a = (M_pinv @ u_batched[..., None])[..., 0]  # (n_points, n_paths)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

    return a, tau


def _cluster_paths(
    tau: np.ndarray, amp: np.ndarray, delay_cluster: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Single-linkage merge of consecutive close-delay paths (eq. 16).

    ``tau`` is assumed to be sorted in increasing order.
    """
    tau_list = list(tau)
    amp_list = list(amp)
    i = 0
    while i < len(tau_list) - 1:
        if tau_list[i + 1] - tau_list[i] < delay_cluster:
            a0, a1 = amp_list[i], amp_list[i + 1]
            w0 = abs(a0) ** 2
            w1 = abs(a1) ** 2
            wsum = w0 + w1
            if wsum == 0.0:
                merged_tau = 0.5 * (tau_list[i] + tau_list[i + 1])
            else:
                merged_tau = (w0 * tau_list[i] + w1 * tau_list[i + 1]) / wsum
            amp_list[i] = a0 + a1
            tau_list[i] = merged_tau
            del tau_list[i + 1]
            del amp_list[i + 1]
            # Recheck against new neighbour — do not advance i.
        else:
            i += 1
    return np.asarray(tau_list, dtype=float), np.asarray(amp_list, dtype=complex)
