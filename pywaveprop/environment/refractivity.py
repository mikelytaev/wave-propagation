"""Atmospheric refractivity physics and ducting diagnostics.

Converts numerical-weather-prediction thermodynamic fields (pressure,
temperature, humidity) into radio refractivity ``N`` and modified refractivity
``M``, the quantity the tropospheric parabolic-equation solvers in
:mod:`pywaveprop.rwp` and :mod:`pywaveprop.rwp_jax` propagate through.

The module is deliberately dependency-light (NumPy only) so it can be used on
its own, without the GFS download layer in
:mod:`pywaveprop.environment.gfs` or the ``xarray``-based pipeline in
:mod:`pywaveprop.environment.nwp`.

References
----------
* ITU-R P.453 "The radio refractive index: its formula and refractivity data".
* Smith & Weintraub (1953), the two-term refractivity formula.
* Buck (1996) saturation vapour pressure.
"""
from __future__ import annotations

import numpy as np

#: Mean Earth radius [m] used for the Earth-flattening term in ``M``.
EARTH_RADIUS_M = 6_371_000.0

#: dM/dh contribution of Earth curvature, = 1e6 / EARTH_RADIUS_M  [N-units per m].
M_CURVATURE_PER_M = 1.0e6 / EARTH_RADIUS_M  # ~0.157


def saturation_vapour_pressure(temperature_k: np.ndarray) -> np.ndarray:
    """Saturation vapour pressure over water [hPa] (Buck 1996 / ITU-R P.453).

    Parameters
    ----------
    temperature_k : array
        Temperature [K].
    """
    t_c = np.asarray(temperature_k, dtype=float) - 273.15
    # Buck coefficients over water (ITU-R P.453 form), result in hPa.
    return 6.1121 * np.exp((18.678 - t_c / 234.5) * (t_c / (257.14 + t_c)))


def vapour_pressure_from_rh(
    temperature_k: np.ndarray, relative_humidity_pct: np.ndarray
) -> np.ndarray:
    """Water-vapour partial pressure ``e`` [hPa] from RH [%] and T [K]."""
    es = saturation_vapour_pressure(temperature_k)
    return np.asarray(relative_humidity_pct, dtype=float) / 100.0 * es


def vapour_pressure_from_q(
    specific_humidity: np.ndarray, pressure_hpa: np.ndarray
) -> np.ndarray:
    """Water-vapour partial pressure ``e`` [hPa] from specific humidity [kg/kg]."""
    q = np.asarray(specific_humidity, dtype=float)
    p = np.asarray(pressure_hpa, dtype=float)
    # e = q*P / (eps + (1-eps)*q), eps = Rd/Rv = 0.622
    eps = 0.622
    return q * p / (eps + (1.0 - eps) * q)


def refractivity_n(
    pressure_hpa: np.ndarray,
    temperature_k: np.ndarray,
    vapour_pressure_hpa: np.ndarray,
) -> np.ndarray:
    """Radio refractivity ``N`` (Smith-Weintraub two-term form).

    ``N = 77.6 * P/T + 3.73e5 * e / T**2``

    Parameters
    ----------
    pressure_hpa : array
        Total air pressure [hPa].
    temperature_k : array
        Temperature [K].
    vapour_pressure_hpa : array
        Water-vapour partial pressure ``e`` [hPa].
    """
    p = np.asarray(pressure_hpa, dtype=float)
    t = np.asarray(temperature_k, dtype=float)
    e = np.asarray(vapour_pressure_hpa, dtype=float)
    return 77.6 * p / t + 3.73e5 * e / (t * t)


def modified_refractivity_m(
    refractivity_n: np.ndarray, height_m: np.ndarray
) -> np.ndarray:
    """Modified refractivity ``M = N + 1e6 * h / a`` (~ N + 0.157*h).

    A trapping / ducting layer is where ``dM/dh < 0``.

    Parameters
    ----------
    refractivity_n : array
        Refractivity ``N`` [N-units].
    height_m : array
        Geometric height above sea level [m].
    """
    return np.asarray(refractivity_n, dtype=float) + M_CURVATURE_PER_M * np.asarray(
        height_m, dtype=float
    )


def profile_from_rh(
    pressure_hpa: np.ndarray,
    temperature_k: np.ndarray,
    relative_humidity_pct: np.ndarray,
    height_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience: compute ``(N, M)`` from RH-based NWP fields.

    All inputs are broadcast together. Returns ``(N, M)`` with the same shape.
    """
    e = vapour_pressure_from_rh(temperature_k, relative_humidity_pct)
    n = refractivity_n(pressure_hpa, temperature_k, e)
    m = modified_refractivity_m(n, height_m)
    return n, m


def duct_diagnostics(profile_M: np.ndarray, height_m: np.ndarray) -> dict:
    """Basic ducting diagnostics for a single ``M(h)`` profile.

    Returns dict with trapping-layer presence, base/top height and duct strength
    (max ``M`` above a local minimum minus that minimum). ``dM/dh < 0`` marks a
    trapping layer.
    """
    m = np.asarray(profile_M, dtype=float)
    h = np.asarray(height_m, dtype=float)
    good = np.isfinite(m) & np.isfinite(h)
    m, h = m[good], h[good]
    if m.size < 3:
        return {"has_duct": False}

    dM_dh = np.gradient(m, h)
    trapping = dM_dh < 0.0
    if not trapping.any():
        return {"has_duct": False, "min_gradient": float(dM_dh.min())}

    idx = np.where(trapping)[0]
    base_i, top_i = idx[0], idx[-1]
    return {
        "has_duct": True,
        "base_height_m": float(h[base_i]),
        "top_height_m": float(h[top_i + 1]) if top_i + 1 < h.size else float(h[top_i]),
        "strength_M": float(m[base_i] - m[min(top_i + 1, m.size - 1)]),
        "min_gradient": float(dM_dh.min()),
    }


def duct_diagnostics_field(M: np.ndarray, height_m: np.ndarray) -> dict:
    """Apply :func:`duct_diagnostics` to every column of an ``M(z, lat, lon)`` cube.

    Returns a dict of ``(nlat, nlon)`` arrays: ``strength``, ``base``, ``top``,
    ``min_gradient`` (NaN where undefined) and the boolean ``has_duct``.
    """
    M = np.asarray(M, dtype=float)
    height_m = np.asarray(height_m, dtype=float)
    if M.ndim != 3 or M.shape[0] != height_m.size:
        raise ValueError(f"M shape {M.shape} does not match {height_m.size} heights")
    _, nlat, nlon = M.shape
    strength = np.full((nlat, nlon), np.nan)
    base = np.full((nlat, nlon), np.nan)
    top = np.full((nlat, nlon), np.nan)
    mingrad = np.full((nlat, nlon), np.nan)
    has = np.zeros((nlat, nlon), bool)
    for j in range(nlat):
        for i in range(nlon):
            d = duct_diagnostics(M[:, j, i], height_m)
            if d.get("min_gradient") is not None:
                mingrad[j, i] = d["min_gradient"]
            if d.get("has_duct"):
                has[j, i] = True
                strength[j, i] = d["strength_M"]
                base[j, i] = d["base_height_m"]
                top[j, i] = d["top_height_m"]
    return {"strength": strength, "base": base, "top": top,
            "min_gradient": mingrad, "has_duct": has}


def horizontal_gradient(field2d: np.ndarray, lat: np.ndarray,
                        lon: np.ndarray) -> np.ndarray:
    """Magnitude of the horizontal gradient of a lat/lon field, per km.

    Used to locate where refraction changes fast enough along the path that a
    range-independent (horizontally homogeneous) assumption breaks down.
    """
    field2d = np.asarray(field2d, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    dlat_km = 111.32
    dlon_km = 111.32 * np.cos(np.deg2rad(lat))  # per degree lon at each latitude
    gy, gx = np.gradient(field2d)  # per grid-step in (lat, lon)
    gy_km = gy / (np.gradient(lat)[:, None] * dlat_km)
    gx_km = gx / (np.gradient(lon)[None, :] * dlon_km[:, None])
    return np.hypot(gx_km, gy_km)


def range_dependent_M_profile(x_m, z_m, M, normalize_top=True):
    """Wrap a sampled ``M(x, z)`` field into the callable the PE solvers expect.

    :attr:`pywaveprop.rwp.environment.Troposphere.M_profile` must be a callable
    ``f(x, z)`` where ``x`` is a scalar range [m] and ``z`` is either a scalar
    or a 1-D array of heights [m]. It MUST preserve the shape of ``z`` (return
    a scalar for scalar ``z``), because the upper non-local boundary condition
    probes the profile at single points; NumPy fancy-indexing breaks that, but
    :func:`numpy.interp` preserves it.

    Ranges outside the column set clamp to the nearest column; heights outside
    the level set clamp to the profile ends, so the top of the profile stays
    linear as the non-local boundary condition requires.

    Parameters
    ----------
    x_m : (nx,) array
        Range of each column [m], increasing.
    z_m : (nz,) array
        Height grid [m], increasing.
    M : (nx, nz) array
        Modified refractivity [M-units].
    normalize_top : bool
        Shift every column by a constant so that ``M(x, z_m[-1])`` equals the
        launch-column value for all ``x``. The transparent (non-local) upper
        boundary condition of the split-step Pade solver is built ONCE, from the
        launch column: ``rwp.sspade`` evaluates the refractive index and its
        vertical gradient at ``x = 0, z = z_max`` and applies that operator at
        every range step. With an NWP-derived field, ``M(x, z_max)`` typically
        drifts by tens of M-units along the path, the operator then no longer
        matches the medium at the boundary, and the boundary reflects: on a
        573 km Gulf of Oman transect (M(x, z_max) spread 29 M-units) the field
        below 1000 m departed from a tall-domain reference by 9.7 dB RMS
        (90th percentile 13.5 dB), against 3.2 dB RMS (90th percentile 1.2 dB)
        after normalization. The shift is physically inert for transmission
        loss: it leaves every vertical gradient — and therefore every duct —
        untouched, and a height-independent offset of ``n^2 - 1`` only adds a
        range-dependent phase, which ``|u|`` does not see.
    """
    x_m = np.asarray(x_m, dtype=float)
    z_m = np.asarray(z_m, dtype=float)
    M = np.asarray(M, dtype=float)
    if M.shape != (x_m.size, z_m.size):
        raise ValueError(f"M shape {M.shape} != ({x_m.size},{z_m.size})")
    if normalize_top and x_m.size > 1:
        M = M - (M[:, -1:] - M[0, -1])

    def f(x, z):
        # locate x between two columns
        xi = np.interp(x, x_m, np.arange(x_m.size))
        i0 = int(np.clip(np.floor(xi), 0, x_m.size - 1))
        i1 = min(i0 + 1, x_m.size - 1)
        w = xi - i0
        col = (1.0 - w) * M[i0] + w * M[i1]
        return np.interp(z, z_m, col)  # preserves scalar/array shape

    return f


def uniform_M_profile(z_m, M_col):
    """Range-INDEPENDENT ``f(x, z)`` built from a single ``M(z)`` column.

    This is the classical horizontally-homogeneous assumption, and the natural
    control run against :func:`range_dependent_M_profile`.
    """
    z_m = np.asarray(z_m, dtype=float)
    M_col = np.asarray(M_col, dtype=float)

    def f(x, z):
        return np.interp(z, z_m, M_col)

    return f
