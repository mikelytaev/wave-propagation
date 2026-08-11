"""Surface-layer refractivity: evaporation ducts over sea, thermal over land.

Evaporation ducts occupy the lowest few tens of metres over the sea and are not
resolved by NWP levels. They arise because air right at the sea surface is
saturated, and humidity falls off sharply with height -- driving the wet term of
refractivity down fast enough that modified refractivity ``M`` has a minimum at
the *evaporation duct height* (EDH), the top of the trapping layer.

We reconstruct the near-surface T and q profiles from bulk fields (SST, 2 m air
T and RH, 10 m wind, surface pressure) using Monin-Obukhov similarity theory:
COARE-style roughness lengths, Businger-Dyer unstable and Beljaars-Holtslag
stable stability functions. Then M(z) is formed and its minimum gives EDH.

Monin-Obukhov similarity describes the *surface layer* only, so the profiles are
valid up to a few tens of metres; ``z_max`` defaults to 40 m for that reason.
Above it, splice into the NWP profile (see
:func:`pywaveprop.environment.nwp.refractivity_transect`).

References
----------
* Paulus (1985); Babin, Young & Carton (1997) -- evaporation-duct models.
* Fairall et al. (2003) COARE 3.0 -- bulk flux algorithm / roughness.
* Beljaars & Holtslag (1991) -- stable-stratification profile functions.
"""
from __future__ import annotations

import numpy as np

from .refractivity import M_CURVATURE_PER_M, refractivity_n

KAPPA = 0.4          # von Karman
G = 9.81             # m/s^2
CP = 1004.67         # J/kg/K
RD = 287.05          # J/kg/K
NU = 1.5e-5          # kinematic viscosity of air [m^2/s]
CHARNOCK = 0.011

#: Surface-layer ceiling [m]: MOST profiles are not valid above this.
DEFAULT_Z_MAX = 40.0


def _qsat(temp_k, pressure_hpa):
    """Saturation specific humidity [kg/kg] over water (Buck es)."""
    t_c = temp_k - 273.15
    es = 6.1121 * np.exp((18.678 - t_c / 234.5) * (t_c / (257.14 + t_c)))
    return 0.622 * es / (pressure_hpa - 0.378 * es)


_BH = (1.0, 0.667, 5.0, 0.35)  # Beljaars-Holtslag stable coefficients


def _psi_m(zeta):
    """Momentum stability function (vectorised; Businger-Dyer / Beljaars-Holtslag)."""
    zeta = np.clip(np.asarray(zeta, dtype=float), -50.0, 50.0)
    zn = np.minimum(zeta, 0.0)
    x = (1.0 - 15.0 * zn) ** 0.25
    unst = (2.0 * np.log((1.0 + x) / 2.0) + np.log((1.0 + x * x) / 2.0)
            - 2.0 * np.arctan(x) + np.pi / 2.0)
    a, b, c, d = _BH
    zp = np.maximum(zeta, 0.0)
    st = -(a * zp + b * (zp - c / d) * np.exp(-d * zp) + b * c / d)
    return np.where(zeta < 0.0, unst, st)


def _psi_h(zeta):
    """Heat/moisture stability function (vectorised)."""
    zeta = np.clip(np.asarray(zeta, dtype=float), -50.0, 50.0)
    zn = np.minimum(zeta, 0.0)
    x = (1.0 - 15.0 * zn) ** 0.25
    unst = 2.0 * np.log((1.0 + x * x) / 2.0)
    a, b, c, d = _BH
    zp = np.maximum(zeta, 0.0)
    st = -((1.0 + 2.0 / 3.0 * a * zp) ** 1.5
           + b * (zp - c / d) * np.exp(-d * zp) + b * c / d - 1.0)
    return np.where(zeta < 0.0, unst, st)


def evaporation_duct(
    sst_k: float,
    air_temp_k: float,
    rh_pct: float,
    wind10_ms: float,
    pressure_hpa: float,
    z_air: float = 2.0,
    z_wind: float = 10.0,
    z_max: float = DEFAULT_Z_MAX,
    n_iter: int = 25,
    n_levels: int = 400,
):
    """Compute the evaporation-duct height for one set of bulk conditions.

    Parameters
    ----------
    sst_k : float
        Sea-surface (skin) temperature [K].
    air_temp_k : float
        Air temperature at ``z_air`` [K].
    rh_pct : float
        Relative humidity at ``z_air`` [%].
    wind10_ms : float
        Wind speed at ``z_wind`` [m/s].
    pressure_hpa : float
        Surface pressure [hPa].

    Returns
    -------
    dict with keys:
        edh_m       -- evaporation duct height [m] (0 if no trapping)
        deficit_M   -- M(surface) - M(edh) [M-units], the trapping depth
        z, M        -- the reconstructed height grid and M(z) profile
    """
    if not np.isfinite([sst_k, air_temp_k, rh_pct, wind10_ms, pressure_hpa]).all():
        return {"edh_m": np.nan, "deficit_M": np.nan, "z": None, "M": None}

    P = float(pressure_hpa)
    U = max(float(wind10_ms), 0.5)

    # surface (saturated, reduced 2% for salinity) and air humidity
    q_s = 0.98 * _qsat(sst_k, P)
    q_a = min(rh_pct / 100.0 * _qsat(air_temp_k, P), _qsat(air_temp_k, P))

    # potential temperatures referenced to 1000 hPa
    exner = (P / 1000.0) ** (RD / CP)
    th_s = sst_k / exner
    th_a = air_temp_k / exner  # 2 m ~ same pressure over this depth
    dth = th_a - th_s          # air minus sea
    dq = q_a - q_s
    th_v = th_a * (1.0 + 0.61 * q_a)

    # --- iterate MOST -------------------------------------------------------
    z0 = 1.0e-4
    u_star = KAPPA * U / np.log(z_wind / z0)
    z0t = 1.0e-4
    t_star = KAPPA * dth / np.log(z_air / z0t)
    q_star = KAPPA * dq / np.log(z_air / z0t)
    L = 1.0e5

    for _ in range(n_iter):
        tv_star = t_star * (1.0 + 0.61 * q_a) + 0.61 * th_a * q_star
        L = th_v * u_star * u_star / (KAPPA * G * tv_star)
        if not np.isfinite(L) or abs(L) < 1e-3:
            L = 1e5 if (tv_star >= 0) else -1e5

        z0 = CHARNOCK * u_star * u_star / G + 0.11 * NU / u_star
        Rr = z0 * u_star / NU
        z0t = min(1.1e-4, 5.5e-5 * Rr ** (-0.6))
        z0q = z0t

        u_star = KAPPA * U / (np.log(z_wind / z0) - _psi_m(z_wind / L) + _psi_m(z0 / L))
        t_star = KAPPA * dth / (np.log(z_air / z0t) - _psi_h(z_air / L) + _psi_h(z0t / L))
        q_star = KAPPA * dq / (np.log(z_air / z0q) - _psi_h(z_air / L) + _psi_h(z0q / L))
        u_star = max(u_star, 1e-4)

    # --- reconstruct M(z) ---------------------------------------------------
    z = np.linspace(0.1, z_max, n_levels)
    psih_z = _psi_h(z / L)
    th_z = th_s + t_star / KAPPA * (np.log(z / z0t) - psih_z + _psi_h(z0t / L))
    q_z = q_s + q_star / KAPPA * (np.log(z / z0q) - psih_z + _psi_h(z0q / L))

    # actual temperature and vapour pressure at height z
    Pz = P * np.exp(-G * z / (RD * air_temp_k))
    T_z = th_z * (Pz / 1000.0) ** (RD / CP)
    e_z = q_z * Pz / (0.622 + 0.378 * q_z)
    N_z = refractivity_n(Pz, T_z, e_z)
    M_z = N_z + M_CURVATURE_PER_M * z

    imin = int(np.argmin(M_z))
    if imin == 0 or imin == len(z) - 1:
        # monotone rising -> no trapping; or min pinned at top -> duct deeper
        edh = 0.0 if imin == 0 else float(z[imin])
    else:
        edh = float(z[imin])
    deficit = float(M_z[0] - M_z[imin])
    return {"edh_m": edh, "deficit_M": max(deficit, 0.0), "z": z, "M": M_z}


def evaporation_duct_field(
    sst_k: np.ndarray,
    air_temp_k: np.ndarray,
    rh_pct: np.ndarray,
    wind10_ms: np.ndarray,
    pressure_hpa: np.ndarray,
    land_fraction: np.ndarray | None = None,
    z_max: float = DEFAULT_Z_MAX,
    n_levels: int = 150,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaporation-duct height and M-deficit over a grid of bulk fields.

    All inputs are 2-D arrays of the same shape. Columns where
    ``land_fraction >= 0.5`` are left as NaN: the sea-surface saturation
    assumption behind :func:`evaporation_duct` does not hold over land (use
    :func:`land_surface_profile` there).

    Returns ``(edh_m, deficit_M)``.
    """
    sst_k = np.asarray(sst_k, dtype=float)
    shape = sst_k.shape
    air_temp_k = np.asarray(air_temp_k, dtype=float)
    rh_pct = np.asarray(rh_pct, dtype=float)
    wind10_ms = np.asarray(wind10_ms, dtype=float)
    pressure_hpa = np.asarray(pressure_hpa, dtype=float)
    lsm = (np.zeros(shape) if land_fraction is None
           else np.asarray(land_fraction, dtype=float))

    edh = np.full(shape, np.nan)
    deficit = np.full(shape, np.nan)
    for idx in np.ndindex(shape):
        if lsm[idx] >= 0.5:
            continue
        r = evaporation_duct(float(sst_k[idx]), float(air_temp_k[idx]),
                             float(rh_pct[idx]), float(wind10_ms[idx]),
                             float(pressure_hpa[idx]),
                             z_max=z_max, n_levels=n_levels)
        edh[idx] = r["edh_m"]
        deficit[idx] = r["deficit_M"]
    return edh, deficit


def land_surface_profile(
    skin_temp_k: float,
    air_temp_k: float,
    rh_pct: float,
    wind10_ms: float,
    pressure_hpa: float,
    z0: float = 0.03,
    z_air: float = 2.0,
    z_wind: float = 10.0,
    z_max: float = 100.0,
    n_iter: int = 25,
    n_levels: int = 250,
):
    """Reconstruct the near-surface N(z), M(z) over LAND from bulk fields.

    Over land the surface is not saturated, so we drive the profile with the
    *thermal* contrast only: the skin-to-2 m air temperature difference sets the
    Monin-Obukhov temperature profile, while specific humidity is held constant
    with height (the over-land anomaly here is thermal). Momentum roughness
    ``z0`` defaults to 0.03 m (short vegetation / desert); scalar roughness is
    ``z0/10``.

    A super-adiabatic layer (hot skin, unstable) makes the dry term of N rise
    with height -> sub-refraction. A surface inversion (cool skin, stable) does
    the opposite -> super-refraction / trapping.

    Returns dict with ``z`` [m], ``N`` [N-units], ``M`` [M-units].
    """
    P = float(pressure_hpa)
    U = max(float(wind10_ms), 0.5)
    z0t = z0 / 10.0

    exner = (P / 1000.0) ** (RD / CP)
    th_s = skin_temp_k / exner
    th_a = air_temp_k / exner
    dth = th_a - th_s  # + = stable (inversion), - = unstable (super-adiabatic)

    q = min(rh_pct / 100.0 * _qsat(air_temp_k, P), _qsat(air_temp_k, P))
    th_v = th_a * (1.0 + 0.61 * q)

    u_star = KAPPA * U / np.log(z_wind / z0)
    t_star = KAPPA * dth / np.log(z_air / z0t)
    L = 1.0e5
    for _ in range(n_iter):
        tv_star = t_star  # dry: neglect moisture flux over land
        L = th_v * u_star * u_star / (KAPPA * G * tv_star)
        if not np.isfinite(L) or abs(L) < 1e-3:
            L = 1e5 if tv_star >= 0 else -1e5
        u_star = KAPPA * U / (np.log(z_wind / z0) - _psi_m(z_wind / L) + _psi_m(z0 / L))
        t_star = KAPPA * dth / (np.log(z_air / z0t) - _psi_h(z_air / L) + _psi_h(z0t / L))
        u_star = max(float(u_star), 1e-4)

    z = np.linspace(0.1, z_max, n_levels)
    psih_z = _psi_h(z / L)
    th_z = th_s + t_star / KAPPA * (np.log(z / z0t) - psih_z + _psi_h(z0t / L))
    Pz = P * np.exp(-G * z / (RD * air_temp_k))
    T_z = th_z * (Pz / 1000.0) ** (RD / CP)
    e_z = q * Pz / (0.622 + 0.378 * q)
    N_z = refractivity_n(Pz, T_z, e_z)
    M_z = N_z + M_CURVATURE_PER_M * z
    return {"z": z, "N": N_z, "M": M_z}
