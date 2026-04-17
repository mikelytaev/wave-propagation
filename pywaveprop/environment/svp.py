"""Sound velocity profile retrieval via the Argo float network.

Uses :mod:`argopy` to fetch temperature and practical salinity at a point of
interest and computes the sound-speed profile with the Chen-Millero (1977)
equation, which is the same formulation used by
:func:`pywaveprop.uwa.environment.sound_speed_mps` but evaluated on the full
pressure-dependent form.

The raw float dataset is written to a NetCDF cache file on first retrieval so
subsequent calls work fully offline:

    ~/.cache/pywaveprop/argo/argo_<lat>_<lon>_<radius>_<start>_<end>.nc

A parallel NumPy cache stores the derived ``(depth, c)`` profile, which is
much smaller than the raw float traces:

    ~/.cache/pywaveprop/argo/ssp_<hash>.npz
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import numpy as np

from . import _paths
from .models import SoundVelocityProfile


def _chen_millero_sound_speed(t: np.ndarray,
                              s: np.ndarray,
                              p_bar: np.ndarray) -> np.ndarray:
    """Chen & Millero (1977) sound speed in sea water.

    Parameters
    ----------
    t : ndarray
        Temperature in degrees Celsius (ITS-90).
    s : ndarray
        Practical salinity (PSU).
    p_bar : ndarray
        Pressure in bar (1 bar ≈ 10 dbar ≈ 1 m depth at the surface).
    """
    t = np.asarray(t, dtype=float)
    s = np.asarray(s, dtype=float)
    p = np.asarray(p_bar, dtype=float)

    c00, c01, c02, c03, c04, c05 = (1402.388, 5.03830, -5.81090e-2,
                                    3.3432e-4, -1.47797e-6, 3.1419e-9)
    c10, c11, c12, c13, c14 = (0.153563, 6.8999e-4, -8.1829e-6,
                               1.3632e-7, -6.1260e-10)
    c20, c21, c22, c23, c24 = (3.1260e-5, -1.7111e-6, 2.5986e-8,
                               -2.5353e-10, 1.0415e-12)
    c30, c31, c32 = (-9.7729e-9, 3.8513e-10, -2.3654e-12)

    cw = (c00 + c01 * t + c02 * t**2 + c03 * t**3 + c04 * t**4 + c05 * t**5
          + (c10 + c11 * t + c12 * t**2 + c13 * t**3 + c14 * t**4) * p
          + (c20 + c21 * t + c22 * t**2 + c23 * t**3 + c24 * t**4) * p**2
          + (c30 + c31 * t + c32 * t**2) * p**3)

    a00, a01, a02, a03, a04 = (1.389, -1.262e-2, 7.166e-5, 2.008e-6, -3.21e-8)
    a10, a11, a12, a13, a14 = (9.4742e-5, -1.2583e-5, -6.4928e-8,
                               1.0515e-8, -2.0142e-10)
    a20, a21, a22, a23 = (-3.9064e-7, 9.1061e-9, -1.6009e-10, 7.994e-12)
    a30, a31, a32 = (1.100e-10, 6.651e-12, -3.391e-13)

    a = (a00 + a01 * t + a02 * t**2 + a03 * t**3 + a04 * t**4
         + (a10 + a11 * t + a12 * t**2 + a13 * t**3 + a14 * t**4) * p
         + (a20 + a21 * t + a22 * t**2 + a23 * t**3) * p**2
         + (a30 + a31 * t + a32 * t**2) * p**3)

    b00, b01 = (-1.922e-2, -4.42e-5)
    b10, b11 = (7.3637e-5, 1.7945e-7)
    b = b00 + b01 * t + (b10 + b11 * t) * p

    d00, d10 = (1.727e-3, -7.9836e-6)
    d = d00 + d10 * p

    return cw + a * s + b * s**1.5 + d * s**2


def _depth_from_pressure(p_dbar: np.ndarray, lat: float) -> np.ndarray:
    """UNESCO 1983 depth from pressure (dbar) and latitude (degrees)."""
    p = np.asarray(p_dbar, dtype=float)
    phi = np.radians(lat)
    g = (9.780318 * (1.0 + 5.2788e-3 * np.sin(phi)**2
                     + 2.36e-5 * np.sin(phi)**4)
         + 1.092e-6 * p)
    return (9.72659e2 * p - 2.2512e-1 * p**2
            + 2.279e-4 * p**3 - 1.82e-7 * p**4) / g


def _profile_cache_key(lat: float, lon: float, radius_deg: float,
                       start: str, end: str, max_depth_m: float) -> str:
    tag = f"{lat:.4f}_{lon:.4f}_r{radius_deg:.3f}_{start}_{end}_d{max_depth_m:.0f}"
    return hashlib.md5(tag.encode()).hexdigest()[:16]


def load_argo_svp(lat: float, lon: float, *,
                  radius_deg: float = 1.0,
                  start_date: str = "2020-01-01",
                  end_date: str = "2023-12-31",
                  max_depth_m: float = 2000.0,
                  n_depth_bins: int = 80,
                  cache_dir: str | Path | None = None,
                  ) -> SoundVelocityProfile:
    """Fetch Argo data around ``(lat, lon)`` and return a mean SVP.

    The raw :class:`xarray.Dataset` returned by :mod:`argopy` is serialised to
    NetCDF under the cache directory on first retrieval and read back from
    disk on subsequent calls.
    """
    cache_dir = Path(cache_dir) if cache_dir else _paths.ARGO_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_path = cache_dir / (
        f"argo_{lat:.4f}_{lon:.4f}_r{radius_deg:.3f}_"
        f"{start_date}_{end_date}.nc"
    )
    profile_path = cache_dir / (
        f"ssp_{_profile_cache_key(lat, lon, radius_deg, start_date, end_date, max_depth_m)}.npz"
    )

    if profile_path.exists():
        with np.load(profile_path) as d:
            return SoundVelocityProfile(
                depths_m=d["depths_m"].copy(),
                speeds_m_s=d["speeds_m_s"].copy(),
                source="argopy",
                lat=float(d["lat"]), lon=float(d["lon"]),
                date_range=(start_date, end_date),
            )

    try:
        import xarray as xr
    except ImportError as e:
        raise ImportError("xarray is required for SVP loading") from e

    if raw_path.exists():
        ds = xr.open_dataset(raw_path)
    else:
        try:
            from argopy import DataFetcher
        except ImportError as e:
            raise ImportError(
                "argopy is required to download Argo data. "
                "Install with `pip install argopy`.") from e
        box = [lon - radius_deg, lon + radius_deg,
               lat - radius_deg, lat + radius_deg,
               0.0, max_depth_m, start_date, end_date]
        fetcher = DataFetcher().region(box)
        ds = fetcher.to_xarray()
        ds.to_netcdf(raw_path)

    temp = np.asarray(ds["TEMP"].values).ravel()
    sal = np.asarray(ds["PSAL"].values).ravel()
    pres = np.asarray(ds["PRES"].values).ravel()

    mask = (np.isfinite(temp) & np.isfinite(sal) & np.isfinite(pres)
            & (pres >= 0.0) & (pres <= max_depth_m * 1.1))
    temp, sal, pres = temp[mask], sal[mask], pres[mask]
    if temp.size == 0:
        raise RuntimeError(
            f"no valid Argo measurements in box around ({lat}, {lon}) "
            f"during {start_date}..{end_date}")

    depth = _depth_from_pressure(pres, lat=lat)
    speed = _chen_millero_sound_speed(temp, sal, pres / 10.0)

    depth_grid = np.linspace(0.0, max_depth_m, n_depth_bins)
    bins = np.digitize(depth, depth_grid)
    mean_speed = np.full(depth_grid.shape, np.nan, dtype=float)
    for i in range(1, len(depth_grid)):
        sel = bins == i
        if np.any(sel):
            mean_speed[i] = float(np.mean(speed[sel]))

    if np.isnan(mean_speed[0]):
        first = np.nanargmax(~np.isnan(mean_speed))
        mean_speed[0] = mean_speed[first]
    not_nan = ~np.isnan(mean_speed)
    if not np.all(not_nan):
        mean_speed = np.interp(depth_grid, depth_grid[not_nan],
                               mean_speed[not_nan])

    np.savez_compressed(profile_path, depths_m=depth_grid,
                        speeds_m_s=mean_speed, lat=lat, lon=lon)

    return SoundVelocityProfile(
        depths_m=depth_grid, speeds_m_s=mean_speed,
        source="argopy", lat=lat, lon=lon,
        date_range=(start_date, end_date),
    )
