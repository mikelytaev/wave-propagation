"""Lightweight geospatial environment data structures.

These are intentionally decoupled from the JAX-based propagation models in
``pywaveprop.rwp_jax`` and ``pywaveprop.uwa_jax``. They are plain dataclasses
that describe the environment data pulled from external sources (SRTM,
WorldCover, GEBCO, Argo) so the same objects can be re-used by the
tropospheric and underwater propagation factories.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


@dataclass
class GeoTerrain:
    """Regular-grid terrain over a WGS84 bounding box.

    Elevation values are in metres above sea level. For tiles that also contain
    bathymetry, negative values (below sea level) are preserved here; callers
    that only need land elevation can use ``elevation_masked`` which clamps
    below-sea-level pixels to zero.
    """
    lons: np.ndarray
    lats: np.ndarray
    grid: np.ndarray
    has_bathymetry: bool = False

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (float(self.lons[0]), float(self.lats[0]),
                float(self.lons[-1]), float(self.lats[-1]))

    @property
    def elevation_masked(self) -> np.ndarray:
        """Copy of ``grid`` with underwater cells clamped to 0 m.

        Propagation above sea level treats the ocean surface as z=0; the
        actual depth is handled separately by the bathymetry loader.
        """
        if not self.has_bathymetry:
            return self.grid
        return np.maximum(self.grid, 0.0)

    def elevation_at(self, lon: float, lat: float,
                     mask_underwater: bool = True) -> float:
        grid = self.elevation_masked if mask_underwater else self.grid
        return float(_bilinear(self.lons, self.lats, grid, lon, lat))

    def profile(self, lon1: float, lat1: float,
                lon2: float, lat2: float,
                n_points: int = 300,
                mask_underwater: bool = True
                ) -> tuple[np.ndarray, np.ndarray]:
        """Extract a terrain profile between two points.

        Returns ``(distances_m, elevations_m)`` with the first sample at 0 m
        and the last at the great-circle distance between the endpoints.
        """
        grid = self.elevation_masked if mask_underwater else self.grid
        lons = np.linspace(lon1, lon2, n_points)
        lats = np.linspace(lat1, lat2, n_points)
        z = _bilinear_vec(self.lons, self.lats, grid, lons, lats)
        dist = _haversine_m(lat1, lon1, lat2, lon2)
        return np.linspace(0.0, dist, n_points), z


@dataclass
class LandCoverClassification:
    """ESA WorldCover class code and derived ground material parameters."""
    class_id: int
    eps_dielect: float
    sgm_conductivity: float


@dataclass
class BathymetryProfile:
    """Seafloor depth sampled along a range axis (depth positive, in metres)."""
    ranges_m: np.ndarray
    depths_m: np.ndarray
    source: str = "gebco"

    @property
    def max_depth_m(self) -> float:
        return float(np.max(self.depths_m))


@dataclass
class RefractivityTransect:
    """Range-dependent modified refractivity ``M(x, z)`` along a great-circle path.

    Produced by :func:`pywaveprop.environment.nwp.refractivity_transect`.
    ``M_profile()`` returns the callable that
    :attr:`pywaveprop.rwp.environment.Troposphere.M_profile` expects, so a
    transect can be handed straight to the range-dependent PE solver.
    """
    x_m: np.ndarray            # (nx,) range along the path [m]
    z_m: np.ndarray            # (nz,) height grid [m]
    M: np.ndarray              # (nx, nz) modified refractivity [M-units]
    lat_pts: np.ndarray        # (nx,) sample latitudes [deg]
    lon_pts: np.ndarray        # (nx,) sample longitudes [deg]
    lsm_pts: np.ndarray        # (nx,) land fraction (1 = land)
    edh_pts: np.ndarray        # (nx,) evaporation duct height [m]
    duct_base: np.ndarray      # (nx,) trapping-layer base [m], NaN if none
    duct_top: np.ndarray       # (nx,) trapping-layer top [m], NaN if none
    duct_strength: np.ndarray  # (nx,) duct strength [M-units], NaN if none
    p1: Optional[np.ndarray] = None
    p2: Optional[np.ndarray] = None
    name: str = ""
    cycle: str = ""

    @property
    def length_m(self) -> float:
        return float(self.x_m[-1])

    def M_profile(self) -> Callable:
        """Range-dependent ``f(x, z)`` callable for the PE solvers."""
        from .refractivity import range_dependent_M_profile
        return range_dependent_M_profile(self.x_m, self.z_m, self.M)

    def uniform_M_profile(self, column: int = 0) -> Callable:
        """Range-INDEPENDENT ``f(x, z)`` from one column (the classical control)."""
        from .refractivity import uniform_M_profile
        return uniform_M_profile(self.z_m, self.M[column])

    def as_dict(self) -> dict:
        """Plain-array view, suitable for :func:`numpy.savez_compressed`."""
        return {
            "x_m": self.x_m, "z_m": self.z_m, "M": self.M,
            "lat_pts": self.lat_pts, "lon_pts": self.lon_pts,
            "lsm_pts": self.lsm_pts, "edh_pts": self.edh_pts,
            "duct_base": self.duct_base, "duct_top": self.duct_top,
            "duct_strength": self.duct_strength,
            "p1": np.asarray(self.p1 if self.p1 is not None else []),
            "p2": np.asarray(self.p2 if self.p2 is not None else []),
            "name": self.name, "cycle": self.cycle,
        }


@dataclass
class SoundVelocityProfile:
    """Vertical sound-speed profile derived from Argo or other CTD data."""
    depths_m: np.ndarray
    speeds_m_s: np.ndarray
    source: str = "argopy"
    lat: Optional[float] = None
    lon: Optional[float] = None
    date_range: Optional[tuple[str, str]] = None

    @property
    def max_depth_m(self) -> float:
        return float(np.max(self.depths_m))


def _haversine_m(lat1: float, lon1: float,
                 lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _bilinear(lons: np.ndarray, lats: np.ndarray, grid: np.ndarray,
              lon: float, lat: float) -> float:
    n_lon = len(lons)
    n_lat = len(lats)
    if n_lon < 2 or n_lat < 2:
        return float(grid.flat[0])
    dlon = lons[1] - lons[0]
    dlat = lats[1] - lats[0]
    c = (lon - float(lons[0])) / dlon
    r = (lat - float(lats[0])) / dlat
    ci = int(max(0, min(n_lon - 2, int(c))))
    ri = int(max(0, min(n_lat - 2, int(r))))
    dc = c - ci
    dr = r - ri
    return (grid[ri, ci] * (1 - dr) * (1 - dc) +
            grid[ri, ci + 1] * (1 - dr) * dc +
            grid[ri + 1, ci] * dr * (1 - dc) +
            grid[ri + 1, ci + 1] * dr * dc)


def _bilinear_vec(lons: np.ndarray, lats: np.ndarray, grid: np.ndarray,
                  lon_arr: np.ndarray, lat_arr: np.ndarray) -> np.ndarray:
    n_lon = len(lons)
    n_lat = len(lats)
    if n_lon < 2 or n_lat < 2:
        return np.full_like(lon_arr, float(grid.flat[0]), dtype=float)
    dlon = lons[1] - lons[0]
    dlat = lats[1] - lats[0]
    c = (lon_arr - float(lons[0])) / dlon
    r = (lat_arr - float(lats[0])) / dlat
    ci = np.clip(c.astype(int), 0, n_lon - 2)
    ri = np.clip(r.astype(int), 0, n_lat - 2)
    dc = c - ci
    dr = r - ri
    return (grid[ri, ci] * (1 - dr) * (1 - dc) +
            grid[ri, ci + 1] * (1 - dr) * dc +
            grid[ri + 1, ci] * dr * (1 - dc) +
            grid[ri + 1, ci + 1] * dr * dc)
