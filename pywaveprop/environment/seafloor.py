"""Seafloor (bathymetry) loader backed by the elevation tile cache.

The public ``elevation-tiles-prod`` bucket that powers
:mod:`pywaveprop.environment.terrain` is a Mapzen/Terrarium-style blend of
SRTM (dry land) and GEBCO (bathymetry), so we can reuse the exact same tile
cache and simply treat *negative* elevations as seafloor depths. A dedicated
GEBCO netCDF loader is also supported for higher-resolution offline workflows.

All data written by this module lands under
``~/.cache/pywaveprop/bathymetry`` (profile caches) and re-uses the elevation
tile cache under ``~/.cache/pywaveprop/elevation``.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import numpy as np

from . import _paths
from .models import BathymetryProfile, GeoTerrain
from .terrain import ElevationTileCache, DEFAULT_RESOLUTION, load_terrain


def _profile_cache_key(lat1: float, lon1: float,
                       lat2: float, lon2: float,
                       n_points: int, resolution: float) -> str:
    tag = (f"{lat1:.5f}_{lon1:.5f}__{lat2:.5f}_{lon2:.5f}"
           f"__n{n_points}_r{resolution:.5f}")
    return hashlib.md5(tag.encode()).hexdigest()[:16]


class SeafloorProvider:
    """Bathymetry profile builder with transparent disk caching.

    The first call for a given path stores the resulting profile under
    ``~/.cache/pywaveprop/bathymetry/profile_<hash>.npz`` so subsequent
    calls return instantly without touching the tile store.
    """

    def __init__(self,
                 resolution: float = DEFAULT_RESOLUTION,
                 cache_dir: str | Path | None = None,
                 elevation_cache: ElevationTileCache | None = None):
        self.resolution = resolution
        self.cache_dir = Path(cache_dir) if cache_dir else _paths.BATHYMETRY_CACHE_DIR
        self.elevation_cache = (elevation_cache if elevation_cache is not None
                                else ElevationTileCache(resolution=resolution))

    def load_area(self, bbox: tuple[float, float, float, float]) -> GeoTerrain:
        """Return a ``GeoTerrain`` over the area, with bathymetry preserved."""
        return self.elevation_cache.load(bbox)

    def profile(self, lat1: float, lon1: float,
                lat2: float, lon2: float,
                n_points: int = 300,
                default_depth_m: float = 100.0,
                ) -> BathymetryProfile:
        """Seafloor depth (positive, metres) sampled along a path.

        Grid points with non-negative elevation are treated as shoreline and
        assigned ``default_depth_m`` so downstream acoustic solvers always
        receive a water column. If the whole path is dry this raises
        ``RuntimeError``.
        """
        cache_path = self.cache_dir / (
            f"profile_{_profile_cache_key(lat1, lon1, lat2, lon2, n_points, self.resolution)}.npz"
        )
        if cache_path.exists():
            with np.load(cache_path) as d:
                return BathymetryProfile(
                    ranges_m=d["ranges_m"].copy(),
                    depths_m=d["depths_m"].copy(),
                    source=str(d["source"]) if "source" in d else "tile-cache",
                )

        lon_min, lon_max = min(lon1, lon2), max(lon1, lon2)
        lat_min, lat_max = min(lat1, lat2), max(lat1, lat2)
        margin = max(5 * self.resolution, 0.01)
        bbox = (lon_min - margin, lat_min - margin,
                lon_max + margin, lat_max + margin)
        terrain = self.elevation_cache.load(bbox)

        ranges_m, raw = terrain.profile(lon1, lat1, lon2, lat2,
                                        n_points=n_points,
                                        mask_underwater=False)
        depths = np.where(raw < 0.0, -raw, default_depth_m)
        if np.all(depths <= 0.0):
            raise RuntimeError(
                "no seafloor along path: entire profile is at or above sea level")

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, ranges_m=ranges_m,
                            depths_m=depths, source="tile-cache")
        return BathymetryProfile(ranges_m=ranges_m, depths_m=depths,
                                 source="tile-cache")


def load_bathymetry_profile(lat1: float, lon1: float,
                            lat2: float, lon2: float,
                            n_points: int = 300,
                            resolution: float = DEFAULT_RESOLUTION,
                            default_depth_m: float = 100.0,
                            ) -> BathymetryProfile:
    return SeafloorProvider(resolution=resolution).profile(
        lat1, lon1, lat2, lon2, n_points=n_points,
        default_depth_m=default_depth_m)


def load_bathymetry_area(bbox: tuple[float, float, float, float],
                         resolution: float = DEFAULT_RESOLUTION
                         ) -> GeoTerrain:
    return load_terrain(bbox, resolution=resolution)
