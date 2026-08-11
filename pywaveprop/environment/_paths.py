"""Default cache directory layout for environment datasets.

All downloaded data (elevation GeoTIFFs, tile arrays, ESA WorldCover land cover,
bathymetry, Argo profiles, GFS GRIB subsets) is cached under a single root so
users can share a single cache across projects and purge it in one place.
"""
from __future__ import annotations

import os
from pathlib import Path


def _default_cache_root() -> Path:
    env = os.environ.get("PYWAVEPROP_CACHE_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "pywaveprop"


CACHE_ROOT = _default_cache_root()
ELEV_CACHE_DIR = CACHE_ROOT / "elevation"
LANDCOVER_CACHE_DIR = CACHE_ROOT / "landcover"
BATHYMETRY_CACHE_DIR = CACHE_ROOT / "bathymetry"
ARGO_CACHE_DIR = CACHE_ROOT / "argo"
GFS_CACHE_DIR = CACHE_ROOT / "gfs"


def ensure_cache_dirs() -> None:
    for d in (ELEV_CACHE_DIR, LANDCOVER_CACHE_DIR,
              BATHYMETRY_CACHE_DIR, ARGO_CACHE_DIR, GFS_CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def set_cache_root(root: str | Path) -> None:
    """Override the cache root for the current process."""
    global CACHE_ROOT, ELEV_CACHE_DIR, LANDCOVER_CACHE_DIR
    global BATHYMETRY_CACHE_DIR, ARGO_CACHE_DIR, GFS_CACHE_DIR
    CACHE_ROOT = Path(root).expanduser()
    ELEV_CACHE_DIR = CACHE_ROOT / "elevation"
    LANDCOVER_CACHE_DIR = CACHE_ROOT / "landcover"
    BATHYMETRY_CACHE_DIR = CACHE_ROOT / "bathymetry"
    ARGO_CACHE_DIR = CACHE_ROOT / "argo"
    GFS_CACHE_DIR = CACHE_ROOT / "gfs"
