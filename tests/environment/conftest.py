"""Synthetic fixture data for the environment package.

These fixtures are generated once per test session and written to a temp
cache directory in the exact on-disk format that
``pywaveprop.environment`` uses, so the loaders run their real code paths
against them without ever reaching the network.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from pywaveprop.environment import _paths as paths
from pywaveprop.environment.terrain import (DEFAULT_RESOLUTION, TILE_SIZE_DEG,
                                            _tile_cache_path, _tiles_for_bbox)


# A small bbox centred on (40.25, 10.25). We use it for both the terrain and
# the bathymetry fixture; coords below sea level appear in the southern half
# so the "has_bathymetry" flag is exercised.
FIXTURE_BBOX = (10.1, 40.1, 10.4, 40.4)
FIXTURE_RES = 0.01
FIXTURE_SVP_LAT = 40.15
FIXTURE_SVP_LON = 10.25


def _make_synthetic_elevation(lon_min: float, lat_min: float,
                              lon_max: float, lat_max: float,
                              res: float,
                              with_bathymetry: bool = False
                              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lons = np.arange(lon_min, lon_max + res / 2, res)
    lats = np.arange(lat_min, lat_max + res / 2, res)
    lon_mesh, lat_mesh = np.meshgrid(lons, lats)
    # Smooth hill + a deep trench in the southern half when bathymetry=True
    hill = 400.0 * np.exp(
        -(((lon_mesh - 10.25) / 0.08) ** 2
          + ((lat_mesh - 40.30) / 0.08) ** 2))
    if with_bathymetry:
        trench = -800.0 * np.exp(
            -(((lon_mesh - 10.25) / 0.08) ** 2
              + ((lat_mesh - 40.15) / 0.04) ** 2))
        grid = hill + trench
    else:
        grid = 50.0 + hill
    return lons, lats, grid.astype(float)


@pytest.fixture(scope="session")
def fixture_cache_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("pywaveprop_env")
    original = paths.CACHE_ROOT
    paths.set_cache_root(root)
    paths.ensure_cache_dirs()

    # Elevation tiles (with bathymetry) covering FIXTURE_BBOX
    for key in _tiles_for_bbox(FIXTURE_BBOX):
        lon0, lat0 = key
        lons, lats, grid = _make_synthetic_elevation(
            lon0, lat0, lon0 + TILE_SIZE_DEG, lat0 + TILE_SIZE_DEG,
            res=FIXTURE_RES, with_bathymetry=True)
        has_bathy = bool(np.any(grid < -0.5))
        path = _tile_cache_path(key, FIXTURE_RES, paths.ELEV_CACHE_DIR)
        np.savez_compressed(path, lons=lons, lats=lats, grid=grid,
                            has_bathymetry=has_bathy)

    # Argo-derived SVP fixture (bypasses argopy entirely)
    depths = np.linspace(0.0, 2000.0, 80)
    speeds = 1500.0 + 3.0 * np.exp(-depths / 200.0) - 4e-4 * (depths - 1300.0) ** 2 / 1300.0
    # Reverse-engineer the cache file name used by load_argo_svp
    import hashlib
    start_date = "2020-01-01"
    end_date = "2023-12-31"
    radius_deg = 1.5
    max_depth = 2000.0
    tag = (f"{FIXTURE_SVP_LAT:.4f}_{FIXTURE_SVP_LON:.4f}_r{radius_deg:.3f}_"
           f"{start_date}_{end_date}_d{max_depth:.0f}")
    key = hashlib.md5(tag.encode()).hexdigest()[:16]
    np.savez_compressed(paths.ARGO_CACHE_DIR / f"ssp_{key}.npz",
                        depths_m=depths, speeds_m_s=speeds,
                        lat=FIXTURE_SVP_LAT, lon=FIXTURE_SVP_LON)

    yield {
        "root": root,
        "bbox": FIXTURE_BBOX,
        "resolution": FIXTURE_RES,
        "svp_latlon": (FIXTURE_SVP_LAT, FIXTURE_SVP_LON),
        "svp_start": start_date,
        "svp_end": end_date,
        "svp_radius_deg": radius_deg,
        "svp_max_depth": max_depth,
    }

    paths.set_cache_root(original)
