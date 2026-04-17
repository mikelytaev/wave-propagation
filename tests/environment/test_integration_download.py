"""Integration test: real downloads + caching round-trip.

Skipped by default. Opt in with::

    PYWAVEPROP_NETWORK_TESTS=1 pytest tests/environment/test_integration_download.py

The goal is to verify, against the live public buckets:
  * One elevation tile downloads and ends up at the expected .npz path.
  * A second call for the same tile loads from disk (no S3 access).
  * One WorldCover GeoTIFF downloads and a land-cover lookup works.
  * An Argo region fetch writes both the raw NetCDF and the derived SVP
    cache files.

The geographies are intentionally tiny (one 0.5 deg tile, one 3 deg
WorldCover tile, a 1 deg Argo box) so the download finishes in under a
minute on a reasonable connection.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest


NETWORK = os.environ.get("PYWAVEPROP_NETWORK_TESTS") == "1"
pytestmark = pytest.mark.skipif(
    not NETWORK,
    reason="Set PYWAVEPROP_NETWORK_TESTS=1 to run the live-download integration test",
)


@pytest.fixture
def net_cache(tmp_path):
    from pywaveprop.environment import _paths as paths
    original = paths.CACHE_ROOT
    paths.set_cache_root(tmp_path)
    paths.ensure_cache_dirs()
    yield tmp_path
    paths.set_cache_root(original)


def test_elevation_tile_download_and_reuse(net_cache):
    pytest.importorskip("rasterio")
    pytest.importorskip("boto3")
    pytest.importorskip("mercantile")
    from pywaveprop.environment import ElevationTileCache
    from pywaveprop.environment.terrain import _tile_cache_path
    from pywaveprop.environment import _paths as paths

    bbox = (10.10, 40.10, 10.15, 40.15)
    res = 0.01

    cache = ElevationTileCache(resolution=res, cache_dir=paths.ELEV_CACHE_DIR)
    assert cache.missing_tiles(bbox), "expected at least one missing tile on first run"

    gt = cache.load(bbox)
    assert gt.grid.size > 0

    tile_files = list(paths.ELEV_CACHE_DIR.glob("tile_*.npz"))
    assert tile_files, "no tile cache file was written"

    # Second call must be offline.
    cache2 = ElevationTileCache(resolution=res, cache_dir=paths.ELEV_CACHE_DIR)
    assert cache2.missing_tiles(bbox) == []
    gt2 = cache2.load(bbox)
    assert gt2.grid.shape == gt.grid.shape


def test_landcover_tile_download(net_cache):
    pytest.importorskip("rasterio")
    pytest.importorskip("boto3")
    from pywaveprop.environment import LandCoverLookup

    lc = LandCoverLookup()
    assert lc.available

    # A point in Rome (built-up should map to class 50).
    params = lc.get_ground_params(41.90, 12.50)
    assert len(params) == 2
    tile_files = list(lc.cache_dir.glob("ESA_WorldCover_10m_2021_v200_*_Map.tif"))
    assert tile_files, "ESA WorldCover tile was not cached"


def test_argo_download_and_cache(net_cache):
    pytest.importorskip("argopy")
    pytest.importorskip("xarray")
    from pywaveprop.environment import load_argo_svp, _paths as paths

    svp = load_argo_svp(
        lat=35.0, lon=-25.0,
        radius_deg=1.0,
        start_date="2022-06-01",
        end_date="2022-06-15",
        max_depth_m=1500.0,
        n_depth_bins=40,
    )
    assert svp.depths_m.size == 40
    assert np.all(np.isfinite(svp.speeds_m_s))

    raw = list(paths.ARGO_CACHE_DIR.glob("argo_*.nc"))
    profiles = list(paths.ARGO_CACHE_DIR.glob("ssp_*.npz"))
    assert raw, "raw Argo NetCDF was not cached"
    assert profiles, "derived SVP .npz was not cached"
