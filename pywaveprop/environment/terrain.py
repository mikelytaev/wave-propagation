"""Elevation tile cache for terrain above sea level.

Tiles are fetched from the public AWS ``elevation-tiles-prod`` S3 bucket via
``mercantile`` + ``rasterio`` when available; otherwise the legacy HTTP
elevation-service mode is used. The Mapzen/Terrarium tile set in that bucket
contains both dry land SRTM-style elevations and bathymetry, so the loader
flags the result as potentially carrying bathymetry and
:class:`GeoTerrain.elevation_masked` clamps below-sea-level pixels to 0 m for
tropospheric propagation.

Saved tiles live under
``~/.cache/pywaveprop/elevation/tile_{lon}_{lat}_r{res}.npz`` and contain the
raw grid (negative values preserved) together with a boolean flag. Subsequent
calls for the same bbox load directly from disk with zero network traffic.
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from . import _paths
from .models import GeoTerrain


TILE_SIZE_DEG = 0.5
DEFAULT_RESOLUTION = 0.001


def _tile_origin(lon: float, lat: float) -> tuple[float, float]:
    return (round(math.floor(lon / TILE_SIZE_DEG) * TILE_SIZE_DEG, 6),
            round(math.floor(lat / TILE_SIZE_DEG) * TILE_SIZE_DEG, 6))


def _tiles_for_bbox(bbox: tuple[float, float, float, float]
                    ) -> list[tuple[float, float]]:
    lon_min, lat_min, lon_max, lat_max = bbox
    tiles = []
    lon = math.floor(lon_min / TILE_SIZE_DEG) * TILE_SIZE_DEG
    while lon < lon_max + 1e-9:
        lat = math.floor(lat_min / TILE_SIZE_DEG) * TILE_SIZE_DEG
        while lat < lat_max + 1e-9:
            tiles.append((round(lon, 6), round(lat, 6)))
            lat += TILE_SIZE_DEG
        lon += TILE_SIZE_DEG
    return tiles


def _tile_cache_path(origin: tuple[float, float],
                     resolution: float,
                     cache_dir: Path) -> Path:
    return cache_dir / (
        f"tile_{origin[0]:.6f}_{origin[1]:.6f}_r{resolution:.6f}.npz"
    )


class ElevationTileCache:
    """Disk-persistent tile cache for elevation grids.

    Tiles are laid out on a fixed 0.5 deg x 0.5 deg grid keyed by their
    (lon, lat) south-west corner. A tile is fetched at most once: if the .npz
    exists on disk, it is memory-mapped on demand; otherwise it is downloaded
    from S3 (preferred) or from an HTTP elevation-service, then written to
    disk before the function returns.
    """

    def __init__(self, resolution: float = DEFAULT_RESOLUTION,
                 base_url: str | None = None,
                 cache_dir: str | Path | None = None,
                 geotiff_cache_dir: str | Path | None = None,
                 s3_bucket: str = "elevation-tiles-prod"):
        self.resolution = resolution
        self.base_url = base_url
        self.s3_bucket = s3_bucket
        self.cache_dir = Path(cache_dir) if cache_dir else _paths.ELEV_CACHE_DIR
        self.geotiff_cache_dir = (Path(geotiff_cache_dir)
                                  if geotiff_cache_dir
                                  else self.cache_dir / "geotiff")

        # tile origin -> (lons_1d, lats_1d, grid_2d, has_bathymetry)
        self._tiles: dict[tuple[float, float],
                          tuple[np.ndarray, np.ndarray, np.ndarray, bool]] = {}

        self._direct_available = False
        try:
            import mercantile as _m   # noqa: F401
            import rasterio as _r     # noqa: F401
            import boto3 as _b        # noqa: F401
            self._direct_available = True
        except ImportError:
            pass

    @property
    def direct_available(self) -> bool:
        return self._direct_available

    def load(self, bbox: tuple[float, float, float, float],
             progress_callback=None) -> GeoTerrain:
        """Ensure all tiles for a bbox are cached, then return a ``GeoTerrain``."""
        needed = _tiles_for_bbox(bbox)
        for key in needed:
            if key in self._tiles:
                continue
            path = _tile_cache_path(key, self.resolution, self.cache_dir)
            if path.exists():
                with np.load(path) as data:
                    self._tiles[key] = (
                        data["lons"].copy(),
                        data["lats"].copy(),
                        data["grid"].copy(),
                        bool(data["has_bathymetry"]) if "has_bathymetry" in data else False,
                    )
                continue
            self._fetch_tile(key, progress_callback)
        return self._merge(bbox)

    def missing_tiles(self, bbox: tuple[float, float, float, float]
                      ) -> list[tuple[float, float]]:
        out = []
        for key in _tiles_for_bbox(bbox):
            if key in self._tiles:
                continue
            if _tile_cache_path(key, self.resolution, self.cache_dir).exists():
                continue
            out.append(key)
        return out

    def _fetch_tile(self, key: tuple[float, float], progress_callback=None):
        if self._direct_available:
            self._fetch_tile_s3(key, progress_callback)
        elif self.base_url:
            self._fetch_tile_http(key, progress_callback)
        else:
            raise RuntimeError(
                "no elevation data source available: install rasterio/boto3/"
                "mercantile for direct S3 access, or pass base_url for an "
                "HTTP elevation-service")

    def _fetch_tile_s3(self, key: tuple[float, float], progress_callback=None):
        import mercantile
        import rasterio
        from botocore import UNSIGNED
        from botocore.config import Config
        import boto3

        lon_min, lat_min = key
        lon_max = lon_min + TILE_SIZE_DEG
        lat_max = lat_min + TILE_SIZE_DEG
        res = self.resolution
        zoom = max(1, min(14, math.ceil(math.log2(360.0 / (512 * res)))))

        lons = np.arange(lon_min, lon_max + res / 2, res)
        lats = np.arange(lat_min, lat_max + res / 2, res)
        n_lon, n_lat = len(lons), len(lats)

        aws_tiles = list(mercantile.tiles(
            lon_min, lat_min, lon_max, lat_max, zooms=zoom))

        self.geotiff_cache_dir.mkdir(parents=True, exist_ok=True)
        s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
        for tile in aws_tiles:
            tile_path = self.geotiff_cache_dir / f"{tile.z}/{tile.x}/{tile.y}.tif"
            if tile_path.exists():
                continue
            tile_path.parent.mkdir(parents=True, exist_ok=True)
            s3_key = f"geotiff/{tile.z}/{tile.x}/{tile.y}.tif"
            s3.download_file(self.s3_bucket, s3_key, str(tile_path))

        grid_lons, grid_lats = np.meshgrid(lons, lats)
        elevations = np.zeros((n_lat, n_lon), dtype=float)
        has_bathy = False
        pts_done = 0
        total_pts = n_lon * n_lat

        for tile in aws_tiles:
            bounds = mercantile.bounds(tile)
            tile_path = self.geotiff_cache_dir / f"{tile.z}/{tile.x}/{tile.y}.tif"
            if not tile_path.exists():
                continue
            mask = ((grid_lons >= bounds.west) &
                    (grid_lons <= bounds.east) &
                    (grid_lats >= bounds.south) &
                    (grid_lats <= bounds.north))
            if not np.any(mask):
                continue
            masked_lons = grid_lons[mask].ravel()
            masked_lats = grid_lats[mask].ravel()
            with rasterio.open(tile_path) as ds:
                if ds.crs and str(ds.crs) != 'EPSG:4326':
                    from rasterio.warp import transform as crs_transform
                    xs, ys = crs_transform(
                        'EPSG:4326', ds.crs,
                        masked_lons.tolist(), masked_lats.tolist())
                    coords = list(zip(xs, ys))
                else:
                    coords = list(zip(masked_lons, masked_lats))
                samples = list(ds.sample(coords))
                values = np.array([
                    float(s[0])
                    if (s[0] != ds.nodata and -20000 < float(s[0]) < 20000)
                    else 0.0
                    for s in samples
                ])
                elevations[mask] = values
                if np.any(values < -0.5):
                    has_bathy = True
            pts_done += int(mask.sum())
            if progress_callback:
                progress_callback(pts_done, total_pts)

        self._save_tile(key, lons, lats, elevations, has_bathy)

    def _fetch_tile_http(self, key: tuple[float, float], progress_callback=None):
        import requests as _requests

        lon_min, lat_min = key
        lon_max = lon_min + TILE_SIZE_DEG
        lat_max = lat_min + TILE_SIZE_DEG
        res = self.resolution

        lons = np.arange(lon_min, lon_max + res / 2, res)
        lats = np.arange(lat_min, lat_max + res / 2, res)
        n_lon, n_lat = len(lons), len(lats)
        total_pts = n_lon * n_lat
        grid_lons, grid_lats = np.meshgrid(lons, lats)
        flat_lons = grid_lons.ravel()
        flat_lats = grid_lats.ravel()
        elevations = np.zeros(total_pts, dtype=float)

        base_url = self.base_url
        max_batch = 2000

        def fetch_batch(span):
            start, end = span
            session = _requests.Session()
            payload = {"locations": [
                {"lat": float(flat_lats[i]), "lng": float(flat_lons[i])}
                for i in range(start, end)]}
            resp = session.post(f"{base_url}/elevation",
                                json=payload, timeout=60.0)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") != "OK":
                raise RuntimeError(f"elevation-service error: {data}")
            results = []
            for rec in data["results"]:
                val = rec.get("elevation")
                results.append(val if val is not None else 0.0)
            return start, results

        batches = [(s, min(s + max_batch, total_pts))
                   for s in range(0, total_pts, max_batch)]
        n_done = 0
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(fetch_batch, b): b for b in batches}
            for future in as_completed(futures):
                start, results = future.result()
                for j, val in enumerate(results):
                    elevations[start + j] = val
                n_done += len(results)
                if progress_callback:
                    progress_callback(n_done, total_pts)

        grid = elevations.reshape((n_lat, n_lon))
        has_bathy = bool(np.any(grid < -0.5))
        self._save_tile(key, lons, lats, grid, has_bathy)

    def _save_tile(self, key: tuple[float, float],
                   lons: np.ndarray, lats: np.ndarray,
                   grid: np.ndarray, has_bathy: bool):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = _tile_cache_path(key, self.resolution, self.cache_dir)
        np.savez_compressed(path, lons=lons, lats=lats, grid=grid,
                            has_bathymetry=has_bathy)
        self._tiles[key] = (lons, lats, grid, has_bathy)

    def _merge(self, bbox: tuple[float, float, float, float]) -> GeoTerrain:
        needed = _tiles_for_bbox(bbox)
        res = self.resolution
        tiles = [self._tiles[k] for k in needed if k in self._tiles]
        if not tiles:
            raise RuntimeError(f"no tiles available for bbox {bbox}")

        all_lon_min = min(float(t[0][0]) for t in tiles)
        all_lat_min = min(float(t[1][0]) for t in tiles)
        all_lon_max = max(float(t[0][-1]) for t in tiles)
        all_lat_max = max(float(t[1][-1]) for t in tiles)

        merged_lons = np.arange(all_lon_min, all_lon_max + res / 2, res)
        merged_lats = np.arange(all_lat_min, all_lat_max + res / 2, res)
        merged_grid = np.zeros((len(merged_lats), len(merged_lons)))
        has_bathy = False

        for lons, lats, grid, bathy_flag in tiles:
            ci = int(round((float(lons[0]) - all_lon_min) / res))
            ri = int(round((float(lats[0]) - all_lat_min) / res))
            rows, cols = grid.shape
            ri_end = min(ri + rows, len(merged_lats))
            ci_end = min(ci + cols, len(merged_lons))
            merged_grid[ri:ri_end, ci:ci_end] = grid[:ri_end - ri, :ci_end - ci]
            has_bathy = has_bathy or bathy_flag

        return GeoTerrain(lons=merged_lons, lats=merged_lats,
                          grid=merged_grid, has_bathymetry=has_bathy)


def load_terrain(bbox: tuple[float, float, float, float],
                 resolution: float = DEFAULT_RESOLUTION,
                 base_url: str | None = None,
                 cache_dir: str | Path | None = None,
                 progress_callback=None) -> GeoTerrain:
    """One-shot terrain loader using the default disk cache.

    ``bbox`` is ``(lon_min, lat_min, lon_max, lat_max)``. On first call the
    required tiles are downloaded; subsequent calls are offline.
    """
    cache = ElevationTileCache(resolution=resolution,
                               base_url=base_url,
                               cache_dir=cache_dir)
    return cache.load(bbox, progress_callback=progress_callback)
