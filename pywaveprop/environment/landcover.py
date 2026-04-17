"""ESA WorldCover 2021 v200 land-cover lookup with local caching.

Returns ESA class codes and derived ground electrical parameters
(``eps_dielect``, ``sgm_conductivity``) suitable for the Longley-Rice /
SPLAT! / ITM family of propagation tools and for pywaveprop's lower NLBC.
3x3 deg Cloud-Optimized GeoTIFF tiles are pulled from the public
``s3://esa-worldcover`` bucket at most once per tile per machine.
"""
from __future__ import annotations

import math
from pathlib import Path

from . import _paths
from .models import LandCoverClassification


# ESA WorldCover 2021 v200 class -> (eps_dielect, sgm_conductivity)
# Values follow the SPLAT!/ITM standard ground parameter tables.
WORLDCOVER_GROUND_PARAMS: dict[int, tuple[float, float]] = {
    10: (15.0, 0.005),   # Tree cover        -> farmland/forest
    20: (15.0, 0.005),   # Shrubland          -> farmland/forest
    30: (15.0, 0.005),   # Grassland          -> average ground
    40: (15.0, 0.005),   # Cropland           -> farmland
    50: (5.0, 0.001),    # Built-up           -> city
    60: (13.0, 0.002),   # Bare/sparse veg    -> mountain/sand
    70: (3.0, 0.001),    # Snow and ice       -> poor ground
    80: (80.0, 0.010),   # Permanent water    -> fresh water
    90: (12.0, 0.007),   # Herbaceous wetland -> marshy land
    95: (12.0, 0.007),   # Mangroves          -> marshy land
    100: (15.0, 0.005),  # Moss and lichen    -> average ground
}
DEFAULT_GROUND_PARAMS = (15.0, 0.005)


def _worldcover_tile_key(lat: float, lon: float) -> str:
    lat_base = math.floor(lat / 3) * 3
    lon_base = math.floor(lon / 3) * 3
    ns = "N" if lat_base >= 0 else "S"
    ew = "E" if lon_base >= 0 else "W"
    return (f"v200/2021/map/ESA_WorldCover_10m_2021_v200_"
            f"{ns}{abs(lat_base):02d}{ew}{abs(lon_base):03d}_Map.tif")


class LandCoverLookup:
    """Cached ESA WorldCover point-sampling lookup."""

    def __init__(self, cache_dir: str | Path | None = None,
                 s3_bucket: str = "esa-worldcover"):
        self.cache_dir = Path(cache_dir) if cache_dir else _paths.LANDCOVER_CACHE_DIR
        self.s3_bucket = s3_bucket
        self._available = False
        try:
            import rasterio as _r   # noqa: F401
            import boto3 as _b      # noqa: F401
            self._available = True
        except ImportError:
            pass

    @property
    def available(self) -> bool:
        return self._available

    def _ensure_tile(self, s3_key: str) -> Path:
        filename = s3_key.rsplit("/", 1)[-1]
        local_path = self.cache_dir / filename
        if local_path.exists():
            return local_path
        local_path.parent.mkdir(parents=True, exist_ok=True)

        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config

        s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
        s3.download_file(self.s3_bucket, s3_key, str(local_path))
        return local_path

    def get_class(self, lat: float, lon: float) -> int | None:
        if not self._available:
            return None
        import rasterio
        s3_key = _worldcover_tile_key(lat, lon)
        try:
            tile_path = self._ensure_tile(s3_key)
        except Exception:
            return None
        try:
            with rasterio.open(tile_path) as ds:
                vals = list(ds.sample([(lon, lat)]))
                if vals and vals[0][0] != ds.nodata:
                    return int(vals[0][0])
        except Exception:
            return None
        return None

    def get_ground_params(self, lat: float, lon: float
                          ) -> tuple[float, float]:
        """Return ``(eps_dielect, sgm_conductivity)`` for ``(lat, lon)``."""
        cls = self.get_class(lat, lon)
        if cls is None:
            return DEFAULT_GROUND_PARAMS
        return WORLDCOVER_GROUND_PARAMS.get(cls, DEFAULT_GROUND_PARAMS)

    def classify(self, lat: float, lon: float) -> LandCoverClassification:
        cls = self.get_class(lat, lon)
        eps, sgm = (WORLDCOVER_GROUND_PARAMS.get(cls, DEFAULT_GROUND_PARAMS)
                    if cls is not None else DEFAULT_GROUND_PARAMS)
        return LandCoverClassification(
            class_id=cls if cls is not None else -1,
            eps_dielect=eps,
            sgm_conductivity=sgm,
        )

    def path_mean_ground_params(self, lat1: float, lon1: float,
                                lat2: float, lon2: float,
                                n_samples: int = 9
                                ) -> tuple[float, float]:
        eps_sum = 0.0
        sgm_sum = 0.0
        for i in range(n_samples):
            frac = (i + 0.5) / n_samples
            lat = lat1 + frac * (lat2 - lat1)
            lon = lon1 + frac * (lon2 - lon1)
            eps, sgm = self.get_ground_params(lat, lon)
            eps_sum += eps
            sgm_sum += sgm
        return eps_sum / n_samples, sgm_sum / n_samples
