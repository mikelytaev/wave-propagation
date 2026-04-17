"""Geospatial environment data loaders for pywaveprop.

Submodules:
    - :mod:`~pywaveprop.environment.terrain`: cached elevation tile loader.
    - :mod:`~pywaveprop.environment.landcover`: ESA WorldCover 2021 lookup
      for ground electrical parameters.
    - :mod:`~pywaveprop.environment.seafloor`: bathymetry profile loader
      (shares the elevation tile cache, negative values = depth below sea
      level).
    - :mod:`~pywaveprop.environment.svp`: Argo-based sound velocity profile
      loader.
    - :mod:`~pywaveprop.environment.factory`: high-level factories that
      build :class:`pywaveprop.rwp_jax.TroposphereModel` and
      :class:`pywaveprop.uwa_jax.UnderwaterEnvironmentModel` from
      ``(lat, lon)`` inputs.

All downloaded data is cached to ``~/.cache/pywaveprop/`` (override via the
``PYWAVEPROP_CACHE_DIR`` environment variable or
:func:`~pywaveprop.environment._paths.set_cache_root`). Elevation tiles that
include bathymetry have their below-sea-level pixels clamped to 0 m when
exposed as terrain, because tropospheric propagation is computed above the
sea surface.
"""
from ._paths import (CACHE_ROOT, ELEV_CACHE_DIR, LANDCOVER_CACHE_DIR,
                     BATHYMETRY_CACHE_DIR, ARGO_CACHE_DIR,
                     ensure_cache_dirs, set_cache_root)
from .models import (GeoTerrain, LandCoverClassification,
                     BathymetryProfile, SoundVelocityProfile)
from .terrain import (ElevationTileCache, load_terrain,
                      TILE_SIZE_DEG, DEFAULT_RESOLUTION)
from .landcover import (LandCoverLookup, WORLDCOVER_GROUND_PARAMS,
                        DEFAULT_GROUND_PARAMS)
from .seafloor import (SeafloorProvider, load_bathymetry_profile,
                       load_bathymetry_area)
from .svp import load_argo_svp
from .factory import (get_troposphere_model,
                      get_underwater_environment_model)

__all__ = [
    "CACHE_ROOT", "ELEV_CACHE_DIR", "LANDCOVER_CACHE_DIR",
    "BATHYMETRY_CACHE_DIR", "ARGO_CACHE_DIR",
    "ensure_cache_dirs", "set_cache_root",
    "GeoTerrain", "LandCoverClassification",
    "BathymetryProfile", "SoundVelocityProfile",
    "ElevationTileCache", "load_terrain",
    "TILE_SIZE_DEG", "DEFAULT_RESOLUTION",
    "LandCoverLookup", "WORLDCOVER_GROUND_PARAMS", "DEFAULT_GROUND_PARAMS",
    "SeafloorProvider", "load_bathymetry_profile", "load_bathymetry_area",
    "load_argo_svp",
    "get_troposphere_model", "get_underwater_environment_model",
]
