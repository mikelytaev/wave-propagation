"""Geospatial and atmospheric environment data loaders for pywaveprop.

Submodules:
    - :mod:`~pywaveprop.environment.terrain`: cached elevation tile loader.
    - :mod:`~pywaveprop.environment.landcover`: ESA WorldCover 2021 lookup
      for ground electrical parameters.
    - :mod:`~pywaveprop.environment.seafloor`: bathymetry profile loader
      (shares the elevation tile cache, negative values = depth below sea
      level).
    - :mod:`~pywaveprop.environment.svp`: Argo-based sound velocity profile
      loader.
    - :mod:`~pywaveprop.environment.gfs`: NOAA GFS numerical-weather-prediction
      access (NOMADS subsets and the AWS long-term archive).
    - :mod:`~pywaveprop.environment.refractivity`: radio refractivity ``N`` /
      modified refractivity ``M`` physics and ducting diagnostics.
    - :mod:`~pywaveprop.environment.evaporation`: Monin-Obukhov surface-layer
      model for evaporation ducts over the sea and thermal profiles over land.
    - :mod:`~pywaveprop.environment.nwp`: GFS -> ``M(height, lat, lon)`` cubes
      and range-dependent ``M(x, z)`` transects for the PE solvers.
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
                     BATHYMETRY_CACHE_DIR, ARGO_CACHE_DIR, GFS_CACHE_DIR,
                     ensure_cache_dirs, set_cache_root)
from .models import (GeoTerrain, LandCoverClassification,
                     BathymetryProfile, SoundVelocityProfile,
                     RefractivityTransect)
from .terrain import (ElevationTileCache, load_terrain,
                      TILE_SIZE_DEG, DEFAULT_RESOLUTION)
from .landcover import (LandCoverLookup, WORLDCOVER_GROUND_PARAMS,
                        DEFAULT_GROUND_PARAMS)
from .seafloor import (SeafloorProvider, load_bathymetry_profile,
                       load_bathymetry_area)
from .svp import load_argo_svp
from .gfs import BBox, GFSRequest, latest_available_cycle
from .refractivity import (duct_diagnostics, duct_diagnostics_field,
                           horizontal_gradient, modified_refractivity_m,
                           refractivity_n, profile_from_rh,
                           range_dependent_M_profile, uniform_M_profile)
from .evaporation import (evaporation_duct, evaporation_duct_field,
                          land_surface_profile)
from .nwp import (fetch_refractivity_cube, fetch_surface_bulk,
                  interpolate_to_height, refractivity_transect,
                  refractivity_transect_from_cube, save_transect,
                  load_transect)
from .factory import (get_troposphere_model,
                      get_underwater_environment_model)

__all__ = [
    "CACHE_ROOT", "ELEV_CACHE_DIR", "LANDCOVER_CACHE_DIR",
    "BATHYMETRY_CACHE_DIR", "ARGO_CACHE_DIR", "GFS_CACHE_DIR",
    "ensure_cache_dirs", "set_cache_root",
    "GeoTerrain", "LandCoverClassification",
    "BathymetryProfile", "SoundVelocityProfile", "RefractivityTransect",
    "ElevationTileCache", "load_terrain",
    "TILE_SIZE_DEG", "DEFAULT_RESOLUTION",
    "LandCoverLookup", "WORLDCOVER_GROUND_PARAMS", "DEFAULT_GROUND_PARAMS",
    "SeafloorProvider", "load_bathymetry_profile", "load_bathymetry_area",
    "load_argo_svp",
    "BBox", "GFSRequest", "latest_available_cycle",
    "duct_diagnostics", "duct_diagnostics_field", "horizontal_gradient",
    "modified_refractivity_m", "refractivity_n", "profile_from_rh",
    "range_dependent_M_profile", "uniform_M_profile",
    "evaporation_duct", "evaporation_duct_field", "land_surface_profile",
    "fetch_refractivity_cube", "fetch_surface_bulk", "interpolate_to_height",
    "refractivity_transect", "refractivity_transect_from_cube",
    "save_transect", "load_transect",
    "get_troposphere_model", "get_underwater_environment_model",
]
