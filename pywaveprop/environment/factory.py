"""Build JAX propagation environment models from geolocated data.

These helpers glue the geospatial loaders (elevation, landcover, bathymetry,
Argo) onto :class:`pywaveprop.rwp_jax.TroposphereModel` and
:class:`pywaveprop.uwa_jax.UnderwaterEnvironmentModel` so users can go from
``(lat, lon)`` to a ready-to-propagate environment with a single call.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .landcover import LandCoverLookup
from .models import BathymetryProfile, SoundVelocityProfile
from .seafloor import SeafloorProvider
from .terrain import ElevationTileCache, DEFAULT_RESOLUTION
from .svp import load_argo_svp


def get_troposphere_model(lat1: float, lon1: float,
                          lat2: float, lon2: float, *,
                          n_points: int = 300,
                          resolution: float = DEFAULT_RESOLUTION,
                          use_landcover: bool = True,
                          evaporation_duct_height_m: Optional[float] = None,
                          elevation_cache: ElevationTileCache | None = None,
                          landcover: LandCoverLookup | None = None):
    """Build a :class:`pywaveprop.rwp_jax.TroposphereModel` for a path.

    The terrain is sampled along the great-circle path between the endpoints;
    elevations below sea level (from bathymetry-carrying tiles) are clamped to
    0 m because tropospheric propagation operates above the ocean surface.
    Ground permittivity and conductivity are derived from the mean ESA
    WorldCover class along the path when ``use_landcover=True``.
    """
    from pywaveprop.rwp_jax import (TroposphereModel, GroundMaterial,
                                    EmptyNProfileModel, EvaporationDuctModel)
    from pywaveprop.helmholtz_jax import PiecewiseLinearTerrainModel

    cache = elevation_cache if elevation_cache is not None \
        else ElevationTileCache(resolution=resolution)
    lon_min, lon_max = min(lon1, lon2), max(lon1, lon2)
    lat_min, lat_max = min(lat1, lat2), max(lat1, lat2)
    margin = max(5 * resolution, 0.01)
    bbox = (lon_min - margin, lat_min - margin,
            lon_max + margin, lat_max + margin)
    terrain = cache.load(bbox)
    ranges_m, elevations_m = terrain.profile(
        lon1, lat1, lon2, lat2, n_points=n_points, mask_underwater=True)

    terrain_model = PiecewiseLinearTerrainModel(
        x_grid_m=np.asarray(ranges_m, dtype=float),
        height=np.asarray(elevations_m, dtype=float),
    )

    if use_landcover:
        lc = landcover if landcover is not None else LandCoverLookup()
        if lc.available:
            eps, sgm = lc.path_mean_ground_params(lat1, lon1, lat2, lon2)
        else:
            eps, sgm = 15.0, 0.005
    else:
        eps, sgm = 15.0, 0.005

    ground = GroundMaterial(eps=eps, sigma=sgm)

    if evaporation_duct_height_m is not None:
        n_profile = EvaporationDuctModel(height_m=evaporation_duct_height_m)
    else:
        n_profile = EmptyNProfileModel()

    return TroposphereModel(
        N_profile=n_profile,
        terrain=terrain_model,
        ground_material=ground,
    )


def get_underwater_environment_model(lat1: float, lon1: float,
                                     lat2: float, lon2: float, *,
                                     n_range_points: int = 300,
                                     resolution: float = DEFAULT_RESOLUTION,
                                     svp_radius_deg: float = 1.5,
                                     svp_start_date: str = "2020-01-01",
                                     svp_end_date: str = "2023-12-31",
                                     max_svp_depth_m: float = 2000.0,
                                     bottom_sound_speed_m_s: float = 1700.0,
                                     bottom_density: float = 1.5,
                                     bottom_attenuation_dm_lambda: float = 0.5,
                                     svp: SoundVelocityProfile | None = None,
                                     bathymetry: BathymetryProfile | None = None,
                                     ):
    """Build a :class:`pywaveprop.uwa_jax.UnderwaterEnvironmentModel`.

    The water column SSP is fetched from Argo; the seafloor range-dependent
    depth is taken from the bathymetry tiles. A single homogeneous bottom
    layer is added underneath (``bottom_*`` parameters).
    """
    from pywaveprop.uwa_jax import (UnderwaterEnvironmentModel,
                                    UnderwaterLayerModel)
    from pywaveprop.helmholtz_jax import (PiecewiseLinearTerrainModel,
                                          PiecewiseLinearWaveSpeedModel,
                                          LinearSlopeWaveSpeedModel)

    if svp is None:
        mid_lat = 0.5 * (lat1 + lat2)
        mid_lon = 0.5 * (lon1 + lon2)
        svp = load_argo_svp(
            mid_lat, mid_lon,
            radius_deg=svp_radius_deg,
            start_date=svp_start_date,
            end_date=svp_end_date,
            max_depth_m=max_svp_depth_m,
        )

    if bathymetry is None:
        bathymetry = SeafloorProvider(resolution=resolution).profile(
            lat1, lon1, lat2, lon2, n_points=n_range_points)

    water_layer_depth_m = float(min(svp.max_depth_m,
                                    bathymetry.max_depth_m + 1.0))
    water_ssp = PiecewiseLinearWaveSpeedModel(
        z_grid_m=np.asarray(svp.depths_m, dtype=float),
        sound_speed=np.asarray(svp.speeds_m_s, dtype=float),
    )
    water_layer = UnderwaterLayerModel(
        height_m=water_layer_depth_m,
        sound_speed_profile_m_s=water_ssp,
        density=1.0,
        attenuation_dm_lambda=0.0,
    )
    bottom_layer = UnderwaterLayerModel(
        height_m=water_layer_depth_m,
        sound_speed_profile_m_s=LinearSlopeWaveSpeedModel(
            c0=bottom_sound_speed_m_s, slope_degrees=0),
        density=bottom_density,
        attenuation_dm_lambda=bottom_attenuation_dm_lambda,
    )

    bathy_model = PiecewiseLinearTerrainModel(
        x_grid_m=np.asarray(bathymetry.ranges_m, dtype=float),
        height=np.asarray(bathymetry.depths_m, dtype=float),
    )

    return UnderwaterEnvironmentModel(
        layers=[water_layer, bottom_layer],
        bathymetry=bathy_model,
    )
