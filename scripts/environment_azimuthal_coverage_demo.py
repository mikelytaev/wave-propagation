"""Azimuthal tropospheric coverage map over real-world terrain.

The original version of this example fetched elevations from a local HTTP
service (``localhost:8010``) and relied on an external ``azimuthal_coverage``
module. Here we replace both with :mod:`pywaveprop.environment`: a single
:class:`ElevationTileCache` covers the whole disk around the antenna, and the
radial profiles are sampled from the cached grid for every azimuth.

Run::

    python scripts/environment_azimuthal_coverage_demo.py
"""
from __future__ import annotations

import logging
import math
import os
import sys
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pywaveprop.environment import ElevationTileCache
from pywaveprop.helmholtz_jax import PiecewiseLinearTerrainModel
from pywaveprop.rwp_jax import (GroundMaterial, RWPComputationalParams,
                                RWPGaussSourceModel, TroposphereModel,
                                create_rwp_model)

jax.config.update("jax_enable_x64", True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    force=True,
)
log = logging.getLogger("env_azim_demo")

EARTH_RADIUS_M = 6371000.0


@dataclass
class GeoCoordinate:
    lat: float
    lon: float


def destination_point(lat_deg: float, lon_deg: float,
                      azimuth_deg: float, distance_m: float
                      ) -> tuple[float, float]:
    """Spherical-earth direct geodesic — matches the haversine used in the
    environment package, so terrain ranges line up consistently."""
    ang = distance_m / EARTH_RADIUS_M
    lat1 = math.radians(lat_deg)
    lon1 = math.radians(lon_deg)
    az = math.radians(azimuth_deg)
    lat2 = math.asin(math.sin(lat1) * math.cos(ang)
                     + math.cos(lat1) * math.sin(ang) * math.cos(az))
    lon2 = lon1 + math.atan2(
        math.sin(az) * math.sin(ang) * math.cos(lat1),
        math.cos(ang) - math.sin(lat1) * math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2)


def azimuthal_segments(antenna_coords: GeoCoordinate,
                       max_range_m: float,
                       max_segment_spacing_m: float) -> list[float]:
    """Return azimuth angles (deg, 0..360) chosen so the arc-length step at
    ``max_range_m`` does not exceed ``max_segment_spacing_m``."""
    n = max(1, int(math.ceil(
        2 * math.pi * max_range_m / max_segment_spacing_m)))
    return list(np.linspace(0.0, 360.0, n, endpoint=False))


def bbox_around(antenna_coords: GeoCoordinate, max_range_m: float,
                margin_deg: float = 0.02
                ) -> tuple[float, float, float, float]:
    """Lon/lat bounding box containing the propagation disk."""
    dlat = math.degrees(max_range_m / EARTH_RADIUS_M)
    dlon = dlat / max(math.cos(math.radians(antenna_coords.lat)), 1e-6)
    return (antenna_coords.lon - dlon - margin_deg,
            antenna_coords.lat - dlat - margin_deg,
            antenna_coords.lon + dlon + margin_deg,
            antenna_coords.lat + dlat + margin_deg)


def radial_terrain_profile(terrain, antenna_coords: GeoCoordinate,
                           azimuth_deg: float, max_range_m: float,
                           n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample terrain elevations along one radial of length ``max_range_m``.

    The radial endpoint is computed on a sphere, then ``GeoTerrain.profile``
    vectorizes the bilinear lookup between the antenna and that endpoint —
    much faster than per-point ``elevation_at`` calls.
    """
    end_lat, end_lon = destination_point(
        antenna_coords.lat, antenna_coords.lon, azimuth_deg, max_range_m)
    ranges_m, elevations = terrain.profile(
        antenna_coords.lon, antenna_coords.lat,
        end_lon, end_lat,
        n_points=n_points, mask_underwater=True)
    return ranges_m, elevations


def cylindric_rwp(*, antenna_freq_hz: float, antenna_height_m: float,
                  antenna_beam_width_deg: float,
                  antenna_elevation_deg: float, antenna_polarz: str,
                  antenna_coords: GeoCoordinate,
                  max_range_m: float, max_segment_spacing_m: float = 1500.0,
                  range_resolution_m: float = 100.0):
    azimuths = azimuthal_segments(
        antenna_coords, max_range_m, max_segment_spacing_m)
    n_terrain_points = max(2, int(round(max_range_m / range_resolution_m)))

    cache = ElevationTileCache(resolution=0.001)
    bbox = bbox_around(antenna_coords, max_range_m)
    missing = cache.missing_tiles(bbox)
    log.info("Terrain bbox %s, disk radius %.1f km, %d azimuth segments",
             bbox, max_range_m / 1000, len(azimuths))
    log.info("Tile cache: %d tile(s) need downloading (others on disk)",
             len(missing))

    def tile_progress(done, total):
        # Called per S3 sub-tile while a 0.5° tile is being assembled.
        if total and (done == total or done % max(1, total // 8) == 0):
            log.info("  tile fetch: %d / %d points", done, total)

    tic = time.time()
    geo_terrain = cache.load(bbox, progress_callback=tile_progress)
    log.info("Terrain ready in %.1fs (grid %s, bathy=%s)",
             time.time() - tic, geo_terrain.grid.shape,
             geo_terrain.has_bathymetry)

    # Use a single average-ground default for every azimuth; the WorldCover
    # lookup is intentionally skipped to keep this demo fully offline after
    # the elevation tiles are cached.
    eps, sgm = 15.0, 0.005
    log.info("Ground material (default): eps=%.2f, sigma=%.4f", eps, sgm)

    src = RWPGaussSourceModel(
        freq_hz=antenna_freq_hz,
        height_m=antenna_height_m,
        beam_width_deg=antenna_beam_width_deg,
        elevation_angle_deg=antenna_elevation_deg,
        polarization=antenna_polarz,
    )
    # Seed the terrain with a flat profile of the same shape we will later
    # mutate to; this fixes the JIT-compiled grid size on the first compute().
    init_x = np.linspace(0.0, max_range_m, n_terrain_points)
    init_h = np.zeros(n_terrain_points)
    env = TroposphereModel(
        terrain=PiecewiseLinearTerrainModel(
            jnp.array(init_x), jnp.array(init_h)),
        ground_material=GroundMaterial(eps=eps, sigma=sgm),
    )
    params = RWPComputationalParams(
        max_range_m=max_range_m,
        max_height_m=300.0,
        max_angle_deg=2.0,
        dx_m=200.0,
        dz_m=1.0,
    )
    params.approx_method = 'ratinterp'
    log.info("Building RWP propagator model")
    tic = time.time()
    model = create_rwp_model(src, env, params)
    log.info("  propagator built in %.2fs", time.time() - tic)

    log.info("JIT-compiling first compute() — this can take 10-30s")
    tic = time.time()
    first = model.compute(src.aperture(model.z_computational_grid()))
    first.block_until_ready()
    log.info("  first compute (incl. JIT) done in %.1fs", time.time() - tic)

    result = []
    loop_start = time.time()
    for i, az in enumerate(azimuths):
        tic = time.time()
        x_grid, elevations = radial_terrain_profile(
            geo_terrain, antenna_coords, az,
            max_range_m, n_terrain_points)
        terrain_time = time.time() - tic

        env.terrain.x_grid_m = jnp.array(x_grid)
        env.terrain.height = jnp.array(elevations)

        tic = time.time()
        field = model.compute(src.aperture(model.z_computational_grid()))
        field.block_until_ready()
        compute_time = time.time() - tic

        log.info("Segment %3d/%d  az=%6.1f°  terrain=%.0fms compute=%.0fms",
                 i + 1, len(azimuths), az,
                 terrain_time * 1000, compute_time * 1000)

        result.append({
            "field": np.asarray(field),
            "x_grid": np.asarray(model.x_output_grid()),
            "z_grid": np.asarray(model.z_output_grid()),
            "terrain_x": np.asarray(x_grid),
            "terrain_z": np.asarray(elevations),
            "angle": az,
        })

    log.info("Azimuth loop finished in %.1fs (%d segments)",
             time.time() - loop_start, len(azimuths))
    return result


def _coverage_grid(result, height_m):
    angles, ranges, field_values = [], [], []
    for segment in result:
        z_grid = segment["z_grid"]
        field = segment["field"]
        # Output shape from RationalHelmholtzPropagator.compute is (n_x, n_z).
        height_idx = int(np.argmin(np.abs(z_grid - height_m)))
        field_at_height = field[:, height_idx]
        field_db = 20 * np.log10(np.abs(field_at_height) + 1e-12)
        angle_rad = np.deg2rad(segment["angle"])
        for r, f in zip(segment["x_grid"], field_db):
            angles.append(angle_rad)
            ranges.append(float(r))
            field_values.append(float(f))

    angles = np.array(angles)
    ranges = np.array(ranges)
    field_values = np.array(field_values)

    unique_ranges = np.linspace(ranges.min(), ranges.max(), 200)
    angle_grid, range_grid = np.meshgrid(
        np.linspace(0.0, 2 * np.pi, 360), unique_ranges)
    field_grid = griddata(
        (angles, ranges), field_values,
        (angle_grid, range_grid),
        method="linear", fill_value=np.nan,
    )
    return angle_grid, range_grid, field_grid, field_values


def _format_polar_axis(ax):
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
    ax.set_xticklabels(["0°", "90°", "180°", "270°"])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, p: f"{x/1000:.0f} km" if x >= 1000 else f"{x:.0f} m"))


def plot_azimuthal_coverage(result, height_m=10, vmin=None, vmax=None):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="polar")

    angle_grid, range_grid, field_grid, field_values = _coverage_grid(
        result, height_m)

    if vmin is None:
        vmin = float(np.nanpercentile(field_values, 5))
    if vmax is None:
        vmax = float(np.nanpercentile(field_values, 95))

    pcm = ax.pcolormesh(angle_grid, range_grid, field_grid,
                        cmap="jet", vmin=vmin, vmax=vmax, shading="auto")
    cbar = plt.colorbar(pcm, ax=ax, pad=0.1)
    cbar.set_label("Field strength (dB)", rotation=270, labelpad=20)

    _format_polar_axis(ax)
    plt.title(f"Azimuthal coverage at {height_m} m height", pad=20)
    plt.tight_layout()
    return fig, ax


def plot_azimuthal_coverage_grid(result, heights_m, vmin=None, vmax=None,
                                 nrows=3, ncols=2):
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 18),
                             subplot_kw={"projection": "polar"})
    axes = np.atleast_2d(axes).ravel()

    grids = [_coverage_grid(result, h) for h in heights_m]
    all_vals = np.concatenate([g[3] for g in grids])
    if vmin is None:
        vmin = float(np.nanpercentile(all_vals, 5))
    if vmax is None:
        vmax = float(np.nanpercentile(all_vals, 95))

    pcm = None
    for ax, h, (angle_grid, range_grid, field_grid, _) in zip(
            axes, heights_m, grids):
        pcm = ax.pcolormesh(angle_grid, range_grid, field_grid,
                            cmap="jet", vmin=vmin, vmax=vmax, shading="auto")
        _format_polar_axis(ax)
        ax.set_title(f"{h} m height", pad=15)

    for ax in axes[len(heights_m):]:
        ax.set_visible(False)

    fig.subplots_adjust(right=0.88, hspace=0.35, wspace=0.25)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(pcm, cax=cbar_ax)
    cbar.set_label("Field strength (dB)", rotation=270, labelpad=20)
    fig.suptitle("Azimuthal coverage", fontsize=16, y=0.995)
    return fig, axes


if __name__ == "__main__":
    antenna_coords = GeoCoordinate(lat=60.0, lon=30.0)
    result = cylindric_rwp(
        antenna_freq_hz=2400e6,
        antenna_height_m=30.0,
        antenna_beam_width_deg=2.0,
        antenna_elevation_deg=0.0,
        antenna_polarz="H",
        antenna_coords=antenna_coords,
        max_range_m=30e3,
        max_segment_spacing_m=2000.0,
        range_resolution_m=150.0,
    )

    plot_heights_m = [5, 10, 20, 30, 50, 100]
    log.info("Rendering 3x2 polar coverage grid at heights %s", plot_heights_m)
    fig, _ = plot_azimuthal_coverage_grid(
        result, plot_heights_m, nrows=3, ncols=2)
    out = os.path.join(
        os.path.dirname(__file__),
        "environment_azimuthal_coverage_demo.png")
    plt.savefig(out, dpi=130)
    plt.close(fig)
    log.info("Saved plot to %s", out)
