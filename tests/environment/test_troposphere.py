"""Troposphere environment factory + RWP forward task using cached fixtures."""
from __future__ import annotations

import os

import numpy as np
import pytest


def test_geoterrain_bathymetry_clamped_to_sea_level(fixture_cache_root):
    from pywaveprop.environment import load_terrain

    gt = load_terrain(
        fixture_cache_root["bbox"],
        resolution=fixture_cache_root["resolution"],
    )
    assert gt.has_bathymetry is True
    assert float(np.min(gt.grid)) < -0.5
    assert float(np.min(gt.elevation_masked)) >= 0.0


def test_tropospheric_environment_from_coords(fixture_cache_root):
    from pywaveprop.environment import get_troposphere_model

    lat1, lon1 = 40.15, 10.15
    lat2, lon2 = 40.35, 10.35

    env = get_troposphere_model(
        lat1, lon1, lat2, lon2,
        n_points=80,
        resolution=fixture_cache_root["resolution"],
        use_landcover=False,
    )

    assert env.terrain is not None
    heights = np.asarray(env.terrain.height)
    # Any bathymetric samples must be clamped to sea level.
    assert float(np.min(heights)) >= 0.0
    # The fixture has a 400-m hill roughly on this path.
    assert float(np.max(heights)) > 50.0


def test_rwp_forward_task_with_loaded_terrain(fixture_cache_root):
    jax = pytest.importorskip("jax")
    from pywaveprop.environment import get_troposphere_model
    from pywaveprop.rwp_jax import (RWPComputationalParams,
                                    RWPGaussSourceModel, rwp_forward_task)

    env = get_troposphere_model(
        40.22, 10.22, 40.28, 10.28,
        n_points=40,
        resolution=fixture_cache_root["resolution"],
        use_landcover=False,
    )
    src = RWPGaussSourceModel(
        freq_hz=10e6,
        height_m=100.0,
        beam_width_deg=15.0,
    )
    max_range = float(env.terrain.x_grid_m[-1])
    params = RWPComputationalParams(
        max_range_m=max_range,
        max_height_m=600.0,
        x_output_points=40,
        z_output_points=40,
    )
    params.rational_approx_order = (1, 2)
    field = rwp_forward_task(src, env, params)
    arr = np.asarray(field.field)
    assert np.all(np.isfinite(arr))
    assert float(np.max(np.abs(arr))) > 0.0
