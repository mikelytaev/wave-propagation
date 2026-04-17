"""Underwater environment factory + UWA forward task using cached fixtures."""
from __future__ import annotations

import numpy as np
import pytest


def test_svp_loaded_from_cache(fixture_cache_root):
    from pywaveprop.environment import load_argo_svp

    lat, lon = fixture_cache_root["svp_latlon"]
    svp = load_argo_svp(
        lat, lon,
        radius_deg=fixture_cache_root["svp_radius_deg"],
        start_date=fixture_cache_root["svp_start"],
        end_date=fixture_cache_root["svp_end"],
        max_depth_m=fixture_cache_root["svp_max_depth"],
    )
    assert svp.depths_m.size > 0
    assert np.all(np.isfinite(svp.speeds_m_s))
    assert 1400.0 < float(np.min(svp.speeds_m_s)) < 1600.0
    assert 1400.0 < float(np.max(svp.speeds_m_s)) < 1600.0


def test_bathymetry_profile_from_tiles(fixture_cache_root):
    from pywaveprop.environment import load_bathymetry_profile

    bathy = load_bathymetry_profile(
        40.12, 10.20, 40.18, 10.30,
        n_points=50,
        resolution=fixture_cache_root["resolution"],
        default_depth_m=10.0,
    )
    assert bathy.ranges_m.size == 50
    assert float(np.max(bathy.depths_m)) > 10.0


def test_underwater_environment_model_from_coords(fixture_cache_root):
    jax = pytest.importorskip("jax")
    from pywaveprop.environment import get_underwater_environment_model

    env = get_underwater_environment_model(
        40.12, 10.20, 40.18, 10.30,
        n_range_points=50,
        resolution=fixture_cache_root["resolution"],
        svp_radius_deg=fixture_cache_root["svp_radius_deg"],
        svp_start_date=fixture_cache_root["svp_start"],
        svp_end_date=fixture_cache_root["svp_end"],
        max_svp_depth_m=fixture_cache_root["svp_max_depth"],
    )
    assert len(env.layers) == 2
    z_test = np.linspace(0.0, 200.0, 5)
    c = env.layers[0].sound_speed_profile_m_s(z_test)
    c = np.asarray(c)
    assert np.all(np.isfinite(c))
    assert 1400.0 < float(np.min(c)) < 1600.0


def test_uwa_forward_task_from_environment(fixture_cache_root):
    jax = pytest.importorskip("jax")
    from pywaveprop.environment import get_underwater_environment_model
    from pywaveprop.uwa_jax import UWAGaussSourceModel, uwa_forward_task
    from pywaveprop.uwa_utils import UWAComputationalParams

    env = get_underwater_environment_model(
        40.12, 10.20, 40.18, 10.30,
        n_range_points=40,
        resolution=fixture_cache_root["resolution"],
        svp_radius_deg=fixture_cache_root["svp_radius_deg"],
        svp_start_date=fixture_cache_root["svp_start"],
        svp_end_date=fixture_cache_root["svp_end"],
        max_svp_depth_m=fixture_cache_root["svp_max_depth"],
    )
    src = UWAGaussSourceModel(
        freq_hz=200.0,
        depth_m=30.0,
        beam_width_deg=25.0,
    )
    max_range = float(env.bathymetry.x_grid_m[-1])
    params = UWAComputationalParams(
        max_range_m=max_range,
        dx_m=50.0,
        dz_m=2.0,
    )
    field = uwa_forward_task(src, env, params)
    arr = np.asarray(field.field)
    assert np.all(np.isfinite(arr))
    assert float(np.max(np.abs(arr))) > 0.0
