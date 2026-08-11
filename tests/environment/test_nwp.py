"""GFS pipeline pieces that need no network: transects, grids, bbox cropping."""
from __future__ import annotations

import numpy as np
import pytest

from pywaveprop.environment import nwp
from pywaveprop.environment.gfs import BBox
from pywaveprop.environment.models import RefractivityTransect


LAT = np.linspace(26.5, 25.5, 5)     # descending, as GFS delivers it
LON = np.linspace(51.5, 53.5, 9)
HEIGHT = np.linspace(0.0, 3000.0, 121)


def _synthetic_cube():
    """M(z, lat, lon) with an elevated duct whose depth grows with longitude."""
    duct_top = 100.0 + 20.0 * np.arange(LON.size)          # 100..260 m
    z = HEIGHT[:, None, None]
    top = duct_top[None, None, :]
    standard = 320.0 + 0.118 * z
    # inside the trapping layer M falls at -0.2 M/m, above it recovers
    trapped = 320.0 - 0.2 * z
    M = np.where(z < top, trapped, 320.0 - 0.2 * top + 0.118 * (z - top))
    M = np.broadcast_to(M, (HEIGHT.size, LAT.size, LON.size)).copy()
    assert standard.shape[0] == HEIGHT.size
    return M


def _synthetic_bulk(land_east: bool = False):
    shape = (LAT.size, LON.size)
    lsm = np.zeros(shape)
    if land_east:
        lsm[:, -2:] = 1.0
    return {
        "sst": np.full(shape, 303.0),
        "t2m": np.full(shape, 304.5),
        "rh2": np.full(shape, 65.0),
        "wind10": np.full(shape, 6.0),
        "sp": np.full(shape, 1005.0),
        "lsm": lsm,
    }


def test_haversine_matches_known_distance():
    # one degree of latitude is ~111.2 km
    assert nwp.haversine_m(25.0, 51.0, 26.0, 51.0) == pytest.approx(111195.0, rel=1e-3)
    assert nwp.haversine_m(25.0, 51.0, 25.0, 51.0) == 0.0


def test_surface_layer_height_grid_is_fine_below_the_splice():
    z = nwp.surface_layer_height_grid(1200.0)
    assert z[0] == 0.0 and z[-1] == pytest.approx(1200.0)
    assert np.all(np.diff(z) > 0)
    below = z[z < nwp.Z_SPLICE]
    assert np.allclose(np.diff(below), 1.0)
    above = z[z >= nwp.Z_SPLICE]
    assert np.allclose(np.diff(above), 20.0)


def test_transect_samples_path_and_keeps_duct_structure():
    tr = nwp.refractivity_transect(LAT, LON, HEIGHT, _synthetic_cube(),
                                   p1=(26.4, 51.9), p2=(25.6, 53.1),
                                   bulk=None, n_cols=11, z_top=600.0,
                                   splice_evaporation=False)
    assert isinstance(tr, RefractivityTransect)
    assert tr.M.shape == (11, tr.z_m.size)
    assert tr.x_m[0] == 0.0
    assert tr.length_m == pytest.approx(
        nwp.haversine_m(26.4, 51.9, 25.6, 53.1))
    assert np.all(np.isfinite(tr.M))
    # the synthetic duct top deepens eastward along the path
    assert np.all(np.isfinite(tr.duct_top))
    assert tr.duct_top[-1] > tr.duct_top[0]


def test_transect_splices_evaporation_duct_over_sea_only():
    cube = _synthetic_cube()
    bulk = _synthetic_bulk(land_east=True)
    plain = nwp.refractivity_transect(LAT, LON, HEIGHT, cube,
                                      (26.4, 51.9), (25.6, 53.4), bulk=bulk,
                                      n_cols=9, z_top=600.0,
                                      splice_evaporation=False)
    spliced = nwp.refractivity_transect(LAT, LON, HEIGHT, cube,
                                        (26.4, 51.9), (25.6, 53.4), bulk=bulk,
                                        n_cols=9, z_top=600.0)
    sea = spliced.lsm_pts < 0.5
    land = ~sea
    assert sea.any() and land.any()
    below = spliced.z_m < nwp.Z_SPLICE
    # sea columns gain surface-layer structure, land columns are untouched
    assert np.abs(spliced.M[sea][:, below] - plain.M[sea][:, below]).max() > 1.0
    assert np.allclose(spliced.M[land], plain.M[land])
    # the splice has fully blended back into the NWP profile above z_blend
    above = spliced.z_m > nwp.Z_BLEND
    assert np.allclose(spliced.M[:, above], plain.M[:, above])
    # surface anchor is preserved
    assert np.allclose(spliced.M[:, 0], plain.M[:, 0])
    assert np.all(spliced.edh_pts[sea] > 0.0)


def test_transect_deficit_cap_bounds_the_surface_layer():
    cube = _synthetic_cube()
    bulk = _synthetic_bulk()
    tight = nwp.refractivity_transect(LAT, LON, HEIGHT, cube, (26.4, 51.9),
                                      (25.6, 53.1), bulk=bulk, n_cols=5,
                                      z_top=600.0, deficit_cap=2.0)
    below = tight.z_m <= nwp.Z_SPLICE
    dev = tight.M[:, below] - tight.M[:, [0]]
    assert np.nanmax(np.abs(dev)) <= 2.0 + 1e-6


def test_transect_M_profile_feeds_the_pe_solver():
    tr = nwp.refractivity_transect(LAT, LON, HEIGHT, _synthetic_cube(),
                                   (26.4, 51.9), (25.6, 53.1), n_cols=7,
                                   z_top=600.0, splice_evaporation=False)
    f = tr.M_profile()
    # scalar in -> scalar out (required by the non-local boundary condition)
    assert np.ndim(f(0.0, 10.0)) == 0
    assert f(0.0, 0.0) == pytest.approx(tr.M[0, 0])
    assert np.asarray(f(tr.length_m, tr.z_m)).shape == tr.z_m.shape
    g = tr.uniform_M_profile()
    assert g(0.0, 25.0) == pytest.approx(g(tr.length_m, 25.0))


def test_transect_round_trips_through_npz(tmp_path):
    tr = nwp.refractivity_transect(LAT, LON, HEIGHT, _synthetic_cube(),
                                   (26.4, 51.9), (25.6, 53.1), n_cols=5,
                                   z_top=400.0, splice_evaporation=False,
                                   name="synthetic", cycle="2026-07-15T12:00Z")
    path = nwp.save_transect(tr, str(tmp_path / "t.npz"))
    back = nwp.load_transect(path)
    assert np.allclose(back.M, tr.M)
    assert np.allclose(back.x_m, tr.x_m)
    assert back.name == "synthetic" and back.cycle == "2026-07-15T12:00Z"


def test_subset_bbox_crops_descending_lat_and_0_360_lon():
    xr = pytest.importorskip("xarray")
    lat = np.linspace(40.0, -40.0, 81)          # descending
    lon = np.arange(0.0, 360.0, 0.25)
    ds = xr.Dataset(
        {"v": (("latitude", "longitude"), np.zeros((lat.size, lon.size)))},
        coords={"latitude": lat, "longitude": lon},
    )
    out = nwp.subset_bbox(ds, BBox(48.0, 57.0, 23.0, 30.5))
    assert float(out.latitude.max()) <= 30.5 and float(out.latitude.min()) >= 23.0
    assert float(out.longitude.min()) >= 48.0 and float(out.longitude.max()) <= 57.0
    assert out.latitude.values[0] > out.latitude.values[-1]  # order preserved


def test_subset_bbox_handles_a_wrapping_box():
    xr = pytest.importorskip("xarray")
    lat = np.linspace(40.0, -40.0, 81)
    lon = np.arange(0.0, 360.0, 0.25)
    ds = xr.Dataset(
        {"v": (("latitude", "longitude"), np.zeros((lat.size, lon.size)))},
        coords={"latitude": lat, "longitude": lon},
    )
    out = nwp.subset_bbox(ds, BBox(-10.0, 10.0, 0.0, 20.0))
    lons = out.longitude.values
    assert np.all(np.diff(lons) > 0)            # monotonic after stitching
    assert lons.min() == pytest.approx(-10.0) and lons.max() == pytest.approx(10.0)


def test_global_grid_detection():
    class _DS:
        def __init__(self, n):
            self.sizes = {"longitude": n, "latitude": 721}

    assert nwp._is_global_grid(_DS(1440))
    assert not nwp._is_global_grid(_DS(37))
