"""Refractivity physics, ducting diagnostics and PE profile callables."""
from __future__ import annotations

import numpy as np
import pytest

from pywaveprop.environment import refractivity as rf


def test_refractivity_matches_smith_weintraub():
    # ITU-R P.453 reference point: P = 1013.25 hPa, T = 288.15 K, e = 10.2 hPa
    n = rf.refractivity_n(1013.25, 288.15, 10.2)
    expected = 77.6 * 1013.25 / 288.15 + 3.73e5 * 10.2 / 288.15 ** 2
    assert n == pytest.approx(expected)
    # standard atmosphere near the ground sits around 315 N-units
    assert 300.0 < n < 340.0


def test_vapour_pressure_consistency():
    t, rh = 293.15, 60.0
    e_rh = rf.vapour_pressure_from_rh(t, rh)
    assert e_rh == pytest.approx(0.6 * rf.saturation_vapour_pressure(t))
    # the same e expressed as specific humidity round-trips
    p = 1000.0
    q = 0.622 * e_rh / (p - 0.378 * e_rh)
    assert rf.vapour_pressure_from_q(q, p) == pytest.approx(e_rh, rel=1e-9)


def test_modified_refractivity_adds_earth_curvature():
    h = np.array([0.0, 100.0, 1000.0])
    m = rf.modified_refractivity_m(np.full(3, 300.0), h)
    assert np.allclose(m - 300.0, h * 1e6 / rf.EARTH_RADIUS_M)
    assert rf.M_CURVATURE_PER_M == pytest.approx(0.157, abs=1e-3)


def test_profile_from_rh_broadcasts():
    p = np.array([1000.0, 900.0, 800.0])
    t = np.array([300.0, 294.0, 288.0])
    rh = np.array([80.0, 60.0, 40.0])
    h = np.array([0.0, 900.0, 1900.0])
    n, m = rf.profile_from_rh(p, t, rh, h)
    assert n.shape == m.shape == (3,)
    assert np.all(np.diff(n) < 0)          # refractivity falls with height
    assert np.all(np.isfinite(m))


def _ducted_profile():
    """Surface duct: M falls to 100 m, then rises with the standard gradient."""
    z = np.linspace(0.0, 1000.0, 201)
    m = np.where(z < 100.0, 340.0 - 0.2 * z, 320.0 + 0.118 * (z - 100.0))
    return z, m


def test_duct_diagnostics_finds_trapping_layer():
    z, m = _ducted_profile()
    d = rf.duct_diagnostics(m, z)
    assert d["has_duct"] is True
    assert d["base_height_m"] == pytest.approx(0.0, abs=10.0)
    assert d["top_height_m"] == pytest.approx(100.0, abs=10.0)
    assert d["strength_M"] == pytest.approx(20.0, abs=2.0)
    assert d["min_gradient"] < 0.0


def test_duct_diagnostics_standard_atmosphere_has_no_duct():
    z = np.linspace(0.0, 1000.0, 101)
    d = rf.duct_diagnostics(300.0 + 0.118 * z, z)
    assert d["has_duct"] is False
    assert d["min_gradient"] > 0.0


def test_duct_diagnostics_ignores_nans_and_short_profiles():
    assert rf.duct_diagnostics(np.array([1.0, 2.0]), np.array([0.0, 1.0])) == {
        "has_duct": False}
    z, m = _ducted_profile()
    m = m.copy()
    m[::7] = np.nan
    assert rf.duct_diagnostics(m, z)["has_duct"] is True


def test_duct_diagnostics_field_matches_per_column():
    z, m = _ducted_profile()
    cube = np.stack([np.stack([m, 300.0 + 0.118 * z], axis=-1)] * 2, axis=-1)
    # cube is (nz, nlat=2, nlon=2): column 0 ducted, column 1 standard
    out = rf.duct_diagnostics_field(cube, z)
    assert out["has_duct"][0, 0] and not out["has_duct"][1, 0]
    assert out["strength"][0, 0] == pytest.approx(
        rf.duct_diagnostics(m, z)["strength_M"])
    assert np.isnan(out["strength"][1, 1])


def test_duct_diagnostics_field_rejects_mismatched_shape():
    with pytest.raises(ValueError):
        rf.duct_diagnostics_field(np.zeros((5, 2, 2)), np.zeros(4))


def test_horizontal_gradient_of_linear_field():
    lat = np.linspace(20.0, 21.0, 11)
    lon = np.linspace(50.0, 51.0, 11)
    # 1 M-unit per degree of longitude at ~20 deg N -> 1/(111.32*cos(20)) per km
    field = np.broadcast_to(lon[None, :], (lat.size, lon.size)).copy()
    g = rf.horizontal_gradient(field, lat, lon)
    expected = 1.0 / (111.32 * np.cos(np.deg2rad(lat)))
    assert np.allclose(g[:, 1:-1], expected[:, None], rtol=1e-6)


def test_range_dependent_profile_preserves_scalar_shape():
    """The non-local boundary condition probes M at single (x, z) points."""
    x = np.array([0.0, 1000.0, 2000.0])
    z = np.array([0.0, 50.0, 100.0])
    M = np.array([[300.0, 310.0, 320.0],
                  [301.0, 311.0, 321.0],
                  [302.0, 312.0, 322.0]])
    f = rf.range_dependent_M_profile(x, z, M)

    v = f(0.0, 0.0)
    assert np.isscalar(v) or np.ndim(v) == 0
    assert float(v) == pytest.approx(300.0)
    assert f(1000.0, 50.0) == pytest.approx(311.0)
    # linear in both directions
    assert f(500.0, 25.0) == pytest.approx(305.5)
    assert np.asarray(f(0.0, z)).shape == z.shape


def test_range_dependent_profile_clamps_outside_grid():
    x = np.array([0.0, 1000.0])
    z = np.array([0.0, 100.0])
    M = np.array([[300.0, 320.0], [310.0, 330.0]])
    f = rf.range_dependent_M_profile(x, z, M)
    assert f(-500.0, 0.0) == pytest.approx(300.0)
    assert f(5000.0, 0.0) == pytest.approx(310.0)
    assert f(0.0, -10.0) == pytest.approx(300.0)
    assert f(0.0, 1000.0) == pytest.approx(320.0)


def test_range_dependent_profile_validates_shape():
    with pytest.raises(ValueError):
        rf.range_dependent_M_profile(np.zeros(3), np.zeros(4), np.zeros((4, 3)))


def test_uniform_profile_is_range_independent():
    z = np.array([0.0, 100.0])
    f = rf.uniform_M_profile(z, np.array([300.0, 320.0]))
    assert f(0.0, 50.0) == pytest.approx(310.0)
    assert f(1e6, 50.0) == pytest.approx(310.0)
