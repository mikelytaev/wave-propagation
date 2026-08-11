"""Monin-Obukhov surface-layer refractivity (evaporation duct / land profile)."""
from __future__ import annotations

import numpy as np
import pytest

from pywaveprop.environment import evaporation as ev


def test_evaporation_duct_is_trapping_in_typical_marine_conditions():
    # warm sea, slightly warmer air, moderate humidity -> classic evap duct
    r = ev.evaporation_duct(sst_k=303.0, air_temp_k=304.0, rh_pct=70.0,
                            wind10_ms=6.0, pressure_hpa=1005.0)
    assert 0.0 < r["edh_m"] <= ev.DEFAULT_Z_MAX
    assert r["deficit_M"] > 0.0
    z, M = r["z"], r["M"]
    assert z[0] > 0.0 and z[-1] == pytest.approx(ev.DEFAULT_Z_MAX)
    # M decreases up to the duct height, then rises again
    i = int(np.argmin(M))
    assert z[i] == pytest.approx(r["edh_m"])
    assert M[-1] > M[i]


def test_evaporation_duct_saturated_air_gives_no_trapping():
    r = ev.evaporation_duct(sst_k=290.0, air_temp_k=290.0, rh_pct=100.0,
                            wind10_ms=5.0, pressure_hpa=1013.0)
    assert r["edh_m"] == 0.0
    assert r["deficit_M"] == 0.0


def test_evaporation_duct_deepens_as_air_dries():
    kw = dict(sst_k=301.0, air_temp_k=302.0, wind10_ms=7.0, pressure_hpa=1008.0)
    humid = ev.evaporation_duct(rh_pct=90.0, **kw)
    dry = ev.evaporation_duct(rh_pct=50.0, **kw)
    assert dry["deficit_M"] > humid["deficit_M"]


def test_evaporation_duct_respects_z_max():
    r = ev.evaporation_duct(303.0, 304.0, 70.0, 6.0, 1005.0, z_max=15.0,
                            n_levels=50)
    assert r["z"][-1] == pytest.approx(15.0)
    assert r["edh_m"] <= 15.0


def test_evaporation_duct_propagates_missing_data():
    r = ev.evaporation_duct(np.nan, 304.0, 70.0, 6.0, 1005.0)
    assert np.isnan(r["edh_m"]) and r["z"] is None


def test_evaporation_duct_field_skips_land():
    shape = (2, 3)
    sst = np.full(shape, 303.0)
    t2m = np.full(shape, 304.0)
    rh = np.full(shape, 70.0)
    wind = np.full(shape, 6.0)
    sp = np.full(shape, 1005.0)
    lsm = np.zeros(shape)
    lsm[1, 2] = 1.0
    edh, deficit = ev.evaporation_duct_field(sst, t2m, rh, wind, sp, lsm,
                                             n_levels=60)
    assert np.isnan(edh[1, 2]) and np.isnan(deficit[1, 2])
    assert np.isfinite(edh[0, 0]) and edh[0, 0] > 0.0
    # uniform forcing -> identical sea columns
    assert np.allclose(edh[lsm < 0.5], edh[0, 0])


def test_land_profile_inversion_traps_and_superadiabatic_does_not():
    kw = dict(air_temp_k=305.0, rh_pct=25.0, wind10_ms=4.0, pressure_hpa=990.0)
    inversion = ev.land_surface_profile(skin_temp_k=300.0, **kw)   # cool skin
    superadiabatic = ev.land_surface_profile(skin_temp_k=320.0, **kw)  # hot skin
    grad_inv = np.gradient(inversion["M"], inversion["z"])
    grad_sup = np.gradient(superadiabatic["M"], superadiabatic["z"])
    assert grad_inv[:50].min() < grad_sup[:50].min()
    assert grad_sup.min() > 0.0  # sub-refraction: M rises everywhere
    assert np.all(np.isfinite(inversion["N"]))
