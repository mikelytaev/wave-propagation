"""GFS request building, cycle selection and archive index handling (offline)."""
from __future__ import annotations

import datetime as dt

import pytest

from pywaveprop.environment import gfs


def test_bbox_normalises_to_gfs_grid():
    b = gfs.BBox(-128.0, -120.0, 32.0, 38.0).to_0_360()
    assert (b.lon_min, b.lon_max) == (232.0, 240.0)
    assert not gfs.BBox(-128.0, -120.0, 32.0, 38.0).crosses_prime_meridian()
    assert gfs.BBox(-10.0, 10.0, 40.0, 50.0).crosses_prime_meridian()


@pytest.mark.parametrize("now, expected", [
    (dt.datetime(2026, 7, 15, 18, 0), dt.datetime(2026, 7, 15, 12)),
    (dt.datetime(2026, 7, 15, 16, 59), dt.datetime(2026, 7, 15, 6)),
    (dt.datetime(2026, 7, 15, 3, 0), dt.datetime(2026, 7, 14, 18)),
])
def test_latest_available_cycle_accounts_for_publication_latency(now, expected):
    assert gfs.latest_available_cycle(now, latency_hours=5.0) == expected


def test_build_url_carries_subregion_variables_and_levels():
    req = gfs.GFSRequest(bbox=gfs.BBox(48.0, 57.0, 23.0, 30.5), forecast_hour=6)
    url = gfs.build_url(req, dt.datetime(2026, 7, 15, 12))
    assert "gfs.t12z.pgrb2.0p25.f006" in url
    assert "dir=/gfs.20260715/12/atmos" in url
    assert "leftlon=48.0000" in url and "rightlon=57.0000" in url
    assert "toplat=30.5000" in url and "bottomlat=23.0000" in url
    for var in ("HGT", "TMP", "RH", "PRES"):
        assert f"var_{var}=on" in url
    assert "lev_1000_mb=on" in url and "lev_2_m_above_ground=on" in url
    assert "var_UGRD=on" not in url


def test_build_url_wind_option_adds_bulk_fields():
    req = gfs.GFSRequest(bbox=gfs.BBox(48.0, 57.0, 23.0, 30.5),
                         levels_hpa=[], variables=["TMP", "RH"],
                         include_wind10m=True)
    url = gfs.build_url(req, dt.datetime(2026, 7, 15, 12))
    assert "var_UGRD=on" in url and "var_VGRD=on" in url and "var_LAND=on" in url
    assert "lev_10_m_above_ground=on" in url


def test_archive_url_points_at_the_aws_mirror():
    req = gfs.GFSRequest(bbox=gfs.BBox(48.0, 57.0, 23.0, 30.5), forecast_hour=24)
    url = gfs.archive_url(req, dt.datetime(2026, 7, 15, 0))
    assert url.endswith("gfs.20260715/00/atmos/gfs.t00z.pgrb2.0p25.f024")
    assert url.startswith(gfs.AWS_ARCHIVE_URL)


def test_request_signature_tracks_content_not_cycle():
    box = gfs.BBox(48.0, 57.0, 23.0, 30.5)
    a = gfs.GFSRequest(bbox=box, cycle=dt.datetime(2026, 7, 15, 0))
    b = gfs.GFSRequest(bbox=box, cycle=dt.datetime(2026, 7, 16, 12))
    assert a.signature() == b.signature()
    c = gfs.GFSRequest(bbox=box, levels_hpa=[1000, 900])
    assert c.signature() != a.signature()


def test_cache_path_encodes_cycle_and_forecast_hour(tmp_path):
    req = gfs.GFSRequest(bbox=gfs.BBox(48.0, 57.0, 23.0, 30.5), forecast_hour=12)
    p = gfs.cache_path(req, dt.datetime(2026, 7, 15, 6), cache_dir=str(tmp_path))
    assert p.startswith(str(tmp_path))
    assert "2026071506" in p and "f012" in p and p.endswith(".grib2")


IDX = """1:0:d=2026071512:PRMSL:mean sea level:anl:
2:1000:d=2026071512:HGT:1000 mb:anl:
3:2500:d=2026071512:TMP:1000 mb:anl:
4:4000:d=2026071512:TMP:2 m above ground:anl:
5:5500:d=2026071512:UGRD:10 m above ground:anl:
6:7000:d=2026071512:TMP:850 mb:6 hour fcst:
"""


def test_parse_idx_records_byte_ranges():
    recs = gfs.parse_idx(IDX)
    assert len(recs) == 6
    assert recs[0]["var"] == "PRMSL" and recs[0]["start"] == 0
    assert recs[0]["end"] == 1000
    assert recs[1]["end"] - recs[1]["start"] == 1500
    assert recs[-1]["end"] is None  # last message runs to EOF


def test_select_messages_matches_var_and_level_ignoring_fcst_suffix():
    req = gfs.GFSRequest(bbox=gfs.BBox(48.0, 57.0, 23.0, 30.5),
                         levels_hpa=[1000, 850], variables=["HGT", "TMP"])
    picked = gfs.select_messages(gfs.parse_idx(IDX), req)
    got = {(r["var"], r["level"]) for r in picked}
    assert got == {("HGT", "1000 mb"), ("TMP", "1000 mb"),
                   ("TMP", "2 m above ground"), ("TMP", "850 mb")}
    # PRMSL is never requested; the 6-hour-fcst suffix must not block a match
    assert all(r["var"] != "PRMSL" for r in picked)


def test_backend_choice_follows_nomads_retention():
    recent = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None) - dt.timedelta(days=1)
    old = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None) - dt.timedelta(days=60)
    assert gfs._backends_for(recent, "auto") == ("nomads", "archive")
    assert gfs._backends_for(old, "auto") == ("archive",)
    assert gfs._backends_for(old, "nomads") == ("nomads",)


def test_download_rejects_wrapping_bbox_on_nomads():
    req = gfs.GFSRequest(bbox=gfs.BBox(-10.0, 10.0, 40.0, 50.0),
                         cycle=dt.datetime.now() - dt.timedelta(days=1))
    with pytest.raises(ValueError, match="0/360"):
        gfs.download(req, "/tmp/never-written.grib2", source="nomads")


def test_download_cached_reuses_existing_file(tmp_path, monkeypatch):
    req = gfs.GFSRequest(bbox=gfs.BBox(48.0, 57.0, 23.0, 30.5),
                         cycle=dt.datetime(2026, 7, 15, 12))
    target = gfs.cache_path(req, req.cycle, cache_dir=str(tmp_path))
    with open(target, "wb") as fh:
        fh.write(b"GRIB")

    def fail(*a, **kw):  # pragma: no cover - must not be reached
        raise AssertionError("download() called despite a warm cache")

    monkeypatch.setattr(gfs, "download", fail)
    path, cycle = gfs.download_cached(req, max_cycle_fallbacks=0,
                                      cache_dir=str(tmp_path))
    assert path == target and cycle == req.cycle


def test_download_cached_populates_the_cache(tmp_path, monkeypatch):
    req = gfs.GFSRequest(bbox=gfs.BBox(48.0, 57.0, 23.0, 30.5),
                         cycle=dt.datetime(2026, 7, 15, 12))

    def fake_download(request, out_path, **kw):
        with open(out_path, "wb") as fh:
            fh.write(b"GRIB")
        return request.cycle

    monkeypatch.setattr(gfs, "download", fake_download)
    path, cycle = gfs.download_cached(req, max_cycle_fallbacks=0,
                                      cache_dir=str(tmp_path))
    assert path == gfs.cache_path(req, req.cycle, cache_dir=str(tmp_path))
    assert open(path, "rb").read() == b"GRIB"
    assert cycle == req.cycle
