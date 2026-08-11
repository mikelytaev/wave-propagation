"""NOAA GFS access: NOMADS GRIB-filter subsets and the AWS long-term archive.

Two back ends deliver the same GRIB2 message set:

``nomads``
    The ``filter_gfs_0p25`` CGI byte-range subsets a GFS file down to only the
    variables, pressure levels and lat/lon box requested, so each fetch is a few
    hundred kB instead of hundreds of MB. NOMADS keeps roughly the last 10 days.

``archive``
    The AWS Open Data mirror (``noaa-gfs-bdp-pds``) retains years but offers no
    subsetting CGI, so we parse the ``.idx`` sidecar and pull the wanted GRIB
    messages with HTTP range requests. Messages are global, so the spatial
    subset is applied after loading (see
    :func:`pywaveprop.environment.nwp.subset_bbox`), and a fetch costs tens of MB.

``auto`` (the default) uses NOMADS for recent cycles and falls back to the
archive when NOMADS no longer has the cycle.

GFS runs 4x/day at 00/06/12/18Z and lands on NOMADS roughly 3.5-5 h after the
cycle time. :func:`latest_available_cycle` accounts for that latency.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import logging
import os
import time
from dataclasses import dataclass, field

from . import _paths

logger = logging.getLogger(__name__)

NOMADS_FILTER_URL = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"

#: AWS Open Data long-term GFS archive (no subsetting CGI; ``.idx`` + ranges).
AWS_ARCHIVE_URL = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"

#: How long NOMADS keeps a cycle available. Older cycles need the archive.
NOMADS_RETENTION_DAYS = 10

#: Default isobaric levels [hPa] to request. Dense in the lower troposphere,
#: which is where surface and elevated ducts live. These are all present in the
#: GFS 0.25-degree pgrb2 product for HGT/TMP/RH.
DEFAULT_LEVELS_HPA = [
    1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600, 550,
    500, 450, 400, 350, 300, 250, 200, 150, 100,
]

#: Fields needed to build a refractivity profile on pressure levels.
DEFAULT_VARS = ["HGT", "TMP", "RH"]

GFS_CYCLES = (0, 6, 12, 18)


@dataclass
class BBox:
    """Geographic bounding box in degrees (lon in -180..180 or 0..360)."""

    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float

    def to_0_360(self) -> "BBox":
        """Return a copy with longitudes normalised to the GFS 0..360 grid."""
        def norm(x: float) -> float:
            return x % 360.0

        return BBox(norm(self.lon_min), norm(self.lon_max), self.lat_min, self.lat_max)

    def crosses_prime_meridian(self) -> bool:
        b = self.to_0_360()
        return b.lon_min > b.lon_max


@dataclass
class GFSRequest:
    """A single GFS subset request."""

    bbox: BBox
    forecast_hour: int = 0
    cycle: dt.datetime | None = None  # UTC; if None -> latest available
    levels_hpa: list[int] = field(default_factory=lambda: list(DEFAULT_LEVELS_HPA))
    variables: list[str] = field(default_factory=lambda: list(DEFAULT_VARS))
    resolution: str = "0p25"
    #: Also request surface pressure/orography and 2 m T/RH, so each column can
    #: be anchored down to the real surface (fills the sub-1000 hPa gap that is
    #: critical for surface / evaporation ducts).
    include_surface: bool = True
    #: Also request 10 m wind (UGRD/VGRD) and the land-sea mask. Needed by the
    #: evaporation-duct surface-layer model.
    include_wind10m: bool = False

    def variable_keys(self) -> set[str]:
        """GRIB short names selected by this request."""
        keys = set(self.variables)
        if self.include_surface:
            keys.add("PRES")
        if self.include_wind10m:
            keys.update(("UGRD", "VGRD", "LAND"))
        return keys

    def level_keys(self) -> set[str]:
        """GRIB ``.idx`` level strings selected by this request."""
        keys = {f"{lev} mb" for lev in self.levels_hpa}
        if self.include_surface:
            keys.update(("surface", "2 m above ground"))
        if self.include_wind10m:
            keys.add("10 m above ground")
        return keys

    def signature(self) -> str:
        """Stable short hash of everything that affects the downloaded bytes."""
        b = self.bbox.to_0_360()
        payload = "|".join(str(x) for x in (
            f"{b.lon_min:.4f}", f"{b.lon_max:.4f}",
            f"{self.bbox.lat_min:.4f}", f"{self.bbox.lat_max:.4f}",
            self.resolution, sorted(self.levels_hpa), sorted(self.variables),
            self.include_surface, self.include_wind10m,
        ))
        return hashlib.sha1(payload.encode()).hexdigest()[:10]


def latest_available_cycle(
    now_utc: dt.datetime | None = None, latency_hours: float = 5.0
) -> dt.datetime:
    """Most recent GFS cycle expected to be published on NOMADS.

    Parameters
    ----------
    now_utc : datetime, optional
        Current UTC time (defaults to now).
    latency_hours : float
        How long after a cycle time to assume its files are available.
    """
    now = now_utc or dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    ref = now - dt.timedelta(hours=latency_hours)
    hour = max(c for c in GFS_CYCLES if c <= ref.hour) if ref.hour >= GFS_CYCLES[0] else 18
    day = ref.date()
    if ref.hour < GFS_CYCLES[0]:
        day = day - dt.timedelta(days=1)
    return dt.datetime(day.year, day.month, day.day, hour)


def build_url(req: GFSRequest, cycle: dt.datetime) -> str:
    """Build the NOMADS filter URL for a request at a given cycle."""
    ymd = cycle.strftime("%Y%m%d")
    cc = cycle.strftime("%H")
    fff = f"{req.forecast_hour:03d}"
    bbox = req.bbox.to_0_360()

    params = [
        ("file", f"gfs.t{cc}z.pgrb2.{req.resolution}.f{fff}"),
        ("dir", f"/gfs.{ymd}/{cc}/atmos"),
        ("subregion", ""),
        ("leftlon", f"{bbox.lon_min:.4f}"),
        ("rightlon", f"{bbox.lon_max:.4f}"),
        ("toplat", f"{req.bbox.lat_max:.4f}"),
        ("bottomlat", f"{req.bbox.lat_min:.4f}"),
    ]
    for v in req.variables:
        params.append((f"var_{v}", "on"))
    for lev in req.levels_hpa:
        params.append((f"lev_{lev}_mb", "on"))

    if req.include_surface:
        # PRES/HGT at surface -> surface pressure + orography;
        # TMP/RH at 2 m above ground -> near-surface anchor point.
        params.append(("var_PRES", "on"))
        params.append(("lev_surface", "on"))
        params.append(("lev_2_m_above_ground", "on"))

    if req.include_wind10m:
        # 10 m wind for the surface-layer flux calc; LAND mask to drop land;
        # TMP at surface gives skin temperature ~ SST over the sea.
        params.append(("var_UGRD", "on"))
        params.append(("var_VGRD", "on"))
        params.append(("var_LAND", "on"))
        params.append(("lev_10_m_above_ground", "on"))

    query = "&".join(f"{k}={v}" for k, v in params)
    return f"{NOMADS_FILTER_URL}?{query}"


def archive_url(req: GFSRequest, cycle: dt.datetime) -> str:
    """URL of the full GFS GRIB2 file in the AWS archive (``.idx`` sidecar: +'.idx')."""
    ymd = cycle.strftime("%Y%m%d")
    cc = cycle.strftime("%H")
    fff = f"{req.forecast_hour:03d}"
    return (f"{AWS_ARCHIVE_URL}/gfs.{ymd}/{cc}/atmos/"
            f"gfs.t{cc}z.pgrb2.{req.resolution}.f{fff}")


def cache_path(req: GFSRequest, cycle: dt.datetime,
               cache_dir: str | None = None) -> str:
    """Deterministic on-disk location of the GRIB subset for ``(req, cycle)``."""
    root = cache_dir or str(_paths.GFS_CACHE_DIR)
    name = (f"gfs_{req.resolution}_{cycle:%Y%m%d%H}_f{req.forecast_hour:03d}_"
            f"{req.signature()}.grib2")
    return os.path.join(root, name)


def download(
    req: GFSRequest,
    out_path: str,
    max_cycle_fallbacks: int = 3,
    timeout: float = 120.0,
    retries: int = 3,
    source: str = "auto",
) -> dt.datetime:
    """Download a GFS subset to ``out_path``.

    If ``req.cycle`` is None the latest expected cycle is tried first, falling
    back to progressively older cycles (up to ``max_cycle_fallbacks``) if the
    files are not yet published.

    ``source`` selects the back end: ``"nomads"``, ``"archive"`` or ``"auto"``
    (NOMADS for cycles inside its retention window, the AWS archive otherwise
    and whenever NOMADS reports the cycle missing).

    Returns the cycle datetime actually downloaded.
    """
    if source not in ("auto", "nomads", "archive"):
        raise ValueError(f"unknown source {source!r}")

    start_cycle = req.cycle or latest_available_cycle()
    if ("nomads" in _backends_for(start_cycle, source)
            and req.bbox.crosses_prime_meridian()):
        raise ValueError(
            "bbox crosses the 0/360 meridian; NOMADS subregion cannot wrap. "
            "Split into two requests, shift coordinates, or use "
            "source='archive' (global messages, cropped after loading)."
        )

    last_err: Exception | None = None

    for i in range(max_cycle_fallbacks + 1):
        cycle = start_cycle - dt.timedelta(hours=6 * i)
        for backend in _backends_for(cycle, source):
            try:
                if backend == "nomads":
                    _http_get_to_file(build_url(req, cycle), out_path,
                                      timeout=timeout, retries=retries)
                else:
                    _archive_get_to_file(req, cycle, out_path,
                                         timeout=timeout, retries=retries)
                return cycle
            except FileNotFoundError as e:
                last_err = e
                continue

    raise RuntimeError(
        f"No GFS data found for the last {max_cycle_fallbacks + 1} cycles "
        f"(newest tried: {start_cycle:%Y-%m-%d %HZ}). Last error: {last_err}"
    )


def download_cached(
    req: GFSRequest,
    max_cycle_fallbacks: int = 3,
    timeout: float = 120.0,
    retries: int = 3,
    source: str = "auto",
    cache_dir: str | None = None,
    use_cache: bool = True,
) -> tuple[str, dt.datetime]:
    """Like :func:`download`, but reuse (and populate) the local GRIB cache.

    Returns ``(grib_path, cycle)``. Cached files live in
    ``~/.cache/pywaveprop/gfs`` so repeated analyses of the same cycle -- and
    re-running a published pipeline offline -- cost no network traffic.
    """
    root = cache_dir or str(_paths.GFS_CACHE_DIR)
    os.makedirs(root, exist_ok=True)

    start_cycle = req.cycle or latest_available_cycle()
    if use_cache:
        for i in range(max_cycle_fallbacks + 1):
            cycle = start_cycle - dt.timedelta(hours=6 * i)
            path = cache_path(req, cycle, cache_dir=root)
            if os.path.exists(path) and os.path.getsize(path) > 0:
                logger.debug("GFS cache hit: %s", path)
                return path, cycle

    tmp = cache_path(req, start_cycle, cache_dir=root) + ".dl"
    cycle = download(req, tmp, max_cycle_fallbacks=max_cycle_fallbacks,
                     timeout=timeout, retries=retries, source=source)
    final = cache_path(req, cycle, cache_dir=root)
    os.replace(tmp, final)
    return final, cycle


def _backends_for(cycle: dt.datetime, source: str) -> tuple[str, ...]:
    """Back ends to try, in order, for a given cycle."""
    if source != "auto":
        return (source,)
    age = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None) - cycle
    if age > dt.timedelta(days=NOMADS_RETENTION_DAYS):
        return ("archive",)
    return ("nomads", "archive")


def _http_get_to_file(url: str, out_path: str, timeout: float, retries: int) -> None:
    """GET ``url`` to ``out_path``; raise FileNotFoundError on a 404/empty body."""
    import requests

    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            with requests.get(url, timeout=timeout, stream=True) as r:
                # NOMADS returns 404 (sometimes a short text/html body) when a
                # cycle/level/var is not available.
                ctype = r.headers.get("Content-Type", "")
                if r.status_code in (403, 404) or "html" in ctype.lower():
                    raise FileNotFoundError(f"not available: {url} ({r.status_code})")
                r.raise_for_status()

                tmp = out_path + ".part"
                total = 0
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        if chunk:
                            fh.write(chunk)
                            total += len(chunk)
                if total == 0:
                    os.remove(tmp)
                    raise FileNotFoundError(f"empty response: {url}")
                os.replace(tmp, out_path)
                return
        except FileNotFoundError:
            raise
        except requests.RequestException as e:  # transient network errors
            last_exc = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"download failed after {retries} retries: {url} ({last_exc})")


def parse_idx(text: str) -> list[dict]:
    """Parse a GRIB2 ``.idx`` sidecar into per-message records.

    Each line looks like ``12:5701234:d=2026071512:TMP:2 m above ground:anl:``.
    The returned records carry ``num``, ``start``, ``var``, ``level`` and
    ``end`` (exclusive; None for the final message, which runs to EOF).
    """
    records: list[dict] = []
    for line in text.splitlines():
        parts = line.split(":")
        if len(parts) < 6 or not parts[0].strip():
            continue
        try:
            num, start = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        records.append({"num": num, "start": start, "var": parts[3],
                        "level": parts[4], "fcst": parts[5], "end": None})
    for a, b in zip(records, records[1:]):
        a["end"] = b["start"]
    return records


def select_messages(records: list[dict], req: GFSRequest) -> list[dict]:
    """Records matching the request's variable x level selection.

    The ``anl`` / ``N hour fcst`` suffix is deliberately ignored: it differs
    between analysis and forecast files while the (variable, level) pair does
    not.
    """
    want_vars = req.variable_keys()
    want_levels = req.level_keys()
    return [r for r in records
            if r["var"] in want_vars and r["level"] in want_levels]


def _archive_get_to_file(req: GFSRequest, cycle: dt.datetime, out_path: str,
                         timeout: float, retries: int) -> None:
    """Byte-range subset the AWS archive file into a single GRIB2 at ``out_path``.

    The archive has no subsetting service, so we read the ``.idx``, pick the
    messages the request asks for, and concatenate them in file order. The
    messages cover the whole globe; crop them after loading.
    """
    import requests

    url = archive_url(req, cycle)
    try:
        idx = requests.get(url + ".idx", timeout=timeout)
    except requests.RequestException as e:
        raise RuntimeError(f"archive index request failed: {url}.idx ({e})") from e
    if idx.status_code in (403, 404):
        raise FileNotFoundError(f"not available: {url}.idx ({idx.status_code})")
    idx.raise_for_status()

    messages = select_messages(parse_idx(idx.text), req)
    if not messages:
        raise FileNotFoundError(f"no matching GRIB messages in {url}.idx")

    logger.info("GFS archive: fetching %d messages from %s", len(messages), url)
    tmp = out_path + ".part"
    total = 0
    with open(tmp, "wb") as fh:
        for msg in messages:
            end = "" if msg["end"] is None else str(msg["end"] - 1)
            headers = {"Range": f"bytes={msg['start']}-{end}"}
            chunk = _range_get(url, headers, timeout=timeout, retries=retries)
            fh.write(chunk)
            total += len(chunk)
    if total == 0:
        os.remove(tmp)
        raise FileNotFoundError(f"empty archive response: {url}")
    os.replace(tmp, out_path)


def _range_get(url: str, headers: dict, timeout: float, retries: int) -> bytes:
    import requests

    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            if r.status_code in (403, 404):
                raise FileNotFoundError(f"not available: {url} ({r.status_code})")
            r.raise_for_status()
            return r.content
        except FileNotFoundError:
            raise
        except requests.RequestException as e:
            last_exc = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"range request failed after {retries} retries: "
                       f"{url} {headers} ({last_exc})")
