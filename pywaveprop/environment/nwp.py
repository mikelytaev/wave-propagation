"""GFS -> modified-refractivity pipeline: cubes, bulk fields and PE transects.

The end product is a cube ``M(height, latitude, longitude)`` -- modified
refractivity on a uniform height grid -- plus the range-dependent ``M(x, z)``
transects the parabolic-equation solvers consume:

    from pywaveprop.environment.gfs import BBox
    from pywaveprop.environment.nwp import (fetch_refractivity_cube,
                                            fetch_surface_bulk,
                                            refractivity_transect_from_cube)

    cube = fetch_refractivity_cube(BBox(48, 57, 23, 30.5))
    bulk = fetch_surface_bulk(BBox(48, 57, 23, 30.5))
    tr = refractivity_transect_from_cube(cube, (26.4, 51.9), (25.6, 53.1),
                                         bulk=bulk)
    env.M_profile = tr.M_profile()      # ready for rwp_ss_pade

GFS resolves the troposphere on pressure levels; the evaporation duct in the
lowest tens of metres over the sea is below that resolution, so
:func:`refractivity_transect` splices the Monin-Obukhov surface-layer profile
from :mod:`pywaveprop.environment.evaporation` underneath the NWP column.

``xarray`` (with the ``cfgrib`` engine) is required for the GRIB-reading parts
and imported lazily, so the rest of the package works without it.
"""
from __future__ import annotations

import datetime as dt
import os

import numpy as np

from . import refractivity as rf
from .evaporation import DEFAULT_Z_MAX, evaporation_duct
from .gfs import BBox, GFSRequest, download_cached
from .models import RefractivityTransect

#: Evaporation surface-layer ceiling [m]: MOST is only valid this far up.
Z_SPLICE = DEFAULT_Z_MAX
#: Height [m] by which the spliced surface layer has blended back into GFS.
Z_BLEND = 120.0
#: Physical guard on the sub-``Z_SPLICE`` M-deficit [M-units]. MOST over-predicts
#: in the stable regime (Pastore 2021; Franklin 2022), so clip rather than stack.
DEFICIT_CAP = 50.0


# --------------------------------------------------------------------------
# GRIB readers
# --------------------------------------------------------------------------

def _open_grib(grib_path: str, keys: dict):
    import xarray as xr

    return xr.open_dataset(
        grib_path,
        engine="cfgrib",
        backend_kwargs={"filter_by_keys": keys, "indexpath": ""},
    )


def load_pressure_level_fields(grib_path: str):
    """Open a GFS pgrb2 subset and return HGT/TMP/RH on isobaric levels.

    Returns a dataset with variables ``gh`` [m], ``t`` [K], ``r`` [%] on
    coordinates ``(isobaricInhPa, latitude, longitude)``, ordered so that
    ``isobaricInhPa`` decreases with height (surface first).
    """
    ds = _open_grib(grib_path, {"typeOfLevel": "isobaricInhPa"})
    # Highest pressure (near surface) first.
    return ds.sortby("isobaricInhPa", ascending=False)


def load_surface_layer(grib_path: str):
    """Load surface + 2 m fields and return an anchor layer ``(N, M, z)``.

    Uses surface pressure (``sp``), orography (``orog``), 2 m temperature
    (``t2m``) and 2 m relative humidity (``r2``) to build one near-surface point
    per column at the surface height. Returns None if the fields are absent.
    """
    import xarray as xr

    try:
        sfc = _open_grib(grib_path, {"typeOfLevel": "surface"})       # sp, orog
        agl = _open_grib(grib_path, {"typeOfLevel": "heightAboveGround"})
    except Exception:
        return None
    if "sp" not in sfc or "t2m" not in agl or "r2" not in agl:
        return None

    p_sfc = sfc["sp"] / 100.0            # Pa -> hPa
    orog = sfc["orog"]                   # m (0 over ocean)
    t2m = agl["t2m"]
    r2 = agl["r2"]

    e = xr.apply_ufunc(rf.vapour_pressure_from_rh, t2m, r2)
    n = xr.apply_ufunc(rf.refractivity_n, p_sfc, t2m, e)
    # Anchor at the surface height (orography; 0 over ocean). The 2 m fields are
    # applied at the surface; the ~2 m offset is negligible (~0.3 M-units).
    # Snap sub-metre orography (geoid noise over sea) to 0 so ocean columns
    # anchor exactly at sea level and the 0 m output bin is populated.
    z = xr.where(orog < 1.0, 0.0, orog)
    m = n + rf.M_CURVATURE_PER_M * z

    return xr.Dataset(
        {
            "N_sfc": n.assign_attrs(units="N-units"),
            "M_sfc": m.assign_attrs(units="M-units"),
            "z_sfc": z.assign_attrs(units="m", long_name="surface height ASL"),
        }
    )


def load_bulk_surface_fields(grib_path: str):
    """Load the bulk fields the evaporation-duct model needs.

    Returns a dataset on ``(latitude, longitude)`` with:
    ``sst`` [K] (skin temp ~ SST over sea), ``t2m`` [K], ``rh2`` [%],
    ``wind10`` [m/s], ``sp`` [hPa], ``lsm`` (land fraction, 1 = land).
    """
    import xarray as xr

    # Drop scalar level/time coords that otherwise collide when merging fields
    # from surface, 2 m and 10 m into one dataset.
    drop = ["heightAboveGround", "surface", "step", "time", "valid_time"]

    def clean(da):
        return da.drop_vars([c for c in drop if c in da.coords])

    sfc = _open_grib(grib_path, {"typeOfLevel": "surface"})
    agl2 = _open_grib(grib_path, {"typeOfLevel": "heightAboveGround", "level": 2})
    agl10 = _open_grib(grib_path, {"typeOfLevel": "heightAboveGround", "level": 10})

    wind = np.hypot(agl10["u10"], agl10["v10"])
    return xr.Dataset(
        {
            "sst": clean(sfc["t"]).assign_attrs(units="K", long_name="skin/sea temp"),
            "t2m": clean(agl2["t2m"]),
            "rh2": clean(agl2["r2"]).assign_attrs(long_name="2 m relative humidity"),
            "wind10": clean(wind).assign_attrs(units="m/s", long_name="10 m wind speed"),
            "sp": clean(sfc["sp"] / 100.0).assign_attrs(units="hPa"),
            "lsm": clean(sfc["lsm"]).assign_attrs(long_name="land-sea mask (1=land)"),
        }
    )


def subset_bbox(ds, bbox: BBox):
    """Crop a dataset on ``(latitude, longitude)`` to ``bbox``.

    NOMADS subsets server-side, but the AWS archive serves global messages, so
    the crop happens here instead. Handles descending latitude axes and boxes
    that wrap the 0/360 meridian.
    """
    import xarray as xr

    b = bbox.to_0_360()
    lat = ds["latitude"].values
    lat_slice = (slice(bbox.lat_max, bbox.lat_min) if lat[0] > lat[-1]
                 else slice(bbox.lat_min, bbox.lat_max))
    ds = ds.sel(latitude=lat_slice)

    if b.lon_min <= b.lon_max:
        return ds.sel(longitude=slice(b.lon_min, b.lon_max))

    # Wrapping box: take the two pieces and stitch them into a monotonic axis
    # by shifting the western piece to negative longitudes.
    west = ds.sel(longitude=slice(b.lon_min, 360.0))
    west = west.assign_coords(longitude=west["longitude"] - 360.0)
    east = ds.sel(longitude=slice(0.0, b.lon_max))
    return xr.concat([west, east], dim="longitude")


# --------------------------------------------------------------------------
# refractivity on levels / height grid
# --------------------------------------------------------------------------

def compute_refractivity_on_levels(ds):
    """Add ``N`` and ``M`` (and geometric height ``z``) on the pressure levels."""
    import xarray as xr

    p = ds["isobaricInhPa"]  # hPa, broadcast along the level dim
    t = ds["t"]
    rh = ds["r"]
    z = ds["gh"]  # geopotential height [m] ~ geometric height in troposphere

    e = xr.apply_ufunc(rf.vapour_pressure_from_rh, t, rh)
    n = xr.apply_ufunc(rf.refractivity_n, p, t, e)
    m = n + rf.M_CURVATURE_PER_M * z

    return ds.assign(
        e=e.assign_attrs(units="hPa", long_name="water vapour partial pressure"),
        N=n.assign_attrs(units="N-units", long_name="radio refractivity"),
        M=m.assign_attrs(units="M-units", long_name="modified refractivity"),
        z=z.assign_attrs(units="m", long_name="geometric height ASL"),
    )


def interpolate_to_height(ds_levels, heights_m, surface=None):
    """Interpolate ``N`` and ``M`` from pressure levels onto a height grid.

    Each column has its own height coordinate (``z``), so we interpolate per
    column. Heights above the model top are filled with NaN; below the lowest
    data point they are filled with NaN unless ``surface`` provides the 2 m
    anchor point from :func:`load_surface_layer`, which is prepended to each
    column so the profile reaches the surface.

    Returns a dataset with ``M`` and ``N`` on ``(height, latitude, longitude)``.
    """
    import xarray as xr

    heights_m = np.asarray(heights_m, dtype=float)
    z = ds_levels["z"].values  # (level, lat, lon)
    lev, nlat, nlon = z.shape

    z_sfc = surface["z_sfc"].values if surface is not None else None
    src_sfc = {
        "N": surface["N_sfc"].values if surface is not None else None,
        "M": surface["M_sfc"].values if surface is not None else None,
    }

    def _interp_var(var_name: str) -> np.ndarray:
        src = ds_levels[var_name].values  # (level, lat, lon)
        sfc = src_sfc.get(var_name)
        out = np.full((heights_m.size, nlat, nlon), np.nan, dtype=float)
        for j in range(nlat):
            for i in range(nlon):
                zc = z[:, j, i]
                vc = src[:, j, i]
                if z_sfc is not None and sfc is not None:
                    zc = np.concatenate(([z_sfc[j, i]], zc))
                    vc = np.concatenate(([sfc[j, i]], vc))
                good = np.isfinite(zc) & np.isfinite(vc)
                if good.sum() < 2:
                    continue
                order = np.argsort(zc[good])
                zz = zc[good][order]
                vv = vc[good][order]
                out[:, j, i] = np.interp(
                    heights_m, zz, vv, left=np.nan, right=np.nan
                )
        return out

    coords = {
        "height": ("height", heights_m, {"units": "m", "long_name": "height ASL"}),
        "latitude": ds_levels["latitude"],
        "longitude": ds_levels["longitude"],
    }
    out = xr.Dataset(
        {
            "M": (("height", "latitude", "longitude"), _interp_var("M"),
                  {"units": "M-units", "long_name": "modified refractivity"}),
            "N": (("height", "latitude", "longitude"), _interp_var("N"),
                  {"units": "N-units", "long_name": "radio refractivity"}),
        },
        coords=coords,
    )
    # Carry over useful global attrs (cycle, forecast hour, etc.).
    out.attrs.update(ds_levels.attrs)
    return out


# --------------------------------------------------------------------------
# fetch entry points
# --------------------------------------------------------------------------

def fetch_refractivity_cube(
    bbox: BBox,
    forecast_hour: int = 0,
    cycle: dt.datetime | None = None,
    top_height_m: float = 3000.0,
    dz_m: float = 20.0,
    keep_pressure_levels: bool = True,
    workdir: str | None = None,
    max_cycle_fallbacks: int | None = None,
    source: str = "auto",
    use_cache: bool = True,
):
    """Fetch GFS and return a modified-refractivity cube for a region.

    Parameters
    ----------
    bbox : BBox
        Region of interest.
    forecast_hour : int
        Forecast lead time in hours (0 = analysis).
    cycle : datetime, optional
        Specific GFS cycle (UTC). Defaults to the latest available.
    top_height_m, dz_m : float
        Vertical grid for the output cube (0..top_height_m step dz_m).
    keep_pressure_levels : bool
        If True, also attach the native pressure-level fields as ``M_plev`` etc.
    workdir : str, optional
        Directory for the downloaded GRIB (defaults to the shared cache,
        ``~/.cache/pywaveprop/gfs``).
    max_cycle_fallbacks : int, optional
        How many older cycles to try if the requested one is missing. Defaults
        to 3 when ``cycle`` is None (latest), and 0 when a specific ``cycle`` is
        given (so a historical request never silently returns a different date).
    source : {'auto', 'nomads', 'archive'}
        GFS back end; see :mod:`pywaveprop.environment.gfs`.
    use_cache : bool
        Reuse a previously downloaded GRIB for the same request and cycle.

    Returns
    -------
    xarray.Dataset
        ``M`` and ``N`` on ``(height, latitude, longitude)`` plus metadata.
    """
    req = GFSRequest(bbox=bbox, forecast_hour=forecast_hour, cycle=cycle)
    if max_cycle_fallbacks is None:
        max_cycle_fallbacks = 0 if cycle is not None else 3

    grib_path, used_cycle = download_cached(
        req, max_cycle_fallbacks=max_cycle_fallbacks, source=source,
        cache_dir=workdir, use_cache=use_cache)

    ds_lvl = load_pressure_level_fields(grib_path)
    surface = load_surface_layer(grib_path)
    if _is_global_grid(ds_lvl):
        ds_lvl = subset_bbox(ds_lvl, bbox)
        if surface is not None:
            surface = subset_bbox(surface, bbox)
    ds_lvl = ascending_latitude(ds_lvl)
    if surface is not None:
        surface = ascending_latitude(surface)
    ds_lvl = compute_refractivity_on_levels(ds_lvl)

    heights = np.arange(0.0, top_height_m + dz_m / 2, dz_m)
    cube = interpolate_to_height(ds_lvl, heights, surface=surface)

    cube.attrs.update(
        source="NOAA GFS 0.25deg",
        cycle=used_cycle.strftime("%Y-%m-%dT%H:00Z"),
        forecast_hour=forecast_hour,
        valid_time=(used_cycle + dt.timedelta(hours=forecast_hour)).strftime(
            "%Y-%m-%dT%H:00Z"
        ),
    )
    if keep_pressure_levels:
        cube["M_plev"] = ds_lvl["M"]
        cube["N_plev"] = ds_lvl["N"]
        cube["z_plev"] = ds_lvl["z"]
    return cube


def fetch_surface_bulk(
    bbox: BBox,
    forecast_hour: int = 0,
    cycle: dt.datetime | None = None,
    workdir: str | None = None,
    max_cycle_fallbacks: int | None = None,
    source: str = "auto",
    use_cache: bool = True,
):
    """Download only the surface/2 m/10 m bulk fields for a region.

    Small, fast download (no pressure levels) for evaporation-duct work.
    """
    req = GFSRequest(
        bbox=bbox, forecast_hour=forecast_hour, cycle=cycle,
        levels_hpa=[], variables=["TMP", "RH"],
        include_surface=True, include_wind10m=True,
    )
    if max_cycle_fallbacks is None:
        max_cycle_fallbacks = 0 if cycle is not None else 3

    grib_path, used_cycle = download_cached(
        req, max_cycle_fallbacks=max_cycle_fallbacks, source=source,
        cache_dir=workdir, use_cache=use_cache)

    ds = load_bulk_surface_fields(grib_path)
    if _is_global_grid(ds):
        ds = subset_bbox(ds, bbox)
    ds = ascending_latitude(ds)
    ds.attrs.update(
        source="NOAA GFS 0.25deg",
        cycle=used_cycle.strftime("%Y-%m-%dT%H:00Z"),
        forecast_hour=forecast_hour,
        valid_time=(used_cycle + dt.timedelta(hours=forecast_hour)).strftime(
            "%Y-%m-%dT%H:00Z"
        ),
    )
    return ds


def ascending_latitude(ds):
    """Return ``ds`` with a south-to-north latitude axis.

    The NOMADS subsetter emits ascending latitude while the global archive
    files run north-to-south; normalising here makes the two back ends
    interchangeable for downstream code.
    """
    lat = ds["latitude"].values
    if lat.size > 1 and lat[0] > lat[-1]:
        return ds.isel(latitude=slice(None, None, -1))
    return ds


def _is_global_grid(ds) -> bool:
    """True if a dataset spans the whole globe and still needs cropping.

    The AWS archive serves full-globe messages (1440 x 721 at 0.25 degrees);
    anything smaller came from the NOMADS subsetter and is already cropped.
    """
    return ds.sizes.get("longitude", 0) >= 1440


# --------------------------------------------------------------------------
# range-dependent transects
# --------------------------------------------------------------------------

def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance [m] between two points (broadcasting)."""
    R = rf.EARTH_RADIUS_M
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dphi = np.deg2rad(np.asarray(lat2, dtype=float) - lat1)
    dl = np.deg2rad(np.asarray(lon2, dtype=float) - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def surface_layer_height_grid(z_top: float, dz_low: float = 1.0,
                              z_low: float = Z_SPLICE,
                              dz_high: float = 20.0) -> np.ndarray:
    """Height grid that resolves the surface layer finely and the rest coarsely."""
    low = np.arange(0.0, z_low, dz_low)
    high = np.arange(z_low, z_top + dz_high / 2, dz_high)
    return np.unique(np.concatenate([low, high]))


def _sorted_axis(vals):
    order = np.argsort(vals)
    return vals[order], order


def refractivity_transect(
    lat: np.ndarray,
    lon: np.ndarray,
    height_m: np.ndarray,
    M: np.ndarray,
    p1: tuple[float, float],
    p2: tuple[float, float],
    bulk: dict | None = None,
    n_cols: int = 61,
    z_top: float = 1200.0,
    splice_evaporation: bool = True,
    z_splice: float = Z_SPLICE,
    z_blend: float = Z_BLEND,
    deficit_cap: float = DEFICIT_CAP,
    name: str = "",
    cycle: str = "",
) -> RefractivityTransect:
    """Sample an ``M(height, lat, lon)`` cube along a path into ``M(x, z)``.

    Over the sea the sub-``z_splice`` evaporation-duct structure (Monin-Obukhov
    surface layer) is spliced onto the NWP column: the surface-layer deviation
    from the sea surface is clipped to ``deficit_cap`` M-units, anchored to the
    NWP surface refractivity and blended back into the NWP profile by
    ``z_blend``. MOST over-predicts the deficit in the stable regime, so we clip
    and blend rather than stack.

    Parameters
    ----------
    lat, lon : 1-D arrays
        Cube axes in degrees (either lon convention, used consistently).
    height_m : 1-D array
        Cube height axis [m].
    M : (nz, nlat, nlon) array
        Modified refractivity.
    p1, p2 : (lat, lon)
        Path endpoints.
    bulk : dict, optional
        2-D arrays on ``(lat, lon)``: ``sst``, ``t2m``, ``rh2``, ``wind10``,
        ``sp``, ``lsm`` and optionally ``edh``. Required for the evaporation
        splice; without it only the NWP profile is used.
    n_cols : int
        Number of columns sampled along the path.
    z_top : float
        Top of the output height grid [m].
    """
    from scipy.interpolate import RegularGridInterpolator

    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    height_m = np.asarray(height_m, dtype=float)
    M = np.asarray(M, dtype=float)

    # ascending axes for the interpolator
    lat_s, jo = _sorted_axis(lat)
    lon_s, io = _sorted_axis(lon)
    interp_M = RegularGridInterpolator(
        (height_m, lat_s, lon_s), M[:, jo, :][:, :, io],
        bounds_error=False, fill_value=np.nan)

    bulk_interp = {}
    if bulk is not None:
        for key in ("sst", "t2m", "rh2", "wind10", "sp", "lsm", "edh"):
            if key in bulk:
                v = np.asarray(bulk[key], dtype=float)[jo][:, io]
                bulk_interp[key] = RegularGridInterpolator(
                    (lat_s, lon_s), v, bounds_error=False, fill_value=np.nan)

    lat1, lon1 = p1
    lat2, lon2 = p2
    t = np.linspace(0.0, 1.0, n_cols)
    lat_pts = lat1 + t * (lat2 - lat1)
    lon_pts = lon1 + t * (lon2 - lon1)
    x_m = haversine_m(lat1, lon1, lat_pts, lon_pts)

    z_m = surface_layer_height_grid(z_top, z_low=z_splice)
    out_M = np.full((n_cols, z_m.size), np.nan)
    edh_pts = np.full(n_cols, np.nan)
    lsm_pts = np.full(n_cols, np.nan)
    base = np.full(n_cols, np.nan)
    top = np.full(n_cols, np.nan)
    strength = np.full(n_cols, np.nan)

    can_splice = splice_evaporation and all(
        k in bulk_interp for k in ("sst", "t2m", "rh2", "wind10", "sp", "lsm"))

    for k in range(n_cols):
        la, lo = lat_pts[k], lon_pts[k]
        # NWP column on its native height grid
        col = interp_M((height_m, np.full_like(height_m, la),
                        np.full_like(height_m, lo)))
        good = np.isfinite(col)
        if good.sum() < 3:
            continue
        hz, mz = height_m[good], col[good]
        M_nwp = np.interp(z_m, hz, mz)
        # extrapolate above the cube top with the local slope, so the profile
        # the non-local boundary condition sees stays linear
        if z_m[-1] > hz[-1]:
            slope = (mz[-1] - mz[-2]) / (hz[-1] - hz[-2])
            M_nwp[z_m > hz[-1]] = mz[-1] + slope * (z_m[z_m > hz[-1]] - hz[-1])

        if "lsm" in bulk_interp:
            lsm_pts[k] = float(bulk_interp["lsm"]((la, lo)))
        if "edh" in bulk_interp:
            edh_pts[k] = float(bulk_interp["edh"]((la, lo)))

        M_col = M_nwp.copy()
        if can_splice and np.isfinite(lsm_pts[k]) and lsm_pts[k] < 0.5:
            vals = {key: float(bulk_interp[key]((la, lo)))
                    for key in ("sst", "t2m", "rh2", "wind10", "sp")}
            if np.all(np.isfinite(list(vals.values()))):
                r = evaporation_duct(vals["sst"], vals["t2m"], vals["rh2"],
                                     vals["wind10"], vals["sp"],
                                     z_max=z_splice, n_levels=200)
                if "edh" not in bulk_interp:
                    edh_pts[k] = r["edh_m"]
                Me = np.interp(z_m, r["z"], r["M"])
                # deviation from the sea surface, clipped so the stable-regime
                # MOST blow-up cannot create an unphysical deficit
                dev = np.clip(Me - Me[0], -deficit_cap, deficit_cap)
                # anchor the surface to the NWP refractivity N_sfc = M_nwp(0)
                M_evap = M_nwp[0] + dev
                # blend into the NWP profile: w=1 below the splice, 0 by z_blend
                w = np.clip((z_blend - z_m) / (z_blend - z_splice), 0.0, 1.0)
                M_col = w * M_evap + (1.0 - w) * M_nwp

        out_M[k] = M_col
        dd = rf.duct_diagnostics(M_col, z_m)
        if dd.get("has_duct"):
            base[k] = dd["base_height_m"]
            top[k] = dd["top_height_m"]
            strength[k] = dd["strength_M"]

    return RefractivityTransect(
        x_m=x_m, z_m=z_m, M=out_M, lat_pts=lat_pts, lon_pts=lon_pts,
        lsm_pts=lsm_pts, edh_pts=edh_pts, duct_base=base, duct_top=top,
        duct_strength=strength, p1=np.asarray(p1, dtype=float),
        p2=np.asarray(p2, dtype=float), name=name, cycle=cycle)


def refractivity_transect_from_cube(cube, p1, p2, bulk=None, **kwargs
                                    ) -> RefractivityTransect:
    """:func:`refractivity_transect` for the datasets the fetch helpers return.

    ``cube`` is a :func:`fetch_refractivity_cube` result and ``bulk`` a
    :func:`fetch_surface_bulk` result on the same grid. Longitudes are converted
    to -180..180 so endpoints can be given in either convention.
    """
    lon = ((cube["longitude"].values + 180.0) % 360.0) - 180.0
    bulk_arrays = None
    if bulk is not None:
        bulk_arrays = {k: bulk[k].values for k in
                       ("sst", "t2m", "rh2", "wind10", "sp", "lsm") if k in bulk}
    p1 = (p1[0], ((p1[1] + 180.0) % 360.0) - 180.0)
    p2 = (p2[0], ((p2[1] + 180.0) % 360.0) - 180.0)
    kwargs.setdefault("name", str(cube.attrs.get("source", "")))
    kwargs.setdefault("cycle", str(cube.attrs.get("cycle", "")))
    return refractivity_transect(
        cube["latitude"].values, lon, cube["height"].values, cube["M"].values,
        p1, p2, bulk=bulk_arrays, **kwargs)


def save_transect(transect: RefractivityTransect, path: str) -> str:
    """Write a transect to a compressed ``.npz`` (readable by :func:`load_transect`)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **transect.as_dict())
    return path


def load_transect(path: str) -> RefractivityTransect:
    """Read back a transect written by :func:`save_transect`."""
    d = np.load(path, allow_pickle=False)
    return RefractivityTransect(
        x_m=d["x_m"], z_m=d["z_m"], M=d["M"], lat_pts=d["lat_pts"],
        lon_pts=d["lon_pts"], lsm_pts=d["lsm_pts"], edh_pts=d["edh_pts"],
        duct_base=d["duct_base"], duct_top=d["duct_top"],
        duct_strength=d["duct_strength"], p1=d["p1"], p2=d["p2"],
        name=str(d["name"]), cycle=str(d["cycle"]))
