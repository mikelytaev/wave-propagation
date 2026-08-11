"""CLI: fetch a GFS refractivity cube for a region and save it to netCDF.

Example
-------
    python -m pywaveprop.environment --bbox -128 -120 32 38 \
        --top-height 3000 --dz 20 --profile 34 -125 --out cube.nc
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys

import numpy as np

from .gfs import BBox
from .nwp import fetch_refractivity_cube
from .refractivity import duct_diagnostics


def _parse_cycle(s: str | None) -> dt.datetime | None:
    if not s:
        return None
    return dt.datetime.strptime(s, "%Y%m%d%H")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="pywaveprop.environment", description=__doc__)
    p.add_argument(
        "--bbox", nargs=4, type=float, required=True,
        metavar=("LONMIN", "LONMAX", "LATMIN", "LATMAX"),
        help="Region in degrees (lon -180..180 or 0..360).",
    )
    p.add_argument("--fhr", type=int, default=0, help="Forecast hour (0 = analysis).")
    p.add_argument("--cycle", type=str, default=None, help="GFS cycle YYYYMMDDHH (UTC).")
    p.add_argument("--top-height", type=float, default=3000.0, help="Cube top [m].")
    p.add_argument("--dz", type=float, default=20.0, help="Vertical step [m].")
    p.add_argument("--source", choices=("auto", "nomads", "archive"), default="auto",
                   help="GFS back end (default: NOMADS, AWS archive for old cycles).")
    p.add_argument("--out", type=str, default=None, help="Output netCDF path.")
    p.add_argument(
        "--profile", nargs=2, type=float, default=None, metavar=("LAT", "LON"),
        help="Also print the M(h) profile + duct diagnostics at this point.",
    )
    args = p.parse_args(argv)

    bbox = BBox(*args.bbox)
    print(f"Fetching GFS for bbox={args.bbox} fhr={args.fhr} ...", file=sys.stderr)

    cube = fetch_refractivity_cube(
        bbox,
        forecast_hour=args.fhr,
        cycle=_parse_cycle(args.cycle),
        top_height_m=args.top_height,
        dz_m=args.dz,
        source=args.source,
    )
    print(
        f"cycle={cube.attrs.get('cycle')} valid={cube.attrs.get('valid_time')} "
        f"shape M{tuple(cube.M.shape)} (height x lat x lon)",
        file=sys.stderr,
    )

    if args.profile:
        lat, lon = args.profile
        lon360 = lon % 360.0
        col = cube.M.sel(latitude=lat, longitude=lon360, method="nearest")
        h = cube.height.values
        diag = duct_diagnostics(col.values, h)
        print(f"\nProfile at lat={float(col.latitude):.3f} lon={float(col.longitude):.3f}")
        print(f"{'height[m]':>10} {'M':>10}")
        for hi, mi in zip(h, col.values):
            if np.isfinite(mi):
                print(f"{hi:10.0f} {mi:10.2f}")
        print("\nDuct diagnostics:", diag)

    if args.out:
        cube.to_netcdf(args.out)
        print(f"wrote {args.out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
