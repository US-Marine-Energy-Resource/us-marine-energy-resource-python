"""
Build the bundled spatial data files from the S3 dataset.

Fetches the manifest JSON from S3, derives the b4 summary parquet URL for each
location, then writes to us_marine_energy_resource/data/:

  geometry_{location}.parquet   — triangle vertices + centroid integers, sorted
                                   by (lat_fixed_precision, lon_fixed_precision) for row-group pruning
  location_bounds.parquet        — exact outer mesh boundary per location,
                                   computed as the union of all triangles

This is a maintainer script.  Run it when the upstream dataset or manifest is
updated.  The manifest URL controls which data version is used; pass
--manifest-url to point at a newer manifest when one is released.

Usage
-----
    python scripts/build_geometry_data.py
    python scripts/build_geometry_data.py --manifest-url <url>
    python scripts/build_geometry_data.py --bounds-only     # skip S3 download
    python scripts/build_geometry_data.py --tolerance 0.0005

AK_aleutian_islands crosses the antimeridian (±180°).  Its mesh boundary is
computed with vertex longitudes translated to 0-360° space so the island chain
forms a single connected polygon.  The resulting WKT is stored in 0-360° space;
gate checks for this location apply the same transform to the query coordinate.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import duckdb
import pandas as pd

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent
_DATA_DIR = _REPO_ROOT / "us_marine_energy_resource" / "data"


def _geometry_out(manifest: dict, location: str) -> Path:
    version = manifest["locations"][location]["latest_version"]
    return _DATA_DIR / f"geometry_{location}_v{version}.parquet"


def _bounds_out(manifest: dict) -> Path:
    return _DATA_DIR / f"location_bounds_v{manifest['manifest_version']}.parquet"


# Default manifest URL.  Update this when a new manifest version is published.
_DEFAULT_MANIFEST_URL = (
    "https://marine-energy-data.s3.us-west-2.amazonaws.com"
    "/us-tidal/manifest/v1.0.0/manifest_1.0.0.json"
)

_S3_HTTP_BASE = "https://marine-energy-data.s3.us-west-2.amazonaws.com"

# Locations whose mesh spans the antimeridian (±180°).
# Boundary WKT is stored in 0-360° longitude space for these locations.
_ANTIMERIDIAN_LOCATIONS: frozenset[str] = frozenset({"AK_aleutian_islands"})

# Boundary simplification tolerance in degrees (0.001° ≈ 111 m).
_DEFAULT_TOLERANCE = 0.001

_ROW_GROUP_SIZE = 100_000

# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------


def _fetch_manifest(url: str) -> dict:
    with urllib.request.urlopen(url) as resp:
        return json.load(resp)  # type: ignore[return-value]


def _s3_to_https(uri: str) -> str:
    return uri.replace("s3://marine-energy-data/", f"{_S3_HTTP_BASE}/")


def _b4_url(manifest: dict, location: str) -> str:
    """Derive the b4 summary parquet URL for a location from the manifest."""
    loc = manifest["locations"][location]
    version = loc["latest_version"]
    date = loc["date"]
    time_ = loc["time"]
    dataset = manifest["dataset"]["name"]
    base = _s3_to_https(manifest["storage"]["s3_base_uri"])
    filename = f"{location}.{dataset}-year_average.b4.{date}.{time_}.v{version}.parquet"
    return f"{base}/{location}/v{version}/b4_vap_summary_parquet/{filename}"


# ---------------------------------------------------------------------------
# SQL fragments
# ---------------------------------------------------------------------------

# Centroid coords are extracted from the S3 URI filename (not lat_center/lon_center)
# to avoid float32 precision loss. Stored as fixed-point integers at this many decimal
# places (7 d.p. ≈ 1 cm resolution) for row-group pruning; matches _COORD_DECIMAL_PRECISION
# in _spatial.py — keep the two in sync.
_COORD_DECIMAL_PRECISION: int = 7
_COORD_PRECISION_SCALE: int = 10**_COORD_DECIMAL_PRECISION

_GEOMETRY_COLS = f"""\
    '{{location}}'  AS location,
    face_id,
    element_corner_1_lat  AS c1_lat,
    element_corner_1_lon  AS c1_lon,
    element_corner_2_lat  AS c2_lat,
    element_corner_2_lon  AS c2_lon,
    element_corner_3_lat  AS c3_lat,
    element_corner_3_lon  AS c3_lon,
    CAST(ROUND(
        CAST(regexp_extract(
            full_year_data_s3_uri, '\\.lat=(-?[0-9]+\\.[0-9]+)\\.', 1
        ) AS DOUBLE) * {_COORD_PRECISION_SCALE}
    ) AS INTEGER) AS lat_fixed_precision,
    CAST(ROUND(
        CAST(regexp_extract(
            full_year_data_s3_uri, '\\.lon=(-?[0-9]+\\.[0-9]+)-', 1
        ) AS DOUBLE) * {_COORD_PRECISION_SCALE}
    ) AS INTEGER) AS lon_fixed_precision"""

_TRIANGLE = """\
ST_MakePolygon(ST_MakeLine(ARRAY[
    ST_Point(c1_lon::DOUBLE, c1_lat::DOUBLE),
    ST_Point(c2_lon::DOUBLE, c2_lat::DOUBLE),
    ST_Point(c3_lon::DOUBLE, c3_lat::DOUBLE),
    ST_Point(c1_lon::DOUBLE, c1_lat::DOUBLE)
]))"""

# For antimeridian locations: translate vertex lons to 0-360° before union.
_TRIANGLE_360 = """\
ST_MakePolygon(ST_MakeLine(ARRAY[
    ST_Point(
        CASE WHEN c1_lon < 0 THEN c1_lon::DOUBLE + 360 ELSE c1_lon::DOUBLE END,
        c1_lat::DOUBLE),
    ST_Point(
        CASE WHEN c2_lon < 0 THEN c2_lon::DOUBLE + 360 ELSE c2_lon::DOUBLE END,
        c2_lat::DOUBLE),
    ST_Point(
        CASE WHEN c3_lon < 0 THEN c3_lon::DOUBLE + 360 ELSE c3_lon::DOUBLE END,
        c3_lat::DOUBLE),
    ST_Point(
        CASE WHEN c1_lon < 0 THEN c1_lon::DOUBLE + 360 ELSE c1_lon::DOUBLE END,
        c1_lat::DOUBLE)
]))"""


# ---------------------------------------------------------------------------
# Phase 1 — per-location geometry parquets
# ---------------------------------------------------------------------------


def _build_one_geometry(
    con: duckdb.DuckDBPyConnection, manifest: dict, location: str, url: str
) -> int:
    out = _geometry_out(manifest, location)
    cols = _GEOMETRY_COLS.replace("'{location}'", f"'{location}'")
    table = f"_geom_{location}"

    # Load into a temp table so build_bounds can re-use it without re-reading S3.
    con.execute(f"""
        CREATE OR REPLACE TABLE {table} AS
        SELECT {cols}
        FROM read_parquet('{url}')
        ORDER BY lat_fixed_precision, lon_fixed_precision
    """)
    n: int = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]  # type: ignore[index]
    con.execute(f"""
        COPY (SELECT * FROM {table})
        TO '{out.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD, COMPRESSION_LEVEL 9,
         ROW_GROUP_SIZE {_ROW_GROUP_SIZE})
    """)
    return n


def build_geometry(con: duckdb.DuckDBPyConnection, manifest: dict) -> None:
    """Download per-location geometry parquets from S3 and write versioned files."""
    print("Geometry parquets")
    print(f"  Output: {_DATA_DIR}/geometry_{{location}}_v{{version}}.parquet")
    print()

    t0 = time.time()
    for location in manifest["locations"]:
        url = _b4_url(manifest, location)
        version = manifest["locations"][location]["latest_version"]
        print(f"  [{location}] v{version}  reading S3 ...", end="", flush=True)
        t = time.time()
        n = _build_one_geometry(con, manifest, location, url)
        out = _geometry_out(manifest, location)
        mb = out.stat().st_size / 1024 / 1024
        print(f" {time.time() - t:.1f}s  {n:,} faces  {mb:.1f} MB")

    print(f"\n  Total: {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Location bounds
# ---------------------------------------------------------------------------


def _bounds_row(
    con: duckdb.DuckDBPyConnection, location: str, source: str, tolerance: float
) -> dict:
    antimeridian = location in _ANTIMERIDIAN_LOCATIONS
    triangle = _TRIANGLE_360 if antimeridian else _TRIANGLE
    row = con.execute(f"""
        SELECT
            ST_AsText(
                ST_SimplifyPreserveTopology(ST_Union_Agg({triangle}), {tolerance})
            )                                               AS boundary_wkt,
            MIN(CAST(lat_fixed_precision AS DOUBLE) / {_COORD_PRECISION_SCALE}) AS lat_min,
            MAX(CAST(lat_fixed_precision AS DOUBLE) / {_COORD_PRECISION_SCALE}) AS lat_max,
            MIN(CAST(lon_fixed_precision AS DOUBLE) / {_COORD_PRECISION_SCALE}) AS lon_min,
            MAX(CAST(lon_fixed_precision AS DOUBLE) / {_COORD_PRECISION_SCALE}) AS lon_max,
            COUNT(*)                                        AS face_count
        FROM {source}
    """).fetchone()
    assert row is not None
    return {
        "location": location,
        "boundary_wkt": row[0],
        "lat_min": row[1],
        "lat_max": row[2],
        "lon_min": row[3],
        "lon_max": row[4],
        "crosses_antimeridian": antimeridian,
        "face_count": int(row[5]),
    }


def build_bounds(con: duckdb.DuckDBPyConnection, manifest: dict, tolerance: float) -> None:
    """Compute the union mesh boundary per location and write location_bounds parquet."""
    bounds_file = _bounds_out(manifest)
    con.execute("LOAD spatial;")
    print("Location bounds")
    print(f"  Output: {bounds_file}")
    print(f"  Tolerance: {tolerance}° (~{tolerance * 111_000:.0f} m)")
    print()

    rows = []
    t0 = time.time()

    for location in manifest["locations"]:
        # Re-use the in-memory table from Phase 1 if present; otherwise read
        # the already-written per-location parquet.
        table = f"_geom_{location}"
        try:
            con.execute(f"SELECT 1 FROM {table} LIMIT 1")
            source = table
        except duckdb.CatalogException:
            parquet = _geometry_out(manifest, location)
            if not parquet.exists():
                raise FileNotFoundError(
                    f"{parquet.name} not found. Run without --bounds-only first."
                ) from None
            source = f"read_parquet('{parquet.as_posix()}')"

        note = "  [antimeridian -> 0-360° lon]" if location in _ANTIMERIDIAN_LOCATIONS else ""
        print(f"  [{location}] union of triangles ...{note}", end="", flush=True)
        t = time.time()
        row = _bounds_row(con, location, source, tolerance)
        rows.append(row)
        wkt_len = len(row["boundary_wkt"])
        print(f" {time.time() - t:.1f}s  {row['face_count']:,} faces → {wkt_len:,} chars WKT")

    bounds_df = pd.DataFrame(rows)
    con.execute("CREATE OR REPLACE TABLE _bounds AS SELECT * FROM bounds_df")
    con.execute(f"""
        COPY (SELECT * FROM _bounds ORDER BY location)
        TO '{bounds_file.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    print(f"\n  Total: {time.time() - t0:.1f}s")

    check = con.execute(f"""
        SELECT location, crosses_antimeridian, face_count,
               ROUND(lat_min, 3) AS lat_min, ROUND(lat_max, 3) AS lat_max,
               ROUND(lon_min, 3) AS lon_min, ROUND(lon_max, 3) AS lon_max
        FROM read_parquet('{bounds_file.as_posix()}')
        ORDER BY location
    """).df()
    print()
    print(check.to_string(index=False))


def write_data_index(manifest: dict) -> None:
    """Write data_index.json mapping locations to their versioned parquet filenames."""
    index = {
        "manifest_version": manifest["manifest_version"],
        "bounds_file": _bounds_out(manifest).name,
        "geometry_files": {
            location: _geometry_out(manifest, location).name for location in manifest["locations"]
        },
    }
    out = _DATA_DIR / "data_index.json"
    with open(out, "w") as f:
        json.dump(index, f, indent=2)
    print(f"  Wrote: {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and run the geometry build pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-url",
        default=_DEFAULT_MANIFEST_URL,
        help="Manifest JSON URL (default: current production manifest)",
    )
    parser.add_argument(
        "--bounds-only",
        action="store_true",
        help="Skip geometry download; re-compute bounds from existing geometry parquets.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=_DEFAULT_TOLERANCE,
        help=f"Boundary simplification tolerance in degrees (default: {_DEFAULT_TOLERANCE})",
    )
    args = parser.parse_args()

    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Fetching manifest: {args.manifest_url}")
    manifest = _fetch_manifest(args.manifest_url)
    print(
        f"  spec v{manifest['spec_version']}  "
        f"manifest v{manifest['manifest_version']}  "
        f"generated {manifest['manifest_generated']}"
    )
    print(f"  Locations: {list(manifest['locations'].keys())}")
    print()

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")

    if not args.bounds_only:
        build_geometry(con, manifest)
        print()

    build_bounds(con, manifest, args.tolerance)
    print()
    print("Data index")
    write_data_index(manifest)


if __name__ == "__main__":
    main()
