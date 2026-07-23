"""Build the grid-node index that backs us_marine_energy_resource.wave_hindcast.nodes.

Maintainer tool, not part of the runtime path. It reads the ``coordinates``
dataset out of each published domain's .h5 on S3 and writes one parquet per
domain into ``data/h2o_wave_hindcast_index/v1/`` (Git LFS), plus the small
in-package files
under ``us_marine_energy_resource/data/``: the occupancy-cell domain gate, the
coverage footprints, the SHA256 registry, and the index JSON.

Only the ``coordinates`` dataset is read, never the sea-state arrays, so this
transfers a few MB per domain rather than the hundreds of GB the files weigh.
The read itself lives in ``us_marine_energy_resource.wave_hindcast.index_build``, which
is also the runtime fallback when a node file can be neither found nor
downloaded. This script adds what the fallback does not need.

    python scripts/build_wave_node_index.py                 # all domains
    python scripts/build_wave_node_index.py --domain Hawaii # one
    python scripts/build_wave_node_index.py --verify        # re-check, no rebuild

Every build verifies itself by replaying the configured sites through
``wave.nodes.nearest`` and comparing the grid ids against the values frozen
below. A disagreement means published results would silently stop matching,
so it fails the build.

The footprint dissolve uses DuckDB's spatial extension, which downloads on
first use. That is acceptable here because this is a maintainer tool; the
runtime query path deliberately avoids the extension.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from us_marine_energy_resource.wave_hindcast import index_build  # noqa: E402
from us_marine_energy_resource.wave_hindcast.config import CONFIG  # noqa: E402
from us_marine_energy_resource.wave_hindcast.domains import (  # noqa: E402
    DOMAIN_ENDPOINTS,
    DOMAINS,
    SITES,
)
INDEX_VERSION = CONFIG.index_version

LFS_DIR = REPO_ROOT / "data" / CONFIG.index_subdir
PKG_DATA = REPO_ROOT / "us_marine_energy_resource" / "data"

COORD_SCALE = index_build.COORD_SCALE

# Side of the occupancy cells describing where each domain actually has nodes,
# in degrees. Bounding boxes cannot do this job: Alaska spans the antimeridian,
# so its box is -180..180 and would match every point on earth.
#
# 0.05 degrees is ~5.6 km. That is deliberately coarser than the ~400-700 m
# median node spacing, because these are unstructured meshes whose node density
# varies: at 0.01 degrees the occupied cells stop touching each other and the
# footprint shatters into 183k disconnected fragments rather than describing a
# coastline. 0.05 is the coarsest scale that still traces each domain's real
# outline, and the finest at which that outline is a single connected shape.
EXTENT_CELL_DEG = 0.05

# Grid ids the current index resolves for each configured site, frozen from a
# verified build. Verification replays the site coordinates (not the resolved
# nodes) because asking which node is nearest to a node answers itself.
VERIFY_GIDS = {
    "US_Alaska_Kodiak": 915331,
    "US_California_Humboldt_Bay": 596791,
    "US_California_San_Luis_Obispo": 671590,
    "US_California_Scripps_Pier": 175332,
    "US_Hawaii_BigIsland_NELHA": 602158,
    "US_Hawaii_Oahu_WETS": 635201,
    "US_Massachusetts_WoodsHole_PioneerWec": 1792676,
    "US_North_Carolina_Jenettes_Pier": 1547084,
    "US_Oregon_PacWave_North": 576280,
    "US_Oregon_PacWave_South": 479519,
    "US_PuertoRico_North_Shore": 3712301,
}


def node_file(domain: str) -> Path:
    """Return the LFS parquet path for one domain."""
    return LFS_DIR / f"nodes_{domain}_{INDEX_VERSION}.parquet"


def build_domain(domain: str) -> dict[str, Any]:
    """Read one domain's coordinates from S3 and write its node parquet."""
    print(f"\n{domain}")
    print(f"  reading {CONFIG.s3_bucket_uri}/{DOMAINS[domain]['grid_key']}")
    coords = index_build.read_coordinates(domain)
    print(f"  {len(coords):,} nodes")
    out = node_file(domain)
    out.parent.mkdir(parents=True, exist_ok=True)
    index_build.write_domain_nodes(coords, out)
    print(f"  wrote {out.name}  ({out.stat().st_size / 1e6:.1f} MB)")
    return index_build.domain_bounds(domain, coords)


def build_extents(con: Any, domains: list[str]) -> str:
    """Write the occupancy-cell gate, derived from the node parquet on disk.

    Reads the built indexes rather than S3: the cells are a pure function of
    the coordinates, so rebuilding the gate never needs the source files again.
    """
    import pandas as pd

    parts = []
    for domain in domains:
        path = node_file(domain)
        if not path.exists():
            continue
        frame = con.execute(
            f"""
            SELECT DISTINCT
                CAST(floor(lat_fixed / {float(COORD_SCALE)} / {EXTENT_CELL_DEG}) AS INTEGER)
                    AS lat_cell,
                CAST(floor(lon_fixed / {float(COORD_SCALE)} / {EXTENT_CELL_DEG}) AS INTEGER)
                    AS lon_cell
            FROM read_parquet('{path.as_posix()}')
            """
        ).df()
        frame.insert(0, "domain", domain)
        parts.append(frame)
        print(f"  {domain:34s} {len(frame):>6,} occupancy cells")

    cells = pd.concat(parts, ignore_index=True).sort_values(["lat_cell", "lon_cell"])
    out = PKG_DATA / f"domain_extents_{INDEX_VERSION}.parquet"
    cells.to_parquet(out, index=False, compression="zstd")
    print(f"  wrote {out.name}  ({out.stat().st_size / 1e3:.0f} KB, {len(cells):,} rows)")
    return out.name


def build_footprints(con: Any, domains: list[str]) -> str:
    """Dissolve each domain's occupied cells into an outline, as GeoJSON.

    This is the domain's actual coverage, not a bounding box: it follows the
    coastline, leaves the interior of landmasses empty, and wraps the
    antimeridian correctly for Alaska. Cells are merged into horizontal runs
    before the union purely for speed, which produces identical geometry.
    """
    features = []
    for domain in domains:
        path = node_file(domain)
        if not path.exists():
            continue
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE runs AS
            SELECT i, min(j) AS j0, max(j) AS j1 FROM (
                SELECT i, j, j - row_number() OVER (PARTITION BY i ORDER BY j) AS grp
                FROM (
                    SELECT DISTINCT
                        CAST(floor(lat_fixed / {float(COORD_SCALE)} / {EXTENT_CELL_DEG})
                             AS INTEGER) AS i,
                        CAST(floor(lon_fixed / {float(COORD_SCALE)} / {EXTENT_CELL_DEG})
                             AS INTEGER) AS j
                    FROM read_parquet('{path.as_posix()}')
                )
            ) GROUP BY i, grp
            """
        )
        geometry = con.execute(
            f"""
            SELECT ST_AsGeoJSON(ST_Union_Agg(ST_MakeEnvelope(
                j0 * {EXTENT_CELL_DEG}, i * {EXTENT_CELL_DEG},
                (j1 + 1) * {EXTENT_CELL_DEG}, (i + 1) * {EXTENT_CELL_DEG})))
            FROM runs
            """
        ).fetchone()[0]
        geometry = json.loads(geometry)
        parts = len(geometry["coordinates"]) if geometry["type"] == "MultiPolygon" else 1
        features.append(
            {
                "type": "Feature",
                "geometry": geometry,
                "properties": {
                    "domain": domain,
                    "endpoint": DOMAIN_ENDPOINTS.get(domain),
                    "parts": parts,
                },
            }
        )
        print(f"  {domain:34s} {parts:>5,} part(s)")

    out = PKG_DATA / f"domain_footprints_{INDEX_VERSION}.geojson"
    out.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "properties": {
                    "description": (
                        "Coverage of each WPTO wave hindcast domain, dissolved from the "
                        "cells that contain grid nodes. Not a bounding box and not a hull."
                    ),
                    "cell_size_deg": EXTENT_CELL_DEG,
                    "source_year": 2010,
                },
                "features": features,
            }
        )
        + "\n"
    )
    print(f"  wrote {out.name}  ({out.stat().st_size / 1e6:.2f} MB)")
    return out.name


def write_registry() -> None:
    """SHA256 of every LFS node file, for pooch to verify downloads against."""
    lines = ["# filename  sha256 -- regenerated by scripts/build_wave_node_index.py"]
    for path in sorted(LFS_DIR.glob("*.parquet")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{path.name}  sha256:{digest}")
    out = CONFIG.index_registry_file
    out.write_text("\n".join(lines) + "\n")
    print(f"  wrote {out.name}  ({len(lines) - 1} files)")


def verify() -> bool:
    """Check the index reproduces the frozen grid id for every configured site."""
    from us_marine_energy_resource.wave_hindcast import nodes

    print(f"\nVerifying {len(VERIFY_GIDS)} sites (queried from site coordinates):")
    ok = True
    for site, expected in sorted(VERIFY_GIDS.items()):
        lat, lon, domain = SITES[site]
        found = nodes.nearest(lat, lon, domain=domain)
        assert isinstance(found, nodes.WaveNode)
        if found.location_id == expected:
            print(f"  PASS  {site:42s} gid={expected:<8} ({found.distance_m:6.1f} m)")
        else:
            ok = False
            print(
                f"  FAIL  {site:42s} picked gid={found.location_id} at "
                f"{found.distance_m:.1f} m, expected gid={expected}"
            )
    return ok


def main() -> None:
    """Run the build and verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", action="append", choices=sorted(DOMAINS), metavar="NAME")
    parser.add_argument("--verify", action="store_true", help="verify only, do not rebuild")
    parser.add_argument(
        "--extents-only",
        action="store_true",
        help="rebuild just the gate and footprints from the node parquet on disk (no S3)",
    )
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    if not args.verify:
        import duckdb

        con = duckdb.connect()
        con.execute("INSTALL spatial; LOAD spatial;")

        index_path = CONFIG.index_file
        index = json.loads(index_path.read_text()) if index_path.exists() else {}
        bounds = {b["domain"]: b for b in index.get("bounds", [])}

        if not args.extents_only:
            for domain in args.domain or list(DOMAINS):
                bounds[domain] = build_domain(domain)

        rows = [bounds[d] for d in DOMAINS if d in bounds]
        print("\nDomain gate:")
        extents_file = build_extents(con, [b["domain"] for b in rows])
        print("\nDomain footprints:")
        footprints_file = build_footprints(con, [b["domain"] for b in rows])

        index = {
            "index_version": INDEX_VERSION,
            "coord_scale": COORD_SCALE,
            "extent_cell_deg": EXTENT_CELL_DEG,
            "source_bucket": CONFIG.s3_bucket_uri,
            "source_year": 2010,
            "extents_file": extents_file,
            "footprints_file": footprints_file,
            "node_files": {
                b["domain"]: f"nodes_{b['domain']}_{INDEX_VERSION}.parquet" for b in rows
            },
            "bounds": rows,
        }
        index_path.write_text(json.dumps(index, indent=2) + "\n")
        print(f"\nwrote {index_path.name}")
        write_registry()

    if not verify():
        sys.exit("\nVerification FAILED: the index no longer resolves the frozen grid ids.")
    print("\nOK")


if __name__ == "__main__":
    main()
