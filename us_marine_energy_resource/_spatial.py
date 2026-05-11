"""
DuckDB-backed spatial queries against the bundled geometry index.

All public functions follow this two-layer execution model:

  Layer 1 — Location gate (location_bounds.parquet, 5 rows)
    Determines which dataset location(s) a query geometry intersects.
    Raises a query-type-specific error if the query is outside all domains.
    AK_aleutian_islands boundaries are stored in 0-360° longitude space;
    the gate automatically applies the corresponding lon transform to queries.

  Layer 2 — Triangle query (geometry_{location}.parquet per matched location)
    Runs an exact ST_Contains / ST_Intersects test against triangle vertices.
    Results from multiple locations are unioned into a single DataFrame.

Public API
----------
find_faces_point(lat, lon, location=None)  -> pd.DataFrame
find_faces_area(coords, location=None)     -> pd.DataFrame
find_faces_line(coords, location=None)     -> pd.DataFrame

Each function returns a DataFrame with columns:
  location, face_id, lat_e7, lon_e7,
  c1_lat, c1_lon, c2_lat, c2_lon, c3_lat, c3_lon,
  distance_km  (centroid → query point; 0.0 for containing faces)
  [line queries also return: frac_along, distance_from_line_m]
"""

from __future__ import annotations

import importlib.resources
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _data_dir() -> Path:
    with importlib.resources.path("us_marine_energy_resource", "data") as p:
        return Path(p)


_index_cache: dict[str, Any] | None = None


def _data_index() -> dict[str, Any]:
    global _index_cache
    if _index_cache is not None:
        return _index_cache
    index_path = _data_dir() / "data_index.json"
    if index_path.exists():
        with open(index_path) as f:
            loaded: dict[str, Any] = json.load(f)
            _index_cache = loaded
    else:
        _index_cache = {}
    return _index_cache


def _geometry_path(location: str) -> str:
    idx = _data_index()
    if "geometry_files" in idx and location in idx["geometry_files"]:
        return (_data_dir() / idx["geometry_files"][location]).as_posix()
    # Fallback: glob for any versioned file, then unversioned.
    data_dir = _data_dir()
    matches = sorted(data_dir.glob(f"geometry_{location}_v*.parquet"))
    if matches:
        return matches[-1].as_posix()
    return (data_dir / f"geometry_{location}.parquet").as_posix()


def _bounds_path() -> str:
    idx = _data_index()
    if "bounds_file" in idx:
        return (_data_dir() / idx["bounds_file"]).as_posix()
    # Fallback: glob for any versioned file, then unversioned.
    data_dir = _data_dir()
    matches = sorted(data_dir.glob("location_bounds_v*.parquet"))
    if matches:
        return matches[-1].as_posix()
    return (data_dir / "location_bounds.parquet").as_posix()


# ---------------------------------------------------------------------------
# DuckDB connection
# ---------------------------------------------------------------------------


def _connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("LOAD spatial;")
    return con


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class OutsideDomainError(ValueError):
    """Query geometry does not intersect any dataset domain."""


class PointOutsideDomainError(OutsideDomainError):
    """Point coordinate is not inside any dataset domain."""


class AreaOutsideDomainError(OutsideDomainError):
    """Area polygon does not intersect any dataset domain."""


class TransectOutsideDomainError(OutsideDomainError):
    """Line/transect does not intersect any dataset domain."""


_ERROR_CLASSES: dict[str, type[OutsideDomainError]] = {
    "point": PointOutsideDomainError,
    "area": AreaOutsideDomainError,
    "line": TransectOutsideDomainError,
}

_ERROR_MESSAGES: dict[str, str] = {
    "point": (
        "Coordinate ({lat:.5f}, {lon:.5f}) is not inside any dataset domain.\n"
        "{domains}"
    ),
    "area": (
        "Query polygon does not intersect any dataset domain.\n"
        "{domains}"
    ),
    "line": (
        "Query transect does not pass through any dataset domain.\n"
        "{domains}"
    ),
}


# ---------------------------------------------------------------------------
# WKT geometry builders
# ---------------------------------------------------------------------------


def _point_wkt(lat: float, lon: float) -> str:
    return f"POINT({lon} {lat})"


def _point_wkt_360(lat: float, lon: float) -> str:
    lon360 = lon + 360.0 if lon < 0.0 else lon
    return f"POINT({lon360} {lat})"


def _coords_to_ring(coords: Sequence[tuple[float, float]]) -> list[str]:
    pts = [f"{lon} {lat}" for lat, lon in coords]
    if pts[0] != pts[-1]:
        pts.append(pts[0])
    return pts


def _polygon_wkt(coords: Sequence[tuple[float, float]]) -> str:
    pts = _coords_to_ring(coords)
    return f"POLYGON(({', '.join(pts)}))"


def _polygon_wkt_360(coords: Sequence[tuple[float, float]]) -> str:
    def _lon360(lon: float) -> float:
        return lon + 360.0 if lon < 0.0 else lon
    pts = [f"{_lon360(lon)} {lat}" for lat, lon in coords]
    if pts[0] != pts[-1]:
        pts.append(pts[0])
    return f"POLYGON(({', '.join(pts)}))"


def _line_wkt(coords: Sequence[tuple[float, float]]) -> str:
    pts = [f"{lon} {lat}" for lat, lon in coords]
    return f"LINESTRING({', '.join(pts)})"


def _line_wkt_360(coords: Sequence[tuple[float, float]]) -> str:
    def _lon360(lon: float) -> float:
        return lon + 360.0 if lon < 0.0 else lon
    pts = [f"{_lon360(lon)} {lat}" for lat, lon in coords]
    return f"LINESTRING({', '.join(pts)})"


# ---------------------------------------------------------------------------
# SQL constants
# ---------------------------------------------------------------------------

_FACE_COLS = (
    "location, face_id, lat_e7, lon_e7, "
    "c1_lat, c1_lon, c2_lat, c2_lon, c3_lat, c3_lon"
)

_TRIANGLE = """\
ST_MakePolygon(ST_MakeLine(ARRAY[
    ST_Point(c1_lon::DOUBLE, c1_lat::DOUBLE),
    ST_Point(c2_lon::DOUBLE, c2_lat::DOUBLE),
    ST_Point(c3_lon::DOUBLE, c3_lat::DOUBLE),
    ST_Point(c1_lon::DOUBLE, c1_lat::DOUBLE)
]))"""

# Bounding-box margin for centroid pre-filter (~0.1° ≈ 11 km).
_MARGIN_E7 = 1_000_000


def _face_distance_km_expr(lat: float, lon: float) -> str:
    """Distance from query point to the nearest point ON the triangle face.

    Returns 0.0 for containing faces (query is inside the triangle).
    Returns exact edge distance for non-containing faces.
    Replaces centroid-based distance, which is an approximation.
    """
    query_pt = f"ST_Point({lon}::DOUBLE, {lat}::DOUBLE)"
    return (
        f"ST_Distance_Sphere("
        f"    ST_ClosestPoint({_TRIANGLE}, {query_pt}),"
        f"    {query_pt}"
        f") / 1000.0"
    )


# ---------------------------------------------------------------------------
# Location gate
# ---------------------------------------------------------------------------


def _domain_summary(con: duckdb.DuckDBPyConnection) -> str:
    rows = con.execute(f"""
        SELECT location, lat_min, lat_max, lon_min, lon_max
        FROM read_parquet('{_bounds_path()}')
        ORDER BY location
    """).fetchall()
    lines = ["Available domains:"]
    for loc, lat_min, lat_max, lon_min, lon_max in rows:
        lines.append(
            f"  {loc}: lat [{lat_min:.3f}, {lat_max:.3f}]  "
            f"lon [{lon_min:.3f}, {lon_max:.3f}]"
        )
    return "\n".join(lines)


def _intersects_boundary(
    con: duckdb.DuckDBPyConnection,
    boundary_wkt: str,
    query_wkt: str,
) -> bool:
    result = con.execute(f"""
        SELECT ST_Intersects(
            ST_GeomFromText('{boundary_wkt}'),
            ST_GeomFromText('{query_wkt}')
        )
    """).fetchone()
    return bool(result and result[0])


def _resolve_locations(
    con: duckdb.DuckDBPyConnection,
    query_wkt_standard: str,
    query_wkt_360: str,
    query_kind: str,
    location: str | None,
    lat: float | None = None,
    lon: float | None = None,
) -> list[str]:
    """Return location names whose exact mesh boundary intersects the query.

    Raises a query-type-specific OutsideDomainError when no match is found.
    """
    bounds = con.execute(f"""
        SELECT location, boundary_wkt, crosses_antimeridian
        FROM read_parquet('{_bounds_path()}')
        {f"WHERE location = '{location}'" if location else ""}
        ORDER BY location
    """).fetchall()

    if location and not bounds:
        raise ValueError(f"Unknown location: '{location}'")

    matched: list[str] = []
    for loc, boundary_wkt, crosses_antimeridian in bounds:
        query_wkt = query_wkt_360 if crosses_antimeridian else query_wkt_standard
        if _intersects_boundary(con, boundary_wkt, query_wkt):
            matched.append(loc)

    if matched:
        return matched

    # Build a helpful error message.
    cls = _ERROR_CLASSES[query_kind]
    template = _ERROR_MESSAGES[query_kind]
    domains = _domain_summary(con)
    msg = template.format(lat=lat or 0.0, lon=lon or 0.0, domains=domains)
    raise cls(msg)


# ---------------------------------------------------------------------------
# Triangle query helpers
# ---------------------------------------------------------------------------


def _bbox_clause(lat_e7: int, lon_e7: int, margin: int = _MARGIN_E7) -> str:
    return (
        f"lat_e7 BETWEEN {lat_e7 - margin} AND {lat_e7 + margin} "
        f"AND lon_e7 BETWEEN {lon_e7 - margin} AND {lon_e7 + margin}"
    )


def _envelope_bbox_clause(
    lat_min: float, lat_max: float, lon_min: float, lon_max: float
) -> str:
    margin = _MARGIN_E7
    lat_min_e7 = int(lat_min * 1e7) - margin
    lat_max_e7 = int(lat_max * 1e7) + margin
    lon_min_e7 = int(lon_min * 1e7) - margin
    lon_max_e7 = int(lon_max * 1e7) + margin
    return (
        f"lat_e7 BETWEEN {lat_min_e7} AND {lat_max_e7} "
        f"AND lon_e7 BETWEEN {lon_min_e7} AND {lon_max_e7}"
    )


# ---------------------------------------------------------------------------
# Public query functions
# ---------------------------------------------------------------------------


def find_faces_point(
    lat: float,
    lon: float,
    location: str | None = None,
) -> pd.DataFrame:
    """Find the mesh face that contains (lat, lon).

    Returns a single-row DataFrame with the containing triangle.  If the point
    falls exactly on a triangle edge or outside the mesh, the nearest face
    centroid is returned instead (ORDER BY is_containing DESC, distance_km ASC).

    Parameters
    ----------
    lat, lon : float
        Query coordinate in decimal degrees (WGS84).
    location : str, optional
        Restrict the search to one dataset location.

    Raises
    ------
    PointOutsideDomainError
        If (lat, lon) is outside all dataset domains.
    """
    con = _connect()
    query_wkt = _point_wkt(lat, lon)
    query_wkt_360 = _point_wkt_360(lat, lon)

    matched_locs = _resolve_locations(
        con, query_wkt, query_wkt_360, "point", location, lat=lat, lon=lon
    )

    lat_e7 = round(lat * 1e7)
    lon_e7 = round(lon * 1e7)
    dist_expr = _face_distance_km_expr(lat, lon)
    parts: list[pd.DataFrame] = []

    for loc in matched_locs:
        df = con.execute(f"""
            SELECT {_FACE_COLS},
                {dist_expr} AS distance_km,
                ST_Contains({_TRIANGLE}, ST_Point({lon}::DOUBLE, {lat}::DOUBLE))
                    AS is_containing
            FROM read_parquet('{_geometry_path(loc)}')
            WHERE {_bbox_clause(lat_e7, lon_e7)}
            ORDER BY is_containing DESC, distance_km ASC
            LIMIT 1
        """).df()
        parts.append(df)

    result = pd.concat(parts, ignore_index=True)
    # Across locations, keep the single best match (containing first, then nearest).
    result = (
        result.sort_values(["is_containing", "distance_km"], ascending=[False, True])
        .head(1)
        .drop(columns=["is_containing"])
        .reset_index(drop=True)
    )
    return result


def find_faces_area(
    coords: Sequence[tuple[float, float]],
    location: str | None = None,
) -> pd.DataFrame:
    """Find all mesh faces that intersect a polygon.

    Parameters
    ----------
    coords : sequence of (lat, lon) tuples
        Ring coordinates defining the polygon.  Need not be closed.
    location : str, optional
        Restrict the search to one dataset location.

    Raises
    ------
    AreaOutsideDomainError
        If the polygon does not intersect any dataset domain.
    ValueError
        If fewer than 3 coordinate pairs are provided.
    """
    if len(coords) < 3:
        raise ValueError("Area query requires at least 3 coordinate pairs.")

    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]

    con = _connect()
    query_wkt = _polygon_wkt(coords)
    query_wkt_360 = _polygon_wkt_360(coords)

    matched_locs = _resolve_locations(
        con, query_wkt, query_wkt_360, "area", location
    )

    bbox = _envelope_bbox_clause(min(lats), max(lats), min(lons), max(lons))
    area_geom = f"ST_GeomFromText('{query_wkt}')"
    parts: list[pd.DataFrame] = []

    for loc in matched_locs:
        df = con.execute(f"""
            SELECT {_FACE_COLS},
                ST_Distance_Sphere(
                    ST_ClosestPoint({area_geom}, ST_ClosestPoint({_TRIANGLE}, {area_geom})),
                    ST_ClosestPoint({_TRIANGLE}, {area_geom})
                ) / 1000.0 AS distance_km
            FROM read_parquet('{_geometry_path(loc)}')
            WHERE {bbox}
              AND ST_Intersects({_TRIANGLE}, {area_geom})
            ORDER BY distance_km ASC
        """).df()
        parts.append(df)

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def find_faces_line(
    coords: Sequence[tuple[float, float]],
    location: str | None = None,
) -> pd.DataFrame:
    """Find all mesh faces that the line/transect passes through.

    Returns only faces where the line geometrically intersects the triangle.
    For proximity queries (faces within X metres of a line), build a buffered
    polygon and use :func:`find_faces_area` instead.

    Parameters
    ----------
    coords : sequence of (lat, lon) tuples
        Vertices of the polyline (at least 2 points).
    location : str, optional
        Restrict the search to one dataset location.

    Returns
    -------
    pd.DataFrame
        Matched faces sorted by position along the line.
        Extra column: ``frac_along`` (fractional position 0→1 of the face
        centroid projected onto the line).

    Raises
    ------
    TransectOutsideDomainError
        If the transect does not intersect any dataset domain.
    ValueError
        If fewer than 2 coordinate pairs are provided.
    """
    if len(coords) < 2:
        raise ValueError("Line query requires at least 2 coordinate pairs.")

    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]

    con = _connect()
    query_wkt = _line_wkt(coords)
    query_wkt_360 = _line_wkt_360(coords)

    matched_locs = _resolve_locations(
        con, query_wkt, query_wkt_360, "line", location
    )

    bbox = _envelope_bbox_clause(min(lats), max(lats), min(lons), max(lons))
    line_geom = f"ST_GeomFromText('{query_wkt}')"
    parts: list[pd.DataFrame] = []

    for loc in matched_locs:
        df = con.execute(f"""
            SELECT {_FACE_COLS},
                ST_LineLocatePoint({line_geom}, ST_Centroid({_TRIANGLE}))
                    AS frac_along
            FROM read_parquet('{_geometry_path(loc)}')
            WHERE {bbox}
              AND ST_Intersects({_TRIANGLE}, {line_geom})
            ORDER BY frac_along ASC
        """).df()
        parts.append(df)

    if not parts:
        return pd.DataFrame()
    result = pd.concat(parts, ignore_index=True)
    return result.sort_values("frac_along").reset_index(drop=True)
