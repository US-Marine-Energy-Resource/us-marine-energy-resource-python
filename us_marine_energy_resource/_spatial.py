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

All geometry math goes through :mod:`.gis` (the DuckDB spatial extension),
per the single-source GIS rule.

Public API
----------
find_faces_point(lat, lon, location=None)  -> pd.DataFrame
find_faces_area(coords, location=None)     -> pd.DataFrame
find_faces_line(coords, location=None)     -> pd.DataFrame

Each function returns a DataFrame with columns:
  location, face_id, lat_fixed_precision, lon_fixed_precision,
  c1_lat, c1_lon, c2_lat, c2_lon, c3_lat, c3_lon,
  distance_km  (query point → nearest point on the face; 0.0 for containing)
  [line queries also return: frac_along, chord_m]
"""

from __future__ import annotations

import importlib.resources
import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from . import gis
from .gis import Geom, LatLon

if TYPE_CHECKING:
    import duckdb

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
    return gis.connection()


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
    "point": ("Coordinate ({lat:.5f}, {lon:.5f}) is not inside any dataset domain.\n{domains}"),
    "area": ("Query polygon does not intersect any dataset domain.\n{domains}"),
    "line": ("Query transect does not pass through any dataset domain.\n{domains}"),
}


# ---------------------------------------------------------------------------
# Geometry adapters (the (lat, lon) tuple API adapted onto gis types)
# ---------------------------------------------------------------------------


def _latlons(coords: Sequence[tuple[float, float]]) -> list[LatLon]:
    return [LatLon(lat=lat, lon=lon) for lat, lon in coords]


# ---------------------------------------------------------------------------
# SQL constants
# ---------------------------------------------------------------------------

_FACE_COLS = (
    "location, face_id, lat_fixed_precision, lon_fixed_precision, "
    "c1_lat, c1_lon, c2_lat, c2_lon, c3_lat, c3_lon"
)

# One mesh face as a geometry, from the triangle corner columns.
_TRIANGLE = gis.make_polygon(
    [
        gis.column_point(lon_sql="c1_lon::DOUBLE", lat_sql="c1_lat::DOUBLE"),
        gis.column_point(lon_sql="c2_lon::DOUBLE", lat_sql="c2_lat::DOUBLE"),
        gis.column_point(lon_sql="c3_lon::DOUBLE", lat_sql="c3_lat::DOUBLE"),
    ]
)

# Coordinate precision: decimal places stored in parquet centroid columns
# (≈ 1 cm ground resolution).
_COORD_DECIMAL_PRECISION: int = 7
_COORD_PRECISION_SCALE: int = 10**_COORD_DECIMAL_PRECISION


def _face_distance_km_expr(pt: LatLon) -> str:
    """Distance from query point to the nearest point ON the triangle face.

    Returns 0.0 for containing faces (query is inside the triangle).
    Returns exact edge distance for non-containing faces.
    Replaces centroid-based distance, which is an approximation.
    """
    query_pt = gis.point(pt)
    closest = gis.closest_point(_TRIANGLE, query_pt)
    return f"{gis.distance_m_sql(closest, query_pt)} / 1000.0"


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
            f"  {loc}: lat [{lat_min:.3f}, {lat_max:.3f}]  lon [{lon_min:.3f}, {lon_max:.3f}]"
        )
    return "\n".join(lines)


def _intersects_boundary(
    con: duckdb.DuckDBPyConnection,
    boundary: Geom,
    query: Geom,
) -> bool:
    result = con.execute(f"SELECT {gis.intersects_sql(boundary, query)}").fetchone()
    return bool(result and result[0])


def _resolve_locations(
    con: duckdb.DuckDBPyConnection,
    query_standard: Geom,
    query_360: Geom,
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
        query = query_360 if crosses_antimeridian else query_standard
        if _intersects_boundary(con, gis.from_wkt(boundary_wkt), query):
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
    pt = LatLon(lat=lat, lon=lon)
    matched_locs = _resolve_locations(
        con, gis.point(pt), gis.point(pt, wrap_360=True), "point", location, lat=lat, lon=lon
    )

    dist_expr = _face_distance_km_expr(pt)
    parts: list[pd.DataFrame] = []

    for loc in matched_locs:
        frame = con.execute(f"""
            SELECT {_FACE_COLS},
                {dist_expr} AS distance_km,
                {gis.contains_sql(_TRIANGLE, gis.point(pt))}
                    AS is_containing
            FROM read_parquet('{_geometry_path(loc)}')
            ORDER BY is_containing DESC, distance_km ASC
            LIMIT 1
        """).df()
        parts.append(frame)

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

    con = _connect()
    ring = gis.Polygon(_latlons(coords))
    area = ring.geom()
    matched_locs = _resolve_locations(con, area, ring.geom(wrap_360=True), "area", location)

    face_closest = gis.closest_point(_TRIANGLE, area)
    area_closest = gis.closest_point(area, face_closest)
    parts: list[pd.DataFrame] = []

    for loc in matched_locs:
        frame = con.execute(f"""
            SELECT {_FACE_COLS},
                {gis.distance_m_sql(area_closest, face_closest)} / 1000.0
                    AS distance_km
            FROM read_parquet('{_geometry_path(loc)}')
            WHERE {gis.intersects_sql(_TRIANGLE, area)}
            ORDER BY distance_km ASC
        """).df()
        parts.append(frame)

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

    con = _connect()
    path = gis.Line(_latlons(coords))
    line = path.geom()
    matched_locs = _resolve_locations(con, line, path.geom(wrap_360=True), "line", location)

    parts: list[pd.DataFrame] = []

    for loc in matched_locs:
        frame = con.execute(f"""
            SELECT {_FACE_COLS},
                {gis.line_locate_sql(line, gis.centroid(_TRIANGLE))}
                    AS frac_along,
                {gis.length_m_sql(gis.intersection(_TRIANGLE, line))}
                    AS chord_m
            FROM read_parquet('{_geometry_path(loc)}')
            WHERE {gis.intersects_sql(_TRIANGLE, line)}
            ORDER BY frac_along ASC
        """).df()
        parts.append(frame)

    if not parts:
        return pd.DataFrame()
    # chord_m is the true control-volume width along the transect: the chord
    # where the line crosses each triangle (metres), the correct dx for flux.
    result = pd.concat(parts, ignore_index=True)
    return result.sort_values("frac_along").reset_index(drop=True)
