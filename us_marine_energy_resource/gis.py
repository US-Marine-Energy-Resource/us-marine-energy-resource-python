"""Single source of GIS truth: DuckDB spatial queries behind typed geometry.

Every geospatial computation in the package runs through this module.
Coordinates enter as :class:`LatLon`, become SQL only through the factories
here, and the axis rules live on :class:`LatLon` and :func:`_lat_first`.
A test bans spatial SQL everywhere else.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .explore.lazy import lazy_import


@dataclass(frozen=True, kw_only=True)
class LatLon:
    """A geographic coordinate in degrees, built with explicit keywords.

    The geometry factories map ``lat`` to y and ``lon`` to x whenever a
    coordinate becomes SQL or WKT.
    """

    lat: float
    lon: float

    def distance_m(self, other: LatLon) -> float:
        """Return the great-circle distance to another coordinate, in metres.

        Parameters
        ----------
        other : LatLon
            The far end.

        Returns
        -------
        float
            Metres on DuckDB's sphere.
        """
        query = f"SELECT {distance_m_sql(point(self), point(other))}"
        row = connection().execute(query).fetchone()
        assert row is not None
        return float(row[0])

    def bearing_to(self, other: LatLon) -> float:
        """Return the initial geodesic bearing to ``other``, in degrees.

        Computed by pyproj's geodesic engine rather than DuckDB, whose
        ``ST_Azimuth`` is planar and has no spheroid variant.

        Parameters
        ----------
        other : LatLon
            The far end.

        Returns
        -------
        float
            Degrees clockwise from north, 0 to 360. Zero when the two
            coordinates coincide.
        """
        if self == other:
            return 0.0
        azimuth, _, _ = _geod().inv(self.lon, self.lat, other.lon, other.lat)
        return float(azimuth % 360.0)


@dataclass(frozen=True)
class Geom:
    """A geometry-valued SQL expression, x as longitude.

    Opaque on purpose: instances come only from the factories and
    composition helpers in this module, so coordinates and axis order never
    appear as strings at call sites.
    """

    expr: str


class SpatialUnavailableError(RuntimeError):
    """The DuckDB spatial extension could not be loaded by any route."""


_state: dict[str, Any] = {}


def _wheel_extension(version: str) -> str | None:
    """Path to the extension shipped by duckdb-extension-spatial, if it matches.

    The wheel stores one binary per duckdb version, so a wheel that does not
    match the running duckdb is skipped rather than loaded.

    Parameters
    ----------
    version : str
        The running duckdb version.

    Returns
    -------
    str or None
        Path to the extension file, or None when the package is absent or
        carries a different version.
    """
    spec = importlib.util.find_spec("duckdb_extension_spatial")
    if spec is None or spec.origin is None:
        return None
    path = Path(spec.origin).parent / "extensions" / f"v{version}" / "spatial.duckdb_extension"
    return path.as_posix() if path.exists() else None


def _load_spatial(con: Any, version: str) -> None:
    """Load the spatial extension: local cache, then the wheel, then INSTALL.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        The connection to load into.
    version : str
        The running duckdb version, used to pick the wheel binary.

    Raises
    ------
    SpatialUnavailableError
        Every route failed, usually offline with nothing cached.
    """
    try:
        con.execute("LOAD spatial")
        return
    except Exception:  # fall through to the next route
        pass
    wheel = _wheel_extension(version)
    if wheel is not None:
        con.execute(f"LOAD '{wheel}'")
        return
    try:
        con.execute("INSTALL spatial")
        con.execute("LOAD spatial")
    except Exception as exc:
        raise SpatialUnavailableError(
            "could not load the DuckDB spatial extension. Install the "
            "duckdb-extension-spatial package for offline use, or run "
            "INSTALL spatial once with network access."
        ) from exc


def _geod() -> Any:
    """Return the shared WGS84 geodesic engine.

    Returns
    -------
    pyproj.Geod
        Created on first use and reused after that.
    """
    if "geod" not in _state:
        pyproj = lazy_import("pyproj", "computing geodesic bearings")
        _state["geod"] = pyproj.Geod(ellps="WGS84")
    return _state["geod"]


def connection() -> Any:
    """Return the shared DuckDB connection with the spatial extension loaded.

    Returns
    -------
    duckdb.DuckDBPyConnection
        The connection, created on first use and reused after that.

    Raises
    ------
    SpatialUnavailableError
        The spatial extension could not be loaded by any route.
    """
    if "con" not in _state:
        duckdb = lazy_import("duckdb", "running spatial queries")
        con = duckdb.connect()
        _load_spatial(con, duckdb.__version__)
        _state["con"] = con
    return _state["con"]


# ---------------------------------------------------------------------------
# Geometry factories: the only ways a coordinate becomes SQL
# ---------------------------------------------------------------------------


def _wkt_lon(lon: float, wrap_360: bool) -> float:
    """Return the longitude to write, optionally translated into 0-360 space.

    The shipped Aleutian Islands boundary stores longitude in 0-360 space so
    it stays contiguous across the antimeridian, and queries against it need
    the same translation. This is the one home of that convention.

    Parameters
    ----------
    lon : float
        Longitude in degrees, -180 to 180.
    wrap_360 : bool
        Translate negative longitudes into 0-360 space.

    Returns
    -------
    float
        The longitude to write.
    """
    return lon + 360.0 if wrap_360 and lon < 0.0 else lon


def point(p: LatLon, wrap_360: bool = False) -> Geom:
    """Build a point geometry from a coordinate.

    Parameters
    ----------
    p : LatLon
        The coordinate.
    wrap_360 : bool, default False
        Translate into 0-360 longitude space (see :func:`_wkt_lon`).

    Returns
    -------
    Geom
        The point, x as longitude.
    """
    lon = _wkt_lon(float(p.lon), wrap_360)
    return Geom(f"ST_Point({lon!r}, {float(p.lat)!r})")


def column_point(*, lon_sql: str, lat_sql: str) -> Geom:
    """Build a point geometry from column expressions, keyword-only.

    The keywords are the guard: a call site cannot swap the axes without
    saying so out loud.

    Parameters
    ----------
    lon_sql, lat_sql : str
        SQL expressions for the longitude and latitude columns.

    Returns
    -------
    Geom
        The point, x as longitude.
    """
    return Geom(f"ST_Point({lon_sql}, {lat_sql})")


def from_wkt(wkt: str) -> Geom:
    """Wrap stored WKT text, such as a boundary string shipped in a parquet.

    Parameters
    ----------
    wkt : str
        WKT in the x=longitude convention (0-360 space where the stored
        data uses it).

    Returns
    -------
    Geom
        The parsed geometry.
    """
    return Geom(f"ST_GeomFromText('{wkt}')")


def make_polygon(points: Sequence[Geom]) -> Geom:
    """Build a polygon from point geometries, closing the ring if needed.

    Parameters
    ----------
    points : sequence of Geom
        The ring's points, at least three.

    Returns
    -------
    Geom
        The polygon.
    """
    ring = list(points)
    if ring[0].expr != ring[-1].expr:
        ring.append(ring[0])
    joined = ", ".join(g.expr for g in ring)
    return Geom(f"ST_MakePolygon(ST_MakeLine(ARRAY[{joined}]))")


class Line:
    """A polyline of coordinates with its DuckDB-backed operations.

    Parameters
    ----------
    coords : sequence of LatLon
        The vertices, at least two.
    """

    def __init__(self, coords: Sequence[LatLon]) -> None:
        """Store the vertices as a tuple."""
        self.coords = tuple(coords)

    def wkt(self, wrap_360: bool = False) -> str:
        """Return the LINESTRING WKT, x as longitude.

        Parameters
        ----------
        wrap_360 : bool, default False
            Translate into 0-360 longitude space (see :func:`_wkt_lon`).

        Returns
        -------
        str
            The WKT text.
        """
        pts = [f"{_wkt_lon(float(p.lon), wrap_360)} {float(p.lat)}" for p in self.coords]
        return f"LINESTRING({', '.join(pts)})"

    def geom(self, wrap_360: bool = False) -> Geom:
        """Return the polyline as a geometry expression.

        Parameters
        ----------
        wrap_360 : bool, default False
            Translate into 0-360 longitude space (see :func:`_wkt_lon`).

        Returns
        -------
        Geom
            The polyline geometry.
        """
        return from_wkt(self.wkt(wrap_360))

    def chord_fractions(self, triangle: Polygon) -> tuple[float, float] | None:
        """Locate the fractions along this line where it crosses a triangle.

        Parameters
        ----------
        triangle : Polygon
            The triangle to intersect with.

        Returns
        -------
        tuple of (float, float) or None
            Normalized 0 to 1 positions of the chord's ends along the line,
            or None when the intersection has no line part (a miss or a
            vertex graze).
        """
        line_g = self.geom()
        cross = intersection(triangle.geom(), line_g)
        row = (
            connection()
            .execute(f"""
                WITH parts AS (
                    SELECT UNNEST(ST_Dump({cross.expr})).geom AS g
                ),
                ends AS (
                    SELECT {line_locate_sql(line_g, Geom("ST_StartPoint(g)"))} AS f
                    FROM parts WHERE ST_GeometryType(g) = 'LINESTRING'
                    UNION ALL
                    SELECT {line_locate_sql(line_g, Geom("ST_EndPoint(g)"))}
                    FROM parts WHERE ST_GeometryType(g) = 'LINESTRING'
                )
                SELECT min(f), max(f), count(*) FROM ends
            """)
            .fetchone()
        )
        if row is None or row[0] is None or row[2] < 2:
            return None
        return float(row[0]), float(row[1])

    def intersections(self, polygon: Polygon) -> list[list[LatLon]]:
        """Return the line parts where this line crosses a polygon, for drawing.

        Parameters
        ----------
        polygon : Polygon
            The polygon to intersect with.

        Returns
        -------
        list of list of LatLon
            One vertex list per line part, empty when the two do not cross.
        """
        import json

        cross = intersection(polygon.geom(), self.geom())
        rows = (
            connection()
            .execute(f"SELECT ST_AsGeoJSON(UNNEST(ST_Dump({cross.expr})).geom)")
            .fetchall()
        )
        parts: list[list[LatLon]] = []
        for (text,) in rows:
            geo = json.loads(text)
            # An empty intersection still dumps one row, as an empty LineString.
            if geo["type"] == "LineString" and len(geo["coordinates"]) >= 2:
                parts.append([LatLon(lat=c[1], lon=c[0]) for c in geo["coordinates"]])
        return parts


class Polygon:
    """A polygon ring of coordinates with its DuckDB-backed operations.

    Parameters
    ----------
    ring : sequence of LatLon
        The ring vertices, at least three. Closed automatically.
    """

    def __init__(self, ring: Sequence[LatLon]) -> None:
        """Store the ring vertices as a tuple."""
        self.ring = tuple(ring)

    def wkt(self, wrap_360: bool = False) -> str:
        """Return the POLYGON WKT, x as longitude, closed if it is not already.

        Parameters
        ----------
        wrap_360 : bool, default False
            Translate into 0-360 longitude space (see :func:`_wkt_lon`).

        Returns
        -------
        str
            The WKT text.
        """
        pts = [f"{_wkt_lon(float(p.lon), wrap_360)} {float(p.lat)}" for p in self.ring]
        if pts[0] != pts[-1]:
            pts.append(pts[0])
        return f"POLYGON(({', '.join(pts)}))"

    def geom(self, wrap_360: bool = False) -> Geom:
        """Return the polygon as a geometry expression.

        Parameters
        ----------
        wrap_360 : bool, default False
            Translate into 0-360 longitude space (see :func:`_wkt_lon`).

        Returns
        -------
        Geom
            The polygon geometry.
        """
        return from_wkt(self.wkt(wrap_360))


# ---------------------------------------------------------------------------
# Composition and measurement: typed geometry in, SQL out
# ---------------------------------------------------------------------------


def _lat_first(g: Geom) -> str:
    """Emit SQL flipping a geometry into the latitude first axis order.

    DuckDB's geodesic measurement functions (``ST_Distance_Sphere``,
    ``ST_Distance_Spheroid``, ``ST_Length_Spheroid``) document their input
    as EPSG:4326 with latitude first, the opposite of the x=longitude order
    every geometry in this package uses. This is the only place
    ``ST_FlipCoordinates`` may appear.

    Parameters
    ----------
    g : Geom
        The geometry, x as longitude.

    Returns
    -------
    str
        The flipped expression.
    """
    return f"ST_FlipCoordinates({g.expr})"


def distance_m_sql(a: Geom, b: Geom) -> str:
    """Emit SQL for the great-circle distance between two geometries, in metres.

    Both inputs pass through :func:`_lat_first`, since ``ST_Distance_Sphere``
    reads latitude first. DuckDB's sphere radius is 6,371,000 m.

    Parameters
    ----------
    a, b : Geom
        The two geometries.

    Returns
    -------
    str
        The distance expression.
    """
    return f"ST_Distance_Sphere({_lat_first(a)}, {_lat_first(b)})"


def length_m_sql(g: Geom) -> str:
    """Emit SQL for the geodesic length of a line geometry, in metres.

    The input passes through :func:`_lat_first`, since ``ST_Length_Spheroid``
    reads latitude first. Point parts and empty geometries measure zero, and
    multi-part lines sum.

    Parameters
    ----------
    g : Geom
        The geometry.

    Returns
    -------
    str
        The length expression.
    """
    return f"ST_Length_Spheroid({_lat_first(g)})"


def contains_sql(outer: Geom, inner: Geom) -> str:
    """Emit SQL testing whether ``outer`` contains ``inner``.

    Parameters
    ----------
    outer, inner : Geom
        The geometries.

    Returns
    -------
    str
        A boolean expression.
    """
    return f"ST_Contains({outer.expr}, {inner.expr})"


def intersects_sql(a: Geom, b: Geom) -> str:
    """Emit SQL testing whether two geometries intersect.

    Parameters
    ----------
    a, b : Geom
        The geometries.

    Returns
    -------
    str
        A boolean expression.
    """
    return f"ST_Intersects({a.expr}, {b.expr})"


def line_locate_sql(line: Geom, pt: Geom) -> str:
    """Emit SQL for the normalized 0 to 1 position of a point along a line.

    Parameters
    ----------
    line : Geom
        The line to measure along.
    pt : Geom
        The point to locate.

    Returns
    -------
    str
        A fraction expression.
    """
    return f"ST_LineLocatePoint({line.expr}, {pt.expr})"


def intersection(a: Geom, b: Geom) -> Geom:
    """Compose the intersection of two geometries.

    Parameters
    ----------
    a, b : Geom
        The geometries.

    Returns
    -------
    Geom
        Their intersection.
    """
    return Geom(f"ST_Intersection({a.expr}, {b.expr})")


def closest_point(on: Geom, to: Geom) -> Geom:
    """Compose the point on one geometry nearest to another.

    Parameters
    ----------
    on : Geom
        The geometry the result lies on.
    to : Geom
        The geometry measured toward.

    Returns
    -------
    Geom
        The nearest point on ``on``.
    """
    return Geom(f"ST_ClosestPoint({on.expr}, {to.expr})")


def centroid(g: Geom) -> Geom:
    """Compose the centroid of a geometry.

    Parameters
    ----------
    g : Geom
        The geometry.

    Returns
    -------
    Geom
        Its centroid point.
    """
    return Geom(f"ST_Centroid({g.expr})")
