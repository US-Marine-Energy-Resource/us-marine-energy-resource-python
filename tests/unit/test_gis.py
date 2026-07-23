"""The GIS seam: typed geometry, axis order, and the shared spatial connection."""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

from us_marine_energy_resource import gis
from us_marine_energy_resource.gis import Geom, LatLon

# Two points near PacWave, 149.115 m apart on DuckDB's 6371 km sphere.
A = LatLon(lat=44.5670485, lon=-124.22896475)
B = LatLon(lat=44.5682, lon=-124.2280)

TRIANGLE = gis.Polygon(
    [LatLon(lat=0.0, lon=0.0), LatLon(lat=0.0, lon=1.0), LatLon(lat=1.0, lon=0.0)]
)
CROSSING = gis.Line([LatLon(lat=0.25, lon=-1.0), LatLon(lat=0.25, lon=2.0)])
MISSING = gis.Line([LatLon(lat=5.0, lon=-1.0), LatLon(lat=5.0, lon=2.0)])


def _haversine_m(a: LatLon, b: LatLon) -> float:
    """Compute the reference great-circle distance on DuckDB's 6371 km sphere."""
    radius = 6371000.0
    p1, p2 = math.radians(a.lat), math.radians(b.lat)
    dlat = math.radians(b.lat - a.lat)
    dlon = math.radians(b.lon - a.lon)
    h = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(h))


def test_latlon_requires_keywords() -> None:
    """Coordinates cannot be built positionally, so the axes cannot swap."""
    with pytest.raises(TypeError):
        LatLon(44.5, -124.2)  # type: ignore[misc]


def test_point_puts_longitude_first() -> None:
    """The point factory emits x as longitude."""
    sql = gis.point(LatLon(lat=44.5, lon=-124.2)).expr
    assert sql.startswith("ST_Point(-124.2")
    assert sql.endswith("44.5)")


def test_point_wrap_360() -> None:
    """wrap_360 translates a negative longitude into 0-360 space."""
    assert gis.point(LatLon(lat=52.0, lon=-179.97), wrap_360=True).expr.startswith(
        "ST_Point(180.03"
    )


def test_distance_matches_reference_sphere() -> None:
    """A wrong axis order here would read about 129 m instead of 149 m."""
    assert A.distance_m(B) == pytest.approx(_haversine_m(A, B), rel=1e-9)


def test_distance_crosses_antimeridian() -> None:
    """The dateline needs no wrap math."""
    east = LatLon(lat=52.0, lon=179.985)
    west = LatLon(lat=52.0005, lon=-179.97)
    assert east.distance_m(west) == pytest.approx(_haversine_m(east, west), rel=1e-9)


def test_connection_is_shared() -> None:
    """Repeat calls reuse one connection instead of reloading the extension."""
    assert gis.connection() is gis.connection()


def test_polygon_wkt_closes_ring_and_wraps() -> None:
    """The ring closes itself and wrap_360 translates negative longitudes."""
    ring = gis.Polygon(
        [LatLon(lat=0.0, lon=-1.0), LatLon(lat=1.0, lon=-1.0), LatLon(lat=1.0, lon=1.0)]
    )
    assert ring.wkt() == "POLYGON((-1.0 0.0, -1.0 1.0, 1.0 1.0, -1.0 0.0))"
    assert ring.wkt(wrap_360=True) == "POLYGON((359.0 0.0, 359.0 1.0, 1.0 1.0, 359.0 0.0))"


def test_make_polygon_closes_ring() -> None:
    """A column-built polygon repeats its first point to close the ring."""
    geom = gis.make_polygon(
        [
            gis.column_point(lon_sql="a_lon", lat_sql="a_lat"),
            gis.column_point(lon_sql="b_lon", lat_sql="b_lat"),
            gis.column_point(lon_sql="c_lon", lat_sql="c_lat"),
        ]
    )
    assert geom.expr.count("ST_Point(a_lon, a_lat)") == 2


def test_line_chord_fractions_span_the_triangle() -> None:
    """A line through a triangle reports its entry and exit fractions."""
    fracs = CROSSING.chord_fractions(TRIANGLE)
    assert fracs is not None
    lo, hi = fracs
    assert lo == pytest.approx(1.0 / 3.0, abs=1e-9)
    assert hi == pytest.approx(0.5833333333, abs=1e-6)


def test_line_chord_fractions_none_for_miss() -> None:
    """A line that misses the triangle yields None, as does a vertex graze."""
    assert MISSING.chord_fractions(TRIANGLE) is None


def test_intersections_return_drawable_parts() -> None:
    """The drawing helper returns the crossing as LatLon vertices."""
    parts = CROSSING.intersections(TRIANGLE)
    assert len(parts) == 1
    assert all(p.lat == pytest.approx(0.25) for p in parts[0])
    assert MISSING.intersections(TRIANGLE) == []


def test_axis_conventions_stay_in_gis() -> None:
    """The seam holds: no spatial SQL of any kind outside gis.py."""
    st_call = re.compile(r"ST_[A-Za-z_]+\s*\(")
    package = Path(gis.__file__).parent
    for path in sorted(package.rglob("*.py")):
        if path.name == "gis.py":
            continue
        match = st_call.search(path.read_text(encoding="utf-8"))
        assert match is None, f"spatial SQL outside the seam: {path} ({match and match.group()})"


def test_lat_first_is_the_one_flip() -> None:
    """The measurement helpers route every flip through _lat_first."""
    assert gis._lat_first(Geom("g")) == "ST_FlipCoordinates(g)"
    assert gis.distance_m_sql(Geom("a"), Geom("b")) == (
        "ST_Distance_Sphere(ST_FlipCoordinates(a), ST_FlipCoordinates(b))"
    )
    assert gis.length_m_sql(Geom("g")) == "ST_Length_Spheroid(ST_FlipCoordinates(g))"


def test_great_circle_bearing_cardinals() -> None:
    """Due east and due north from the equator, and the geodesic case at 60 N."""
    origin = LatLon(lat=0.0, lon=0.0)
    assert origin.bearing_to(LatLon(lat=0.0, lon=1.0)) == pytest.approx(90.0)
    assert origin.bearing_to(LatLon(lat=1.0, lon=0.0)) == pytest.approx(0.0)
    assert origin.bearing_to(origin) == 0.0
    # The planar answer would be 90. ST_Azimuth gives that, which is why this
    # one operation delegates to pyproj's geodesic engine instead of DuckDB.
    bearing = LatLon(lat=60.0, lon=0.0).bearing_to(LatLon(lat=60.0, lon=10.0))
    assert bearing == pytest.approx(85.67, abs=0.01)


def test_spatial_unavailable_error_names_the_fixes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every load route failing raises with the offline guidance."""

    class FailingCon:
        def execute(self, sql: str) -> None:
            raise RuntimeError("offline")

    monkeypatch.setattr(gis, "_wheel_extension", lambda version: None)
    with pytest.raises(gis.SpatialUnavailableError, match="duckdb-extension-spatial"):
        gis._load_spatial(FailingCon(), "0.0.0")
