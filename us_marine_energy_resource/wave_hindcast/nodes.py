"""Find the WPTO wave hindcast grid node nearest a coordinate.

The hindcast download API identifies a site by geometry (``wkt``) or by grid
id (``location_ids``), and the grid id is the more dependable of the two, but
nothing in the API tells you what a site's grid id is. This module answers
that offline:

    >>> from us_marine_energy_resource.wave_hindcast import nodes
    >>> nodes.nearest(44.5670485, -124.22896475)
    WaveNode(location_id=479519, domain='West_Coast',
             endpoint='us-west-coast-hindcast-download',
             lat=44.5682, lon=-124.228, distance_m=149.1)

    >>> nodes.nearest(21.46488, -157.751524, k=5)   # DataFrame, ranked
    >>> nodes.nearest(lat, lon, domain='Atlantic')  # skip the domain gate

``location_id`` is the value to pass as the API's ``location_ids`` parameter.
Backed by a parquet index, one file per domain, resolved by :mod:`.index`, so
a query touches the network at most once per domain. Distances come from
DuckDB's spatial extension via :mod:`..gis`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .. import gis
from ..explore.lazy import lazy_import
from . import errors, index
from .domains import DOMAIN_ENDPOINTS

if TYPE_CHECKING:
    import pandas as pd

# How far a queried point may sit from the grid before it is an error rather
# than a snap. A product decision, not geometry: sites just off the last node
# (nearshore buoys) should resolve, points in another ocean should not.
_MAX_SNAP_M = 50_000


@dataclass(frozen=True)
class WaveNode:
    """One grid node, and how far it sits from the requested point."""

    location_id: int
    domain: str
    endpoint: str
    lat: float
    lon: float
    distance_m: float


def domains() -> list[str]:
    """List the domains present in the shipped index.

    Returns
    -------
    list of str
        Domain names in sorted order.
    """
    return sorted(index.load_index()["node_files"])


def within(lat: float, lon: float, rings: int = 1) -> list[str]:
    """List the domains whose coverage includes this point. Stage one of the lookup.

    Reads a small occupancy table rather than any node file, so it is cheap
    regardless of domain size. Bounding boxes are deliberately not used
    because Alaska spans the antimeridian, so its box would claim every point
    on earth.

    Parameters
    ----------
    lat, lon : float
        Point of interest, in degrees.
    rings : int, default 1
        How many cells of slack to allow around the point. The default admits
        the neighbouring cells so a site just outside the last node, such as
        a nearshore buoy, still resolves to its domain. Pass 0 to require the
        point to land in an occupied cell exactly.

    Returns
    -------
    list of str
        Matching domain names, possibly empty. More than one is normal: the
        published domains overlap at their edges.
    """
    idx = index.load_index()
    cell = idx["extent_cell_deg"]
    path = index.data_path(idx["extents_file"])
    lat_cell = math.floor(lat / cell)
    lon_cell = math.floor(lon / cell)

    return [
        row[0]
        for row in gis.connection()
        .execute(
            f"""
            SELECT DISTINCT domain FROM read_parquet('{path.as_posix()}')
            WHERE lat_cell BETWEEN ? AND ? AND lon_cell BETWEEN ? AND ?
            """,
            [lat_cell - rings, lat_cell + rings, lon_cell - rings, lon_cell + rings],
        )
        .fetchall()
    ]


def footprints(domain: str | None = None) -> dict[str, Any]:
    """Coverage outlines as GeoJSON, ready to drop on a map.

    Outlines follow each domain's real coastline rather than a bounding box,
    with Alaska split across the antimeridian. They are dissolved from the
    cells that contain nodes, at the resolution recorded in the file's
    ``cell_size_deg`` property.

    Parameters
    ----------
    domain : str, optional
        Return one domain's Feature instead of the whole FeatureCollection.

    Returns
    -------
    dict
        A GeoJSON FeatureCollection, or a single Feature when ``domain`` is
        set.
    """
    import json

    collection = json.loads(index.data_path(index.load_index()["footprints_file"]).read_text())
    if domain is None:
        return collection
    for feature in collection["features"]:
        if feature["properties"]["domain"] == domain:
            return feature
    raise KeyError(
        f"{domain} is not in the footprints. The available domains are {', '.join(domains())}."
    )


def _query_domain(domain: str, lat: float, lon: float, k: int) -> pd.DataFrame:
    """Find the nearest ``k`` nodes in one domain's parquet, ranked by distance.

    Parameters
    ----------
    domain : str
        Domain whose node file to search.
    lat, lon : float
        Point of interest, in degrees.
    k : int
        How many nodes to return.

    Returns
    -------
    pandas.DataFrame
        Up to ``k`` rows sorted by distance, with columns ``location_id``,
        ``lat``, ``lon``, and ``distance_m``.
    """
    idx = index.load_index()
    path = index.data_path(idx["node_files"][domain])
    scale_f = float(idx["coord_scale"])

    # A full scan handles the antimeridian natively: a bbox prefilter cannot,
    # without wrap logic this module no longer hand-rolls.
    node = gis.column_point(lon_sql=f"lon_fixed / {scale_f}", lat_sql=f"lat_fixed / {scale_f}")
    query = gis.point(gis.LatLon(lat=lat, lon=lon))
    return (
        gis.connection()
        .execute(
            f"""
        SELECT * FROM (
            SELECT location_id,
                   lat_fixed / {scale_f} AS lat,
                   lon_fixed / {scale_f} AS lon,
                   {gis.distance_m_sql(node, query)} AS distance_m
            FROM read_parquet('{path.as_posix()}')
        )
        WHERE distance_m <= {_MAX_SNAP_M}
        ORDER BY distance_m ASC
        LIMIT {int(k)}
        """
        )
        .df()
    )


def nearest(
    lat: float, lon: float, k: int = 1, domain: str | None = None
) -> WaveNode | pd.DataFrame:
    """Grid node(s) nearest a coordinate.

    Parameters
    ----------
    lat, lon : float
        Point of interest, in degrees. Note the argument order is lat then
        lon, unlike the API's WKT, which is longitude-first.
    k : int, default 1
        How many nodes to return. ``k=1`` returns a single :class:`WaveNode`;
        anything larger returns a DataFrame ranked by distance.
    domain : str, optional
        Restrict to one domain instead of letting the coverage gate decide.

    Returns
    -------
    WaveNode or pandas.DataFrame
        A single :class:`WaveNode` when ``k`` is 1, otherwise a DataFrame of
        the nearest nodes ranked by distance.

    Raises
    ------
    PointOutsideDomainError
        The point is outside every domain.
    """
    # Stage one: narrow to covering domains against the small occupancy table.
    # Stage two, below, only touches those domains' node files.
    candidates = [domain] if domain else within(lat, lon)
    if not candidates:
        raise errors.PointOutsideDomainError(
            f"({lat}, {lon}) is outside every hindcast domain",
            lat=lat,
            lon=lon,
            domains=domains(),
        )

    frames = []
    for name in candidates:
        frame = _query_domain(name, lat, lon, k)
        if len(frame):
            frame.insert(0, "domain", name)
            frame["endpoint"] = DOMAIN_ENDPOINTS.get(name)
            frames.append(frame)

    if not frames:
        # Inside a covering cell but no node within the snap cap: the point
        # is in a hole in the grid, e.g. inland or past the domain edge.
        raise errors.PointOutsideDomainError(
            f"({lat}, {lon}) has no grid node within {_MAX_SNAP_M // 1000} km",
            lat=lat,
            lon=lon,
            domains=candidates,
        )

    pd = lazy_import("pandas", "ranking wave grid nodes")

    ranked = (
        pd.concat(frames, ignore_index=True)
        .sort_values("distance_m")
        .head(k)
        .reset_index(drop=True)
    )
    if k == 1:
        row = ranked.iloc[0]
        return WaveNode(
            location_id=int(row.location_id),
            domain=str(row.domain),
            endpoint=str(row.endpoint),
            lat=float(row.lat),
            lon=float(row.lon),
            distance_m=float(row.distance_m),
        )
    return ranked
