"""Parse geometry inputs from CLI strings into (lat, lon) coordinate lists."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def parse_point(s: str) -> tuple[float, float]:
    """Parse a ``'lat,lon'`` string into a (lat, lon) float tuple."""
    parts = s.strip().split(",")
    if len(parts) != 2:
        raise ValueError(f"Expected 'lat,lon', got {s!r}")
    return float(parts[0].strip()), float(parts[1].strip())


def parse_bbox(s: str) -> list[tuple[float, float]]:
    """Parse ``'lat_min,lon_min,lat_max,lon_max'`` into a 4-corner polygon ring.

    Returns the ring as (lat, lon) tuples suitable for
    ``TidalManifestQuery.query_all_within_polygon``.
    """
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError(f"--bbox expects 'lat_min,lon_min,lat_max,lon_max', got {s!r}")
    lat_min, lon_min, lat_max, lon_max = (float(p) for p in parts)
    if lat_min >= lat_max:
        raise ValueError(f"lat_min ({lat_min}) must be less than lat_max ({lat_max})")
    if lon_min >= lon_max:
        raise ValueError(f"lon_min ({lon_min}) must be less than lon_max ({lon_max})")
    return [
        (lat_min, lon_min),
        (lat_min, lon_max),
        (lat_max, lon_max),
        (lat_max, lon_min),
    ]


def parse_wkt(value: str) -> list[tuple[float, float]]:
    """Parse a WKT POLYGON string or path to a ``.wkt`` file.

    WKT coordinates are (lon lat) order; returns (lat, lon) tuples.
    """
    path = Path(value)
    if path.exists():
        value = path.read_text().strip()

    m = re.search(r"POLYGON\s*\(\s*\(([^)]+)\)", value, re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot parse as POLYGON WKT: {value[:80]!r}")

    coords: list[tuple[float, float]] = []
    for pair in m.group(1).split(","):
        parts = pair.strip().split()
        if len(parts) < 2:
            raise ValueError(f"Invalid WKT coordinate pair: {pair!r}")
        lon, lat = float(parts[0]), float(parts[1])
        coords.append((lat, lon))
    return coords


def parse_geojson_file(path: Path) -> list[tuple[float, float]]:
    """Load (lat, lon) ring coordinates from a GeoJSON Polygon file.

    Handles ``Feature``, ``FeatureCollection``, and bare ``Polygon`` geometry.
    GeoJSON coordinates are [lon, lat] order; returns (lat, lon) tuples.
    """
    data: dict[str, Any] = json.loads(path.read_text())

    if data.get("type") == "FeatureCollection":
        features = data.get("features", [])
        if not features:
            raise ValueError("GeoJSON FeatureCollection contains no features")
        data = features[0]

    if data.get("type") == "Feature":
        data = data.get("geometry") or {}

    if data.get("type") != "Polygon":
        raise ValueError(f"GeoJSON geometry must be Polygon, got {data.get('type')!r}")

    ring: list[list[float]] = data["coordinates"][0]
    return [(float(lat), float(lon)) for lon, lat in ring]
