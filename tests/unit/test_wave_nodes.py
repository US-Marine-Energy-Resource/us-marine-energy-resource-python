"""Nearest-node lookup against a tiny fixture index."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from us_marine_energy_resource.wave_hindcast import errors, nodes
from us_marine_energy_resource.wave_hindcast.nodes import WaveNode

PACWAVE = (44.5670485, -124.22896475)


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute the reference great-circle distance on DuckDB's 6371 km sphere."""
    radius = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def test_within_finds_west_coast(wave_index_dir: Path) -> None:
    """The occupancy gate resolves a covered point to its domain."""
    assert nodes.within(*PACWAVE) == ["West_Coast"]


def test_within_empty_for_uncovered_point(wave_index_dir: Path) -> None:
    """A point far from every node matches no domain."""
    assert nodes.within(0.0, 0.0) == []


def test_within_rings_zero_is_strict(wave_index_dir: Path) -> None:
    """rings=0 requires the exact cell; rings=1 admits the neighbours."""
    # One cell (0.05 deg) away from every West_Coast fixture node.
    lat, lon = 44.51, -124.30
    assert nodes.within(lat, lon, rings=0) == []
    assert "West_Coast" in nodes.within(lat, lon, rings=2)


def test_nearest_picks_haversine_neighbour(wave_index_dir: Path) -> None:
    """Return the true haversine nearest node with its distance."""
    node = nodes.nearest(*PACWAVE)
    assert isinstance(node, WaveNode)
    assert node.domain == "West_Coast"
    assert node.endpoint == "us-west-coast-hindcast-download"
    assert node.location_id == 0  # (44.5682, -124.2280) is closest in the fixture
    expected = _haversine_m(*PACWAVE, 44.5682, -124.2280)
    assert node.distance_m == pytest.approx(expected, rel=1e-6)


def test_nearest_k_returns_ranked_frame(wave_index_dir: Path) -> None:
    """k>1 returns a DataFrame sorted by distance."""
    frame = nodes.nearest(*PACWAVE, k=3)
    assert list(frame.columns) == ["domain", "location_id", "lat", "lon", "distance_m", "endpoint"]
    assert len(frame) == 3
    distances = frame["distance_m"].tolist()
    assert distances == sorted(distances)
    assert frame.iloc[0].location_id == 0


def test_nearest_across_antimeridian(wave_index_dir: Path) -> None:
    """Alaska resolves on both sides of the dateline."""
    east = nodes.nearest(52.0, 179.985)
    assert isinstance(east, WaveNode)
    assert east.domain == "Alaska"
    assert east.location_id == 0  # (52.0, 179.98)

    west = nodes.nearest(52.0, -179.99)
    assert isinstance(west, WaveNode)
    assert west.domain == "Alaska"
    assert west.location_id == 1  # (52.0005, -179.97)


def test_nearest_crosses_antimeridian_for_true_nearest(wave_index_dir: Path) -> None:
    """The true nearest node wins even when it sits across the dateline."""
    # From (52.0, -179.999) the eastern node at (51.99, 179.99) is nearer
    # than the western node at -179.97, but only via the antimeridian.
    node = nodes.nearest(52.0, -179.999)
    assert isinstance(node, WaveNode)
    assert node.location_id == 2  # (51.99, 179.99)
    expected = _haversine_m(52.0, -179.999, 51.99, 179.99)
    assert node.distance_m == pytest.approx(expected, rel=1e-6)


def test_outside_every_domain_raises(wave_index_dir: Path) -> None:
    """A point outside all coverage raises with the checked domains attached."""
    with pytest.raises(errors.PointOutsideDomainError) as excinfo:
        nodes.nearest(0.0, 0.0)
    assert excinfo.value.lat == 0.0
    assert excinfo.value.domains


def test_forced_domain_skips_gate(wave_index_dir: Path) -> None:
    """domain= restricts the search without consulting the occupancy table."""
    node = nodes.nearest(21.4660, -157.7520, domain="Hawaii")
    assert isinstance(node, WaveNode)
    assert node.domain == "Hawaii"
    assert node.endpoint == "hawaii-hindcast-download"


def test_forced_domain_with_no_nearby_node_raises(wave_index_dir: Path) -> None:
    """A forced domain with nothing inside the prefilter window is a miss."""
    with pytest.raises(errors.PointOutsideDomainError, match="no grid node"):
        nodes.nearest(30.0, -140.0, domain="West_Coast")
