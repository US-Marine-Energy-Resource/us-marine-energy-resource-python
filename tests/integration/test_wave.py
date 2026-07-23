"""Integration tests for the wave node index and point description.

These touch the real index files and, in one case, the published download URL.
They never call the NLR download API, whose rate limits (2000 requests a day,
one every two seconds) make automated calls a bad idea. The fetch path is
covered by unit tests against a fake API, and manually by:

    mer wave 44.567,-124.229 --yes
"""

from __future__ import annotations

import os

import pytest

from us_marine_energy_resource.wave_hindcast import hindcast, index, nodes

PACWAVE = (44.5670485, -124.22896475)


@pytest.mark.integration
def test_describe_point_pacwave() -> None:
    """PacWave South resolves to its known West Coast grid node."""
    info = hindcast.describe_point(*PACWAVE)
    assert info["domain"] == "West_Coast"
    assert info["location_id"] == 479519
    assert info["endpoint"] == "us-west-coast-hindcast-download"
    assert info["years"] == [1979, 2020]
    assert info["distance_m"] < 300


# One verified in-domain point per domain the index covers well.
_KNOWN_POINTS = (
    (44.5670485, -124.22896475, "West_Coast"),
    (21.46488, -157.751524, "Hawaii"),
    (35.91036, -75.59239, "Atlantic"),
    (57.0, -152.5, "Alaska"),
    (18.6, -66.1, "Gulf_of_Mexico_and_Puerto_Rico"),
)


@pytest.mark.integration
def test_nearest_known_points() -> None:
    """Each known point resolves to a nearby node in its own domain."""
    for lat, lon, domain in _KNOWN_POINTS:
        node = nodes.nearest(lat, lon, domain=domain)
        assert isinstance(node, nodes.WaveNode)
        assert node.domain == domain
        assert node.distance_m < 1_000


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("MER_WAVE_INDEX_PUBLISHED"),
    reason="the LFS files reach the main branch only after this change merges. "
    "Set MER_WAVE_INDEX_PUBLISHED=1 to run.",
)
def test_index_downloads_from_main(tmp_path, monkeypatch) -> None:
    """One real node file downloads from the published URL and verifies."""
    monkeypatch.delenv("MER_WAVE_INDEX_DIR", raising=False)
    monkeypatch.setattr(index, "_checkout_path", lambda filename: None)

    import pooch

    monkeypatch.setattr(pooch, "os_cache", lambda name: tmp_path)
    path = index.data_path("nodes_West_Coast_v1.parquet")
    assert path.stat().st_size > 1_000_000
