"""The shared node-index build core."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from us_marine_energy_resource.wave_hindcast import index_build


@pytest.fixture
def coords() -> np.ndarray:
    """Build a small (n, 2) lat/lon array in source order."""
    return np.array(
        [
            [44.5682, -124.2280],
            [44.5600, -124.2200],
            [44.5750, -124.2350],
        ]
    )


def test_read_coordinates_from_h5(tmp_path: Path, coords: np.ndarray) -> None:
    """The coordinates dataset comes back verbatim from an open handle."""
    import h5py

    path = tmp_path / "grid.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("coordinates", data=coords)
        f.create_dataset("meta/ignored", data=np.arange(4))
    with open(path, "rb") as handle:
        out = index_build.read_coordinates_from(handle)
    np.testing.assert_array_equal(out, coords)


def test_write_domain_nodes_schema(tmp_path: Path, coords: np.ndarray) -> None:
    """Fixed-point columns and a 0..n-1 location_id, matching the shipped index."""
    import pyarrow.parquet as pq

    dest = tmp_path / "nodes.parquet"
    index_build.write_domain_nodes(coords, dest)
    table = pq.read_table(dest)
    assert table.column_names == ["location_id", "lat_fixed", "lon_fixed"]
    assert table["location_id"].to_pylist() == [0, 1, 2]
    assert table["lat_fixed"].to_pylist() == [round(lat * 10**6) for lat, _ in coords.tolist()]
    assert table["lon_fixed"].to_pylist() == [round(lon * 10**6) for _, lon in coords.tolist()]


def test_domain_bounds_antimeridian() -> None:
    """A domain spanning the dateline is flagged; a compact one is not."""
    alaska = np.array([[52.0, 179.99], [52.0, -179.98]])
    bounds = index_build.domain_bounds("Alaska", alaska)
    assert bounds["crosses_antimeridian"] is True
    assert bounds["node_count"] == 2

    west = np.array([[44.5, -124.2], [44.6, -124.3]])
    assert index_build.domain_bounds("West_Coast", west)["crosses_antimeridian"] is False


def test_build_domain_nodes(
    tmp_path: Path, coords: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one-call build reads coordinates and writes the parquet."""
    import pyarrow.parquet as pq

    monkeypatch.setattr(index_build, "read_coordinates", lambda domain: coords)
    dest = tmp_path / "sub" / "nodes_West_Coast_v1.parquet"
    out = index_build.build_domain_nodes("West_Coast", dest)
    assert out == dest
    assert pq.read_table(dest).num_rows == len(coords)
