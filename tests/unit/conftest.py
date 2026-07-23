"""Fixtures for explore unit tests: h5, nc4, and parquet files built on the fly."""

from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest


@pytest.fixture
def h5_file(tmp_path: Path) -> Path:
    """Build a chunked, gzipped HDF5 file with a group, attrs, and a dimension scale."""
    import h5py

    path = tmp_path / "sample.h5"
    with h5py.File(path, "w") as f:
        f.attrs["title"] = "sample"
        f.attrs["_NCProperties"] = "version=2"
        f.create_dataset("meta/latitude", data=np.linspace(40, 41, 20).astype("f4"))
        ti = f.create_dataset("time_index", data=np.arange(200))
        ti.make_scale("time")
        swh = f.create_dataset(
            "significant_wave_height",
            data=(np.arange(200 * 20).reshape(200, 20) % 500).astype("i2"),
            chunks=(50, 10),
            compression="gzip",
        )
        swh.attrs["scale_factor"] = 100.0
        swh.dims[0].attach_scale(ti)
        swh.dims[1].label = "gid"
    return path


@pytest.fixture
def nc4_file(tmp_path: Path) -> Path:
    """Build an HDF5 file with a ``.nc4`` name, to prove nc4 sniffs as HDF5."""
    import h5py

    path = tmp_path / "sample.nc4"
    with h5py.File(path, "w") as f:
        f.create_dataset("temp", data=np.arange(30).astype("f4"))
    return path


@pytest.fixture
def parquet_file(tmp_path: Path) -> Path:
    """Build a parquet file with two columns across several row groups."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "sample.parquet"
    table = pa.table(
        {
            "speed": np.linspace(0, 5, 500).astype("f4"),
            "direction": np.arange(500).astype("i4"),
        }
    )
    pq.write_table(table, path, row_group_size=100)
    return path


@pytest.fixture
def nc3_file(tmp_path: Path) -> Path:
    """Write a file starting with the netCDF-3 classic magic bytes."""
    path = tmp_path / "old.nc"
    path.write_bytes(b"CDF\x01" + b"\x00" * 60)
    return path


# --------------------------------------------------------------------------- #
# wave fixtures
# --------------------------------------------------------------------------- #

_WAVE_COORD_SCALE = 10**6
_WAVE_CELL_DEG = 0.05

# Tiny per-domain node sets. Alaska deliberately straddles the antimeridian.
_WAVE_FIXTURE_NODES: dict[str, list[tuple[float, float]]] = {
    "West_Coast": [
        (44.5682, -124.2280),  # nearest to PacWave South (44.5670, -124.2290)
        (44.5600, -124.2200),
        (44.5750, -124.2350),
        (44.5500, -124.2400),
        (44.5900, -124.2100),
    ],
    "Hawaii": [
        (21.4650, -157.7510),
        (21.4700, -157.7600),
    ],
    "Alaska": [
        (52.0000, 179.9800),
        (52.0005, -179.9700),
        (51.9900, 179.9900),
    ],
}


def _write_wave_nodes(path: Path, coords: list[tuple[float, float]]) -> None:
    """Write a fixed-point node parquet shaped like the real index files."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table(
        {
            "location_id": np.arange(len(coords), dtype=np.int32),
            "lat_fixed": np.array(
                [round(lat * _WAVE_COORD_SCALE) for lat, _ in coords], dtype=np.int32
            ),
            "lon_fixed": np.array(
                [round(lon * _WAVE_COORD_SCALE) for _, lon in coords], dtype=np.int32
            ),
        }
    )
    pq.write_table(table, path)


@pytest.fixture
def wave_index_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build a tiny node index and point ``MER_WAVE_INDEX_DIR`` at it."""
    import math

    import pyarrow as pa
    import pyarrow.parquet as pq

    index_dir = tmp_path / "wave_index"
    index_dir.mkdir()

    cells: dict[str, set[tuple[int, int]]] = {}
    for domain, coords in _WAVE_FIXTURE_NODES.items():
        _write_wave_nodes(index_dir / f"nodes_{domain}_v1.parquet", coords)
        cells[domain] = {
            (math.floor(lat / _WAVE_CELL_DEG), math.floor(lon / _WAVE_CELL_DEG))
            for lat, lon in coords
        }

    rows = [
        (domain, lat_cell, lon_cell)
        for domain, cellset in cells.items()
        for lat_cell, lon_cell in sorted(cellset)
    ]
    extents = pa.table(
        {
            "domain": [r[0] for r in rows],
            "lat_cell": np.array([r[1] for r in rows], dtype=np.int32),
            "lon_cell": np.array([r[2] for r in rows], dtype=np.int32),
        }
    )
    pq.write_table(extents, index_dir / "domain_extents_v1.parquet")

    monkeypatch.setenv("MER_WAVE_INDEX_DIR", str(index_dir))
    return index_dir


def make_wave_archive(
    dest: Path,
    *,
    gid: int = 479519,
    lat: float = 44.57,
    lon: float = -124.23,
    years: tuple[int, ...] = (1979, 1980),
    direction: float = 100.0,
) -> Path:
    """Write a zip shaped like an NLR wave download archive."""
    import zipfile

    preamble_keys = (
        "Source,Location ID,Jurisdiction,Latitude,Longitude,Time Zone,"
        "Local Time Zone,Distance to Shore,Water Depth,Version,"
        "Significant Wave Height,Mean Wave Direction"
    )
    preamble_values = f"WPTO,{gid},Federal,{lat},{lon},0,-8,10000,b'50',v1.0.1,m,deg"
    header = "Year,Month,Day,Hour,Minute,Significant Wave Height,Mean Wave Direction"

    with zipfile.ZipFile(dest, "w") as archive:
        for year in years:
            rows = [
                f"{year},1,1,0,0,2.5,{direction}",
                f"{year},1,1,3,0,2.75,{direction + 10}",
            ]
            body = "\n".join([preamble_keys, preamble_values, header, *rows]) + "\n"
            archive.writestr(f"{gid}_{lat}_{lon}_{year}.csv", body)
    return dest


@pytest.fixture
def wave_archive_zip(tmp_path: Path) -> Path:
    """Write a two-year wave archive named for the site ``mysite``."""
    return make_wave_archive(tmp_path / "mysite.zip")


# --------------------------------------------------------------------------- #
# wave API fakes
# --------------------------------------------------------------------------- #


class FakeResponse:
    """Duck-typed requests.Response."""

    def __init__(
        self,
        status_code: int,
        body: dict[str, Any] | None = None,
        text: str = "",
        headers: dict[str, str] | None = None,
    ):
        self.status_code = status_code
        self._body = body
        self.text = text or (json.dumps(body) if body is not None else "")
        self.reason = "reason"
        self.headers = headers or {}

    def json(self) -> dict[str, Any]:
        """Return the body, or raise ValueError like requests does."""
        if self._body is None:
            raise ValueError("not json")
        return self._body


class FakeTime:
    """A controllable clock so pacing and backoff cost nothing."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        """Return the fake clock."""
        return self.now

    def sleep(self, seconds: float) -> None:
        """Advance the fake clock instead of sleeping."""
        self.slept.append(seconds)
        self.now += seconds


@pytest.fixture
def fake_time(monkeypatch: pytest.MonkeyPatch) -> FakeTime:
    """Replace the clock in both nlr_api modules."""
    from us_marine_energy_resource.wave_hindcast.nlr_api import archive, client

    clock = FakeTime()
    monkeypatch.setattr(client, "time", clock)
    monkeypatch.setattr(archive, "time", clock)
    return clock


def _patch_requests(monkeypatch: pytest.MonkeyPatch, mod: types.SimpleNamespace) -> None:
    """Serve a fake ``requests`` module through lazy_import."""
    import importlib

    # Capture the real function first: the setattr below patches the shared
    # importlib module, so calling importlib.import_module inside the lambda
    # would recurse into itself.
    real_import = importlib.import_module
    monkeypatch.setattr(
        "us_marine_energy_resource.explore.lazy.importlib.import_module",
        lambda name: mod if name == "requests" else real_import(name),
    )
