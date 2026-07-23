"""The direct-S3 backend, against local .h5 files standing in for the bucket."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from us_marine_energy_resource.wave_hindcast import errors, hindcast
from us_marine_energy_resource.wave_hindcast.nodes import WaveNode
from us_marine_energy_resource.wave_hindcast.s3_direct import backend as s3_backend

NODE = WaveNode(
    location_id=4,
    domain="West_Coast",
    endpoint="us-west-coast-hindcast-download",
    lat=44.5682,
    lon=-124.228,
    distance_m=142.3,
)

_TIMES = 8
_NODES = 6


def _write_year_file(path: Path, year: int) -> None:
    """Build a tiny rex-style year file: chunked 2-D variables plus time_index."""
    import h5py

    stamps = [f"{year}-01-0{d + 1} 00:00:00".encode() for d in range(_TIMES)]
    with h5py.File(path, "w") as f:
        f.create_dataset("time_index", data=np.array(stamps, dtype="S25"))
        f.create_dataset("coordinates", data=np.zeros((_NODES, 2), dtype="f4"))
        f.create_dataset("meta", data=np.zeros(_NODES, dtype="f4"))
        swh = f.create_dataset(
            "significant_wave_height",
            data=np.arange(_TIMES * _NODES, dtype="f4").reshape(_TIMES, _NODES),
            chunks=(4, 3),
        )
        swh.attrs["units"] = "m"
        period = f.create_dataset(
            "energy_period",
            # Stored as tens: the rex scale_factor divides on read.
            data=np.full((_TIMES, _NODES), 100, dtype="i2"),
            chunks=(4, 3),
        )
        period.attrs["scale_factor"] = 10.0
        period.attrs["units"] = "s"


@pytest.fixture
def year_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write 2019 and 2020 files and route _year_uri at them."""
    for year in (2019, 2020):
        _write_year_file(tmp_path / f"wave_{year}.h5", year)
    monkeypatch.setattr(
        s3_backend, "_year_uri", lambda domain, year: str(tmp_path / f"wave_{year}.h5")
    )
    return tmp_path


def _fetch(cache: Path, **kwargs: object) -> list[str]:
    events: list[str] = []
    s3_backend.S3Backend().fetch(
        NODE,
        "mysite",
        requested_lat=44.567,
        requested_lon=-124.229,
        force=False,
        timeout_s=60,
        cache_dir=cache,
        on_event=events.append,
        **kwargs,  # type: ignore[arg-type]
    )
    return events


def test_fetch_writes_the_contract_layout(year_files: Path, tmp_path: Path) -> None:
    """Fetch produces the combined CSV and metadata the facade reads."""
    cache = tmp_path / "cache"
    _fetch(cache, years=[2019, 2020])

    frame, metadata = hindcast.load_site("mysite", cache_dir=cache)
    assert len(frame) == 2 * _TIMES
    assert list(frame.columns) == ["Energy Period", "Significant Wave Height"]
    assert metadata["gid"] == 4
    assert metadata["years"] == ["2019", "2020"]
    assert metadata["units"]["Significant Wave Height"] == "m"
    assert metadata["direction_transform"] is None
    assert metadata["source"] == "s3 direct"

    # Node 4's first value is column 4 of row 0, and the scale divides.
    assert frame["Significant Wave Height"].iloc[0] == pytest.approx(4.0)
    assert frame["Energy Period"].iloc[0] == pytest.approx(10.0)


def test_variables_narrow_and_validate(year_files: Path, tmp_path: Path) -> None:
    """A variables subset limits columns, and unknown names list the real ones."""
    cache = tmp_path / "cache"
    _fetch(cache, years=[2019], variables=["energy_period"])
    frame, _ = hindcast.load_site("mysite", cache_dir=cache)
    assert list(frame.columns) == ["Energy Period"]

    with pytest.raises(errors.InvalidAttributeError) as excinfo:
        _fetch(tmp_path / "cache2", years=[2019], variables=["swh_typo"])
    assert "significant_wave_height" in excinfo.value.valid


def test_years_validate(year_files: Path, tmp_path: Path) -> None:
    """A year outside the served range is refused up front."""
    with pytest.raises(errors.InvalidYearError, match="1900"):
        _fetch(tmp_path / "cache", years=[1900])


def test_chunk_blocks_cached_for_neighbors(year_files: Path, tmp_path: Path) -> None:
    """The whole chunk block is kept, and a second run reads from it."""
    cache = tmp_path / "cache"
    events = _fetch(cache, years=[2019])
    assert any(e.startswith("note:") and "chunk blocks" in e for e in events)

    # Node 4 rides in the chunk covering nodes 3-5.
    blocks = sorted(p.name for p in (cache / "s3_chunks" / "West_Coast" / "2019").iterdir())
    assert blocks == ["energy_period_3.npy", "significant_wave_height_3.npy"]
    block = np.load(cache / "s3_chunks" / "West_Coast" / "2019" / "significant_wave_height_3.npy")
    assert block.shape == (_TIMES, 3)

    # A neighboring node in the same block downloads nothing new: no note.
    neighbor = WaveNode(5, "West_Coast", NODE.endpoint, 44.57, -124.22, 99.0)
    events2: list[str] = []
    s3_backend.S3Backend().fetch(
        neighbor,
        "nextdoor",
        requested_lat=44.57,
        requested_lon=-124.22,
        force=False,
        timeout_s=60,
        cache_dir=cache,
        on_event=events2.append,
        years=[2019],
    )
    assert not any(e.startswith("note:") for e in events2)
    frame, _ = hindcast.load_site("nextdoor", cache_dir=cache)
    assert frame["Significant Wave Height"].iloc[0] == pytest.approx(5.0)


def test_describe_reports_no_direction_transform(year_files: Path) -> None:
    """The S3 files store meteorological directions, even for Hawaii."""
    hawaii = WaveNode(1, "Hawaii", "hawaii-hindcast-download", 21.46, -157.75, 10.0)
    info = s3_backend.S3Backend().describe(hawaii)
    assert info.direction_transform is None
    assert (info.first_year, info.last_year) == (1979, 2020)


def test_metadata_json_is_valid(year_files: Path, tmp_path: Path) -> None:
    """The written metadata.json parses and records the request context."""
    cache = tmp_path / "cache"
    _fetch(cache, years=[2020])
    site_dir = next(cache.glob("mysite_*"))
    metadata = json.loads((site_dir / "metadata.json").read_text())
    assert metadata["requested_lat"] == 44.567
    assert metadata["rows"] == _TIMES
