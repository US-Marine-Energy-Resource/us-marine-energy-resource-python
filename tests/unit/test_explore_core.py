"""Parity, sniffing, selection, and stats across the explore backends."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from us_marine_energy_resource.explore import (
    ByteSize,
    FirstN,
    Index,
    NodePath,
    StatsSpec,
    TransferPolicy,
    open_file,
)
from us_marine_energy_resource.explore.errors import UnknownFormatError, UnsupportedFormatError
from us_marine_energy_resource.explore.model import ArrayInfo, StorageInfo
from us_marine_energy_resource.explore.selection import resolve
from us_marine_energy_resource.explore.sniff import sniff_format

# --- parity: one API, every backend --------------------------------------------------------------


@pytest.fixture(params=["h5_file", "nc4_file", "parquet_file"])
def any_file(request: pytest.FixtureRequest) -> Path:
    """Each supported file type in turn."""
    return request.getfixturevalue(request.param)


def test_summary_shape_is_uniform(any_file: Path) -> None:
    """Summary shape is uniform."""
    with open_file(str(any_file)) as f:
        summary = f.summary()
    assert summary.nodes
    assert all(n.path.value.startswith("/") for n in summary.nodes)
    assert summary.nodes[0].path.value == "/"
    assert summary.n_arrays >= 1


def test_summary_json_roundtrips(any_file: Path) -> None:
    """Summary json roundtrips."""
    from us_marine_energy_resource.explore.cli.render import to_json

    with open_file(str(any_file)) as f:
        text = to_json(f.summary())
    parsed = json.loads(text)
    assert parsed["format"] in ("hdf5", "parquet")
    assert isinstance(parsed["nodes"], list)


def test_asdict_is_serializable(any_file: Path) -> None:
    """Asdict is serializable."""
    with open_file(str(any_file)) as f:
        summary = f.summary()
    # Every field survives dataclasses.asdict + json with a str fallback.
    json.dumps(dataclasses.asdict(summary), default=str)


# --- sniff: extension never decides --------------------------------------------------------------


def test_nc4_sniffs_as_hdf5(nc4_file: Path) -> None:
    """Nc4 sniffs as hdf5."""
    with open_file(str(nc4_file)) as f:
        assert f.summary().format == "hdf5"


def test_h5_content_wins_over_extension(h5_file: Path, tmp_path: Path) -> None:
    """H5 content wins over extension."""
    renamed = tmp_path / "mystery.dat"
    renamed.write_bytes(h5_file.read_bytes())
    with open_file(str(renamed)) as f:
        assert f.summary().format == "hdf5"


def test_netcdf3_is_rejected_by_name(nc3_file: Path) -> None:
    """Netcdf3 is rejected by name."""
    with pytest.raises(UnsupportedFormatError, match="netCDF-3"), open_file(str(nc3_file)):
        pass


def test_garbage_is_unknown() -> None:
    """Garbage is unknown."""
    with pytest.raises(UnknownFormatError):
        sniff_format(b"not a file")


# --- selection: wrong states fail early ----------------------------------------------------------


def _array(shape: tuple[int, ...]) -> ArrayInfo:
    return ArrayInfo(
        shape=shape,
        dtype="int16",
        dim_names=tuple(None for _ in shape),
        fill_value=None,
        storage=StorageInfo(None, None, (), None, None),
    )


def test_index_wrong_rank_fails() -> None:
    """Index wrong rank fails."""
    with pytest.raises(ValueError, match="axes"):
        resolve(Index("0:5,0:5,0:5"), _array((10, 10)))


def test_index_out_of_bounds_fails() -> None:
    """Index out of bounds fails."""
    with pytest.raises(ValueError, match="out of range"):
        resolve(Index("50"), _array((10,)))


def test_firstn_must_be_positive() -> None:
    """Firstn must be positive."""
    with pytest.raises(ValueError):
        FirstN(0)
    with pytest.raises(ValueError):
        FirstN(-1)


def test_firstn_clamps_to_length() -> None:
    """Firstn clamps to length."""
    r = resolve(FirstN(999), _array((10, 4)))
    assert r.slices[0].stop == 10


# --- stats: honest sampling ----------------------------------------------------------------------


def test_stats_reports_sampling(h5_file: Path) -> None:
    """Stats reports sampling."""
    with open_file(str(h5_file)) as f:
        path = NodePath("/significant_wave_height")
        plan = f.plan_stats(path, StatsSpec(max_elements=500))
        approved = TransferPolicy().approve(plan, remote=False)
        from us_marine_energy_resource.explore.budget import ApprovedRead

        assert isinstance(approved, ApprovedRead)
        result = f.stats(approved, StatsSpec(max_elements=500))
    assert result.sampled is True
    assert 0 < result.sample_fraction < 1
    assert result.sample_method == "chunk-strided"
    assert result.mean is not None and result.min is not None


def test_stats_full_when_small(parquet_file: Path) -> None:
    """Stats full when small."""
    with open_file(str(parquet_file)) as f:
        path = NodePath("/speed")
        plan = f.plan_stats(path, StatsSpec(max_elements=10_000))
        approved = TransferPolicy().approve(plan, remote=False)
        from us_marine_energy_resource.explore.budget import ApprovedRead

        assert isinstance(approved, ApprovedRead)
        result = f.stats(approved, StatsSpec(max_elements=10_000))
    assert result.sampled is False
    assert result.sample_fraction == pytest.approx(1.0)


def test_decode_cf_scales_values(h5_file: Path) -> None:
    """Decode cf scales values."""
    with open_file(str(h5_file)) as f:
        path = NodePath("/significant_wave_height")
        plan = f.plan_read(path, Index("0:1,0:1"))
        approved = TransferPolicy().approve(plan, remote=False)
        from us_marine_energy_resource.explore.budget import ApprovedRead

        assert isinstance(approved, ApprovedRead)
        raw = f.head(approved, "none")
        scaled = f.head(approved, "cf")
    raw_val = raw.values[0][0]
    scaled_val = scaled.values[0][0]
    assert scaled_val == pytest.approx(raw_val * 100.0)
    assert raw.notes  # names the unapplied scale_factor


def test_bytesize_str() -> None:
    """Bytesize str."""
    assert str(ByteSize(4_500_000_000)) == "4.2 GB"
