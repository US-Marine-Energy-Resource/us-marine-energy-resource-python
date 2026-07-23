"""The NLR archive handling: waiting for builds and organizing downloads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.unit.conftest import FakeTime, make_wave_archive
from us_marine_energy_resource.wave_hindcast import errors
from us_marine_energy_resource.wave_hindcast.nlr_api import archive

# --------------------------------------------------------------------------- #
# waiting
# --------------------------------------------------------------------------- #


def test_wait_for_archive_times_out(monkeypatch: pytest.MonkeyPatch, fake_time: FakeTime) -> None:
    """A never-ready archive returns False once the deadline passes."""
    monkeypatch.setattr(archive, "archive_ready", lambda url: False)
    events: list[str] = []
    ready = archive.wait_for_archive("s", "https://dl/x.zip", timeout_s=60, on_event=events.append)
    assert ready is False
    assert any("waiting" in e for e in events)


def test_wait_escalates_when_slow(monkeypatch: pytest.MonkeyPatch, fake_time: FakeTime) -> None:
    """A wait past the typical build time tells the user to check email once."""
    monkeypatch.setattr(archive, "archive_ready", lambda url: False)
    events: list[str] = []
    archive.wait_for_archive("s", "https://dl/x.zip", timeout_s=1500, on_event=events.append)
    notes = [e for e in events if e.startswith("note:")]
    assert len(notes) == 1
    assert "--force" in notes[0] and "email" in notes[0]


def test_wait_for_archive_ready_immediately(
    monkeypatch: pytest.MonkeyPatch, fake_time: FakeTime
) -> None:
    """An already-built archive returns at once."""
    monkeypatch.setattr(archive, "archive_ready", lambda url: True)
    assert archive.wait_for_archive("s", "https://dl/x.zip", timeout_s=60) is True


# --------------------------------------------------------------------------- #
# organizing
# --------------------------------------------------------------------------- #


def _manifest(domain: str = "West_Coast") -> dict[str, Any]:
    return {
        "mysite": {
            "domain": domain,
            "requested_lat": 44.567,
            "requested_lon": -124.229,
            "download_url": "https://dl/86feb440a614c730332d16ffbb2e2413.zip",
            "requested_at": "2026-07-22T00:00:00+00:00",
        }
    }


def test_organize_archive(tmp_path: Path, wave_archive_zip: Path) -> None:
    """An archive unpacks into the combined CSV, layouts, and metadata."""
    out_dir = archive.organize_archive(wave_archive_zip, _manifest(), tmp_path)

    assert out_dir == tmp_path / "mysite_44.57_-124.23"
    combined = out_dir / "mysite_44.57_-124.23_1979-1980.csv"
    assert combined.exists()
    assert (out_dir / "by_year" / "mysite_44.57_-124.23_1979.csv").exists()
    assert (out_dir / "by_variable" / "mysite_44.57_-124.23_significant_wave_height.csv").exists()

    metadata = json.loads((out_dir / "metadata.json").read_text())
    assert metadata["site"] == "mysite"
    assert metadata["gid"] == 479519
    assert metadata["years"] == ["1979", "1980"]
    assert metadata["rows"] == 4
    assert metadata["units"]["Significant Wave Height"] == "m"
    assert metadata["water_depth"] == 50  # b'50' byte-repr stripped and typed
    assert metadata["direction_transform"] is None
    assert metadata["direction_columns_corrected"] == []

    import pandas as pd

    frame = pd.read_csv(combined)
    assert list(frame.columns[:2]) == ["timestamp", "Year"]
    assert frame["timestamp"].iloc[0] == "1979-01-01T00:00:00Z"


def test_organize_applies_hawaii_direction_transform(tmp_path: Path) -> None:
    """A Hawaii archive gets its direction columns rotated 270-x."""
    zip_path = make_wave_archive(
        tmp_path / "mysite.zip", lat=21.46, lon=-157.75, years=(1979,), direction=100.0
    )
    out_dir = archive.organize_archive(zip_path, _manifest("Hawaii"), tmp_path)

    metadata = json.loads((out_dir / "metadata.json").read_text())
    assert metadata["direction_transform"] == "270-x"
    assert metadata["direction_columns_corrected"] == ["Mean Wave Direction"]

    import pandas as pd

    frame = pd.read_csv(out_dir / "mysite_21.46_-157.75_1979-1979.csv")
    assert frame["Mean Wave Direction"].tolist() == [170.0, 160.0]  # (270 - x) % 360
    # The non-direction column is untouched.
    assert frame["Significant Wave Height"].tolist() == [2.5, 2.75]


def test_organize_matches_by_request_hash(tmp_path: Path) -> None:
    """A hand-downloaded archive named by hash matches through the manifest URL."""
    zip_path = make_wave_archive(tmp_path / "86feb440a614c730332d16ffbb2e2413.zip", years=(1979,))
    out_dir = archive.organize_archive(zip_path, _manifest(), tmp_path)
    assert json.loads((out_dir / "metadata.json").read_text())["site"] == "mysite"


def test_organize_unmatched_raises(tmp_path: Path) -> None:
    """An archive matching nothing in the manifest is an explicit error."""
    zip_path = make_wave_archive(tmp_path / "stranger.zip", years=(1979,))
    with pytest.raises(errors.ArchiveUnmatchedError):
        archive.organize_archive(zip_path, {}, tmp_path)


def test_organize_empty_zip_raises(tmp_path: Path) -> None:
    """A zip with no year CSVs is corrupt."""
    import zipfile

    path = tmp_path / "empty.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("readme.txt", "nothing here")
    with pytest.raises(errors.ArchiveCorruptError):
        archive.organize_archive(path, {}, tmp_path)
