"""Listing the published wave bucket, against an in-memory S3."""

from __future__ import annotations

from typing import Any

import pytest

from us_marine_energy_resource.wave_hindcast.s3_direct import bucket

_OBJECTS = [
    ("v1.0.1/West_Coast/West_Coast_wave_2010.h5", 80_000_000_000),
    ("v1.0.1/West_Coast/West_Coast_wave_2011.h5", 81_000_000_000),
    ("v1.0.0/West_Coast/West_Coast_wave_2010.h5", 79_000_000_000),
    ("v1.0.0/Hawaii/Hawaii_wave_2010.h5", 30_000_000_000),
    ("v1.0.1/Gulf_of_Mexico_and_Puerto_Rico/GOM_PR_2010.h5", 600_000_000_000),
    ("v1.0.1/West_Coast_virtual_buoy/buoy_2010.h5", 1_000),  # skipped
    ("v1.0.1/deprecated_old/old_2009.h5", 1_000),  # skipped
    ("v1.0.1/West_Coast/readme.txt", 100),  # not .h5
    ("v1.0.1/West_Coast/oddname.h5", 100),  # no year
]


class FakeS3:
    """Just enough of an S3 client: paginated list_objects_v2."""

    def __init__(self, objects: list[tuple[str, int]], page_size: int = 3) -> None:
        self._objects = objects
        self._page_size = page_size
        self.pages = 0

    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]:
        """Serve one page of the fixture objects."""
        self.pages += 1
        start = int(kwargs.get("ContinuationToken") or 0)
        chunk = self._objects[start : start + self._page_size]
        page: dict[str, Any] = {
            "Contents": [{"Key": key, "Size": size} for key, size in chunk],
        }
        if start + self._page_size < len(self._objects):
            page["IsTruncated"] = True
            page["NextContinuationToken"] = str(start + self._page_size)
        return page


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> FakeS3:
    """Serve the fixture objects through wave.bucket's client factory."""
    client = FakeS3(_OBJECTS)
    monkeypatch.setattr(bucket, "_client", lambda: client)
    return client


def test_list_files_filters_and_sorts(fake_client: FakeS3) -> None:
    """Only real year .h5 files survive; virtual-buoy and deprecated are skipped."""
    files = bucket.list_files()
    assert [f.key for f in files] == [
        "v1.0.0/Hawaii/Hawaii_wave_2010.h5",
        "v1.0.0/West_Coast/West_Coast_wave_2010.h5",
        "v1.0.1/Gulf_of_Mexico_and_Puerto_Rico/GOM_PR_2010.h5",
        "v1.0.1/West_Coast/West_Coast_wave_2010.h5",
        "v1.0.1/West_Coast/West_Coast_wave_2011.h5",
    ]
    assert fake_client.pages > 1  # pagination was followed

    first = files[0]
    assert (first.version, first.domain, first.year) == ("v1.0.0", "Hawaii", 2010)
    assert first.uri == "s3://wpto-pds-us-wave/v1.0.0/Hawaii/Hawaii_wave_2010.h5"


def test_list_files_domain_and_version_filters(fake_client: FakeS3) -> None:
    """The domain and version filters compose."""
    assert len(bucket.list_files(domain="West_Coast")) == 3
    assert len(bucket.list_files(domain="West_Coast", version="v1.0.1")) == 2
    assert bucket.list_files(domain="Atlantic") == []


def test_summary_groups_by_version_domain(fake_client: FakeS3) -> None:
    """One summary row per (version, domain) with the year span and size."""
    rows = {(r.version, r.domain): r for r in bucket.summary()}
    west = rows[("v1.0.1", "West_Coast")]
    assert (west.first_year, west.last_year, west.n_files) == (2010, 2011, 2)
    assert west.total_gb == pytest.approx(161.0)


def test_available_years_unions_versions(fake_client: FakeS3) -> None:
    """With no version, years union across every version that has the domain."""
    assert bucket.available_years("West_Coast") == [2010, 2011]
    assert bucket.available_years("West_Coast", version="v1.0.0") == [2010]


def test_latest_version(fake_client: FakeS3) -> None:
    """The newest version publishing the domain wins; absent domains give None."""
    assert bucket.latest_version("West_Coast") == "v1.0.1"
    assert bucket.latest_version("Atlantic") is None
