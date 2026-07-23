"""The wave facade, with a fake backend injected at the seam."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from us_marine_energy_resource.wave_hindcast import backend as backend_mod
from us_marine_energy_resource.wave_hindcast import errors, hindcast
from us_marine_energy_resource.wave_hindcast.backend import BackendInfo
from us_marine_energy_resource.wave_hindcast.nodes import WaveNode

PACWAVE = (44.5670485, -124.22896475)


def _write_site(cache_dir: Path, name: str, node_lat: float = 44.57, node_lon: float = -124.23):
    """Materialize the on-disk layout a backend's fetch must produce."""
    site_dir = cache_dir / f"{name}_{node_lat}_{node_lon}"
    site_dir.mkdir(parents=True, exist_ok=True)
    (site_dir / f"{site_dir.name}_1979-1980.csv").write_text(
        "timestamp,Year,Significant Wave Height\n"
        "1979-01-01T00:00:00Z,1979,2.5\n"
        "1980-01-01T00:00:00Z,1980,3.0\n"
    )
    (site_dir / "metadata.json").write_text(
        json.dumps({"site": name, "gid": 479519, "domain": "West_Coast"})
    )
    return site_dir


class FakeBackend:
    """A backend that materializes the contract layout, counting calls."""

    def __init__(self) -> None:
        self.fetches: list[str] = []

    def describe(self, node: WaveNode) -> BackendInfo:
        """Report fixed facts for any domain."""
        return BackendInfo(
            endpoint=node.endpoint,
            first_year=1979,
            last_year=2020,
            interval_minutes=180,
            direction_transform=None,
        )

    def fetch(self, node: WaveNode, name: str, *, cache_dir: Path, **kwargs: Any) -> None:
        """Write the contract layout without any network."""
        self.fetches.append(name)
        _write_site(cache_dir, name)


@pytest.fixture
def fake_backend(monkeypatch: pytest.MonkeyPatch) -> FakeBackend:
    """Route get_backend() to a FakeBackend instance."""
    fake = FakeBackend()
    monkeypatch.setattr(backend_mod, "get_backend", lambda name="api": fake)
    return fake


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create an isolated wave cache dir and set it as the default."""
    root = tmp_path / "wave_cache"
    monkeypatch.setenv("MER_WAVE_CACHE_DIR", str(root))
    return root


def test_get_data_at_point_fetches_and_loads(
    wave_index_dir: Path, fake_backend: FakeBackend, cache: Path
) -> None:
    """A cold cache resolves the node, fetches, and returns the record."""
    frame = hindcast.get_data_at_point(*PACWAVE, name="mysite")
    assert fake_backend.fetches == ["mysite"]
    assert frame.index.name == "timestamp"
    assert frame.attrs["gid"] == 479519
    assert len(frame) == 2


def test_cache_hit_skips_backend(
    wave_index_dir: Path, fake_backend: FakeBackend, cache: Path
) -> None:
    """A warm cache never touches the backend."""
    _write_site(cache, "mysite")
    frame = hindcast.get_data_at_point(*PACWAVE, name="mysite")
    assert fake_backend.fetches == []
    assert len(frame) == 2


def test_force_refetches(wave_index_dir: Path, fake_backend: FakeBackend, cache: Path) -> None:
    """force=True goes back to the backend despite the cache."""
    _write_site(cache, "mysite")
    hindcast.get_data_at_point(*PACWAVE, name="mysite", force=True)
    assert fake_backend.fetches == ["mysite"]


def test_return_metadata_shapes(
    wave_index_dir: Path, fake_backend: FakeBackend, cache: Path
) -> None:
    """Metadata rides on attrs by default and is returned when asked for."""
    frame = hindcast.get_data_at_point(*PACWAVE, name="a")
    assert frame.attrs["site"] == "a"

    result = hindcast.get_data_at_point(*PACWAVE, name="a", return_metadata=True)
    assert isinstance(result, tuple)
    frame, metadata = result
    assert metadata["site"] == "a"
    assert frame.attrs == metadata


def test_default_name_from_coordinates(
    wave_index_dir: Path, fake_backend: FakeBackend, cache: Path
) -> None:
    """An unnamed point gets a coordinate-derived, dash-free label."""
    hindcast.get_data_at_point(*PACWAVE)
    assert fake_backend.fetches == ["point_44.5670_m124.2290"]


def test_api_outage_blocks_fetch(
    wave_index_dir: Path, fake_backend: FakeBackend, cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit api fetch on a domain in API_OUTAGES fails fast."""
    broken = WaveNode(
        location_id=7,
        domain="Gulf_of_Mexico_and_Puerto_Rico",
        endpoint="us-wave-v1-0-0-gom-and-pr-download",
        lat=18.6,
        lon=-66.1,
        distance_m=5.0,
    )
    monkeypatch.setattr(
        "us_marine_energy_resource.wave_hindcast.nodes.nearest", lambda *args, **kwargs: broken
    )
    with pytest.raises(errors.ApiOutageError) as excinfo:
        hindcast.get_data_at_point(18.6, -66.1, name="pr", backend="api")
    assert excinfo.value.domain == "Gulf_of_Mexico_and_Puerto_Rico"
    assert fake_backend.fetches == []


def test_api_outage_does_not_block_s3_backend(
    wave_index_dir: Path, fake_backend: FakeBackend, cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The outage is the API's alone, so backend="s3" still fetches."""
    broken = WaveNode(
        location_id=7,
        domain="Gulf_of_Mexico_and_Puerto_Rico",
        endpoint="us-wave-v1-0-0-gom-and-pr-download",
        lat=18.6,
        lon=-66.1,
        distance_m=5.0,
    )
    monkeypatch.setattr(
        "us_marine_energy_resource.wave_hindcast.nodes.nearest", lambda *args, **kwargs: broken
    )
    frame = hindcast.get_data_at_point(18.6, -66.1, name="pr", backend="s3")
    assert fake_backend.fetches == ["pr"]
    assert len(frame) == 2


def test_api_outage_cached_data_still_loads(
    wave_index_dir: Path, fake_backend: FakeBackend, cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Data downloaded before an outage stays readable during it."""
    broken = WaveNode(
        location_id=7,
        domain="Gulf_of_Mexico_and_Puerto_Rico",
        endpoint="us-wave-v1-0-0-gom-and-pr-download",
        lat=18.6,
        lon=-66.1,
        distance_m=5.0,
    )
    monkeypatch.setattr(
        "us_marine_energy_resource.wave_hindcast.nodes.nearest", lambda *args, **kwargs: broken
    )
    _write_site(cache, "pr")
    frame = hindcast.get_data_at_point(18.6, -66.1, name="pr")
    assert len(frame) == 2


def test_describe_point(wave_index_dir: Path, fake_backend: FakeBackend) -> None:
    """describe_point merges the node with the backend's view of the domain."""
    info = hindcast.describe_point(*PACWAVE)
    assert info["domain"] == "West_Coast"
    assert info["location_id"] == 0
    assert info["endpoint"] == "us-west-coast-hindcast-download"
    assert info["years"] == [1979, 2020]
    assert info["n_years"] == 42
    assert info["interval_minutes"] == 180
    assert info["requested_lat"] == PACWAVE[0]
    assert info["distance_m"] > 0


def test_load_site_missing_raises(cache: Path) -> None:
    """An empty cache is a CacheMissError, which is a FileNotFoundError."""
    with pytest.raises(errors.CacheMissError):
        hindcast.load_site("nothing")
    with pytest.raises(FileNotFoundError):
        hindcast.load_site("nothing")


def test_sites_on_disk(cache: Path) -> None:
    """sites_on_disk lists organized directories, ignoring archives/."""
    assert hindcast.sites_on_disk() == []
    _write_site(cache, "alpha")
    _write_site(cache, "beta", node_lat=21.46, node_lon=-157.75)
    (cache / "archives").mkdir()
    assert hindcast.sites_on_disk() == ["alpha", "beta"]


def test_sites_on_disk_ignores_backend_working_dirs(cache: Path) -> None:
    """Neither backend working directory surfaces as a phantom site."""
    _write_site(cache, "alpha")
    (cache / "archives").mkdir()
    (cache / "s3_chunks").mkdir()
    assert hindcast.sites_on_disk() == ["alpha"]


def test_unknown_backend_raises() -> None:
    """An unknown backend name fails with the available options listed."""
    with pytest.raises(ValueError, match="api"):
        backend_mod.get_backend("hsds")


def test_resolve_backend_passthrough() -> None:
    """Explicit names resolve to themselves with no explanation."""
    assert backend_mod.resolve_backend("api", "West_Coast") == ("api", None)
    assert backend_mod.resolve_backend("s3", "West_Coast") == ("s3", None)


def test_resolve_backend_small_query_reads_s3() -> None:
    """At or under the seam, auto reads S3 with no key involved."""
    name, reason = backend_mod.resolve_backend(
        "auto", "West_Coast", years=[2020], variables=["a", "b", "c", "d"]
    )
    assert name == "s3" and reason is None


def test_resolve_backend_large_query_uses_api_with_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Past the seam, auto hands the query to the api when a key exists."""
    monkeypatch.setattr(
        "us_marine_energy_resource.wave_hindcast.nlr_api.client.has_credentials", lambda: True
    )
    name, reason = backend_mod.resolve_backend(
        "auto", "West_Coast", years=list(range(2011, 2021)), variables=["a", "b", "c", "d"]
    )
    assert name == "api" and reason is not None


def test_resolve_backend_large_query_stays_on_s3_without_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Past the seam with no key, auto stays on S3 and says why."""
    monkeypatch.setattr(
        "us_marine_energy_resource.wave_hindcast.nlr_api.client.has_credentials", lambda: False
    )
    name, reason = backend_mod.resolve_backend(
        "auto", "West_Coast", years=list(range(2011, 2021)), variables=["a", "b", "c", "d"]
    )
    assert name == "s3" and reason is not None and "key" in reason


def test_resolve_backend_outage_domain_reads_s3() -> None:
    """A large query on a domain with a broken API reads S3."""
    name, reason = backend_mod.resolve_backend(
        "auto",
        "Gulf_of_Mexico_and_Puerto_Rico",
        years=list(range(2011, 2021)),
        variables=["a", "b", "c", "d"],
    )
    assert name == "s3" and reason is not None and "not working" in reason


def test_resolve_backend_api_capped_years_read_s3() -> None:
    """Years past the api's cap for a domain read S3, key or no key."""
    name, reason = backend_mod.resolve_backend(
        "auto", "Atlantic", years=list(range(2011, 2021)), variables=["a", "b", "c", "d"]
    )
    assert name == "s3" and reason is not None and "2010" in reason
