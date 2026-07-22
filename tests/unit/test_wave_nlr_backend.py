"""The NLR API backend: describing nodes and the full fetch flow."""

from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any

import pytest

from tests.unit.conftest import FakeResponse, FakeTime, _patch_requests, make_wave_archive
from us_marine_energy_resource.wave_hindcast.nlr_api import client
from us_marine_energy_resource.wave_hindcast.nlr_api.backend import ApiBackend
from us_marine_energy_resource.wave_hindcast.nodes import WaveNode

NODE = WaveNode(
    location_id=479519,
    domain="West_Coast",
    endpoint="us-west-coast-hindcast-download",
    lat=44.5682,
    lon=-124.228,
    distance_m=142.3,
)


def test_backend_describe_reports_domain_facts() -> None:
    """describe() carries the API's own year caps and direction corrections."""
    backend = ApiBackend()
    info = backend.describe(NODE)
    assert info.endpoint == "us-west-coast-hindcast-download"
    assert (info.first_year, info.last_year) == (1979, 2020)

    hawaii = WaveNode(1, "Hawaii", "hawaii-hindcast-download", 21.46, -157.75, 10.0)
    info = backend.describe(hawaii)
    assert info.last_year == 2010
    assert info.direction_transform == "270-x"


def test_backend_fetch_full_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_time: FakeTime
) -> None:
    """fetch() requests, waits, downloads, and organizes in one pass."""
    monkeypatch.setenv("NLR_DEVELOPER_API_KEY", "k")
    monkeypatch.setenv("NLR_DEVELOPER_EMAIL", "e@example.org")
    monkeypatch.setitem(client._attribute_cache, "West_Coast", ["significant_wave_height"])

    zip_bytes = make_wave_archive(tmp_path / "src.zip", years=(1979,)).read_bytes()

    def fake_post(url: str, params: dict[str, str], data: dict[str, str], timeout: int) -> Any:
        return FakeResponse(200, {"outputs": {"downloadUrl": "https://dl/a.zip", "message": "ok"}})

    def fake_head(url: str, timeout: int) -> Any:
        return FakeResponse(200, {})

    class StreamingResponse(FakeResponse):
        def iter_content(self, chunk_size: int) -> Any:
            yield zip_bytes

    def fake_get(url: str, stream: bool, timeout: int) -> Any:
        return StreamingResponse(200, {})

    _patch_requests(
        monkeypatch,
        types.SimpleNamespace(
            post=fake_post, head=fake_head, get=fake_get, RequestException=Exception
        ),
    )

    cache = tmp_path / "cache"
    events: list[str] = []
    ApiBackend().fetch(
        NODE,
        "mysite",
        requested_lat=44.567,
        requested_lon=-124.229,
        force=False,
        timeout_s=60,
        cache_dir=cache,
        on_event=events.append,
    )

    manifest = json.loads((cache / "requests.json").read_text())
    assert manifest["mysite"]["download_url"] == "https://dl/a.zip"
    assert (cache / "archives" / "mysite.zip").exists()
    assert (cache / "mysite_44.57_-124.23" / "metadata.json").exists()
    assert any("requesting mysite" in e for e in events)
