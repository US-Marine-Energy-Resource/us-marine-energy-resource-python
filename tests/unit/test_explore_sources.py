"""Source dispatch and the HTTP range-read path, without touching the network."""

from __future__ import annotations

from pathlib import Path

import pytest

from us_marine_energy_resource.explore.errors import SourceError
from us_marine_energy_resource.explore.sources import (
    HttpSource,
    LocalSource,
    S3Source,
    resolve_source,
)


def test_resolve_dispatches_by_scheme(tmp_path: Path) -> None:
    """Resolve dispatches by scheme."""
    local = tmp_path / "x.bin"
    local.write_bytes(b"data")
    assert isinstance(resolve_source(str(local)), LocalSource)
    assert isinstance(resolve_source(f"file://{local}"), LocalSource)


def test_unsupported_scheme_errors() -> None:
    """Unsupported scheme errors."""
    with pytest.raises(SourceError, match="scheme"):
        resolve_source("ftp://host/path")


def test_local_source_rejects_missing(tmp_path: Path) -> None:
    """Local source rejects missing."""
    with pytest.raises(SourceError, match="not a file"):
        LocalSource(tmp_path / "nope.h5")


class _FakeResponse:
    """A minimal requests-like response backed by an in-memory blob."""

    def __init__(self, blob: bytes, headers: dict[str, str]) -> None:
        self._blob = blob
        self.headers = headers
        self.content = blob

    def raise_for_status(self) -> None:
        """No-op; the fake always succeeds."""


def _fake_requests(monkeypatch: pytest.MonkeyPatch, blob: bytes) -> None:
    """Patch the lazily imported ``requests`` module used by HttpSource."""
    import types

    mod = types.SimpleNamespace()

    def head(url: str, allow_redirects: bool = True, timeout: int = 30) -> _FakeResponse:
        return _FakeResponse(b"", {"Accept-Ranges": "bytes", "Content-Length": str(len(blob))})

    def get(url: str, headers: dict[str, str] | None = None, timeout: int = 30) -> _FakeResponse:
        rng = (headers or {}).get("Range", "")
        start, end = rng.removeprefix("bytes=").split("-")
        return _FakeResponse(blob[int(start) : int(end) + 1], {})

    mod.head = head
    mod.get = get
    monkeypatch.setattr(
        "us_marine_energy_resource.explore.lazy.importlib.import_module",
        lambda name: mod if name == "requests" else __import__(name),
    )


def test_http_source_peek_and_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """Http source peek and read."""
    blob = bytes(range(256)) * 40
    _fake_requests(monkeypatch, blob)
    src = HttpSource("https://example.org/data.bin")
    assert src.ref.size is not None and src.ref.size.bytes == len(blob)
    assert src.peek(8) == blob[:8]
    with src.open_binary() as handle:
        handle.seek(1000)
        assert handle.read(16) == blob[1000:1016]


def test_http_without_ranges_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Http without ranges errors."""
    import types

    mod = types.SimpleNamespace()

    def _head(url, allow_redirects=True, timeout=30):
        return _FakeResponse(b"", {"Content-Length": "10"})

    mod.head = _head
    monkeypatch.setattr(
        "us_marine_energy_resource.explore.lazy.importlib.import_module",
        lambda name: mod if name == "requests" else __import__(name),
    )
    src = HttpSource("https://example.org/noranges.bin")
    with pytest.raises(SourceError, match="range"), src.open_binary() as handle:
        handle.read(1)


def test_s3_uri_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    """S3 uri parsing."""
    import types

    captured = {}

    class _Info:
        size = 4096

    class _FS:
        def get_file_info(self, path: str) -> _Info:
            captured["path"] = path
            return _Info()

    fs_mod = types.SimpleNamespace(
        resolve_s3_region=lambda bucket: "us-west-2",
        S3FileSystem=lambda **kw: _FS(),
    )
    monkeypatch.setattr(
        "us_marine_energy_resource.explore.lazy.importlib.import_module",
        lambda name: fs_mod if name == "pyarrow.fs" else __import__(name),
    )
    src = S3Source("marine-energy-data", "us-tidal/x.h5")
    assert captured["path"] == "marine-energy-data/us-tidal/x.h5"
    assert src.ref.uri == "s3://marine-energy-data/us-tidal/x.h5"
    assert src.ref.size is not None and src.ref.size.bytes == 4096
