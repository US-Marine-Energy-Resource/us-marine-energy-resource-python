"""Resolution order and fallbacks for the wave node-index files."""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from us_marine_energy_resource.wave_hindcast import errors, index
from us_marine_energy_resource.wave_hindcast.config import CONFIG


@pytest.fixture(autouse=True)
def _fresh_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear the module cache so each test sees its own environment."""
    monkeypatch.setattr(index, "_state", {})
    monkeypatch.delenv("MER_WAVE_INDEX_DIR", raising=False)
    monkeypatch.delenv("MER_WAVE_INDEX_URL", raising=False)


def _fake_pooch(
    monkeypatch: pytest.MonkeyPatch, cache: Path, retrieve: object
) -> types.SimpleNamespace:
    """Patch the lazily imported ``pooch`` module."""
    import importlib

    # Capture the real function first: the setattr below patches the shared
    # importlib module, so calling importlib.import_module inside the lambda
    # would recurse into itself.
    real_import = importlib.import_module
    mod = types.SimpleNamespace(os_cache=lambda name: cache, retrieve=retrieve)
    monkeypatch.setattr(
        "us_marine_energy_resource.explore.lazy.importlib.import_module",
        lambda name, *args: mod if name == "pooch" else real_import(name, *args),
    )
    return mod


def test_registry_parses() -> None:
    """The packaged registry maps every node file to a sha256."""
    entries = index.registry()
    assert len(entries) == 6
    assert all(name.startswith("nodes_") for name in entries)
    assert all(checksum.startswith("sha256:") for checksum in entries.values())


def test_load_index_shape() -> None:
    """The packaged JSON records the scale, gate, and node files."""
    idx = index.load_index()
    assert idx["coord_scale"] == 10**6
    assert idx["extent_cell_deg"] == 0.05
    assert set(idx["node_files"]) == {
        "West_Coast",
        "Atlantic",
        "Alaska",
        "Hawaii",
        "Gulf_of_Mexico_and_Puerto_Rico",
        "CNMI_and_Guam",
    }


def test_env_override_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """MER_WAVE_INDEX_DIR takes priority over every other source."""
    (tmp_path / "some_file.parquet").write_bytes(b"PAR1data")
    monkeypatch.setenv("MER_WAVE_INDEX_DIR", str(tmp_path))
    assert index.data_path("some_file.parquet") == tmp_path / "some_file.parquet"


def test_env_override_missing_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A set-but-wrong override errors instead of falling through to a download."""
    monkeypatch.setenv("MER_WAVE_INDEX_DIR", str(tmp_path))
    with pytest.raises(errors.IndexMissingError, match="MER_WAVE_INDEX_DIR"):
        index.data_path("nodes_West_Coast_v1.parquet")


def test_packaged_files_resolve() -> None:
    """The small in-package files resolve with no env and no network."""
    path = index.data_path("domain_extents_v1.parquet")
    assert path.is_file()
    assert path.parent.name == "data"


def test_lfs_pointer_is_not_real(tmp_path: Path) -> None:
    """A Git LFS pointer file is treated as absent, not as content."""
    pointer = tmp_path / "nodes_X_v1.parquet"
    pointer.write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\n")
    assert not index._is_real_file(pointer)
    real = tmp_path / "real.parquet"
    real.write_bytes(b"PAR1data")
    assert index._is_real_file(real)


def test_checkout_resolves_node_file() -> None:
    """A repo checkout's LFS directory serves the node files directly."""
    # This test runs inside the repo, where data/h2o_wave_hindcast_index/v1/
    # holds real parquet (or LFS pointers, if git-lfs was absent at checkout).
    path = index._checkout_path("nodes_West_Coast_v1.parquet")
    if path is not None:
        assert path.is_file()
        assert index.data_path("nodes_West_Coast_v1.parquet") == path


def test_fetch_via_pooch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With no local copy anywhere, the file is downloaded and cached."""
    monkeypatch.setattr(index, "_checkout_path", lambda filename: None)
    cache = tmp_path / "cache"
    calls: list[tuple[str, str | None]] = []

    def retrieve(
        url: str, known_hash: str | None, fname: str, path: Path, progressbar: bool
    ) -> str:
        calls.append((url, known_hash))
        dest = Path(path) / fname
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"PAR1data")
        return str(dest)

    _fake_pooch(monkeypatch, cache, retrieve)
    path = index.data_path("nodes_West_Coast_v1.parquet")
    assert path.read_bytes() == b"PAR1data"
    (url, known_hash) = calls[0]
    assert url == CONFIG.index_base_url + "nodes_West_Coast_v1.parquet"
    assert known_hash is not None and known_hash.startswith("sha256:")

    # Second call short-circuits on the cached file; no new download.
    assert index.data_path("nodes_West_Coast_v1.parquet") == path
    assert len(calls) == 1


def test_fetch_url_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """MER_WAVE_INDEX_URL repoints the download without a code change."""
    monkeypatch.setattr(index, "_checkout_path", lambda filename: None)
    monkeypatch.setenv("MER_WAVE_INDEX_URL", "https://example.org/idx/")
    seen: list[str] = []

    def retrieve(
        url: str, known_hash: str | None, fname: str, path: Path, progressbar: bool
    ) -> str:
        seen.append(url)
        dest = Path(path) / fname
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"PAR1data")
        return str(dest)

    _fake_pooch(monkeypatch, tmp_path / "cache", retrieve)
    index.data_path("nodes_Hawaii_v1.parquet")
    assert seen == ["https://example.org/idx/nodes_Hawaii_v1.parquet"]


def test_generation_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed download falls back to generating the file from S3."""
    monkeypatch.setattr(index, "_checkout_path", lambda filename: None)
    cache = tmp_path / "cache"

    def retrieve(**kwargs: object) -> str:
        raise OSError("404 Not Found")

    _fake_pooch(monkeypatch, cache, retrieve)

    built: list[str] = []

    def fake_build(domain: str, dest: Path, *, coord_scale: int) -> Path:
        built.append(domain)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"PAR1generated")
        return dest

    monkeypatch.setattr(
        "us_marine_energy_resource.wave_hindcast.index_build.build_domain_nodes", fake_build
    )

    with pytest.warns(UserWarning, match="built"):
        path = index.data_path("nodes_Alaska_v1.parquet")
    assert built == ["Alaska"]
    assert path.read_bytes() == b"PAR1generated"
    assert (cache / CONFIG.index_subdir / "nodes_Alaska_v1.parquet.generated").exists()

    # Subsequent calls use the generated file without another attempt.
    assert index.data_path("nodes_Alaska_v1.parquet") == path
    assert built == ["Alaska"]


def test_generation_refuses_non_node_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only node files can be generated; anything else is a hard miss."""
    monkeypatch.setattr(index, "_checkout_path", lambda filename: None)

    def retrieve(**kwargs: object) -> str:
        raise OSError("404 Not Found")

    _fake_pooch(monkeypatch, tmp_path / "cache", retrieve)
    with pytest.raises(errors.IndexMissingError, match="generated"):
        index.data_path("nodes_bogus_v9.parquet")
