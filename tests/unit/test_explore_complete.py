"""Shell completion for the shared path argument."""

from __future__ import annotations

from pathlib import Path

import pytest

from us_marine_energy_resource.explore.cli import complete

_WEST = (
    ["v1.0.1/West_Coast/"],
    ["v1.0.1/index.html"],
)

_TIDAL = (
    ["us-tidal/AK_aleutian_islands/", "us-tidal/AK_cook_inlet/", "us-tidal/manifest/"],
    [],
)

# Bound before the autouse fixture replaces the module attribute, for the one
# test that exercises the real spawn logic.
_REAL_SPAWN = complete._spawn_prefetch


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep cache I/O away from the user's cache and never spawn prefetchers."""
    monkeypatch.setattr(complete, "_CACHE_FILE", tmp_path / "completion.json")
    monkeypatch.setattr(complete, "_spawn_prefetch", lambda bucket, prefixes: None)


@pytest.fixture
def prefetches(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, list[str]]]:
    """Record prefetch requests instead of spawning processes."""
    seen: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(
        complete, "_spawn_prefetch", lambda bucket, prefixes: seen.append((bucket, prefixes))
    )
    return seen


def test_endpoint_names_complete() -> None:
    """A partial endpoint name completes to the endpoints, slash included."""
    assert complete.complete_path("ti") == ["tidal/"]
    assert complete.complete_path("wav") == ["wave/"]


def test_exact_endpoint_gains_slash() -> None:
    """An exact endpoint name completes to itself with a trailing slash."""
    assert complete.complete_path("tidal") == ["tidal/"]


def test_endpoint_children_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    """endpoint/partial completes from a one-level S3 listing."""
    seen: list[tuple[str, str]] = []

    def fake_children(bucket: str, prefix: str) -> tuple[list[str], list[str]]:
        seen.append((bucket, prefix))
        return _TIDAL

    monkeypatch.setattr(complete, "_s3_children", fake_children)
    assert complete.complete_path("tidal/AK_c") == ["tidal/AK_cook_inlet/"]
    assert seen == [("marine-energy-data", "us-tidal/")]


def test_endpoint_deeper_level(monkeypatch: pytest.MonkeyPatch) -> None:
    """A deeper partial lists the parent level, not the root."""
    seen: list[tuple[str, str]] = []

    def fake_children(bucket: str, prefix: str) -> tuple[list[str], list[str]]:
        seen.append((bucket, prefix))
        return (["v1.0.1/West_Coast/"], [])

    monkeypatch.setattr(complete, "_s3_children", fake_children)
    assert complete.complete_path("wave/v1.0.1/We") == ["wave/v1.0.1/West_Coast/"]
    assert seen == [("wpto-pds-us-wave", "v1.0.1/")]


def test_s3_uri_completes(monkeypatch: pytest.MonkeyPatch) -> None:
    """An s3:// prefix completes its children with the scheme kept."""
    monkeypatch.setattr(complete, "_s3_children", lambda bucket, prefix: _WEST)
    out = complete.complete_path("s3://wpto-pds-us-wave/v1.0.1/W")
    assert out == ["s3://wpto-pds-us-wave/v1.0.1/West_Coast/"]


def test_s3_bucket_alone_completes_nothing() -> None:
    """Bucket names cannot be listed anonymously, so no suggestions."""
    assert complete.complete_path("s3://wpto") == []


def test_local_paths_complete(tmp_path: Path) -> None:
    """Local files and directories complete, directories with a slash."""
    (tmp_path / "data.h5").write_bytes(b"x")
    (tmp_path / "deeper").mkdir()
    (tmp_path / ".hidden").write_bytes(b"x")
    out = complete.complete_path(f"{tmp_path}/d")
    assert out == [f"{tmp_path}/data.h5", f"{tmp_path}/deeper/"]


def test_failures_complete_to_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Network trouble on TAB never raises, it just suggests nothing."""

    def boom(bucket: str, prefix: str) -> tuple[list[str], list[str]]:
        raise TimeoutError("no network")

    monkeypatch.setattr(complete, "_s3_children", boom)
    assert complete.complete_path("tidal/AK_c") == []


def test_repeat_presses_hit_the_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """A second TAB at the same level answers from disk with no listing."""
    calls: list[str] = []

    def fake_children(bucket: str, prefix: str) -> tuple[list[str], list[str]]:
        calls.append(prefix)
        return _TIDAL

    monkeypatch.setattr(complete, "_s3_children", fake_children)
    assert complete.complete_path("tidal/AK_c") == ["tidal/AK_cook_inlet/"]
    assert complete.complete_path("tidal/AK_a") == ["tidal/AK_aleutian_islands/"]
    assert calls == ["us-tidal/"]


def test_cache_answers_when_the_network_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """A press that fails online still completes from a recent cached listing."""
    monkeypatch.setattr(complete, "_s3_children", lambda bucket, prefix: _TIDAL)
    complete.complete_path("tidal/AK_c")  # warm the cache

    def boom(bucket: str, prefix: str) -> tuple[list[str], list[str]]:
        raise TimeoutError("no network")

    monkeypatch.setattr(complete, "_s3_children", boom)
    assert complete.complete_path("tidal/AK_c") == ["tidal/AK_cook_inlet/"]


def test_stale_cache_refreshes(monkeypatch: pytest.MonkeyPatch) -> None:
    """An entry older than the TTL is fetched again."""
    calls: list[str] = []

    def fake_children(bucket: str, prefix: str) -> tuple[list[str], list[str]]:
        calls.append(prefix)
        return _TIDAL

    monkeypatch.setattr(complete, "_s3_children", fake_children)
    complete.complete_path("tidal/AK_c")

    import json

    cache = json.loads(complete._CACHE_FILE.read_text())
    for entry in cache.values():
        entry["at"] -= complete._CACHE_TTL_S + 1
    complete._CACHE_FILE.write_text(json.dumps(cache))

    complete.complete_path("tidal/AK_c")
    assert calls == ["us-tidal/", "us-tidal/"]


def test_corrupt_cache_is_survived(monkeypatch: pytest.MonkeyPatch) -> None:
    """A broken cache file falls back to a plain listing."""
    complete._CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    complete._CACHE_FILE.write_text("{not json")
    monkeypatch.setattr(complete, "_s3_children", lambda bucket, prefix: _TIDAL)
    assert complete.complete_path("tidal/AK_c") == ["tidal/AK_cook_inlet/"]


def test_matched_directories_are_prefetched(
    monkeypatch: pytest.MonkeyPatch, prefetches: list[tuple[str, list[str]]]
) -> None:
    """Completing a level asks for its candidate directories ahead of time."""
    monkeypatch.setattr(complete, "_s3_children", lambda bucket, prefix: _TIDAL)
    complete.complete_path("tidal/AK_c")
    assert prefetches == [("marine-energy-data", ["us-tidal/AK_cook_inlet/"])]


def test_endpoint_names_prefetch_their_roots(prefetches: list[tuple[str, list[str]]]) -> None:
    """Completing an endpoint name warms that endpoint's first level."""
    complete.complete_path("tid")
    assert prefetches == [("marine-energy-data", ["us-tidal/"])]

    prefetches.clear()
    complete.complete_path("wave")
    assert prefetches == [("wpto-pds-us-wave", [""])]


def test_prefetch_main_fills_the_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """The detached prefetcher writes each requested level into the cache."""
    calls: list[str] = []

    def fake_children(bucket: str, prefix: str) -> tuple[list[str], list[str]]:
        calls.append(prefix)
        return ([f"{prefix}sub/"], [])

    monkeypatch.setattr(complete, "_s3_children", fake_children)
    complete._prefetch_main(["marine-energy-data", "us-tidal/AK_cook_inlet/", "us-tidal/manifest/"])
    assert calls == ["us-tidal/AK_cook_inlet/", "us-tidal/manifest/"]

    # The fetched levels now answer without a listing.
    monkeypatch.setattr(
        complete,
        "_s3_children",
        lambda bucket, prefix: (_ for _ in ()).throw(TimeoutError("offline")),
    )
    assert complete.complete_path("tidal/AK_cook_inlet/") == ["tidal/AK_cook_inlet/sub/"]


def test_spawn_skips_fresh_prefixes(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fully warmed path spawns no background process at all."""
    monkeypatch.setattr(complete, "_s3_children", lambda bucket, prefix: ([], []))
    complete._cached_children("marine-energy-data", "us-tidal/AK_cook_inlet/")

    spawned: list[list[str]] = []
    monkeypatch.setattr("subprocess.Popen", lambda cmd, **kwargs: spawned.append(cmd) or None)
    _REAL_SPAWN("marine-energy-data", ["us-tidal/AK_cook_inlet/"])
    assert spawned == []
    _REAL_SPAWN("marine-energy-data", ["us-tidal/other/"])
    assert len(spawned) == 1 and "us-tidal/other/" in spawned[0]
