"""The wave hindcast configuration dataclass."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from us_marine_energy_resource.wave_hindcast.config import CONFIG


def test_bucket_uri_derives_from_bare_name() -> None:
    """The s3:// form always matches the bare bucket name."""
    assert CONFIG.s3_bucket_uri == f"s3://{CONFIG.s3_bucket}"
    assert CONFIG.s3_bucket_uri == "s3://wpto-pds-us-wave"


def test_non_site_dirnames_covers_both_backends() -> None:
    """Both backend working directories are excluded from site scans."""
    assert CONFIG.non_site_dirnames == frozenset({"archives", "s3_chunks"})


def test_default_timeout() -> None:
    """The archive wait ceiling is two hours."""
    assert CONFIG.default_timeout_s == 7200


def test_config_is_frozen() -> None:
    """Assignments raise so no module can mutate shared settings."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        CONFIG.s3_bucket = "other"  # type: ignore[misc]


def test_default_cache_dir_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """The env var wins over the home directory default."""
    monkeypatch.setenv(CONFIG.cache_dir_env, "/tmp/somewhere")
    assert CONFIG.default_cache_dir() == Path("/tmp/somewhere")

    monkeypatch.delenv(CONFIG.cache_dir_env)
    assert CONFIG.default_cache_dir() == Path.home() / CONFIG.cache_dir_name
