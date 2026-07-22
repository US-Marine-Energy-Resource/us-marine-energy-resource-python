"""The explore configuration dataclass."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from us_marine_energy_resource.explore.config import CONFIG
from us_marine_energy_resource.wave_hindcast.config import CONFIG as WAVE_CONFIG


def test_endpoints_cover_tidal_and_wave() -> None:
    """Both endpoints resolve, and the wave bucket is single-sourced."""
    assert CONFIG.endpoints["tidal"] == (CONFIG.tidal_bucket, CONFIG.tidal_prefix)
    assert CONFIG.endpoints["wave"] == (WAVE_CONFIG.s3_bucket, "")


def test_policy_defaults_are_positive() -> None:
    """Every transfer ceiling is a positive number of megabytes."""
    assert CONFIG.max_transfer_mb > 0
    assert CONFIG.max_memory_mb > 0
    assert CONFIG.max_download_mb > 0
    assert CONFIG.confirm_above_mb > 0


def test_settings_path_is_under_home() -> None:
    """The settings file lives directly under the user's home."""
    assert CONFIG.settings_path() == Path.home() / CONFIG.settings_filename


def test_completion_cache_path_is_under_home_cache() -> None:
    """The completion cache lives under the user's cache directory."""
    path = CONFIG.completion_cache_path()
    assert path == Path.home() / ".cache" / "mer" / "completion.json"


def test_data_extensions_are_lowercase_dotted() -> None:
    """Every extension starts with a dot and is lowercase for matching."""
    for ext in CONFIG.data_extensions:
        assert ext.startswith(".")
        assert ext == ext.lower()


def test_config_is_frozen() -> None:
    """Assignments raise so no module can mutate shared settings."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        CONFIG.tidal_bucket = "other"  # type: ignore[misc]
