"""
US Marine Energy Resource — Python library for tidal energy data.

Download, cache, and query the H20 High Resolution Tidal Hindcast dataset
stored on AWS (marine-energy-data S3 bucket).

Quick start (high-level API)::

    from us_marine_energy_resource import tidal_hindcast as tidal

    df = tidal.get_data_at_point(lat=60.73, lon=-151.43)
    fig = tidal.plot_tidal_time_series(df)
    fig, stats = tidal.plot_velocity_exceedance(df)
    fig = tidal.generate_tidal_joint_probability(df, sigma_layer=4)

Low-level API::

    from us_marine_energy_resource.cache import S3CacheManager
    from us_marine_energy_resource.manifest import TidalManifestQuery

    cache = S3CacheManager(bucket="marine-energy-data", prefix="us-tidal")
    manifest_path, _ = find_latest_manifest_s3(cache)
    query = TidalManifestQuery(manifest_path, s3_cache=cache)
    result = query.query_nearest_point(lat=60.73, lon=-151.43)
"""

from __future__ import annotations

import importlib
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import tidal_hindcast
    from .cache import S3CacheManager
    from .manifest import TidalManifestQuery, find_latest_manifest_hpc, find_latest_manifest_s3

__all__ = [
    "S3CacheManager",
    "TidalManifestQuery",
    "find_latest_manifest_hpc",
    "find_latest_manifest_s3",
    "tidal_hindcast",
]

# Submodules that can be returned directly.
_SUBMODULES = {"tidal_hindcast"}

# Symbols that live inside a submodule — (submodule_path, attr_name).
_SYMBOLS: dict[str, tuple[str, str]] = {
    "S3CacheManager": (".cache", "S3CacheManager"),
    "TidalManifestQuery": (".manifest", "TidalManifestQuery"),
    "find_latest_manifest_hpc": (".manifest", "find_latest_manifest_hpc"),
    "find_latest_manifest_s3": (".manifest", "find_latest_manifest_s3"),
}


def __getattr__(name: str) -> object:
    this = sys.modules[__name__]

    if name in _SUBMODULES:
        module = importlib.import_module(f".{name}", __name__)
        setattr(this, name, module)  # cache so __getattr__ is not called again
        return module

    if name in _SYMBOLS:
        submod_path, attr = _SYMBOLS[name]
        module = importlib.import_module(submod_path, __name__)
        value = getattr(module, attr)
        setattr(this, name, value)
        return value

    raise AttributeError(f"module 'us_marine_energy_resource' has no attribute {name!r}")
