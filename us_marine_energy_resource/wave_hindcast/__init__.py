"""WPTO wave hindcast: point queries, node lookup, and S3 layout.

Quick start::

    from us_marine_energy_resource import wave_hindcast

    info = wave_hindcast.describe_point(44.567, -124.229)   # no download, no API key
    df = wave_hindcast.get_data_at_point(44.567, -124.229)  # blocks while the data downloads

Point queries pick a backend automatically. Small queries read straight from
the published files on S3 with no credentials, and large ones go through the
NLR developer download API, which needs ``NLR_DEVELOPER_API_KEY`` and
``NLR_DEVELOPER_EMAIL`` (force one with ``backend="s3"`` or
``backend="api"``). Node lookup (``describe_point``, ``nearest``) is offline
against a packaged index. Import is cheap: heavy dependencies load on first
use.
"""

from __future__ import annotations

import importlib
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import backend, config, domains, errors, index, index_build, nlr_api, nodes, s3_direct
    from .hindcast import (
        default_cache_dir,
        describe_point,
        get_data_at_point,
        load_site,
        sites_on_disk,
    )
    from .nodes import WaveNode, nearest

__all__ = [
    "WaveNode",
    "backend",
    "config",
    "default_cache_dir",
    "describe_point",
    "domains",
    "errors",
    "get_data_at_point",
    "index",
    "index_build",
    "load_site",
    "nearest",
    "nlr_api",
    "nodes",
    "s3_direct",
    "sites_on_disk",
]

# Submodules resolved on first attribute access, keeping `import
# us_marine_energy_resource.wave_hindcast` free of duckdb/pooch/boto3.
_SUBMODULES = {
    "backend",
    "config",
    "domains",
    "errors",
    "index",
    "index_build",
    "nlr_api",
    "nodes",
    "s3_direct",
}

# Symbols that live inside a submodule, as (submodule_path, attr_name).
_SYMBOLS: dict[str, tuple[str, str]] = {
    "default_cache_dir": (".hindcast", "default_cache_dir"),
    "describe_point": (".hindcast", "describe_point"),
    "get_data_at_point": (".hindcast", "get_data_at_point"),
    "load_site": (".hindcast", "load_site"),
    "sites_on_disk": (".hindcast", "sites_on_disk"),
    "WaveNode": (".nodes", "WaveNode"),
    "nearest": (".nodes", "nearest"),
}


def __getattr__(name: str) -> object:
    """Resolve a submodule or facade symbol on first access.

    Parameters
    ----------
    name : str
        Attribute being looked up.

    Returns
    -------
    object
        The submodule or symbol, cached on the package afterwards.

    Raises
    ------
    AttributeError
        The name is neither a submodule nor a facade symbol.
    """
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

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
