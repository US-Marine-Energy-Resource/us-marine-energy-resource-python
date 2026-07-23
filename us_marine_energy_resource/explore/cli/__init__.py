"""The ``mer`` file verbs: ls, info, explore, download.

All four share one path grammar (endpoint names, endpoint sub-paths, ``s3://``
URLs, local paths) resolved by :func:`~us_marine_energy_resource.explore.catalog.
resolve_path` into a directory-like *prefix* or a *file*:

- ``ls``       one-level listing (terse)
- ``info``     metadata: prefix -> sizes + truncated tree; file -> format + attrs
- ``explore``  list + info combined: prefix -> browse tree; file -> structure + values
- ``download`` fetch a file to disk

This package's public surface is the four verb functions plus the help and
epilog text the umbrella CLI registers them with.
"""

from __future__ import annotations

from ._shared import (
    DOWNLOAD_HELP,
    EXPLORE_EPILOG,
    EXPLORE_HELP,
    INFO_HELP,
    LS_HELP,
)
from .download import download
from .explore import explore
from .info import info
from .ls import ls

__all__ = [
    "DOWNLOAD_HELP",
    "EXPLORE_EPILOG",
    "EXPLORE_HELP",
    "INFO_HELP",
    "LS_HELP",
    "download",
    "explore",
    "info",
    "ls",
]
