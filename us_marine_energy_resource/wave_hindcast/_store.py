"""The on-disk site layout shared by every wave backend.

A backend must leave ``<cache_dir>/<name>_<lat>_<lon>/`` holding a combined
CSV and a ``metadata.json``. The helpers here are the single spelling of that
contract, so the backends and the readers in :mod:`.hindcast` can never
drift apart. Layouts only one backend writes stay in that backend.
"""

from __future__ import annotations

import json
from pathlib import Path

METADATA_FILENAME = "metadata.json"


def _noop(_message: str) -> None:
    """Swallow an event, the default sink for library callers.

    Parameters
    ----------
    _message : str
        The event text, ignored.
    """


def site_stem(name: str, lat: object, lon: object) -> str:
    """Build the directory and file stem for one site.

    Parameters
    ----------
    name : str
        Site label.
    lat, lon : object
        Node coordinates, already coerced to the form the caller wants
        rendered.

    Returns
    -------
    str
        ``<name>_<lat>_<lon>``.
    """
    return f"{name}_{lat}_{lon}"


def combined_csv_name(stem: str, first: object, last: object) -> str:
    """Build the file name of the combined record CSV.

    Parameters
    ----------
    stem : str
        A :func:`site_stem` result.
    first, last : object
        First and last year in the record.

    Returns
    -------
    str
        ``<stem>_<first>-<last>.csv``.
    """
    return f"{stem}_{first}-{last}.csv"


def combined_csv_glob(stem: str) -> str:
    """Build the glob that finds a site's combined CSV.

    Parameters
    ----------
    stem : str
        A :func:`site_stem` result.

    Returns
    -------
    str
        Glob pattern matching :func:`combined_csv_name` output.
    """
    return f"{stem}_*-*.csv"


def point_name(lat: float, lon: float) -> str:
    """Build the default site label for a coordinate query.

    The CLI and the library must agree on this, because it is the cache key
    a later call uses to find the download.

    Parameters
    ----------
    lat, lon : float
        The requested coordinate in degrees.

    Returns
    -------
    str
        A filesystem safe label with minus signs spelled as ``m``.
    """
    return f"point_{lat:.4f}_{lon:.4f}".replace("-", "m")


def write_json(path: Path, payload: dict) -> None:
    """Write a JSON file in the package's one true format.

    Creates parent directories, indents, sorts keys, and ends with a
    newline.

    Parameters
    ----------
    path : Path
        Destination file.
    payload : dict
        JSON serializable content.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
