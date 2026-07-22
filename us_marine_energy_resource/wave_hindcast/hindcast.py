"""Coordinates in, wave hindcast out.

    >>> from us_marine_energy_resource import wave_hindcast
    >>> df = wave_hindcast.get_data_at_point(19.7283, -156.0624)

``df`` is indexed by UTC timestamp with one column per wave variable, covering
every year the hindcast serves for that location, and ``df.attrs`` carries the
metadata. Behind that call the coordinate is resolved to a grid node offline,
the full record is requested from the API, and the archive is downloaded and
organized. The call blocks, since archives take minutes to build server-side.
Results are cached under ``~/.mer_wave_cache`` (override with ``cache_dir=``
or ``MER_WAVE_CACHE_DIR``), so repeat calls are instant and offline.

Everything else here is optional:

    wave_hindcast.describe_point(lat, lon)   # which node/domain/years, no download
    wave_hindcast.load_site("US_Hawaii_Oahu_WETS")   # re-read something already on disk
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..explore.lazy import lazy_import
from . import _store, errors, nodes
from . import backend as _backend
from .config import CONFIG
from .domains import check_api_outage

if TYPE_CHECKING:
    import pandas as pd


def default_cache_dir() -> Path:
    """Return the wave download cache root.

    ``MER_WAVE_CACHE_DIR`` overrides the default ``~/.mer_wave_cache`` (a
    sibling of the tidal ``~/.us_tidal_cache``).

    Returns
    -------
    Path
        The cache root directory.
    """
    return CONFIG.default_cache_dir()


def _site_dir(name: str, cache_dir: Path) -> Path | None:
    """Find the organized directory for a site, if it has been downloaded.

    Parameters
    ----------
    name : str
        Site label.
    cache_dir : Path
        The wave cache root.

    Returns
    -------
    Path or None
        The site directory, or None when nothing is on disk yet.
    """
    matches = sorted(p for p in cache_dir.glob(f"{name}_*") if p.is_dir())
    return matches[0] if matches else None


def load_site(name: str, *, cache_dir: Path | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read an already-downloaded site from disk.

    Parameters
    ----------
    name : str
        Site label, e.g. the ``point_...`` label a coordinate query used.
    cache_dir : Path, optional
        The wave cache root; defaults to :func:`default_cache_dir`.

    Returns
    -------
    (pandas.DataFrame, dict)
        The combined record indexed by UTC timestamp, and its metadata.

    Raises
    ------
    CacheMissError
        Nothing on disk for this site yet. Subclasses ``FileNotFoundError``.
    """
    pd = lazy_import("pandas", "loading a cached wave hindcast site")
    root = cache_dir or default_cache_dir()
    directory = _site_dir(name, root)
    if directory is None:
        raise errors.CacheMissError(
            f"{name} is not cached in {root}. "
            "Use wave_hindcast.get_data_at_point(lat, lon) to fetch it."
        )
    combined = sorted(directory.glob(_store.combined_csv_glob(directory.name)))
    frame = pd.read_csv(combined[0], parse_dates=["timestamp"]).set_index("timestamp")
    metadata = json.loads((directory / _store.METADATA_FILENAME).read_text())
    return frame, metadata


def describe_point(
    lat: float, lon: float, *, domain: str | None = None, backend: str = "api"
) -> dict[str, Any]:
    """Resolve what a point maps to, without downloading anything.

    Free and offline (after the node index is cached): no credentials, no
    request, no archive.

    Parameters
    ----------
    lat, lon : float
        Degrees. Note the order is lat then lon, unlike the API's WKT.
    domain : str, optional
        Force a hindcast domain instead of resolving it from coverage.
    backend : str, default "api"
        Which backend's view to report; year ranges differ per backend.

    Returns
    -------
    dict
        ``location_id``, ``domain``, ``endpoint``, the node's own coordinates,
        ``distance_m`` from the requested point, the year range the backend
        serves, and the direction correction it needs (``None`` for every
        domain except Hawaii).

    Raises
    ------
    PointOutsideDomainError
        The point is outside every hindcast domain.
    """
    node = nodes.nearest(lat, lon, domain=domain)
    assert isinstance(node, nodes.WaveNode)
    info = _backend.get_backend(backend).describe(node)
    return {
        "location_id": node.location_id,
        "domain": node.domain,
        "endpoint": info.endpoint,
        "requested_lat": lat,
        "requested_lon": lon,
        "node_lat": node.lat,
        "node_lon": node.lon,
        "distance_m": node.distance_m,
        "years": [info.first_year, info.last_year],
        "n_years": info.last_year - info.first_year + 1,
        "interval_minutes": info.interval_minutes,
        "direction_transform": info.direction_transform,
    }


def get_data_at_point(
    lat: float,
    lon: float,
    *,
    name: str | None = None,
    domain: str | None = None,
    force: bool = False,
    cache_dir: Path | None = None,
    backend: str = "api",
    timeout_s: int = CONFIG.default_timeout_s,
    years: list[int] | None = None,
    variables: list[str] | None = None,
    on_event: Callable[[str], None] | None = None,
    return_metadata: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch the full hindcast record for the grid node nearest a coordinate.

    Blocks while the archive is built server-side, then caches the result so
    repeat calls are instant.

    Parameters
    ----------
    lat, lon : float
        Degrees. lat first, unlike the API's longitude-first WKT.
    name : str, optional
        Label for the site, used for the directory and cache key. Defaults to
        ``point_<lat>_<lon>``.
    domain : str, optional
        Force a hindcast domain rather than resolving it from coverage.
    force : bool, default False
        Re-request even if the site is already on disk.
    cache_dir : Path, optional
        The wave cache root; defaults to :func:`default_cache_dir`.
    backend : str, default "api"
        ``"api"`` for the NLR developer download API (needs
        ``NLR_DEVELOPER_API_KEY``/``NLR_DEVELOPER_EMAIL``), or ``"s3"`` for
        direct reads of the published .h5 files (no key, slower for big
        requests).
    timeout_s : int, default 7200
        Ceiling on the archive wait. Timing out loses the wait, not the
        archive. The request is saved and a retry resumes.
    years, variables : list, optional
        Narrow the record; ``None`` means everything the backend serves.
        Narrowed results cache under a name that includes the year span.
    on_event : callable, optional
        Sink for progress messages (the CLI passes one; library use defaults
        to silent).
    return_metadata : bool, default False
        Also return the metadata dict instead of only attaching it to
        ``df.attrs``.

    Returns
    -------
    pandas.DataFrame or (pandas.DataFrame, dict)
        Indexed by UTC timestamp, one column per variable, with
        ``Year``/``Month``/``Day``/``Hour``/``Minute`` retained alongside.
        Metadata rides on ``df.attrs`` either way.

    Raises
    ------
    PointOutsideDomainError
        The point is outside every hindcast domain.
    ApiOutageError
        The domain's API is recorded as broken upstream and ``backend="api"``
        was asked for (cached data and the s3 backend still work).
    ArchiveTimeoutError
        The archive was not ready within ``timeout_s``.

    Notes
    -----
    Directions use the meteorological convention (degrees clockwise from
    north, the direction waves come FROM), with the Hawaii domain's values
    corrected on the way through. ``metadata['direction_transform']`` records
    whether that happened.
    """
    node = nodes.nearest(lat, lon, domain=domain)
    assert isinstance(node, nodes.WaveNode)
    if name is None:
        name = _store.point_name(lat, lon)
        # A narrowed record must not shadow the full one in the cache.
        if years:
            name += f"_y{min(years)}-{max(years)}"
    root = cache_dir or default_cache_dir()

    frame = metadata = None
    if not force:
        with contextlib.suppress(errors.CacheMissError):
            frame, metadata = load_site(name, cache_dir=root)

    if frame is None or metadata is None:
        # Checked only once the cache has been consulted: data already
        # downloaded stays readable even after its domain goes down.
        if backend == "api":
            check_api_outage(node.domain)

        _backend.get_backend(backend).fetch(
            node,
            name,
            requested_lat=lat,
            requested_lon=lon,
            force=force,
            timeout_s=timeout_s,
            cache_dir=root,
            on_event=on_event or _store._noop,
            years=years,
            variables=variables,
        )
        frame, metadata = load_site(name, cache_dir=root)

    frame.attrs.update(metadata)
    if return_metadata:
        return frame, metadata
    return frame


def sites_on_disk(cache_dir: Path | None = None) -> list[str]:
    """List the names of sites already downloaded and organized.

    Parameters
    ----------
    cache_dir : Path, optional
        The wave cache root; defaults to :func:`default_cache_dir`.

    Returns
    -------
    list of str
        Sorted site labels.
    """
    root = cache_dir or default_cache_dir()
    if not root.is_dir():
        return []
    return sorted(
        p.name.rsplit("_", 2)[0]
        for p in root.iterdir()
        if p.is_dir() and p.name not in CONFIG.non_site_dirnames
    )
