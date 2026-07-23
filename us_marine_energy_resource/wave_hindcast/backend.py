"""The seam between the wave functions and backend that fetches the data.

Two backends exist: the NLR developer download API (:mod:`.nlr_api`, the
default) and direct range reads of the published .h5 files on S3
(:mod:`.s3_direct`, slower but with no server-side build to fail).

A backend's job is one grid node's record. Location resolution
(:mod:`.nodes`) and reading the on-disk result (:mod:`.hindcast`) live
outside it. The contract between a backend and the facade is the on-disk
layout: ``fetch`` must leave ``<cache_dir>/<name>_<lat>_<lon>/`` holding a
combined CSV and a ``metadata.json``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .nodes import WaveNode


@dataclass(frozen=True)
class BackendInfo:
    """What one backend serves for a domain.

    Backend-specific on purpose: the download API caps Atlantic and Hawaii at
    2010, while the .h5 files on S3 carry the full record, so a backend
    reading those files reports a different range here.
    """

    endpoint: str
    first_year: int
    last_year: int
    interval_minutes: int
    direction_transform: str | None


class WaveBackend(Protocol):
    """Fetch the hindcast record for one grid node."""

    def describe(self, node: WaveNode) -> BackendInfo:
        """Report what this backend serves for the node's domain.

        Parameters
        ----------
        node : WaveNode
            The resolved grid node.

        Returns
        -------
        BackendInfo
            Year range, interval, and direction handling.
        """
        ...

    def fetch(
        self,
        node: WaveNode,
        name: str,
        *,
        requested_lat: float,
        requested_lon: float,
        force: bool,
        timeout_s: int,
        cache_dir: Path,
        on_event: Callable[[str], None],
        years: list[int] | None = None,
        variables: list[str] | None = None,
    ) -> None:
        """Block until the node's record is organized under ``cache_dir``.

        Parameters
        ----------
        node : WaveNode
            The resolved grid node.
        name : str
            Site label naming the organized directory.
        requested_lat, requested_lon : float
            The coordinate originally asked for, kept in the metadata.
        force : bool
            Fetch again even when already on disk.
        timeout_s : int
            Ceiling on any server side wait.
        cache_dir : Path
            The wave cache root.
        on_event : callable
            Sink for progress messages.
        years, variables : list, optional
            Narrow the record. ``None`` means all.
        """
        ...


def get_backend(name: str = "api") -> WaveBackend:
    """Return the named backend.

    Parameters
    ----------
    name : str, default "api"
        ``"api"`` for the NLR developer download API, or ``"s3"`` for direct
        reads of the published .h5 files.

    Returns
    -------
    WaveBackend
        The backend instance.

    Raises
    ------
    ValueError
        The name is not a known backend.
    """
    # These imports must stay function local: both backend modules import
    # this module for BackendInfo, so a module level import here would cycle.
    if name == "api":
        from .nlr_api.backend import ApiBackend

        return ApiBackend()
    if name == "s3":
        from .s3_direct.backend import S3Backend

        return S3Backend()
    raise ValueError(f"unknown wave backend {name!r}. The choices are 'api' and 's3'.")
