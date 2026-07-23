"""The seam between the wave functions and backend that fetches the data.

Two backends exist: the NLR developer download API (:mod:`.nlr_api`) and
direct range reads of the published .h5 files on S3 (:mod:`.s3_direct`,
slower for big requests but with no credentials and no server-side build to
fail). The default is ``"auto"``: :func:`resolve_backend` reads small
queries straight from S3, so discovery needs no account, and hands large
ones to the API.

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

from .config import CONFIG
from .domains import API_OUTAGES, domain_config
from .nodes import WaveNode

# The auto backend reads small queries straight from S3 and hands large ones
# to the download API. The seam is measured in variable-years, the years
# fetched times the variables fetched: at the measured S3 cost of about
# 15 MB and 20 seconds per variable-year, 30 variable-years is roughly
# 450 MB and ten minutes of direct reading, which is where a server-built
# archive becomes the better deal.
AUTO_SEAM_VARIABLE_YEARS = 30


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


def resolve_backend(
    name: str,
    domain: str,
    *,
    years: list[int] | None = None,
    variables: list[str] | None = None,
) -> tuple[str, str | None]:
    """Resolve a backend name, turning ``"auto"`` into ``"api"`` or ``"s3"``.

    Auto reads small queries straight from the published files on S3, which
    needs no credentials, and hands large queries to the download API, which
    builds the archive server-side. A query is large past
    :data:`AUTO_SEAM_VARIABLE_YEARS` variable-years. A large query still
    stays on S3 when the API cannot serve it: a recorded outage, a requested
    year past the API's cap for the domain, or missing credentials.

    Parameters
    ----------
    name : str
        ``"auto"``, ``"api"``, or ``"s3"``. Explicit names pass through
        untouched.
    domain : str
        The resolved hindcast domain.
    years, variables : list, optional
        The narrowed request. ``None`` means everything served.

    Returns
    -------
    tuple of (str, str or None)
        The concrete backend name and, when auto made a choice worth
        explaining, one sentence saying why.
    """
    if name != "auto":
        return name, None

    config = domain_config(domain)
    n_years = len(years) if years else config["last_year"] - config["first_year"] + 1
    if variables:
        n_variables = len(variables)
    else:
        # Function local so this module stays importable by the backends.
        from .s3_direct.backend import TYPICAL_VARIABLES_PER_FILE

        n_variables = TYPICAL_VARIABLES_PER_FILE

    if n_years * n_variables <= AUTO_SEAM_VARIABLE_YEARS:
        return "s3", None
    if domain in API_OUTAGES:
        return "s3", (
            f"large query, but the {domain} API download service is not "
            "working right now, so this reads directly from S3"
        )
    if years and max(years) > config["last_year"]:
        return "s3", (
            f"large query, but the API serves {domain} only through "
            f"{config['last_year']}, so this reads directly from S3"
        )
    from .nlr_api.client import has_credentials

    if not has_credentials():
        return "s3", (
            "large query with no API key configured, so this reads directly "
            "from S3. The api backend is usually faster at this size, and a "
            f"free key is available at {CONFIG.signup_url}"
        )
    return "api", "large query, so the api backend builds the archive server-side"


def get_backend(name: str = "api") -> WaveBackend:
    """Return the named backend.

    Parameters
    ----------
    name : str, default "api"
        ``"api"`` for the NLR developer download API, or ``"s3"`` for direct
        reads of the published .h5 files. ``"auto"`` is not accepted here:
        resolve it first with :func:`resolve_backend`.

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
