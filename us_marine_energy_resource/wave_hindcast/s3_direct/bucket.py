"""List the published wave hindcast .h5 files on S3.

The download API serves tidy CSV subsets, but the full source is the set of
.h5 files in the public ``wpto-pds-us-wave`` bucket. This module enumerates
them, answering what actually exists (versions, domains, years) and where the
API's own limits differ from the data.

    >>> from us_marine_energy_resource.wave_hindcast.s3_direct import bucket
    >>> files = bucket.list_files()               # list[WaveFile]
    >>> bucket.list_files(domain="Alaska")
    >>> bucket.summary()                          # list[DomainSummary]
    >>> bucket.available_years("West_Coast")

Reads are anonymous and unsigned, so stale AWS credentials in the environment
cannot break access to this public bucket. Virtual-buoy datasets are skipped.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from ...explore.lazy import lazy_import
from ..config import CONFIG

# Files are {version}/{Domain}/{stem}_{year}.h5; the year is the only reliably
# parseable part. Domain and version come from the key's leading path segments.
_YEAR_RE = re.compile(r"_(\d{4})\.h5$")


@dataclass(frozen=True)
class WaveFile:
    """One published .h5 file."""

    version: str
    domain: str
    year: int
    size_bytes: int
    key: str

    @property
    def uri(self) -> str:
        """Return the file's full ``s3://`` URI.

        Returns
        -------
        str
            The bucket URI joined with the object key.
        """
        return f"{CONFIG.s3_bucket_uri}/{self.key}"


@dataclass(frozen=True)
class DomainSummary:
    """Coverage of one (version, domain): year span, file count, total size."""

    version: str
    domain: str
    first_year: int
    last_year: int
    n_files: int
    total_gb: float


def _client() -> Any:
    """Build an anonymous S3 client. Imports are lazy so boto3 loads only here.

    Returns
    -------
    Any
        A boto3 S3 client that makes unsigned requests.
    """
    boto3 = lazy_import("boto3", "listing the wave hindcast bucket")
    botocore = lazy_import("botocore", "listing the wave hindcast bucket")
    config = lazy_import("botocore.config", "listing the wave hindcast bucket")

    return boto3.client("s3", config=config.Config(signature_version=botocore.UNSIGNED))


def _iter_objects(prefix: str = "") -> Iterator[tuple[str, int]]:
    """Every (key, size) under a prefix, following pagination.

    Parameters
    ----------
    prefix : str, optional
        Only list objects whose keys start with this string.

    Yields
    ------
    tuple of (str, int)
        The object key and its size in bytes.
    """
    client = _client()
    token = None
    while True:
        kwargs: dict[str, Any] = {"Bucket": CONFIG.s3_bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        page = client.list_objects_v2(**kwargs)
        for obj in page.get("Contents", []):
            yield obj["Key"], obj["Size"]
        if not page.get("IsTruncated"):
            return
        token = page["NextContinuationToken"]


def list_files(domain: str | None = None, version: str | None = None) -> list[WaveFile]:
    """Every published wave hindcast .h5 file.

    Parameters
    ----------
    domain : str, optional
        Restrict to one domain (e.g. ``"West_Coast"``, ``"Alaska"``).
    version : str, optional
        Restrict to one version (e.g. ``"v1.0.1"``).

    Returns
    -------
    list of WaveFile
        Sorted by version, domain, year. Non-``.h5`` objects and virtual-buoy
        datasets are excluded.
    """
    files = []
    for key, size in _iter_objects():
        if not key.endswith(".h5"):
            continue
        parts = key.split("/")
        if len(parts) < 3:
            continue
        ver, dom = parts[0], parts[1]
        if "virtual_buoy" in dom.lower() or "deprecated" in dom.lower():
            continue
        match = _YEAR_RE.search(parts[-1])
        if match is None:
            continue
        files.append(WaveFile(ver, dom, int(match.group(1)), size, key))

    if domain is not None:
        files = [f for f in files if f.domain == domain]
    if version is not None:
        files = [f for f in files if f.version == version]
    return sorted(files, key=lambda f: (f.version, f.domain, f.year))


def summary(files: list[WaveFile] | None = None) -> list[DomainSummary]:
    """One :class:`DomainSummary` per (version, domain).

    Parameters
    ----------
    files : list of WaveFile, optional
        Output of :func:`list_files`. Fetched if omitted.

    Returns
    -------
    list of DomainSummary
        Sorted by domain, then version.
    """
    if files is None:
        files = list_files()

    groups: dict[tuple[str, str], list[WaveFile]] = {}
    for f in files:
        groups.setdefault((f.version, f.domain), []).append(f)

    out = []
    for (ver, dom), group in groups.items():
        years = [f.year for f in group]
        out.append(
            DomainSummary(
                version=ver,
                domain=dom,
                first_year=min(years),
                last_year=max(years),
                n_files=len(group),
                total_gb=round(sum(f.size_bytes for f in group) / 1e9, 1),
            )
        )
    return sorted(out, key=lambda s: (s.domain, s.version))


def available_years(domain: str, version: str | None = None) -> list[int]:
    """Sorted list of years published for a domain.

    With no ``version``, returns the union across versions, the full set of
    years obtainable from S3, which for Atlantic and Hawaii exceeds what the
    download API serves.

    Parameters
    ----------
    domain : str
        Domain name.
    version : str, optional
        Restrict to one version.

    Returns
    -------
    list of int
        Ascending years.
    """
    return sorted({f.year for f in list_files(domain=domain, version=version)})


def latest_version(domain: str) -> str | None:
    """Return the newest version string that publishes this domain, or None.

    Parameters
    ----------
    domain : str
        Domain name.

    Returns
    -------
    str or None
        E.g. ``"v1.0.1"``; None when the domain is absent from the bucket.
    """
    versions = sorted({f.version for f in list_files(domain=domain)})
    return versions[-1] if versions else None
