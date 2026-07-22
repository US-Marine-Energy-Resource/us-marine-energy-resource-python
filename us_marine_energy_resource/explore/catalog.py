"""Resolve a path to a prefix or a file, and browse prefixes one level at a time.

Every ``mer`` verb (ls, info, explore, download) shares this path grammar:
endpoint names (``tidal``, ``wave``), endpoint sub-paths, ``s3://…`` URLs, and
local paths. A path resolves to a directory-like *prefix* or a *file*. Prefix
listing is always delimited, capped per level, and page-bounded, so it never
recurses into buckets holding millions of objects.
"""

from __future__ import annotations

import dataclasses
import fnmatch
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from .config import CONFIG

# Endpoint name -> (bucket, root prefix). The wave root is the bucket itself.
ENDPOINTS: dict[str, tuple[str, str]] = CONFIG.endpoints

# Extensions that mark a path as a file rather than a directory prefix.
_DATA_EXTS = CONFIG.data_extensions

_MAX_PAGES = 5  # hard bound on S3 round-trips per level, even with a narrow filter


@dataclasses.dataclass(frozen=True)
class Entry:
    """One immediate child of a prefix: a subdirectory or an object."""

    name: str
    is_dir: bool
    size: int | None


@dataclasses.dataclass(frozen=True)
class Listing:
    """The capped set of immediate children under one prefix."""

    prefix: str
    entries: tuple[Entry, ...]
    truncated: bool

    @property
    def n_files(self) -> int:
        """Return the number of file entries shown.

        Returns
        -------
        int
            Count of entries that are files.
        """
        return sum(1 for e in self.entries if not e.is_dir)

    @property
    def n_dirs(self) -> int:
        """Return the number of directory entries shown.

        Returns
        -------
        int
            Count of entries that are directories.
        """
        return sum(1 for e in self.entries if e.is_dir)

    @property
    def total_file_bytes(self) -> int:
        """Return the sum of shown file sizes.

        Returns
        -------
        int
            Total bytes across the file entries.
        """
        return sum(e.size or 0 for e in self.entries if not e.is_dir)


@dataclasses.dataclass(frozen=True)
class TreeNode:
    """A node in a browsed tree; dirs may be expanded or left closed."""

    name: str
    is_dir: bool
    size: int | None
    children: tuple[TreeNode, ...]
    truncated: bool
    expanded: bool


@dataclasses.dataclass(frozen=True)
class PrefixPointer:
    """A directory-like location to list or browse."""

    kind: str  # "s3" or "local"
    bucket: str
    prefix: str
    label: str

    @property
    def uri(self) -> str:
        """Return the display URI for the prefix.

        Returns
        -------
        str
            URI shown to the user.
        """
        return f"s3://{self.bucket}/{self.prefix}" if self.kind == "s3" else self.prefix


@dataclasses.dataclass(frozen=True)
class FilePointer:
    """A single file to inspect or download."""

    uri: str
    label: str


def _has_data_ext(name: str) -> bool:
    """Return whether a name ends in a recognized data-file extension.

    Parameters
    ----------
    name : str
        File or object name to test.

    Returns
    -------
    bool
        True if the name has a data-file extension.
    """
    return name.lower().endswith(_DATA_EXTS)


def resolve_path(arg: str) -> PrefixPointer | FilePointer:
    """Classify a path argument as a prefix or a file.

    Uses only the argument's shape (endpoint, scheme, extension, trailing
    slash, local ``is_dir``); it makes no network calls.

    Parameters
    ----------
    arg : str
        Path to classify.

    Returns
    -------
    PrefixPointer or FilePointer
        Pointer to a directory-like prefix or to a single file.
    """
    if arg in ENDPOINTS:
        bucket, prefix = ENDPOINTS[arg]
        return PrefixPointer("s3", bucket, prefix, arg)

    for name, (bucket, root) in ENDPOINTS.items():
        if arg.startswith(name + "/"):
            sub = arg[len(name) + 1 :]
            key = root + sub
            if _has_data_ext(sub):
                return FilePointer(f"s3://{bucket}/{key}", arg)
            return PrefixPointer("s3", bucket, key.rstrip("/") + "/", arg.rstrip("/"))

    if arg.startswith("s3://"):
        bucket, _, key = arg[len("s3://") :].partition("/")
        if arg.endswith("/"):
            return PrefixPointer("s3", bucket, key, arg.rstrip("/"))
        if _has_data_ext(key):
            return FilePointer(arg, arg)
        return PrefixPointer("s3", bucket, key.rstrip("/") + "/", arg)

    if arg.startswith(("http://", "https://")):
        return FilePointer(arg, arg)

    path = Path(arg)
    if path.is_dir():
        return PrefixPointer("local", "", str(path) + "/", arg)
    return FilePointer(arg, arg)


class Lister(Protocol):
    """Lists the immediate children of a prefix."""

    def list_children(self, prefix: str, limit: int, name_filter: str | None) -> Listing:
        """Return capped immediate children of ``prefix``.

        Parameters
        ----------
        prefix : str
            Prefix to list.
        limit : int
            Maximum number of entries to return.
        name_filter : str or None
            Optional glob applied to entry names.

        Returns
        -------
        Listing
            Capped immediate children of the prefix.
        """
        ...


def make_client(aws_profile: str | None = None) -> Any:
    """Build an S3 client: anonymous by default, or a named profile.

    Parameters
    ----------
    aws_profile : str or None
        Named AWS profile to use. Anonymous access when None.

    Returns
    -------
    Any
        The boto3 S3 client.
    """
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    if aws_profile:
        return boto3.Session(profile_name=aws_profile).client("s3")
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


class S3Lister:
    """Lists an S3 bucket via delimited, page-bounded ``list_objects_v2``.

    Parameters
    ----------
    client : Any
        The boto3 S3 client to call.
    bucket : str
        Bucket to list.
    """

    def __init__(self, client: Any, bucket: str) -> None:
        """Hold the client and bucket."""
        self._client = client
        self._bucket = bucket

    def list_children(self, prefix: str, limit: int, name_filter: str | None) -> Listing:
        """Return immediate children of a prefix, capped and page-bounded.

        Parameters
        ----------
        prefix : str
            Prefix to list.
        limit : int
            Maximum number of entries to return.
        name_filter : str or None
            Optional glob applied to entry names.

        Returns
        -------
        Listing
            Capped immediate children of the prefix.
        """
        entries: list[Entry] = []
        token: str | None = None
        more = False
        for _ in range(_MAX_PAGES):
            kw: dict[str, Any] = {
                "Bucket": self._bucket,
                "Prefix": prefix,
                "Delimiter": "/",
                "MaxKeys": 1000,
            }
            if token:
                kw["ContinuationToken"] = token
            resp = self._client.list_objects_v2(**kw)
            for cp in resp.get("CommonPrefixes", []):
                name = cp["Prefix"][len(prefix) :].rstrip("/")
                if name and _match(name, name_filter):
                    entries.append(Entry(name, True, None))
            for obj in resp.get("Contents", []):
                name = obj["Key"][len(prefix) :]
                if name and "/" not in name and _match(name, name_filter):
                    entries.append(Entry(name, False, obj["Size"]))
            token = resp.get("NextContinuationToken")
            if not resp.get("IsTruncated"):
                break
            if len(entries) >= limit:
                more = True
                break
        else:
            more = True
        return Listing(prefix, tuple(entries[:limit]), more or len(entries) > limit)


class LocalLister:
    """Lists a local directory one level deep."""

    def list_children(self, prefix: str, limit: int, name_filter: str | None) -> Listing:
        """Return immediate children of a local directory, capped.

        Parameters
        ----------
        prefix : str
            Directory to list.
        limit : int
            Maximum number of entries to return.
        name_filter : str or None
            Optional glob applied to entry names.

        Returns
        -------
        Listing
            Capped immediate children of the directory.
        """
        entries: list[Entry] = []
        with os.scandir(prefix) as it:
            for de in sorted(it, key=lambda d: (not d.is_dir(), d.name)):
                if not _match(de.name, name_filter):
                    continue
                is_dir = de.is_dir()
                entries.append(Entry(de.name, is_dir, None if is_dir else de.stat().st_size))
        truncated = len(entries) > limit
        return Listing(prefix, tuple(entries[:limit]), truncated)


def make_lister(pointer: PrefixPointer, aws_profile: str | None = None) -> Lister:
    """Build the right lister for a prefix pointer.

    Parameters
    ----------
    pointer : PrefixPointer
        Location to list.
    aws_profile : str or None
        Named AWS profile passed to the S3 client. Anonymous when None.

    Returns
    -------
    Lister
        Lister matched to the pointer kind.
    """
    if pointer.kind == "local":
        return LocalLister()
    return S3Lister(make_client(aws_profile), pointer.bucket)


def _match(name: str, name_filter: str | None) -> bool:
    """Return whether a name passes an optional glob filter.

    Parameters
    ----------
    name : str
        Name to test.
    name_filter : str or None
        Glob pattern. Every name passes when None.

    Returns
    -------
    bool
        True if the name matches or no filter is set.
    """
    return name_filter is None or fnmatch.fnmatch(name, name_filter)


def list_children(
    lister: Lister, prefix: str, limit: int = 200, name_filter: str | None = None
) -> Listing:
    """List one level under a prefix.

    Parameters
    ----------
    lister : Lister
        Backend that performs the listing.
    prefix : str
        Prefix to list.
    limit : int
        Maximum number of entries to return.
    name_filter : str or None
        Optional glob applied to entry names.

    Returns
    -------
    Listing
        Capped immediate children of the prefix.
    """
    return lister.list_children(prefix, limit, name_filter)


def build_tree(
    lister: Lister,
    prefix: str,
    *,
    depth: int = 1,
    limit: int = 200,
    name_filter: str | None = None,
    on_list: Callable[[str], None] | None = None,
) -> TreeNode:
    """Build a browse tree, expanding directories up to ``depth`` levels.

    Parameters
    ----------
    lister : Lister
        Backend that performs the listing.
    prefix : str
        Prefix at the root of the tree.
    depth : int
        Levels to expand. ``1`` lists only immediate children; ``2`` also lists
        each child directory, and so on.
    limit : int
        Per-level cap on children.
    name_filter : str, optional
        Glob applied at the top level only.
    on_list : callable, optional
        Called with each prefix just before it is listed, for progress display.

    Returns
    -------
    TreeNode
        Root node of the browsed tree.
    """
    if on_list is not None:
        on_list(prefix)
    listing = lister.list_children(prefix, limit, name_filter)
    children: list[TreeNode] = []
    for entry in listing.entries:
        if entry.is_dir and depth > 1:
            sub = build_tree(
                lister,
                prefix + entry.name + "/",
                depth=depth - 1,
                limit=limit,
                name_filter=None,
                on_list=on_list,
            )
            children.append(
                TreeNode(entry.name, True, None, sub.children, sub.truncated, expanded=True)
            )
        else:
            children.append(
                TreeNode(entry.name, entry.is_dir, entry.size, (), False, expanded=False)
            )
    return TreeNode(prefix or "/", True, None, tuple(children), listing.truncated, expanded=True)
