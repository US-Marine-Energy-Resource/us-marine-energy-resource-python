"""Inspect h5 / nc / parquet files on local disk, S3, or over HTTP.

The public entry point is :func:`open_file`, a context manager yielding an
``OpenFile`` with a uniform read API. Importing this package pulls in only the
stdlib data model; h5py and pyarrow load when a file of their format is opened.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from .budget import ApprovedRead, NeedsConfirm, ReadPlan, Refusal, TransferPolicy
from .errors import (
    DependencyError,
    ExploreError,
    NodeNotFoundError,
    SourceError,
    TransferBudgetExceededError,
    UnknownFormatError,
    UnsupportedFormatError,
)
from .model import (
    ArrayInfo,
    AttrValue,
    ByteSize,
    FileSummary,
    Format,
    HeadResult,
    NodeInfo,
    NodePath,
    SourceRef,
    StatsResult,
    StatsSpec,
    StorageInfo,
)
from .protocols import OpenFile, Source
from .selection import FirstN, Index, Selection

__all__ = [
    "ApprovedRead",
    "ArrayInfo",
    "AttrValue",
    "ByteSize",
    "DependencyError",
    "ExploreError",
    "FileSummary",
    "FirstN",
    "Format",
    "HeadResult",
    "Index",
    "NeedsConfirm",
    "NodeInfo",
    "NodeNotFoundError",
    "NodePath",
    "OpenFile",
    "ReadPlan",
    "Refusal",
    "Selection",
    "Source",
    "SourceError",
    "SourceRef",
    "StatsResult",
    "StatsSpec",
    "StorageInfo",
    "TransferBudgetExceededError",
    "TransferPolicy",
    "UnknownFormatError",
    "UnsupportedFormatError",
    "open_file",
]


# Block size for metadata-only reads of remote HDF5. HDF5 metadata is small
# and scattered, so large blocks fetch far more than they use. Parquet keeps
# the default large blocks because its footer is contiguous.
_METADATA_BLOCK = 64 * 1024


@contextmanager
def _open_with_reader(
    uri: str,
    *,
    policy: TransferPolicy,
    aws_profile: str | None,
    metadata_only: bool,
) -> Iterator[tuple[OpenFile, object, SourceRef]]:
    """Open a file, yielding the raw reader and source ref too.

    The CLI uses this seam to reach the reader's byte counter and the source
    ref for its large file heads-up. The raw reader is ``None`` for local
    files, which move no bytes.

    Parameters
    ----------
    uri : str
        A local path, ``s3://bucket/key``, or ``https://...`` URL.
    policy : TransferPolicy
        Volume limits. Its ``max_transfer`` becomes the runtime fuse on
        remote reads.
    aws_profile : str or None
        AWS profile for signed S3 access. Anonymous when ``None``.
    metadata_only : bool
        Tune remote reads for a metadata walk instead of payload reads.

    Yields
    ------
    tuple of (OpenFile, object, SourceRef)
        The open file, the raw block-cached reader or ``None``, and the
        resolved source ref.
    """
    from .formats import get_backend
    from .sniff import sniff_format
    from .sources import resolve_source

    source = resolve_source(uri, aws_profile=aws_profile)
    max_bytes = None if source.ref.scheme == "file" else policy.max_transfer.bytes
    fmt = sniff_format(source.peek(16))
    backend = get_backend(fmt)
    block = _METADATA_BLOCK if metadata_only and fmt == "hdf5" else None
    with (
        source.open_binary(max_bytes, block_size=block) as handle,
        backend.open(handle, source.ref) as opened,
    ):
        yield opened, getattr(handle, "raw", None), source.ref


@contextmanager
def open_file(
    uri: str,
    *,
    policy: TransferPolicy | None = None,
    aws_profile: str | None = None,
    metadata_only: bool = False,
) -> Iterator[OpenFile]:
    """Open a file for exploration.

    Parameters
    ----------
    uri : str
        A local path, ``s3://bucket/key``, or ``https://...`` URL.
    policy : TransferPolicy, optional
        Volume limits. Its ``max_transfer`` becomes the runtime fuse on remote
        reads. Defaults to :class:`TransferPolicy`.
    aws_profile : str, optional
        AWS profile for signed S3 access. Anonymous when omitted.
    metadata_only : bool
        Tune remote reads for a metadata walk instead of payload reads. For
        remote HDF5 this fetches small aligned blocks so scattered object
        headers do not drag whole megabytes each.

    Yields
    ------
    OpenFile
        A handle with the uniform read API.
    """
    with _open_with_reader(
        uri,
        policy=policy or TransferPolicy(),
        aws_profile=aws_profile,
        metadata_only=metadata_only,
    ) as (opened, _reader, _ref):
        yield opened
