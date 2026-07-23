"""The two axes of exploration: where a file lives, and what format it is.

A ``Source`` yields bytes; a ``FormatBackend`` consumes them. They meet at
``BinaryIO`` and nowhere else, so a source never learns the format and a backend
never learns the location. ``OpenFile`` is the single read API every backend
implements the same way.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import BinaryIO, Protocol, runtime_checkable

from .budget import ApprovedRead, ReadPlan
from .model import (
    Decode,
    FileHeader,
    FileSummary,
    Format,
    HeadResult,
    NodeInfo,
    NodePath,
    SourceRef,
    StatsResult,
    StatsSpec,
)
from .selection import Selection


@runtime_checkable
class Source(Protocol):
    """A file location. Knows nothing about the file's format."""

    ref: SourceRef

    def open_binary(
        self, max_bytes: int | None = None, block_size: int | None = None
    ) -> AbstractContextManager[BinaryIO]:
        """Open a seekable handle, tripping past ``max_bytes`` fetched (remote only).

        Parameters
        ----------
        max_bytes : int, optional
            Stop the read once this many bytes have been fetched.
        block_size : int, optional
            Alignment of cached blocks for remote reads.
        """
        ...

    def peek(self, n: int) -> bytes:
        """Read the first ``n`` bytes, for format sniffing.

        Parameters
        ----------
        n : int
            Number of bytes to read.
        """
        ...


@runtime_checkable
class OpenFile(Protocol):
    """The read API. Identical across backends."""

    def header(self) -> FileHeader:
        """Read format and root attributes only, without walking the structure."""
        ...

    def summary(self, *, storage: bool = False) -> FileSummary:
        """Describe the whole file. ``storage=True`` adds on-disk sizes (may be slow).

        Parameters
        ----------
        storage : bool, optional
            Include on-disk sizes for each array.
        """
        ...

    def node(self, path: NodePath) -> NodeInfo | None:
        """Return one node, or ``None`` if the path does not exist.

        Parameters
        ----------
        path : NodePath
            Path of the node to look up.
        """
        ...

    def plan_read(self, path: NodePath, selection: Selection) -> ReadPlan:
        """Estimate the cost of a value read from metadata only.

        Parameters
        ----------
        path : NodePath
            Path of the array to read.
        selection : Selection
            The requested read shape.
        """
        ...

    def plan_stats(self, path: NodePath, spec: StatsSpec) -> ReadPlan:
        """Estimate the cost of a stats read from metadata only.

        Parameters
        ----------
        path : NodePath
            Path of the array to read.
        spec : StatsSpec
            Limits for the stats read.
        """
        ...

    def head(self, approved: ApprovedRead, decode: Decode = "none") -> HeadResult:
        """Read a small slice of values, optionally decoding scale/offset.

        Parameters
        ----------
        approved : ApprovedRead
            The approved plan to execute.
        decode : Decode, optional
            How to decode raw values.
        """
        ...

    def stats(self, approved: ApprovedRead, spec: StatsSpec) -> StatsResult:
        """Compute summary statistics, sampling within the spec's budget.

        Parameters
        ----------
        approved : ApprovedRead
            The approved plan to execute.
        spec : StatsSpec
            Limits for the stats read.
        """
        ...


@runtime_checkable
class FormatBackend(Protocol):
    """A file format. Knows nothing about the file's location."""

    format: Format

    def open(self, handle: BinaryIO, ref: SourceRef) -> AbstractContextManager[OpenFile]:
        """Open the file from a binary handle.

        Parameters
        ----------
        handle : BinaryIO
            Seekable binary stream positioned at the start of the file.
        ref : SourceRef
            Where the file was opened from.
        """
        ...
