"""The two axes of exploration: where a file lives, and what format it is.

A ``Source`` yields bytes; a ``FormatBackend`` consumes them. They meet at
``BinaryIO`` and nowhere else, so a source never learns the format and a backend
never learns the location. ``OpenFile`` is the single read API every backend
implements the same way.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
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
        """Open a seekable handle, tripping past ``max_bytes`` fetched (remote only)."""
        ...

    def peek(self, n: int) -> bytes:
        """Read the first ``n`` bytes, for format sniffing."""
        ...

    def materialize(self, approved: ApprovedRead) -> Path:
        """Download the whole file to local disk and return its path."""
        ...


@runtime_checkable
class OpenFile(Protocol):
    """The read API. Identical across backends."""

    def header(self) -> FileHeader:
        """Read format and root attributes only, without walking the structure."""
        ...

    def summary(self, *, storage: bool = False) -> FileSummary:
        """Describe the whole file. ``storage=True`` adds on-disk sizes (may be slow)."""
        ...

    def node(self, path: NodePath) -> NodeInfo | None:
        """Return one node, or ``None`` if the path does not exist."""
        ...

    def plan_read(self, path: NodePath, selection: Selection) -> ReadPlan:
        """Estimate the cost of a value read from metadata only."""
        ...

    def plan_stats(self, path: NodePath, spec: StatsSpec) -> ReadPlan:
        """Estimate the cost of a stats read from metadata only."""
        ...

    def head(self, approved: ApprovedRead, decode: Decode = "none") -> HeadResult:
        """Read a small slice of values, optionally decoding scale/offset."""
        ...

    def stats(self, approved: ApprovedRead, spec: StatsSpec) -> StatsResult:
        """Compute summary statistics, sampling within the spec's budget."""
        ...


@runtime_checkable
class FormatBackend(Protocol):
    """A file format. Knows nothing about the file's location."""

    format: Format

    @staticmethod
    def sniff(head: bytes) -> bool:
        """Return whether these leading bytes are this format."""
        ...

    def open(self, handle: BinaryIO, ref: SourceRef) -> AbstractContextManager[OpenFile]:
        """Open the file from a binary handle."""
        ...
