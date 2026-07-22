"""Shared data model for file exploration.

Every backend normalises its format into these types, so renderers and JSON
output work the same for HDF5 and parquet. Stdlib only: importing this module
must not pull in h5py, pyarrow, or numpy.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

Format = Literal["hdf5", "parquet"]
NodeKind = Literal["group", "array", "column"]
Scheme = Literal["file", "s3", "https"]
Decode = Literal["none", "cf", "rex"]

# JSON-safe by construction. Backends convert bytes, numpy scalars, and object
# references to these types at the boundary, so json.dumps needs no encoder.
AttrValue = str | int | float | bool | None | list["AttrValue"] | dict[str, "AttrValue"]

_MB = 1024 * 1024


@dataclasses.dataclass(frozen=True)
class ByteSize:
    """A non-negative byte count, readable as bytes or megabytes."""

    bytes: int

    def __post_init__(self) -> None:
        """Reject negative counts."""
        if self.bytes < 0:
            raise ValueError(f"negative size: {self.bytes}")

    @property
    def mb(self) -> float:
        """Size in megabytes."""
        return self.bytes / _MB

    def __str__(self) -> str:
        """Human-readable size, e.g. ``4.2 GB``."""
        n = float(self.bytes)
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if n < 1024 or unit == "TB":
                return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
            n /= 1024
        return f"{n:.1f} TB"


@dataclasses.dataclass(frozen=True)
class NodePath:
    """A ``/``-rooted path to a group, array, or column inside a file."""

    value: str

    def __post_init__(self) -> None:
        """Reject paths that are not rooted at ``/``."""
        if not self.value.startswith("/"):
            raise ValueError(f"path must start with '/': {self.value!r}")

    @property
    def name(self) -> str:
        """Final path segment (``""`` for the root)."""
        return self.value.rstrip("/").rsplit("/", 1)[-1]

    @property
    def depth(self) -> int:
        """Number of segments below the root."""
        return self.value.strip("/").count("/") + 1 if self.value.strip("/") else 0

    def __str__(self) -> str:
        """Return the raw path string."""
        return self.value


@dataclasses.dataclass(frozen=True)
class SourceRef:
    """Where a file was opened from."""

    uri: str
    scheme: Scheme
    display: str
    size: ByteSize | None


@dataclasses.dataclass(frozen=True)
class StorageInfo:
    """How an array sits on disk."""

    chunks: tuple[int, ...] | None
    compression: str | None
    filters: tuple[str, ...]
    stored: ByteSize | None
    logical: ByteSize | None

    @property
    def compression_ratio(self) -> float | None:
        """Logical bytes divided by stored bytes, or ``None`` if unknown."""
        if self.stored is None or self.logical is None or self.stored.bytes == 0:
            return None
        return round(self.logical.bytes / self.stored.bytes, 2)


@dataclasses.dataclass(frozen=True)
class ArrayInfo:
    """Shape, type, and storage of an array or column."""

    shape: tuple[int, ...]
    dtype: str
    dim_names: tuple[str | None, ...]
    fill_value: AttrValue
    storage: StorageInfo


@dataclasses.dataclass(frozen=True)
class NodeInfo:
    """One group, array, or column in a file."""

    path: NodePath
    kind: NodeKind
    attrs: dict[str, AttrValue]
    array: ArrayInfo | None
    n_children: int

    @property
    def name(self) -> str:
        """Final segment of the node path."""
        return self.path.name


@dataclasses.dataclass(frozen=True)
class FileHeader:
    """Cheap descriptive metadata: format and root attrs, without walking nodes."""

    source: SourceRef
    format: Format
    format_detail: str
    root_attrs: dict[str, AttrValue]


@dataclasses.dataclass(frozen=True)
class FileSummary:
    """Structure of a whole file: root attrs plus a flat, pre-order node list."""

    source: SourceRef
    format: Format
    format_detail: str
    root_attrs: dict[str, AttrValue]
    nodes: tuple[NodeInfo, ...]
    n_arrays: int
    n_groups: int
    warnings: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class HeadResult:
    """A small slice of values read from one array."""

    path: NodePath
    shape: tuple[int, ...]
    dtype: str
    selection: str
    values: AttrValue
    decode: Decode
    notes: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class StatsSpec:
    """Limits for a stats read."""

    max_elements: int = 1_000_000
    nan_policy: Literal["omit", "propagate"] = "omit"
    exact: bool = False


@dataclasses.dataclass(frozen=True)
class StatsResult:
    """Summary statistics for one array, with the sampling that produced them."""

    path: NodePath
    count: int
    n_nan: int
    mean: float | None
    std: float | None
    min: float | None
    max: float | None
    sampled: bool
    sample_fraction: float
    sample_method: Literal["full", "chunk-strided"]
