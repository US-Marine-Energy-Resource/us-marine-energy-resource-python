"""Parquet backend.

A parquet file is presented as a root group with one column node per column, so
it fits the same tree and read API as HDF5. Structure and footer statistics come
from the file footer with no row reads.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, BinaryIO

from ..budget import ApprovedRead, ReadPlan
from ..errors import NodeNotFoundError
from ..lazy import lazy_import
from ..model import (
    ArrayInfo,
    AttrValue,
    ByteSize,
    Decode,
    FileHeader,
    FileSummary,
    HeadResult,
    NodeInfo,
    NodePath,
    SourceRef,
    StatsResult,
    StatsSpec,
    StorageInfo,
)
from ..selection import FirstN, Selection, resolve

_ROOT = NodePath("/")


class ParquetBackend:
    """Open parquet files."""

    format = "parquet"

    @staticmethod
    def sniff(head: bytes) -> bool:
        """Return whether the leading bytes are the parquet magic."""
        return head.startswith(b"PAR1")

    @contextmanager
    def open(self, handle: BinaryIO, ref: SourceRef) -> Iterator[ParquetOpenFile]:
        """Open a parquet file from a binary handle."""
        yield ParquetOpenFile(handle, ref)


class ParquetOpenFile:
    """Read structure and values from one parquet file."""

    def __init__(self, handle: BinaryIO, ref: SourceRef) -> None:
        """Read the footer and column metadata."""
        pq = lazy_import("pyarrow.parquet", "reading parquet files")
        self._ref = ref
        self._pf = pq.ParquetFile(handle)
        self._meta = self._pf.metadata
        self._schema = self._pf.schema_arrow
        self._num_rows = self._meta.num_rows
        self._names = list(self._schema.names)

    def _column_storage(self, col_idx: int) -> tuple[StorageInfo, str | None]:
        """Sum compressed and uncompressed sizes for a column across row groups."""
        stored = 0
        logical = 0
        compression: str | None = None
        for rg in range(self._meta.num_row_groups):
            chunk = self._meta.row_group(rg).column(col_idx)
            stored += chunk.total_compressed_size
            logical += chunk.total_uncompressed_size
            compression = chunk.compression
        comp = None if compression in (None, "UNCOMPRESSED") else str(compression)
        return (
            StorageInfo(
                chunks=None,
                compression=comp,
                filters=(),
                stored=ByteSize(stored),
                logical=ByteSize(logical),
            ),
            comp,
        )

    def _array_info(self, col_idx: int) -> ArrayInfo:
        """Build the array description for a column."""
        field = self._schema.field(col_idx)
        storage, _ = self._column_storage(col_idx)
        return ArrayInfo(
            shape=(self._num_rows,),
            dtype=str(field.type),
            dim_names=(None,),
            fill_value=None,
            storage=storage,
        )

    def _detail(self) -> str:
        """Format string, e.g. ``Parquet 2.6, 4 row groups``."""
        return f"Parquet {self._meta.format_version}, {self._meta.num_row_groups} row groups"

    def header(self) -> FileHeader:
        """Read file-level metadata from the footer; no per-column work."""
        from ...analysis.preprocessing import _extract_parquet_footer_info

        footer = _extract_parquet_footer_info(self._meta)
        return FileHeader(
            source=self._ref,
            format="parquet",
            format_detail=self._detail(),
            root_attrs=dict(footer["file_meta"]),
        )

    def summary(self, *, storage: bool = False) -> FileSummary:
        """Describe the file: root attrs plus one node per column.

        Parquet sizes come from the footer, so ``storage`` costs nothing extra.
        """
        from ...analysis.preprocessing import _extract_parquet_footer_info

        footer = _extract_parquet_footer_info(self._meta)
        root_attrs: dict[str, AttrValue] = dict(footer["file_meta"])
        root = NodeInfo(
            path=_ROOT, kind="group", attrs=root_attrs, array=None, n_children=len(self._names)
        )
        nodes = [root]
        for i, name in enumerate(self._names):
            col_meta = footer["var_meta"].get(name, {})
            nodes.append(
                NodeInfo(
                    path=NodePath(f"/{name}"),
                    kind="column",
                    attrs=dict(col_meta),
                    array=self._array_info(i),
                    n_children=0,
                )
            )
        detail = self._detail()
        return FileSummary(
            source=self._ref,
            format="parquet",
            format_detail=detail,
            root_attrs=root_attrs,
            nodes=tuple(nodes),
            n_arrays=len(self._names),
            n_groups=1,
            warnings=(),
        )

    def node(self, path: NodePath) -> NodeInfo | None:
        """Return the root group or one column node."""
        if path.value == "/":
            return NodeInfo(
                path=_ROOT, kind="group", attrs={}, array=None, n_children=len(self._names)
            )
        name = path.name
        if name not in self._names:
            return None
        idx = self._names.index(name)
        return NodeInfo(
            path=path, kind="column", attrs={}, array=self._array_info(idx), n_children=0
        )

    def _require_column(self, path: NodePath) -> int:
        """Return a column index or raise if the path is not a column."""
        name = path.name
        if name not in self._names:
            raise NodeNotFoundError(f"no column at {path}")
        return self._names.index(name)

    def _row_groups_for(self, start: int, stop: int) -> list[int]:
        """Return the row groups overlapping the half-open row range."""
        groups = []
        row0 = 0
        for rg in range(self._meta.num_row_groups):
            n = self._meta.row_group(rg).num_rows
            if row0 < stop and row0 + n > start:
                groups.append(rg)
            row0 += n
        return groups

    def plan_read(self, path: NodePath, selection: Selection) -> ReadPlan:
        """Estimate a read from footer metadata only."""
        idx = self._require_column(path)
        node = self.node(path)
        assert node is not None and node.array is not None
        resolved = resolve(selection, node.array)
        n = resolved.n_elements
        itemsize = _itemsize(node.array.dtype)
        logical = ByteSize(n * itemsize)
        row_slice = resolved.slices[0]
        groups = self._row_groups_for(row_slice.start, row_slice.stop)
        transferred = sum(
            self._meta.row_group(rg).column(idx).total_compressed_size for rg in groups
        )
        return ReadPlan(
            node=node,
            selection=resolved,
            logical=logical,
            transferred=ByteSize(transferred),
            n_chunks=len(groups),
        )

    def plan_stats(self, path: NodePath, spec: StatsSpec) -> ReadPlan:
        """Estimate a stats read: all rows if exact, else the first max_elements."""
        idx = self._require_column(path)
        node = self.node(path)
        assert node is not None and node.array is not None
        itemsize = _itemsize(node.array.dtype)
        rows = self._num_rows if spec.exact else min(spec.max_elements, self._num_rows)
        resolved = resolve(FirstN(max(1, rows)), node.array)
        groups = self._row_groups_for(0, rows)
        transferred = sum(
            self._meta.row_group(rg).column(idx).total_compressed_size for rg in groups
        )
        return ReadPlan(
            node=node,
            selection=resolved,
            logical=ByteSize(rows * itemsize),
            transferred=ByteSize(transferred),
            n_chunks=len(groups),
        )

    def _read_column_slice(self, idx: int, row_slice: slice) -> Any:
        """Read a column over the row groups covering a slice, then apply the slice."""
        import pyarrow as pa  # noqa: F401 - ensures pyarrow present for compute

        name = self._names[idx]
        groups = self._row_groups_for(row_slice.start, row_slice.stop)
        if not groups:
            table = self._pf.read_row_groups([0], columns=[name])
            return table.column(0).slice(0, 0)
        table = self._pf.read_row_groups(groups, columns=[name])
        offset = sum(self._meta.row_group(rg).num_rows for rg in range(groups[0]))
        local_start = row_slice.start - offset
        local_stop = row_slice.stop - offset
        col = table.column(0)
        step = row_slice.step or 1
        sliced = col.slice(local_start, local_stop - local_start)
        return sliced[::step] if step != 1 else sliced

    def head(self, approved: ApprovedRead, decode: Decode = "none") -> HeadResult:
        """Read the approved slice of one column. Parquet has no scale/offset to decode."""
        node = approved.plan.node
        idx = self._require_column(node.path)
        resolved = approved.plan.selection
        arr = self._read_column_slice(idx, resolved.slices[0])
        assert node.array is not None
        return HeadResult(
            path=node.path,
            shape=(len(arr),),
            dtype=node.array.dtype,
            selection=resolved.text,
            values=arr.to_pylist(),
            decode="none",
            notes=(),
        )

    def stats(self, approved: ApprovedRead, spec: StatsSpec) -> StatsResult:
        """Compute statistics over the first rows of one column, per the spec."""
        np = lazy_import("numpy", "computing statistics")
        node = approved.plan.node
        idx = self._require_column(node.path)
        rows = self._num_rows if spec.exact else min(spec.max_elements, self._num_rows)
        arr = self._read_column_slice(idx, slice(0, rows, 1))
        values = np.asarray(arr.to_pylist())
        return _summarize(np, values, node.path, self._num_rows, spec)


def _itemsize(dtype: str) -> int:
    """Best-effort element size in bytes from a dtype string."""
    for token, size in (("64", 8), ("32", 4), ("16", 2), ("8", 1)):
        if token in dtype:
            return size
    return 8


def _summarize(np: Any, values: Any, path: NodePath, total: int, spec: StatsSpec) -> StatsResult:
    """Reduce a 1-D array to a StatsResult, reporting how much was covered."""
    read = int(values.size)
    numeric = values.dtype.kind in "iufc"
    if not numeric or read == 0:
        return StatsResult(
            path=path,
            count=read,
            n_nan=0,
            mean=None,
            std=None,
            min=None,
            max=None,
            sampled=read < total,
            sample_fraction=(read / total) if total else 1.0,
            sample_method="full" if read >= total else "chunk-strided",
        )
    flat = values.astype("float64").ravel()
    n_nan = int(np.isnan(flat).sum())
    clean = flat[~np.isnan(flat)] if spec.nan_policy == "omit" else flat
    return StatsResult(
        path=path,
        count=read,
        n_nan=n_nan,
        mean=float(np.mean(clean)) if clean.size else None,
        std=float(np.std(clean)) if clean.size else None,
        min=float(np.min(clean)) if clean.size else None,
        max=float(np.max(clean)) if clean.size else None,
        sampled=read < total,
        sample_fraction=(read / total) if total else 1.0,
        sample_method="full" if read >= total else "chunk-strided",
    )
