"""HDF5 backend, covering ``.h5`` and netCDF-4 (both are HDF5 on disk).

Structure comes from h5py metadata with no payload reads. Value reads are
chunk-aware: estimates count whole chunks, and stats sample strided chunks so a
500 GB array never loads in full. Dimension names come from HDF5 dimension
scales, which is most of what xarray would show.
"""

from __future__ import annotations

import math
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

_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"


class Hdf5Backend:
    """Open HDF5 and netCDF-4 files."""

    format = "hdf5"

    @staticmethod
    def sniff(head: bytes) -> bool:
        """Return whether the leading bytes are the HDF5 magic."""
        return head.startswith(_HDF5_MAGIC)

    @contextmanager
    def open(self, handle: BinaryIO, ref: SourceRef) -> Iterator[Hdf5OpenFile]:
        """Open an HDF5 file from a binary handle.

        Parameters
        ----------
        handle : BinaryIO
            Open binary handle to the file.
        ref : SourceRef
            Reference to where the file came from.

        Yields
        ------
        Hdf5OpenFile
            Reader for the opened file.
        """
        h5py = lazy_import("h5py", "reading HDF5/netCDF-4 files")
        f = h5py.File(handle, "r")
        try:
            yield Hdf5OpenFile(f, ref, h5py)
        finally:
            f.close()


class Hdf5OpenFile:
    """Read structure and values from one HDF5 file.

    Parameters
    ----------
    f : Any
        Open h5py file object.
    ref : SourceRef
        Reference to where the file came from.
    h5py : Any
        Imported h5py module.
    """

    def __init__(self, f: Any, ref: SourceRef, h5py: Any) -> None:
        """Hold the open file and the h5py and numpy modules.

        Parameters
        ----------
        f : Any
            Open h5py file object.
        ref : SourceRef
            Reference to where the file came from.
        h5py : Any
            Imported h5py module.
        """
        self._f = f
        self._ref = ref
        self._h5py = h5py
        self._np = lazy_import("numpy", "reading HDF5 values")

    def _attr(self, val: Any) -> AttrValue:
        """Convert an h5py attribute value to a JSON-safe form.

        Parameters
        ----------
        val : Any
            Attribute value as h5py returns it.

        Returns
        -------
        AttrValue
            JSON-safe value.
        """
        np = self._np
        if isinstance(val, bytes):
            return val.decode("utf-8", "replace")
        if isinstance(val, np.ndarray):
            return [self._attr(v) for v in val.tolist()]
        if isinstance(val, np.generic):
            return self._attr(val.item())
        if isinstance(val, str | int | float | bool) or val is None:
            return val
        return str(val)

    def _attrs(self, obj: Any) -> dict[str, AttrValue]:
        """Convert all attributes of a node.

        Parameters
        ----------
        obj : Any
            Group or dataset with attributes.

        Returns
        -------
        dict[str, AttrValue]
            Attribute names mapped to JSON-safe values.
        """
        return {k: self._attr(v) for k, v in obj.attrs.items()}

    def _filters(self, dset: Any) -> tuple[str, ...]:
        """List filter names from the dataset creation property list.

        Parameters
        ----------
        dset : Any
            Open h5py dataset.

        Returns
        -------
        tuple[str, ...]
            Filter names, empty if none or unreadable.
        """
        names: list[str] = []
        try:
            dcpl = dset.id.get_create_plist()
            for i in range(dcpl.get_nfilters()):
                info = dcpl.get_filter(i)
                names.append(info[3].decode() if isinstance(info[3], bytes) else str(info[3]))
        except (KeyError, ValueError, RuntimeError):
            return ()
        return tuple(names)

    def _dim_names(self, dset: Any) -> tuple[str | None, ...]:
        """Read per-axis dimension names from attached dimension scales.

        Parameters
        ----------
        dset : Any
            Open h5py dataset.

        Returns
        -------
        tuple[str or None, ...]
            One name per axis, with ``None`` where no scale is attached.
        """
        names: list[str | None] = []
        try:
            for dim in dset.dims:
                if dim.label:
                    names.append(dim.label)
                elif len(dim):
                    names.append(dim[0].name.rsplit("/", 1)[-1])
                else:
                    names.append(None)
        except (RuntimeError, ValueError):
            return tuple(None for _ in dset.shape)
        return tuple(names)

    def _storage(self, dset: Any, *, sizes: bool) -> StorageInfo:
        """Describe on-disk storage of a dataset.

        ``get_storage_size`` walks the chunk index, which is expensive for a
        large chunked dataset read over the network, so it runs only when
        ``sizes`` is set.

        Parameters
        ----------
        dset : Any
            Open h5py dataset.
        sizes : bool
            Whether to read the stored size from the chunk index.

        Returns
        -------
        StorageInfo
            Chunking, compression, and size details.
        """
        stored = ByteSize(dset.id.get_storage_size()) if sizes else None
        return StorageInfo(
            chunks=tuple(dset.chunks) if dset.chunks else None,
            compression=dset.compression,
            filters=self._filters(dset),
            stored=stored,
            logical=ByteSize(dset.nbytes),
        )

    def _array(self, dset: Any, *, sizes: bool = False) -> ArrayInfo:
        """Describe a dataset as an array.

        Parameters
        ----------
        dset : Any
            Open h5py dataset.
        sizes : bool, optional
            Whether to read the stored size from the chunk index.

        Returns
        -------
        ArrayInfo
            Shape, dtype, dimension names, fill value, and storage.
        """
        return ArrayInfo(
            shape=tuple(dset.shape),
            dtype=str(dset.dtype),
            dim_names=self._dim_names(dset),
            fill_value=self._attr(dset.fillvalue) if dset.shape else None,
            storage=self._storage(dset, sizes=sizes),
        )

    def _format_detail(self) -> str:
        """Format string, e.g. ``netCDF-4, libhdf5 1.14.6``.

        Returns
        -------
        str
            Format kind and the linked libhdf5 version.
        """
        kind = "netCDF-4" if "_NCProperties" in self._f.attrs else "HDF5"
        return f"{kind}, libhdf5 {self._h5py.version.hdf5_version}"

    def header(self) -> FileHeader:
        """Read root attributes and format only; no structure walk.

        Returns
        -------
        FileHeader
            Source, format, and root attributes.
        """
        return FileHeader(
            source=self._ref,
            format="hdf5",
            format_detail=self._format_detail(),
            root_attrs=self._attrs(self._f),
        )

    def summary(self, *, storage: bool = False) -> FileSummary:
        """Walk the file and describe every group and dataset.

        Parameters
        ----------
        storage : bool, optional
            Whether to read stored sizes for each dataset.

        Returns
        -------
        FileSummary
            All nodes plus file level details.
        """
        nodes: list[NodeInfo] = [
            NodeInfo(
                path=NodePath("/"),
                kind="group",
                attrs=self._attrs(self._f),
                array=None,
                n_children=len(self._f.keys()),
            )
        ]
        counts = {"arrays": 0, "groups": 1}

        def visit(name: str, obj: Any) -> None:
            """Record one visited node.

            Parameters
            ----------
            name : str
                Path of the node relative to the root.
            obj : Any
                Visited group or dataset.
            """
            path = NodePath("/" + name)
            if isinstance(obj, self._h5py.Dataset):
                counts["arrays"] += 1
                nodes.append(
                    NodeInfo(
                        path=path,
                        kind="array",
                        attrs=self._attrs(obj),
                        array=self._array(obj, sizes=storage),
                        n_children=0,
                    )
                )
            else:
                counts["groups"] += 1
                nodes.append(
                    NodeInfo(
                        path=path,
                        kind="group",
                        attrs=self._attrs(obj),
                        array=None,
                        n_children=len(obj.keys()),
                    )
                )

        self._f.visititems(visit)
        detail = self._format_detail()
        return FileSummary(
            source=self._ref,
            format="hdf5",
            format_detail=detail,
            root_attrs=self._attrs(self._f),
            nodes=tuple(nodes),
            n_arrays=counts["arrays"],
            n_groups=counts["groups"],
            warnings=(),
        )

    def node(self, path: NodePath) -> NodeInfo | None:
        """Return the node at a path, or ``None`` if it does not exist.

        Parameters
        ----------
        path : NodePath
            Path of the node to look up.

        Returns
        -------
        NodeInfo or None
            Description of the node, or ``None`` if the path is missing.
        """
        key = "/" if path.value == "/" else path.value
        obj = self._f.get(key)
        if obj is None:
            return None
        if isinstance(obj, self._h5py.Dataset):
            return NodeInfo(
                path=path,
                kind="array",
                attrs=self._attrs(obj),
                array=self._array(obj),
                n_children=0,
            )
        return NodeInfo(
            path=path,
            kind="group",
            attrs=self._attrs(obj),
            array=None,
            n_children=len(obj.keys()),
        )

    def _require_dataset(self, path: NodePath) -> Any:
        """Return a dataset or raise if the path is missing or a group.

        Parameters
        ----------
        path : NodePath
            Path that must name a dataset.

        Returns
        -------
        Any
            Open h5py dataset.

        Raises
        ------
        NodeNotFoundError
            If the path is missing or names a group.
        """
        obj = self._f.get(path.value)
        if obj is None:
            raise NodeNotFoundError(f"no node at {path}")
        if not isinstance(obj, self._h5py.Dataset):
            raise NodeNotFoundError(f"{path} is a group, not an array")
        return obj

    def _chunk_transfer(
        self, dset: Any, slices: tuple[slice, ...], logical: int
    ) -> tuple[int, int]:
        """Estimate transferred bytes and chunk count for reading slices.

        Parameters
        ----------
        dset : Any
            Open h5py dataset.
        slices : tuple[slice, ...]
            Resolved per-axis slices to read.
        logical : int
            Logical size of the selection in bytes.

        Returns
        -------
        tuple[int, int]
            Estimated transferred bytes and number of touched chunks.
        """
        if not dset.chunks or dset.id.get_storage_size() == 0:
            return logical, 1
        touched = 1
        total = 1
        for sl, dim, chunk in zip(slices, dset.shape, dset.chunks, strict=False):
            first = sl.start // chunk
            last = max(sl.start, sl.stop - 1) // chunk
            touched *= last - first + 1
            total *= math.ceil(dim / chunk)
        stored = dset.id.get_storage_size()
        transferred = int(stored * touched / total) if total else stored
        return transferred, touched

    def plan_read(self, path: NodePath, selection: Selection) -> ReadPlan:
        """Estimate a value read from chunk geometry only.

        Parameters
        ----------
        path : NodePath
            Path of the dataset to read.
        selection : Selection
            Requested part of the array.

        Returns
        -------
        ReadPlan
            Estimated cost of the read.
        """
        dset = self._require_dataset(path)
        node = self.node(path)
        assert node is not None and node.array is not None
        resolved = resolve(selection, node.array)
        logical = resolved.n_elements * dset.dtype.itemsize
        transferred, n_chunks = self._chunk_transfer(dset, resolved.slices, logical)
        return ReadPlan(
            node=node,
            selection=resolved,
            logical=ByteSize(logical),
            transferred=ByteSize(transferred),
            n_chunks=n_chunks,
        )

    def _sample_rows(self, dset: Any, spec: StatsSpec) -> int:
        """Return how many axis-0 rows a sampled stats read will cover.

        Parameters
        ----------
        dset : Any
            Open h5py dataset.
        spec : StatsSpec
            Stats request with the element budget.

        Returns
        -------
        int
            Number of axis-0 rows to sample.
        """
        shape = dset.shape
        rows_per = math.prod(shape[1:]) if len(shape) > 1 else 1
        max_rows = max(1, spec.max_elements // max(1, rows_per))
        return min(shape[0], max_rows)

    def plan_stats(self, path: NodePath, spec: StatsSpec) -> ReadPlan:
        """Estimate a stats read: all chunks if exact, else strided chunks.

        Parameters
        ----------
        path : NodePath
            Path of the dataset to summarize.
        spec : StatsSpec
            Stats request with the element budget.

        Returns
        -------
        ReadPlan
            Estimated cost of the read.
        """
        dset = self._require_dataset(path)
        node = self.node(path)
        assert node is not None and node.array is not None
        shape = dset.shape
        total = math.prod(shape) if shape else 1
        itemsize = dset.dtype.itemsize
        if spec.exact or total <= spec.max_elements:
            resolved = resolve(FirstN(max(1, shape[0])) if shape else FirstN(1), node.array)
            transferred = dset.id.get_storage_size() or total * itemsize
            n_chunks = _n_chunks(dset)
            logical = total * itemsize
        else:
            rows = self._sample_rows(dset, spec)
            resolved = resolve(FirstN(rows), node.array)
            logical = rows * (math.prod(shape[1:]) if len(shape) > 1 else 1) * itemsize
            stored = dset.id.get_storage_size()
            frac = rows / shape[0] if shape[0] else 1.0
            transferred = int(stored * frac) if stored else logical
            n_chunks = max(1, math.ceil(rows / (dset.chunks[0] if dset.chunks else rows)))
        return ReadPlan(
            node=node,
            selection=resolved,
            logical=ByteSize(logical),
            transferred=ByteSize(transferred),
            n_chunks=n_chunks,
        )

    def head(self, approved: ApprovedRead, decode: Decode = "none") -> HeadResult:
        """Read the approved slice, optionally applying scale/offset.

        Parameters
        ----------
        approved : ApprovedRead
            Read plan that passed the budget check.
        decode : Decode, optional
            Scaling convention to apply.

        Returns
        -------
        HeadResult
            Values plus shape, dtype, and decode notes.
        """
        node = approved.plan.node
        dset = self._require_dataset(node.path)
        resolved = approved.plan.selection
        data = dset[resolved.slices]
        values, applied, notes = self._decode(dset, data, decode)
        return HeadResult(
            path=node.path,
            shape=tuple(getattr(values, "shape", ())),
            dtype=str(dset.dtype),
            selection=resolved.text,
            values=_to_jsonable(values),
            decode=applied,
            notes=notes,
        )

    def _decode(self, dset: Any, data: Any, decode: Decode) -> tuple[Any, Decode, tuple[str, ...]]:
        """Apply CF or rex scaling, or report unapplied scale attributes.

        Parameters
        ----------
        dset : Any
            Open h5py dataset with the scale attributes.
        data : Any
            Raw values read from the dataset.
        decode : Decode
            Scaling convention to apply.

        Returns
        -------
        tuple[Any, Decode, tuple[str, ...]]
            Values, the convention applied, and any notes.
        """
        sf = dset.attrs.get("scale_factor")
        offset = dset.attrs.get("add_offset")
        add = float(offset) if offset is not None else 0.0
        if decode == "cf" and sf is not None:
            return data.astype("float64") * float(sf) + add, "cf", ()
        if decode == "rex" and sf is not None:
            return data.astype("float64") / float(sf), "rex", ()
        notes: list[str] = []
        if sf is not None:
            notes.append(
                f"raw values; unapplied scale_factor={float(sf)}"
                + (f", add_offset={float(offset)}" if offset is not None else "")
                + " (use --decode cf or --decode rex)"
            )
        return data, "none", tuple(notes)

    def stats(self, approved: ApprovedRead, spec: StatsSpec) -> StatsResult:
        """Compute statistics over a strided chunk sample, or the full array if exact.

        Parameters
        ----------
        approved : ApprovedRead
            Read plan that passed the budget check.
        spec : StatsSpec
            Stats request with the element budget.

        Returns
        -------
        StatsResult
            Summary statistics and how they were sampled.
        """
        np = lazy_import("numpy", "computing statistics")
        node = approved.plan.node
        dset = self._require_dataset(node.path)
        shape = dset.shape
        total = math.prod(shape) if shape else int(dset.size)

        if not shape:
            data = np.asarray(dset[()]).reshape(1)
            method = "full"
        elif spec.exact or total <= spec.max_elements:
            data = np.asarray(dset[...])
            method = "full"
        else:
            data = self._strided_sample(dset, spec)
            method = "chunk-strided"

        read = int(data.size)
        return _summarize(np, data, node.path, total, read, method, spec)

    def _strided_sample(self, dset: Any, spec: StatsSpec) -> Any:
        """Read evenly spaced axis-0 blocks until the element budget is met.

        Parameters
        ----------
        dset : Any
            Open h5py dataset.
        spec : StatsSpec
            Stats request with the element budget.

        Returns
        -------
        Any
            Flat numpy array of the sampled values.
        """
        np = lazy_import("numpy", "computing statistics")
        shape = dset.shape
        block = dset.chunks[0] if dset.chunks else max(1, self._sample_rows(dset, spec))
        n_blocks = math.ceil(shape[0] / block)
        want_rows = self._sample_rows(dset, spec)
        want_blocks = max(1, math.ceil(want_rows / block))
        stride = max(1, n_blocks // want_blocks)
        pieces = []
        taken = 0
        for b in range(0, n_blocks, stride):
            start = b * block
            stop = min(start + block, shape[0])
            pieces.append(np.asarray(dset[start:stop]))
            taken += stop - start
            if taken >= want_rows:
                break
        return np.concatenate([p.ravel() for p in pieces])


def _n_chunks(dset: Any) -> int:
    """Total number of chunks in a dataset, or 1 if contiguous.

    Parameters
    ----------
    dset : Any
        Open h5py dataset.

    Returns
    -------
    int
        Chunk count.
    """
    if not dset.chunks:
        return 1
    return math.prod(math.ceil(d / c) for d, c in zip(dset.shape, dset.chunks, strict=False))


def _to_jsonable(data: Any) -> AttrValue:
    """Convert a numpy array or scalar to nested JSON-safe lists.

    Parameters
    ----------
    data : Any
        Numpy array or scalar, or an already plain value.

    Returns
    -------
    AttrValue
        JSON-safe value.
    """
    if hasattr(data, "tolist"):
        out = data.tolist()
        return _bytes_to_str(out)
    return data


def _bytes_to_str(obj: Any) -> AttrValue:
    """Recursively decode bytes in nested lists.

    Parameters
    ----------
    obj : Any
        Value that may be or contain bytes.

    Returns
    -------
    AttrValue
        Value with all bytes decoded to text.
    """
    if isinstance(obj, bytes):
        return obj.decode("utf-8", "replace")
    if isinstance(obj, list):
        return [_bytes_to_str(x) for x in obj]
    return obj


def _summarize(
    np: Any, data: Any, path: NodePath, total: int, read: int, method: str, spec: StatsSpec
) -> StatsResult:
    """Reduce sampled data to a StatsResult with honest coverage fields."""
    numeric = data.dtype.kind in "iufc"
    sampled = read < total
    fraction = (read / total) if total else 1.0
    if not numeric or read == 0:
        return StatsResult(
            path=path,
            count=read,
            n_nan=0,
            mean=None,
            std=None,
            min=None,
            max=None,
            sampled=sampled,
            sample_fraction=fraction,
            sample_method=method,  # type: ignore[arg-type]
        )
    flat = data.astype("float64").ravel()
    n_nan = int(np.isnan(flat).sum())
    clean = flat[~np.isnan(flat)] if spec.nan_policy == "omit" else flat
    has = clean.size > 0
    return StatsResult(
        path=path,
        count=read,
        n_nan=n_nan,
        mean=float(np.mean(clean)) if has else None,
        std=float(np.std(clean)) if has else None,
        min=float(np.min(clean)) if has else None,
        max=float(np.max(clean)) if has else None,
        sampled=sampled,
        sample_fraction=fraction,
        sample_method=method,  # type: ignore[arg-type]
    )
