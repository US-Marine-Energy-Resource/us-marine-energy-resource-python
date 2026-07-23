"""Render exploration results as rich terminal output or JSON."""

from __future__ import annotations

import dataclasses
import json
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from ...wave_hindcast.config import CONFIG as _WAVE_CONFIG
from ..catalog import ENDPOINTS, Listing, TreeNode
from ..model import ByteSize, FileHeader, FileSummary, HeadResult, NodeInfo, NodePath, StatsResult


def to_json(obj: Any, *, extra: dict[str, Any] | None = None) -> str:
    """Serialize a result to JSON, with ByteSize as int and NodePath as str.

    Parameters
    ----------
    obj : Any
        Result object to serialize.
    extra : dict, optional
        Extra top level keys to merge into the payload.

    Returns
    -------
    str
        The JSON text.
    """
    payload = _jsonify(obj)
    if extra:
        assert isinstance(payload, dict)
        payload.update(extra)
    return json.dumps(payload, indent=2)


def _jsonify(obj: Any) -> Any:
    """Recursively convert model objects to JSON-safe values.

    Parameters
    ----------
    obj : Any
        Value to convert.

    Returns
    -------
    Any
        A JSON safe equivalent of the value.
    """
    if isinstance(obj, ByteSize):
        return obj.bytes
    if isinstance(obj, NodePath):
        return obj.value
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _jsonify(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, list | tuple):
        return [_jsonify(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    return obj


def render_info(console: Console, summary: FileSummary) -> None:
    """Print a one-screen overview of a file.

    Parameters
    ----------
    console : Console
        Terminal to print to.
    summary : FileSummary
        Summary of the file.
    """
    s = summary
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(style="dim", no_wrap=True)
    table.add_column()
    table.add_row("source", s.source.display)
    table.add_row("format", f"{s.format}  ({s.format_detail})")
    if s.source.size is not None:
        table.add_row("size", str(s.source.size))
    table.add_row("arrays", str(s.n_arrays))
    table.add_row("groups", str(s.n_groups))
    console.print(table)
    if s.root_attrs:
        console.print("\n[bright_blue]root attributes[/]")
        _print_attrs(console, s.root_attrs)
    for w in s.warnings:
        console.print(f"[bright_blue]warning:[/] {w}")


def render_tree(
    console: Console,
    summary: FileSummary,
    *,
    show_storage: bool = False,
    root_path: str = "/",
    max_depth: int | None = None,
) -> None:
    """Print the file's group/array hierarchy as a tree.

    Parameters
    ----------
    console : Console
        Terminal to print to.
    summary : FileSummary
        Summary of the file.
    show_storage : bool
        Add chunk and compression details to array labels.
    root_path : str
        Node path to start the tree at.
    max_depth : int, optional
        Deepest level below the root to show.
    """
    base_depth = NodePath(root_path).depth
    tree = Tree(f"[bright_blue]{summary.source.display}[/]")
    branches: dict[str, Tree] = {"/": tree}
    for node in summary.nodes:
        if node.path.value == "/":
            continue
        if not node.path.value.startswith(root_path.rstrip("/") + "/") and root_path != "/":
            continue
        if max_depth is not None and node.path.depth - base_depth > max_depth:
            continue
        parent_key = node.path.value.rsplit("/", 1)[0] or "/"
        parent = branches.get(parent_key, tree)
        branches[node.path.value] = parent.add(_node_label(node, show_storage))
    console.print(tree)


def _node_label(node: NodeInfo, show_storage: bool) -> str:
    """Build a one-line label for a tree node.

    Parameters
    ----------
    node : NodeInfo
        Node to label.
    show_storage : bool
        Add chunk and compression details to array labels.

    Returns
    -------
    str
        The label with rich markup.
    """
    if node.array is None:
        return f"[bold bright_blue]{node.name}/[/]"
    a = node.array
    dims = ",".join(str(d) for d in a.shape)
    label = f"[green]{node.name}[/]  ({dims}) [dim]{a.dtype}[/]"
    if a.dim_names and any(a.dim_names):
        label += f" [dim]dims={a.dim_names}[/]"
    if show_storage:
        bits = []
        if a.storage.chunks:
            bits.append(f"chunks={a.storage.chunks}")
        if a.storage.compression:
            bits.append(str(a.storage.compression))
        if a.storage.compression_ratio:
            bits.append(f"{a.storage.compression_ratio}x")
        if bits:
            label += f"  [dim]{'  '.join(bits)}[/]"
    return label


def render_attrs(console: Console, summary: FileSummary, path: str | None) -> None:
    """Print root attributes, or the attributes of one node.

    Parameters
    ----------
    console : Console
        Terminal to print to.
    summary : FileSummary
        Summary of the file.
    path : str or None
        Node path to show, or None for the root and every node with
        attributes.
    """
    if path is None:
        console.print("[bright_blue]root[/]")
        _print_attrs(console, summary.root_attrs)
        for node in summary.nodes:
            if node.attrs and node.path.value != "/":
                console.print(f"\n[bright_blue]{node.path}[/]")
                _print_attrs(console, node.attrs)
        return
    target = NodePath(path).value
    for node in summary.nodes:
        if node.path.value == target:
            console.print(f"[bright_blue]{node.path}[/]")
            _print_attrs(console, node.attrs)
            return
    console.print(f"[bright_blue]no node at {path}[/]")


def _print_attrs(console: Console, attrs: dict[str, Any]) -> None:
    """Print a key/value attribute table.

    Parameters
    ----------
    console : Console
        Terminal to print to.
    attrs : dict
        Attribute names and values.
    """
    if not attrs:
        console.print("  [dim](none)[/]")
        return
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(style="bright_blue", no_wrap=True)
    table.add_column()
    for k, v in attrs.items():
        table.add_row(k, str(v))
    console.print(table)


def render_head(console: Console, result: HeadResult) -> None:
    """Print a value slice and any decode notes.

    Parameters
    ----------
    console : Console
        Terminal to print to.
    result : HeadResult
        Slice of values to show.
    """
    shape = ",".join(str(d) for d in result.shape)
    console.print(f"[bright_blue]{result.path}[/]  ({shape}) {result.dtype}")
    console.print(f"[dim]selection {result.selection}  decode={result.decode}[/]")
    # Print as a plain repr so rich soft-wraps wide arrays instead of expanding
    # every element onto its own line.
    console.print(repr(result.values), soft_wrap=True)
    for note in result.notes:
        console.print(f"[bright_blue]note:[/] {note}")


def render_ls(console: Console, label: str, listing: Listing) -> None:
    """Print a terse one-level listing: dirs first, then files with sizes.

    Parameters
    ----------
    console : Console
        Terminal to print to.
    label : str
        Name of the listed location.
    listing : Listing
        Entries to show.
    """
    for entry in listing.entries:
        if entry.is_dir:
            console.print(f"[bright_blue]{entry.name}/[/]")
        else:
            size = ByteSize(entry.size) if entry.size is not None else None
            console.print(f"[green]{entry.name}[/]" + (f"  [dim]{size}[/]" if size else ""))
    if listing.truncated:
        console.print("[bright_blue]… more (raise --limit or narrow with --filter)[/]")
    if not listing.entries:
        console.print("[dim](empty)[/]")


def render_variables(console: Console, summary: FileSummary) -> None:
    """Print each variable with its shape, type, and attributes.

    Parquet columns carry their schema field metadata as attributes, and
    HDF5/netCDF variables carry their variable attributes. Groups appear only
    when they have attributes of their own.

    Parameters
    ----------
    console : Console
        Terminal to print to.
    summary : FileSummary
        Summary of the file.
    """
    for node in summary.nodes:
        if node.path.value == "/":
            continue
        if node.array is None and not node.attrs:
            continue
        label = f"\n[bright_blue]{node.path}[/]"
        if node.array is not None:
            dims = ",".join(str(d) for d in node.array.shape)
            label += f"  ({dims}) [dim]{node.array.dtype}[/]"
        console.print(label)
        if node.attrs:
            _print_attrs(console, node.attrs)


def render_file_header(console: Console, header: FileHeader) -> None:
    """Print a file's descriptive metadata: format, size, and root attributes.

    Parameters
    ----------
    console : Console
        Terminal to print to.
    header : FileHeader
        Header metadata to show.
    """
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(style="dim", no_wrap=True)
    table.add_column()
    table.add_row("source", header.source.display)
    table.add_row("format", f"{header.format}  ({header.format_detail})")
    if header.source.size is not None:
        table.add_row("size", str(header.source.size))
    console.print(table)
    console.print("\n[bright_blue]root attributes[/]")
    _print_attrs(console, header.root_attrs)


def render_prefix_info(console: Console, label: str, uri: str, node: TreeNode) -> None:
    """Print a truncated tree of a prefix with sizes and an aggregate summary.

    Parameters
    ----------
    console : Console
        Terminal to print to.
    label : str
        Name of the prefix.
    uri : str
        Full S3 URI of the prefix.
    node : TreeNode
        Browsed tree to show.
    """
    n_dirs = sum(1 for c in node.children if c.is_dir)
    files = [c for c in node.children if not c.is_dir]
    total = sum(c.size or 0 for c in files)
    tree = Tree(f"[bright_blue]{label}[/]  [dim]{uri}[/]")
    _attach_children(tree, node)
    console.print(tree)
    summary = f"{n_dirs} dir(s), {len(files)} file(s) shown"
    if total:
        summary += f", {ByteSize(total)} in shown files"
    console.print(f"[dim]{summary}[/]")


def render_catalog(console: Console, label: str, root_uri: str, node: TreeNode) -> None:
    """Print a browsed S3 tree, marking truncated levels and closed directories.

    Parameters
    ----------
    console : Console
        Terminal to print to.
    label : str
        Name of the dataset.
    root_uri : str
        Full S3 URI of the browsed root.
    node : TreeNode
        Browsed tree to show.
    """
    tree = Tree(f"[bright_blue]{label}[/]  [dim]{root_uri}[/]")
    _attach_children(tree, node)
    console.print(tree)
    console.print(
        f"[dim]drill in:[/] mer explore {label}/[bright_blue]<name>[/]   "
        "[dim]· open a file:[/] mer explore [bright_blue]s3://<bucket>/<...>.h5[/]"
    )


def _attach_children(branch: Tree, node: TreeNode) -> None:
    """Recursively add a browsed node's children to a rich tree branch.

    Parameters
    ----------
    branch : Tree
        Branch to add the children to.
    node : TreeNode
        Browsed node whose children are added.
    """
    for child in node.children:
        if child.is_dir:
            label = f"[bright_blue]{child.name}/[/]"
            if not child.expanded and child.children:
                label += "  [dim]…[/]"
            sub = branch.add(label)
            if child.expanded:
                _attach_children(sub, child)
        else:
            size = f"  [dim]{ByteSize(child.size)}[/]" if child.size is not None else ""
            branch.add(f"[green]{child.name}[/]{size}")
    if node.truncated:
        branch.add("[bright_blue]… more (raise --limit or narrow with --filter)[/]")


def render_catalog_json(node: TreeNode) -> str:
    """Serialize a browsed tree to JSON.

    Parameters
    ----------
    node : TreeNode
        Browsed tree to serialize.

    Returns
    -------
    str
        The JSON text.
    """
    return json.dumps(_catalog_dict(node), indent=2)


def _catalog_dict(node: TreeNode) -> dict[str, Any]:
    """Convert a TreeNode to a JSON-safe dict.

    Parameters
    ----------
    node : TreeNode
        Node to convert.

    Returns
    -------
    dict
        The node and its children as plain values.
    """
    return {
        "name": node.name,
        "is_dir": node.is_dir,
        "size": node.size,
        "truncated": node.truncated,
        "children": [_catalog_dict(c) for c in node.children],
    }


# One-line description per path verb, used by the no-argument overview so
# `mer ls` talks about ls rather than about explore.
_OVERVIEW_TAGLINES = {
    "ls": "lists the immediate children of a directory, dataset, or s3:// prefix.",
    "info": "shows metadata for a dataset prefix or a single file.",
    "explore": "inspects files and browses a dataset's S3 layout.",
}

# File examples per verb, so the overview only shows flags the verb accepts.
_OVERVIEW_FILE_EXAMPLES = {
    "ls": ["[bright_blue]data.h5[/]                       [dim]name and size[/]"],
    "info": ["[bright_blue]data.h5[/]                       [dim]format and root attributes[/]"],
    "explore": [
        "[bright_blue]data.h5[/]                       [dim]overview[/]",
        "[bright_blue]cook_inlet.nc[/] --tree --storage",
        f"[bright_blue]{_WAVE_CONFIG.s3_bucket_uri}/v1.0.1/West_Coast/West_Coast_wave_2010.h5[/]",
    ],
}


def render_explore_overview(console: Console, verb: str = "explore") -> None:
    """Print the overview shown when a path verb runs with no argument.

    Parameters
    ----------
    console : Console
        Terminal to print to.
    verb : str
        Command name the overview describes.
    """
    console.print(f"[bright_blue]mer {verb}[/] {_OVERVIEW_TAGLINES[verb]}\n")
    console.print("[bright_blue]Browse a dataset[/] (drill in by extending the path):")
    rows = [(name, f"s3://{bucket}/{prefix}") for name, (bucket, prefix) in ENDPOINTS.items()] + [
        ("tidal/AK_cook_inlet", "drill into a location")
    ]
    column = max(len(path) for path, _ in rows) + 2
    for path, description in rows:
        console.print(f"  mer {verb} [bright_blue]{path:<{column}}[/][dim]{description}[/]")
    if verb == "explore":
        console.print("  mer explore [bright_blue]wave[/] --filter 'v1.0.1' --depth 2")
    console.print()
    console.print("[bright_blue]Inspect one file[/] (local path, s3://..., or https://...):")
    for example in _OVERVIEW_FILE_EXAMPLES[verb]:
        console.print(f"  mer {verb} {example}")
    console.print(f"\nRun [bright_blue]mer {verb} --help[/] for all options.")


def render_stats(console: Console, result: StatsResult) -> None:
    """Print summary statistics and how much of the array they cover.

    Parameters
    ----------
    console : Console
        Terminal to print to.
    result : StatsResult
        Statistics to show.
    """
    r = result
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(style="dim", no_wrap=True)
    table.add_column()
    table.add_row("path", str(r.path))
    table.add_row("count", f"{r.count:,}")
    table.add_row("nan", f"{r.n_nan:,}")
    for name, val in (("mean", r.mean), ("std", r.std), ("min", r.min), ("max", r.max)):
        table.add_row(name, "n/a" if val is None else f"{val:.6g}")
    coverage = "full" if not r.sampled else f"{r.sample_fraction:.3%} ({r.sample_method})"
    table.add_row("coverage", coverage)
    console.print(table)
