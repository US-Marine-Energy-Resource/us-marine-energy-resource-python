"""The ``mer explore`` verb: inspect a file or browse a dataset's S3 layout."""

from __future__ import annotations

import dataclasses
from typing import Any, cast

import typer

from ...cli._display import console, error
from .. import _open_with_reader
from ..budget import ApprovedRead, NeedsConfirm, ReadPlan, Refusal, TransferPolicy
from ..catalog import PrefixPointer, build_tree, make_lister, resolve_path
from ..model import Decode, NodePath, StatsSpec
from ..protocols import OpenFile
from ..selection import FirstN, Index, Selection
from ._shared import (
    _bytes_fetched,
    _handle_errors,
    _listing_status,
    _policy,
    _progress,
    _warn_if_large,
)
from .options import (
    AttrsModeOpt,
    AwsProfileOpt,
    DecodeOpt,
    DepthOpt,
    DryRunOpt,
    ExactOpt,
    FilterOpt,
    HeadModeOpt,
    IndexOpt,
    InfoModeOpt,
    JsonOpt,
    LimitOpt,
    MaxElementsOpt,
    MaxMemoryOpt,
    MaxTransferOpt,
    NOpt,
    PathOpt,
    StatsModeOpt,
    StorageOpt,
    TreeModeOpt,
    UriArg,
    YesOpt,
)
from .render import (
    render_attrs,
    render_catalog,
    render_catalog_json,
    render_explore_overview,
    render_head,
    render_info,
    render_stats,
    render_tree,
    to_json,
)


def explore(
    uri: UriArg = None,
    info_mode: InfoModeOpt = False,
    tree: TreeModeOpt = False,
    attrs: AttrsModeOpt = False,
    head: HeadModeOpt = False,
    stats: StatsModeOpt = False,
    path: PathOpt = None,
    json_out: JsonOpt = False,
    aws_profile: AwsProfileOpt = None,
    storage: StorageOpt = False,
    depth: DepthOpt = None,
    limit: LimitOpt = 200,
    name_filter: FilterOpt = None,
    n: NOpt = None,
    index: IndexOpt = None,
    decode: DecodeOpt = "none",
    max_elements: MaxElementsOpt = 1_000_000,
    exact: ExactOpt = False,
    max_transfer_mb: MaxTransferOpt = None,
    max_memory_mb: MaxMemoryOpt = None,
    dry_run: DryRunOpt = False,
    yes: YesOpt = False,
) -> None:
    """Inspect a file's contents, or browse a dataset's S3 layout.

    Parameters
    ----------
    uri : str, optional
        Endpoint name, endpoint sub-path, ``s3://`` prefix, local path, or
        URL. Prints the overview when omitted.
    info_mode : bool
        Show only the header.
    tree : bool
        Show only the hierarchy.
    attrs : bool
        Show attributes of the ``path`` node.
    head : bool
        Read values from ``path``.
    stats : bool
        Compute statistics for ``path``.
    path : str, optional
        Node path inside the file.
    json_out : bool
        Emit JSON instead of formatted text.
    aws_profile : str, optional
        AWS profile for signed S3 access. Anonymous when omitted.
    storage : bool
        Show chunks, compression, and ratio in the tree.
    depth : int, optional
        Levels of a tree to expand.
    limit : int
        Maximum entries listed per level when browsing.
    name_filter : str, optional
        Glob applied to entry names at the top level when browsing.
    n : int, optional
        Read the first ``n`` entries along axis 0.
    index : str, optional
        Numpy style slice to read instead of ``n``.
    decode : str
        Value decoding: ``none``, ``cf``, or ``rex``.
    max_elements : int
        Sampling budget for statistics.
    exact : bool
        Read the whole array for statistics, gated by the policy.
    max_transfer_mb : float, optional
        Network transfer ceiling in megabytes.
    max_memory_mb : float, optional
        Per-read memory ceiling in megabytes.
    dry_run : bool
        Show the read plan and exit without reading.
    yes : bool
        Skip confirmation prompts.
    """
    if uri is None:
        render_explore_overview(console)
        return

    flags = {"info": info_mode, "tree": tree, "attrs": attrs, "head": head, "stats": stats}
    chosen = [name for name, on in flags.items() if on]

    pointer = resolve_path(uri)
    if isinstance(pointer, PrefixPointer):
        if chosen:
            error("mode flags (--head, --stats, …) apply to files, not directories")
            raise typer.Exit(1)
        _do_browse(pointer, depth or 1, limit, name_filter, aws_profile, json_out)
        return

    if len(chosen) > 1:
        error(f"choose one mode, got: {', '.join('--' + m for m in chosen)}")
        raise typer.Exit(1)
    mode = chosen[0] if chosen else "overview"

    if mode in ("head", "stats") and path is None:
        error(f"--{mode} requires --path")
        raise typer.Exit(1)
    if n is not None and index is not None:
        error("give -n or --index, not both")
        raise typer.Exit(1)

    reads_data = mode in ("head", "stats")
    # Structure inspection keeps the modest transfer ceiling. A file with
    # unusually large metadata trips the fuse instead of fetching it silently.
    policy = _policy(max_transfer_mb, max_memory_mb if reads_data else None, yes, dry_run)

    with (
        _handle_errors(),
        _open_with_reader(
            uri, policy=policy, aws_profile=aws_profile, metadata_only=not reads_data
        ) as (f, reader, ref),
    ):
        _warn_if_large(ref)
        if mode == "head":
            _do_head(f, reader, path, n, index, decode, policy, json_out)
        elif mode == "stats":
            _do_stats(f, reader, path, max_elements, exact, policy, json_out)
        else:
            _do_metadata(f, reader, mode, path, storage, depth, json_out)


def _do_browse(
    pointer: PrefixPointer,
    depth: int,
    limit: int,
    name_filter: str | None,
    aws_profile: str | None,
    json_out: bool,
) -> None:
    """List a prefix as a tree, with a progress spinner while it walks.

    Parameters
    ----------
    pointer : PrefixPointer
        The prefix to browse.
    depth : int
        Levels to expand.
    limit : int
        Maximum entries listed per level.
    name_filter : str or None
        Glob applied to entry names at the top level.
    aws_profile : str or None
        AWS profile for signed S3 access. Anonymous when ``None``.
    json_out : bool
        Emit JSON instead of formatted text.
    """
    lister = make_lister(pointer, aws_profile)
    if json_out:
        node = build_tree(
            lister,
            pointer.prefix,
            depth=depth,
            limit=limit,
            name_filter=name_filter,
        )
        typer.echo(render_catalog_json(node))
        return
    with _listing_status(pointer) as status:
        node = build_tree(
            lister,
            pointer.prefix,
            depth=depth,
            limit=limit,
            name_filter=name_filter,
            on_list=_progress(status),
        )
    render_catalog(console, pointer.label, pointer.uri, node)


def _do_metadata(
    f: OpenFile,
    reader: Any,
    mode: str,
    path: str | None,
    storage: bool,
    depth: int | None,
    json_out: bool,
) -> None:
    """Render a file view: overview, info, tree, or attrs (full structure walk).

    Parameters
    ----------
    f : OpenFile
        The open file.
    reader : object
        The raw reader behind the file, or ``None`` for local files.
    mode : str
        The chosen view.
    path : str or None
        Node path the view is rooted at.
    storage : bool
        Show chunks, compression, and ratio.
    depth : int or None
        Levels of the tree to show.
    json_out : bool
        Emit JSON instead of formatted text.
    """
    if _bytes_fetched(reader) is not None:
        note = " (+ per-array sizes)" if storage else ""
        with console.status(f"[bright_blue]reading structure over the network{note}…"):
            summary = f.summary(storage=storage)
    else:
        summary = f.summary(storage=storage)
    if json_out:
        fetched = _bytes_fetched(reader)
        extra = {"bytes_fetched": fetched} if fetched is not None else None
        typer.echo(to_json(summary, extra=extra))
        return
    if mode == "info":
        render_info(console, summary)
    elif mode == "tree":
        render_tree(console, summary, show_storage=storage, root_path=path or "/", max_depth=depth)
    elif mode == "attrs":
        render_attrs(console, summary, path)
    else:
        render_info(console, summary)
        console.print()
        render_tree(console, summary, show_storage=storage, root_path=path or "/", max_depth=depth)


def _do_head(
    f: OpenFile,
    reader: Any,
    path: str | None,
    n: int | None,
    index: str | None,
    decode: str,
    policy: TransferPolicy,
    json_out: bool,
) -> None:
    """Read and render a value slice from one array.

    Parameters
    ----------
    f : OpenFile
        The open file.
    reader : object
        The raw reader behind the file, or ``None`` for local files.
    path : str or None
        Node path of the array to read.
    n : int or None
        Read the first ``n`` entries along axis 0.
    index : str or None
        Numpy style slice to read instead of ``n``.
    decode : str
        Value decoding: ``none``, ``cf``, or ``rex``.
    policy : TransferPolicy
        Volume limits gating the read.
    json_out : bool
        Emit JSON instead of formatted text.

    Raises
    ------
    typer.Exit
        When the read is refused, declined, or was a dry run.
    """
    assert path is not None
    selection: Selection = Index(index) if index is not None else FirstN(n or 5)
    plan = f.plan_read(NodePath(path), selection)
    approved = _gate(policy, plan, remote=_bytes_fetched(reader) is not None)
    if approved is None:
        raise typer.Exit(0 if policy.dry_run else 1)
    result = f.head(approved, cast(Decode, decode))
    typer.echo(to_json(result)) if json_out else render_head(console, result)


def _do_stats(
    f: OpenFile,
    reader: Any,
    path: str | None,
    max_elements: int,
    exact: bool,
    policy: TransferPolicy,
    json_out: bool,
) -> None:
    """Compute and render statistics for one array.

    Parameters
    ----------
    f : OpenFile
        The open file.
    reader : object
        The raw reader behind the file, or ``None`` for local files.
    path : str or None
        Node path of the array to summarize.
    max_elements : int
        Sampling budget for statistics.
    exact : bool
        Read the whole array, gated by the policy.
    policy : TransferPolicy
        Volume limits gating the read.
    json_out : bool
        Emit JSON instead of formatted text.

    Raises
    ------
    typer.Exit
        When the read is refused, declined, or was a dry run.
    """
    assert path is not None
    spec = StatsSpec(max_elements=max_elements, exact=exact)
    plan = f.plan_stats(NodePath(path), spec)
    approved = _gate(policy, plan, remote=_bytes_fetched(reader) is not None)
    if approved is None:
        raise typer.Exit(0 if policy.dry_run else 1)
    result = f.stats(approved, spec)
    typer.echo(to_json(result)) if json_out else render_stats(console, result)


def _gate(policy: TransferPolicy, plan: ReadPlan, *, remote: bool) -> ApprovedRead | None:
    """Apply the policy to a plan, prompting or refusing as needed.

    Parameters
    ----------
    policy : TransferPolicy
        Volume limits gating the read.
    plan : ReadPlan
        What the read would cost.
    remote : bool
        Whether the source moves bytes over the network.

    Returns
    -------
    ApprovedRead or None
        ``None`` when the read is refused, declined, or was a dry run.
    """
    if policy.dry_run:
        console.print(
            f"[dim]plan[/] logical={plan.logical} transferred={plan.transferred} "
            f"chunks={plan.n_chunks} amplification={plan.amplification:.1f}x"
        )
        return None
    outcome = policy.approve(plan, remote=remote)
    if isinstance(outcome, ApprovedRead):
        return outcome
    if isinstance(outcome, Refusal):
        error(outcome.message())
        return None
    if isinstance(outcome, NeedsConfirm):
        if not console.file.isatty():
            error(f"read would use {outcome.size}. Rerun with --yes to proceed")
            return None
        if not typer.confirm(f"Read {outcome.size}?"):
            return None
        forced = dataclasses.replace(policy, assume_yes=True)
        result = forced.approve(plan, remote=remote)
        return result if isinstance(result, ApprovedRead) else None
    return None
