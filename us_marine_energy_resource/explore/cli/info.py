"""The ``mer info`` verb: metadata for a prefix or a file."""

from __future__ import annotations

import typer

from ...cli._display import console
from .. import _open_with_reader
from ..catalog import PrefixPointer, build_tree, make_lister, resolve_path
from ..errors import TransferBudgetExceededError
from ..model import MB
from ._shared import (
    _bytes_fetched,
    _handle_errors,
    _listing_status,
    _policy,
    _progress,
    _warn_if_large,
)
from .options import (
    AwsProfileOpt,
    DepthOpt,
    FilterOpt,
    JsonOpt,
    LimitOpt,
    MaxTransferOpt,
    UriArg,
    VariablesOpt,
)
from .render import (
    render_catalog_json,
    render_explore_overview,
    render_file_header,
    render_info,
    render_prefix_info,
    render_variables,
    to_json,
)


def info(
    path: UriArg = None,
    depth: DepthOpt = None,
    limit: LimitOpt = 200,
    name_filter: FilterOpt = None,
    variables: VariablesOpt = False,
    max_transfer_mb: MaxTransferOpt = None,
    aws_profile: AwsProfileOpt = None,
    json_out: JsonOpt = False,
) -> None:
    """Show metadata for a prefix (sizes + truncated tree) or a file (format + attrs).

    Parameters
    ----------
    path : str, optional
        Endpoint name, endpoint sub-path, ``s3://`` prefix, or local path.
        Prints the overview when omitted.
    depth : int, optional
        Levels of a prefix tree to expand.
    limit : int
        Maximum entries listed per level.
    name_filter : str, optional
        Glob applied to entry names at the top level.
    variables : bool
        Also read each variable's attributes from a remote HDF5 file.
    max_transfer_mb : float, optional
        Network transfer ceiling in megabytes.
    aws_profile : str, optional
        AWS profile for signed S3 access. Anonymous when omitted.
    json_out : bool
        Emit JSON instead of formatted text.
    """
    if path is None:
        render_explore_overview(console, verb="info")
        return
    with _handle_errors():
        pointer = resolve_path(path)
        if isinstance(pointer, PrefixPointer):
            lister = make_lister(pointer, aws_profile)
            with _listing_status(pointer) as status:
                node = build_tree(
                    lister,
                    pointer.prefix,
                    depth=depth or 2,
                    limit=limit,
                    name_filter=name_filter,
                    on_list=_progress(status),
                )
            if json_out:
                typer.echo(render_catalog_json(node))
            else:
                render_prefix_info(console, pointer.label, pointer.uri, node)
            return
        # Show the quick header right away. Walking every variable's attributes
        # means many small remote reads, so by default it runs only where it is
        # quick (local files and parquet) and otherwise waits for --variables.
        trip: Exception | None = None
        summary = None
        with _open_with_reader(
            pointer.uri,
            policy=_policy(max_transfer_mb, None, False, False),
            aws_profile=aws_profile,
            metadata_only=True,
        ) as (f, reader, ref):
            _warn_if_large(ref)
            remote = _bytes_fetched(reader) is not None
            header = f.header()
            if variables or not remote or header.format == "parquet":
                try:
                    if remote:
                        with console.status(
                            "[bright_blue]reading each variable's attributes over the network "
                            "(this can take minutes for a large file)…"
                        ):
                            summary = f.summary()
                    else:
                        summary = f.summary()
                except TransferBudgetExceededError as exc:
                    trip = exc
        if json_out:
            typer.echo(to_json(summary if summary is not None else header))
            return
        if summary is not None:
            render_info(console, summary)
            render_variables(console, summary)
            return
        render_file_header(console, header)
        if trip is not None:
            mb = getattr(trip, "fetched", 0) / MB
            console.print(
                f"\n[bright_blue]note:[/] variable attributes were stopped after {mb:.0f} MB "
                "because this file's metadata is unusually large. Rerun with a higher "
                f"ceiling, for example --max-transfer-mb {max(200, int(mb * 2))}.",
                highlight=False,
            )
        else:
            console.print(
                "\n[dim]Each variable's attributes were not read: they are scattered "
                "through this remote file and take a while to collect. Add "
                "[bright_blue]--variables[/] to read them, or use "
                "[bright_blue]mer explore PATH --tree[/] for the structure alone.[/]",
                highlight=False,
            )
