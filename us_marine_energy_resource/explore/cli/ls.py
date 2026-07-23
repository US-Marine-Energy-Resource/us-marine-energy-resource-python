"""The ``mer ls`` verb: a terse one-level listing."""

from __future__ import annotations

from typing import Any

import typer

from ...cli._display import console
from ..catalog import FilePointer, list_children, make_lister, resolve_path
from ._shared import _handle_errors, _listing_status
from .options import AwsProfileOpt, FilterOpt, JsonOpt, LimitOpt, UriArg
from .render import render_catalog_json, render_explore_overview, render_ls


def ls(
    path: UriArg = None,
    limit: LimitOpt = 200,
    name_filter: FilterOpt = None,
    aws_profile: AwsProfileOpt = None,
    json_out: JsonOpt = False,
) -> None:
    """List the immediate children of a directory, dataset, or prefix.

    Parameters
    ----------
    path : str, optional
        Endpoint name, endpoint sub-path, ``s3://`` prefix, or local path.
        Prints the overview when omitted.
    limit : int
        Maximum entries listed.
    name_filter : str, optional
        Glob applied to entry names.
    aws_profile : str, optional
        AWS profile for signed S3 access. Anonymous when omitted.
    json_out : bool
        Emit JSON instead of formatted text.
    """
    if path is None:
        render_explore_overview(console, verb="ls")
        return
    with _handle_errors():
        pointer = resolve_path(path)
        if isinstance(pointer, FilePointer):
            _print_file_entry(pointer, aws_profile)
            return
        lister = make_lister(pointer, aws_profile)
        with _listing_status(pointer):
            listing = list_children(lister, pointer.prefix, limit, name_filter)
        if json_out:
            typer.echo(render_catalog_json(_listing_as_node(pointer, listing)))
        else:
            render_ls(console, pointer.label, listing)


def _print_file_entry(pointer: FilePointer, aws_profile: str | None) -> None:
    """Print a single file's name and size (one HEAD).

    Parameters
    ----------
    pointer : FilePointer
        The file to describe.
    aws_profile : str or None
        AWS profile for signed S3 access. Anonymous when ``None``.
    """
    from ..sources import resolve_source

    ref = resolve_source(pointer.uri, aws_profile=aws_profile).ref
    size = f"  [dim]{ref.size}[/]" if ref.size is not None else ""
    console.print(f"[green]{pointer.label}[/]{size}")


def _listing_as_node(pointer: Any, listing: Any) -> Any:
    """Wrap a one-level listing as a TreeNode for JSON output.

    Parameters
    ----------
    pointer : PrefixPointer
        The prefix that was listed.
    listing : Listing
        The one-level listing to wrap.

    Returns
    -------
    TreeNode
        The listing as a single-level tree.
    """
    from ..catalog import TreeNode

    children = tuple(
        TreeNode(e.name, e.is_dir, e.size, (), False, expanded=False) for e in listing.entries
    )
    return TreeNode(pointer.prefix, True, None, children, listing.truncated, expanded=True)
