"""Shared plumbing for the ``mer`` file verbs.

Holds what every verb needs: the help and epilog text, the error handler that
turns expected failures into clean exits, the listing spinners, the large
file heads-up, and the policy builder for the volume flags.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import typer

from ...cli._display import console, error
from ..budget import TransferPolicy, build_policy
from ..catalog import PrefixPointer
from ..config import CONFIG
from ..errors import ExploreError, TransferBudgetExceededError
from ..model import MB

LS_HELP = "List the immediate children of a directory, dataset, or s3:// prefix."
INFO_HELP = (
    "Show metadata: a prefix's sizes and a truncated tree, or a file's global "
    "attributes plus each variable's attributes (HDF5/netCDF) or column schema "
    "(parquet). For a remote HDF5 file, add --variables to collect every "
    "variable's attributes."
)
EXPLORE_HELP = (
    "Inspect a file's contents (structure, values, statistics), or browse a "
    "dataset's S3 layout. For a file, prints an overview by default. --info, "
    "--tree, --attrs, --head, and --stats select a single view."
)
DOWNLOAD_HELP = (
    "Download a file, or one level of a directory, with a size limit and a progress bar."
)

EXPLORE_EPILOG = (
    "[bright_blue]Examples[/bright_blue]\n\n"
    "mer explore tidal/AK_cook_inlet                  Browse an S3 location\n\n"
    "mer explore data.h5                              File overview (header + tree)\n\n"
    "mer explore data.h5 --tree --storage            Hierarchy with chunks/compression\n\n"
    "mer explore data.h5 --head -p /var -n 5         First 5 rows of an array\n\n"
    "mer explore data.h5 --stats -p /var --dry-run   Show the read plan, read nothing"
)


def _large_warning(ref: Any) -> str | None:
    """Return a heads-up for a large remote file, or ``None`` if it is not large.

    Uses the object size from the HEAD done at open time, so it costs no read.

    Parameters
    ----------
    ref : SourceRef
        The resolved source of the file about to be read.

    Returns
    -------
    str or None
        The message to print, or ``None`` when no heads-up applies.
    """
    if ref.scheme == "file" or ref.size is None or ref.size.bytes < CONFIG.large_remote_bytes:
        return None
    return (
        f"[bright_blue]note:[/] {ref.display} is [bright_blue]{ref.size}[/]. Only what you "
        "ask for is read: the header is quick, walking the full structure or every "
        "variable's attributes takes a while, and values (--head/--stats) read at most "
        "one slice."
    )


def _warn_if_large(ref: Any) -> None:
    """Print the large-file heads-up when one applies.

    Parameters
    ----------
    ref : SourceRef
        The resolved source of the file about to be read.
    """
    message = _large_warning(ref)
    if message is not None:
        console.print(message, highlight=False)


def _bytes_fetched(reader: Any) -> int | None:
    """Return bytes fetched by a block-cached reader, if it has a counter.

    Parameters
    ----------
    reader : object
        The raw reader behind an open file, or ``None`` for local files.

    Returns
    -------
    int or None
        Bytes fetched so far, or ``None`` when the reader has no counter.
    """
    return getattr(reader, "bytes_fetched", None)


def _listing_status(pointer: PrefixPointer) -> Any:
    """Return a progress spinner for a listing, active only for remote prefixes.

    Parameters
    ----------
    pointer : PrefixPointer
        The prefix about to be listed.

    Returns
    -------
    context manager
        A live spinner for remote prefixes, or a no-op for local ones.
    """
    if pointer.kind == "local":
        return _null_status()
    return console.status(f"[bright_blue]listing {pointer.uri}")


@contextmanager
def _null_status() -> Iterator[None]:
    """Yield a no-op status context for local, instant listings.

    Yields
    ------
    None
        Nothing. The context exists only to match the spinner's shape.
    """
    yield None


def _progress(status: Any) -> Any:
    """Build an on_list callback that updates a status spinner, or None for local.

    Parameters
    ----------
    status : object
        The live spinner, or ``None`` when listing locally.

    Returns
    -------
    callable or None
        A callback that updates the spinner, or ``None`` when there is none.
    """
    if status is None:
        return None
    return lambda p: status.update(f"[bright_blue]listing {p}")


def _policy(
    max_transfer_mb: float | None, max_memory_mb: float | None, yes: bool, dry_run: bool
) -> TransferPolicy:
    """Build a policy from CLI volume flags.

    Parameters
    ----------
    max_transfer_mb : float or None
        Network transfer ceiling in megabytes, or ``None`` for the default.
    max_memory_mb : float or None
        Per-read memory ceiling in megabytes, or ``None`` for the default.
    yes : bool
        Skip confirmation prompts.
    dry_run : bool
        Estimate and stop instead of reading.

    Returns
    -------
    TransferPolicy
        The resolved policy.
    """
    return build_policy(
        max_transfer_mb=max_transfer_mb,
        max_memory_mb=max_memory_mb,
        assume_yes=yes,
        dry_run=dry_run,
    )


@contextmanager
def _handle_errors() -> Iterator[None]:
    """Turn expected exploration errors into clean CLI exits.

    Yields
    ------
    None
        Control, while watching for expected exploration errors.

    Raises
    ------
    typer.Exit
        With code 1 when an expected exploration error is caught.
    """
    try:
        yield
    except TransferBudgetExceededError as exc:
        mb = exc.fetched / MB
        error(
            f"stopped after {mb:.0f} MB: this file has unusually large or scattered "
            "metadata.\n  Rerun with a higher ceiling, e.g. --max-transfer-mb 200."
        )
        raise typer.Exit(1) from exc
    except ExploreError as exc:
        error(str(exc))
        raise typer.Exit(1) from exc
