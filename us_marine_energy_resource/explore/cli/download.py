"""The ``mer download`` verb: fetch a file, or one level of a directory."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import typer

from ...cli._display import console, error
from ...wave_hindcast.config import CONFIG as _WAVE_CONFIG
from ..budget import build_policy
from ..catalog import PrefixPointer, list_children, make_lister, resolve_path
from ..model import ByteSize
from ._shared import _handle_errors, _listing_status
from .options import AwsProfileOpt, MaxDownloadOpt, OutputOpt, UriArg, YesOpt


def download(
    path: UriArg = None,
    output: OutputOpt = None,
    max_download_mb: MaxDownloadOpt = None,
    aws_profile: AwsProfileOpt = None,
    yes: YesOpt = False,
) -> None:
    """Download a file, or one level of a directory, with a size gate and progress.

    Parameters
    ----------
    path : str, optional
        Endpoint name, endpoint sub-path, ``s3://`` prefix, local path, or
        URL of what to download.
    output : Path, optional
        Directory to download into. Defaults to the current directory.
    max_download_mb : float, optional
        Confirm before downloading more than this many megabytes.
    aws_profile : str, optional
        AWS profile for signed S3 access. Anonymous when omitted.
    yes : bool
        Skip confirmation prompts.
    """
    if path is None:
        error("give a file or directory to download (see 'mer download --help')")
        raise typer.Exit(1)
    with _handle_errors():
        pointer = resolve_path(path)
        if isinstance(pointer, PrefixPointer):
            _do_download_prefix(pointer, output, max_download_mb, aws_profile, yes)
        else:
            _do_download(pointer.uri, output, max_download_mb, aws_profile, yes)


def _do_download(
    uri: str, output: Path | None, max_download_mb: float | None, aws_profile: str | None, yes: bool
) -> None:
    """Download one file to disk with a size gate, progress, and atomic write.

    Parameters
    ----------
    uri : str
        The file to download.
    output : Path or None
        Directory to download into, or ``None`` for the current directory.
    max_download_mb : float or None
        Download ceiling in megabytes, or ``None`` for the default.
    aws_profile : str or None
        AWS profile for signed S3 access. Anonymous when ``None``.
    yes : bool
        Skip confirmation prompts.

    Raises
    ------
    typer.Exit
        When the download is declined or there is not enough free space.
    """
    from ..sources import resolve_source

    ref = resolve_source(uri, aws_profile=aws_profile).ref
    if ref.scheme == "file":
        # A local path may use backslashes, which the URI split below misses.
        name = Path(ref.uri).name or "download"
    else:
        name = uri.rstrip("/").rsplit("/", 1)[-1] or "download"
    dest_dir = output or Path.cwd()
    dest = dest_dir / name
    limit = build_policy(max_download_mb=max_download_mb).max_download

    if ref.size is not None and ref.size.bytes > limit.bytes:
        if yes or (console.file.isatty() and typer.confirm(f"Download {ref.size} to {dest}?")):
            pass
        else:
            error(
                f"{ref.display} is {ref.size}, over the {limit} download limit. "
                "Rerun with --yes or a higher --max-download-mb."
            )
            raise typer.Exit(1)

    dest_dir.mkdir(parents=True, exist_ok=True)
    if ref.size is not None:
        free = shutil.disk_usage(dest_dir).free
        if ref.size.bytes > free:
            error(f"not enough free space: need {ref.size}, have {ByteSize(free)} in {dest_dir}")
            raise typer.Exit(1)

    part = dest.with_suffix(dest.suffix + ".part")
    _fetch_to(uri, ref, part, aws_profile)
    part.replace(dest)
    console.print(f"[green]saved[/] {dest}")


def _child_uri(pointer: PrefixPointer, name: str) -> str:
    """Build the URI of one immediate child of a prefix.

    Parameters
    ----------
    pointer : PrefixPointer
        The parent prefix.
    name : str
        The child's name.

    Returns
    -------
    str
        The child's URI.
    """
    if pointer.kind == "s3":
        prefix = (
            pointer.prefix
            if pointer.prefix.endswith("/") or not pointer.prefix
            else (pointer.prefix + "/")
        )
        return f"s3://{pointer.bucket}/{prefix}{name}"
    return str(Path(pointer.prefix) / name)


def _do_download_prefix(
    pointer: PrefixPointer,
    output: Path | None,
    max_download_mb: float | None,
    aws_profile: str | None,
    yes: bool,
) -> None:
    """Download every file one level under a prefix, gating on the total size.

    The listing is delimited and capped (never recursive), and the gate is
    applied to the aggregate before the first byte moves — a wave domain
    prefix sums to multiple TB and is refused with guidance rather than
    started.

    Parameters
    ----------
    pointer : PrefixPointer
        The prefix to download from.
    output : Path or None
        Directory to download into, or ``None`` for the current directory.
    max_download_mb : float or None
        Download ceiling in megabytes for the aggregate, or ``None`` for the
        default.
    aws_profile : str or None
        AWS profile for signed S3 access. Anonymous when ``None``.
    yes : bool
        Skip confirmation prompts.

    Raises
    ------
    typer.Exit
        When there are no files, or the aggregate download is declined.
    """
    lister = make_lister(pointer, aws_profile)
    with _listing_status(pointer):
        listing = list_children(lister, pointer.prefix, 1000, None)

    files = [e for e in listing.entries if not e.is_dir]
    n_dirs = listing.n_dirs
    if n_dirs:
        console.print(
            f"[dim]skipping {n_dirs} subdirector{'y' if n_dirs == 1 else 'ies'}, "
            "only files one level down are downloaded[/]"
        )
    if not files:
        error(f"no files directly under {pointer.uri}")
        raise typer.Exit(1)

    total = ByteSize(listing.total_file_bytes)
    suffix = " (listing truncated, more exist)" if listing.truncated else ""
    console.print(f"{len(files)} file(s), {total} total{suffix}")

    limit = build_policy(max_download_mb=max_download_mb).max_download
    if total.bytes > limit.bytes:
        prompt_ok = yes or (
            console.file.isatty() and typer.confirm(f"Download {len(files)} files ({total})?")
        )
        if not prompt_ok:
            # A wave domain prefix is far over any sane limit, so point a
            # refused bulk request at the per-point path instead.
            hint = (
                "\n  For a point time series use 'mer wave LAT,LON' to fetch "
                "one grid node instead of whole files."
                if pointer.bucket == _WAVE_CONFIG.s3_bucket
                else ""
            )
            error(
                f"{pointer.uri} holds {total}, over the {limit} download limit. "
                f"Rerun with --yes or a higher --max-download-mb.{hint}"
            )
            raise typer.Exit(1)

    dest_dir = output or Path.cwd()
    for entry in files:
        dest = dest_dir / entry.name
        if dest.exists() and entry.size is not None and dest.stat().st_size == entry.size:
            console.print(f"[dim]exists[/] {dest}")
            continue
        # The aggregate was approved above, so each file downloads ungated.
        _do_download(_child_uri(pointer, entry.name), dest_dir, None, aws_profile, True)


def _fetch_to(uri: str, ref: Any, part: Path, aws_profile: str | None) -> None:
    """Stream a file to ``part`` with a progress bar, by scheme.

    Parameters
    ----------
    uri : str
        The file to fetch.
    ref : SourceRef
        The resolved source, carrying scheme and size.
    part : Path
        The partial file the bytes stream into.
    aws_profile : str or None
        AWS profile for signed S3 access. Anonymous when ``None``.
    """
    from rich.progress import (
        BarColumn,
        DownloadColumn,
        Progress,
        TransferSpeedColumn,
    )

    total = ref.size.bytes if ref.size is not None else None
    columns = ["[bright_blue]downloading", BarColumn(), DownloadColumn(), TransferSpeedColumn()]
    with Progress(*columns, console=console) as progress:
        task = progress.add_task("download", total=total)

        def advance(n: int) -> None:
            """Move the progress bar forward by ``n`` bytes.

            Parameters
            ----------
            n : int
                Bytes just written.
            """
            progress.update(task, advance=n)

        if ref.scheme == "s3":
            from ..catalog import make_client

            bucket, _, key = uri[len("s3://") :].partition("/")
            make_client(aws_profile).download_file(bucket, key, str(part), Callback=advance)
        elif ref.scheme in ("http", "https"):
            from ..lazy import lazy_import

            requests = lazy_import("requests", "downloading over HTTP(S)")
            with requests.get(uri, stream=True, timeout=60) as resp, open(part, "wb") as fh:
                resp.raise_for_status()
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    fh.write(chunk)
                    advance(len(chunk))
        else:
            src = Path(uri)
            shutil.copyfile(src, part)
            advance(total or src.stat().st_size)
