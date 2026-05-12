"""Shared post-query execution: download, display, and output handling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from ._display import (
    cache_location,
    console,
    dataset_info_panel,
    error,
    faces_table,
    point_result,
    size_estimate,
    stats_table,
)
from ._download import (
    check_size_limit,
    copy_to_output_dir,
    download_with_progress,
    estimate_download,
    multi_face_file_paths,
    parquet_to_csv,
    point_file_path,
)
from ._options import QueryOptions, Session


def run_info(
    session: Session,
    opts: QueryOptions,
    file_paths: list[str],
) -> None:
    """Fetch parquet footer info and render the --info display.

    Uses range requests to read only the parquet footer — no full download.
    For multiple files the footers are fetched in parallel and stats are
    aggregated across all files.
    """
    footer_infos: list[dict[str, Any]] = []

    if session.conn.use_hpc:
        from ..analysis.preprocessing import read_parquet_footer_info

        for fp in file_paths:
            local_path = Path(session.query.get_hpc_path(fp))
            if local_path.exists():
                footer_infos.append(dict(read_parquet_footer_info(local_path)))
    else:
        if session.cache is None:
            error("No S3 cache available — cannot fetch dataset info.")
            raise typer.Exit(1)
        fetched = session.cache.get_many_parquet_footer_infos(file_paths)
        footer_infos = [fetched[p] for p in file_paths if p in fetched]

    if not footer_infos:
        error("Could not retrieve footer info for any matched files.")
        raise typer.Exit(1)

    dataset_info_panel(
        footer_infos=footer_infos,
        filter_categories=opts.info_categories,
        layers=list(opts.layers),
        depth_target=opts.depth_target,
        depth_avg=opts.depth_avg,
        n_files=len(file_paths),
    )


def run_point(
    session: Session,
    opts: QueryOptions,
    lat: float,
    lon: float,
) -> None:
    """Execute a point query then download and display results."""
    with console.status("[cyan]Searching for nearest face…"):
        result = session.query.query_nearest_point(lat, lon)

    if result is None:
        error(
            f"No grid face found near ({lat}, {lon}). "
            "Coordinate may be outside all dataset domains."
        )
        raise typer.Exit(1)

    if opts.max_distance_km is not None and result["distance_km"] > opts.max_distance_km:
        error(
            f"Nearest face is {result['distance_km']:.2f} km away, "
            f"which exceeds --max-distance-km {opts.max_distance_km}."
        )
        raise typer.Exit(1)

    point_result(result)

    if opts.info_mode:
        file_path = point_file_path(result)
        with console.status("[cyan]Fetching dataset info…"):
            run_info(session, opts, [file_path])
        return

    file_path = point_file_path(result)
    paths_by_location = {result["location"]: [file_path]}

    if opts.dry_run:
        if session.cache is not None:
            total_mb, cached_mb, to_dl_mb, is_estimate = estimate_download(
                paths_by_location, session.cache
            )
            size_estimate(1, total_mb, cached_mb, to_dl_mb, is_estimate)
        return

    if session.conn.use_hpc:
        local_path = Path(session.query.get_hpc_path(file_path))
        if not local_path.exists():
            error(f"HPC file not found: {local_path}")
            raise typer.Exit(1)
    else:
        if session.cache is None:
            error("No S3 cache available — cannot download.")
            raise typer.Exit(1)
        check_size_limit(paths_by_location, session.cache, opts.max_size_mb)
        with console.status("[cyan]Downloading…"):
            local_path = session.cache.get(file_path)

    stats_table(local_path)
    _finalize({file_path: local_path}, session, opts)


def run_multi(
    results: list[dict[str, Any]],
    session: Session,
    opts: QueryOptions,
) -> None:
    """Download and display results for line / bbox / polygon queries."""
    if not results:
        console.print("[yellow]No matching grid faces found.[/]")
        raise typer.Exit(0)

    faces_table(results)

    if opts.info_mode:
        file_paths = multi_face_file_paths(results, session.query)
        with console.status("[cyan]Fetching dataset info…"):
            run_info(session, opts, file_paths)
        return

    file_paths = multi_face_file_paths(results, session.query)

    paths_by_location: dict[str, list[str]] = {}
    for r, fp in zip(results, file_paths):
        paths_by_location.setdefault(r["location"], []).append(fp)

    if opts.dry_run:
        if session.cache is not None:
            total_mb, cached_mb, to_dl_mb, is_estimate = estimate_download(
                paths_by_location, session.cache
            )
            size_estimate(len(file_paths), total_mb, cached_mb, to_dl_mb, is_estimate)
        return

    if session.conn.use_hpc:
        downloaded = {
            p: Path(session.query.get_hpc_path(p))
            for p in file_paths
            if Path(session.query.get_hpc_path(p)).exists()
        }
    else:
        if session.cache is None:
            error("No S3 cache available — cannot download.")
            raise typer.Exit(1)
        check_size_limit(paths_by_location, session.cache, opts.max_size_mb)
        downloaded = download_with_progress(file_paths, session.cache)

    _finalize(downloaded, session, opts)


def _finalize(downloaded: dict[str, Path], session: Session, opts: QueryOptions) -> None:
    """Copy to output_dir if given; convert to CSV if requested."""
    if opts.output_dir is not None:
        copy_to_output_dir(downloaded, opts.output_dir)
        n = len(downloaded)
        console.print(
            f"\n  [bold green]✓[/]  {n} file{'s' if n != 1 else ''} "
            f"saved to [cyan]{opts.output_dir}[/]"
        )
    elif session.cache is not None:
        cache_location(session.cache.cache_dir, len(downloaded))

    if opts.csv_output:
        csv_dir = opts.output_dir or Path.cwd()
        written = parquet_to_csv(downloaded, csv_dir)
        cwd = Path.cwd()
        for csv_path in written:
            try:
                display = csv_path.relative_to(cwd)
            except ValueError:
                display = csv_path
            console.print(f"\n  [bold green]✓[/]  CSV saved: [cyan]{display}[/]")
