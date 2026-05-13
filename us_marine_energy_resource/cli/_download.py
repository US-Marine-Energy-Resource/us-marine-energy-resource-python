"""Download orchestration for the us-tidal CLI."""

from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

if TYPE_CHECKING:
    from ..cache import S3CacheManager
    from ..manifest import TidalManifestQuery

from ._display import console, error


def point_file_path(result: dict[str, Any]) -> str:
    """Return the parquet file path for a point query result."""
    return result["point"]["file_path"]


def multi_face_file_paths(
    results: list[dict[str, Any]],
    query: TidalManifestQuery,
) -> list[str]:
    """Return parquet file paths for a list of multi-face query results."""
    return [query.get_file_path(r) for r in results]


_SAMPLE_SIZE = 5


def estimate_download(
    paths_by_location: dict[str, list[str]],
    cache: S3CacheManager,
) -> tuple[float, float, float, bool]:
    """Return (total_mb, cached_mb, to_download_mb, is_estimate).

    Processes each location group independently — file sizes differ between
    locations (hourly vs half-hourly data). Cached files use local stat (no
    network). Uncached files are estimated by sampling up to _SAMPLE_SIZE
    paths per location and extrapolating the average to the full group.
    is_estimate is True when any location had more uncached files than the
    sample, meaning sizes were extrapolated rather than measured exactly.
    """
    total_cached_bytes = 0
    total_uncached_bytes = 0
    is_estimate = False

    for paths in paths_by_location.values():
        cached = [p for p in paths if cache.is_cached(p)]
        uncached = [p for p in paths if not cache.is_cached(p)]

        total_cached_bytes += sum((cache.cache_dir / p).stat().st_size for p in cached)

        if uncached:
            sample = uncached[:_SAMPLE_SIZE]
            sizes = cache.estimate_sizes(sample, max_workers=len(sample))
            if sizes:
                avg_bytes = sum(sizes.values()) / len(sizes)
                total_uncached_bytes += int(avg_bytes * len(uncached))
            if len(uncached) > len(sample):
                is_estimate = True

    total_bytes = total_cached_bytes + total_uncached_bytes
    mb = 1024 * 1024
    return (
        total_bytes / mb,
        total_cached_bytes / mb,
        total_uncached_bytes / mb,
        is_estimate,
    )


def check_size_limit(
    paths_by_location: dict[str, list[str]],
    cache: S3CacheManager,
    max_size_mb: float,
) -> tuple[float, float, float, bool]:
    """Estimate download size and abort via typer.Exit if the limit is exceeded.

    Returns (total_mb, cached_mb, to_download_mb, is_estimate) when within the limit.
    """
    total_mb, cached_mb, to_dl_mb, is_estimate = estimate_download(paths_by_location, cache)
    if max_size_mb > 0 and to_dl_mb > max_size_mb:
        error(
            f"{to_dl_mb:.1f} MB to download exceeds --max-size-mb {max_size_mb:.0f} MB. "
            f"Use --dry-run to see a breakdown, or increase --max-size-mb."
        )
        raise typer.Exit(1)
    return total_mb, cached_mb, to_dl_mb, is_estimate


def download_with_progress(
    paths: list[str],
    cache: S3CacheManager,
    max_workers: int = 4,
) -> dict[str, Path]:
    """Download files in parallel with a Rich progress bar."""
    downloaded: dict[str, Path] = {}
    label = f"Downloading {len(paths)} file{'s' if len(paths) != 1 else ''}"

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"[cyan]{label}[/]", total=len(paths))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_path = {pool.submit(cache.get, p): p for p in paths}
            for future in as_completed(future_to_path):
                rel = future_to_path[future]
                local = future.result()
                downloaded[rel] = local
                progress.advance(task)

    return downloaded


def copy_to_output_dir(downloaded: dict[str, Path], output_dir: Path) -> None:
    """Copy downloaded parquet files to the user-specified output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for local_path in downloaded.values():
        shutil.copy2(local_path, output_dir / local_path.name)


def parquet_to_csv(downloaded: dict[str, Path], output_dir: Path) -> list[Path]:
    """Convert downloaded parquet files to CSV and write them to output_dir.

    Returns the list of CSV paths written.
    """
    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for local_path in downloaded.values():
        df = pd.read_parquet(local_path)
        csv_path = output_dir / local_path.with_suffix(".csv").name
        df.to_csv(csv_path, index=False)
        written.append(csv_path)
    return written
