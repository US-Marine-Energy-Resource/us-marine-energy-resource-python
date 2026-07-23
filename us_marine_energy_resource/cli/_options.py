"""Dataclasses and Annotated type aliases for the us-tidal CLI."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

if TYPE_CHECKING:
    from ..cache import S3CacheManager
    from ..manifest import TidalManifestQuery

try:
    import tomllib  # pyright: ignore[reportMissingImports]
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ConnectionConfig:
    """Infrastructure settings — how to connect and where to cache.

    Populated from ``~/.us_tidal.toml`` (or a custom config file), then
    overridden by any CLI flags the user provides at runtime.
    """

    aws_profile: str | None = None
    cache_dir: Path | None = None
    s3_bucket: str = "marine-energy-data"
    s3_prefix: str = "us-tidal"
    use_hpc: bool = False
    hpc_base_path: str | None = None
    clear_cache: bool = False

    @classmethod
    def from_config_file(cls, path: Path | None = None) -> ConnectionConfig:
        """Load settings from a TOML config file.

        Looks for ``~/.us_tidal.toml`` unless *path* is given.
        Returns a default ``ConnectionConfig`` when no file exists.
        """
        config_path = path or Path.home() / ".us_tidal.toml"
        if not config_path.exists():
            return cls()
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
        return cls(
            aws_profile=data.get("aws_profile"),
            cache_dir=Path(data["cache_dir"]) if "cache_dir" in data else None,
            s3_bucket=data.get("s3_bucket", "marine-energy-data"),
            s3_prefix=data.get("s3_prefix", "us-tidal"),
            use_hpc=data.get("use_hpc", False),
            hpc_base_path=data.get("hpc_base_path"),
        )

    def with_cli_overrides(
        self,
        *,
        aws_profile: str | None,
        cache_dir: Path | None,
        use_hpc: bool,
        hpc_base_path: str | None,
        clear_cache: bool,
    ) -> ConnectionConfig:
        """Return a new ``ConnectionConfig`` with CLI-provided values merged in.

        Non-None CLI values override the config file. Boolean flags are OR'd
        (CLI ``--use-hpc`` can enable HPC mode even if the config says False,
        but cannot disable it if the config says True).
        """
        return dataclasses.replace(
            self,
            aws_profile=aws_profile or self.aws_profile,
            cache_dir=cache_dir or self.cache_dir,
            use_hpc=use_hpc or self.use_hpc,
            hpc_base_path=hpc_base_path or self.hpc_base_path,
            clear_cache=clear_cache or self.clear_cache,
        )

    def create_session(self) -> Session:
        """Initialise ``S3CacheManager`` + ``TidalManifestQuery``.

        This is the only place that touches the network. It is called once per
        CLI invocation, after all geometry validation has passed.
        """
        from ..cache import S3CacheManager
        from ..config import config as _defaults
        from ..manifest import TidalManifestQuery, find_latest_manifest_hpc, find_latest_manifest_s3
        from ._display import console, error

        if self.use_hpc:
            base = self.hpc_base_path or _defaults["storage"]["hpc_base_path"]
            with console.status("[bright_blue]Locating HPC manifest…"):
                manifest_path = find_latest_manifest_hpc(base)
            if manifest_path is None:
                error(f"No manifest found on HPC at {base!r}")
                raise typer.Exit(1)
            return Session(cache=None, query=TidalManifestQuery(manifest_path), conn=self)

        effective_cache = (self.cache_dir or Path.home() / ".us_tidal_cache") / self.s3_bucket
        cache = S3CacheManager(
            bucket=self.s3_bucket,
            prefix=self.s3_prefix,
            cache_dir=effective_cache,
            aws_profile=self.aws_profile,
        )

        if self.clear_cache:
            with console.status("[bright_blue]Clearing cache…"):
                cache.clear_cache()

        with console.status("[bright_blue]Locating manifest…"):
            result = find_latest_manifest_s3(cache)

        if result is None:
            error(f"No manifest found at s3://{self.s3_bucket}/{self.s3_prefix}/manifest/")
            raise typer.Exit(1)

        manifest_path, _ = result
        return Session(
            cache=cache,
            query=TidalManifestQuery(manifest_path, s3_cache=cache),
            conn=self,
        )


@dataclasses.dataclass(frozen=True)
class QueryOptions:
    """Per-query output options."""

    info_mode: bool = False
    info_categories: tuple[str, ...] | None = None
    layers: tuple[int, ...] = (0,)
    depth_target: float | None = None
    depth_avg: bool = False
    dry_run: bool = False
    max_size_mb: float = 500.0
    output_dir: Path | None = None
    max_distance_km: float | None = None
    csv_output: bool = False


@dataclasses.dataclass(frozen=True)
class Session:
    """Initialised connection — S3 cache + manifest query handle."""

    cache: S3CacheManager | None
    query: TidalManifestQuery
    conn: ConnectionConfig


# ---------------------------------------------------------------------------
# Annotated type aliases — help text and defaults defined exactly once
# ---------------------------------------------------------------------------

_INFO_PANEL = "Dataset Info"

InfoOpt = Annotated[
    bool,
    typer.Option(
        "--info",
        help=(
            "Show dataset metadata, schema, and statistics without downloading. "
            "Reads only the parquet footer (fast range requests)."
        ),
        rich_help_panel=_INFO_PANEL,
    ),
]
InfoSpeedOpt = Annotated[
    bool,
    typer.Option(
        "--info-speed",
        help="Show speed category info only (implies [bright_blue]--info[/]).",
        rich_help_panel=_INFO_PANEL,
    ),
]
InfoDirectionOpt = Annotated[
    bool,
    typer.Option(
        "--info-direction",
        help="Show direction category info only (implies [bright_blue]--info[/]).",
        rich_help_panel=_INFO_PANEL,
    ),
]
InfoPowerOpt = Annotated[
    bool,
    typer.Option(
        "--info-power",
        help="Show power density category info only (implies [bright_blue]--info[/]).",
        rich_help_panel=_INFO_PANEL,
    ),
]
InfoDepthOpt = Annotated[
    bool,
    typer.Option(
        "--info-depth",
        help="Show depth/water-level category info only (implies [bright_blue]--info[/]).",
        rich_help_panel=_INFO_PANEL,
    ),
]
LayerOpt = Annotated[
    list[int] | None,
    typer.Option(
        "--layer",
        help=(
            "Sigma layer for [bright_blue]--info[/] statistics (0=surface, 9=near-bed). "
            "Repeat to select multiple layers."
        ),
        rich_help_panel=_INFO_PANEL,
    ),
]
DepthTargetOpt = Annotated[
    float | None,
    typer.Option(
        "--depth",
        help=(
            "Select the sigma layer nearest to this depth (m from surface) for "
            "[bright_blue]--info[/] statistics. Approximate — uses footer depth stats."
        ),
        rich_help_panel=_INFO_PANEL,
    ),
]
DepthAvgOpt = Annotated[
    bool,
    typer.Option(
        "--depth-avg",
        help="Average [bright_blue]--info[/] statistics across all sigma layers.",
        rich_help_panel=_INFO_PANEL,
    ),
]
DryRun = Annotated[
    bool,
    typer.Option("--dry-run", help="Show size estimate without downloading."),
]
MaxSizeMb = Annotated[
    float,
    typer.Option(
        "--max-size-mb",
        envvar="US_TIDAL_MAX_SIZE_MB",
        help="Abort if uncached data to download exceeds this limit (MB). 0 = no limit.",
    ),
]
OutputDir = Annotated[
    Path | None,
    typer.Option("--output-dir", "-o", help="Copy downloaded parquet files to this directory."),
]
MaxDistKm = Annotated[
    float | None,
    typer.Option(
        "--max-distance-km",
        help="Reject if nearest face is farther than this (km). Point queries only.",
    ),
]
AwsProfileOpt = Annotated[
    str | None,
    typer.Option("--aws-profile", help="Override AWS profile from config."),
]
CacheDirOpt = Annotated[
    Path | None,
    typer.Option("--cache-dir", help="Override local cache directory from config."),
]
ConfigFileOpt = Annotated[
    Path | None,
    typer.Option("--config", help="Path to config file (default: ~/.us_tidal.toml)."),
]
UseHpcOpt = Annotated[
    bool,
    typer.Option("--use-hpc", help="Use HPC local filesystem instead of S3."),
]
HpcBasePathOpt = Annotated[
    str | None,
    typer.Option("--hpc-base-path", help="Override HPC dataset root path from config."),
]
ClearCacheOpt = Annotated[
    bool,
    typer.Option("--clear-cache", help="Clear the local cache before running."),
]
CsvOpt = Annotated[
    bool,
    typer.Option(
        "--csv",
        help=(
            "Export downloaded data as CSV files. "
            "Written to [bright_blue]--output-dir[/] if set, otherwise to the current directory."
        ),
    ),
]
