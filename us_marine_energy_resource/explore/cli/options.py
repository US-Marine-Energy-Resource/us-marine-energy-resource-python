"""Typer options for ``mer explore``, with help text defined once."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from .complete import complete_path as _complete_path

_MODE_PANEL = "Mode (choose one, default shows an overview)"
_BROWSE_PANEL = "Browse (datasets and s3:// prefixes)"
_STRUCTURE_PANEL = "Structure (--tree)"
_VALUES_PANEL = "Values (--head)"
_STATS_PANEL = "Statistics (--stats)"
_VOLUME_PANEL = "Volume limits (--head / --stats)"

UriArg = Annotated[
    str | None,
    typer.Argument(
        help="Path, [bright_blue]s3://bucket/key[/], or [bright_blue]https://...[/] URL.",
        autocompletion=_complete_path,
    ),
]

VariablesOpt = Annotated[
    bool,
    typer.Option(
        "--variables",
        help="Also read each variable's attributes from a remote HDF5 file. "
        "They are scattered through the file, so this can take minutes for a "
        "large file. Local files and parquet include them without this flag.",
    ),
]

# ----- mode flags (mutually exclusive; default is the overview) -----
InfoModeOpt = Annotated[
    bool,
    typer.Option("--info", help="Show only the header.", rich_help_panel=_MODE_PANEL),
]
TreeModeOpt = Annotated[
    bool,
    typer.Option("--tree", help="Show only the hierarchy.", rich_help_panel=_MODE_PANEL),
]
AttrsModeOpt = Annotated[
    bool,
    typer.Option("--attrs", help="Show attributes of --path node.", rich_help_panel=_MODE_PANEL),
]
HeadModeOpt = Annotated[
    bool,
    typer.Option("--head", help="Read values from --path.", rich_help_panel=_MODE_PANEL),
]
StatsModeOpt = Annotated[
    bool,
    typer.Option("--stats", help="Statistics for --path.", rich_help_panel=_MODE_PANEL),
]

JsonOpt = Annotated[
    bool,
    typer.Option("--json", help="Emit JSON instead of formatted text."),
]
PathOpt = Annotated[
    str | None,
    typer.Option(
        "--path", "-p", help="Node path inside the file, e.g. [bright_blue]/group/array[/]."
    ),
]
AwsProfileOpt = Annotated[
    str | None,
    typer.Option("--aws-profile", help="AWS profile for signed S3 access (anonymous otherwise)."),
]
DepthOpt = Annotated[
    int | None,
    typer.Option(
        "--depth", help="Limit the tree to this many levels.", rich_help_panel=_STRUCTURE_PANEL
    ),
]
LimitOpt = Annotated[
    int,
    typer.Option("--limit", help="Max entries listed per level.", rich_help_panel=_BROWSE_PANEL),
]
OutputOpt = Annotated[
    Path | None,
    typer.Option("--output", "-o", help="Directory to download into (default: current)."),
]
MaxDownloadOpt = Annotated[
    float | None,
    typer.Option("--max-download-mb", help="Refuse a download larger than this many MB."),
]
FilterOpt = Annotated[
    str | None,
    typer.Option(
        "--filter",
        help="Glob on names at the top level, e.g. [bright_blue]AK_*[/].",
        rich_help_panel=_BROWSE_PANEL,
    ),
]
StorageOpt = Annotated[
    bool,
    typer.Option(
        "--storage", help="Show chunks, compression, ratio.", rich_help_panel=_STRUCTURE_PANEL
    ),
]
NOpt = Annotated[
    int | None,
    typer.Option(
        "-n",
        help="Read the first N entries along axis 0. Excludes [bright_blue]--index[/].",
        rich_help_panel=_VALUES_PANEL,
    ),
]
IndexOpt = Annotated[
    str | None,
    typer.Option(
        "--index",
        help="Numpy-style slice, e.g. [bright_blue]0:5,::2[/]. Excludes [bright_blue]-n[/].",
        rich_help_panel=_VALUES_PANEL,
    ),
]
DecodeOpt = Annotated[
    str,
    typer.Option(
        "--decode",
        help="[bright_blue]none[/] raw, [bright_blue]cf[/] value*scale+offset, "
        "[bright_blue]rex[/] value/scale.",
        rich_help_panel=_VALUES_PANEL,
    ),
]
MaxElementsOpt = Annotated[
    int,
    typer.Option("--max-elements", help="Sampling budget for stats.", rich_help_panel=_STATS_PANEL),
]
ExactOpt = Annotated[
    bool,
    typer.Option("--exact", help="Read the whole array (gated).", rich_help_panel=_STATS_PANEL),
]
MaxTransferOpt = Annotated[
    float | None,
    typer.Option(
        "--max-transfer-mb", help="Network transfer ceiling.", rich_help_panel=_VOLUME_PANEL
    ),
]
MaxMemoryOpt = Annotated[
    float | None,
    typer.Option("--max-memory-mb", help="Per-read memory ceiling.", rich_help_panel=_VOLUME_PANEL),
]
DryRunOpt = Annotated[
    bool,
    typer.Option("--dry-run", help="Show the read plan and exit.", rich_help_panel=_VOLUME_PANEL),
]
YesOpt = Annotated[
    bool,
    typer.Option("--yes", "-y", help="Skip confirmation prompts.", rich_help_panel=_VOLUME_PANEL),
]
