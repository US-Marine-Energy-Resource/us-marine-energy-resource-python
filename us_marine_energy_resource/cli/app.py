"""us-tidal — query and download the US DOE High Resolution Tidal Hindcast dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
import typer.rich_utils

from ._display import console, error
from ._geometry import parse_bbox, parse_geojson_file, parse_point, parse_wkt
from ._links import DATASET_CITATION, DOCS, GEOJSON_TOOL, S3_BROWSER, _link
from ._options import (
    AwsProfileOpt,
    CacheDirOpt,
    ClearCacheOpt,
    ConfigFileOpt,
    ConnectionConfig,
    CsvOpt,
    DepthAvgOpt,
    DepthTargetOpt,
    DryRun,
    HpcBasePathOpt,
    InfoDepthOpt,
    InfoDirectionOpt,
    InfoOpt,
    InfoPowerOpt,
    InfoSpeedOpt,
    LayerOpt,
    MaxDistKm,
    MaxSizeMb,
    OutputDir,
    QueryOptions,
    UseHpcOpt,
)
from ._run import run_multi, run_point

# Blank out Typer's default help-text styling and pin its help colors to the
# CLI palette (white, blue, and green, with bold only on blue and green).
# Typer's defaults use cyan, yellow, magenta, and red. Set at import time,
# before any command renders.
typer.rich_utils.STYLE_HELPTEXT = ""
typer.rich_utils.STYLE_OPTION = "bold bright_blue"
typer.rich_utils.STYLE_SWITCH = "bold green"
typer.rich_utils.STYLE_NEGATIVE_OPTION = "bright_blue"
typer.rich_utils.STYLE_NEGATIVE_SWITCH = "green"
# typer 0.27 renamed STYLE_METAVAR* to STYLE_TYPES* (fastapi/typer#1863).
# Set both spellings via setattr so the palette holds on either side of the
# rename and the type checker accepts whichever name the installed typer lacks.
for _name, _style in (
    ("STYLE_METAVAR", "bright_blue"),
    ("STYLE_TYPES", "bright_blue"),
    ("STYLE_TYPES_SEPARATOR", "dim bright_blue"),
):
    setattr(typer.rich_utils, _name, _style)
typer.rich_utils.STYLE_USAGE = "bright_blue"
typer.rich_utils.STYLE_USAGE_COMMAND = "bold bright_blue"
typer.rich_utils.STYLE_DEPRECATED = "bold bright_blue"
typer.rich_utils.STYLE_REQUIRED_SHORT = "bold bright_blue"
typer.rich_utils.STYLE_REQUIRED_LONG = "dim bright_blue"
typer.rich_utils.STYLE_OPTION_ENVVAR = "dim green"
typer.rich_utils.STYLE_COMMANDS_TABLE_FIRST_COLUMN = "bold green"
typer.rich_utils.STYLE_ERRORS_PANEL_BORDER = "bright_blue"
typer.rich_utils.STYLE_ABORTED = "bright_blue"

_MAIN_HELP = (
    "Query and download modeled tidal current data from the U.S. DOE H2O High"
    " Resolution Tidal Hindcast, FVCOM simulations covering five U.S. coastal"
    " regions: Cook Inlet AK, Aleutian Islands AK, Salish Sea WA,"
    " Piscataqua River NH, and Western Passage ME.\n\n"
    "A point query returns the mesh face containing the coordinate."
    " Area and transect queries return all faces whose triangles"
    " geometrically intersect the specified geometry."
    " Each matched face downloads as a full-year, hourly or half-hourly"
    " time series of current speed, direction, and kinetic power density"
    " at 10 depth layers (sea surface to seafloor).\n\n"
    f"Dataset citation: {_link(DATASET_CITATION)}\n"
    f"Documentation:    {_link(DOCS)}\n"
    f"AWS S3 browser:   {_link(S3_BROWSER)}\n\n"
    "Provide [bright_blue]exactly one[/] geometry input: a positional [bright_blue]lat,lon[/]"
    " for\na point query, or one of [bright_blue]--coord[/], [bright_blue]--bbox[/],"
    " [bright_blue]--file[/],\nor [bright_blue]--wkt[/] for area queries."
)

tidal_app = typer.Typer(
    name="tidal",
    rich_markup_mode="rich",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
    add_completion=True,
)

_TIDAL_EPILOG = (
    "[bright_blue]Examples[/bright_blue]\n\n"
    "mer tidal 60.73,-151.43                              Point query\n\n"
    "mer tidal --coord 60.7,-151.4 --coord 60.9,-151.2   Transect\n\n"
    "mer tidal --bbox 60.7,-151.5,60.9,-151.2            Bounding box\n\n"
    "mer tidal --file study_area.geojson                  Polygon from file\n\n"
    'mer tidal --wkt "POLYGON((-151.5 60.7,...))"         Polygon from WKT\n\n'
    "mer tidal 60.73,-151.43 --dry-run                    Size estimate\n\n"
    "mer tidal 60.73,-151.43 --info                       Dataset info (no download)\n\n"
    "mer tidal 60.73,-151.43 --info --layer 3             Layer 3 stats\n\n"
    "mer tidal 60.73,-151.43 --output-dir ./data          Save parquet files\n\n"
    "mer tidal 60.73,-151.43 --csv                        Export CSV to current dir\n\n"
    "Config file ([italic]~/.us_tidal.toml[/italic]) sets defaults for AWS, cache, and HPC options."
)


def main(
    # ----- Geometry (exactly one required) -----
    location: Annotated[
        str | None,
        typer.Argument(
            help="Point as [bright_blue]lat,lon[/] (e.g. [bright_blue]60.73,-151.43[/])."
        ),
    ] = None,
    coord: Annotated[
        list[str] | None,
        typer.Option(
            "--coord",
            "-c",
            help="Transect waypoint as [bright_blue]lat,lon[/]. Repeat for multi-segment lines.",
        ),
    ] = None,
    bbox: Annotated[
        str | None,
        typer.Option(
            "--bbox", help="Bounding box as [bright_blue]lat_min,lon_min,lat_max,lon_max[/]."
        ),
    ] = None,
    file: Annotated[
        Path | None,
        typer.Option(
            "--file", "-f", help=f"Polygon from a GeoJSON file. Draw one at {_link(GEOJSON_TOOL)}."
        ),
    ] = None,
    wkt: Annotated[
        str | None,
        typer.Option("--wkt", help="Polygon as a WKT POLYGON string or path to a .wkt file."),
    ] = None,
    # ----- Output -----
    output_dir: OutputDir = None,
    csv_output: CsvOpt = False,
    dry_run: DryRun = False,
    max_size_mb: MaxSizeMb = 500.0,
    max_distance_km: MaxDistKm = None,
    # ----- Dataset info -----
    info: InfoOpt = False,
    info_speed: InfoSpeedOpt = False,
    info_direction: InfoDirectionOpt = False,
    info_power: InfoPowerOpt = False,
    info_depth: InfoDepthOpt = False,
    layer: LayerOpt = None,
    depth: DepthTargetOpt = None,
    depth_avg: DepthAvgOpt = False,
    # ----- Advanced (override config file) -----
    config_file: ConfigFileOpt = None,
    aws_profile: AwsProfileOpt = None,
    cache_dir: CacheDirOpt = None,
    use_hpc: UseHpcOpt = False,
    hpc_base_path: HpcBasePathOpt = None,
    clear_cache: ClearCacheOpt = False,
) -> None:
    """Query the tidal hindcast for one geometry and print or download the result."""
    # -- validate geometry (exactly one active) --
    geometry_sources = {
        "lat,lon (positional)": location is not None,
        "--coord": bool(coord),
        "--bbox": bbox is not None,
        "--file": file is not None,
        "--wkt": wkt is not None,
    }
    active = [name for name, present in geometry_sources.items() if present]

    if not active:
        error(
            "No geometry provided. Supply a positional lat,lon, "
            "or one of --coord, --bbox, --file, --wkt."
        )
        raise typer.Exit(1)

    if len(active) > 1:
        error(f"Multiple geometry inputs given: {', '.join(active)}. Provide exactly one.")
        raise typer.Exit(1)

    # -- build connection config (file → CLI overrides) --
    conn = ConnectionConfig.from_config_file(config_file).with_cli_overrides(
        aws_profile=aws_profile,
        cache_dir=cache_dir,
        use_hpc=use_hpc,
        hpc_base_path=hpc_base_path,
        clear_cache=clear_cache,
    )

    # Resolve info mode and category filter
    info_mode = info or info_speed or info_direction or info_power or info_depth
    if info_mode and not info:
        cats: list[str] = []
        if info_speed:
            cats.append("speed")
        if info_direction:
            cats.append("direction")
        if info_power:
            cats.append("power")
        if info_depth:
            cats.append("depth")
        info_categories: tuple[str, ...] | None = tuple(cats) if cats else None
    else:
        info_categories = None  # all categories

    opts = QueryOptions(
        info_mode=info_mode,
        info_categories=info_categories,
        layers=tuple(layer) if layer else (0,),
        depth_target=depth,
        depth_avg=depth_avg,
        dry_run=dry_run,
        max_size_mb=max_size_mb,
        output_dir=output_dir,
        max_distance_km=max_distance_km,
        csv_output=csv_output,
    )

    # -- parse geometry and dispatch --
    try:
        if location is not None:
            lat, lon = parse_point(location)
            session = conn.create_session()
            run_point(session, opts, lat, lon)

        elif coord:
            waypoints = [parse_point(c) for c in coord]
            if len(waypoints) < 2:
                error("Transect requires at least 2 --coord values.")
                raise typer.Exit(1)
            session = conn.create_session()
            with console.status("[bright_blue]Searching for intersecting faces…"):
                results = session.query.query_all_on_path(waypoints)
            run_multi(results, session, opts)

        elif bbox is not None:
            polygon = parse_bbox(bbox)
            session = conn.create_session()
            with console.status("[bright_blue]Searching for faces in bounding box…"):
                results = session.query.query_all_within_polygon(polygon)
            run_multi(results, session, opts)

        elif file is not None:
            if not file.exists():
                error(f"File not found: {file}")
                raise typer.Exit(1)
            polygon = parse_geojson_file(file)
            session = conn.create_session()
            with console.status("[bright_blue]Searching for faces within polygon…"):
                results = session.query.query_all_within_polygon(polygon)
            run_multi(results, session, opts)

        elif wkt is not None:
            polygon = parse_wkt(wkt)
            session = conn.create_session()
            with console.status("[bright_blue]Searching for faces within polygon…"):
                results = session.query.query_all_within_polygon(polygon)
            run_multi(results, session, opts)

    except ValueError as exc:
        error(str(exc))
        raise typer.Exit(1) from exc


# Single-command app that runs the query without naming a subcommand, for the
# standalone ``us-tidal`` console script. ``mer`` registers ``main`` directly as
# its ``tidal`` leaf command (see mer.py), so the query stays positional there.
tidal_app.command(help=_MAIN_HELP, epilog=_TIDAL_EPILOG)(main)

# Backward-compatible alias.
app = tidal_app


# TODO(deprecate): Deprecate after v0.7
def us_tidal_entry() -> None:
    """Run the tidal query under the deprecated ``us-tidal`` name."""
    console.print(
        "[dim]note: 'us-tidal' is now 'mer tidal'; this alias will be removed in a"
        " future release[/dim]"
    )
    tidal_app()
