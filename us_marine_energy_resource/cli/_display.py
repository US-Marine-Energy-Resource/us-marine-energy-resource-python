"""Rich display helpers for the us-tidal CLI."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich import box
from rich.console import Console
from rich.highlighter import NullHighlighter
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

if TYPE_CHECKING:
    import pandas as pd

# The CLI palette: white, blue, and green, with bold only on blue and green.
# Blue is the bright variant because standard ANSI blue is nearly unreadable
# on dark terminals. Rich's defaults color paths magenta, progress bars pink,
# and table headers bold white, so every default style that can show up in
# our output is pinned to the palette here.
# tests/unit/test_cli_palette.py checks rendered output against this rule.
PALETTE = Theme(
    {
        "rule.line": "bright_blue",
        "bar.complete": "bright_blue",
        "bar.finished": "green",
        "bar.pulse": "bright_blue",
        "progress.percentage": "green",
        "progress.download": "green",
        "progress.data.speed": "bright_blue",
        "progress.remaining": "bright_blue",
        "progress.elapsed": "bright_blue",
        "progress.spinner": "green",
        "status.spinner": "green",
        "table.header": "bright_blue",
        "table.footer": "bright_blue",
        "prompt.choices": "green",
        "prompt.default": "bright_blue",
    }
)

# No automatic highlighting: rich would otherwise color every number, path,
# and date inside prose, which turns a plain sentence into a patchwork.
# Color appears only where a style is set deliberately.
console = Console(theme=PALETTE, highlighter=NullHighlighter())


def _df_to_rich_table(
    df: pd.DataFrame,
    title: str = "",
    dim_cols: set[str] | None = None,
) -> Table:
    """Convert a pandas DataFrame to a Rich Table for terminal display."""
    import pandas as pd

    _dim = dim_cols or set()
    table = Table(
        title=title or None,
        box=box.SIMPLE,
        show_header=True,
        header_style="bright_blue",
        padding=(0, 1),
    )
    for col in df.columns:
        justify = "right" if pd.api.types.is_numeric_dtype(df[col]) else "left"
        table.add_column(str(col), justify=justify, style="dim" if col in _dim else None)
    for row in df.itertuples(index=False):
        table.add_row(*[str(v) for v in row])
    return table


def header(title: str, subtitle: str = "") -> None:
    """Print a header panel for a query."""
    body = Text(subtitle, style="dim") if subtitle else Text("")
    console.print(
        Panel(body, title=f"[bold bright_blue]us-tidal[/] · {title}", box=box.ROUNDED, expand=False)
    )


def point_result(result: dict[str, Any]) -> None:
    """Print metadata for a single-point query result."""
    from ._links import _link, http_url, s3_uri

    p = result["point"]
    dist = result["distance_km"]
    dist_label = "0.00 km (containing cell)" if dist == 0.0 else f"{dist:.4f} km"
    rel = p["file_path"]

    rows = [
        {"field": "face_id", "value": p["face_id"]},
        {"field": "location", "value": result["location"]},
        {"field": "latitude", "value": str(p["lat"])},
        {"field": "longitude", "value": str(p["lon"])},
        {"field": "distance", "value": dist_label},
        {"field": "file", "value": rel},
        {"field": "s3", "value": s3_uri(rel)},
        {"field": "url", "value": _link(http_url(rel))},
    ]

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(style="dim", no_wrap=True)
    table.add_column()
    for row in rows:
        table.add_row(row["field"], row["value"])
    console.print(table)


def faces_table(results: list[dict[str, Any]], max_rows: int = 20) -> None:
    """Print a summary and table of matched mesh faces."""
    import pandas as pd

    locations = sorted({r.get("location", "?") for r in results})
    loc_str = "  ·  ".join(f"[bright_blue]{loc}[/]" for loc in locations)
    console.print(f"\n  Matched [bold green]{len(results):,}[/] faces  ·  {loc_str}")

    rows = []
    for r in results[:max_rows]:
        lat, lon = r["centroid"]
        rows.append(
            {
                "face_id": r["face_id"],
                "location": r["location"],
                "lat": round(lat, 5),
                "lon": round(lon, 5),
                "dist_km": round(r.get("distance_km", 0.0), 2),
            }
        )

    df = pd.DataFrame(rows)
    console.print(_df_to_rich_table(df, dim_cols={"face_id"}))

    if len(results) > max_rows:
        console.print(f"  [dim]… and {len(results) - max_rows:,} more[/]")


def size_estimate(
    n_files: int,
    total_mb: float,
    cached_mb: float,
    to_download_mb: float,
    is_estimate: bool = False,
) -> None:
    """Print a dry-run size estimate."""
    import pandas as pd

    tilde = "~" if is_estimate else ""
    df = pd.DataFrame(
        [
            {"metric": "Files matched", "value": str(n_files)},
            {"metric": "Total size", "value": f"{tilde}{total_mb:.1f} MB"},
            {"metric": "Already cached", "value": f"{cached_mb:.1f} MB"},
            {"metric": "To download", "value": f"{tilde}{to_download_mb:.1f} MB"},
        ]
    )
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim")
    table.add_column(justify="right")
    for row in df.itertuples(index=False):
        table.add_row(row.metric, row.value)
    console.print(table)


def stats_table(local_path: Path, label: str = "Statistics  (surface layer)") -> None:
    """Load a parquet file and print a speed + power density summary table."""
    import pandas as pd

    df = pd.read_parquet(local_path)
    speed_col = "vap_sea_water_speed_layer_0"
    power_col = "vap_sea_water_power_density_layer_0"

    if speed_col not in df.columns:
        return

    rows: list[dict[str, Any]] = []
    s = df[speed_col]
    rows.append(
        {
            "metric": "Speed (m/s)",
            "mean": round(float(s.mean()), 3),
            "p90": round(float(s.quantile(0.9)), 3),
            "max": round(float(s.max()), 3),
        }
    )
    if power_col in df.columns:
        p = df[power_col]
        rows.append(
            {
                "metric": "Power density (W/m²)",
                "mean": round(float(p.mean()), 1),
                "p90": round(float(p.quantile(0.9)), 1),
                "max": round(float(p.max()), 1),
            }
        )

    stats_df = pd.DataFrame(rows)
    console.print(_df_to_rich_table(stats_df, title=label))


def cache_location(cache_dir: Path, n_files: int) -> None:
    """Print a summary line showing where files were cached."""
    console.print(
        f"\n  [bold green]✓[/]  {n_files} file{'s' if n_files != 1 else ''} "
        f"cached at [bright_blue]{cache_dir}[/]"
    )


def error(msg: str) -> None:
    """Print a formatted error message."""
    console.print(f"[bold bright_blue]Error:[/] {msg}")


# ---------------------------------------------------------------------------
# --info display
# ---------------------------------------------------------------------------

_META_SKIP = frozenset({"pandas", "ARROW:schema"})


def _fmt_stat(val: float | None) -> str:
    if val is None:
        return "—"
    if abs(val) >= 10_000:
        return f"{val:,.0f}"
    if abs(val) >= 10:
        return f"{val:.1f}"
    return f"{val:.3f}"


def _render_file_meta(
    file_meta: dict[str, Any],
    n_files: int,
    total_rows: int,
) -> None:
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim", no_wrap=True)
    table.add_column()
    for k, v in file_meta.items():
        if k in _META_SKIP:
            continue
        table.add_row(k, str(v)[:120])
    if n_files > 1:
        table.add_row("matched files", str(n_files))
    table.add_row("total rows", f"{total_rows:,}")
    console.print(
        Panel(table, title="[bright_blue]Dataset Metadata[/]", box=box.ROUNDED, expand=False)
    )


def _render_schema_table(categories: list[Any]) -> None:
    import pandas as pd

    rows = [
        {
            "Category": c["name"],
            "n": c["n"],
            "Columns": c["pattern"],
            "Units": c["units"],
        }
        for c in categories
    ]
    df = pd.DataFrame(rows)
    console.print(_df_to_rich_table(df, title="Column Schema"))


def _render_stats_table(
    stat_rows: list[Any],
    layer_label: str,
    n_files: int,
    total_rows: int,
) -> None:
    import pandas as pd

    file_note = f" · {n_files} files" if n_files > 1 else ""
    title = (
        f"Statistics — {layer_label}  ·  {total_rows:,} rows{file_note}"
        "  [dim](parquet row-group min/max)[/]"
    )
    rows = [
        {
            "Category": s["category"],
            "Layer": s["layer_label"],
            "Min": _fmt_stat(s["col_min"]),
            "Max": _fmt_stat(s["col_max"]),
            "Units": s["units"],
        }
        for s in stat_rows
    ]
    df = pd.DataFrame(rows)
    console.print(_df_to_rich_table(df, title=title))


def dataset_info_panel(
    footer_infos: list[dict[str, Any]],
    filter_categories: tuple[str, ...] | None,
    layers: list[int],
    depth_target: float | None,
    depth_avg: bool,
    n_files: int,
) -> None:
    """Render the full --info display: metadata, column schema, and statistics.

    Parameters
    ----------
    footer_infos : list of dict
        One ParquetFooterInfo-shaped dict per matched file.
    filter_categories : tuple of str or None
        Category filter keys to include (``None`` shows all).
    layers : list of int
        Sigma layers for the stats table.
    depth_target : float or None
        If given, select the nearest sigma layer by depth.
    depth_avg : bool
        If True, average stats across all sigma layers.
    n_files : int
        Total number of matched files (may exceed len(footer_infos) if some
        fetches failed).
    """
    from ..analysis.resource import categorize_columns, compute_footer_stats

    first = footer_infos[0]
    total_rows = sum(int(info.get("num_rows", 0)) for info in footer_infos)

    all_categories = categorize_columns(list(first["column_stats"].keys()))
    cats_to_show = (
        [c for c in all_categories if c["filter_key"] in filter_categories]
        if filter_categories
        else all_categories
    )

    if filter_categories is None:
        _render_file_meta(first["file_meta"], n_files, total_rows)
    _render_schema_table(cats_to_show)

    stat_rows, layer_label = compute_footer_stats(
        footer_infos=footer_infos,
        categories=cats_to_show,
        layers=layers,
        depth_target=depth_target,
        depth_avg=depth_avg,
    )
    _render_stats_table(stat_rows, layer_label, n_files, total_rows)
