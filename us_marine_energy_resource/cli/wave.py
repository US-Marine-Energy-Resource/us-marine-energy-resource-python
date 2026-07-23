"""``mer wave``: query the WPTO wave hindcast at a point.

Two backends fetch the data, and the default choice is automatic: small
queries read straight from the published .h5 files on S3 with no key, and
large ones go through the NLR developer download API, which builds a zip of
the node's record and needs a free key. Either way the command describes
what it will do and asks before starting. ``--info`` answers "which
node/domain/years is this point?" with no key and no wave data download.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from ..wave_hindcast import _store, hindcast
from ..wave_hindcast import errors as wave_errors
from ..wave_hindcast.backend import AUTO_SEAM_VARIABLE_YEARS, resolve_backend
from ..wave_hindcast.config import CONFIG
from ..wave_hindcast.domains import API_OUTAGES, BASE_DOMAIN, DOMAINS
from ._display import console, error
from ._geometry import parse_point
from ._links import _link

if TYPE_CHECKING:
    from collections.abc import Iterator

# What a query fetches unless told otherwise: the variables most resource
# work starts from (Hm0, Te, Tp, and J) for the most recent served year.
# The summary tells the user what was held back and how to get more.
_DEFAULT_VARIABLES: tuple[str, ...] = (
    "significant_wave_height",
    "energy_period",
    "peak_period",
    "omni-directional_wave_power",
)
_DEFAULT_YEARS_SPAN = 1

def _prose_list(items: list[str]) -> str:
    """Join items into prose with commas and a final "and".

    Parameters
    ----------
    items : list of str
        The items to join.

    Returns
    -------
    str
        "a", "a and b", or "a, b, and c".
    """
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _count_word(n: int) -> str:
    """Spell a small count out for prose, falling back to digits.

    Parameters
    ----------
    n : int
        The count.

    Returns
    -------
    str
        "four", "six", and so on, or ``str(n)`` past twelve.
    """
    words = (
        "zero", "one", "two", "three", "four", "five", "six",
        "seven", "eight", "nine", "ten", "eleven", "twelve",
    )
    return words[n] if 0 <= n < len(words) else str(n)


def _build_wave_help(domains: dict[str, dict[str, Any]], base_domain: dict[str, Any]) -> str:
    """Build the command help from the canonical dataset facts.

    Every enumerable fact (regions, years, default variables, credential
    variable names, the backend seam) is derived from the modules that
    define it, so the help cannot drift from the code.

    Parameters
    ----------
    domains : dict
        The per-domain overrides, normally :data:`DOMAINS`.
    base_domain : dict
        The shared domain settings, normally :data:`BASE_DOMAIN`.

    Returns
    -------
    str
        The rich-markup help text.
    """
    regions = [name.replace("_", " ") for name in domains]
    years = f"{base_domain['first_year']}-{base_domain['last_year']}"
    variables = _prose_list([v.replace("_", " ") for v in _DEFAULT_VARIABLES])
    capped = [
        name.replace("_", " ")
        for name, cfg in domains.items()
        if cfg.get("last_year", base_domain["last_year"]) < base_domain["last_year"]
    ]
    cap_sentence = ""
    if capped:
        cap_year = max(
            cfg["last_year"]
            for cfg in domains.values()
            if cfg.get("last_year", base_domain["last_year"]) < base_domain["last_year"]
        )
        cap_sentence = (
            f" The api backend serves {_prose_list(capped)} only through"
            f" {cap_year}, so requests past that read from S3."
        )
    region_word = "region" if len(regions) == 1 else "regions"
    return (
        "Query the U.S. DOE WPTO High-Resolution Wave Hindcast at a point. The"
        f" dataset covers {years} in {_count_word(len(regions))} {region_word}:"
        f" {_prose_list(regions)}.\n\n"
        "A query finds the hindcast grid node nearest the coordinate and fetches, by"
        f" default, the {_count_word(len(_DEFAULT_VARIABLES))} high impact variables"
        f" ({variables}) for the most recent served year, so a first"
        " look is quick. Use [bright_blue]--years[/] and [bright_blue]--variables[/]"
        " for more, or [bright_blue]--all[/] for everything, which is slow because of"
        " the volume. The result is cached locally, so repeat queries read from disk,"
        " and saved as a CSV in the current directory unless"
        " [bright_blue]--cache-only[/] is given.\n\n"
        "Two backends serve the data and the default picks between them by size."
        " A small query reads straight from the published files in"
        f" {CONFIG.s3_bucket_uri}, which needs no key or account. A large query"
        " switches to the NLR api backend, which builds the archive server side"
        f" and needs a free API key ({_link(CONFIG.signup_url)}) in"
        f" [bright_blue]{CONFIG.api_key_env}[/] and a contact email in"
        f" [bright_blue]{CONFIG.email_env}[/], set in the environment, in a"
        " [bright_blue].env[/] file in the current directory, or in"
        " [bright_blue]~/.mer.env[/] to cover every directory. A query counts as"
        f" large past {AUTO_SEAM_VARIABLE_YEARS} variable-years, which is the"
        " years times the variables it asks for, and a large query without a key"
        " stays on S3 and takes longer. Choose a backend yourself with"
        " [bright_blue]--backend s3[/] or [bright_blue]--backend api[/]."
        f"{cap_sentence}"
        " [bright_blue]--info[/] needs no key at all.\n\n"
        "To browse the source .h5 files on S3, use [bright_blue]mer ls wave[/] or"
        " [bright_blue]mer explore wave/...[/]"
    )


# Coordinates the examples use: PacWave South off Newport, Oregon, and
# Kaneohe Bay off Oahu, Hawaii.
_EXAMPLE_POINT = (44.57, -124.23)
_EXAMPLE_POINT_HI = (21.46, -157.75)
_EXAMPLE_ARG = "{},{}".format(*_EXAMPLE_POINT)
_EXAMPLE_ARG_HI = "{},{}".format(*_EXAMPLE_POINT_HI)

# Column where an example's description starts.
_EXAMPLE_COLUMN = 45


def _example(command: str, description: str) -> str:
    """Lay one example out with its description in a fixed column.

    Parameters
    ----------
    command : str
        The command to show.
    description : str
        What it does.

    Returns
    -------
    str
        The command padded to the description column, or split over two
        lines when the command is too long for one. Typer joins the two
        lines with a space when it rewraps the epilog, so the description
        renders on its own line at the terminal's left edge.
    """
    if len(command) >= _EXAMPLE_COLUMN:
        return f"{command}\n{' ' * _EXAMPLE_COLUMN}{description}\n\n"
    return f"{command:<{_EXAMPLE_COLUMN}}{description}\n\n"


def _build_wave_epilog(base_domain: dict[str, Any]) -> str:
    """Build the examples epilog with years the dataset really serves.

    Parameters
    ----------
    base_domain : dict
        The shared domain settings, normally :data:`BASE_DOMAIN`.

    Returns
    -------
    str
        The rich-markup epilog text.
    """
    last = base_domain["last_year"]
    n_default = _count_word(len(_DEFAULT_VARIABLES)).capitalize()
    cached = f"{_store.point_name(*_EXAMPLE_POINT)}_y{last}-{last}"
    examples = [
        (f"mer wave {_EXAMPLE_ARG} --info", "Show the node, domain, and years"),
        (f"mer wave {_EXAMPLE_ARG}", f"{n_default} key variables, most recent year, no key"),
        (
            f"mer wave {_EXAMPLE_ARG} --years {last - 9}-{last}",
            "The same variables for a decade",
        ),
        (f"mer wave {_EXAMPLE_ARG} --all", "Every variable and year (slow, heavy)"),
        (f"mer wave {_EXAMPLE_ARG_HI} -o ./data", "Save the CSV to a chosen directory"),
        (f"mer wave {_EXAMPLE_ARG} --cache-only", "Fetch and cache without writing a CSV"),
        (f"mer wave {_EXAMPLE_ARG} --stats-csv", "Also save the statistics tables as a CSV"),
        (
            f"mer wave {_EXAMPLE_ARG} --backend s3 --years {last}",
            "Force one year straight from S3",
        ),
        (
            f"mer wave {_EXAMPLE_ARG} --backend s3 --years {last - 5}-{last} "
            "--variables significant_wave_height",
            "One variable for six years from S3",
        ),
        (f"mer wave {_EXAMPLE_ARG} --dry-run", "Show the request without sending it"),
        ("mer wave --cache", "Show what the cache holds"),
        (f"mer wave --clear {cached} -y", "Remove one cached item"),
        ("mer wave --clear-all", "Empty the wave cache"),
    ]
    lines = "".join(_example(command, description) for command, description in examples)
    return "[bright_blue]Examples[/bright_blue]\n\n" + lines.rstrip("\n")


_WAVE_HELP = _build_wave_help(DOMAINS, BASE_DOMAIN)
_WAVE_EPILOG = _build_wave_epilog(BASE_DOMAIN)


@contextmanager
def _handle_errors() -> Iterator[None]:
    """Turn expected wave errors into clean CLI exits."""
    try:
        yield
    except wave_errors.CredentialsMissingError as exc:
        error(
            f"{exc}\n  --info and --dry-run work without a key, and "
            "--backend s3 fetches the same data with no key."
        )
        raise typer.Exit(1) from exc
    except wave_errors.ApiOutageError as exc:
        error(f"{exc}\n\n{exc.detail}")
        raise typer.Exit(1) from exc
    except wave_errors.WaveHindcastError as exc:
        error(str(exc))
        raise typer.Exit(1) from exc


def _describe_table(info: dict[str, Any]) -> None:
    """Render a describe_point result."""
    from rich.table import Table

    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="bright_blue")
    table.add_column()
    table.add_row("Domain", info["domain"])
    table.add_row("Grid node", str(info["location_id"]))
    table.add_row("Node position", f"{info['node_lat']:.4f}, {info['node_lon']:.4f}")
    table.add_row("Distance", f"{info['distance_m']:,.0f} m from the requested point")
    table.add_row("Years", f"{info['years'][0]}-{info['years'][1]} ({info['n_years']} years)")
    table.add_row("Interval", f"{info['interval_minutes']} minutes")
    table.add_row("Endpoint", info["endpoint"])
    if info["direction_transform"]:
        table.add_row("Direction fix", f"{info['direction_transform']} (applied automatically)")
    console.print(table)


def _warn_outage(domain: str, backend: str) -> None:
    """Print the outage notice for a domain whose API is broken upstream."""
    if domain not in API_OUTAGES:
        return
    if backend == "api":
        console.print(
            f"\n[bright_blue]warning:[/] the {domain} API download service is not working "
            "right now. Requests are accepted but never finish. Already downloaded data "
            "still loads, and [bright_blue]--backend s3[/] still works.",
            highlight=False,
        )
    else:
        console.print(
            f"\n[bright_blue]warning:[/] the {domain} API download service is not working "
            "right now, so queries here read from S3.",
            highlight=False,
        )


def _request_plan(
    point: dict[str, Any],
    names: list[str],
    email: str,
    cache_dest: Path,
    save_path: Path | None,
    year_list: list[int] | None = None,
) -> None:
    """Show what the request will do, one fact per line, before the confirmation."""
    from rich.table import Table

    n_years = len(year_list) if year_list else point["n_years"]
    rows = n_years * (365 * 24 * 60 // point["interval_minutes"])
    values = rows * len(names)
    amount = f"{values / 1e6:.1f} million" if values >= 1e6 else f"{values:,}"

    from urllib.parse import urlparse

    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="bright_blue")
    table.add_column()
    table.add_row("Server", urlparse(CONFIG.api_base_url).netloc)
    table.add_row("Extract", f"{rows:,} timestamps x {len(names)} variables, {amount} values")
    table.add_row("From", f"the {n_years} yearly {point['domain']} archive files")
    table.add_row(
        "Wait",
        "as long as the server takes to build the archive. Rerunning resumes a stopped wait",
    )
    table.add_row("Email", f"{email} gets the link when the archive is ready")
    if save_path is not None:
        table.add_row("Saves to", str(save_path))
    table.add_row(
        "Cached at",
        f"{cache_dest.parent} under the name {cache_dest.name}, kept for instant reuse",
    )
    console.print(f"\n[bright_blue]Variables[/] ({len(names)}): {', '.join(names)}\n")
    console.print(table)
    if year_list is None:
        console.print(
            "\n[bright_blue]note:[/] this asks for everything the endpoint serves. "
            "The archive takes the server longer to build.",
            highlight=False,
        )
    console.print()


def _defaults_note(point: dict[str, Any], defaulted_years: bool, defaulted_variables: bool) -> None:
    """Tell the user what the quick defaults held back and how to get it."""
    if not (defaulted_years or defaulted_variables):
        return
    served = f"{point['years'][0]}-{point['years'][1]}"
    parts = []
    if defaulted_years:
        parts.append(f"only {point['years'][1]} out of the served {served}")
    if defaulted_variables:
        parts.append("only the four key variables")
    console.print(
        f"\n[bright_blue]note:[/] to keep this quick, the defaults fetched "
        f"{' and '.join(parts)}. Get more with [bright_blue]--years[/] and "
        f"[bright_blue]--variables[/], or [bright_blue]--all[/] for everything.",
        highlight=False,
    )


def _parse_years(text: str) -> list[int]:
    """Parse a years option: single years, ranges, and comma lists."""
    chosen: set[int] = set()
    try:
        for token in text.split(","):
            token = token.strip()
            if "-" in token:
                first, last = token.split("-", 1)
                chosen.update(range(int(first), int(last) + 1))
            elif token:
                chosen.add(int(token))
    except ValueError as exc:
        error(f"could not read --years {text!r}. Use forms like 2020, 2015-2020, or 2000,2010.")
        raise typer.Exit(1) from exc
    return sorted(chosen)


def _s3_request_plan(
    point: dict[str, Any],
    year_list: list[int] | None,
    variable_list: list[str] | None,
    cache_dest: Path,
    save_path: Path | None,
) -> None:
    """Show what a direct S3 read will do, with its volume and time cost."""
    from rich.table import Table

    from ..wave_hindcast.s3_direct.backend import (
        MB_PER_VARIABLE_YEAR,
        SECONDS_PER_VARIABLE_YEAR,
        TYPICAL_VARIABLES_PER_FILE,
    )

    n_years = len(year_list) if year_list else point["n_years"]
    n_vars = len(variable_list) if variable_list else TYPICAL_VARIABLES_PER_FILE
    vars_label = str(n_vars) if variable_list else f"about {TYPICAL_VARIABLES_PER_FILE}"
    rows = n_years * (365 * 24 * 60 // point["interval_minutes"])
    mb = n_years * n_vars * MB_PER_VARIABLE_YEAR
    seconds = n_years * n_vars * SECONDS_PER_VARIABLE_YEAR
    if seconds < 120:
        wait = f"roughly {seconds} seconds"
    elif seconds < 5400:
        wait = f"roughly {seconds / 60:.0f} minutes"
    else:
        wait = f"roughly {seconds / 3600:.1f} hours"

    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="bright_blue")
    table.add_column()
    table.add_row("Source", f"{CONFIG.s3_bucket_uri}, read directly. No API and no key")
    table.add_row("Extract", f"{rows:,} timestamps x {vars_label} variables")
    table.add_row(
        "Download",
        f"about {mb:,} MB, since the files store data in chunks that bundle many neighboring nodes",
    )
    table.add_row(
        "Wait",
        f"{wait} at typical speeds. Downloaded chunks stay in the cache, so nearby "
        "points are much faster afterward",
    )
    if save_path is not None:
        table.add_row("Saves to", str(save_path))
    table.add_row(
        "Cached at",
        f"{cache_dest.parent} under the name {cache_dest.name}, kept for instant reuse",
    )
    console.print()
    console.print(table)
    if not year_list and not variable_list:
        console.print(
            "\n[bright_blue]note:[/] reading the full record directly is slow. "
            "Narrow it with --years and --variables, or use the default api backend.",
            highlight=False,
        )
    console.print()


def _dir_bytes(path: Path) -> int:
    """Sum the file sizes under a path."""
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _fmt_bytes(n: int) -> str:
    """Format a byte count for people."""
    if n >= 1e9:
        return f"{n / 1e9:.1f} GB"
    if n >= 1e6:
        return f"{n / 1e6:.1f} MB"
    return f"{n / 1e3:.0f} KB"


def _cached_items(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collect the cached sites and the manifest, tolerating a missing cache."""
    import json

    from ..wave_hindcast.nlr_api.archive import load_manifest

    manifest = load_manifest(root / CONFIG.manifest_filename) if root.exists() else {}
    items: list[dict[str, Any]] = []
    if root.is_dir():
        for entry in sorted(root.iterdir()):
            if not entry.is_dir() or entry.name in CONFIG.non_site_dirnames:
                continue
            item = {"dir": entry, "name": entry.name, "bytes": _dir_bytes(entry)}
            meta_path = entry / _store.METADATA_FILENAME
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                item["name"] = str(meta.get("site") or entry.name)
                years = meta.get("years") or ["?", "?"]
                item["years"] = f"{years[0]}-{years[1]}"
                item["variables"] = len(meta.get("variables") or [])
                item["source"] = str(meta.get("source") or "api")
            items.append(item)
    return items, manifest


def _cache_stats(root: Path) -> None:
    """Show what the wave cache holds and how to clear it."""
    from rich.table import Table

    items, manifest = _cached_items(root)
    downloaded = {item["name"] for item in items}
    pending = sorted(set(manifest) - downloaded)

    if not items and not pending:
        console.print(f"The wave cache at {root} is empty.")
        return

    if items:
        table = Table(box=None)
        table.add_column("Cached item", style="green")
        table.add_column("Size", justify="right")
        table.add_column("Years")
        table.add_column("Variables", justify="right")
        table.add_column("Source")
        for item in items:
            table.add_row(
                item["name"],
                _fmt_bytes(item["bytes"]),
                str(item.get("years", "?")),
                str(item.get("variables", "?")),
                str(item.get("source", "?")),
            )
        console.print(table)

    if pending:
        console.print(
            f"\n[bright_blue]Requested but never downloaded[/] ({len(pending)}): "
            f"{', '.join(pending)}"
        )

    chunk_bytes = _dir_bytes(root / CONFIG.chunks_dirname)
    archive_bytes = _dir_bytes(root / CONFIG.archives_dirname)
    if chunk_bytes:
        console.print(
            f"\nChunk blocks from --backend s3: {_fmt_bytes(chunk_bytes)}. "
            "These make nearby points fast and are safe to clear."
        )
    if archive_bytes:
        console.print(f"Downloaded archives: {_fmt_bytes(archive_bytes)}")
    console.print(f"Total: {_fmt_bytes(_dir_bytes(root))} in {root}")
    console.print(
        "\n[dim]Remove one item with --clear NAME, several by repeating it, or "
        "everything with --clear-all.[/]"
    )


def _cache_clear(root: Path, names: list[str], yes: bool) -> None:
    """Remove named items: their site data, archive, and manifest entry."""
    import shutil

    from ..wave_hindcast.nlr_api.archive import save_manifest

    items, manifest = _cached_items(root)
    by_name = {item["name"]: item for item in items}
    known = sorted(set(by_name) | set(manifest))
    missing = [n for n in names if n not in known]
    if missing:
        error(
            f"not in the cache: {', '.join(missing)}. "
            f"Cached: {', '.join(known) if known else 'nothing'}."
        )
        raise typer.Exit(1)

    if not yes and not typer.confirm(f"Remove {', '.join(names)} from the cache?"):
        raise typer.Exit(0)
    for name in names:
        item = by_name.get(name)
        if item is not None:
            shutil.rmtree(item["dir"])
        archive = root / CONFIG.archives_dirname / f"{name}.zip"
        if archive.exists():
            archive.unlink()
        if name in manifest:
            del manifest[name]
        console.print(f"removed {name}")
    save_manifest(root / CONFIG.manifest_filename, manifest)


def _cache_clear_all(root: Path, yes: bool) -> None:
    """Remove the whole wave cache."""
    import shutil

    total = _dir_bytes(root)
    if total == 0:
        console.print(f"The wave cache at {root} is already empty.")
        return
    if not yes and not typer.confirm(f"Remove everything in {root} ({_fmt_bytes(total)})?"):
        raise typer.Exit(0)
    shutil.rmtree(root)
    console.print(f"removed {_fmt_bytes(total)} from {root}")


def _summary(frame: Any, metadata: dict[str, Any], cache_dir: Path) -> None:
    """Print what was fetched and where it lives."""
    years = metadata.get("years") or ["?", "?"]
    variables = metadata.get("variables") or []
    console.print(
        f"\n[green]{metadata.get('site')}[/]: {len(frame):,} rows, "
        f"{years[0]}-{years[1]}, {len(variables)} variables"
    )
    if metadata.get("organized_at"):
        console.print(f"[dim]downloaded {str(metadata['organized_at'])[:16]}[/]")
    console.print(f"[dim]cached under {cache_dir}[/]")


def _fmt(value: float) -> str:
    """Format a statistic: two decimals for small values, thousands for large."""
    return f"{value:,.0f}" if abs(value) >= 1000 else f"{value:.2f}"


# The statistic columns of the monthly and all-time tables, in display order.
_QUANTILE_COLUMNS: tuple[tuple[str, float | None], ...] = (
    ("Min", None),
    ("P0.1", 0.001),
    ("P1", 0.01),
    ("P5", 0.05),
    ("P25", 0.25),
    ("Median", 0.5),
    ("P75", 0.75),
    ("P95", 0.95),
    ("P99", 0.99),
    ("P99.9", 0.999),
    ("Max", None),
    ("Mean", None),
)


def _stat_values(series: Any) -> dict[str, float]:
    """Compute one row of statistics in _QUANTILE_COLUMNS order."""
    values: dict[str, float] = {}
    for label, quantile in _QUANTILE_COLUMNS:
        if quantile is not None:
            values[label] = float(series.quantile(quantile))
        elif label == "Min":
            values[label] = float(series.min())
        elif label == "Max":
            values[label] = float(series.max())
        else:
            values[label] = float(series.mean())
    return values


def _stats_frame(frame: Any, metadata: dict[str, Any]) -> Any:
    """Build the tidy statistics table: one row per variable and period.

    Periods are the twelve months (combined across years) followed by
    All-time. Direction variables are left out because linear statistics
    mislead on angles, and None is returned when nothing remains.
    """
    import calendar

    import pandas as pd

    from ..wave_hindcast.nlr_api.archive import DIRECTION_COLUMNS

    skip = set(DIRECTION_COLUMNS) | {"Year", "Month", "Day", "Hour", "Minute"}
    numeric = frame.select_dtypes("number")
    columns = [c for c in numeric.columns if c not in skip]
    records = []
    for column in columns:
        series = numeric[column].dropna()
        if series.empty:
            continue
        if isinstance(frame.index, pd.DatetimeIndex):
            for month, values in series.groupby(series.index.month):
                records.append(
                    {"Variable": column, "Period": calendar.month_abbr[int(month)]}
                    | _stat_values(values)
                )
        records.append({"Variable": column, "Period": "All-time"} | _stat_values(series))
    return pd.DataFrame(records) if records else None


def _render_stats(stats: Any, metadata: dict[str, Any]) -> None:
    """Print monthly statistics per variable, then the all-time table."""
    from rich.markup import escape
    from rich.table import Table

    units = metadata.get("units") or {}
    years = metadata.get("years") or ["?", "?"]
    span = f"({years[0]} through {years[1]})"
    labels = [label for label, _quantile in _QUANTILE_COLUMNS]

    def labelled(column: str) -> str:
        unit = f" [{units[column]}]" if units.get(column) else ""
        # escape() keeps a unit like [m] literal instead of it being read as
        # rich markup and vanishing.
        return escape(f"{column}{unit}")

    monthly_rows = stats[stats["Period"] != "All-time"]
    for variable, group in monthly_rows.groupby("Variable", sort=False):
        monthly = Table(
            box=None, title=f"Monthly {labelled(str(variable))} {span}", title_justify="left"
        )
        monthly.add_column("Month", style="green")
        for label in labels:
            monthly.add_column(label, justify="right")
        for _, row in group.iterrows():
            monthly.add_row(str(row["Period"]), *[_fmt(row[label]) for label in labels])
        console.print()
        console.print(monthly)

    table = Table(box=None, title=f"All-time {span}", title_justify="left")
    table.add_column("Variable", style="green")
    for label in labels:
        table.add_column(label, justify="right")
    for _, row in stats[stats["Period"] == "All-time"].iterrows():
        table.add_row(labelled(str(row["Variable"])), *[_fmt(row[label]) for label in labels])
    console.print()
    console.print(table)
    console.print("[dim]direction variables are angles and are left out[/]")


def _export(frame: Any, metadata: dict[str, Any], output_dir: Path) -> None:
    """Write the combined CSV and metadata beside each other in output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    site = str(metadata.get("site"))
    csv_path = output_dir / f"{site}.csv"
    frame.to_csv(csv_path)
    _store.write_json(output_dir / f"{site}_metadata.json", metadata)
    console.print(f"[dim]wrote {csv_path}[/]")


def wave_query(
    location: Annotated[
        str | None,
        typer.Argument(
            help=f"Point as [bright_blue]lat,lon[/] (e.g. [bright_blue]{_EXAMPLE_ARG}[/])."
        ),
    ] = None,
    info: Annotated[
        bool,
        typer.Option(
            "--info",
            "-i",
            help="Show the node, domain, and years, then exit. "
            "Needs no API key and downloads no wave data.",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Show the request that would be sent, send nothing."),
    ] = False,
    domain: Annotated[
        str | None,
        typer.Option("--domain", help="Use this domain instead of finding it from the point."),
    ] = None,
    name: Annotated[
        str | None,
        typer.Option("--name", help="Label for the cache directory (default: the coordinates)."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Re-request even if the point is already cached."),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip the confirmation prompt."),
    ] = False,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            "-o",
            help="Write the CSV and metadata files here instead of the current directory.",
        ),
    ] = None,
    cache_only: Annotated[
        bool,
        typer.Option("--cache-only", help="Keep the data in the cache without writing a CSV."),
    ] = False,
    stats_csv: Annotated[
        bool,
        typer.Option(
            "--stats-csv",
            help="Also save the printed statistics tables as a CSV beside the data.",
        ),
    ] = False,
    cache_dir: Annotated[
        Path | None,
        typer.Option(
            "--cache-dir",
            help=f"Cache directory (default ~/{CONFIG.cache_dir_name}, "
            f"or set {CONFIG.cache_dir_env} to move it).",
        ),
    ] = None,
    timeout_min: Annotated[
        int,
        typer.Option("--timeout", help="Minutes to wait for the archive to build."),
    ] = CONFIG.default_timeout_s // 60,
    backend: Annotated[
        str,
        typer.Option(
            "--backend",
            help="auto (default) reads small queries directly from S3 with no key and "
            "switches to the api for large ones. api always requests a server-built "
            "archive, which needs a free key. s3 always reads the published .h5 files "
            "directly, which needs no key and cannot fail server-side, but is slower "
            "for big requests, so narrow them with --years and --variables.",
        ),
    ] = "auto",
    years: Annotated[
        str | None,
        typer.Option("--years", help="Narrow to years, e.g. 2020, 2015-2020, or 2000,2010."),
    ] = None,
    variables: Annotated[
        str | None,
        typer.Option(
            "--variables",
            help="Narrow to a comma separated list, e.g. significant_wave_height,energy_period.",
        ),
    ] = None,
    all_data: Annotated[
        bool,
        typer.Option(
            "--all",
            help="Fetch every variable for every served year instead of the defaults. "
            "Intended to work, but slow and heavy because of the volume, especially "
            "with --backend s3.",
        ),
    ] = False,
    cache_stats: Annotated[
        bool,
        typer.Option("--cache", help="Show what is in the wave cache, then exit."),
    ] = False,
    clear: Annotated[
        list[str] | None,
        typer.Option("--clear", help="Remove one cached item by name. Repeat for several."),
    ] = None,
    clear_all: Annotated[
        bool,
        typer.Option("--clear-all", help="Remove everything in the wave cache."),
    ] = False,
) -> None:
    """Query the wave hindcast at a point and print or download the result."""
    if cache_stats:
        _cache_stats(cache_dir or hindcast.default_cache_dir())
        return
    if clear:
        _cache_clear(cache_dir or hindcast.default_cache_dir(), clear, yes)
        return
    if clear_all:
        _cache_clear_all(cache_dir or hindcast.default_cache_dir(), yes)
        return

    if location is None:
        error(f"No point was given. Provide a lat,lon coordinate, e.g. {_EXAMPLE_ARG}.")
        raise typer.Exit(1)
    try:
        lat, lon = parse_point(location)
    except ValueError as exc:
        error(str(exc))
        raise typer.Exit(1) from exc

    explicit_name = name is not None
    with _handle_errors():
        point = hindcast.describe_point(lat, lon, domain=domain, backend=backend)
        _describe_table(point)
        if backend != "s3":
            _warn_outage(point["domain"], backend)

        if info:
            return

        # Defaults are deliberately narrow so nobody downloads every year of
        # everything by accident. --all is there for whoever asks.
        if all_data and (years or variables):
            error("--all fetches every variable and year. Give it alone, or narrow instead.")
            raise typer.Exit(1)
        if all_data:
            year_list = None
            variable_list = None
        else:
            last_year = int(point["years"][1])
            year_list = (
                _parse_years(years)
                if years
                else list(range(last_year - _DEFAULT_YEARS_SPAN + 1, last_year + 1))
            )
            variable_list = (
                [v.strip() for v in variables.split(",")] if variables else list(_DEFAULT_VARIABLES)
            )

        requested_backend = backend
        backend, backend_reason = resolve_backend(
            backend, point["domain"], years=year_list, variables=variable_list
        )
        if requested_backend == "auto" and backend == "api":
            # The auto describe reports the s3 view, and the api serves
            # fewer years in some domains, so refresh the numbers.
            point = hindcast.describe_point(lat, lon, domain=domain, backend="api")
        if backend_reason:
            console.print(f"\n[bright_blue]note:[/] {backend_reason}.", highlight=False)

        if dry_run:
            span = (
                f"{min(year_list)}-{max(year_list)}"
                if year_list
                else f"{point['years'][0]}-{point['years'][1]}"
            )
            wanted = ", ".join(variable_list) if variable_list else "every variable served"
            source = point["endpoint"] if backend == "api" else CONFIG.s3_bucket_uri
            console.print(
                f"\n[bright_blue]Would request[/] grid node {point['location_id']} from "
                f"{source}: years {span}, interval "
                f"{point['interval_minutes']} minutes, UTC, leap days included. "
                f"Variables: {wanted}."
            )
            return

        # Fail on missing credentials before the confirmation, not after a
        # wait. Skipped for a cached point and the s3 backend, which need none.
        name = name or _store.point_name(lat, lon)
        # A narrowed record must not shadow a full one already in the cache,
        # and a custom variable set must not shadow the default one.
        if year_list and not explicit_name:
            name = f"{name}_y{min(year_list)}-{max(year_list)}"
        if variable_list and variable_list != list(_DEFAULT_VARIABLES) and not explicit_name:
            import hashlib

            digest = hashlib.md5(",".join(sorted(variable_list)).encode()).hexdigest()[:6]
            name = f"{name}_v{digest}"
        cache_root = cache_dir or hindcast.default_cache_dir()
        cached = name in hindcast.sites_on_disk(cache_dir)
        if not (cached and not force):
            save_path = None if cache_only else (output_dir or Path.cwd()) / f"{name}.csv"
            if backend == "s3":
                _s3_request_plan(point, year_list, variable_list, cache_root / name, save_path)
            else:
                from ..wave_hindcast.nlr_api import client as nlr_client

                api_key, email = nlr_client.credentials()
                # The endpoint is asked which variables it serves before any
                # job is queued. The answer is cached per domain, so the real
                # request reuses it.
                with console.status("[bright_blue]asking the endpoint which variables it serves…"):
                    names = nlr_client.attributes_for(point["domain"], api_key, email)
                if variable_list:
                    names = [n for n in names if n in set(variable_list)]
                _request_plan(point, names, email, cache_root / name, save_path, year_list)
            if not yes and not typer.confirm("Continue?"):
                raise typer.Exit(0)

        def on_event(message: str) -> None:
            # Quota and note lines are worth keeping on screen; progress lines
            # only update the spinner.
            if message.startswith(("quota:", "note:")):
                console.print(f"[dim]{message}[/]")
            else:
                status.update(f"[bright_blue]{message}")

        with console.status("[bright_blue]querying the wave hindcast…") as status:
            result = hindcast.get_data_at_point(
                lat,
                lon,
                name=name,
                domain=domain,
                force=force,
                cache_dir=cache_dir,
                backend=backend,
                timeout_s=timeout_min * 60,
                years=year_list,
                variables=variable_list,
                on_event=on_event,
                return_metadata=True,
            )
        assert isinstance(result, tuple)
        frame, metadata = result

        _summary(frame, metadata, cache_root)
        _defaults_note(point, defaulted_years=not years, defaulted_variables=not variables)
        stats = _stats_frame(frame, metadata)
        if stats is not None:
            _render_stats(stats, metadata)
        if not cache_only:
            _export(frame, metadata, output_dir or Path.cwd())
        if stats is not None:
            dest_dir = output_dir or Path.cwd()
            stats_path = dest_dir / f"{metadata.get('site')}_stats.csv"
            if stats_csv:
                dest_dir.mkdir(parents=True, exist_ok=True)
                stats.to_csv(stats_path, index=False)
                console.print(f"[dim]wrote {stats_path}[/]")
            else:
                console.print(f"[dim]add --stats-csv to save these tables as {stats_path.name}[/]")
