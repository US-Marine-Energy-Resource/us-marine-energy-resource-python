"""Read a grid node's record straight from the published .h5 files on S3.

No API, no key, and no server-side archive build to fail. The trade is
volume: the files store data in chunks spanning many timestamps and many
neighboring nodes, and whole chunks must be fetched, so one node costs
megabytes per variable and year even though its own values are kilobytes.
Every fetched chunk block is kept under ``<cache_dir>/s3_chunks/``, so later
queries near a fetched point cost nothing to download.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...explore.lazy import lazy_import
from .. import _store, errors
from ..backend import BackendInfo
from ..config import CONFIG
from ..domains import DOMAINS, domain_config
from ..nodes import WaveNode

if TYPE_CHECKING:
    import pandas as pd

# Block size for the range reader. A measured data chunk is about 2 MB, so
# blocks this size fetch a chunk in one request instead of dozens.
_BLOCK_SIZE = 2 * 1024 * 1024

# Datasets that describe the grid rather than the sea state.
_NON_VARIABLES = {"coordinates", "meta", "time_index"}

# Cost of one variable for one year at one node, measured on a West Coast
# file and used only for the estimates the CLI shows.
MB_PER_VARIABLE_YEAR = 15
SECONDS_PER_VARIABLE_YEAR = 20


def _year_uri(domain: str, year: int) -> str:
    """Build the s3:// URI of one domain-year file.

    Every domain's ``grid_key`` names its 2010 file; the other years differ
    only in the year.

    Parameters
    ----------
    domain : str
        Domain name.
    year : int
        Year of the file.

    Returns
    -------
    str
        The full ``s3://`` URI of the file for that domain and year.
    """
    return f"{CONFIG.s3_bucket_uri}/{DOMAINS[domain]['grid_key'].replace('2010', str(year))}"


def _label(dataset: str) -> str:
    """Turn a dataset name into the column label the API CSVs use.

    Parameters
    ----------
    dataset : str
        Dataset name from the .h5 file.

    Returns
    -------
    str
        The matching column label.
    """
    return dataset.replace("_", " ").title()


def _node_column(
    ds: Any, dataset: str, gid: int, chunk_dir: Path, saved: dict[str, int]
) -> tuple[Any, str | None]:
    """Read one node's column, caching the whole chunk block it rides in.

    The chunks bundle adjacent nodes, so keeping the block makes every other
    node in it free later.

    Parameters
    ----------
    ds : Any
        Open h5py dataset for one variable.
    dataset : str
        Name of the dataset, used in the cached block's filename.
    gid : int
        Grid id of the node to read.
    chunk_dir : Path
        Directory that holds the cached chunk blocks.
    saved : dict of str to int
        Running totals for the cache note, updated in place.

    Returns
    -------
    tuple of (Any, str or None)
        The node's values as a float array and the variable's unit if the
        file records one.
    """
    np = lazy_import("numpy", "reading wave data from S3")

    width = ds.chunks[1] if ds.chunks else ds.shape[1]
    saved["width"] = max(saved.get("width", 0), int(width))
    start = (gid // width) * width
    stop = min(start + width, ds.shape[1])
    block_path = chunk_dir / f"{dataset}_{start}.npy"
    if block_path.exists():
        block = np.load(block_path, mmap_mode="r")
    else:
        block = ds[:, start:stop]
        chunk_dir.mkdir(parents=True, exist_ok=True)
        np.save(block_path, block)
        saved["bytes"] = saved.get("bytes", 0) + int(block.nbytes)

    column = np.asarray(block[:, gid - start], dtype="float64")
    scale = ds.attrs.get("scale_factor")
    if scale:
        # rex convention: the stored integers divide by scale_factor.
        column = column / float(scale)
    unit = ds.attrs.get("units")
    if isinstance(unit, bytes):
        unit = unit.decode()
    return column, unit


def _read_year(
    domain: str,
    year: int,
    gid: int,
    variables: list[str] | None,
    cache_dir: Path,
    saved: dict[str, int],
    on_event: Callable[[str], None],
) -> tuple[pd.DataFrame, dict[str, str | None]]:
    """Read one year of one node from its .h5 file on S3.

    Parameters
    ----------
    domain : str
        Domain name.
    year : int
        Year to read.
    gid : int
        Grid id of the node to read.
    variables : list of str, optional
        Variables to read. All variables in the file when omitted.
    cache_dir : Path
        Root of the on disk cache.
    saved : dict of str to int
        Running totals for the cache note, updated in place.
    on_event : callable
        Called with a short message as each variable is read.

    Returns
    -------
    tuple of (pd.DataFrame, dict)
        The year's values indexed by timestamp and each column's unit.
    """
    import pandas as pd

    from ...explore.sources import SourceError, resolve_source

    h5py = lazy_import("h5py", "reading wave data from S3")
    uri = _year_uri(domain, year)
    try:
        source = resolve_source(uri)
    except SourceError as exc:
        raise errors.InvalidYearError(f"no published file for {domain} {year}: {exc}") from exc

    chunk_dir = cache_dir / CONFIG.chunks_dirname / domain / str(year)
    with (
        source.open_binary(max_bytes=None, block_size=_BLOCK_SIZE) as handle,
        h5py.File(handle, "r") as f,
    ):
        available = sorted(
            name for name, ds in f.items() if name not in _NON_VARIABLES and ds.ndim == 2
        )
        names = variables or available
        missing = sorted(set(names) - set(available))
        if missing:
            raise errors.InvalidAttributeError(
                f"not in the {domain} {year} file: {', '.join(missing)}",
                valid=available,
            )
        stamps = pd.to_datetime([t.decode() for t in f["time_index"][:]], utc=True)
        data: dict[str, Any] = {}
        units: dict[str, str | None] = {}
        for i, name in enumerate(names):
            on_event(f"{year}: reading {name} ({i + 1}/{len(names)})")
            column, unit = _node_column(f[name], name, gid, chunk_dir, saved)
            data[_label(name)] = column
            units[_label(name)] = unit
        frame = pd.DataFrame(data, index=stamps)
        frame.index.name = "timestamp"
        return frame, units


class S3Backend:
    """Fetch a grid node's record by range-reading the published .h5 files."""

    def describe(self, node: WaveNode) -> BackendInfo:
        """Report what the S3 files serve for the node's domain.

        The published files carry their directions in the meteorological
        convention already, so no correction applies here even where the
        API's CSVs need one.

        Parameters
        ----------
        node : WaveNode
            The grid node to describe.

        Returns
        -------
        BackendInfo
            Year span, sampling interval, and endpoint for the node's domain.
        """
        config = domain_config(node.domain)
        return BackendInfo(
            endpoint=CONFIG.s3_bucket_uri,
            first_year=1979,
            last_year=2020,
            interval_minutes=int(config["interval"]),
            direction_transform=None,
        )

    def fetch(
        self,
        node: WaveNode,
        name: str,
        *,
        requested_lat: float,
        requested_lon: float,
        force: bool,
        timeout_s: int,
        cache_dir: Path,
        on_event: Callable[[str], None],
        years: list[int] | None = None,
        variables: list[str] | None = None,
    ) -> None:
        """Read the selected years and variables and organize them on disk.

        Parameters
        ----------
        node : WaveNode
            The grid node to fetch.
        name : str
            Site name used for the output directory and files.
        requested_lat : float
            Latitude the user asked for, recorded in the metadata.
        requested_lon : float
            Longitude the user asked for, recorded in the metadata.
        force : bool
            Accepted for backend parity. Unused here.
        timeout_s : int
            Accepted for backend parity. Unused here.
        cache_dir : Path
            Root of the on disk cache.
        on_event : callable
            Called with a short message as work progresses.
        years : list of int, optional
            Years to read. The full published span when omitted.
        variables : list of str, optional
            Variables to read. All variables in the files when omitted.
        """
        from datetime import datetime, timezone

        import pandas as pd

        info = self.describe(node)
        chosen = years or list(range(info.first_year, info.last_year + 1))
        out_of_range = [y for y in chosen if not info.first_year <= y <= info.last_year]
        if out_of_range:
            raise errors.InvalidYearError(
                f"outside {info.first_year}-{info.last_year}: "
                f"{', '.join(str(y) for y in out_of_range)}"
            )

        saved: dict[str, int] = {}
        frames = []
        units: dict[str, str | None] = {}
        for year in sorted(chosen):
            frame, units = _read_year(
                node.domain, year, node.location_id, variables, cache_dir, saved, on_event
            )
            frames.append(frame)
        combined = pd.concat(frames)

        stem = _store.site_stem(name, node.lat, node.lon)
        out_dir = cache_dir / stem
        out_dir.mkdir(parents=True, exist_ok=True)
        first, last = min(chosen), max(chosen)
        combined.to_csv(out_dir / _store.combined_csv_name(stem, first, last))

        metadata = {
            "site": name,
            "domain": node.domain,
            "gid": node.location_id,
            "node_lat": node.lat,
            "node_lon": node.lon,
            "requested_lat": requested_lat,
            "requested_lon": requested_lon,
            "years": [str(first), str(last)],
            "interval_minutes": info.interval_minutes,
            "rows": len(combined),
            "variables": list(combined.columns),
            "units": units,
            "direction_transform": None,
            "source": "s3 direct",
            "organized_at": datetime.now(timezone.utc).isoformat(),
        }
        _store.write_json(out_dir / _store.METADATA_FILENAME, metadata)

        if saved.get("bytes"):
            width = saved.get("width", 0)
            on_event(
                f"note: kept {saved['bytes'] / 1e6:,.0f} MB of chunk blocks in the cache. "
                f"They bundle about {width:,} neighboring nodes each, so nearby points "
                "now read from disk"
            )
