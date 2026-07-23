"""Build a domain's node parquet from its source .h5 on S3.

Only the ``coordinates`` dataset is read, never the sea-state arrays, so the
transfer is a tiny fraction of what the source files weigh.

Shared by three callers: the runtime fallback in :mod:`.index` (when a node
file can be neither found nor downloaded), the maintainer script
``scripts/build_wave_node_index.py``, and the tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, BinaryIO

from ..explore.lazy import lazy_import
from . import domains
from .config import CONFIG

# Coordinates are stored as fixed-point integers rather than floats, which is
# what makes the index small enough to ship. 1e-6 degrees (~10 cm) is far
# finer than the grid spacing.
COORD_SCALE = 10**6

ROW_GROUP_SIZE = 100_000

# Ceiling on bytes fetched while reading one coordinates dataset. Sized well
# above the largest domain's float64 pairs plus HDF5 metadata and
# block-granularity overhead.
_MAX_COORD_BYTES = 512 * 1024 * 1024


def read_coordinates_from(handle: BinaryIO) -> Any:
    """Read the ``(n, 2)`` lat/lon array out of an open rex-style .h5 handle.

    Parameters
    ----------
    handle : BinaryIO
        A seekable binary handle to the HDF5 file.

    Returns
    -------
    numpy.ndarray
        Shape ``(n, 2)``: latitude in column 0, longitude in column 1.
    """
    h5py = lazy_import("h5py", "reading the wave grid from HDF5")
    np = lazy_import("numpy", "reading the wave grid from HDF5")
    with h5py.File(handle, "r") as h5:
        return np.asarray(h5["coordinates"])


def read_coordinates(domain: str) -> Any:
    """Read a domain's ``(n, 2)`` lat/lon array, straight from S3.

    Parameters
    ----------
    domain : str
        One of the keys of :data:`.domains.DOMAINS`.

    Returns
    -------
    numpy.ndarray
        Shape ``(n, 2)``: latitude in column 0, longitude in column 1.
    """
    from ..explore.sources import resolve_source

    uri = f"{CONFIG.s3_bucket_uri}/{domains.DOMAINS[domain]['grid_key']}"
    source = resolve_source(uri)
    with source.open_binary(_MAX_COORD_BYTES) as handle:
        return read_coordinates_from(handle)


def write_domain_nodes(coords: Any, dest: Path, *, coord_scale: int = COORD_SCALE) -> None:
    """Write one domain's nodes to a parquet file.

    Rows stay in the source file's own order, which is what keeps the file
    small. ``location_id`` is stored explicitly rather than left implicit in
    the row order so the files stand on their own.

    Parameters
    ----------
    coords : numpy.ndarray
        Shape ``(n, 2)`` lat/lon array in source order.
    dest : Path
        Parquet file to write.
    coord_scale : int, default ``COORD_SCALE``
        Fixed-point multiplier for the coordinate columns.
    """
    np = lazy_import("numpy", "building the wave node index")
    pa = lazy_import("pyarrow", "building the wave node index")
    pq = lazy_import("pyarrow.parquet", "building the wave node index")

    table = pa.table(
        {
            # The grid id the download API wants as `location_ids` is exactly
            # the row index into the source file's `coordinates` dataset.
            "location_id": np.arange(len(coords), dtype=np.int32),
            "lat_fixed": np.round(coords[:, 0] * coord_scale).astype(np.int32),
            "lon_fixed": np.round(coords[:, 1] * coord_scale).astype(np.int32),
        }
    )
    # Source order is spatially coherent, so delta-encoded coordinates
    # compress well. Sorting (by latitude, say) would scramble location_id
    # into a random permutation and inflate the file.
    pq.write_table(
        table,
        dest,
        compression="zstd",
        compression_level=22,
        use_dictionary=False,
        column_encoding={
            "location_id": "DELTA_BINARY_PACKED",
            "lat_fixed": "DELTA_BINARY_PACKED",
            "lon_fixed": "DELTA_BINARY_PACKED",
        },
        row_group_size=ROW_GROUP_SIZE,
    )


def domain_bounds(domain: str, coords: Any) -> dict[str, Any]:
    """Compute the bounds record for one domain, as stored in the index JSON.

    Parameters
    ----------
    domain : str
        Domain name.
    coords : numpy.ndarray
        Shape ``(n, 2)`` lat/lon array.

    Returns
    -------
    dict
        Domain name, lat/lon extremes, node count, and whether the domain
        crosses the antimeridian.
    """
    lat, lon = coords[:, 0], coords[:, 1]
    lon_min, lon_max = float(lon.min()), float(lon.max())
    return {
        "domain": domain,
        "lat_min": float(lat.min()),
        "lat_max": float(lat.max()),
        "lon_min": lon_min,
        "lon_max": lon_max,
        "node_count": len(coords),
        # Alaska runs out past the dateline through the Aleutians, so its
        # longitude box spans -180..180 and would match every point on earth.
        # Recorded for reference; the occupancy-cell gate is what actually
        # decides coverage, precisely because boxes cannot express this.
        "crosses_antimeridian": bool(lon_max - lon_min > 180),
    }


def build_domain_nodes(domain: str, dest: Path, *, coord_scale: int = COORD_SCALE) -> Path:
    """Generate one domain's node parquet from S3 into ``dest``.

    Used as the last-resort fallback when the published index file can be
    neither found locally nor downloaded, and by the maintainer build script.

    Parameters
    ----------
    domain : str
        One of the keys of :data:`.domains.DOMAINS`.
    dest : Path
        Parquet file to write. Parent directories are created.
    coord_scale : int, default ``COORD_SCALE``
        Fixed-point multiplier for the coordinate columns.

    Returns
    -------
    Path
        ``dest``, once written.
    """
    coords = read_coordinates(domain)
    dest.parent.mkdir(parents=True, exist_ok=True)
    write_domain_nodes(coords, dest, coord_scale=coord_scale)
    return dest
