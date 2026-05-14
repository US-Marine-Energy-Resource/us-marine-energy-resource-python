"""Data loading and preprocessing for tidal energy parquet files."""

import enum
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


_N_LAYERS = 10


class DepthMode(enum.Enum):
    """Depth coordinate convention for sigma-layer data.

    ``FixedSurface``
        Depth below the instantaneous sea surface in metres (σ × D).
        ``D = vap_sea_floor_depth`` is the instantaneous total water column depth.
        Stored in ``vap_sigma_depth_layer_{i}`` columns.

    ``FixedBottom``
        Height above the instantaneous seafloor in metres ((1 − σ) × D).
        Stored in ``vap_sigma_height_layer_{i}`` columns.

    ``Navd88Depth``
        Depth below NAVD88 datum in metres (σ × D − ζ), where ζ is
        ``vap_surface_elevation`` (tidal surface elevation relative to NAVD88).
        Stored in ``vap_sigma_depth_navd88_layer_{i}`` columns.
        Requires ``vap_surface_elevation`` in the DataFrame.

    ``Navd88Elevation``
        Height above NAVD88 datum in metres (ζ − σ × D).
        Stored in ``vap_sigma_height_navd88_layer_{i}`` columns.
        Requires ``vap_surface_elevation`` in the DataFrame.
    """

    FixedSurface = enum.auto()
    FixedBottom = enum.auto()
    Navd88Depth = enum.auto()
    Navd88Elevation = enum.auto()


def sigma_layer_depth_col(layer: int, mode: DepthMode) -> str:
    """Return the DataFrame column name for a sigma layer's depth coordinate.

    Parameters
    ----------
    layer : int
        Zero-based sigma layer index (0 = surface, 9 = near-bed).
    mode : DepthMode
        Depth coordinate convention.

    Returns
    -------
    str
        Column name present after :func:`prepare_dataframe` has been called.
    """
    match mode:
        case DepthMode.FixedSurface:
            return f"vap_sigma_depth_layer_{layer}"
        case DepthMode.FixedBottom:
            return f"vap_sigma_height_layer_{layer}"
        case DepthMode.Navd88Depth:
            return f"vap_sigma_depth_navd88_layer_{layer}"
        case DepthMode.Navd88Elevation:
            return f"vap_sigma_height_navd88_layer_{layer}"


def sigma_depth_scalar(df: pd.DataFrame, layer: int, mode: DepthMode) -> float:
    """Return a representative scalar depth coordinate for one sigma layer.

    Returns the time-mean of the depth coordinate column for *layer*.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame prepared by :func:`prepare_dataframe`.
    layer : int
        Zero-based sigma layer index (0 = surface, 9 = near-bed).
    mode : DepthMode
        Depth coordinate convention.

    Returns
    -------
    float
        Time-mean depth coordinate value.
    """
    return float(df[sigma_layer_depth_col(layer, mode)].mean())


def sigma_depths_array(df: pd.DataFrame, mode: DepthMode) -> np.ndarray:
    """Return all sigma layer depths as a 2-D array.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame prepared by :func:`prepare_dataframe`.
    mode : DepthMode
        Depth coordinate convention.

    Returns
    -------
    np.ndarray
        Shape ``(n_timesteps, 10)``.  Each column is one sigma layer.
    """
    cols = [sigma_layer_depth_col(i, mode) for i in range(_N_LAYERS)]
    return df[cols].to_numpy(dtype=float)


def sigma_depth_axis_label(mode: DepthMode) -> str:
    """Return a y-axis label appropriate for the chosen depth mode.

    Parameters
    ----------
    mode : DepthMode
        Depth coordinate convention.

    Returns
    -------
    str
        Human-readable axis label.
    """
    match mode:
        case DepthMode.FixedSurface:
            return "Depth [m from surface]"
        case DepthMode.FixedBottom:
            return "Altitude [m from seafloor]"
        case DepthMode.Navd88Depth:
            return "Depth [m from NAVD88]"
        case DepthMode.Navd88Elevation:
            return "Altitude [m from NAVD88]"


def prepare_dataframe(
    df: pd.DataFrame,
    file_meta: dict[str, str],
) -> pd.DataFrame:
    """Prepare a tidal hindcast DataFrame for analysis and visualization.

    Applies all required preprocessing steps in order:

    1. Adds ``lat_center`` / ``lon_center`` column aliases from ``lat`` / ``lon``.
    2. Adds ``dataset_name`` column from *file_meta*.
    3. Computes sigma depth boundary columns (``vap_sigma_depth_bound_{0..10}``),
       using existing layer depth data if available, otherwise sea floor depth only.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame as returned by :func:`load_parquet`.
    file_meta : dict[str, str]
        File-level metadata dict as returned by :func:`load_parquet`.

    Returns
    -------
    pd.DataFrame
        The input DataFrame mutated with all preprocessing applied.
    """
    if "lat_center" not in df.columns and "lat" in df.columns:
        df["lat_center"] = df["lat"]
    if "lon_center" not in df.columns and "lon" in df.columns:
        df["lon_center"] = df["lon"]

    if "dataset_name" not in df.columns:
        df["dataset_name"] = file_meta.get("dataset_name", "WPTO Tidal Hindcast")

    # Recompute sigma layer depths from first principles.  The stored
    # vap_sigma_depth_layer_{i} values are interpolated model output and
    # cannot be assumed reliable; D = vap_sea_floor_depth = h + ζ is the
    # authoritative instantaneous water-column depth.
    if "vap_sea_floor_depth" in df.columns:
        D = df["vap_sea_floor_depth"]
        for i in range(_N_LAYERS):
            sigma = (i + 0.5) / _N_LAYERS
            df[f"vap_sigma_depth_layer_{i}"] = sigma * D
            df[f"vap_sigma_height_layer_{i}"] = (1 - sigma) * D
            df[f"vap_sigma_relative_depth_layer_{i}"] = sigma  # kept for backward compat

        if "vap_surface_elevation" in df.columns:
            zeta = df["vap_surface_elevation"]
            for i in range(_N_LAYERS):
                sigma = (i + 0.5) / _N_LAYERS
                df[f"vap_sigma_depth_navd88_layer_{i}"] = sigma * D - zeta
                df[f"vap_sigma_height_navd88_layer_{i}"] = zeta - sigma * D

    if "vap_sigma_depth_bound_0" not in df.columns:
        if "vap_sigma_depth_layer_0" in df.columns:
            compute_sigma_bounds_from_layers(df)
        else:
            compute_sigma_bounds_from_seafloor(df)

    return df


def standardize_metadata(
    schema_metadata: dict[Any, Any],
    standardize_values: bool = True,
) -> dict[str, Any]:
    """Standardize PyArrow schema metadata keys and optionally values.

    Decodes byte-string keys and values from PyArrow schema metadata into
    plain Python strings, recursively handling nested dicts and lists.

    Parameters
    ----------
    schema_metadata : dict
        The original metadata dictionary, typically with ``bytes`` keys and
        values as returned by ``pyarrow.Schema.metadata``.
    standardize_values : bool, optional
        Whether to also decode byte-string values. Default is ``True``.

    Returns
    -------
    dict[str, Any]
        A new dictionary with all keys decoded to ``str`` and, if
        ``standardize_values`` is ``True``, all byte values decoded as well.
    """
    standardized: dict[str, Any] = {}

    for key, value in schema_metadata.items():
        std_key = key.decode("utf-8") if isinstance(key, bytes) else str(key)

        if standardize_values:
            if isinstance(value, bytes):
                std_value: Any = value.decode("utf-8")
            elif isinstance(value, dict):
                std_value = standardize_metadata(value, standardize_values=True)
            elif isinstance(value, list):
                std_value = [
                    item.decode("utf-8") if isinstance(item, bytes) else item for item in value
                ]
            else:
                std_value = value
        else:
            std_value = value

        standardized[std_key] = std_value

    return standardized


def load_parquet(
    parquet_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, dict[str, str]]]:
    """Load a tidal hindcast parquet file and its netCDF-compatible metadata.

    Reads the parquet file into a ``pandas.DataFrame`` with a
    ``DatetimeIndex``, and separately extracts file-level (global) and
    variable-level CF metadata stored in the PyArrow schema.

    Parameters
    ----------
    parquet_path : str or Path
        Path to the parquet file to load.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with a ``DatetimeIndex`` named ``"time"`` and columns such
        as ``vap_sea_water_speed_layer_{0..9}``,
        ``vap_sea_water_power_density_layer_{0..9}``,
        ``vap_sigma_depth_layer_{0..9}``, ``vap_surface_elevation``,
        ``vap_sea_floor_depth``, ``lat``, and ``lon``.
    file_meta : dict[str, str]
        Global file-level attributes (e.g. ``title``, ``institution``,
        ``source``), decoded from the PyArrow schema metadata.
    var_meta : dict[str, dict[str, str]]
        Per-variable CF attributes keyed by column name (e.g.
        ``{"vap_surface_elevation": {"units": "m", "long_name": "..."}}``)
    """
    table = pq.read_table(parquet_path)
    result = table.to_pandas().set_index("time", drop=True)

    file_meta: dict[str, str] = {}
    if table.schema.metadata:
        for key, value in table.schema.metadata.items():
            file_meta[key.decode("utf-8")] = value.decode("utf-8")

    var_meta: dict[str, dict[str, str]] = {}
    for field in table.schema:
        if field.metadata:
            field_attrs = {k.decode("utf-8"): v.decode("utf-8") for k, v in field.metadata.items()}
            if field_attrs:
                var_meta[field.name] = field_attrs

    return result, file_meta, var_meta


def compute_sigma_bounds_from_layers(df: pd.DataFrame) -> pd.DataFrame:
    """Compute sigma depth boundary columns from existing layer depth columns.

    Derives 11 sigma depth bound columns (``vap_sigma_depth_bound_0`` through
    ``vap_sigma_depth_bound_10``) by midpointing adjacent
    ``vap_sigma_depth_layer_{i}`` values already present in the DataFrame.

    ``vap_sea_floor_depth`` stores the instantaneous total water column depth
    ``D = h + ζ`` (bathymetric depth *h* plus tidal surface elevation *ζ*
    relative to NAVD88).  ``bound_0`` is fixed at 0 (the free surface) and
    ``bound_10`` equals ``vap_sea_floor_depth`` (the seafloor face of the
    uniform sigma grid, sigma = 1.0).

    .. note::
        A common mistake is to compute ``bound_10`` as
        ``vap_surface_elevation + vap_sea_floor_depth``, which yields
        ``h + 2ζ`` instead of ``h + ζ``.  At low tide (ζ < 0) this causes
        ``bound_10 < bound_9``, inverting the bottom polygon and producing
        sign-flip artefacts in depth-section plots.

    Use this function when the parquet file already contains
    ``vap_sigma_depth_layer_{i}`` columns from the hindcast output.
    For files without those columns, use
    :func:`compute_sigma_bounds_from_seafloor` instead.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ``vap_sigma_depth_layer_{0..9}`` and
        ``vap_sea_floor_depth`` columns.

    Returns
    -------
    pd.DataFrame
        The input DataFrame with 11 new ``vap_sigma_depth_bound_{i}`` columns
        added in-place.
    """
    for i in range(11):
        if i == 0:
            df[f"vap_sigma_depth_bound_{i}"] = 0
        elif i == 10:
            # vap_sea_floor_depth = h + ζ = D (total water column depth).
            # Do NOT add vap_surface_elevation again — that would double-count
            # ζ and produce h + 2ζ, inverting the bottom polygon at low tide.
            df[f"vap_sigma_depth_bound_{i}"] = df["vap_sea_floor_depth"]
        else:
            df[f"vap_sigma_depth_bound_{i}"] = (
                df[f"vap_sigma_depth_layer_{i - 1}"] + df[f"vap_sigma_depth_layer_{i}"]
            ) / 2

    return df


def compute_sigma_bounds_from_seafloor(df: pd.DataFrame) -> pd.DataFrame:
    """Compute sigma layer centers and bounds from sea floor depth only.

    Derives both 10 sigma layer center columns
    (``vap_sigma_depth_layer_{0..9}``) and 11 sigma bound columns
    (``vap_sigma_depth_bound_{0..10}``) assuming uniform sigma spacing based
    solely on ``vap_sea_floor_depth``.  Surface elevation is not used.

    Use this function when ``vap_sigma_depth_layer_{i}`` columns are absent
    from the parquet file and must be synthesised from bathymetry.  If the
    actual hindcast sigma columns are present, use
    :func:`compute_sigma_bounds_from_layers` instead.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a ``vap_sea_floor_depth`` column.

    Returns
    -------
    pd.DataFrame
        The input DataFrame with 10 ``vap_sigma_depth_layer_{i}`` columns and
        11 ``vap_sigma_depth_bound_{i}`` columns added in-place.
    """
    sea_floor = df["vap_sea_floor_depth"]

    for i in range(10):
        sigma_factor = (i + 0.5) / 10
        df[f"vap_sigma_depth_layer_{i}"] = sea_floor * sigma_factor

    for i in range(11):
        if i == 0:
            df[f"vap_sigma_depth_bound_{i}"] = 0
        elif i == 10:
            df[f"vap_sigma_depth_bound_{i}"] = sea_floor
        else:
            df[f"vap_sigma_depth_bound_{i}"] = sea_floor * (i / 10)

    return df


# ---------------------------------------------------------------------------
# Parquet footer — metadata and column statistics without loading row data
# ---------------------------------------------------------------------------


class ColumnStats(TypedDict):
    """Min/max statistics for one column, aggregated across all row groups."""

    col_min: float | None
    col_max: float | None
    null_count: int


class ParquetFooterInfo(TypedDict):
    """Parquet footer metadata: file-level attrs, schema, and column stats."""

    file_meta: dict[str, str]
    var_meta: dict[str, dict[str, str]]
    column_stats: dict[str, ColumnStats]
    num_rows: int
    num_row_groups: int


_FOOTER_SKIP_KEYS = frozenset({"pandas", "ARROW:schema"})


def _extract_parquet_footer_info(metadata: Any) -> ParquetFooterInfo:
    """Extract metadata, schema, and column stats from a PyArrow FileMetaData object.

    Parameters
    ----------
    metadata : pyarrow.parquet.FileMetaData
        Footer metadata object returned by ``pq.read_metadata()``.

    Returns
    -------
    ParquetFooterInfo
        Extracted global metadata, per-variable metadata, per-column min/max
        statistics aggregated across all row groups, row count, and row group count.
    """
    file_meta: dict[str, str] = {}
    if metadata.metadata:
        for k, v in metadata.metadata.items():
            key = k.decode("utf-8") if isinstance(k, bytes) else str(k)
            if key in _FOOTER_SKIP_KEYS:
                continue
            val = v.decode("utf-8") if isinstance(v, bytes) else str(v)
            file_meta[key] = val

    arrow_schema = metadata.schema.to_arrow_schema()

    var_meta: dict[str, dict[str, str]] = {}
    for field in arrow_schema:
        if field.metadata:
            var_meta[field.name] = {
                (k.decode("utf-8") if isinstance(k, bytes) else str(k)): (
                    v.decode("utf-8") if isinstance(v, bytes) else str(v)
                )
                for k, v in field.metadata.items()
            }

    col_names: list[str] = arrow_schema.names
    column_stats: dict[str, ColumnStats] = {}

    for col_idx, col_name in enumerate(col_names):
        col_min: float | None = None
        col_max: float | None = None
        total_null: int = 0

        for rg_idx in range(metadata.num_row_groups):
            chunk = metadata.row_group(rg_idx).column(col_idx)
            stats = chunk.statistics
            if stats is not None:
                total_null += stats.null_count or 0
                if stats.has_min_max:
                    try:
                        rg_min = float(stats.min)
                        rg_max = float(stats.max)
                        col_min = rg_min if col_min is None else min(col_min, rg_min)
                        col_max = rg_max if col_max is None else max(col_max, rg_max)
                    except (TypeError, ValueError):
                        pass

        column_stats[col_name] = ColumnStats(
            col_min=col_min,
            col_max=col_max,
            null_count=total_null,
        )

    return ParquetFooterInfo(
        file_meta=file_meta,
        var_meta=var_meta,
        column_stats=column_stats,
        num_rows=metadata.num_rows,
        num_row_groups=metadata.num_row_groups,
    )


def read_parquet_footer_info(local_path: str | Path) -> ParquetFooterInfo:
    """Read parquet footer metadata from a local file without loading row data.

    Reads only the parquet footer (schema + file-level metadata + column
    statistics) rather than the full file.  Useful for inspecting dataset
    structure and data ranges without a full download.

    Parameters
    ----------
    local_path : str or Path
        Path to the local parquet file.

    Returns
    -------
    ParquetFooterInfo
        Extracted metadata, per-variable attributes, and per-column min/max
        statistics aggregated across all row groups.
    """
    metadata = pq.read_metadata(str(local_path))
    return _extract_parquet_footer_info(metadata)
