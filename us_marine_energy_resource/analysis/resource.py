"""Tidal energy resource computation functions."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, TypedDict

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.signal import find_peaks

from .preprocessing import DepthMode, sigma_depth_scalar

logger = logging.getLogger(__name__)

_N_LAYERS = 10
_DATASET_RHO: float = 1025.0
# Seawater density (kg/m3) used by the H2O High Resolution Tidal Hindcast
# to pre-compute the stored vap_sea_water_power_density_layer_{i} columns.


def compute_power_density(
    speed: np.ndarray,
    rho: float = _DATASET_RHO,
) -> np.ndarray:
    r"""Compute tidal current power density from speed.

    Applies the fluid kinetic energy flux equation:

    .. math::

        P = \tfrac{1}{2} \rho v^3

    This is the same formula used by the H2O High Resolution Tidal Hindcast
    dataset to produce the stored ``vap_sea_water_power_density_layer_{i}``
    columns (dataset default rho = 1025.0 kg/m3).  Passing a different *rho*
    lets callers override that assumption without modifying the underlying
    parquet data.

    Parameters
    ----------
    speed : np.ndarray
        Sea-water speed in m/s.  Any shape is accepted; the result has the same
        shape.
    rho : float, optional
        Seawater density in kg/m³.  Defaults to the dataset value of
        ``1025.0``.

    Returns
    -------
    np.ndarray
        Power density in W/m², same shape as *speed*.
    """
    return 0.5 * rho * speed**3


def select_layer_for_depth(
    df: pd.DataFrame,
    target_depth_m: float,
    relative_to: str = "surface",
    mode: DepthMode | None = None,
) -> tuple[int, float]:
    """Select the sigma layer whose mean depth is closest to a target depth.

    A lightweight alternative to turbine-specific layer selection for cases where
    only a depth is known and no turbine geometry is required.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed tidal hindcast DataFrame containing
        ``vap_sigma_depth_layer_{0..9}`` and ``vap_sea_floor_depth`` columns.
    target_depth_m : float
        Target depth in metres (positive value).
    relative_to : {"surface", "sea_floor"}, optional
        Reference datum for *target_depth_m* used for layer selection:

        * ``"surface"`` — depth measured **downward** from the sea surface
          (e.g. ``10.0`` means 10 m below the surface).
        * ``"sea_floor"`` — height measured **upward** from the seafloor
          (e.g. ``10.0`` means 10 m above the seabed).

        Default is ``"surface"``.
    mode : DepthMode, optional
        Depth coordinate convention for the *returned* depth value.  When
        ``None`` (default) the active global depth perspective is used (set via
        ``set_depth_perspective``).  The layer selection itself always uses
        surface-relative depths internally regardless of this parameter.

    Returns
    -------
    layer : int
        Index (0-9) of the best-matching sigma layer.
    layer_mean_depth_m : float
        Mean depth of the selected sigma layer expressed in the coordinate
        system given by *mode*.

    Raises
    ------
    ValueError
        If *relative_to* is not ``"surface"`` or ``"sea_floor"``.
    """
    if relative_to not in {"surface", "sea_floor"}:
        raise ValueError(f"relative_to must be 'surface' or 'sea_floor', got {relative_to!r}")

    if mode is None:
        from us_marine_energy_resource.viz.settings import get_depth_perspective
        mode = get_depth_perspective().mode

    mean_depths = np.array(
        [float(df[f"vap_sigma_depth_layer_{i}"].mean()) for i in range(_N_LAYERS)]
    )

    if relative_to == "sea_floor":
        mean_seafloor = float(df["vap_sea_floor_depth"].mean())
        abs_depth = mean_seafloor - target_depth_m
    else:
        abs_depth = target_depth_m

    layer = int(np.argmin(np.abs(mean_depths - abs_depth)))
    return layer, sigma_depth_scalar(df, layer, mode)


def compute_power_density_summary(
    df: pd.DataFrame,
    layer: int | None = None,
    rho: float = 1025.0,
    cut_in_speed: float = 0.5,
) -> dict[str, Any]:
    """Compute power density statistics for a tidal current data layer.

    Selects the depth layer with the highest mean power density when
    ``layer`` is not specified.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ``vap_sea_water_speed_layer_{i}``,
        ``vap_sea_water_power_density_layer_{i}`` (used for layer selection
        only), and ``vap_sigma_depth_layer_{i}`` columns.
    layer : int or None, optional
        Depth layer index (0-based) to analyze.  If ``None``, the layer with
        the highest mean power density is selected automatically.
    rho : float, optional
        Seawater density in kg/m³.  Default is ``1025.0``.
    cut_in_speed : float, optional
        Turbine cut-in speed in m/s.  Default is ``0.5``.

    Returns
    -------
    dict[str, Any]
        Dictionary with the following keys:

        - ``layer`` (int): selected depth layer index
        - ``depth`` (float): depth in metres at the selected layer
        - ``mean_speed`` (float): mean current speed in m/s
        - ``p90_speed`` (float): 90th-percentile current speed in m/s
        - ``p95_speed`` (float): 95th-percentile current speed in m/s
        - ``max_speed`` (float): maximum current speed in m/s
        - ``mean_power_density`` (float): mean power density in W/m²
        - ``max_power_density`` (float): maximum power density in W/m²
        - ``usable_time_pct`` (float): % of time above cut-in speed
    """
    if layer is None:
        mean_powers = [
            df[f"vap_sea_water_power_density_layer_{i}"].mean() for i in range(_N_LAYERS)
        ]
        layer = int(np.argmax(mean_powers))

    speeds: np.ndarray = df[f"vap_sea_water_speed_layer_{layer}"].to_numpy()
    power_densities: np.ndarray = compute_power_density(speeds, rho=rho)
    depth: float = float(df[f"vap_sigma_depth_layer_{layer}"].iloc[0])

    usable_time = float(100 * np.mean(speeds >= cut_in_speed))

    return {
        "layer": layer,
        "depth": depth,
        "mean_speed": float(np.mean(speeds)),
        "p90_speed": float(np.nanpercentile(speeds, 90)),
        "p95_speed": float(np.nanpercentile(speeds, 95)),
        "max_speed": float(np.max(speeds)),
        "mean_power_density": float(np.mean(power_densities)),
        "max_power_density": float(np.max(power_densities)),
        "usable_time_pct": usable_time,
    }


def calculate_tidal_periods(
    surface_elevation: ArrayLike,
    times: pd.DatetimeIndex | pd.Series | None = None,
) -> dict[str, Any]:
    """Calculate tidal period statistics from a surface elevation time series.

    Identifies high and low tides using peak detection and computes period
    and tidal range statistics for each consecutive tidal cycle.

    Parameters
    ----------
    surface_elevation : array-like
        Array of modelled water surface elevations in metres.
    times : pd.DatetimeIndex or pd.Series, optional
        Timestamps corresponding to each elevation value.  Required for
        period calculations; if ``None``, period fields will be zero.

    Returns
    -------
    dict[str, Any]
        Dictionary with the following keys:

        - ``average_period_seconds`` (float)
        - ``min_period_seconds`` (float)
        - ``max_period_seconds`` (float)
        - ``average_period_str`` (str): e.g. ``"12.42h"``
        - ``min_period_str`` (str)
        - ``max_period_str`` (str)
        - ``tide_type`` (str): ``"Twice Daily Tides"``, ``"Once Daily Tides"``,
          ``"Mixed Pattern Tides"``, or ``"Unknown"``
        - ``average_range`` (float): mean tidal range in metres
        - ``min_range`` (float)
        - ``max_range`` (float)
        - ``min_range_cycle`` (dict or None): cycle data for smallest range
        - ``max_range_cycle`` (dict or None): cycle data for largest range
        - ``tidal_ranges`` (list[float])
        - ``cycle_data`` (list[dict])
    """
    _empty: dict[str, Any] = {
        "average_period_seconds": 0,
        "min_period_seconds": 0,
        "max_period_seconds": 0,
        "average_period_str": "0.00h",
        "min_period_str": "0.00h",
        "max_period_str": "0.00h",
        "tide_type": "Unknown",
        "average_range": 0,
        "min_range": 0,
        "max_range": 0,
        "min_range_cycle": None,
        "max_range_cycle": None,
        "tidal_ranges": [],
        "cycle_data": [],
    }

    elev = np.asarray(surface_elevation)
    high_tide_indices, _ = find_peaks(elev, prominence=0.05)
    low_tide_indices, _ = find_peaks(-elev, prominence=0.05)
    high_tide_indices = np.sort(high_tide_indices)
    low_tide_indices = np.sort(low_tide_indices)

    if len(high_tide_indices) < 2:
        return _empty

    high_tide_periods: list[float] = []
    tidal_ranges: list[float] = []
    tidal_cycles_data: list[dict[str, Any]] = []

    for i in range(1, len(high_tide_indices)):
        prev_idx = high_tide_indices[i - 1]
        curr_idx = high_tide_indices[i]

        between_low = low_tide_indices[
            (low_tide_indices > prev_idx) & (low_tide_indices < curr_idx)
        ]

        if len(between_low) > 0:
            lowest_low_idx = between_low[np.argmin(elev[between_low])]
            tidal_range = float(elev[prev_idx] - elev[lowest_low_idx])
            tidal_ranges.append(tidal_range)

            high_tide_time = None
            low_tide_time = None
            if times is not None:
                try:
                    if isinstance(times, pd.DatetimeIndex):
                        high_tide_time = times[prev_idx]
                        low_tide_time = times[lowest_low_idx]
                    elif hasattr(times, "iloc"):
                        high_tide_time = times.iloc[prev_idx]
                        low_tide_time = times.iloc[lowest_low_idx]
                    else:
                        high_tide_time = times[prev_idx]  # type: ignore[index]
                        low_tide_time = times[lowest_low_idx]  # type: ignore[index]
                except (IndexError, KeyError):
                    pass

            tidal_cycles_data.append(
                {
                    "high_tide_index": prev_idx,
                    "high_tide_value": float(elev[prev_idx]),
                    "high_tide_time": high_tide_time,
                    "low_tide_index": lowest_low_idx,
                    "low_tide_value": float(elev[lowest_low_idx]),
                    "low_tide_time": low_tide_time,
                    "tidal_range": tidal_range,
                }
            )

        if times is not None:
            try:
                if isinstance(times, pd.DatetimeIndex):
                    time_diff = (times[curr_idx] - times[prev_idx]).total_seconds()
                elif hasattr(times, "iloc"):
                    time_diff = (times.iloc[curr_idx] - times.iloc[prev_idx]).total_seconds()
                else:
                    time_diff = (times[curr_idx] - times[prev_idx]).total_seconds()  # type: ignore[index]

                if (10 * 3600 < time_diff < 14 * 3600) or (20 * 3600 < time_diff < 26 * 3600):
                    high_tide_periods.append(float(time_diff))
            except (TypeError, AttributeError):
                continue

    if not high_tide_periods:
        return {**_empty, "cycle_data": tidal_cycles_data}

    avg_period = float(np.mean(high_tide_periods))
    min_period = float(np.min(high_tide_periods))
    max_period = float(np.max(high_tide_periods))

    if 10 * 3600 < avg_period < 14 * 3600:
        tide_type = "Twice Daily Tides"
    elif 20 * 3600 < avg_period < 26 * 3600:
        tide_type = "Once Daily Tides"
    else:
        tide_type = "Mixed Pattern Tides"

    def _fmt(seconds: float) -> str:
        return f"{seconds / 3600:.2f}h"

    avg_range = float(np.mean(tidal_ranges)) if tidal_ranges else 0.0
    min_range = float(np.min(tidal_ranges)) if tidal_ranges else 0.0
    max_range = float(np.max(tidal_ranges)) if tidal_ranges else 0.0
    min_range_cycle = tidal_cycles_data[int(np.argmin(tidal_ranges))] if tidal_ranges else None
    max_range_cycle = tidal_cycles_data[int(np.argmax(tidal_ranges))] if tidal_ranges else None

    return {
        "average_period_seconds": avg_period,
        "min_period_seconds": min_period,
        "max_period_seconds": max_period,
        "average_period_str": _fmt(avg_period),
        "min_period_str": _fmt(min_period),
        "max_period_str": _fmt(max_period),
        "tide_type": tide_type,
        "average_range": avg_range,
        "min_range": min_range,
        "max_range": max_range,
        "min_range_cycle": min_range_cycle,
        "max_range_cycle": max_range_cycle,
        "tidal_ranges": tidal_ranges,
        "cycle_data": tidal_cycles_data,
    }


def calculate_tidal_levels(
    surface_positions: ArrayLike,
    times: pd.DatetimeIndex | pd.Series | None = None,
) -> dict[str, Any]:
    """Calculate tidal reference levels from a surface elevation record.

    Identifies high and low tides via peak detection and derives standard
    tidal datums using plain-language keys.  Falls back to the top/bottom
    20th percentile when no peaks are detected (e.g. very short records).

    Parameters
    ----------
    surface_positions : array-like
        Modelled water surface elevations in metres.
    times : pd.DatetimeIndex or pd.Series, optional
        Timestamps (unused in computation; reserved for future use).

    Returns
    -------
    dict[str, Any]
        Dictionary with the following keys:

        - ``"Max High Tide"`` (float): highest recorded high tide
        - ``"Min High Tide"`` (float): lowest recorded high tide
        - ``"Mean High Tide"`` (float): mean of all high tides
        - ``"Mean Water Level"`` (float): mean of all water levels
        - ``"Max Low Tide"`` (float): highest recorded low tide
        - ``"Mean Low Tide"`` (float): mean of all low tides
        - ``"Min Low Tide"`` (float): lowest recorded low tide
        - ``"high_tide_indices"`` (np.ndarray): indices of high tide peaks
        - ``"low_tide_indices"`` (np.ndarray): indices of low tide troughs
    """
    elev = np.asarray(surface_positions, dtype=float)
    model_msl = float(np.mean(elev))

    high_tide_indices, _ = find_peaks(elev, prominence=0.05)
    low_tide_indices, _ = find_peaks(-elev, prominence=0.05)

    if len(high_tide_indices) == 0 or len(low_tide_indices) == 0:
        logger.warning(
            "Could not detect tidal peaks and troughs; falling back to top/bottom 20th percentile."
        )
        n20 = max(1, int(len(elev) * 0.2))
        sorted_elev = np.sort(elev)
        high_tides = sorted_elev[-n20:]
        low_tides = sorted_elev[:n20]
        high_tide_indices = np.argsort(elev)[-n20:]
        low_tide_indices = np.argsort(elev)[:n20]
    else:
        high_tides = elev[high_tide_indices]
        low_tides = elev[low_tide_indices]

    return {
        "Max High Tide": float(np.max(high_tides)),
        "Min High Tide": float(np.min(high_tides)),
        "Mean High Tide": float(np.mean(high_tides)),
        "Mean Water Level": model_msl,
        "Max Low Tide": float(np.max(low_tides)),
        "Mean Low Tide": float(np.mean(low_tides)),
        "Min Low Tide": float(np.min(low_tides)),
        "high_tide_indices": high_tide_indices,
        "low_tide_indices": low_tide_indices,
    }




class SiteSummaryMetrics(TypedDict):
    """Aggregated resource metrics for one tidal candidate site.

    Returned by :func:`collect_site_metrics`.  All speed and power values
    are for the depth layer with the highest mean power density.
    """

    site_name: str
    mean_speed: float
    p90_speed: float
    p95_speed: float
    max_speed: float
    mean_power_density: float
    usable_time_pct: float
    average_tidal_range: float
    lat: float
    lon: float


def collect_site_metrics(
    df: pd.DataFrame,
    site_name: str,
    rho: float = 1025.0,
    cut_in_speed: float = 0.5,
) -> SiteSummaryMetrics:
    """Compute a concise resource summary for one tidal candidate site.

    Calls :func:`compute_power_density_summary` (auto-selecting the best
    depth layer) and :func:`calculate_tidal_periods` to populate a
    :class:`SiteSummaryMetrics` dictionary suitable for multi-site comparison.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed tidal hindcast DataFrame containing all ten sigma-layer
        speed, power-density, and depth columns, plus ``vap_surface_elevation``.
    site_name : str
        Human-readable label for this site, used in plots and tables.
    rho : float, optional
        Seawater density in kg/m³.  Default is ``1025.0``.
    cut_in_speed : float, optional
        Minimum current speed to count as usable resource (m/s).
        Default is ``0.5``.

    Returns
    -------
    SiteSummaryMetrics
        Dictionary of aggregated resource metrics for this site.
    """
    power_summary = compute_power_density_summary(
        df,
        rho=rho,
        cut_in_speed=cut_in_speed,
    )
    period_stats = calculate_tidal_periods(
        df["vap_surface_elevation"],
        times=df.index if isinstance(df.index, pd.DatetimeIndex) else None,
    )
    lat = float(df["lat"].iloc[0]) if "lat" in df.columns else 0.0
    lon = float(df["lon"].iloc[0]) if "lon" in df.columns else 0.0

    return {
        "site_name": site_name,
        "mean_speed": power_summary["mean_speed"],
        "p90_speed": power_summary["p90_speed"],
        "p95_speed": power_summary["p95_speed"],
        "max_speed": power_summary["max_speed"],
        "mean_power_density": power_summary["mean_power_density"],
        "usable_time_pct": power_summary["usable_time_pct"],
        "average_tidal_range": period_stats["average_range"],
        "lat": lat,
        "lon": lon,
    }


# ---------------------------------------------------------------------------
# Column categorization and footer-based statistics
# ---------------------------------------------------------------------------

# (display_name, filter_key, column_prefix, units, is_directional)
_LAYER_CATEGORIES: list[tuple[str, str, str, str, bool]] = [
    ("Speed",         "speed",     "vap_sea_water_speed_layer_",          "m/s",  False),
    ("Direction",     "direction", "vap_sea_water_to_direction_layer_",   "°",    True),
    ("Power Density", "power",     "vap_sea_water_power_density_layer_",  "W/m²", False),
    ("Depth",         "depth",     "vap_sigma_depth_layer_",              "m",    False),
    ("Height",        "depth",     "vap_sigma_height_layer_",             "m",    False),
]

# (display_name, filter_key, column_names, units)
_SCALAR_CATEGORIES: list[tuple[str, str, list[str], str]] = [
    ("Water Level", "depth",    ["vap_surface_elevation"], "m"),
    ("Sea Floor",   "depth",    ["vap_sea_floor_depth"],   "m"),
    ("Position",    "position", ["lat", "lon"],             "°"),
]


class CategoryInfo(TypedDict):
    """Metadata about one group of related columns."""

    name: str
    filter_key: str
    columns: list[str]
    n: int
    units: str
    is_directional: bool
    is_layered: bool
    prefix: str
    pattern: str


class StatRow(TypedDict):
    """One row in the footer-statistics display table."""

    category: str
    filter_key: str
    layer_label: str
    col_min: float | None
    col_max: float | None
    units: str
    is_directional: bool


def categorize_columns(columns: Sequence[str]) -> list[CategoryInfo]:
    """Group DataFrame column names by physical category.

    Matches columns against the known H2O tidal hindcast naming conventions
    and returns a list of :class:`CategoryInfo` records suitable for display.

    Parameters
    ----------
    columns : sequence of str
        Column names present in the parquet file.

    Returns
    -------
    list of CategoryInfo
        One entry per detected category, in a fixed display order.
        Categories whose columns are entirely absent from *columns* are omitted.
    """
    available = set(columns)
    result: list[CategoryInfo] = []

    for display_name, filter_key, prefix, units, is_dir in _LAYER_CATEGORIES:
        cols = [f"{prefix}{i}" for i in range(_N_LAYERS) if f"{prefix}{i}" in available]
        if not cols:
            continue
        n = len(cols)
        pattern = f"{prefix}0 … _{n - 1}" if n > 1 else cols[0]
        result.append(
            CategoryInfo(
                name=display_name,
                filter_key=filter_key,
                columns=cols,
                n=n,
                units=units,
                is_directional=is_dir,
                is_layered=True,
                prefix=prefix,
                pattern=pattern,
            )
        )

    for display_name, filter_key, col_names, units in _SCALAR_CATEGORIES:
        cols = [c for c in col_names if c in available]
        if not cols:
            continue
        result.append(
            CategoryInfo(
                name=display_name,
                filter_key=filter_key,
                columns=cols,
                n=len(cols),
                units=units,
                is_directional=False,
                is_layered=False,
                prefix="",
                pattern=", ".join(cols),
            )
        )

    return result


def _aggregate_col_stats(
    footer_infos: list[dict[str, Any]],
    col: str,
) -> tuple[float | None, float | None]:
    """Return global min/max for *col* across all footer infos."""
    col_min: float | None = None
    col_max: float | None = None
    for info in footer_infos:
        s = info["column_stats"].get(col)
        if s:
            if s["col_min"] is not None:
                col_min = s["col_min"] if col_min is None else min(col_min, s["col_min"])
            if s["col_max"] is not None:
                col_max = s["col_max"] if col_max is None else max(col_max, s["col_max"])
    return col_min, col_max


def _aggregate_depth_avg(
    footer_infos: list[dict[str, Any]],
    prefix: str,
) -> tuple[float | None, float | None]:
    """Return the depth-averaged min/max across all 10 sigma layers and all files."""
    layer_mins: list[float] = []
    layer_maxes: list[float] = []
    for i in range(_N_LAYERS):
        mn, mx = _aggregate_col_stats(footer_infos, f"{prefix}{i}")
        if mn is not None:
            layer_mins.append(mn)
        if mx is not None:
            layer_maxes.append(mx)
    col_min = float(np.mean(layer_mins)) if layer_mins else None
    col_max = float(np.mean(layer_maxes)) if layer_maxes else None
    return col_min, col_max


def _find_layer_for_depth(
    footer_infos: list[dict[str, Any]],
    target_depth: float,
) -> int:
    """Find the sigma layer index whose midpoint depth is closest to *target_depth*.

    Uses ``vap_sigma_depth_layer_{i}`` column statistics (min/max midpoint)
    from the footer as a proxy for mean depth.  Approximate — actual layer
    depths vary with tidal stage and location.
    """
    midpoints: list[float] = []
    for i in range(_N_LAYERS):
        col = f"vap_sigma_depth_layer_{i}"
        mn, mx = _aggregate_col_stats(footer_infos, col)
        if mn is not None and mx is not None:
            midpoints.append((mn + mx) / 2.0)
        else:
            midpoints.append(float("nan"))

    diffs = [
        abs(m - target_depth) if not np.isnan(m) else float("inf") for m in midpoints
    ]
    return int(np.argmin(diffs))


def _layer_label(layer: int) -> str:
    if layer == 0:
        return "0 (surf.)"
    if layer == _N_LAYERS - 1:
        return f"{layer} (bed)"
    return str(layer)


def compute_footer_stats(
    footer_infos: list[dict[str, Any]],
    categories: list[CategoryInfo],
    layers: list[int],
    depth_target: float | None = None,
    depth_avg: bool = False,
) -> tuple[list[StatRow], str]:
    """Compute per-category statistics from parquet footer row-group data.

    Statistics are derived entirely from the parquet footer (row-group min/max),
    so no full file download is required.  For multiple files the global min
    and max across all files are returned.

    Parameters
    ----------
    footer_infos : list of dict
        One :class:`ParquetFooterInfo`-shaped dict per matched parquet file.
    categories : list of CategoryInfo
        Column categories to include (as returned by :func:`categorize_columns`).
    layers : list of int
        Sigma layer indices to display (0 = surface, 9 = near-bed).
    depth_target : float, optional
        If provided, find and use the sigma layer nearest to this depth (m from
        surface). Overrides *layers*.
    depth_avg : bool
        If True, average statistics across all 10 sigma layers.

    Returns
    -------
    tuple of (list[StatRow], str)
        Stat rows for the display table and a human-readable layer-selection label.
    """
    if depth_avg:
        resolved: list[int] = []
        layer_label_str = "Depth Average (all layers)"
    elif depth_target is not None:
        best = _find_layer_for_depth(footer_infos, depth_target)
        resolved = [best]
        layer_label_str = f"~{depth_target:.1f} m (layer {best})"
    else:
        resolved = layers if layers else [0]
        if resolved == [0]:
            layer_label_str = "Surface Layer (layer 0)"
        elif len(resolved) == 1:
            layer_label_str = f"Layer {resolved[0]}"
        else:
            layer_label_str = f"Layers {', '.join(map(str, resolved))}"

    rows: list[StatRow] = []

    for cat in categories:
        if cat["is_layered"]:
            if depth_avg:
                col_min, col_max = _aggregate_depth_avg(footer_infos, cat["prefix"])
                rows.append(
                    StatRow(
                        category=cat["name"],
                        filter_key=cat["filter_key"],
                        layer_label="avg",
                        col_min=col_min,
                        col_max=col_max,
                        units=cat["units"],
                        is_directional=cat["is_directional"],
                    )
                )
            else:
                for lyr in resolved:
                    col = f"{cat['prefix']}{lyr}"
                    col_min, col_max = _aggregate_col_stats(footer_infos, col)
                    rows.append(
                        StatRow(
                            category=cat["name"],
                            filter_key=cat["filter_key"],
                            layer_label=_layer_label(lyr),
                            col_min=col_min,
                            col_max=col_max,
                            units=cat["units"],
                            is_directional=cat["is_directional"],
                        )
                    )
        else:
            for col in cat["columns"]:
                col_min, col_max = _aggregate_col_stats(footer_infos, col)
                lbl = col if len(cat["columns"]) > 1 else "—"
                rows.append(
                    StatRow(
                        category=cat["name"],
                        filter_key=cat["filter_key"],
                        layer_label=lbl,
                        col_min=col_min,
                        col_max=col_max,
                        units=cat["units"],
                        is_directional=False,
                    )
                )

    return rows, layer_label_str
