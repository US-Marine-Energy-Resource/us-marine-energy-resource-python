"""Tidal energy resource computation functions."""

from __future__ import annotations

import logging
from typing import Any, TypedDict

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.signal import find_peaks

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
        Reference datum for *target_depth_m*:

        * ``"surface"`` — depth measured **downward** from the sea surface
          (e.g. ``10.0`` means 10 m below the surface).
        * ``"sea_floor"`` — height measured **upward** from the seafloor
          (e.g. ``10.0`` means 10 m above the seabed).

        Default is ``"surface"``.

    Returns
    -------
    layer : int
        Index (0-9) of the best-matching sigma layer.
    layer_mean_depth_m : float
        Mean depth of the selected sigma layer (m from the sea surface).

    Raises
    ------
    ValueError
        If *relative_to* is not ``"surface"`` or ``"sea_floor"``.
    """
    if relative_to not in {"surface", "sea_floor"}:
        raise ValueError(f"relative_to must be 'surface' or 'sea_floor', got {relative_to!r}")

    mean_depths = np.array(
        [float(df[f"vap_sigma_depth_layer_{i}"].mean()) for i in range(_N_LAYERS)]
    )

    if relative_to == "sea_floor":
        mean_seafloor = float(df["vap_sea_floor_depth"].mean())
        abs_depth = mean_seafloor - target_depth_m
    else:
        abs_depth = target_depth_m

    layer = int(np.argmin(np.abs(mean_depths - abs_depth)))
    return layer, float(mean_depths[layer])


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


def calculate_haversine_distance_meters(
    point_a: tuple[float, float],
    point_b: tuple[float, float],
) -> float:
    """Calculate the great-circle distance between two points in metres.

    Delegates to :meth:`TidalManifestQuery.haversine_distance
    <us_marine_energy_resource.manifest.TidalManifestQuery.haversine_distance>`
    and converts the result from kilometres to metres.

    Parameters
    ----------
    point_a : tuple[float, float]
        ``(latitude, longitude)`` of the first point in decimal degrees.
    point_b : tuple[float, float]
        ``(latitude, longitude)`` of the second point in decimal degrees.

    Returns
    -------
    float
        Distance in metres.
    """
    from us_marine_energy_resource.manifest import TidalManifestQuery

    return (
        TidalManifestQuery.haversine_distance(point_a[0], point_a[1], point_b[0], point_b[1])
        * 1000.0
    )


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
