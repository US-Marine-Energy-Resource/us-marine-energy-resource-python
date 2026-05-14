"""Shared DRY components for tidal visualizations."""

import contextlib
from typing import Any

import cmocean  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colormaps
from matplotlib.figure import Figure

from us_marine_energy_resource.analysis.preprocessing import (
    DepthMode,
    sigma_depth_axis_label,
    sigma_depth_scalar,
    sigma_depths_array,
    sigma_layer_depth_col,
)
from us_marine_energy_resource.viz.settings import PlotSettings

with contextlib.suppress(AttributeError, ValueError):
    colormaps.register(name="cmocean_thermal", cmap=cmocean.cm.thermal)  # type: ignore[attr-defined]

# Constants
_N_LAYERS = 10
_COMPLEX_TOLERANCE = 1e-10


def _validate_dataframe(df: pd.DataFrame, depth_mode: DepthMode) -> None:
    """Validate that DataFrame contains expected columns for the requested depth mode.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    depth_mode : DepthMode
        Depth coordinate convention to validate columns for.

    Raises
    ------
    KeyError
        If any required column is absent.
    """
    required_columns = [f"vap_sea_water_speed_layer_{i}" for i in range(_N_LAYERS)]
    required_columns += [sigma_layer_depth_col(i, depth_mode) for i in range(_N_LAYERS)]
    required_columns += ["vap_sea_floor_depth", "vap_surface_elevation"]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def _validate_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Validate that specific columns exist in *df*.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    columns : list of str
        Column names that must be present.

    Raises
    ------
    KeyError
        If any listed column is absent from *df*.
    """
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def _calculate_tidal_statistics(
    df: pd.DataFrame, layer: int, depth_mode: DepthMode
) -> dict[str, Any]:
    """Calculate common tidal statistics for a single sigma layer.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with speed and depth columns for all layers.
    layer : int
        Zero-based sigma layer index.
    depth_mode : DepthMode
        Depth coordinate convention.

    Returns
    -------
    dict
        Keys: ``mean_speed``, ``max_speed``, ``std_speed``, ``depth``.
    """
    speeds: np.ndarray = df[f"vap_sea_water_speed_layer_{layer}"].to_numpy(
        dtype=float, na_value=np.nan
    )
    depth = sigma_depth_scalar(df, layer, depth_mode)

    return {
        "mean_speed": float(np.nanmean(speeds)),
        "max_speed": float(np.nanmax(speeds)),
        "std_speed": float(np.nanstd(speeds)),
        "depth": depth,
    }


def _format_date_time(ax: Any, times: pd.DatetimeIndex, rotation: int = 45) -> None:
    """Format a numeric x-axis with datetime labels.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis whose x-ticks will be reformatted.
    times : pd.DatetimeIndex
        Datetime values corresponding to integer x positions ``0..len(times)-1``.
    rotation : int, optional
        Label rotation in degrees. Default 45.
    """
    if isinstance(times, pd.DatetimeIndex):
        n_ticks = min(12, len(times))
        if n_ticks > 1:
            tick_indices = np.linspace(0, len(times) - 1, n_ticks, dtype=int)
            ax.set_xticks(tick_indices)
            ax.set_xticklabels(
                [times[i].strftime("%Y-%m-%d %H:%M") for i in tick_indices],
                rotation=rotation,
            )


def _setup_standard_grid(fig: Figure, n_rows: int = 1, n_cols: int = 1) -> list[Any]:
    """Create a standard GridSpec layout and return flat list of axes.

    Parameters
    ----------
    fig : Figure
        Parent figure.
    n_rows : int
        Number of grid rows.
    n_cols : int
        Number of grid columns.

    Returns
    -------
    list of Axes
        Axes in row-major order.
    """
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.3, wspace=0.25)
    axes: list[Any] = [fig.add_subplot(gs[i, j]) for i in range(n_rows) for j in range(n_cols)]
    return axes


def _apply_plotting_options(ax: Any, grid: bool = True, legend: bool = False) -> None:
    """Apply standard grid/legend options to *ax*.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    grid : bool
        Whether to add a dashed grid.
    legend : bool
        Whether to add an auto-located legend.
    """
    if grid:
        ax.grid(True, linestyle="--", alpha=0.7)
    if legend:
        ax.legend(loc="best")


def _get_layer_colors(n_layers: int = _N_LAYERS) -> list[tuple[float, float, float]]:
    """Return a viridis color palette for *n_layers* sigma layers.

    Parameters
    ----------
    n_layers : int
        Number of colors required.

    Returns
    -------
    list of RGB tuples
        One color per layer.
    """
    return sns.color_palette("viridis", n_layers)  # type: ignore[return-value]


def _get_sigma_depth_bounds(df: pd.DataFrame, layer: int) -> tuple[float, float]:
    """Return ``(min_depth, max_depth)`` for a sigma layer column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``vap_sigma_depth_layer_{layer}`` column.
    layer : int
        Zero-based layer index.

    Returns
    -------
    tuple of float
        ``(min_depth, max_depth)`` in metres.
    """
    col = df[f"vap_sigma_depth_layer_{layer}"]
    return float(col.min()), float(col.max())


def _safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide *numerator* by *denominator*, returning *default* when near zero.

    Parameters
    ----------
    numerator : float
        Dividend.
    denominator : float
        Divisor.
    default : float
        Value returned when ``abs(denominator) <= _COMPLEX_TOLERANCE``.

    Returns
    -------
    float
        Result of division or *default*.
    """
    if abs(denominator) > _COMPLEX_TOLERANCE:
        return numerator / denominator
    return default


def _trim_time(df: pd.DataFrame, settings: PlotSettings | None) -> pd.DataFrame:
    """Slice *df* to the time window specified in *settings*.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a :class:`pandas.DatetimeIndex`.
    settings : PlotSettings, optional
        When ``None``, or when both dates are ``None``, returns *df* unchanged.

    Returns
    -------
    pd.DataFrame
        A slice of *df* within ``[start_date, end_date]``.  The slice shares
        memory with the original (no copy).

    Raises
    ------
    TypeError
        If *df* does not have a :class:`pandas.DatetimeIndex`.
    """
    if settings is None:
        return df

    start = pd.to_datetime(settings.start_date) if settings.start_date is not None else None
    end = pd.to_datetime(settings.end_date) if settings.end_date is not None else None

    if start is None and end is None:
        return df

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("_trim_time requires a DataFrame with a DatetimeIndex.")

    return df.loc[start:end]  # type: ignore[misc]
