"""Polar and rose plots for tidal current direction analysis."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from windrose import WindroseAxes  # type: ignore[import-untyped]

from us_marine_energy_resource.viz._style import styled
from us_marine_energy_resource.viz.settings import PlotSettings

from ._components import _validate_columns


@styled
def plot_current_rose(
    df: pd.DataFrame,
    settings: PlotSettings | None = None,
    layer: int = 0,
    bins: int = 16,
    vmax: float | None = None,
) -> Figure:
    """Plot a stacked-bar current rose using native matplotlib polar projection.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ``vap_sea_water_speed_layer_{layer}``,
        ``vap_sea_water_to_direction_layer_{layer}``, and
        ``vap_sigma_depth_layer_{layer}`` columns.
    settings : PlotSettings, optional
        See :class:`~us_marine_energy_resource.viz.settings.PlotSettings`.
    layer : int, optional
        Sigma layer index to visualize. Default 0.
    bins : int, optional
        Number of direction bins. Default 16.
    vmax : float, optional
        Upper bound for velocity color scaling. Defaults to the data maximum
        rounded up to the nearest 0.1 m/s.

    Returns
    -------
    fig : Figure
        The created matplotlib figure.

    Raises
    ------
    KeyError
        If required columns are absent from *df*.
    """
    _validate_columns(
        df,
        [
            f"vap_sea_water_speed_layer_{layer}",
            f"vap_sea_water_to_direction_layer_{layer}",
            f"vap_sigma_depth_layer_{layer}",
        ],
    )

    speeds: np.ndarray = df[f"vap_sea_water_speed_layer_{layer}"].to_numpy(
        dtype=float, na_value=np.nan
    )
    directions: np.ndarray = df[f"vap_sea_water_to_direction_layer_{layer}"].to_numpy(
        dtype=float, na_value=np.nan
    )
    depth = float(df[f"vap_sigma_depth_layer_{layer}"].iloc[0])

    directions_rad = np.deg2rad(directions)
    bin_width = 2 * np.pi / bins
    direction_bins = np.linspace(0, 2 * np.pi, bins + 1)

    if vmax is None:
        vmax = float(np.ceil(np.nanmax(speeds) * 10) / 10)

    vel_bins = np.linspace(0, vmax, 6)
    freq = np.zeros((len(vel_bins) - 1, bins))

    for i in range(len(vel_bins) - 1):
        mask = (speeds >= vel_bins[i]) & (speeds < vel_bins[i + 1])
        for j in range(bins):
            dir_mask = (directions_rad >= direction_bins[j]) & (
                directions_rad < direction_bins[j + 1]
            )
            freq[i, j] = float(np.sum(mask & dir_mask))

    freq = freq / len(speeds) * 100

    fig = plt.figure(figsize=(10, 10))
    ax: Any = fig.add_subplot(111, projection="polar")

    width = bin_width * 0.8
    colors = plt.cm.viridis(np.linspace(0, 1, len(vel_bins) - 1))  # type: ignore[attr-defined]

    for i in range(len(vel_bins) - 1):
        ax.bar(
            direction_bins[:-1],
            freq[i],
            width=width,
            bottom=0.0 if i == 0 else np.sum(freq[:i], axis=0),
            color=colors[i],
            alpha=0.8,
            label=f"{vel_bins[i]:.1f}-{vel_bins[i + 1]:.1f} m/s",
        )

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
    plt.title(f"Current Rose at {depth:.1f} m Depth (Layer {layer})")
    plt.legend(loc="lower right", bbox_to_anchor=(1.1, -0.1))
    return fig


@styled
def plot_tidal_rose(
    df: pd.DataFrame,
    settings: PlotSettings | None = None,
    layer: int = 4,
) -> Figure:
    """Plot a windrose-style current rose using ``WindroseAxes``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ``vap_sea_water_speed_layer_{layer}``,
        ``vap_sea_water_to_direction_layer_{layer}``, and
        ``vap_sigma_depth_layer_{layer}`` columns.
    settings : PlotSettings, optional
        See :class:`~us_marine_energy_resource.viz.settings.PlotSettings`.
    layer : int, optional
        Sigma layer index to visualize. Default 4 (mid-column).

    Returns
    -------
    fig : Figure
        The created matplotlib figure.

    Raises
    ------
    KeyError
        If required columns are absent from *df*.
    """
    _validate_columns(
        df,
        [
            f"vap_sea_water_speed_layer_{layer}",
            f"vap_sea_water_to_direction_layer_{layer}",
            f"vap_sigma_depth_layer_{layer}",
        ],
    )

    speeds: np.ndarray = df[f"vap_sea_water_speed_layer_{layer}"].to_numpy(
        dtype=float, na_value=np.nan
    )
    directions: np.ndarray = df[f"vap_sea_water_to_direction_layer_{layer}"].to_numpy(
        dtype=float, na_value=np.nan
    )
    depth = float(df[f"vap_sigma_depth_layer_{layer}"].mean())

    fig = plt.figure(figsize=(10, 10))
    rect = [0.1, 0.1, 0.8, 0.8]
    ax = WindroseAxes(fig, rect)
    fig.add_axes(ax)

    ax.bar(
        directions,
        speeds,
        normed=True,
        opening=0.8,
        edgecolor="white",
        cmap=plt.cm.viridis,  # type: ignore[attr-defined]
    )
    ax.set_legend(title="Speed [m/s]")
    ax.set_title(f"Tidal Current Rose at {depth:.1f} m Depth")
    return fig
