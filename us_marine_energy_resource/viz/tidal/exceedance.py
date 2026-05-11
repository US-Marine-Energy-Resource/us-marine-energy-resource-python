"""Velocity, power, and combined exceedance plots for tidal energy assessment."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from us_marine_energy_resource.viz._style import styled
from us_marine_energy_resource.viz.settings import PlotSettings

from ._components import _N_LAYERS


@styled
def plot_velocity_exceedance(
    df: pd.DataFrame,
    settings: PlotSettings | None = None,
    layers: list[int] | None = None,
    key_percentiles: list[float] | None = None,
    annotate: bool = False,
) -> tuple[Figure, dict[str, dict[str, float | int]]]:
    """Plot velocity exceedance curves for tidal current data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ``vap_sea_water_speed_layer_{0..9}`` and
        ``vap_sigma_depth_layer_{0..9}`` columns.
    settings : PlotSettings, optional
        See :class:`~us_marine_energy_resource.viz.settings.PlotSettings`.
    layers : list of int, optional
        Depth layers to plot. If ``None``, plots all 10 layers (0-9).
    key_percentiles : list of float, optional
        List of percentiles to highlight. If ``None``, uses
        ``[50, 25, 10, 5, 1, 0.1]`` for max annotations and
        ``[50, 25, 10]`` for min annotations.
    annotate : bool, optional
        If ``True``, draw min/max percentile markers and callout boxes on the
        curves.  Defaults to ``False`` for a cleaner plot.

    Returns
    -------
    fig : Figure
        The created matplotlib figure.
    stats : dict
        Nested dict keyed by ``"Layer {i}"`` containing per-layer exceedance
        statistics: one entry per percentile key, plus ``"mean"`` and ``"max"``.

    Raises
    ------
    KeyError
        If expected speed or depth columns are absent from *df*.
    """
    if layers is None:
        layers = list(range(_N_LAYERS))

    max_key_percentiles: list[float] = [50, 25, 10, 5, 1, 0.1]
    min_key_percentiles: list[float] = [50, 25, 10]
    if key_percentiles is None:
        key_percentiles = list(set(max_key_percentiles + min_key_percentiles))

    fig, ax = plt.subplots(figsize=(16, 9))
    colors = sns.color_palette("viridis", len(layers))

    stats: dict[str, dict[str, float | int]] = {}
    percentile_values: dict[float, dict[str, float | int | None]] = {
        p: {
            "min": float("inf"),
            "max": -float("inf"),
            "min_layer": None,
            "max_layer": None,
        }
        for p in key_percentiles
    }

    for i, layer in enumerate(layers):
        velocities: np.ndarray = df[f"vap_sea_water_speed_layer_{layer}"].to_numpy(
            dtype=float, na_value=np.nan
        )
        depth: float = float(df[f"vap_sigma_depth_layer_{layer}"].iloc[0])

        sorted_velocities = np.sort(velocities)[::-1]
        exceedance_pct = np.arange(1, len(sorted_velocities) + 1) / len(sorted_velocities) * 100

        ax.plot(
            sorted_velocities,
            exceedance_pct,
            "-",
            color=colors[i],
            label=f"Layer {layer} (~{depth:.1f} m)",
            linewidth=2,
        )

        layer_stats: dict[str, float | int] = {}
        for percentile in key_percentiles:
            exceeded_velocity = float(np.percentile(velocities, 100 - percentile))
            layer_stats[f"{percentile}%"] = exceeded_velocity

            pv = percentile_values[percentile]
            if exceeded_velocity < float(pv["min"]):  # type: ignore[arg-type]
                pv["min"] = exceeded_velocity
                pv["min_layer"] = layer
            if exceeded_velocity > float(pv["max"]):  # type: ignore[arg-type]
                pv["max"] = exceeded_velocity
                pv["max_layer"] = layer

        layer_stats["mean"] = float(np.nanmean(velocities))
        layer_stats["max"] = float(np.nanmax(velocities))
        stats[f"Layer {layer}"] = layer_stats

    # Annotate the fastest layer at each max percentile
    if annotate:
        for percentile in max_key_percentiles:
            max_val = float(percentile_values[percentile]["max"])  # type: ignore[arg-type]
            max_layer = int(percentile_values[percentile]["max_layer"])  # type: ignore[arg-type]
            color_idx = layers.index(max_layer)

            ax.plot(
                max_val,
                percentile,
                "o",
                color=colors[color_idx],
                markersize=8,
                markeredgecolor="black",
                markeredgewidth=1,
            )
            ax.annotate(
                f"{percentile}% Max: {max_val:.2f} m/s\n(Layer {max_layer})",
                xy=(max_val, percentile),
                xytext=(max_val + 0.1, percentile + 8),
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightcoral", "alpha": 0.7},
                arrowprops={"arrowstyle": "->", "color": "red"},
                fontsize=9,
                ha="center",
            )

        # Annotate the slowest layer at each min percentile
        for percentile in min_key_percentiles:
            min_val = float(percentile_values[percentile]["min"])  # type: ignore[arg-type]
            min_layer = int(percentile_values[percentile]["min_layer"])  # type: ignore[arg-type]
            color_idx = layers.index(min_layer)

            ax.plot(
                min_val,
                percentile,
                "o",
                color=colors[color_idx],
                markersize=8,
                markeredgecolor="black",
                markeredgewidth=1,
            )
            ax.annotate(
                f"{percentile}% Min: {min_val:.2f} m/s\n(Layer {min_layer})",
                xy=(min_val, percentile),
                xytext=(min_val - 0.1, percentile - 8),
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightblue", "alpha": 0.7},
                arrowprops={"arrowstyle": "->", "color": "blue"},
                fontsize=9,
                ha="center",
            )

    ax.set_xlabel("Sea Water Speed [m/s]", fontsize=14)
    ax.set_ylabel("Probability of Exceedance [%]", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
        framealpha=0.9,
    )
    ax.set_ylim(0, 100)
    ax.set_xlim(left=0)
    plt.tight_layout()

    return fig, stats


@styled
def plot_power_exceedance(
    df: pd.DataFrame,
    settings: PlotSettings | None = None,
    layers: list[int] | None = None,
    key_percentiles: list[float] | None = None,
) -> tuple[Figure, dict[str, dict[str, float | int]]]:
    """Plot power density exceedance curves for tidal current data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ``vap_sea_water_power_density_layer_{0..9}`` and
        ``vap_sigma_depth_layer_{0..9}`` columns.
    settings : PlotSettings, optional
        See :class:`~us_marine_energy_resource.viz.settings.PlotSettings`.
    layers : list of int, optional
        Depth layers to plot. If ``None``, plots all 10 layers (0-9).
    key_percentiles : list of float, optional
        List of percentiles to highlight. If ``None``, uses
        ``[50, 25, 10, 5, 1, 0.1]``.

    Returns
    -------
    fig : Figure
        The created matplotlib figure.
    stats : dict
        Nested dict keyed by ``"Layer {i}"`` containing per-layer exceedance
        statistics: one entry per percentile key, plus ``"mean"`` and ``"max"``.

    Raises
    ------
    KeyError
        If expected power density or depth columns are absent from *df*.
    """
    if layers is None:
        layers = list(range(_N_LAYERS))

    if key_percentiles is None:
        key_percentiles = [50, 25, 10, 5, 1, 0.1]

    fig, ax = plt.subplots(figsize=(16, 9))
    colors = sns.color_palette("viridis", len(layers))

    stats: dict[str, dict[str, float | int]] = {}

    for i, layer in enumerate(layers):
        power: np.ndarray = df[f"vap_sea_water_power_density_layer_{layer}"].to_numpy(
            dtype=float, na_value=np.nan
        )
        depth: float = float(df[f"vap_sigma_depth_layer_{layer}"].iloc[0])

        sorted_power = np.sort(power)[::-1]
        exceedance_pct = np.arange(1, len(sorted_power) + 1) / len(sorted_power) * 100

        ax.plot(
            sorted_power,
            exceedance_pct,
            "-",
            color=colors[i],
            label=f"Layer {layer} (~{depth:.1f} m)",
            linewidth=2,
        )

        layer_stats: dict[str, float | int] = {}
        for percentile in key_percentiles:
            exceeded_power = float(np.percentile(power, 100 - percentile))
            layer_stats[f"{percentile}%"] = exceeded_power

        layer_stats["mean"] = float(np.nanmean(power))
        layer_stats["max"] = float(np.nanmax(power))
        stats[f"Layer {layer}"] = layer_stats

    ax.set_xlabel("Power Density [W/m²]", fontsize=14)
    ax.set_ylabel("Probability of Exceedance [%]", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
        framealpha=0.9,
    )
    ax.set_ylim(0, 100)
    ax.set_xlim(left=0)
    plt.tight_layout()

    return fig, stats


@styled
def plot_tidal_exceedance(
    df: pd.DataFrame,
    settings: PlotSettings | None = None,
    layers: list[int] | None = None,
) -> Figure:
    """Side-by-side speed and power density exceedance curves.

    Left panel: current speed exceedance with 1.0 and 2.0 m/s reference lines.
    Right panel: power density exceedance on a log y-scale.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ``vap_sea_water_speed_layer_{i}``,
        ``vap_sea_water_power_density_layer_{i}``, and
        ``vap_sigma_depth_layer_{i}`` columns for requested layers.
    settings : PlotSettings, optional
        See :class:`~us_marine_energy_resource.viz.settings.PlotSettings`.
    layers : list of int, optional
        Sigma layers to plot. Defaults to ``[0, 4, 9]``.

    Returns
    -------
    fig : Figure
        The created matplotlib figure.

    Raises
    ------
    KeyError
        If required columns are absent from *df*.
    """
    from ._components import _validate_columns

    if layers is None:
        layers = [0, 4, 9]

    required: list[str] = []
    for layer in layers:
        required += [
            f"vap_sea_water_speed_layer_{layer}",
            f"vap_sea_water_power_density_layer_{layer}",
            f"vap_sigma_depth_layer_{layer}",
        ]
    _validate_columns(df, required)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    colors = plt.cm.viridis(np.linspace(0, 1, len(layers)))  # type: ignore[attr-defined]

    for i, layer in enumerate(layers):
        speeds = df[f"vap_sea_water_speed_layer_{layer}"].to_numpy(dtype=float, na_value=np.nan)
        power = df[f"vap_sea_water_power_density_layer_{layer}"].to_numpy(
            dtype=float, na_value=np.nan
        )
        depth = float(df[f"vap_sigma_depth_layer_{layer}"].mean())
        n = len(speeds)
        exc = np.arange(1, n + 1) / n * 100

        ax1.plot(
            exc, np.sort(speeds)[::-1], label=f"Depth: {depth:.1f} m", color=colors[i], linewidth=2
        )
        ax2.plot(
            exc, np.sort(power)[::-1], label=f"Depth: {depth:.1f} m", color=colors[i], linewidth=2
        )

    ax1.axhline(y=1.0, color="r", linestyle="--", alpha=0.7)
    ax1.axhline(y=2.0, color="r", linestyle="--", alpha=0.7)
    ax1.text(95, 1.0, "1.0 m/s", va="bottom", ha="right", color="r")
    ax1.text(95, 2.0, "2.0 m/s", va="bottom", ha="right", color="r")
    ax1.set_xlabel("Exceedance Probability [%]")
    ax1.set_ylabel("Current Speed [m/s]")
    ax1.set_title("Tidal Current Speed Exceedance")
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.legend(loc="best")

    ax2.set_xlabel("Exceedance Probability [%]")
    ax2.set_ylabel("Power Density [W/m^2]")
    ax2.set_title("Tidal Power Density Exceedance")
    ax2.set_yscale("log")
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.legend(loc="best")

    plt.tight_layout()
    return fig


@styled
def plot_multi_site_exceedance_overlay(
    site_records: list[tuple[str, pd.DataFrame, int, str]],
    settings: PlotSettings | None = None,
    cut_in_speed_ms: float = 0.5,
    show_cut_in_line: bool = False,
    show_cut_in_zones: bool = False,
    show_generating_pct: bool | None = None,
) -> tuple[Figure, dict[str, dict[str, float]]]:
    """Plot current-speed exceedance curves for all sites on one figure.

    Each site contributes one curve at its turbine hub-depth layer.
    Speed is on the x-axis and probability of exceedance on the y-axis,
    matching the convention of :func:`plot_velocity_exceedance`.

    Parameters
    ----------
    site_records : list of (name, df, layer, color)
        One tuple per site.  *layer* is the sigma-layer index chosen for the
        turbine hub depth (e.g. from :func:`select_turbine_layer`).
    settings : PlotSettings, optional
        See :class:`~us_marine_energy_resource.viz.settings.PlotSettings`.
    cut_in_speed_ms : float, optional
        Turbine cut-in speed (m/s).  Used for ``usable_pct`` stats and
        optional decorations.
    show_cut_in_line : bool, optional
        If ``True``, draws a dashed vertical line at *cut_in_speed_ms* with a
        label annotation.  Defaults to ``False``.
    show_cut_in_zones : bool, optional
        If ``True``, draws background shading for the below-cut-in (red) and
        generating (green) speed regions.  Defaults to ``False``.
    show_generating_pct : bool, optional
        If ``True``, appends ``"(generating N% of time)"`` to each site's
        legend label.  Defaults to ``True`` when *show_cut_in_line* is
        ``True``, ``False`` otherwise.

    Returns
    -------
    fig : Figure
        Single-panel exceedance figure.
    stats : dict
        Nested dict keyed by site name.  Each entry contains ``usable_pct``,
        ``mean``, ``max``, and one key per standard percentile
        (``"50%"``, ``"25%"``, ``"10%"``, ``"5%"``, ``"1%"``, ``"0.1%"``).
    """
    _key_percentiles: list[float] = [50, 25, 10, 5, 1, 0.1]
    _show_pct = show_cut_in_line if show_generating_pct is None else show_generating_pct

    fig, ax = plt.subplots(figsize=(13, 7))
    stats: dict[str, dict[str, float]] = {}

    for name, df, layer, color in site_records:
        speeds = df[f"vap_sea_water_speed_layer_{layer}"].dropna().to_numpy(dtype=float)
        speeds_sorted = np.sort(speeds)[::-1]
        n = len(speeds_sorted)
        exc_pct = np.arange(1, n + 1) / n * 100.0

        usable_pct = float(np.mean(speeds >= cut_in_speed_ms) * 100.0)
        site_label = name.split(",")[0]
        if _show_pct:
            site_label += f"  (generating {usable_pct:.0f}% of time)"

        ax.plot(speeds_sorted, exc_pct, color=color, linewidth=2.2, label=site_label, alpha=0.9)

        site_stats: dict[str, float] = {"usable_pct": usable_pct}
        for p in _key_percentiles:
            site_stats[f"{p}%"] = float(np.percentile(speeds, 100 - p))
        site_stats["mean"] = float(np.nanmean(speeds))
        site_stats["max"] = float(np.nanmax(speeds))
        stats[name] = site_stats

    if show_cut_in_zones:
        x_max = ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else 5.0
        ax.axvspan(
            0,
            cut_in_speed_ms,
            alpha=0.06,
            color="red",
            label=f"Below cut-in ({cut_in_speed_ms} m/s)",
        )
        ax.axvspan(
            cut_in_speed_ms,
            max(x_max, cut_in_speed_ms * 4),
            alpha=0.04,
            color="green",
            label=f"Generating (v \u2265 {cut_in_speed_ms} m/s)",
        )

    if show_cut_in_line:
        ax.axvline(cut_in_speed_ms, color="tomato", linestyle="--", linewidth=1.4, alpha=0.9)
        ax.text(
            cut_in_speed_ms,
            101,
            f"cut-in\n{cut_in_speed_ms} m/s",
            va="bottom",
            ha="center",
            fontsize=8,
            color="tomato",
        )

    ax.set_xlabel("Sea Water Speed [m/s]", fontsize=11)
    ax.set_ylabel("Probability of Exceedance [%]", fontsize=11)
    ax.set_title(
        "Multi-Site Current Speed Exceedance\n(turbine hub-depth layer per site)",
        fontsize=12,
    )
    ax.set_xlim(left=0)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
        fontsize=8.5,
        framealpha=0.9,
    )

    plt.tight_layout()
    return fig, stats
