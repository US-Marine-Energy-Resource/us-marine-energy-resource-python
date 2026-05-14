"""Time series, tidal asymmetry, and surface elevation statistics plots."""

from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from us_marine_energy_resource.viz._style import styled
from us_marine_energy_resource.viz.settings import PlotSettings, get_depth_perspective

from ._components import _validate_columns


@styled
def plot_tidal_time_series(
    df: pd.DataFrame,
    settings: PlotSettings | None = None,
    layers: list[int] | None = None,
) -> Figure:
    """Create a three-panel time series plot of tidal current data.

    Panels show current speed, current direction, and power density
    for a selected set of sigma layers over a configurable date range.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a ``DatetimeIndex`` and columns
        ``vap_sea_water_speed_layer_{i}``,
        ``vap_sea_water_to_direction_layer_{i}``,
        ``vap_sea_water_power_density_layer_{i}``, and
        ``vap_sigma_depth_layer_{i}`` for the requested layers.
    settings : PlotSettings, optional
        Shared plot settings (e.g. ``start_date``, ``end_date``). When
        provided, the DataFrame is automatically trimmed to the requested
        time window before plotting.
    layers : list of int, optional
        Sigma layer indices to plot. If ``None``, defaults to
        ``[0, 4, 9]`` (surface, mid-column, bottom).

    Returns
    -------
    fig : Figure
        The created matplotlib figure.

    Raises
    ------
    KeyError
        If any required speed, direction, power-density, or depth column is
        absent from *df*.
    """
    if layers is None:
        layers = [0, 4, 9]

    perspective = get_depth_perspective(settings)

    required = []
    for layer in layers:
        required += [
            f"vap_sea_water_speed_layer_{layer}",
            f"vap_sea_water_to_direction_layer_{layer}",
            f"vap_sea_water_power_density_layer_{layer}",
            perspective.depth_col(layer),
        ]
    _validate_columns(df, required)

    colors = plt.cm.viridis(np.linspace(0, 1, len(layers)))  # type: ignore[attr-defined]

    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Panel 1 — current speed
    for i, layer in enumerate(layers):
        depth = float(df[perspective.depth_col(layer)].iloc[0])
        axs[0].plot(
            df.index,
            df[f"vap_sea_water_speed_layer_{layer}"],
            color=colors[i],
            label=f"Layer {layer} (~{depth:.1f} m)",
        )
    axs[0].set_ylabel("Current Speed [m/s]")
    axs[0].set_title("Tidal Current Speed")
    axs[0].grid(True, linestyle="--", alpha=0.7)
    axs[0].legend(loc="upper right")

    # Panel 2 — current direction
    for i, layer in enumerate(layers):
        axs[1].plot(
            df.index,
            df[f"vap_sea_water_to_direction_layer_{layer}"],
            color=colors[i],
            label=f"Layer {layer}",
        )
    axs[1].set_ylabel("Direction [°]")
    axs[1].set_title("Tidal Current Direction")
    axs[1].set_yticks(np.arange(0, 361, 45))
    axs[1].set_yticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"])
    axs[1].set_ylim(0, 360)
    axs[1].grid(True, linestyle="--", alpha=0.7)

    # Panel 3 — power density
    for i, layer in enumerate(layers):
        axs[2].plot(
            df.index,
            df[f"vap_sea_water_power_density_layer_{layer}"],
            color=colors[i],
            label=f"Layer {layer}",
        )
    axs[2].set_ylabel("Power Density [W/m²]")
    axs[2].set_title("Tidal Current Power Density")
    axs[2].set_xlabel("Date / Time")
    axs[2].grid(True, linestyle="--", alpha=0.7)

    # X-axis date formatting on the shared bottom axis
    fig.autofmt_xdate()
    axs[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    axs[2].xaxis.set_major_locator(mdates.DayLocator())
    axs[2].xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
    axs[2].grid(True, which="minor", linestyle=":", alpha=0.4)

    plt.tight_layout()
    return fig


@styled
def plot_tidal_asymmetry(
    df: pd.DataFrame,
    settings: PlotSettings | None = None,
    layer: int = 4,
) -> Figure:
    """Analyze and visualize flood vs. ebb tidal current asymmetry.

    Four panels: flood/ebb speed histograms, CDFs, polar direction scatter,
    and an asymmetry metrics text box.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ``vap_sea_water_speed_layer_{layer}``,
        ``vap_sea_water_to_direction_layer_{layer}``,
        ``u_layer_{layer}``, ``v_layer_{layer}``, and
        ``vap_sigma_depth_layer_{layer}`` columns.
    settings : PlotSettings, optional
        Shared plot settings (e.g. ``start_date``, ``end_date``). When
        provided, the DataFrame is automatically trimmed to the requested
        time window before plotting.
    layer : int, optional
        Sigma layer to analyze. Default 4.

    Returns
    -------
    fig : Figure
        The created matplotlib figure.

    Raises
    ------
    KeyError
        If required columns are absent from *df*.
    """
    from sklearn.decomposition import PCA as _PCA

    from ._components import _validate_columns

    perspective = get_depth_perspective(settings)

    _validate_columns(
        df,
        [
            f"vap_sea_water_speed_layer_{layer}",
            f"vap_sea_water_to_direction_layer_{layer}",
            f"u_layer_{layer}",
            f"v_layer_{layer}",
            perspective.depth_col(layer),
        ],
    )

    speeds: np.ndarray = df[f"vap_sea_water_speed_layer_{layer}"].to_numpy(dtype=float)
    directions: np.ndarray = df[f"vap_sea_water_to_direction_layer_{layer}"].to_numpy(dtype=float)
    depth_value = float(df[perspective.depth_col(layer)].mean())

    u_dir = -speeds * np.sin(np.radians(directions))
    v_dir = -speeds * np.cos(np.radians(directions))

    pca = _PCA(n_components=2)
    pca.fit(np.column_stack([u_dir, v_dir]))
    principal_axis = float(
        np.degrees(np.arctan2(pca.components_[0, 1], pca.components_[0, 0])) % 360
    )

    principal_comp = u_dir * np.cos(np.radians(principal_axis)) + v_dir * np.sin(
        np.radians(principal_axis)
    )
    flood_speeds = principal_comp[principal_comp > 0]
    ebb_speeds = -principal_comp[principal_comp < 0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    _pal = sns.color_palette()
    flood_color = _pal[0]
    ebb_color = _pal[1]

    # Panel 1 — histograms
    ax1 = axes[0, 0]
    bins = np.linspace(0, max(float(np.max(flood_speeds)), float(np.max(ebb_speeds))) * 1.1, 30)
    ax1.hist(flood_speeds, bins=list(bins), alpha=0.7, label="Flood", color=flood_color)
    ax1.hist(ebb_speeds, bins=list(bins), alpha=0.7, label="Ebb", color=ebb_color)
    ax1.set_xlabel("Current Speed [m/s]")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Flood vs Ebb Speed Distribution")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.7)

    # Panel 2 — CDFs
    ax2 = axes[0, 1]
    ax2.plot(
        np.sort(flood_speeds),
        np.arange(1, len(flood_speeds) + 1) / len(flood_speeds),
        "-",
        linewidth=2,
        label="Flood",
        color=flood_color,
    )
    ax2.plot(
        np.sort(ebb_speeds),
        np.arange(1, len(ebb_speeds) + 1) / len(ebb_speeds),
        "-",
        linewidth=2,
        label="Ebb",
        color=ebb_color,
    )
    ax2.set_xlabel("Current Speed [m/s]")
    ax2.set_ylabel("Cumulative Probability")
    ax2.set_title("Flood vs Ebb Speed CDF")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.7)

    # Panel 3 — polar scatter
    ax3 = axes[1, 0]
    ax3.remove()
    from matplotlib.projections.polar import PolarAxes as _PolarAxes

    ax3_polar: _PolarAxes = fig.add_subplot(223, projection="polar")  # type: ignore[assignment]
    dir_rad = np.radians(directions)
    ax3_polar.scatter(dir_rad, speeds, s=5, alpha=0.3, c=speeds, cmap="viridis")
    flood_dir = principal_axis
    ebb_dir = (principal_axis + 180) % 360
    ax3_polar.scatter(
        [np.radians(flood_dir)],
        [float(np.median(flood_speeds))],
        s=100,
        color=flood_color,
        marker="^",
        label="Flood Dir",
    )
    ax3_polar.scatter(
        [np.radians(ebb_dir)],
        [float(np.median(ebb_speeds))],
        s=100,
        color=ebb_color,
        marker="v",
        label="Ebb Dir",
    )
    ax3_polar.set_theta_zero_location("N")
    ax3_polar.set_theta_direction(-1)
    ax3_polar.set_title("Current Direction Distribution")
    ax3_polar.legend(loc="upper right", bbox_to_anchor=(1.2, 1.0))

    # Panel 4 — metrics
    ax4 = axes[1, 1]
    ax4.axis("off")
    fm, em = float(np.mean(flood_speeds)), float(np.mean(ebb_speeds))
    fx, ex = float(np.max(flood_speeds)), float(np.max(ebb_speeds))
    fp, ep = (
        float(np.percentile(flood_speeds, 95)),
        float(np.percentile(ebb_speeds, 95)),
    )
    ratio_mean = fm / em if em > 0 else float("inf")
    ratio_max = fx / ex if ex > 0 else float("inf")
    fp / ep if ep > 0 else float("inf")
    fp_pow = float(np.mean(flood_speeds**3))
    ep_pow = float(np.mean(ebb_speeds**3))
    pow_ratio = fp_pow / ep_pow if ep_pow > 0 else float("inf")
    flood_pct = len(flood_speeds) / len(speeds) * 100
    ebb_pct = len(ebb_speeds) / len(speeds) * 100

    ax4.text(
        0,
        1.0,
        f"TIDAL ASYMMETRY METRICS\n"
        f"======================\n\n"
        f"Principal Flow Axis: {principal_axis:.1f} / {ebb_dir:.1f} deg\n\n"
        f"SPEEDS:\n"
        f"  Flood Mean: {fm:.2f} m/s    Ebb Mean: {em:.2f} m/s\n"
        f"  Ratio (Flood/Ebb): {ratio_mean:.2f}\n"
        f"  Flood Max: {fx:.2f} m/s    Ebb Max: {ex:.2f} m/s\n"
        f"  Max Ratio: {ratio_max:.2f}\n\n"
        f"POWER:\n"
        f"  Flood (m/s)^3: {fp_pow:.2f}    Ebb (m/s)^3: {ep_pow:.2f}\n"
        f"  Power Ratio: {pow_ratio:.2f}\n\n"
        f"DURATION:\n"
        f"  Flood: {flood_pct:.1f}%    Ebb: {ebb_pct:.1f}%",
        va="top",
        fontfamily="monospace",
    )

    plt.tight_layout()
    plt.suptitle(f"Tidal Asymmetry Analysis at {depth_value:.1f} m Depth", fontsize=16, y=1.02)
    return fig


@styled
def plot_tidal_statistics(
    surface_positions: pd.Series | np.ndarray,
    times: pd.DatetimeIndex | None = None,
    x_positions: pd.DatetimeIndex | np.ndarray | None = None,
    ax: Any | None = None,
    model_name: str = "FVCOM",
    data_year: str | None = None,
    location_label: str = "",
    settings: PlotSettings | None = None,
) -> tuple[Figure, Any, dict[str, Any]]:
    """Plot tidal reference levels and statistics for a surface elevation series.

    Draws the water surface elevation with horizontal reference lines for Max /
    Min / Mean High Tide, Mean Water Level, and Max / Min / Mean Low Tide.
    Markers are added at each detected high and low tide event.  A statistics
    text box is placed to the right of the axes.

    Parameters
    ----------
    surface_positions : array-like
        Water surface elevation values (m).
    times : pd.DatetimeIndex, optional
        Timestamps corresponding to *surface_positions*.
    x_positions : array-like, optional
        X-axis values for plotting.  Defaults to *times* when available,
        otherwise integer indices.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.  A new figure is created when ``None``.
    model_name : str, optional
        Model name shown in the title. Default ``"FVCOM"``.
    data_year : str, optional
        Year or range string for the title (e.g. ``"2020-2021"``).
    location_label : str, optional
        Location description for the title.
    settings : PlotSettings, optional
        Shared plot settings. Reserved for future use; this function takes a
        pre-extracted ``surface_positions`` array rather than a DataFrame, so
        time windowing is not applied automatically.

    Returns
    -------
    fig : Figure
        Parent figure.
    ax : matplotlib.axes.Axes
        Axes containing the plot.
    tidal_data : dict
        Tidal reference levels from
        :func:`us_marine_energy_resource.analysis.resource.calculate_tidal_levels`.
    """
    from matplotlib.lines import Line2D as _Line2D

    from us_marine_energy_resource.analysis.resource import (
        calculate_tidal_levels,
        calculate_tidal_periods,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure

    sp = np.asarray(surface_positions, dtype=float)
    sp = sp - float(np.mean(sp))

    if x_positions is None:
        x_positions = times if times is not None else np.arange(len(sp))

    len(x_positions)

    tidal_data: dict[str, Any] = calculate_tidal_levels(sp, times)
    period_stats: dict[str, Any] = calculate_tidal_periods(sp, times)

    levels = {
        "Max High Tide*": tidal_data["Max High Tide"],
        "Min High Tide*": tidal_data["Min High Tide"],
        "Mean High Tide*": tidal_data["Mean High Tide"],
        "Mean Water Level*": tidal_data["Mean Water Level"],
        "Max Low Tide*": tidal_data["Max Low Tide"],
        "Mean Low Tide*": tidal_data["Mean Low Tide"],
        "Min Low Tide*": tidal_data["Min Low Tide"],
    }
    _pal = sns.color_palette()
    surface_color = _pal[0]
    colors_map = {
        "Max High Tide": _pal[1],
        "Min High Tide": _pal[2],
        "Mean High Tide": _pal[3],
        "Mean Water Level": _pal[4],
        "Max Low Tide": _pal[5],
        "Mean Low Tide": _pal[6],
        "Min Low Tide": _pal[7],
    }
    styles_map = {
        "Max High Tide": (0, (5, 5)),
        "Min High Tide": (0, (1, 1)),
        "Mean High Tide": "-",
        "Mean Water Level": "-",
        "Max Low Tide": (0, (1, 1)),
        "Mean Low Tide": "-",
        "Min Low Tide": (0, (5, 5)),
    }

    ax.plot(x_positions, sp, color=surface_color, linewidth=0.5, zorder=10)
    legend_elems: list[Any] = [
        _Line2D([0], [0], color=surface_color, lw=1.5, label="Water Surface Elevation")
    ]

    for label, value in levels.items():
        base = label.replace("*", "")
        ax.axhline(
            y=value,
            color=colors_map.get(base, "gray"),
            linestyle=styles_map.get(base, "--"),
            linewidth=1.2,
            alpha=0.8,
            zorder=5,
        )
        legend_elems.append(
            _Line2D(
                [0],
                [0],
                color=colors_map.get(base, "gray"),
                linestyle=styles_map.get(base, "--"),
                lw=1.2,
                label=f"{label}: {value:.2f} m",
            )
        )

    high_idx = tidal_data.get("high_tide_indices", [])
    low_idx = tidal_data.get("low_tide_indices", [])

    for idx in high_idx:
        if idx < len(sp):
            ex = x_positions[idx] if not isinstance(x_positions, np.ndarray) else idx
            ax.plot(ex, sp[idx], "o", color=colors_map["Mean High Tide"], markersize=3, alpha=0.7)

    for idx in low_idx:
        if idx < len(sp):
            ex = x_positions[idx] if not isinstance(x_positions, np.ndarray) else idx
            ax.plot(ex, sp[idx], "o", color=colors_map["Mean Low Tide"], markersize=3, alpha=0.7)

    legend_elems += [
        _Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=colors_map["Mean High Tide"],
            markersize=6,
            label="High Tide",
        ),
        _Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=colors_map["Mean Low Tide"],
            markersize=6,
            label="Low Tide",
        ),
    ]

    # Stats text box to the right
    days = 1.0
    if isinstance(times, pd.DatetimeIndex) and len(times) > 1:
        days = float((times[-1] - times[0]).total_seconds()) / 86400.0
    tidal_cycles = len(high_idx)
    cpd = tidal_cycles / days if days > 0 else 0.0

    sim_period = data_year or ""
    if not sim_period and times is not None and isinstance(times, pd.DatetimeIndex):
        sim_period = f"{times[0].strftime('%b %d, %Y')} to {times[-1].strftime('%b %d, %Y')}"

    tidal_range = float(levels["Max High Tide*"]) - float(levels["Min Low Tide*"])
    stats_text = (
        f"Surface Elevation Statistics ({sim_period}):\n"
        f"Total Range: {tidal_range:.2f} m\n"
        f"Tidal Range: avg {period_stats['average_range']:.2f} m, "
        f"min {period_stats['min_range']:.2f} m, max {period_stats['max_range']:.2f} m\n"
        f"Cycles: {tidal_cycles} ({cpd:.1f}/day)\n"
        f"Period: avg {period_stats['average_period_str']}, "
        f"range {period_stats['min_period_str']} to {period_stats['max_period_str']}\n"
        f"Tide Pattern: {period_stats.get('tide_type', 'Unknown')}\n"
        f"* Model-derived, not measured"
    )
    ax.text(
        1.01,
        -0.01,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.8, "ec": "gray"},
    )

    max_h = float(levels["Max High Tide*"])
    min_l = float(levels["Min Low Tide*"])
    y_pad = 0.05 * (max_h - min_l)
    ax.set_ylim(min_l - y_pad, max_h + y_pad)
    plt.ylabel("Surface Elevation [m]")
    plt.xlabel("Time [UTC]")
    plt.grid(True, alpha=0.3, which="both")
    plt.title(sim_period)
    ax.legend(
        handles=legend_elems,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        title="Legend",
        title_fontsize=10,
    )
    plt.tight_layout()
    return fig, ax, tidal_data
