"""Multi-panel dashboard figures for tidal resource characterization."""

from typing import Any

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from windrose import WindroseAxes  # type: ignore[import-untyped]

from us_marine_energy_resource.viz._style import styled
from us_marine_energy_resource.viz.settings import PlotSettings

from ._components import _N_LAYERS, _validate_columns

# Aggregated water-column columns produced by some dataset versions
_WATER_COLUMN_COLS = [
    "vap_water_column_mean_sea_water_speed",
    "vap_water_column_max_sea_water_speed",
    "vap_water_column_95th_percentile_sea_water_speed",
    "vap_water_column_mean_sea_water_power_density",
    "vap_water_column_max_sea_water_power_density",
    "vap_water_column_95th_percentile_sea_water_power_density",
    "vap_water_column_mean_sea_water_to_direction",
]


@styled
def create_tidal_resource_dashboard(
    df: pd.DataFrame,
    settings: PlotSettings | None = None,
    timestamp_index: int = 0,
) -> Figure:
    """Create an 8-panel tidal resource characterization dashboard.

    Panels:

    1. Velocity profile at a selected timestamp.
    2. Power density profile at a selected timestamp.
    3. Speed-vs-depth scatter (all time steps).
    4. Speed time series (+/- 100 steps around selected timestamp).
    5. Direction rose (polar histogram).
    6. Speed exceedance curve (middle layer).
    7. Power density exceedance curve on log scale (middle layer).
    8. Site resource summary text box.

    .. note::
        Panels 8 (summary text) requires the aggregated water-column columns
        (e.g. ``vap_water_column_mean_sea_water_speed``).  If they are absent
        the panel is rendered blank with an informational message.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a ``DatetimeIndex`` and standard speed, direction,
        power density, depth, and location columns.
    settings : PlotSettings, optional
        See :class:`~us_marine_energy_resource.viz.settings.PlotSettings`.
    timestamp_index : int, optional
        Row index of the reference timestamp. Default 0.

    Returns
    -------
    fig : Figure
        The created matplotlib figure.

    Raises
    ------
    KeyError
        If core per-layer columns are absent from *df*.
    """
    speed_cols = [f"vap_sea_water_speed_layer_{i}" for i in range(_N_LAYERS)]
    power_cols = [f"vap_sea_water_power_density_layer_{i}" for i in range(_N_LAYERS)]
    depth_cols = [f"vap_sigma_depth_layer_{i}" for i in range(_N_LAYERS)]
    dir_cols = [f"vap_sea_water_to_direction_layer_{i}" for i in range(_N_LAYERS)]
    _validate_columns(df, speed_cols + power_cols + depth_cols + dir_cols + ["vap_sea_floor_depth"])

    data = df.iloc[timestamp_index]
    depths = [float(data[f"vap_sigma_depth_layer_{i}"]) for i in range(_N_LAYERS)]
    speeds_ts = [float(data[f"vap_sea_water_speed_layer_{i}"]) for i in range(_N_LAYERS)]
    power_ts = [float(data[f"vap_sea_water_power_density_layer_{i}"]) for i in range(_N_LAYERS)]

    fig = plt.figure(figsize=(22, 16))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 1.2])

    # 1 — velocity profile
    ax1: Any = fig.add_subplot(gs[0, 0])
    ax1.plot(speeds_ts, depths, "o-", linewidth=2, markersize=8)
    ax1.set_xlabel("Current Speed [m/s]")
    ax1.set_ylabel("Depth [m]")
    ax1.set_title("Velocity Profile")
    ax1.grid(True)
    ax1.invert_yaxis()

    # 2 — power density profile
    ax2: Any = fig.add_subplot(gs[0, 1])
    ax2.plot(power_ts, depths, "o-", linewidth=2, markersize=8, color=sns.color_palette()[1])
    ax2.set_xlabel("Power Density [W/m^2]")
    ax2.set_ylabel("Depth [m]")
    ax2.set_title("Power Density Profile")
    ax2.grid(True)
    ax2.invert_yaxis()

    # 3 — speed-vs-depth scatter
    ax3: Any = fig.add_subplot(gs[0, 2])
    all_depths = np.array([df[c].to_numpy(dtype=float) for c in depth_cols]).T.flatten()
    all_speeds = np.array([df[c].to_numpy(dtype=float) for c in speed_cols]).T.flatten()
    ax3.scatter(all_speeds, all_depths, s=10, alpha=0.3)
    ax3.set_xlabel("Current Speed [m/s]")
    ax3.set_ylabel("Depth [m]")
    ax3.set_title("Speed vs Depth Distribution")
    ax3.grid(True)
    ax3.invert_yaxis()

    # 4 — time series slice
    ax4: Any = fig.add_subplot(gs[1, :2])
    window = 100
    df_slice = df.iloc[max(0, timestamp_index - window) : min(len(df), timestamp_index + window)]
    for layer in [0, 4, 9]:
        ax4.plot(
            df_slice.index,
            df_slice[f"vap_sea_water_speed_layer_{layer}"],
            label=f"Depth: {depths[layer]:.1f} m",
        )
    ax4.axvline(x=df.index[timestamp_index], color="r", linestyle="--")
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Current Speed [m/s]")
    ax4.set_title("Tidal Current Speed Time Series")
    ax4.legend()
    ax4.grid(True)

    # 5 — direction rose (polar)
    ax5: Any = fig.add_subplot(gs[1, 2], polar=True)
    directions = df["vap_sea_water_to_direction_layer_4"].to_numpy(dtype=float)
    dir_rad = np.radians(directions)
    rose_bins = np.linspace(0, 2 * np.pi, 17)
    n, _ = np.histogram(dir_rad, bins=rose_bins)
    ax5.bar(rose_bins[:-1], n, width=rose_bins[1] - rose_bins[0], bottom=0.0, alpha=0.7)
    ax5.set_theta_zero_location("N")
    ax5.set_theta_direction(-1)
    ax5.set_title("Current Direction (Layer 4)")

    # 6 — speed exceedance
    ax6: Any = fig.add_subplot(gs[2, 0])
    speeds_mid = df["vap_sea_water_speed_layer_4"].to_numpy(dtype=float)
    speeds_sorted = np.sort(speeds_mid)[::-1]
    exceedance = np.arange(1, len(speeds_mid) + 1) / len(speeds_mid) * 100
    ax6.plot(exceedance, speeds_sorted, linewidth=2)
    ax6.axhline(y=0.5, color="g", linestyle="--")
    ax6.axhline(y=2.0, color="r", linestyle="--")
    ax6.text(95, 0.5, "Cut-in: 0.5 m/s", ha="right", va="bottom", color="g")
    ax6.text(95, 2.0, "Rated: 2.0 m/s", ha="right", va="bottom", color="r")
    ax6.set_xlabel("Exceedance Probability [%]")
    ax6.set_ylabel("Current Speed [m/s]")
    ax6.set_title("Speed Exceedance Curve (Layer 4)")
    ax6.grid(True)

    # 7 — power exceedance
    ax7: Any = fig.add_subplot(gs[2, 1])
    power_mid = df["vap_sea_water_power_density_layer_4"].to_numpy(dtype=float)
    power_sorted = np.sort(power_mid)[::-1]
    ax7.plot(exceedance, power_sorted, linewidth=2, color=sns.color_palette()[1])
    ax7.set_xlabel("Exceedance Probability [%]")
    ax7.set_ylabel("Power Density [W/m^2]")
    ax7.set_title("Power Density Exceedance (Layer 4)")
    ax7.set_yscale("log")
    ax7.grid(True)

    # 8 — resource summary text
    ax8: Any = fig.add_subplot(gs[2, 2])
    ax8.axis("off")
    missing_wc = [c for c in _WATER_COLUMN_COLS[:6] if c not in df.columns]
    if missing_wc:
        ax8.text(
            0.05,
            0.95,
            "Water-column aggregate columns not available.\n"
            "Run compute_water_column_stats() to generate them.",
            va="top",
            transform=ax8.transAxes,
            fontsize=9,
            color="gray",
        )
    else:
        pct_1ms = float(np.mean(speeds_mid >= 1.0)) * 100
        pct_2ms = float(np.mean(speeds_mid >= 2.0)) * 100
        s_mean = float(df["vap_water_column_mean_sea_water_speed"].mean())
        s_max = float(df["vap_water_column_max_sea_water_speed"].max())
        s_p95 = float(df["vap_water_column_95th_percentile_sea_water_speed"].mean())
        p_mean = float(df["vap_water_column_mean_sea_water_power_density"].mean())
        p_max = float(df["vap_water_column_max_sea_water_power_density"].max())
        info = (
            "SITE RESOURCE SUMMARY\n"
            "=====================\n\n"
            f"Water Depth: {float(data['vap_sea_floor_depth']):.1f} m\n\n"
            f"Mean Speed: {s_mean:.2f} m/s\n"
            f"Max Speed: {s_max:.2f} m/s\n"
            f"95th Pct Speed: {s_p95:.2f} m/s\n\n"
            f"Mean Power: {p_mean:.2f} W/m^2\n"
            f"Max Power: {p_max:.2f} W/m^2\n\n"
            f"Time >= 1.0 m/s: {pct_1ms:.1f}%\n"
            f"Time >= 2.0 m/s: {pct_2ms:.1f}%\n\n"
            f"Lat: {float(data['lat_center']):.4f}  Lon: {float(data['lon_center']):.4f}"
        )
        ax8.text(0, 1.0, info, va="top", fontfamily="monospace")

    plt.suptitle(
        "Tidal Energy Resource Characterization Dashboard",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


@styled
def generate_tidal_site_assessment(
    df: pd.DataFrame,
    settings: PlotSettings | None = None,
    site_name: str = "Tidal Site",
) -> list[Figure]:
    """Generate a comprehensive tidal site assessment figure.

    Produces a single six-panel figure summarising key site metrics:

    1. Summary text box (location, speed, power).
    2. Water-column mean speed exceedance curve.
    3. Speed and power density time series (dual y-axis).
    4. Speed distribution histogram.
    5. Current rose (WindroseAxes).
    6. Vertical velocity profile (layer means).

    .. note::
        Requires water-column aggregate columns.  See
        :data:`_WATER_COLUMN_COLS`.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with standard layer columns and water-column aggregates.
    settings : PlotSettings, optional
        See :class:`~us_marine_energy_resource.viz.settings.PlotSettings`.
    site_name : str, optional
        Site label used in the figure title. Default ``"Tidal Site"``.

    Returns
    -------
    figs : list of Figure
        List containing the single assessment figure (returned as a list for
        forward-compatibility with multi-figure reports).

    Raises
    ------
    KeyError
        If required columns are absent from *df*.
    """
    required_wc = [
        "vap_water_column_mean_sea_water_speed",
        "vap_water_column_max_sea_water_speed",
        "vap_water_column_95th_percentile_sea_water_speed",
        "vap_water_column_mean_sea_water_power_density",
        "vap_water_column_max_sea_water_power_density",
        "vap_water_column_mean_sea_water_to_direction",
        "vap_sea_floor_depth",
        "lat_center",
        "lon_center",
    ]
    depth_cols = [f"vap_sigma_depth_layer_{i}" for i in range(_N_LAYERS)]
    speed_layer_cols = [f"vap_sea_water_speed_layer_{i}" for i in range(_N_LAYERS)]
    _validate_columns(df, required_wc + depth_cols + speed_layer_cols)

    mean_depth = float(df["vap_sea_floor_depth"].mean())
    max_speed = float(df["vap_water_column_max_sea_water_speed"].max())
    mean_speed = float(df["vap_water_column_mean_sea_water_speed"].mean())
    p95_speed = float(df["vap_water_column_95th_percentile_sea_water_speed"].mean())
    max_power = float(df["vap_water_column_max_sea_water_power_density"].max())
    mean_power = float(df["vap_water_column_mean_sea_water_power_density"].mean())

    speeds: np.ndarray = df["vap_water_column_mean_sea_water_speed"].to_numpy(dtype=float)
    speeds_sorted = np.sort(speeds)[::-1]
    exceedance = np.arange(1, len(speeds) + 1) / len(speeds) * 100

    exc_1ms = float(np.mean(speeds >= 1.0)) * 100
    exc_1_5ms = float(np.mean(speeds >= 1.5)) * 100
    exc_2ms = float(np.mean(speeds >= 2.0)) * 100

    fig = plt.figure(figsize=(20, 15))
    gs_fig = gridspec.GridSpec(3, 3, figure=fig)

    # 1 — summary text
    ax_sum: Any = fig.add_subplot(gs_fig[0, 0])
    ax_sum.axis("off")
    ax_sum.text(
        0,
        1.0,
        f"SITE ASSESSMENT: {site_name}\n"
        "====================================\n\n"
        f"LOCATION\n"
        f"  Lat: {float(df['lat_center'].iloc[0]):.4f}  "
        f"Lon: {float(df['lon_center'].iloc[0]):.4f}\n"
        f"  Mean Water Depth: {mean_depth:.1f} m\n\n"
        f"CURRENT VELOCITY\n"
        f"  Mean: {mean_speed:.2f} m/s\n"
        f"  Maximum: {max_speed:.2f} m/s\n"
        f"  95th Percentile: {p95_speed:.2f} m/s\n\n"
        f"POWER RESOURCE\n"
        f"  Mean: {mean_power:.2f} W/m^2\n"
        f"  Maximum: {max_power:.2f} W/m^2\n\n"
        f"EXCEEDANCE\n"
        f"  >= 1.0 m/s: {exc_1ms:.1f}%\n"
        f"  >= 1.5 m/s: {exc_1_5ms:.1f}%\n"
        f"  >= 2.0 m/s: {exc_2ms:.1f}%",
        va="top",
        fontfamily="monospace",
        fontsize=11,
        transform=ax_sum.transAxes,
    )

    # 2 — exceedance
    ax_exc: Any = fig.add_subplot(gs_fig[0, 1:])
    ax_exc.plot(exceedance, speeds_sorted, "-", color=sns.color_palette()[0], linewidth=2)
    ax_exc.axhline(y=1.0, color="g", linestyle="--", label="1.0 m/s")
    ax_exc.axhline(y=1.5, color="orange", linestyle="--", label="1.5 m/s")
    ax_exc.axhline(y=2.0, color="r", linestyle="--", label="2.0 m/s")
    ax_exc.set_xlabel("Exceedance Probability [%]")
    ax_exc.set_ylabel("Current Speed [m/s]")
    ax_exc.set_title("Water-Column Mean Speed Exceedance")
    ax_exc.grid(True)
    ax_exc.legend()

    # 3 — time series
    ax_ts: Any = fig.add_subplot(gs_fig[1, :])
    n_pts = min(1000, len(df))
    _speed_color = sns.color_palette()[0]
    _power_color = sns.color_palette()[1]
    ax_ts.plot(
        df.index[:n_pts],
        df["vap_water_column_mean_sea_water_speed"].iloc[:n_pts],
        "-",
        color=_speed_color,
        linewidth=1,
        label="Mean Speed",
    )
    ax_ts.set_xlabel("Time")
    ax_ts.set_ylabel("Current Speed [m/s]", color=_speed_color)
    ax_ts.tick_params(axis="y", labelcolor=_speed_color)
    ax_ts2 = ax_ts.twinx()
    ax_ts2.plot(
        df.index[:n_pts],
        df["vap_water_column_mean_sea_water_power_density"].iloc[:n_pts],
        "-",
        color=_power_color,
        linewidth=1,
        label="Power Density",
    )
    ax_ts2.set_ylabel("Power Density [W/m^2]", color=_power_color)
    ax_ts2.tick_params(axis="y", labelcolor=_power_color)
    lines1, lab1 = ax_ts.get_legend_handles_labels()
    lines2, lab2 = ax_ts2.get_legend_handles_labels()
    ax_ts.legend(lines1 + lines2, lab1 + lab2, loc="upper right")
    ax_ts.set_title("Speed and Power Density Time Series")
    ax_ts.grid(True)

    # 4 — speed histogram
    ax_hist: Any = fig.add_subplot(gs_fig[2, 0])
    ax_hist.hist(speeds, bins=30, color=sns.color_palette()[0], edgecolor="black")
    ax_hist.axvline(x=mean_speed, color="r", linestyle="-", label=f"Mean: {mean_speed:.2f} m/s")
    ax_hist.axvline(x=p95_speed, color="g", linestyle="--", label=f"95th: {p95_speed:.2f} m/s")
    ax_hist.set_xlabel("Current Speed [m/s]")
    ax_hist.set_ylabel("Frequency")
    ax_hist.set_title("Speed Distribution")
    ax_hist.grid(True)
    ax_hist.legend()

    # 5 — windrose
    ax_rose_ph: Any = fig.add_subplot(gs_fig[2, 1])
    ax_rose = WindroseAxes(fig, ax_rose_ph.get_position())
    ax_rose_ph.remove()
    fig.add_axes(ax_rose)
    dir_all: np.ndarray = df["vap_water_column_mean_sea_water_to_direction"].to_numpy(dtype=float)
    spd_all: np.ndarray = df["vap_water_column_mean_sea_water_speed"].to_numpy(dtype=float)
    ax_rose.bar(dir_all, spd_all, normed=True, opening=0.8, edgecolor="white")
    ax_rose.set_legend(title="Speed [m/s]")
    ax_rose.set_title("Current Rose")

    # 6 — vertical profile
    ax_prof: Any = fig.add_subplot(gs_fig[2, 2])
    mean_speeds = [float(df[f"vap_sea_water_speed_layer_{i}"].mean()) for i in range(_N_LAYERS)]
    mean_depths = [float(df[f"vap_sigma_depth_layer_{i}"].mean()) for i in range(_N_LAYERS)]
    ax_prof.plot(mean_speeds, mean_depths, "o-", linewidth=2, markersize=8)
    opt = int(np.argmax(mean_speeds))
    ax_prof.annotate(
        f"Optimal: Layer {opt} ({mean_depths[opt]:.1f} m)",
        xy=(mean_speeds[opt], mean_depths[opt]),
        xytext=(mean_speeds[opt] - 0.2, mean_depths[opt]),
        arrowprops={"facecolor": "black", "shrink": 0.05},
    )
    ax_prof.set_xlabel("Mean Current Speed [m/s]")
    ax_prof.set_ylabel("Depth [m]")
    ax_prof.set_title("Vertical Velocity Profile")
    ax_prof.grid(True)
    ax_prof.invert_yaxis()

    plt.suptitle(f"Tidal Energy Resource Assessment: {site_name}", fontsize=16, y=0.98)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    return [fig]
