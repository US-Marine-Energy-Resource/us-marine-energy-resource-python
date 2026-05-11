"""Velocity profile, shear, and power density plots."""

from typing import Any

import cmocean  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from us_marine_energy_resource.viz._style import styled
from us_marine_energy_resource.viz.settings import PlotSettings

from ._components import _N_LAYERS, _validate_columns


@styled
def plot_velocity_profile_with_histograms(
    df: pd.DataFrame,
    settings: PlotSettings | None = None,
    filter_dry_points: bool = True,
    dry_threshold: float = 0.0,
    filter_smushed_layers: bool = True,
    min_layer_thickness: float = 0.1,
    min_total_depth: float = 1.0,
    show_filtered_stats: bool = True,
    invert_depth_axis: bool = True,
    layout: str = "stacked",
    verbose: bool = False,
) -> tuple[Figure, dict[str, Any]]:
    """Plot velocity profiles with per-layer histograms.

    Produces a five-panel figure:

    1. **Velocity profile** — horizontal box plots per sigma layer with a
       mean-velocity line overlay.
    2. **Depth vs. speed scatter** — coloured by direction with quadratic
       mean/max fit lines.
    3-5. **Per-layer histograms** of speed, direction, and depth (one
       sub-plot per layer).

    Data quality filtering is applied before plotting:

    * *Dry points* — time steps where the surface layer depth is at or below
      *dry_threshold* are removed.
    * *Shallow points* — time steps where the maximum layer depth is below
      *min_total_depth* are removed.
    * *Smushed layers* — individual sigma layers thinner than
      *min_layer_thickness* are masked to NaN.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ``vap_sea_water_speed_layer_{i}``,
        ``vap_sea_water_to_direction_layer_{i}``, and
        ``vap_sigma_depth_layer_{i}`` for layers 0-9.
    settings : PlotSettings, optional
        See :class:`~us_marine_energy_resource.viz.settings.PlotSettings`.
    filter_dry_points : bool, optional
        Remove time steps where the surface layer is ``<= dry_threshold``.
        Default ``True``.
    dry_threshold : float, optional
        Depth (m) at or below which the surface layer is considered dry.
        Default 0.0.
    filter_smushed_layers : bool, optional
        Mask individual layers thinner than *min_layer_thickness*. Default
        ``True``.
    min_layer_thickness : float, optional
        Minimum layer thickness (m) to retain. Default 0.1.
    min_total_depth : float, optional
        Minimum total water column depth (m) to retain a time step.
        Default 1.0.
    show_filtered_stats : bool, optional
        Annotate the figure with a data-quality summary. Default ``True``.
    invert_depth_axis : bool, optional
        Use oceanographic convention (surface at top). Default ``True``.
    layout : str, optional
        Panel arrangement. ``"stacked"`` (default) places the velocity profile
        and scatter plot in the top row and the three per-layer histogram groups
        in the bottom row. ``"flat"`` uses the original single-row arrangement
        with all five panels side by side.
    verbose : bool, optional
        Print filtering progress to stdout. Default ``False``.

    Returns
    -------
    fig : Figure
        The created matplotlib figure.
    stats : dict
        Data-quality metrics with keys ``original_points``,
        ``filtered_points``, ``dry_points_removed``,
        ``shallow_points_removed``, ``smushed_layers_removed``,
        ``total_layers_original``, ``total_layers_after_filtering``.

    Raises
    ------
    KeyError
        If any required speed, direction, or depth column is absent from *df*.
    ValueError
        If no data points remain after filtering.
    """
    velocity_cols = [f"vap_sea_water_speed_layer_{i}" for i in range(_N_LAYERS)]
    direction_cols = [f"vap_sea_water_to_direction_layer_{i}" for i in range(_N_LAYERS)]
    depth_cols = [f"vap_sigma_depth_layer_{i}" for i in range(_N_LAYERS)]
    _validate_columns(df, velocity_cols + direction_cols + depth_cols)

    stats: dict[str, Any] = {
        "original_points": len(df),
        "filtered_points": 0,
        "dry_points_removed": 0,
        "shallow_points_removed": 0,
        "smushed_layers_removed": 0,
        "total_layers_original": 0,
        "total_layers_after_filtering": 0,
    }

    if verbose:
        print(f"Starting with {len(df)} data points")

    df_filtered = df.copy()

    # 1. Remove dry points
    if filter_dry_points:
        dry_mask = df_filtered[depth_cols[0]] <= dry_threshold
        stats["dry_points_removed"] = int(dry_mask.sum())
        df_filtered = df_filtered[~dry_mask]
        if verbose:
            print(
                f"Removed {stats['dry_points_removed']} dry points "
                f"(surface depth <= {dry_threshold} m)"
            )

    # 2. Remove shallow points
    if min_total_depth > 0:
        max_depths = df_filtered[depth_cols].max(axis=1)
        shallow_mask = max_depths < min_total_depth
        stats["shallow_points_removed"] = int(shallow_mask.sum())
        df_filtered = df_filtered[~shallow_mask]
        if verbose:
            print(
                f"Removed {stats['shallow_points_removed']} shallow points "
                f"(max depth < {min_total_depth} m)"
            )

    stats["filtered_points"] = len(df_filtered)
    if len(df_filtered) == 0:
        raise ValueError("No data points remain after filtering.")

    n_loc = len(df_filtered)
    # shape: (n_loc, _N_LAYERS)
    all_vel = df_filtered[velocity_cols].to_numpy(dtype=float)
    all_dir = df_filtered[direction_cols].to_numpy(dtype=float)
    all_dep = df_filtered[depth_cols].to_numpy(dtype=float)

    stats["total_layers_original"] = n_loc * _N_LAYERS

    # 3. Mask smushed layers
    layer_valid = np.ones((n_loc, _N_LAYERS), dtype=bool)
    if filter_smushed_layers:
        for loc in range(n_loc):
            loc_depths = all_dep[loc, :]
            depth_order = np.argsort(loc_depths)
            thicknesses = np.diff(loc_depths[depth_order])
            for k, layer_idx in enumerate(depth_order[:-1]):
                if thicknesses[k] < min_layer_thickness:
                    layer_valid[loc, layer_idx] = False
                    stats["smushed_layers_removed"] += 1

    stats["total_layers_after_filtering"] = int(layer_valid.sum())
    if verbose:
        print(
            f"Removed {stats['smushed_layers_removed']} smushed layers "
            f"(thickness < {min_layer_thickness} m)"
        )

    all_vel = np.where(layer_valid, all_vel, np.nan)
    all_dir = np.where(layer_valid, all_dir, np.nan)
    all_dep = np.where(layer_valid, all_dep, np.nan)

    with np.errstate(invalid="ignore"):
        mean_dep: np.ndarray = np.nanmean(all_dep, axis=0)
        max_dep: np.ndarray = np.nanmax(all_dep, axis=0)
        min_dep: np.ndarray = np.nanmin(all_dep, axis=0)
        mean_vel: np.ndarray = np.nanmean(all_vel, axis=0)

    # --- Colour palette ---
    palette = sns.color_palette()
    bp_color = palette[0]
    bp_edge = palette[0]
    median_color = palette[1]
    whisker_color = "#333333"
    hist_vel_color = palette[0]
    hist_dir_color = palette[2]
    hist_dep_color = palette[3]

    # --- Build figure ---
    bot_row: Any = None  # only assigned in "stacked" layout
    if layout == "stacked":
        fig = plt.figure(figsize=(20, 16))
        gs_outer = fig.add_gridspec(2, 1, height_ratios=[3, 2], hspace=0.35)
        top_row = gs_outer[0].subgridspec(1, 2, width_ratios=[1, 1], wspace=0.35)
        bot_row = gs_outer[1].subgridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.35)
        ax_profile = fig.add_subplot(top_row[0])
        ax_scatter = fig.add_subplot(top_row[1])
        ax_hist_ph = fig.add_subplot(bot_row[0])
        ax_dir_ph = fig.add_subplot(bot_row[1])
        ax_dep_ph = fig.add_subplot(bot_row[2])
    else:
        fig = plt.figure(figsize=(24, 12))
        gs_outer = fig.add_gridspec(1, 1)
        top_row = gs_outer[0].subgridspec(1, 5, width_ratios=[1, 1, 0.75, 0.75, 0.75])
        ax_profile = fig.add_subplot(top_row[0])
        ax_scatter = fig.add_subplot(top_row[1])
        ax_hist_ph = fig.add_subplot(top_row[2])
        ax_dir_ph = fig.add_subplot(top_row[3])
        ax_dep_ph = fig.add_subplot(top_row[4])

    for _ax in [ax_profile, ax_scatter]:
        _ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))  # type: ignore[attr-defined]
        _ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))  # type: ignore[attr-defined]

    # Panel 1 — box plots + mean profile
    for i in range(_N_LAYERS):
        layer_vel = all_vel[:, i]
        valid_vel = layer_vel[~np.isnan(layer_vel)]
        if len(valid_vel) == 0 or np.isnan(mean_dep[i]):
            continue
        ax_profile.boxplot(
            valid_vel,
            positions=[mean_dep[i]],
            vert=False,
            widths=0.2,
            patch_artist=True,
            showfliers=False,
            boxprops={"facecolor": bp_color, "alpha": 0.3, "edgecolor": bp_edge},
            medianprops={"color": median_color, "linewidth": 1.5},
            whiskerprops={"color": whisker_color, "linewidth": 0.75},
            capprops={"color": whisker_color},
        )

    valid_layers = ~np.isnan(mean_dep) & ~np.isnan(mean_vel)
    if np.any(valid_layers):
        ax_profile.plot(
            mean_vel[valid_layers],
            mean_dep[valid_layers],
            "-",
            linewidth=2,
            label="Mean Velocity",
            marker=".",
            markersize=10,
            color=bp_color,
            zorder=10,
        )

    ax_profile.plot([], [], "-", color=whisker_color, label="Whiskers (min/max)")
    ax_profile.plot(
        [],
        [],
        "s",
        color=bp_edge,
        markerfacecolor=bp_color,
        alpha=0.3,
        markersize=8,
        label="IQR (25th/75th)",
    )
    ax_profile.plot([], [], "-", color=median_color, linewidth=1.5, label="Median")
    ax_profile.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    ax_profile.set_xlabel("Sea Water Speed [m/s]")
    ax_profile.set_ylabel("Depth [m]")

    title = "Velocity Profile"
    if show_filtered_stats:
        title += f" (n={stats['filtered_points']}/{stats['original_points']} points)"
    ax_profile.set_title(title)

    valid_dep_layers = ~np.isnan(mean_dep)
    if np.any(valid_dep_layers):
        y_ticks = mean_dep[valid_dep_layers]
        y_labels = [
            f"{min_dep[i]:.1f}-{max_dep[i]:.1f} m" for i in range(_N_LAYERS) if valid_dep_layers[i]
        ]
        ax_profile.set_yticks(y_ticks)
        ax_profile.set_yticklabels(y_labels)

    if invert_depth_axis:
        ax_profile.invert_yaxis()
    ax_profile.grid(True)

    # Panel 2 — scatter: depth vs speed coloured by direction
    flat_dep = all_dep.flatten()
    flat_vel = all_vel.flatten()
    flat_dir = all_dir.flatten()
    valid_mask = ~(np.isnan(flat_dep) | np.isnan(flat_vel) | np.isnan(flat_dir))
    flat_dep = flat_dep[valid_mask]
    flat_vel = flat_vel[valid_mask]
    flat_dir = flat_dir[valid_mask]

    if len(flat_dep) > 0:
        scatter = ax_scatter.scatter(
            flat_vel,
            flat_dep,
            alpha=1.0,
            c=flat_dir,
            cmap=cmocean.cm.phase,  # type: ignore[attr-defined]
            s=3,
            edgecolor="none",
        )
        cbar = plt.colorbar(scatter, ax=ax_scatter)
        cbar.set_label("Direction [°]")

        layer_mean_sp = np.nanmean(all_vel, axis=0)
        layer_max_sp = np.nanmax(all_vel, axis=0)
        fit_valid = ~(np.isnan(mean_dep) | np.isnan(layer_mean_sp) | np.isnan(layer_max_sp))

        if int(np.sum(fit_valid)) >= 3:
            mean_coefs = np.polyfit(mean_dep[fit_valid], layer_mean_sp[fit_valid], 2)
            max_coefs = np.polyfit(mean_dep[fit_valid], layer_max_sp[fit_valid], 2)
            line_x = np.linspace(np.nanmin(flat_dep), np.nanmax(flat_dep), 100)

            ax_scatter.plot(
                np.polyval(mean_coefs, line_x),
                line_x,
                color=sns.color_palette()[0],
                linewidth=2,
                label="Mean Speed (quad fit)",
            )
            ax_scatter.plot(
                np.polyval(max_coefs, line_x),
                line_x,
                color=sns.color_palette()[3],
                linewidth=2,
                linestyle="--",
                label="Max Speed (quad fit)",
            )
            a_m, b_m, c_m = mean_coefs
            a_x, b_x, c_x = max_coefs
            fig.text(
                0.5,
                0.02,
                (
                    f"Mean Speed = {a_m:.4f}·Depth² + {b_m:.4f}·Depth + {c_m:.4f}\n"
                    f"Max Speed  = {a_x:.4f}·Depth² + {b_x:.4f}·Depth + {c_x:.4f}"
                ),
                ha="center",
                fontsize=10,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
            )

        ax_scatter.scatter(
            layer_mean_sp[fit_valid],
            mean_dep[fit_valid],
            color=sns.color_palette()[0],
            s=50,
            zorder=5,
            label="Layer Mean Speeds",
        )
        ax_scatter.scatter(
            layer_max_sp[fit_valid],
            mean_dep[fit_valid],
            color=sns.color_palette()[3],
            s=50,
            zorder=5,
            marker="s",
            label="Layer Max Speeds",
        )

    ax_scatter.set_xlabel("Sea Water Speed [m/s]")
    ax_scatter.set_ylabel("Depth [m]")
    ax_scatter.set_title("Depth vs. Speed")
    ax_scatter.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    if invert_depth_axis:
        ax_scatter.invert_yaxis()
    ax_scatter.grid(True, linestyle="--", alpha=0.7)

    # Panels 3-5 — per-layer histograms
    ax_hist_ph.remove()
    ax_dir_ph.remove()
    ax_dep_ph.remove()
    if layout == "stacked":
        hist_grid = bot_row[0].subgridspec(_N_LAYERS, 1, hspace=0.3)
        dir_grid = bot_row[1].subgridspec(_N_LAYERS, 1, hspace=0.3)
        dep_grid = bot_row[2].subgridspec(_N_LAYERS, 1, hspace=0.3)
    else:
        hist_grid = top_row[2].subgridspec(_N_LAYERS, 1, hspace=0.3)
        dir_grid = top_row[3].subgridspec(_N_LAYERS, 1, hspace=0.3)
        dep_grid = top_row[4].subgridspec(_N_LAYERS, 1, hspace=0.3)

    n_bins = 50
    all_valid_vel = all_vel[~np.isnan(all_vel)]
    all_valid_dep_fl = all_dep[~np.isnan(all_dep)]

    vel_edges: np.ndarray = (
        np.linspace(all_valid_vel.min() * 0.9, all_valid_vel.max() * 1.1, n_bins + 1)
        if len(all_valid_vel) > 0
        else np.linspace(0, 1, n_bins + 1)
    )
    dir_edges: np.ndarray = np.linspace(0, 360, n_bins + 1)
    dep_edges: np.ndarray = (
        np.linspace(all_valid_dep_fl.min() * 0.95, all_valid_dep_fl.max() * 1.05, n_bins + 1)
        if len(all_valid_dep_fl) > 0
        else np.linspace(0, 10, n_bins + 1)
    )

    hist_axes: list[Any] = []
    dir_axes: list[Any] = []
    dep_axes: list[Any] = []

    for i in range(_N_LAYERS):
        ax_v = fig.add_subplot(hist_grid[i])
        ax_d = fig.add_subplot(dir_grid[i])
        ax_p = fig.add_subplot(dep_grid[i])
        hist_axes.append(ax_v)
        dir_axes.append(ax_d)
        dep_axes.append(ax_p)

        lv = all_vel[:, i][~np.isnan(all_vel[:, i])]
        ld = all_dir[:, i][~np.isnan(all_dir[:, i])]
        lp = all_dep[:, i][~np.isnan(all_dep[:, i])]

        if len(lv) > 0:
            ax_v.hist(
                lv,
                bins=list(vel_edges),
                color=hist_vel_color,
                edgecolor=hist_vel_color,
                alpha=0.8,
            )
        if len(ld) > 0:
            ax_d.hist(
                ld,
                bins=list(dir_edges),
                color=hist_dir_color,
                edgecolor=hist_dir_color,
                alpha=0.8,
            )
        if len(lp) > 0:
            ax_p.hist(
                lp,
                bins=list(dep_edges),
                color=hist_dep_color,
                edgecolor=hist_dep_color,
                alpha=0.8,
            )

        if not np.isnan(min_dep[i]) and not np.isnan(max_dep[i]):
            depth_label = (
                f"Layer {i}: {min_dep[i]:.1f}-{max_dep[i]:.1f} m "
                f"(n={int(np.sum(~np.isnan(all_vel[:, i])))})"
            )
        else:
            depth_label = f"Layer {i}: no valid data"

        for _a in [ax_v, ax_d, ax_p]:
            _a.text(
                0.02,
                0.8,
                depth_label,
                transform=_a.transAxes,
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.7},
            )
            _a.grid(True, linestyle="--", alpha=0.6)

        if i == _N_LAYERS // 2:
            ax_v.set_ylabel("Count")
            ax_d.set_ylabel("Count")
            ax_p.set_ylabel("Count")

        if i < _N_LAYERS - 1:
            ax_v.set_xticklabels([])
            ax_d.set_xticklabels([])
            ax_p.set_xticklabels([])
        else:
            ax_v.set_xlabel("Sea Water Speed [m/s]")
            ax_d.set_xlabel("Direction [°]")
            ax_p.set_xlabel("Depth [m]")

        ax_v.set_xlim(vel_edges[0], vel_edges[-1])
        ax_d.set_xlim(0, 360)
        ax_p.set_xlim(dep_edges[0], dep_edges[-1])

    hist_axes[0].set_title("Velocity Distributions by Depth")
    dir_axes[0].set_title("Direction Distributions by Depth")
    dep_axes[0].set_title("Depth Distributions by Layer")

    if show_filtered_stats:
        stats_text = (
            "Data Filtering Summary:\n"
            f"• Original points: {stats['original_points']}\n"
            f"• Dry points removed: {stats['dry_points_removed']}\n"
            f"• Shallow points removed: {stats['shallow_points_removed']}\n"
            f"• Smushed layers removed: {stats['smushed_layers_removed']}\n"
            f"• Final points: {stats['filtered_points']}\n"
            f"• Valid layer-points: "
            f"{stats['total_layers_after_filtering']}/{stats['total_layers_original']}"
        )
        fig.text(
            0.02,
            0.98,
            stats_text,
            transform=fig.transFigure,
            fontsize=9,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "lightblue", "alpha": 0.8},
        )

    if layout == "stacked":
        fig.subplots_adjust(bottom=0.08, left=0.08, right=0.97, top=0.93)
    else:
        fig.subplots_adjust(bottom=0.20, left=0.15)
    return fig, stats


@styled
def plot_velocity_profile(
    df: pd.DataFrame,
    settings: PlotSettings | None = None,
    timestamp_idx: int = 0,
) -> Figure:
    """Plot a single-timestamp velocity profile (speed vs depth).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ``vap_sea_water_speed_layer_{i}`` and
        ``vap_sigma_depth_layer_{i}`` columns.
    settings : PlotSettings, optional
        See :class:`~us_marine_energy_resource.viz.settings.PlotSettings`.
    timestamp_idx : int, optional
        Row index of the timestamp to plot. Default 0.

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
        [f"vap_sea_water_speed_layer_{i}" for i in range(_N_LAYERS)]
        + [f"vap_sigma_depth_layer_{i}" for i in range(_N_LAYERS)],
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    depths = [float(df.iloc[timestamp_idx][f"vap_sigma_depth_layer_{i}"]) for i in range(_N_LAYERS)]
    velocities = [
        float(df.iloc[timestamp_idx][f"vap_sea_water_speed_layer_{i}"]) for i in range(_N_LAYERS)
    ]
    ax.plot(velocities, depths, "o-", linewidth=2, markersize=8)
    ax.set_xlabel("Sea Water Speed [m/s]")
    ax.set_ylabel("Depth [m]")
    ax.set_title(f"Velocity Profile at {df.index[timestamp_idx]}")
    ax.invert_yaxis()
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    return fig


@styled
def plot_tidal_velocity_profile(
    df: pd.DataFrame,
    settings: PlotSettings | None = None,
    timestamp_index: int = 0,
) -> Figure:
    """Plot speed and direction profiles side-by-side for a single timestamp.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ``vap_sea_water_speed_layer_{i}``,
        ``vap_sea_water_to_direction_layer_{i}``, and
        ``vap_sigma_depth_layer_{i}`` columns.
    settings : PlotSettings, optional
        See :class:`~us_marine_energy_resource.viz.settings.PlotSettings`.
    timestamp_index : int, optional
        Row index of the timestamp to plot. Default 0.

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
        [f"vap_sea_water_speed_layer_{i}" for i in range(_N_LAYERS)]
        + [f"vap_sea_water_to_direction_layer_{i}" for i in range(_N_LAYERS)]
        + [f"vap_sigma_depth_layer_{i}" for i in range(_N_LAYERS)],
    )
    data = df.iloc[timestamp_index]
    depths = [float(data[f"vap_sigma_depth_layer_{i}"]) for i in range(_N_LAYERS)]
    speeds = [float(data[f"vap_sea_water_speed_layer_{i}"]) for i in range(_N_LAYERS)]
    directions = [float(data[f"vap_sea_water_to_direction_layer_{i}"]) for i in range(_N_LAYERS)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

    ax1.plot(speeds, depths, "o-", linewidth=2, markersize=8)
    ax1.set_xlabel("Current Speed [m/s]")
    ax1.set_ylabel("Depth [m]")
    ax1.set_title("Velocity Profile")
    ax1.grid(True)
    ax1.invert_yaxis()

    ax2.plot(directions, depths, "o-", linewidth=2, markersize=8, color=sns.color_palette()[1])
    ax2.set_xlabel("Current Direction [degrees]")
    ax2.set_ylabel("Depth [m]")
    ax2.set_title("Direction Profile")
    ax2.grid(True)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 360)

    plt.suptitle(f"Profiles at {df.index[timestamp_index]}")
    plt.tight_layout()
    return fig


@styled
def plot_power_density_profile(
    df: pd.DataFrame,
    settings: PlotSettings | None = None,
    timestamp_index: int = 0,
) -> Figure:
    """Plot power density and speed profiles on twin x-axes for a single timestamp.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ``vap_sea_water_power_density_layer_{i}``,
        ``vap_sea_water_speed_layer_{i}``, and
        ``vap_sigma_depth_layer_{i}`` columns.
    settings : PlotSettings, optional
        See :class:`~us_marine_energy_resource.viz.settings.PlotSettings`.
    timestamp_index : int, optional
        Row index of the timestamp to plot. Default 0.

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
        [f"vap_sea_water_power_density_layer_{i}" for i in range(_N_LAYERS)]
        + [f"vap_sea_water_speed_layer_{i}" for i in range(_N_LAYERS)]
        + [f"vap_sigma_depth_layer_{i}" for i in range(_N_LAYERS)],
    )
    data = df.iloc[timestamp_index]
    depths = [float(data[f"vap_sigma_depth_layer_{i}"]) for i in range(_N_LAYERS)]
    power = [float(data[f"vap_sea_water_power_density_layer_{i}"]) for i in range(_N_LAYERS)]
    speeds = [float(data[f"vap_sea_water_speed_layer_{i}"]) for i in range(_N_LAYERS)]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax2 = ax.twiny()

    ax.plot(
        power,
        depths,
        "o-",
        linewidth=2,
        markersize=8,
        color=sns.color_palette()[0],
        label="Power Density",
    )
    ax.set_xlabel("Power Density [W/m^2]")
    ax.set_ylabel("Depth [m]")
    ax.set_title("Tidal Power Density Profile")
    ax.grid(True)
    ax.invert_yaxis()

    ax2.plot(
        speeds,
        depths,
        "o--",
        linewidth=1.5,
        markersize=6,
        color=sns.color_palette()[1],
        label="Current Speed",
    )
    ax2.set_xlabel("Current Speed [m/s]")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.suptitle(f"Power Density at {df.index[timestamp_index]}")
    plt.tight_layout()
    return fig


@styled
def plot_velocity_shear_profile(df: pd.DataFrame, settings: PlotSettings | None = None) -> Figure:
    """Analyse and visualise velocity shear across the water column.

    Three panels: mean velocity profile, mean shear profile (with error bars),
    and shear variability box plots.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ``vap_sea_water_speed_layer_{i}`` and
        ``vap_sigma_depth_layer_{i}`` columns.
    settings : PlotSettings, optional
        See :class:`~us_marine_energy_resource.viz.settings.PlotSettings`.

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
        [f"vap_sea_water_speed_layer_{i}" for i in range(_N_LAYERS)]
        + [f"vap_sigma_depth_layer_{i}" for i in range(_N_LAYERS)],
    )

    depths = [float(df[f"vap_sigma_depth_layer_{i}"].mean()) for i in range(_N_LAYERS)]

    velocity_diffs: list[np.ndarray] = []
    depth_diffs: list[np.ndarray] = []
    for i in range(_N_LAYERS - 1):
        v1 = df[f"vap_sea_water_speed_layer_{i}"].to_numpy(dtype=float)
        v2 = df[f"vap_sea_water_speed_layer_{i + 1}"].to_numpy(dtype=float)
        d1 = df[f"vap_sigma_depth_layer_{i}"].to_numpy(dtype=float)
        d2 = df[f"vap_sigma_depth_layer_{i + 1}"].to_numpy(dtype=float)
        velocity_diffs.append(v1 - v2)
        depth_diffs.append(d2 - d1)

    shear: list[np.ndarray] = [velocity_diffs[i] / depth_diffs[i] for i in range(_N_LAYERS - 1)]
    iface_depths = [(depths[i] + depths[i + 1]) / 2.0 for i in range(_N_LAYERS - 1)]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 10))

    mean_vel = [float(df[f"vap_sea_water_speed_layer_{i}"].mean()) for i in range(_N_LAYERS)]
    ax1.plot(mean_vel, depths, "o-", linewidth=2, markersize=8)
    ax1.set_xlabel("Mean Current Speed [m/s]")
    ax1.set_ylabel("Depth [m]")
    ax1.set_title("Vertical Velocity Profile")
    ax1.grid(True)
    ax1.invert_yaxis()

    mean_shear = [float(np.mean(s)) for s in shear]
    std_shear = [float(np.std(s)) for s in shear]
    ax2.plot(
        mean_shear, iface_depths, "o-", linewidth=2, markersize=8, color=sns.color_palette()[0]
    )
    ax2.errorbar(
        mean_shear,
        iface_depths,
        xerr=std_shear,
        fmt="none",
        ecolor=sns.color_palette()[0],
        alpha=0.5,
        capsize=5,
    )
    ax2.set_xlabel("Velocity Shear [1/s]")
    ax2.set_ylabel("Depth [m]")
    ax2.set_title("Vertical Shear Profile")
    ax2.grid(True)
    ax2.invert_yaxis()

    bp = ax3.boxplot(shear, positions=iface_depths, vert=False, patch_artist=True, widths=0.5)
    for box in bp["boxes"]:
        box.set(facecolor=sns.color_palette()[0], alpha=0.8)
    for median in bp["medians"]:
        median.set(color=sns.color_palette()[1], linewidth=2)
    ax3.set_xlabel("Velocity Shear [1/s]")
    ax3.set_ylabel("Depth [m]")
    ax3.set_title("Shear Variability")
    ax3.grid(True, axis="x")
    ax3.invert_yaxis()

    total_depth = depths[-1] - depths[0]
    surf_speed = float(df["vap_sea_water_speed_layer_0"].mean())
    bot_speed = float(df["vap_sea_water_speed_layer_9"].mean())
    overall_shear = (surf_speed - bot_speed) / total_depth if total_depth > 0 else 0.0

    fig.text(
        0.5,
        0.01,
        f"Overall Water Column Shear: {overall_shear:.5f} (1/s)",
        ha="center",
        fontsize=12,
        bbox={"facecolor": "white", "alpha": 0.8},
    )

    plt.tight_layout()
    plt.suptitle("Vertical Velocity Shear Analysis", fontsize=16, y=1.02)
    return fig
