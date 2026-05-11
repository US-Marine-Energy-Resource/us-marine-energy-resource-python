"""Multi-site tidal resource comparison plots."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from us_marine_energy_resource.analysis.resource import SiteSummaryMetrics
from us_marine_energy_resource.viz._style import styled
from us_marine_energy_resource.viz.settings import PlotSettings


def _shorten_name(name: str) -> str:
    """Return the portion of a site name before the first comma."""
    return name.split(",")[0].strip()


@styled
def plot_multi_site_comparison(
    site_summaries: list[SiteSummaryMetrics],
    turbine_config_label: str = "",
    settings: PlotSettings | None = None,
) -> Figure:
    """Create a six-panel bar chart comparing resource metrics across sites.

    Each panel shows one key metric as a horizontal bar chart with one bar
    per site.  Sites are color-coded consistently across all panels.

    Parameters
    ----------
    site_summaries : list[SiteSummaryMetrics]
        One entry per site, as returned by
        :func:`us_marine_energy_resource.analysis.collect_site_metrics`.
    turbine_config_label : str, optional
        Short description of the turbine configuration to include in the
        figure title (e.g. ``"D=10 m, rated=1.5 m/s, η=40%"``).
    settings : PlotSettings, optional
        See :class:`~us_marine_energy_resource.viz.settings.PlotSettings`.

    Returns
    -------
    fig : Figure
        The created matplotlib figure.
    """
    n_sites = len(site_summaries)
    short_names = [_shorten_name(s["site_name"]) for s in site_summaries]
    colors = plt.cm.tab10(np.linspace(0, 0.9, n_sites))  # type: ignore[attr-defined]

    metrics: list[tuple[str, str, str]] = [
        ("mean_speed", "Mean Speed", "m/s"),
        ("mean_power_density", "Mean Power Density", "W/m²"),
        ("usable_time_pct", "Usable Time (≥ cut-in)", "%"),
        ("p90_speed", "P90 Current Speed", "m/s"),
        ("max_speed", "Max Current Speed", "m/s"),
        ("average_tidal_range", "Avg Tidal Range", "m"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()

    for ax, (key, label, unit) in zip(axes_flat, metrics, strict=False):
        values = [float(s[key]) for s in site_summaries]  # type: ignore[literal-required]
        bars = ax.barh(
            short_names,
            values,
            color=colors,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_xlabel(f"[{unit}]", fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.grid(True, axis="x", linestyle="--", alpha=0.5)
        ax.set_xlim(left=0)
        ax.tick_params(axis="y", labelsize=9)

        for bar, val in zip(bars, values, strict=False):
            ax.text(
                bar.get_width() * 1.01,
                bar.get_y() + bar.get_height() / 2.0,
                f"{val:.2f}",
                va="center",
                ha="left",
                fontsize=8,
            )

    title = "Multi-Site Tidal Resource Comparison"
    if turbine_config_label:
        title += f"\n{turbine_config_label}"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    plt.tight_layout()
    return fig


@styled
def plot_jpd_comparison_grid(
    site_records: list[tuple[str, pd.DataFrame, int]],
    settings: PlotSettings | None = None,
    direction_bin_width_deg: float = 1.0,
    velocity_bin_width_ms: float = 0.1,
    colorbar_max: float | None = None,
    ncols: int = 3,
) -> Figure:
    """Plot a grid of joint probability distributions for any number of tidal sites.

    Each panel has its own colorbar showing the joint probability [%] for that
    site.  The radial speed rings are normalized to a shared maximum across all
    panels so that current magnitudes are directly comparable.

    Parameters
    ----------
    site_records : list of (name, df, sigma_layer)
        One tuple per site.  *sigma_layer* is the sigma-layer index to use for
        that site (e.g. as returned by
        :func:`~us_marine_energy_resource.analysis.select_layer_for_depth`).
        Any number of entries >= 1 is accepted.
    settings : PlotSettings, optional
        See :class:`~us_marine_energy_resource.viz.settings.PlotSettings`.
    direction_bin_width_deg : float, optional
        Width of direction bins in degrees.  Default 1.0.
    velocity_bin_width_ms : float, optional
        Width of velocity bins in m/s.  Default 0.1.
    colorbar_max : float, optional
        When set, clamps every panel's color scale to this upper bound (%)
        so sites can be compared on a shared probability scale.  When
        ``None`` (default) each panel uses its own data-driven maximum.
    ncols : int, optional
        Number of columns in the grid.  The number of rows is computed
        automatically as ``ceil(len(site_records) / ncols)``.  Default 3.

    Returns
    -------
    fig : Figure
        The comparison grid figure.

    Raises
    ------
    ValueError
        If *site_records* is empty or *ncols* < 1.
    """
    import math

    from us_marine_energy_resource.viz.tidal.joint_probability import (
        _add_jpd_colorbar,
        _render_jpd_scatter,
        _set_radial_ticks,
    )

    if len(site_records) == 0:
        raise ValueError("site_records must contain at least one entry.")
    if ncols < 1:
        raise ValueError(f"ncols must be >= 1, got {ncols}.")

    n = len(site_records)
    nrows = math.ceil(n / ncols)

    # Use the caller-supplied dimensions as the total figure size; fall back to
    # a sensible per-panel default of 5x5 so the figure scales with grid shape.
    _panel_w, _panel_h = 5.0, 5.0
    fig_w: float = float(getattr(settings, "fig_width", None) or _panel_w * ncols)
    fig_h: float = float(getattr(settings, "fig_height", None) or _panel_h * nrows)

    # constrained_layout handles inset axes (the per-panel colorbars) far
    # better than tight_layout: it eliminates the excess vertical whitespace
    # that tight_layout introduces when inset axes extend beyond subplot bounds.
    fig, axes_arr = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_w, fig_h),
        subplot_kw={"projection": "polar"},
        constrained_layout=True,
    )

    # Normalize axes to a flat list regardless of grid shape.
    if nrows == 1 and ncols == 1:
        axes_flat: list[Any] = [axes_arr]
    elif nrows == 1 or ncols == 1:
        axes_flat = list(np.asarray(axes_arr).flatten())
    else:
        axes_flat = list(axes_arr.flatten())  # type: ignore[union-attr]

    # Hide any unused trailing panels.
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    # Render each panel's scatter and collect (ax, scatter) pairs.
    panel_data: list[tuple[Any, Any]] = []
    for (name, df, layer), ax in zip(site_records, axes_flat[:n], strict=True):
        to_direction = df[f"vap_sea_water_to_direction_layer_{layer}"]
        speed = df[f"vap_sea_water_speed_layer_{layer}"]
        depth = df[f"vap_sigma_depth_layer_{layer}"]
        depth_str = f"~{float(depth.mean()):.1f} m"

        ax, sx = _render_jpd_scatter(
            to_direction,
            speed,
            direction_bin_width_deg,
            velocity_bin_width_ms,
            ax=ax,
            metadata=None,
            metadata_fontsize=8.0,
        )
        ax.set_title(f"{name}\n({depth_str})", fontsize=9, fontweight="bold")
        panel_data.append((ax, sx))

    # Normalize radial (speed) axis so all panels share the same rings.
    global_rmax = max(float(ax.get_ylim()[1]) for ax, _ in panel_data)
    for ax, _ in panel_data:
        ax.set_ylim(0, global_rmax)
        _set_radial_ticks(ax, global_rmax)

    # Optionally clamp color scale across all panels.
    if colorbar_max is not None:
        for _, sx in panel_data:
            sx.set_clim(0.0, colorbar_max)

    # Add a properly-sized inset colorbar for each panel.
    # The inset rect is in axes-fraction coords: [x0, y0, width, height].
    # x0=1.06 places the bar just outside the right edge of the polar circle;
    # height=0.75 keeps it proportional to the plot rather than the full figure.
    for ax, sx in panel_data:
        _add_jpd_colorbar(ax, sx, inset_rect=(1.06, 0.125, 0.045, 0.75))
    return fig
