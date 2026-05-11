"""Joint probability distribution plots for tidal current speed and direction."""

import bisect
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from mhkit.tidal.resource import _histogram as _mhkit_histogram  # type: ignore[import-untyped]
from mhkit.utils import convert_to_dataarray as _mhkit_convert  # type: ignore[import-untyped]
from scipy.interpolate import interpn as _scipy_interpn

from us_marine_energy_resource.viz._style import styled
from us_marine_energy_resource.viz.settings import PlotSettings

from ._components import _validate_columns


def _setup_polar_axes(
    ax: Any,
    metadata: dict[str, Any] | None = None,
    flood: float | None = None,
    ebb: float | None = None,
    metadata_fontsize: float = 11.0,
) -> None:
    """Configure a polar axes with cardinal directions and an optional metadata box.

    This replicates ``mhkit.tidal.graphics._initialize_polar`` but uses
    ``ax``-scoped calls instead of ``plt.*`` globals so it is safe to use
    inside multi-panel figures.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        A polar axes to configure in-place.
    metadata : dict, optional
        When provided must contain ``"name"`` (title string), ``"lat"``, and
        ``"lon"`` keys.  The name is set as the axes title; lat/lon appear in
        a text box inside the plot.
    flood : float, optional
        Flood direction in degrees added as a major tick label.
    ebb : float, optional
        Ebb direction in degrees added as a major tick label.
    metadata_fontsize : float, optional
        Font size for the lat/lon metadata text box.  Default 11.
    """
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    xticks = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    xtick_degrees = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]

    if flood is not None:
        bisect.insort(xtick_degrees, float(flood))
        idx = xtick_degrees.index(float(flood))
        xticks[idx:idx] = ["\nFlood"]

    if ebb is not None:
        bisect.insort(xtick_degrees, float(ebb))
        idx = xtick_degrees.index(float(ebb))
        xticks[idx:idx] = ["\nEbb"]

    ax.set_xticks(np.array(xtick_degrees) * np.pi / 180.0)
    ax.set_xticklabels(xticks)

    if metadata is not None:
        ax.set_title(metadata["name"])
        bouy_data = "\n".join([
            f'Lat = {float(metadata["lat"]):0.2f}\u00b0',
            f'Lon = {float(metadata["lon"]):0.2f}\u00b0',
        ])
        ax.text(
            -0.3,
            0.80,
            bouy_data,
            transform=ax.transAxes,
            fontsize=metadata_fontsize,
            verticalalignment="top",
            bbox={"facecolor": "none", "edgecolor": "k", "pad": 5},
        )


def _render_jpd_scatter(
    directions: pd.Series,
    velocities: pd.Series,
    direction_bin_width_deg: float,
    velocity_bin_width_ms: float,
    ax: Any,
    metadata: dict[str, Any] | None = None,
    flood: float | None = None,
    ebb: float | None = None,
    metadata_fontsize: float = 11.0,
) -> tuple[Any, Any]:
    """Scatter-plot a JPD on a polar axes without adding a colorbar.

    Replicates the core of
    ``mhkit.tidal.graphics.plot_joint_probability_distribution`` but separates
    colorbar creation so callers can size it correctly for their layout
    (single plot vs. multi-panel grid).

    Parameters
    ----------
    directions : pd.Series
        Current direction in degrees (0 = true north).
    velocities : pd.Series
        Current speed in m/s.
    direction_bin_width_deg : float
        Width of direction bins for the 2-D histogram in degrees.
    velocity_bin_width_ms : float
        Width of velocity bins for the 2-D histogram in m/s.
    ax : matplotlib.axes.Axes
        Polar axes to draw into.
    metadata : dict, optional
        Passed to :func:`_setup_polar_axes`.
    flood : float, optional
        Flood direction in degrees for tick label.
    ebb : float, optional
        Ebb direction in degrees for tick label.
    metadata_fontsize : float, optional
        Font size for the lat/lon metadata box.  Default 11.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes (same object passed in) with the scatter rendered.
    sx : matplotlib.collections.PathCollection
        The scatter ``PathCollection``; pass to :func:`_add_jpd_colorbar`.
    """
    directions_da = _mhkit_convert(directions)
    velocities_da = _mhkit_convert(velocities)

    histogram, dir_edges_raw, vel_edges_raw = _mhkit_histogram(
        directions.to_numpy(), velocities.to_numpy(),
        direction_bin_width_deg, velocity_bin_width_ms,
    )
    # mhkit returns plain lists for edges; convert so numpy arithmetic works.
    dir_edges: np.ndarray = np.asarray(dir_edges_raw)
    vel_edges: np.ndarray = np.asarray(vel_edges_raw)

    _setup_polar_axes(
        ax, metadata=metadata, flood=flood, ebb=ebb, metadata_fontsize=metadata_fontsize
    )

    dir_bins = 0.5 * (dir_edges[1:] + dir_edges[:-1])
    vel_bins = 0.5 * (vel_edges[1:] + vel_edges[:-1])
    dir_bins[[0, -1]] = dir_edges[[0, -1]]
    vel_bins[[0, -1]] = vel_edges[[0, -1]]

    pts = np.vstack([directions_da.to_numpy(), velocities_da.to_numpy()]).T
    z: np.ndarray = _scipy_interpn(  # type: ignore[call-overload]
        (dir_bins, vel_bins), histogram, pts, method="splinef2d", bounds_error=False,  # type: ignore[arg-type]
    )

    idx = z.argsort()
    theta = directions_da.to_numpy()[idx] * np.pi / 180
    r = velocities_da.to_numpy()[idx]

    # zorder=3 keeps scatter above grid lines (z≈2) and the polar patch (z=1)
    # after set_axisbelow(False) is applied in _set_radial_ticks.
    sx = ax.scatter(theta, r, c=z[idx], s=5, edgecolor=None, zorder=3)
    return ax, sx


def _set_radial_ticks(ax: Any, rmax: float) -> None:
    """Set explicit, stable speed-ring tick positions and labels on a polar axes.

    Uses :class:`~matplotlib.ticker.MaxNLocator` to compute a clean set of
    tick values for [0, *rmax*], then calls ``set_yticks`` + ``set_yticklabels``
    together so the rings are frozen and independent of figure size.

    Calling ``set_yticklabels`` alone (without ``set_yticks``) is unsafe:
    matplotlib's auto-locator recomputes tick positions when the figure is
    resized, making the manually-set labels slide off the wrong rings.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Polar axes to configure.
    rmax : float
        Upper radial limit (shared ``global_rmax`` for grid plots, or the
        single-plot axes maximum).
    """
    from matplotlib.ticker import MaxNLocator

    locator = MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10])
    ring_ticks = np.asarray([
        t for t in locator.tick_values(0.0, rmax) if 0.0 < t <= rmax
    ])
    ax.set_yticks(ring_ticks)
    ax.set_yticklabels([f"{t:.1f} m/s" for t in ring_ticks])

    # seaborn's theme sets axes.axisbelow=True which drops the whole Axis object
    # (including tick labels) to zorder=0.5 — below the polar background patch
    # (zorder=1) and scatter data (zorder=3).  Setting zorder on individual Text
    # children cannot override the parent Axis zorder.
    # set_axisbelow(False) raises the Axis to zorder=2.5 so labels render above
    # the background patch.  Scatter is rendered at zorder=3 (set in
    # _render_jpd_scatter) so it stays above the grid rings.
    ax.set_axisbelow(False)


def _add_jpd_colorbar(
    ax: Any,
    sx: Any,
    label: str = "Joint Probability [%]",
    inset_rect: tuple[float, float, float, float] = (1.05, 0.1, 0.05, 0.8),
) -> Any:
    """Add a colorbar anchored to *ax* rather than the full figure.

    ``plt.colorbar(sx, ax=polar_ax)`` expands to full figure height when the
    polar axes lives inside a multi-panel grid because matplotlib's
    ``make_axes`` mis-calculates the bounding box for polar subplots.
    Using ``ax.inset_axes`` positions the colorbar relative to *ax*'s own
    bounding box, keeping it correctly proportioned in any layout.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Polar axes the colorbar should be anchored to.
    sx : matplotlib.collections.PathCollection
        Scatter mappable returned by :func:`_render_jpd_scatter`.
    label : str, optional
        Colorbar axis label.  Default ``"Joint Probability [%]"``.
    inset_rect : tuple of float, optional
        ``(x0, y0, width, height)`` in axes-fraction coordinates passed to
        ``ax.inset_axes``.  Default ``(1.05, 0.1, 0.05, 0.8)`` places a slim
        bar just outside the right edge, vertically centred on the plot circle.

    Returns
    -------
    cb : matplotlib.colorbar.Colorbar
        The created colorbar object.
    """
    cax = ax.inset_axes(list(inset_rect))
    fig: Figure = ax.get_figure()  # type: ignore[assignment]
    cb = fig.colorbar(sx, cax=cax, label=label)
    return cb


@styled
def generate_tidal_joint_probability(
    df: pd.DataFrame,
    sigma_layer: int,
    settings: PlotSettings | None = None,
    direction_bin_width_deg: float = 1.0,
    velocity_bin_width_ms: float = 0.1,
    ax: Any | None = None,
) -> Figure:
    """Plot a joint probability distribution of tidal current speed and direction.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ``vap_sea_water_to_direction_layer_{sigma_layer}``,
        ``vap_sea_water_speed_layer_{sigma_layer}``,
        ``vap_sigma_depth_layer_{sigma_layer}``, ``dataset_name``,
        ``lat_center``, and ``lon_center`` columns.
    sigma_layer : int
        Zero-based sigma layer index.
    settings : PlotSettings, optional
        See :class:`~us_marine_energy_resource.viz.settings.PlotSettings`.
    direction_bin_width_deg : float, optional
        Width of direction bins in degrees. Default 1.0.
    velocity_bin_width_ms : float, optional
        Width of velocity bins in m/s. Default 0.1.
    ax : matplotlib.axes.Axes, optional
        Existing polar axes to draw on. When provided the plot is drawn into
        *ax* and ``ax.figure`` is returned.  When ``None`` a new figure is
        created.

    Returns
    -------
    fig : Figure
        The created matplotlib figure.

    Raises
    ------
    KeyError
        If required columns are absent from *df*.
    """
    required = [
        f"vap_sea_water_to_direction_layer_{sigma_layer}",
        f"vap_sea_water_speed_layer_{sigma_layer}",
        f"vap_sigma_depth_layer_{sigma_layer}",
        "dataset_name",
        "lat_center",
        "lon_center",
    ]
    _validate_columns(df, required)

    to_direction = df[f"vap_sea_water_to_direction_layer_{sigma_layer}"]
    speed = df[f"vap_sea_water_speed_layer_{sigma_layer}"]
    depth = df[f"vap_sigma_depth_layer_{sigma_layer}"]

    time_str = f"Time Range: {df.index[0]} - {df.index[-1]} [UTC]"
    depth_str = f"Depth Range: {depth.min():.2f} - {depth.max():.2f} [m]"
    speed_str = f"Speed Range: {speed.min():.2f} - {speed.max():.2f} [m/s]"

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={"projection": "polar"})
    else:
        fig = ax.get_figure()  # type: ignore[assignment]

    ax, sx = _render_jpd_scatter(
        to_direction,
        speed,
        direction_bin_width_deg,
        velocity_bin_width_ms,
        ax=ax,
        metadata={
            "name": f"{time_str}\n{speed_str}\n{depth_str}",
            "lat": float(df["lat_center"].iloc[0]),
            "lon": float(df["lon_center"].iloc[0]),
        },
    )

    _set_radial_ticks(ax, float(ax.get_ylim()[1]))  # type: ignore[union-attr]
    _add_jpd_colorbar(ax, sx)

    plt.tight_layout()
    return fig  # type: ignore[return-value]
