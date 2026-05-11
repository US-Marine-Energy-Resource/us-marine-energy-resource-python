"""Sigma-layer cross-section and mesh plots for tidal current data."""

import warnings
from typing import Any, Literal

import cmocean  # type: ignore[import-untyped]
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure

from us_marine_energy_resource.viz._style import _resolve_cmap, styled
from us_marine_energy_resource.viz.settings import PlotSettings

from ._components import _N_LAYERS, _validate_columns


def _plot_sigma_layers(
    df: pd.DataFrame,
    cmap: Any,
    layer_name: str = "speed",
    units: str = "m/s",
    anchor: Literal["bottom", "surface"] = "bottom",
    show_surface_elevation: bool = False,
    ax: Any = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> tuple[Figure, Any]:
    import matplotlib.pyplot as plt
    import numpy as np

    required: list[str] = [f"vap_sea_water_{layer_name}_layer_{i}" for i in range(_N_LAYERS)]
    required += [f"vap_sigma_depth_bound_{i}" for i in range(_N_LAYERS + 1)]
    required += ["vap_sea_floor_depth"]
    _validate_columns(df, required)

    label = layer_name.replace("_", " ").title()

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.figure

    times = df.index

    if not isinstance(times, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex")

    # Convert to matplotlib date numbers so we can use Matplotlib Date formatters
    time_nums = mdates.date2num(times.to_pydatetime())
    n_times = len(time_nums)

    verts: list[list[tuple[float, float]]] = []
    values: list[float] = []
    surface_positions: list[float] = []

    if anchor == "bottom":
        max_bottom = float(df["vap_sigma_depth_bound_10"].max())

        for t in range(n_times):
            current_bottom = float(df["vap_sigma_depth_bound_10"].iloc[t])
            offset = max_bottom - current_bottom
            surface_positions.append(offset)

            # define time span
            if t < n_times - 1:
                t0 = time_nums[t]
                t1 = time_nums[t + 1]
            else:
                dt = time_nums[t] - time_nums[t - 1]
                t0 = time_nums[t]
                t1 = time_nums[t] + dt

            for i in range(_N_LAYERS):
                top = float(df[f"vap_sigma_depth_bound_{i}"].iloc[t]) + offset
                bot = float(df[f"vap_sigma_depth_bound_{i + 1}"].iloc[t]) + offset
                val = float(df[f"vap_sea_water_{layer_name}_layer_{i}"].iloc[t])

                values.append(val)
                verts.append([(t0, top), (t1, top), (t1, bot), (t0, bot)])

    else:
        for t in range(n_times):
            if t < n_times - 1:
                t0 = time_nums[t]
                t1 = time_nums[t + 1]
            else:
                dt = time_nums[t] - time_nums[t - 1]
                t0 = time_nums[t]
                t1 = time_nums[t] + dt

            for i in range(_N_LAYERS):
                top = float(df[f"vap_sigma_depth_bound_{i}"].iloc[t])
                bot = float(df[f"vap_sigma_depth_bound_{i + 1}"].iloc[t])
                val = float(df[f"vap_sea_water_{layer_name}_layer_{i}"].iloc[t])

                values.append(val)
                verts.append([(t0, top), (t1, top), (t1, bot), (t0, bot)])

    poly = PolyCollection(
        verts,
        array=np.array(values),
        cmap=cmap,
        edgecolors="face",
        linewidths=0,
    )

    if vmin is not None or vmax is not None:
        _clim_min = vmin if vmin is not None else float(np.nanmin(values))
        _clim_max = vmax if vmax is not None else float(np.nanmax(values))
        poly.set_clim(_clim_min, _clim_max)

        arr = np.asarray(values)
        n_total = arr.size
        if n_total > 0:
            n_clipped = int(np.sum((arr < _clim_min) | (arr > _clim_max)))
            frac = n_clipped / n_total
            if frac > 0.01:
                warnings.warn(
                    f"{frac * 100:.1f}% of data values ({n_clipped:,} of {n_total:,}) "
                    f"fall outside the colorbar range [{_clim_min}, {_clim_max}] and "
                    "will be clipped to the colorbar extremes.",
                    UserWarning,
                    stacklevel=4,
                )

    ax.add_collection(poly)

    # PolyCollection doesn't reliably drive y-axis auto-scaling, so compute
    # the full y extent from the actual vertex data and set it explicitly.
    all_y = [coord[1] for quad in verts for coord in quad]
    ax.set_ylim(max(all_y), min(all_y))  # inverted: deep at bottom, surface at top

    if anchor == "bottom" and show_surface_elevation and surface_positions:
        ax.plot(
            time_nums,
            surface_positions,
            color=sns.color_palette()[0],
            linewidth=0.5,
            label="Surface Elevation",
        )

    cbar = plt.colorbar(poly, ax=ax)
    if len(units) > 10:
        cbar.set_label(f"{label}\n[{units}]")
    else:
        cbar.set_label(f"{label} [{units}]")

    ax.set_xlim(time_nums[0], time_nums[-1])
    ax.set_xlabel("Time [UTC]")
    ax.set_ylabel("Elevation [m]" if anchor == "bottom" else "Depth [m]")

    # matplotlib datetime formatting
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    plt.tight_layout()
    return fig, ax


@styled
def plot_sigma_layers_speed(
    df: pd.DataFrame,
    anchor: Literal["bottom", "surface"] = "bottom",
    show_surface_elevation: bool = False,
    settings: PlotSettings | None = None,
    ax: Any = None,
) -> tuple[Figure, Any]:
    """Plot sea water speed across sigma layers over time.

    A depth-time cross-section coloured by current speed using the
    ``cmocean thermal`` colormap.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a ``DatetimeIndex`` and columns
        ``vap_sea_water_speed_layer_{i}`` (0-9),
        ``vap_sigma_depth_bound_{i}`` (0-10), and
        ``vap_sea_floor_depth``.
    anchor : {"bottom", "surface"}, optional
        Which boundary to pin on the y-axis:

        * ``"bottom"`` — seafloor fixed; tidal surface visible (default).
        * ``"surface"`` — water surface fixed at 0; seafloor oscillates.
    show_surface_elevation : bool, optional
        When ``True`` and *anchor* is ``"bottom"``, overlay a line showing
        the tidal surface elevation. Default is ``False``.
    settings : PlotSettings, optional
        Shared plot settings (e.g. ``start_date``, ``end_date``).
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on. Creates a new figure when ``None``.

    Returns
    -------
    fig : Figure
        Parent figure.
    ax : matplotlib.axes.Axes
        Axes containing the plot.

    Raises
    ------
    KeyError
        If required columns are absent from *df*.
    """
    return _plot_sigma_layers(
        df,
        cmap=_resolve_cmap(settings, cmocean.cm.thermal),  # type: ignore[attr-defined]
        layer_name="speed",
        units="m/s",
        anchor=anchor,
        show_surface_elevation=show_surface_elevation,
        ax=ax,
        vmin=getattr(settings, "colorbar_min", None),
        vmax=getattr(settings, "colorbar_max", None),
    )


@styled
def plot_sigma_layers_direction(
    df: pd.DataFrame,
    anchor: Literal["bottom", "surface"] = "bottom",
    show_surface_elevation: bool = False,
    settings: PlotSettings | None = None,
    ax: Any = None,
) -> tuple[Figure, Any]:
    """Plot sea water direction across sigma layers over time.

    A depth-time cross-section coloured by current direction (°CW from
    True North) using the ``cmocean phase`` colormap.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a ``DatetimeIndex`` and columns
        ``vap_sea_water_to_direction_layer_{i}`` (0-9),
        ``vap_sigma_depth_bound_{i}`` (0-10), and
        ``vap_sea_floor_depth``.
    anchor : {"bottom", "surface"}, optional
        Which boundary to pin on the y-axis:

        * ``"bottom"`` — seafloor fixed; tidal surface visible (default).
        * ``"surface"`` — water surface fixed at 0; seafloor oscillates.
    show_surface_elevation : bool, optional
        When ``True`` and *anchor* is ``"bottom"``, overlay a line showing
        the tidal surface elevation. Default is ``False``.
    settings : PlotSettings, optional
        Shared plot settings (e.g. ``start_date``, ``end_date``).
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on. Creates a new figure when ``None``.

    Returns
    -------
    fig : Figure
        Parent figure.
    ax : matplotlib.axes.Axes
        Axes containing the plot.

    Raises
    ------
    KeyError
        If required columns are absent from *df*.
    """
    return _plot_sigma_layers(
        df,
        cmap=_resolve_cmap(settings, cmocean.cm.phase),  # type: ignore[attr-defined]
        layer_name="to_direction",
        units="Deg CW from True North",
        anchor=anchor,
        show_surface_elevation=show_surface_elevation,
        ax=ax,
        vmin=getattr(settings, "colorbar_min", None),
        vmax=getattr(settings, "colorbar_max", None),
    )


@styled
def plot_speed_mesh(
    df: pd.DataFrame,
    settings: PlotSettings | None = None,
) -> Figure:
    """Create a 2-D colour-mesh of current speed over time and depth.

    Uses ``imshow`` with the viridis colormap, treating each sigma layer as a
    uniform depth band and the time axis as the x-axis.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ``vap_sea_water_speed_layer_{i}`` and
        ``vap_sigma_depth_layer_{i}`` columns.
    settings : PlotSettings, optional
        Shared plot settings (e.g. ``start_date``, ``end_date``).

    Returns
    -------
    fig : Figure
        The created matplotlib figure.

    Raises
    ------
    KeyError
        If required columns are absent from *df*.
    """
    import matplotlib.dates as _mdates

    _validate_columns(
        df,
        [f"vap_sea_water_speed_layer_{i}" for i in range(_N_LAYERS)]
        + [f"vap_sigma_depth_layer_{i}" for i in range(_N_LAYERS)],
    )

    timestamps = df.index
    all_vel: np.ndarray = np.column_stack(
        [df[f"vap_sea_water_speed_layer_{i}"].to_numpy(dtype=float) for i in range(_N_LAYERS)]
    )
    all_dep: np.ndarray = np.column_stack(
        [df[f"vap_sigma_depth_layer_{i}"].to_numpy(dtype=float) for i in range(_N_LAYERS)]
    )

    vmin, vmax = float(np.nanmin(all_vel)), float(np.nanmax(all_vel))
    time_num = _mdates.date2num(timestamps)
    bottom_depths = float(np.nanmax(all_dep))

    # Simple sinusoidal surface elevation proxy (M2 period)
    hours = np.array([(t - timestamps[0]).total_seconds() / 3600.0 for t in timestamps])
    surface_elev = -0.5 * np.sin(2 * np.pi * hours / 12.4)

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(
        all_vel.T,
        aspect="auto",
        cmap=_resolve_cmap(settings, plt.cm.viridis),  # type: ignore[attr-defined]
        vmin=vmin,
        vmax=vmax,
        extent=(time_num[0], time_num[-1], bottom_depths, float(np.min(surface_elev))),
        origin="upper",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Current Speed [m/s]")

    ax.xaxis_date()
    ax.xaxis.set_major_formatter(_mdates.DateFormatter("%b %d"))
    ax.set_xlabel("Time")
    ax.set_ylabel("Depth [m]")
    ax.set_title("Current Speed Throughout Water Column")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    return fig
