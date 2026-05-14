"""Shared settings dataclass for tidal visualization functions."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from us_marine_energy_resource.analysis.preprocessing import (
    DepthMode,
    sigma_depth_axis_label,
    sigma_layer_depth_col,
)

OutputFormat = Literal["svg", "png", "pdf", "eps"]


@dataclass
class DepthPerspective:
    """Depth coordinate convention and display sign for sigma-layer visualizations.

    Controls which reference frame is used for the depth axis and whether larger
    values represent shallower or deeper positions.

    Parameters
    ----------
    mode : DepthMode
        Reference frame. Determines which DataFrame columns are read and the
        natural axis direction. Defaults to :attr:`DepthMode.FixedBottom`.
    positive_up : bool, optional
        When ``True`` larger values represent shallower positions (y-axis
        increases upward). When ``False`` larger values represent deeper
        positions (y-axis increases downward). ``None`` (default) infers from
        *mode*: ``FixedBottom`` and ``Navd88Elevation`` default to ``True``;
        ``FixedSurface`` and ``Navd88Depth`` default to ``False``.

    Examples
    --------
    Fixed-bottom elevation (default global):

    >>> DepthPerspective()

    Oceanographic convention (depth from surface):

    >>> DepthPerspective(mode=DepthMode.FixedSurface)

    Negative depth (0 at surface, −30 m at seafloor):

    >>> DepthPerspective(mode=DepthMode.FixedSurface, positive_up=True)

    NAVD88 elevation:

    >>> DepthPerspective(mode=DepthMode.Navd88Elevation)
    """

    mode: DepthMode = field(default=DepthMode.FixedBottom)
    positive_up: bool | None = field(default=None)
    depth_label_override: str | None = field(default=None)

    def effective_positive_up(self) -> bool:
        """Resolve ``positive_up``, inferring from *mode* when ``None``."""
        if self.positive_up is not None:
            return self.positive_up
        return self.mode in (DepthMode.FixedBottom, DepthMode.Navd88Elevation)

    def display_array(self, values: np.ndarray) -> np.ndarray:
        """Apply sign convention to depth values before plotting.

        Negates *values* when the mode's natural direction disagrees with
        the requested :attr:`positive_up` setting.
        """
        is_upward_mode = self.mode in (DepthMode.FixedBottom, DepthMode.Navd88Elevation)
        if is_upward_mode != self.effective_positive_up():
            return -values
        return values

    def should_invert_axis(self) -> bool:
        """Whether to call ``ax.invert_yaxis()`` after plotting."""
        return not self.effective_positive_up()

    def depth_col(self, layer: int) -> str:
        """Return the DataFrame column name for sigma *layer* in this mode."""
        return sigma_layer_depth_col(layer, self.mode)

    def depth_label(self) -> str:
        """Return the depth axis label for this perspective.

        Returns :attr:`depth_label_override` when set, otherwise
        auto-generates from *mode* (e.g. ``"Height Above Seafloor [m]"``
        for :attr:`DepthMode.FixedBottom`). May be applied to either the
        x or y axis depending on the plot type.
        """
        if self.depth_label_override is not None:
            return self.depth_label_override
        return sigma_depth_axis_label(self.mode)


_DEFAULT_DEPTH_PERSPECTIVE: DepthPerspective = DepthPerspective()


def _as_depth_perspective(v: DepthPerspective | DepthMode) -> DepthPerspective:
    if isinstance(v, DepthMode):
        return DepthPerspective(mode=v)
    return v


def set_depth_perspective(perspective: DepthPerspective | DepthMode) -> None:
    """Set the module-level default :class:`DepthPerspective` for all plot functions.

    All subsequent plot calls that do not supply an explicit
    ``depth_perspective`` in their :class:`PlotSettings` will use this
    perspective.

    Parameters
    ----------
    perspective : DepthPerspective | DepthMode
        New default depth perspective.  A bare :class:`DepthMode` is
        automatically wrapped in a :class:`DepthPerspective` with default
        display options.

    See Also
    --------
    depth_perspective_context : Scoped, reversible alternative.
    get_depth_perspective : Resolves the effective perspective for a call.
    """
    global _DEFAULT_DEPTH_PERSPECTIVE
    _DEFAULT_DEPTH_PERSPECTIVE = _as_depth_perspective(perspective)


def get_depth_perspective(settings: PlotSettings | None = None) -> DepthPerspective:
    """Resolve the effective :class:`DepthPerspective` for a plot call.

    Priority: ``settings.depth_perspective`` > module-level default.

    Parameters
    ----------
    settings : PlotSettings, optional
        Per-call settings.  When ``None`` or when
        ``settings.depth_perspective`` is ``None``, returns the global
        default set by :func:`set_depth_perspective`.

    Returns
    -------
    DepthPerspective
    """
    if settings is not None and settings.depth_perspective is not None:
        return _as_depth_perspective(settings.depth_perspective)
    return _DEFAULT_DEPTH_PERSPECTIVE


@contextmanager
def depth_perspective_context(perspective: DepthPerspective | DepthMode) -> Iterator[None]:
    """Context manager for a scoped :class:`DepthPerspective` override.

    Temporarily replaces the module-level default for the duration of the
    ``with`` block, then restores the previous value on exit.

    Parameters
    ----------
    perspective : DepthPerspective | DepthMode
        Perspective to apply within the context.  A bare :class:`DepthMode`
        is automatically wrapped in a :class:`DepthPerspective` with default
        display options.

    Examples
    --------
    >>> with depth_perspective_context(DepthMode.FixedSurface):
    ...     tidal.plot_velocity_profile_with_histograms(df, settings=s)
    """
    global _DEFAULT_DEPTH_PERSPECTIVE
    previous = _DEFAULT_DEPTH_PERSPECTIVE
    _DEFAULT_DEPTH_PERSPECTIVE = _as_depth_perspective(perspective)
    try:
        yield
    finally:
        _DEFAULT_DEPTH_PERSPECTIVE = previous


@dataclass
class PlotSettings:
    """Configuration options shared across all tidal visualization functions.

    Pass an instance to any ``plot_*`` or ``generate_*`` function to control
    time windowing, figure sizing, titles, labels, colormaps, and datetime
    axis formatting.

    Parameters
    ----------
    start_date : str or pd.Timestamp, optional
        Inclusive start of the time window.  Accepts any string parseable by
        :func:`pandas.to_datetime` (e.g. ``"2010-01"``, ``"2010-01-15 06:00"``).
        ``None`` keeps the earliest timestamp in the DataFrame.
    end_date : str or pd.Timestamp, optional
        Inclusive end of the time window.  Same format as *start_date*.
        ``None`` keeps the latest timestamp in the DataFrame.
    fig_width : float, optional
        Override the figure width (inches) for this call only.  Supersedes
        ``PLOT_CONFIG["fig_width"]``.  ``None`` defers to the global config.
    fig_height : float, optional
        Override the figure height (inches) for this call only.  When set,
        the ``PLOT_CONFIG["max_fig_height"]`` cap is also bypassed.
        ``None`` defers to the global config.
    title : str, optional
        Override the primary figure title.  On single-axis figures this calls
        ``ax.set_title(title)``; on multi-axis figures it calls
        ``fig.suptitle(title)``.  ``None`` keeps the function's default title.
    subtitle : str, optional
        Secondary label placed in smaller italic text near the top of the
        figure via ``fig.text``.  Appears below the main title.
    caption : str, optional
        Annotation placed at the bottom-left of the figure (e.g. coordinate
        info, data source).  Rendered via ``fig.text(0.01, 0.01, ...)``.
    xlabel : str, optional
        Override the x-axis label.  Only applied on **single-axis** figures.
        Ignored on multi-panel figures.
    ylabel : str, optional
        Override the y-axis label.  Only applied on **single-axis** figures.
        Ignored on multi-panel figures.
    colormap : str or Colormap, optional
        Override the **primary** colormap used by the plot (the main data
        fill / heatmap color scale).  Accepts any matplotlib colormap name
        or object.  Secondary layer-color palettes are unaffected.
        ``None`` uses the function's built-in default.
    colorbar_min : float, optional
        Lower bound for the colorbar color scale.  When set, values below
        this threshold are clipped to the minimum color.  A
        :class:`UserWarning` is issued if more than 1 % of the plotted data
        falls below this value.  ``None`` uses the data minimum.
    colorbar_max : float, optional
        Upper bound for the colorbar color scale.  When set, values above
        this threshold are clipped to the maximum color.  A
        :class:`UserWarning` is issued if more than 1 % of the plotted data
        exceeds this value.  ``None`` uses the data maximum.
    datetime_format : str, optional
        Explicit :func:`~datetime.datetime.strftime` format string for
        datetime tick labels (e.g. ``"%Y-%m"``).  Takes priority over
        *datetime_style* when both are set.  Applied to every axis in the
        figure that already has a date-type formatter.
    datetime_style : {"auto", "concise", "short", "long"}, optional
        Named datetime formatting strategy applied when *datetime_format* is
        ``None``:

        * ``"auto"`` / ``"concise"`` — :class:`~matplotlib.dates.ConciseDateFormatter`
          with :class:`~matplotlib.dates.AutoDateLocator` (smart tick density).
        * ``"short"`` — ``DateFormatter("%b %d")`` with daily major ticks.
        * ``"long"`` — ``DateFormatter("%Y-%m-%d %H:%M")`` with daily major
          ticks and 6-hourly minor ticks.
    output_format : {"svg", "png", "pdf", "eps"}, optional
        File format hint passed to :func:`matplotlib.figure.Figure.savefig`
        when saving a figure returned by a plot function.  Defaults to
        ``"svg"`` for lossless, resolution-independent output.
    save_path : str or Path, optional
        If set, the figure is automatically saved to this path after all
        styling is applied.  The file format is inferred from the extension
        (e.g. ``".png"``, ``".svg"``).  Parent directories are created
        automatically.  When ``None`` (default) no file is written.
    depth_perspective : DepthPerspective | DepthMode, optional
        Override the depth coordinate convention for this call only.  A bare
        :class:`DepthMode` is automatically wrapped in a
        :class:`DepthPerspective` with default display options.  When
        ``None`` (default) the module-level default set by
        :func:`set_depth_perspective` is used (initially
        :attr:`DepthMode.FixedBottom`).  Affects which depth columns are
        read, the axis direction, and the depth axis label.

    Examples
    --------
    >>> from us_marine_energy_resource.viz import PlotSettings
    >>> s = PlotSettings(
    ...     start_date="2010-01",
    ...     end_date="2010-02",
    ...     fig_width=6,
    ...     fig_height=2,
    ...     title="Current Speed — Cook Inlet, Near Nikiski, AK",
    ...     caption="Latitude: 60.68, Longitude: -151.40",
    ...     datetime_style="concise",
    ... )
    >>> tidal.plot_sigma_layers_speed(df, settings=s)
    """

    start_date: str | pd.Timestamp | None = field(default=None)
    end_date: str | pd.Timestamp | None = field(default=None)
    fig_width: float | None = field(default=None)
    fig_height: float | None = field(default=None)
    title: str | None = field(default=None)
    subtitle: str | None = field(default=None)
    caption: str | None = field(default=None)
    xlabel: str | None = field(default=None)
    ylabel: str | None = field(default=None)
    colormap: str | Any | None = field(default=None)
    colorbar_min: float | None = field(default=None)
    colorbar_max: float | None = field(default=None)
    datetime_format: str | None = field(default=None)
    datetime_style: Literal["auto", "concise", "short", "long"] | None = field(default=None)
    output_format: OutputFormat = field(default="svg")
    save_path: str | Path | None = field(default=None)
    depth_perspective: DepthPerspective | DepthMode | None = field(default=None)
