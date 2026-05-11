"""Shared settings dataclass for tidal visualization functions."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

OutputFormat = Literal["svg", "png", "pdf", "eps"]


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
