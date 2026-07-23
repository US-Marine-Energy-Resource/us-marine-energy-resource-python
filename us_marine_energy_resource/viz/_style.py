"""Shared visualization theme for tidal energy resource plots."""

import contextlib
import functools
from collections.abc import Callable
from typing import Any, TypeVar

import cmocean
import matplotlib
import matplotlib.dates as mdates
import matplotlib.figure
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import colormaps

_F = TypeVar("_F", bound=Callable[..., Any])

_CMOCEAN_THERMAL_NAME = "cmocean_thermal"

PLOT_CONFIG: dict[str, float | None] = {
    "fig_width": 6.0,
    "max_fig_height": 3.0,
}

_SOURCE_CAPTION = "Source: U.S. DOE H2O High Resolution Tidal Hindcast, Yang et al., 2025"

# Reference height (inches) used by figure_fontsize when no override is given.
# Matches PLOT_CONFIG["max_fig_height"] so that fonts at the default figure
# size come out exactly as specified.
_REF_HEIGHT: float = 3.0

# Formatter types that indicate a datetime x-axis.
_DATE_FORMATTER_TYPES = (
    mdates.DateFormatter,
    mdates.ConciseDateFormatter,
    mdates.AutoDateFormatter,
)


def _apply_theme() -> None:
    """Apply the standard tidal resource visualization theme.

    Sets the seaborn base theme, matplotlib font and weight rcParams, and
    registers (if not already registered) the ``cmocean_thermal`` colormap as
    the default.
    """
    sns.set_theme()
    plt.rcParams["font.family"] = "Public Sans, Arial, sans-serif"
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    if _CMOCEAN_THERMAL_NAME not in colormaps:
        colormaps.register(name=_CMOCEAN_THERMAL_NAME, cmap=cmocean.cm.thermal)  # type: ignore[attr-defined]

    plt.set_cmap(_CMOCEAN_THERMAL_NAME)


def _apply_settings_hook(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, ...]:
    """Apply ``PlotSettings`` trimming to the first DataFrame argument.

    Inspects the ``settings`` keyword argument.  When present and non-None,
    applies :func:`~us_marine_energy_resource.viz.tidal._components._trim_time`
    to the first positional argument if it is a :class:`pandas.DataFrame`, or
    to each element if it is a list of DataFrames.

    Parameters
    ----------
    args : tuple
        Positional arguments passed to the decorated function.
    kwargs : dict
        Keyword arguments passed to the decorated function.

    Returns
    -------
    tuple
        Updated positional arguments with the first DataFrame (or each
        DataFrame in the first list) replaced by its trimmed version.
    """
    settings = kwargs.get("settings")
    if settings is None or not args:
        return args

    # Defer import to avoid circular dependency at module level.
    from us_marine_energy_resource.viz.tidal._components import _trim_time

    first = args[0]
    if isinstance(first, pd.DataFrame):
        return (_trim_time(first, settings), *args[1:])
    if isinstance(first, list) and first:
        head = first[0]
        # list[pd.DataFrame]
        if isinstance(head, pd.DataFrame):
            return ([_trim_time(df, settings) for df in first], *args[1:])
        # list[tuple[str, pd.DataFrame, ...]] — site_records pattern
        if isinstance(head, tuple) and len(head) >= 2 and isinstance(head[1], pd.DataFrame):
            return (
                [(t[0], _trim_time(t[1], settings), *t[2:]) for t in first],
                *args[1:],
            )

    return args


def _apply_figure_sizing(result: Any, settings: Any) -> None:
    """Resize the figure returned by a plot function using :data:`PLOT_CONFIG`.

    Called by :func:`styled` after each decorated function.  The target width
    and height are resolved in this order (highest priority first):

    1. Per-call ``settings.fig_width`` / ``settings.fig_height`` override.
    2. Global :data:`PLOT_CONFIG` ``fig_width`` (exact) and
       ``max_fig_height`` (cap — only shrinks, never grows).
    3. The figure's current size (no change).

    Parameters
    ----------
    result : Any
        Return value of the decorated function.  May be a
        :class:`~matplotlib.figure.Figure`, a tuple whose first element is a
        Figure, or any other value (in which case nothing is done).
    settings : Any
        The ``settings`` keyword argument passed to the decorated function, or
        ``None``.
    """
    from matplotlib.figure import Figure

    fig: Figure | None = None
    if isinstance(result, Figure):
        fig = result
    elif isinstance(result, tuple) and result and isinstance(result[0], Figure):
        fig = result[0]

    if fig is None:
        return

    current_w, current_h = fig.get_size_inches()

    # Per-call settings override global config.
    target_w: float | None = getattr(settings, "fig_width", None)
    target_h: float | None = getattr(settings, "fig_height", None)

    if target_w is None:
        target_w = PLOT_CONFIG.get("fig_width")
    if target_h is None:
        max_h = PLOT_CONFIG.get("max_fig_height")
        if max_h is not None:
            target_h = min(current_h, max_h)

    new_w = target_w if target_w is not None else current_w
    new_h = target_h if target_h is not None else current_h
    if new_w != current_w or new_h != current_h:
        fig.set_size_inches(new_w, new_h)


def _resolve_cmap(settings: Any, default: Any) -> Any:
    """Return the colormap from *settings* or fall back to *default*.

    Parameters
    ----------
    settings : PlotSettings or None
        The settings object passed to a plot function.
    default : colormap
        Fallback colormap (name string or matplotlib Colormap object) used
        when ``settings`` is ``None`` or ``settings.colormap`` is ``None``.

    Returns
    -------
    colormap
        ``settings.colormap`` when set, otherwise *default*.
    """
    if settings is not None and getattr(settings, "colormap", None) is not None:
        return settings.colormap
    return default


def _apply_figure_annotations(result: Any, settings: Any) -> None:
    """Apply title, subtitle, caption, and axis label overrides to a figure.

    Called by :func:`styled` after each decorated function.  All fields are
    optional — only those that are non-``None`` in *settings* are applied.

    * **title** — calls ``axes[0].set_title(title)`` so the title sits tightly
      above the main axes content regardless of whether a colorbar or other
      secondary axes is present.
    * **subtitle** — rendered as italic ``fig.text`` near the top center.
    * **caption** — appended after the source attribution line which appears
      on every figure.
    * **xlabel** / **ylabel** — applied only when the figure has exactly one
      axis (ignored for multi-panel figures).

    Parameters
    ----------
    result : Any
        Return value of the decorated function.
    settings : Any
        The ``settings`` keyword argument, or ``None``.
    """
    from matplotlib.figure import Figure

    if settings is None:
        return

    fig: Figure | None = None
    if isinstance(result, Figure):
        fig = result
    elif isinstance(result, tuple) and result and isinstance(result[0], Figure):
        fig = result[0]

    if fig is None:
        return

    axes = fig.axes
    # Subplot axes have a SubplotSpec; inset and colorbar axes do not.
    # Use this to distinguish true multi-panel grids from single plots that
    # happen to have a colorbar (which adds a second axes).
    content_axes = [ax for ax in axes if ax.get_subplotspec() is not None]
    is_multi_panel = len(content_axes) > 1

    title: str | None = getattr(settings, "title", None)
    subtitle: str | None = getattr(settings, "subtitle", None)
    caption: str | None = getattr(settings, "caption", None)
    xlabel: str | None = getattr(settings, "xlabel", None)
    ylabel: str | None = getattr(settings, "ylabel", None)

    # Always show source attribution; append user caption when provided.
    display_caption = f"{_SOURCE_CAPTION}  |  {caption}" if caption else _SOURCE_CAPTION

    # For multi-panel figures use suptitle so the figure-level title doesn't
    # overwrite the first subplot's own title.  For single-panel (with or
    # without a colorbar) axes[0].set_title() keeps the title tightly above
    # the plot content.
    if title is not None:
        if is_multi_panel:
            fig.suptitle(title, fontsize=11, fontweight="bold")
        else:
            axes[0].set_title(title)

    if subtitle is not None:
        fig.text(0.5, 0.98, subtitle, ha="center", va="top", fontsize=9, style="italic")

    fig.text(0.01, 0.01, display_caption, ha="left", va="bottom", fontsize=8, color="0.55")

    is_single = not is_multi_panel
    if is_single:
        if xlabel is not None:
            axes[0].set_xlabel(xlabel)
        if ylabel is not None:
            axes[0].set_ylabel(ylabel)

    fig_h = fig.get_figheight()

    def _margin(fontsize_pt: float, pad_in: float = 0.04) -> float:
        """Convert font size + padding to a figure-height fraction."""
        return (fontsize_pt / 72.0 + pad_in) / fig_h

    # Re-run tight_layout after all annotations so spacing is correct for
    # the final figure size.
    with contextlib.suppress(Exception):
        fig.tight_layout()

    # Push axes down just enough for the caption (always present).
    with contextlib.suppress(Exception):
        fig.subplots_adjust(bottom=fig.subplotpars.bottom + _margin(8.0))


def _apply_datetime_formatting(result: Any, settings: Any) -> None:
    """Re-apply datetime tick formatting on all date axes in the figure.

    Detects axes that already have a date-type major formatter and replaces it
    according to *settings*.  A ``datetime_format`` string takes priority over
    ``datetime_style``; if neither is set this function is a no-op.

    Strategies for ``datetime_style``:

    * ``"auto"`` / ``"concise"`` — :class:`~matplotlib.dates.ConciseDateFormatter`
      with :class:`~matplotlib.dates.AutoDateLocator`.
    * ``"short"`` — ``DateFormatter("%b %d")``.
    * ``"long"`` — ``DateFormatter("%Y-%m-%d %H:%M")`` with daily major ticks
      and 6-hourly minor ticks.

    Parameters
    ----------
    result : Any
        Return value of the decorated function.
    settings : Any
        The ``settings`` keyword argument, or ``None``.
    """
    from matplotlib.figure import Figure

    if settings is None:
        return

    fmt_str: str | None = getattr(settings, "datetime_format", None)
    style: str | None = getattr(settings, "datetime_style", None)

    if fmt_str is None and style is None:
        return

    fig: Figure | None = None
    if isinstance(result, Figure):
        fig = result
    elif isinstance(result, tuple) and result and isinstance(result[0], Figure):
        fig = result[0]

    if fig is None:
        return

    for ax in fig.axes:
        if not isinstance(ax.xaxis.get_major_formatter(), _DATE_FORMATTER_TYPES):
            continue

        if fmt_str is not None:
            ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt_str))
        elif style in ("auto", "concise"):
            locator = mdates.AutoDateLocator()
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        elif style == "short":
            ax.xaxis.set_major_locator(mdates.DayLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        elif style == "long":
            ax.xaxis.set_major_locator(mdates.DayLocator())
            ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))


def figure_fontsize(
    pt: float,
    fig_or_height: "matplotlib.figure.Figure | float",
    ref_height: float | None = None,
) -> float:
    """Scale *pt* proportionally to a figure's height.

    Use this instead of hard-coded ``fontsize=N`` throughout the viz layer so
    that text stays legible at any output size — whether the user requests a
    thumbnail or a print-resolution figure.

    Inside a plot function (before ``styled`` resizes the figure), pass the
    *intended* final height via :func:`target_fig_height` rather than the
    current ``fig.get_figheight()``:

    .. code-block:: python

        fh = target_fig_height(settings)
        ax.set_xlabel("Speed [m/s]", fontsize=figure_fontsize(12, fh))

    Inside a post-processing hook (after ``styled`` resizes the figure), pass
    the ``Figure`` object directly:

    .. code-block:: python

        fontsize = figure_fontsize(10, fig)

    Parameters
    ----------
    pt : float
        Base font size in points, calibrated for the reference height.
    fig_or_height : Figure or float
        Either a :class:`~matplotlib.figure.Figure` (its current height is
        used) or an explicit height in inches.
    ref_height : float, optional
        Reference height in inches at which *pt* is the exact desired size.
        Defaults to :data:`_REF_HEIGHT` (``PLOT_CONFIG["max_fig_height"]``).

    Returns
    -------
    float
        Scaled font size in points, floored at ``6.0``.
    """
    from matplotlib.figure import Figure

    h: float = (
        fig_or_height.get_figheight()  # type: ignore[union-attr]
        if isinstance(fig_or_height, Figure)
        else float(fig_or_height)
    )
    base = ref_height if ref_height is not None else _REF_HEIGHT
    return max(6.0, pt * (h / base))


def target_fig_height(settings: Any = None) -> float:
    """Return the expected final figure height for font-size calculations.

    Resolves in priority order:

    1. ``settings.fig_height`` (explicit per-call override).
    2. :data:`PLOT_CONFIG` ``max_fig_height`` (global default).
    3. ``3.0`` (hard fallback).

    Call this at the *top* of a plot function and pass the result to
    :func:`figure_fontsize` so that text is sized for the figure's final
    dimensions rather than the temporary creation size.

    Parameters
    ----------
    settings : PlotSettings or None
        The ``settings`` keyword argument, or ``None``.

    Returns
    -------
    float
        Expected figure height in inches.
    """
    h: float | None = getattr(settings, "fig_height", None)
    if h is not None:
        return float(h)
    return float(PLOT_CONFIG.get("max_fig_height") or _REF_HEIGHT)


def _apply_legend_styling(result: Any) -> None:
    """Scale every legend's font size to the final figure dimensions.

    Called by :func:`styled` **after** :func:`_apply_figure_sizing` so the
    font size is computed against the true rendered height.

    For each legend the font size is the lesser of:

    * :func:`figure_fontsize` ``(10, fig)`` — a 10 pt base scaled to height.
    * ``(fig_height * 72 / n_entries) * 0.55`` — keeps all entries legible
      within the figure bounds.
    * ``9.0`` pt — hard upper cap so text stays readable on large figures.

    The result is floored at ``6.0`` pt.

    Parameters
    ----------
    result : Any
        Return value of the decorated function.  May be a
        :class:`~matplotlib.figure.Figure`, a tuple whose first element is a
        Figure, or any other value (in which case this is a no-op).
    """
    from matplotlib.figure import Figure

    fig: Figure | None = None
    if isinstance(result, Figure):
        fig = result
    elif isinstance(result, tuple) and result and isinstance(result[0], Figure):
        fig = result[0]

    if fig is None:
        return

    fig_h = fig.get_figheight()

    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is None:
            continue
        n = len(legend.get_texts())
        if n == 0:
            continue
        base = figure_fontsize(10.0, fig_h)
        entry_cap = (fig_h * 72.0 / n) * 0.55
        fontsize = max(6.0, min(base, entry_cap, 9.0))
        for text in legend.get_texts():
            text.set_fontsize(fontsize)


def _save_figure(result: Any, settings: Any) -> None:
    """Save the figure to disk if ``settings.save_path`` is set.

    Called by :func:`styled` after all styling hooks have been applied, so the
    saved file reflects the final annotated, sized figure.  Parent directories
    are created automatically.  File format is inferred from the extension of
    *save_path*.

    Parameters
    ----------
    result : Any
        Return value of the decorated function.
    settings : Any
        The ``settings`` keyword argument, or ``None``.
    """
    from pathlib import Path

    from matplotlib.figure import Figure

    save_path = getattr(settings, "save_path", None)
    if save_path is None:
        return

    fig: Figure | None = None
    if isinstance(result, Figure):
        fig = result
    elif isinstance(result, tuple) and result and isinstance(result[0], Figure):
        fig = result[0]

    if fig is None:
        return

    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # A figure may opt out of tight cropping (e.g. a hand-laid-out composite with
    # a cartopy GeoAxes, whose NaN tight bbox would otherwise crop the canvas).
    bbox = getattr(fig, "_mer_save_bbox_inches", "tight")
    fig.savefig(p, dpi=150, bbox_inches=bbox)
    plt.close(fig)


def styled(func: _F) -> _F:
    """Decorate a plot function to apply the standard tidal resource theme.

    The theme is applied fresh before each call so that the visual style is
    consistent regardless of any global state changes made by other libraries
    between calls.

    If the caller passes a ``settings`` keyword argument (a
    :class:`~us_marine_energy_resource.viz.settings.PlotSettings` instance),
    the following are applied automatically:

    * **Time trimming** — the first ``pd.DataFrame`` positional argument is
      sliced to ``[start_date, end_date]`` before the function executes.
    * **Figure sizing** — ``fig_width`` / ``fig_height`` override
      :data:`PLOT_CONFIG` after the function returns.
    * **Annotations** — ``title``, ``subtitle``, ``caption``, ``xlabel``,
      and ``ylabel`` are applied post-call via
      :func:`_apply_figure_annotations`.
    * **Datetime formatting** — ``datetime_format`` / ``datetime_style`` are
      applied to all date axes post-call via
      :func:`_apply_datetime_formatting`.

    Parameters
    ----------
    func : callable
        A plot function to wrap. Must return a ``matplotlib.figure.Figure``
        or a tuple whose first element is a Figure.

    Returns
    -------
    callable
        The wrapped function with theme application, settings hooks, and
        figure sizing applied.

    Examples
    --------
    >>> @styled
    ... def plot_something(df, settings=None):
    ...     fig, ax = plt.subplots()
    ...     ax.plot(df.index, df["speed"])
    ...     return fig
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        _apply_theme()
        args = _apply_settings_hook(args, kwargs)
        result = func(*args, **kwargs)
        settings = kwargs.get("settings")
        _apply_figure_sizing(result, settings)
        _apply_legend_styling(result)
        _apply_figure_annotations(result, settings)
        _apply_datetime_formatting(result, settings)
        _save_figure(result, settings)
        return result

    return wrapper  # type: ignore[return-value]
