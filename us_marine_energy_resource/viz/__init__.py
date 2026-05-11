"""
US Marine Energy Resource — viz subpackage.

Visualization functions for tidal energy resource analysis.

Every public plot function is decorated with ``@styled`` from ``viz._style``,
which applies the standard tidal resource theme (seaborn, rcParams, cmocean)
before each call.

Quick start::

    from us_marine_energy_resource.viz.velocity import plot_velocity_exceedance
    from us_marine_energy_resource.viz.tidal import plot_tidal_time_series

    fig = plot_velocity_exceedance(df)
    fig.savefig("velocity_exceedance.png", dpi=300, bbox_inches="tight")
"""

from .settings import OutputFormat, PlotSettings

__all__ = ["OutputFormat", "PlotSettings"]
