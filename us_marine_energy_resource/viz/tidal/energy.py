"""Tidal energy resource analysis plots."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from us_marine_energy_resource.analysis.resource import compute_power_density
from us_marine_energy_resource.viz._style import styled
from us_marine_energy_resource.viz.settings import PlotSettings

from ._components import _N_LAYERS, _validate_columns


@styled
def analyze_power_density(
    df: pd.DataFrame,
    settings: PlotSettings | None = None,
    layer: int | None = None,
    rho: float = 1025.0,
    cut_in_speed: float = 0.5,
) -> tuple[Figure, dict[str, Any]]:
    """Analyse tidal current power density with a four-panel diagnostic figure.

    Panels:

    1. Current speed histogram with cut-in speed marker.
    2. Power density vs. current speed scatter with theoretical P = 0.5*rho*V^3 curve.
    3. Power density histogram.
    4. Power duration curve (W/m²).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ``vap_sea_water_speed_layer_{i}``,
        ``vap_sea_water_power_density_layer_{i}``, and
        ``vap_sigma_depth_layer_{i}`` columns for all 10 layers.
    settings : PlotSettings, optional
        See :class:`~us_marine_energy_resource.viz.settings.PlotSettings`.
    layer : int, optional
        Sigma layer to analyse.  If ``None``, the layer with the highest mean
        power density is used.
    rho : float, optional
        Water density in kg/m^3.  Default 1025.
    cut_in_speed : float, optional
        Cut-in speed for turbine operation (m/s).  Default 0.5.

    Returns
    -------
    fig : Figure
        The created matplotlib figure.
    summary : dict
        Keys: ``layer``, ``depth``, ``mean_speed``, ``p90_speed``,
        ``p95_speed``, ``max_speed``, ``mean_power_density``,
        ``max_power_density``, ``usable_time_pct``.

    Raises
    ------
    KeyError
        If required columns are absent from *df*.
    """
    speed_cols = [f"vap_sea_water_speed_layer_{i}" for i in range(_N_LAYERS)]
    power_cols = [f"vap_sea_water_power_density_layer_{i}" for i in range(_N_LAYERS)]
    depth_cols = [f"vap_sigma_depth_layer_{i}" for i in range(_N_LAYERS)]
    _validate_columns(df, speed_cols + power_cols + depth_cols)

    if layer is None:
        layer = int(np.argmax([df[c].mean() for c in power_cols]))

    speeds: np.ndarray = df[f"vap_sea_water_speed_layer_{layer}"].to_numpy(
        dtype=float, na_value=np.nan
    )
    power_densities: np.ndarray = compute_power_density(speeds, rho=rho)
    depth = float(df[f"vap_sigma_depth_layer_{layer}"].iloc[0])

    fig, axs = plt.subplots(2, 2, figsize=(14, 12))

    bins_speed = np.linspace(0, float(np.nanmax(speeds)) * 1.05, 30)
    axs[0, 0].hist(
        speeds, bins=list(bins_speed), alpha=0.7, color=sns.color_palette()[0], edgecolor="black"
    )
    axs[0, 0].axvline(
        x=cut_in_speed, color="r", linestyle="--", label=f"Cut-in: {cut_in_speed} m/s"
    )
    axs[0, 0].set_xlabel("Current Speed [m/s]")
    axs[0, 0].set_ylabel("Frequency")
    axs[0, 0].set_title(f"Speed Distribution at {depth:.1f} m (Layer {layer})")
    axs[0, 0].grid(True, linestyle="--", alpha=0.7)
    axs[0, 0].legend()

    v_range = np.linspace(0, float(np.nanmax(speeds)) * 1.1, 100)
    p_theoretical = compute_power_density(v_range, rho=rho)
    axs[0, 1].plot(v_range, p_theoretical, "k-", label="P = 0.5*rho*V^3")
    axs[0, 1].scatter(
        speeds, power_densities, alpha=0.3, color=sns.color_palette()[0], label="Observed", s=5
    )
    axs[0, 1].set_xlabel("Current Speed [m/s]")
    axs[0, 1].set_ylabel("Power Density [W/m\u00b2]")
    axs[0, 1].set_title("Power Density vs. Current Speed")
    axs[0, 1].grid(True, linestyle="--", alpha=0.7)
    axs[0, 1].legend()

    bins_pd = np.linspace(0, float(np.nanmax(power_densities)) * 1.05, 30)
    axs[1, 0].hist(
        power_densities,
        bins=list(bins_pd),
        alpha=0.7,
        color=sns.color_palette()[1],
        edgecolor="black",
    )
    axs[1, 0].set_xlabel("Power Density [W/m\u00b2]")
    axs[1, 0].set_ylabel("Frequency")
    axs[1, 0].set_title("Power Density Distribution")
    axs[1, 0].grid(True, linestyle="--", alpha=0.7)

    # Power duration curve: power density above cut-in, sorted descending.
    active = speeds >= cut_in_speed
    pd_active = np.where(active, compute_power_density(speeds, rho=rho), 0.0)
    sorted_pd = np.sort(pd_active)[::-1]
    exceedance = np.arange(1, len(sorted_pd) + 1) / len(sorted_pd) * 100

    axs[1, 1].plot(exceedance, sorted_pd, "-", color=sns.color_palette()[1])
    axs[1, 1].axvline(
        x=float(np.mean(~active) * 100),
        color="r",
        linestyle="--",
        alpha=0.7,
        label=f"Cut-in ({cut_in_speed} m/s)",
    )
    axs[1, 1].set_xlabel("Exceedance Probability [%]")
    axs[1, 1].set_ylabel("Power Density [W/m\u00b2]")
    axs[1, 1].set_title("Power Duration Curve")
    axs[1, 1].grid(True, linestyle="--", alpha=0.7)
    axs[1, 1].legend()

    summary: dict[str, Any] = {
        "layer": layer,
        "depth": depth,
        "mean_speed": float(np.nanmean(speeds)),
        "p90_speed": float(np.nanpercentile(speeds, 90)),
        "p95_speed": float(np.nanpercentile(speeds, 95)),
        "max_speed": float(np.nanmax(speeds)),
        "mean_power_density": float(np.nanmean(power_densities)),
        "max_power_density": float(np.nanmax(power_densities)),
        "usable_time_pct": float(100.0 * np.mean(speeds >= cut_in_speed)),
    }

    summary_text = (
        f"Layer {layer} at {depth:.1f} m\n"
        f"Mean Speed: {summary['mean_speed']:.2f} m/s  |  "
        f"P90: {summary['p90_speed']:.2f} m/s  |  "
        f"Max: {summary['max_speed']:.2f} m/s\n"
        f"Mean Power Density: {summary['mean_power_density']:.2f} W/m\u00b2  |  "
        f"Max: {summary['max_power_density']:.2f} W/m\u00b2\n"
        f"Time above cut-in: {summary['usable_time_pct']:.1f}%"
    )
    fig.text(
        0.5,
        0.01,
        summary_text,
        ha="center",
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.8, "boxstyle": "round"},
    )

    plt.tight_layout(rect=(0, 0.05, 1, 0.95))
    plt.suptitle("Tidal Energy Resource Assessment", fontsize=16, y=0.98)
    return fig, summary
