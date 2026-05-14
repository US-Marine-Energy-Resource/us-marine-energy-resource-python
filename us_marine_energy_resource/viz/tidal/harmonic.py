"""Tidal harmonic analysis, FFT, and phase analysis plots."""

import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from scipy.fft import fft, fftfreq  # type: ignore[import-untyped]

from us_marine_energy_resource.analysis.preprocessing import sigma_depth_scalar
from us_marine_energy_resource.viz._style import styled
from us_marine_energy_resource.viz.settings import PlotSettings, get_depth_perspective

from ._components import _validate_columns


def _utide_imports() -> tuple:
    # utide bundles a .npy constants file compiled with an older numpy that used
    # align=0 (int) instead of align=False (bool), triggering a NumPy 2.4 warning.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=Warning, module="numpy.*")
        from utide import reconstruct, solve  # type: ignore[import-untyped]
    return reconstruct, solve

# Known principal tidal constituents: name → period in hours
_PRINCIPAL_PERIODS: dict[str, float] = {
    "M2": 12.42,
    "S2": 12.00,
    "N2": 12.66,
    "K1": 23.93,
    "O1": 25.82,
    "M4": 6.21,
    "M6": 4.14,
}

# Descriptive labels for constituent bar chart
_CONSTITUENT_INFO: dict[str, str] = {
    "M2": "Principal lunar semidiurnal",
    "S2": "Principal solar semidiurnal",
    "N2": "Larger lunar elliptic semidiurnal",
    "K1": "Lunar diurnal",
    "O1": "Lunar diurnal",
    "M4": "Shallow water overtide of M2",
    "M6": "Shallow water overtide of M2",
    "K2": "Solar semidiurnal",
    "L2": "Smaller lunar elliptic semidiurnal",
    "P1": "Solar diurnal",
    "Q1": "Larger lunar elliptic diurnal",
    "MK3": "Shallow water terdiurnal",
    "MN4": "Shallow water quarter diurnal",
    "MS4": "Shallow water quarter diurnal",
}


@styled
def plot_tidal_harmonic_analysis(
    df: pd.DataFrame,
    settings: PlotSettings | None = None,
    layer: int = 4,
    n_components: int = 5,
) -> Figure:
    """Perform tidal harmonic analysis and produce a three-panel diagnostic figure.

    The three panels are:

    1. **Speed time series** — observed vs. ``utide``-reconstructed current speed
       for the chosen sigma layer, annotated with RMSE and R².
    2. **FFT spectrum** — power spectrum of the full speed record on a linear
       scale, with vertical reference lines for principal tidal constituents.
    3. **Constituent bar chart** — amplitude of the *n_components* strongest
       harmonic constituents identified by ``utide``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a ``DatetimeIndex`` and columns
        ``vap_sea_water_speed_layer_{layer}``,
        ``u_layer_{layer}``, ``v_layer_{layer}``,
        ``vap_sigma_depth_layer_{layer}``, and ``lat_center``.
    settings : PlotSettings, optional
        See :class:`~us_marine_energy_resource.viz.settings.PlotSettings`.
    layer : int, optional
        Zero-based sigma layer index used for the analysis. Default 4
        (mid-column).
    n_components : int, optional
        Number of harmonic constituents to display in the bar chart. Default 5.

    Returns
    -------
    fig : Figure
        The created matplotlib figure.

    Raises
    ------
    KeyError
        If any required column (speed, u/v components, depth, or latitude)
        is absent from *df*.
    """
    perspective = get_depth_perspective(settings)
    required = [
        f"vap_sea_water_speed_layer_{layer}",
        f"u_layer_{layer}",
        f"v_layer_{layer}",
        perspective.depth_col(layer),
        "lat_center",
    ]
    _validate_columns(df, required)

    speeds: np.ndarray = df[f"vap_sea_water_speed_layer_{layer}"].to_numpy(
        dtype=float, na_value=np.nan
    )
    u: np.ndarray = df[f"u_layer_{layer}"].to_numpy(dtype=float, na_value=np.nan)
    v: np.ndarray = df[f"v_layer_{layer}"].to_numpy(dtype=float, na_value=np.nan)
    t = df.index.to_numpy()
    lat = float(df["lat_center"].iloc[0])
    depth_value = sigma_depth_scalar(df, layer, perspective.mode)

    # Harmonic analysis via utide (lazy-loaded to defer the numpy warning)
    reconstruct, solve = _utide_imports()
    coef = solve(t, u, v, lat=lat, conf_int="linear")
    major_names: list[str] = list(coef["name"][:n_components])
    major_freqs: np.ndarray = coef["aux"]["frq"][:n_components]

    # Convert utide frequencies (rad s⁻¹) → period in hours
    major_periods: list[float] = [float(2 * np.pi / (freq * 3600)) for freq in major_freqs]
    major_amps: np.ndarray = np.sqrt(
        coef["Lsmaj"][:n_components] ** 2 + coef["Lsmin"][:n_components] ** 2
    )

    # Reconstruct tidal signal (capped at 1 000 time steps for speed)
    plot_limit = min(1000, len(df))
    t_plot = df.index[:plot_limit]
    recon = reconstruct(t_plot, coef)

    if hasattr(recon, "h"):
        speed_recon: np.ndarray = recon.h
    else:
        speed_recon = np.sqrt(recon.u**2 + recon.v**2)

    # --- Build figure ---
    fig = plt.figure(figsize=(12, 15))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.3)
    ax_speed = fig.add_subplot(gs[0])
    ax_fft = fig.add_subplot(gs[1])
    ax_harm = fig.add_subplot(gs[2])

    # Panel 1 — speed time series
    ax_speed.plot(
        t_plot, speeds[:plot_limit], "-", color=sns.color_palette()[0], alpha=0.7, label="Observed"
    )
    ax_speed.plot(t_plot, speed_recon, "-", color=sns.color_palette()[1], label="Reconstructed")
    ax_speed.set_ylabel("Current Speed [m/s]")
    ax_speed.set_title("Tidal Current Speed")
    ax_speed.legend()
    ax_speed.grid(True)
    ax_speed.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    rmse = float(np.sqrt(np.mean((speeds[:plot_limit] - speed_recon) ** 2)))
    ss_res = float(np.sum((speeds[:plot_limit] - speed_recon) ** 2))
    ss_tot = float(np.sum((speeds[:plot_limit] - np.mean(speeds[:plot_limit])) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    ax_speed.text(
        0.02,
        0.95,
        f"RMSE: {rmse:.3f} m/s\nR²: {r2:.3f}",
        transform=ax_speed.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7},
    )

    # Panel 2 — FFT spectrum
    if len(t) > 1:
        time_diffs = np.diff(t.astype("datetime64[s]").astype(np.int64))
        dt = float(np.mean(time_diffs)) / 86400.0  # seconds → days
    else:
        dt = 1.0 / 48.0  # fallback: 30-minute data

    n_fft = len(speeds)
    yf = fft(speeds)
    xf: np.ndarray = fftfreq(n_fft, dt)[: n_fft // 2]  # cycles per day

    # Guard against zero frequency before converting to periods
    with np.errstate(divide="ignore", invalid="ignore"):
        periods: np.ndarray = np.where(xf > 0, 24.0 / xf, np.inf)

    power: np.ndarray = 2.0 / n_fft * np.abs(yf[: n_fft // 2])
    period_mask = (periods >= 1) & (periods <= 30)

    ax_fft.plot(
        periods[period_mask],
        power[period_mask],
        "-",
        linewidth=1.5,
        color=sns.color_palette()[0],
    )
    ax_fft.set_xlabel("Period [h]")
    ax_fft.set_ylabel("Amplitude")
    ax_fft.set_title("FFT Frequency Spectrum (1-30 h)")
    ax_fft.grid(True, linestyle="--", alpha=0.7)

    ref_colors = plt.cm.tab10(np.linspace(0, 1, len(_PRINCIPAL_PERIODS)))  # type: ignore[attr-defined]
    for j, (name, period) in enumerate(_PRINCIPAL_PERIODS.items()):
        if 1 <= period <= 30:
            ax_fft.axvline(
                x=period,
                color=ref_colors[j],
                linestyle="--",
                alpha=0.7,
                label=f"{name} ({period:.2f} h)",
            )
    ax_fft.legend(loc="upper right", fontsize=8)

    # Panel 3 — constituent bar chart
    bar_colors = plt.cm.viridis(np.linspace(0, 0.8, len(major_names)))  # type: ignore[attr-defined]
    ax_harm.bar(range(len(major_names)), major_amps, alpha=0.8, color=bar_colors)
    ax_harm.set_xticks(range(len(major_names)))
    ax_harm.set_xticklabels(major_names)

    for k, name in enumerate(major_names):
        description = _CONSTITUENT_INFO.get(name, name)
        ax_harm.text(
            k,
            -0.08,
            description,
            ha="center",
            va="top",
            transform=ax_harm.get_xaxis_transform(),
            fontsize=9,
        )
        ax_harm.text(
            k,
            float(major_amps[k]) + 0.02,
            f"{major_periods[k]:.2f} h",
            ha="center",
            va="bottom",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7},
        )

    ax_harm.set_ylabel("Amplitude [m/s]")
    ax_harm.set_title("Major Tidal Constituents")
    ax_harm.text(
        0.5,
        -0.22,
        "The strongest constituents indicate primary tidal forces at this location.",
        ha="center",
        va="center",
        transform=ax_harm.transAxes,
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "#f0f0f0", "alpha": 0.7},
    )
    ax_harm.grid(True, axis="y")

    plt.suptitle(
        f"Tidal Harmonic Analysis at {depth_value:.1f} m Depth",
        fontsize=14,
        y=0.98,
    )
    plt.tight_layout(rect=(0, 0.05, 1, 0.95))

    return fig


@styled
def plot_fft(
    df: pd.DataFrame,
    settings: PlotSettings | None = None,
    sample_rate: float | None = None,
) -> Figure:
    """Generate a per-layer FFT amplitude spectrum grid (2 rows x 5 columns).

    Each sub-plot shows the amplitude spectrum for one sigma layer on a
    log-period x-axis (4-30 h), with reference lines for M2, K1, M4, S2,
    and O1 constituents.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ``vap_sea_water_speed_layer_{i}`` and
        ``vap_sigma_depth_layer_{i}`` columns for layers 0-9.
    settings : PlotSettings, optional
        See :class:`~us_marine_energy_resource.viz.settings.PlotSettings`.
    sample_rate : float, optional
        Sampling rate in samples per hour.  If ``None``, derived from the
        DataFrame's ``DatetimeIndex``.

    Returns
    -------
    fig : Figure
        The created matplotlib figure.

    Raises
    ------
    KeyError
        If required columns are absent from *df*.
    """
    from ._components import _N_LAYERS, _validate_columns

    perspective = get_depth_perspective(settings)
    _validate_columns(
        df,
        [f"vap_sea_water_speed_layer_{i}" for i in range(_N_LAYERS)]
        + [perspective.depth_col(i) for i in range(_N_LAYERS)],
    )

    colors = sns.color_palette()
    timestamps = df.index
    if sample_rate is None and len(timestamps) > 1:
        dt_h = float((timestamps[1] - timestamps[0]).total_seconds()) / 3600.0
        sample_rate = 1.0 / dt_h
    elif sample_rate is None:
        sample_rate = 1.0

    all_vel: np.ndarray = np.column_stack(
        [df[f"vap_sea_water_speed_layer_{i}"].to_numpy(dtype=float) for i in range(_N_LAYERS)]
    )
    all_dep: np.ndarray = np.column_stack(
        [df[perspective.depth_col(i)].to_numpy(dtype=float) for i in range(_N_LAYERS)]
    )
    min_depths = np.min(all_dep, axis=0)
    max_depths = np.max(all_dep, axis=0)

    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    axes_flat = axes.flatten()

    sig_periods = [12.42, 24.0, 6.21, 12.0, 25.82]
    sig_names = ["M2", "K1", "M4", "S2", "O1"]

    for i in range(_N_LAYERS):
        ax = axes_flat[i]
        vel = all_vel[:, i] - float(np.mean(all_vel[:, i]))
        window = np.hanning(len(vel))
        fft_result = np.fft.rfft(vel * window)
        fft_freq = np.fft.rfftfreq(len(vel), d=1.0 / sample_rate)

        periods = 1.0 / fft_freq[1:]
        amplitudes = np.abs(fft_result)[1:]

        ax.plot(periods, amplitudes, "-", color=colors[0], linewidth=1.5)

        for period, name in zip(sig_periods, sig_names, strict=False):
            closest = int(np.argmin(np.abs(periods - period)))
            if closest < len(amplitudes):
                ax.axvline(x=periods[closest], color="red", alpha=0.3, linestyle="--")
                if amplitudes[closest] > 0.1 * float(np.max(amplitudes)):
                    ax.text(
                        periods[closest],
                        amplitudes[closest] * 1.1,
                        name,
                        ha="center",
                        fontsize=8,
                        bbox={"facecolor": "white", "alpha": 0.8},
                    )

        ax.text(
            0.02,
            0.95,
            f"Depth: {min_depths[i]:.1f}-{max_depths[i]:.1f} m",
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.7},
        )
        ax.set_xscale("log")
        ax.set_xlim(4, 30)
        ax.grid(True, linestyle="--", alpha=0.6)
        if i >= 5:
            ax.set_xlabel("Period [h]")
        if i % 5 == 0:
            ax.set_ylabel("Amplitude")

    fig.suptitle("FFT Analysis of Current Speed by Depth Layer", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


@styled
def plot_tidal_phase_analysis(
    df: pd.DataFrame,
    settings: PlotSettings | None = None,
    layer: int = 4,
) -> Figure:
    """Analyze tidal phases and the current speed-vs-water-level relationship.

    Three main panels (stacked, shared x-axis):

    1. Water level time series with high/low tide markers.
    2. Current speed with high/low tide reference lines.
    3. Speed vs. water level scatter with quadratic best-fit.

    Plus a fourth panel (phase lag histogram) placed alongside panel 3.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ``vap_sea_water_speed_layer_{layer}``,
        ``vap_surface_elevation``, and ``vap_sigma_depth_layer_{layer}``
        columns.
    settings : PlotSettings, optional
        See :class:`~us_marine_energy_resource.viz.settings.PlotSettings`.
    layer : int, optional
        Sigma layer index to analyze. Default 4.

    Returns
    -------
    fig : Figure
        The created matplotlib figure.

    Raises
    ------
    KeyError
        If required columns are absent from *df*.
    """
    from scipy.optimize import curve_fit as _curve_fit
    from scipy.signal import find_peaks as _find_peaks

    from ._components import _validate_columns

    perspective = get_depth_perspective(settings)
    _validate_columns(
        df,
        [
            f"vap_sea_water_speed_layer_{layer}",
            "vap_surface_elevation",
            perspective.depth_col(layer),
        ],
    )

    speeds: np.ndarray = df[f"vap_sea_water_speed_layer_{layer}"].to_numpy(dtype=float)
    water_level: np.ndarray = df["vap_surface_elevation"].to_numpy(dtype=float)
    timestamps = df.index
    depth_value = sigma_depth_scalar(df, layer, perspective.mode)

    high_idx, _ = _find_peaks(water_level, distance=5)
    low_idx, _ = _find_peaks(-water_level, distance=5)

    tidal_ranges: list[float] = []
    for k in range(len(high_idx) - 1):
        between = [j for j in low_idx if high_idx[k] < j < high_idx[k + 1]]
        if between:
            tidal_ranges.append(float(water_level[high_idx[k]] - water_level[between[0]]))
    mean_range = float(np.mean(tidal_ranges)) if tidal_ranges else 0.0

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 15), sharex=True)

    _pal = sns.color_palette()
    ax1.plot(timestamps, water_level, "-", color=_pal[0], linewidth=1.5)
    ax1.plot(timestamps[high_idx], water_level[high_idx], "o", color=_pal[1], label="High Tide")
    ax1.plot(timestamps[low_idx], water_level[low_idx], "o", color=_pal[2], label="Low Tide")
    ax1.set_ylabel("Water Level [m]")
    ax1.set_title("Tidal Elevation")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(timestamps, speeds, "-", color=_pal[0], linewidth=1.5)
    for idx in high_idx:
        ax2.axvline(x=timestamps[int(idx)], color=_pal[1], linestyle="--", alpha=0.3)
    for idx in low_idx:
        ax2.axvline(x=timestamps[int(idx)], color=_pal[2], linestyle="--", alpha=0.3)
    ax2.set_ylabel("Current Speed [m/s]")
    ax2.set_title("Tidal Current Speed")
    ax2.grid(True)

    ax3.scatter(water_level, speeds, s=10, alpha=0.5)
    ax3.set_xlabel("Water Level [m]")
    ax3.set_ylabel("Current Speed [m/s]")
    ax3.set_title("Speed vs. Water Level (Phase Relationship)")
    ax3.grid(True)

    def _quad(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        return a * x**2 + b * x + c

    try:
        popt, _ = _curve_fit(_quad, water_level, speeds)
        a, b, c = popt
        x_fit = np.linspace(float(water_level.min()), float(water_level.max()), 100)
        ax3.plot(
            x_fit,
            _quad(x_fit, a, b, c),
            "-",
            color=_pal[1],
            linewidth=2,
            label=f"Fit: {a:.4f}x^2 + {b:.4f}x + {c:.4f}",
        )
        ax3.legend()
    except Exception:
        pass

    # Phase lag panel alongside ax3
    gs2 = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])
    ax3.set_position(gs2[2, 0].get_position(fig))
    ax4 = fig.add_subplot(gs2[2, 1])

    phase_lags: list[float] = []
    for idx in high_idx:
        ht = timestamps[int(idx)]
        window = pd.Timedelta(hours=3)
        mask = (timestamps >= ht - window) & (timestamps <= ht + window)
        ws = speeds[mask]
        wt = timestamps[mask]
        if len(ws) > 0:
            mi = int(np.argmax(ws))
            lag = float((wt[mi] - ht).total_seconds()) / 3600.0
            phase_lags.append(lag)

    mean_lag = 0.0
    if phase_lags:
        ax4.hist(phase_lags, bins=15, alpha=0.7)
        mean_lag = float(np.mean(phase_lags))
        ax4.axvline(x=mean_lag, color=_pal[0], linestyle="-", label=f"Mean: {mean_lag:.2f} h")
        ax4.set_xlabel("Phase Lag [h]")
        ax4.set_ylabel("Frequency")
        ax4.set_title("Phase Lag Distribution")
        ax4.grid(True)
        ax4.legend()

    plt.tight_layout()
    plt.suptitle(f"Tidal Phase Analysis at {depth_value:.1f} m Depth", fontsize=16, y=1.02)

    if phase_lags:
        fig.text(
            0.5,
            0.01,
            f"Mean Tidal Range: {mean_range:.2f} m  |  "
            f"Mean Phase Lag: {mean_lag:.2f} h  |  "
            f"{'Flood Dominant' if mean_lag < 0 else 'Ebb Dominant'}",
            ha="center",
            fontsize=12,
            bbox={"facecolor": "white", "alpha": 0.8},
        )

    return fig
