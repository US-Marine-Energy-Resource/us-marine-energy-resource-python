"""
High-level entrypoint for the H20 tidal hindcast dataset.

Provides a single-namespace API for data access and visualization:

    from us_marine_energy_resource import tidal_hindcast as tidal

    df = tidal.get_data_at_point(lat=60.73, lon=-151.43)
    fig = tidal.plot_tidal_time_series(df)
    fig, stats = tidal.plot_velocity_exceedance(df)
    fig = tidal.generate_tidal_joint_probability(df, sigma_layer=4)

The S3 connection and manifest are initialized lazily on the first call to
:func:`get_data_at_point` and reused for subsequent calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Lazy-init singleton
# ---------------------------------------------------------------------------


@dataclass
class _State:
    """Holds the shared S3 cache and manifest query objects."""

    cache: Any  # S3CacheManager
    query: Any  # TidalManifestQuery


_state: _State | None = None


def _ensure_initialized(cache_dir: Path | None = None, verbose: bool = False) -> None:
    """Initialize S3 cache and manifest on first use.

    Parameters
    ----------
    cache_dir : Path, optional
        Local directory for cached S3 files. Defaults to ``./us_tidal_cache``.
    verbose : bool, optional
        If True, print manifest loading details. Defaults to False.
    """
    global _state
    if _state is not None:
        return

    # Defer heavy imports to keep module-level import fast.
    from .cache import S3CacheManager
    from .manifest import TidalManifestQuery, find_latest_manifest_s3

    cache = S3CacheManager(
        bucket="marine-energy-data",
        prefix="us-tidal",
        cache_dir=cache_dir,
    )

    result = find_latest_manifest_s3(cache)
    if result is None:
        raise RuntimeError(
            "Could not locate a tidal hindcast manifest on S3. "
            "Check your network connection and AWS credentials."
        )

    manifest_path, _ = result
    query = TidalManifestQuery(manifest_path, s3_cache=cache, verbose=verbose)
    _state = _State(cache=cache, query=query)


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------


def get_data_at_point(
    lat: float,
    lon: float,
    max_km: float | None = None,
    cache_dir: Path | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Fetch tidal hindcast data for the grid point nearest to a coordinate.

    Downloads and caches the parquet file from S3 on the first call; subsequent
    calls for the same point return the cached file immediately.

    Parameters
    ----------
    lat : float
        Latitude in decimal degrees (WGS84).
    lon : float
        Longitude in decimal degrees (WGS84).
    max_km : float, optional
        Maximum allowed distance (km) between the query coordinate and the
        nearest grid point. Raises :exc:`ValueError` if exceeded.
    cache_dir : Path, optional
        Override the local cache directory (only used on the very first call
        before the connection is initialized).
    verbose : bool, optional
        If True, print manifest loading and cache details. Defaults to False.

    Returns
    -------
    pd.DataFrame
        Preprocessed tidal hindcast DataFrame with a ``DatetimeIndex`` and
        all sigma-layer speed, direction, power-density, and depth columns.

    Raises
    ------
    ValueError
        If ``max_km`` is set and the nearest grid point is farther than that
        distance.
    RuntimeError
        If no manifest is found on S3, or no grid point is near the coordinate.

    Examples
    --------
    >>> import us_marine_energy_resource.tidal_hindcast as tidal
    >>> df = tidal.get_data_at_point(lat=60.73, lon=-151.43)
    >>> df = tidal.get_data_at_point(lat=47.27, lon=-122.55, max_km=20.0)
    >>> df = tidal.get_data_at_point(lat=60.73, lon=-151.43, verbose=True)
    """
    _ensure_initialized(cache_dir, verbose=verbose)
    assert _state is not None  # narrowing for type checker

    result = _state.query.query_nearest_point(lat, lon)
    if result is None:
        raise RuntimeError(
            f"No tidal grid point found near ({lat:.4f}, {lon:.4f}). "
            "The coordinate may be outside the dataset domain."
        )

    distance_km: float = float(result["distance_km"])
    if max_km is not None and distance_km > max_km:
        raise ValueError(
            f"Nearest grid point is {distance_km:.1f} km away from "
            f"({lat:.4f}, {lon:.4f}), which exceeds max_km={max_km}."
        )

    from .analysis.preprocessing import load_parquet, prepare_dataframe

    point = result["point"]
    local_path = _state.cache.get(point["file_path"])
    raw_df, file_meta, _ = load_parquet(local_path)
    return prepare_dataframe(raw_df, file_meta)


# ---------------------------------------------------------------------------
# Re-exports — analysis helpers
# ---------------------------------------------------------------------------
from .analysis import (  # noqa: E402
    SiteSummaryMetrics,
    calculate_tidal_levels,
    calculate_tidal_periods,
    collect_site_metrics,
    compute_power_density,
    compute_power_density_summary,
    compute_sigma_bounds_from_layers,
    compute_sigma_bounds_from_seafloor,
    load_parquet,
    prepare_dataframe,
    select_layer_for_depth,
    standardize_metadata,
)
from .viz._style import PLOT_CONFIG  # noqa: E402
from .viz.tidal import (  # noqa: E402
    PlotSettings,
    analyze_power_density,
    create_tidal_resource_dashboard,
    generate_tidal_joint_probability,
    generate_tidal_site_assessment,
    plot_current_rose,
    plot_fft,
    plot_jpd_comparison_grid,
    plot_multi_site_comparison,
    plot_multi_site_exceedance_overlay,
    plot_power_density_profile,
    plot_power_exceedance,
    plot_sigma_layers_direction,
    plot_sigma_layers_speed,
    plot_speed_mesh,
    plot_tidal_asymmetry,
    plot_tidal_exceedance,
    plot_tidal_harmonic_analysis,
    plot_tidal_phase_analysis,
    plot_tidal_rose,
    plot_tidal_statistics,
    plot_tidal_time_series,
    plot_tidal_velocity_profile,
    plot_velocity_exceedance,
    plot_velocity_profile,
    plot_velocity_profile_with_histograms,
    plot_velocity_shear_profile,
)

__all__ = [
    "PLOT_CONFIG",
    "PlotSettings",
    "SiteSummaryMetrics",
    "analyze_power_density",
    "calculate_tidal_levels",
    "calculate_tidal_periods",
    "collect_site_metrics",
    "compute_power_density",
    "compute_power_density_summary",
    "compute_sigma_bounds_from_layers",
    "compute_sigma_bounds_from_seafloor",
    "create_tidal_resource_dashboard",
    "generate_tidal_joint_probability",
    "generate_tidal_site_assessment",
    "get_data_at_point",
    "load_parquet",
    "plot_current_rose",
    "plot_fft",
    "plot_jpd_comparison_grid",
    "plot_multi_site_comparison",
    "plot_multi_site_exceedance_overlay",
    "plot_power_density_profile",
    "plot_power_exceedance",
    "plot_sigma_layers_direction",
    "plot_sigma_layers_speed",
    "plot_speed_mesh",
    "plot_tidal_asymmetry",
    "plot_tidal_exceedance",
    "plot_tidal_harmonic_analysis",
    "plot_tidal_phase_analysis",
    "plot_tidal_rose",
    "plot_tidal_statistics",
    "plot_tidal_time_series",
    "plot_tidal_velocity_profile",
    "plot_velocity_exceedance",
    "plot_velocity_profile",
    "plot_velocity_profile_with_histograms",
    "plot_velocity_shear_profile",
    "prepare_dataframe",
    "select_layer_for_depth",
    "standardize_metadata",
]
