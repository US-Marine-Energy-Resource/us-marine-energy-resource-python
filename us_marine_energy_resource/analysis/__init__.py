"""
US Marine Energy Resource — analysis subpackage.

Pure computation functions for tidal energy resource analysis.
No matplotlib dependency.

Quick start::

    from us_marine_energy_resource.analysis import load_parquet, prepare_dataframe
    from us_marine_energy_resource.analysis import select_layer_for_depth
    from us_marine_energy_resource.analysis import calculate_tidal_periods

    df, file_meta, var_meta = load_parquet("path/to/point.parquet")
    df = prepare_dataframe(df, file_meta)
    layer, depth = select_layer_for_depth(df, target_depth_m=10.0)
    periods = calculate_tidal_periods(df["vap_surface_elevation"], df.index)
"""

from .preprocessing import (
    DepthMode,
    compute_sigma_bounds_from_layers,
    compute_sigma_bounds_from_seafloor,
    load_parquet,
    prepare_dataframe,
    sigma_depth_axis_label,
    sigma_depth_scalar,
    sigma_depths_array,
    sigma_layer_depth_col,
    standardize_metadata,
)
from .resource import (
    SiteSummaryMetrics,
    calculate_tidal_levels,
    calculate_tidal_periods,
    collect_site_metrics,
    compute_power_density,
    compute_power_density_summary,
    select_layer_for_depth,
)

__all__ = [
    "DepthMode",
    "SiteSummaryMetrics",
    "calculate_tidal_levels",
    "calculate_tidal_periods",
    "collect_site_metrics",
    "compute_power_density",
    "compute_power_density_summary",
    "compute_sigma_bounds_from_layers",
    "compute_sigma_bounds_from_seafloor",
    "load_parquet",
    "prepare_dataframe",
    "select_layer_for_depth",
    "sigma_depth_axis_label",
    "sigma_depth_scalar",
    "sigma_depths_array",
    "sigma_layer_depth_col",
    "standardize_metadata",
]
