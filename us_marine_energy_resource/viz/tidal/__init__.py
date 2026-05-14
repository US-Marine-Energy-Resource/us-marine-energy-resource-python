"""Tidal visualization submodule."""

from ..settings import DepthPerspective, PlotSettings, set_depth_perspective
from ...analysis.preprocessing import DepthMode
from .comparison import (
    plot_jpd_comparison_grid,
    plot_multi_site_comparison,
)
from .dashboard import create_tidal_resource_dashboard, generate_tidal_site_assessment
from .energy import (
    analyze_power_density,
)
from .exceedance import (
    plot_multi_site_exceedance_overlay,
    plot_power_exceedance,
    plot_tidal_exceedance,
    plot_velocity_exceedance,
)
from .harmonic import plot_fft, plot_tidal_harmonic_analysis, plot_tidal_phase_analysis
from .joint_probability import generate_tidal_joint_probability
from .polar import plot_current_rose, plot_tidal_rose
from .profiles import (
    plot_power_density_profile,
    plot_tidal_velocity_profile,
    plot_velocity_profile,
    plot_velocity_profile_with_histograms,
    plot_velocity_shear_profile,
)
from .sigma import plot_sigma_layers_direction, plot_sigma_layers_speed, plot_speed_mesh
from .time_series import (
    plot_tidal_asymmetry,
    plot_tidal_statistics,
    plot_tidal_time_series,
)

__all__ = [
    "DepthMode",
    "DepthPerspective",
    "PlotSettings",
    "set_depth_perspective",
    "analyze_power_density",
    "create_tidal_resource_dashboard",
    "generate_tidal_joint_probability",
    "generate_tidal_site_assessment",
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
]
