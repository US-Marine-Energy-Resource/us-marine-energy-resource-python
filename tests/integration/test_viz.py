"""Smoke tests for core visualization functions.

Verifies that each function executes without raising and returns the
expected types.  Visual correctness is not verified here — these tests
exist to catch API breakage and import-level regressions.
"""

import pandas as pd
import pytest
from matplotlib.figure import Figure

import us_marine_energy_resource.tidal_hindcast as tidal


@pytest.mark.integration
def test_plot_sigma_layers_speed(cook_inlet_df: pd.DataFrame) -> None:
    """plot_sigma_layers_speed returns a (Figure, Axes) tuple without raising."""
    import matplotlib.pyplot as plt

    fig, ax = tidal.plot_sigma_layers_speed(cook_inlet_df)
    assert isinstance(fig, Figure)
    plt.close(fig)


@pytest.mark.integration
def test_plot_sigma_layers_direction(cook_inlet_df: pd.DataFrame) -> None:
    """plot_sigma_layers_direction returns a (Figure, Axes) tuple without raising."""
    import matplotlib.pyplot as plt

    fig, ax = tidal.plot_sigma_layers_direction(cook_inlet_df)
    assert isinstance(fig, Figure)
    plt.close(fig)


@pytest.mark.integration
def test_plot_velocity_exceedance(cook_inlet_df: pd.DataFrame) -> None:
    """plot_velocity_exceedance returns a Figure and a 10-layer stats dict."""
    import matplotlib.pyplot as plt

    fig, stats = tidal.plot_velocity_exceedance(cook_inlet_df)
    assert isinstance(fig, Figure)
    assert isinstance(stats, dict)
    assert len(stats) == 10, f"Expected stats for 10 layers; got {len(stats)}"
    plt.close(fig)


@pytest.mark.integration
def test_generate_tidal_joint_probability(cook_inlet_df: pd.DataFrame) -> None:
    """generate_tidal_joint_probability returns a Figure without raising."""
    import matplotlib.pyplot as plt

    fig = tidal.generate_tidal_joint_probability(cook_inlet_df, sigma_layer=4)
    assert isinstance(fig, Figure)
    plt.close(fig)


@pytest.mark.integration
def test_plot_velocity_profile_with_histograms(cook_inlet_df: pd.DataFrame) -> None:
    """plot_velocity_profile_with_histograms returns (Figure, stats dict) without raising."""
    import matplotlib.pyplot as plt

    fig, stats = tidal.plot_velocity_profile_with_histograms(cook_inlet_df)
    assert isinstance(fig, Figure)
    assert isinstance(stats, dict)
    plt.close(fig)
