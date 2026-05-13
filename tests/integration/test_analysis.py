"""Integration tests for the analysis pipeline on real downloaded data."""

import numpy as np
import pandas as pd
import pytest

import us_marine_energy_resource.tidal_hindcast as tidal
from us_marine_energy_resource.analysis import (
    compute_power_density,
    load_parquet,
    prepare_dataframe,
    select_layer_for_depth,
)


@pytest.mark.integration
def test_load_parquet_and_prepare_dataframe(cook_inlet_df: pd.DataFrame) -> None:
    """load_parquet + prepare_dataframe produce a fully-populated DataFrame.

    Verifies the end-to-end load path: parquet I/O, metadata extraction, sigma
    depth computation, and column completeness — without re-downloading from S3.
    """
    assert tidal._state is not None
    query = tidal._state.query
    nearest = query.query_nearest_point(lat=60.735016, lon=-151.431396)
    assert nearest is not None

    local_path = tidal._state.cache.get(nearest["point"]["file_path"])
    raw_df, file_meta, var_meta = load_parquet(local_path)

    # File-level metadata must carry dataset identity fields
    assert "title" in file_meta or "institution" in file_meta, (
        f"file_meta missing expected keys; got: {list(file_meta.keys())}"
    )

    # Variable-level metadata must cover at least the speed columns
    speed_col = "vap_sea_water_speed_layer_0"
    assert speed_col in var_meta, f"{speed_col} missing from var_meta"

    df = prepare_dataframe(raw_df, file_meta)

    # All 10 sigma depth columns must be present and positive
    depth_cols = [f"vap_sigma_depth_layer_{i}" for i in range(10)]
    missing = [c for c in depth_cols if c not in df.columns]
    assert not missing, f"prepare_dataframe did not add sigma depth columns: {missing}"
    assert (df[depth_cols] > 0).all().all(), "All sigma layer depths must be positive"

    # Mean layer depth increases monotonically from surface (layer 0) to near-bed (layer 9)
    mean_depths = [df[f"vap_sigma_depth_layer_{i}"].mean() for i in range(10)]
    for i in range(9):
        assert mean_depths[i] < mean_depths[i + 1], (
            f"Mean depth not monotonically increasing: layer {i} ({mean_depths[i]:.3f} m) "
            f">= layer {i + 1} ({mean_depths[i + 1]:.3f} m)"
        )

    # Sigma bounds (11 values) must span from 0 to seafloor
    bound_cols = [f"vap_sigma_depth_bound_{i}" for i in range(11)]
    assert all(c in df.columns for c in bound_cols), "Sigma depth bound columns missing"
    assert (df["vap_sigma_depth_bound_0"] == 0).all(), "Bound 0 must be 0 (sea surface)"
    pd.testing.assert_series_equal(
        df["vap_sigma_depth_bound_10"],
        df["vap_sea_floor_depth"],
        check_names=False,
        check_exact=False,
        rtol=1e-5,
    )


@pytest.mark.integration
def test_computed_power_density_matches_stored_values(cook_inlet_df: pd.DataFrame) -> None:
    """compute_power_density(speed) agrees with the pre-computed dataset values.

    The dataset stores power density computed as 0.5 * 1025 * speed^3.  This
    test verifies that the library's formula reproduces those stored values to
    within 5% mean relative error — catching any change to the formula or the
    default rho constant.
    """
    df = cook_inlet_df

    for layer in range(10):
        speed = df[f"vap_sea_water_speed_layer_{layer}"].to_numpy()
        stored = df[f"vap_sea_water_power_density_layer_{layer}"].to_numpy()
        computed = compute_power_density(speed)

        nonzero = stored > 0
        assert nonzero.sum() > 100, (
            f"Layer {layer}: fewer than 100 non-zero power density values; "
            "cannot reliably compute relative error"
        )

        rel_error = np.abs(computed[nonzero] - stored[nonzero]) / stored[nonzero]
        mean_err = float(rel_error.mean())
        assert mean_err < 0.05, (
            f"Layer {layer}: mean relative error between computed and stored power density "
            f"is {mean_err:.3%} — expected < 5%"
        )


@pytest.mark.integration
def test_select_layer_for_depth_returns_valid_layers(cook_inlet_df: pd.DataFrame) -> None:
    """select_layer_for_depth returns a valid layer index for surface and seafloor references."""
    df = cook_inlet_df

    for depth_m in (5.0, 10.0, 20.0):
        layer, actual_depth = select_layer_for_depth(df, depth_m, relative_to="surface")
        assert 0 <= layer <= 9, (
            f"depth={depth_m} m from surface: layer index {layer} out of [0, 9]"
        )
        assert actual_depth > 0, f"depth={depth_m} m from surface: returned depth {actual_depth} <= 0"

        layer_sf, actual_depth_sf = select_layer_for_depth(df, depth_m, relative_to="sea_floor")
        assert 0 <= layer_sf <= 9, (
            f"depth={depth_m} m from sea_floor: layer index {layer_sf} out of [0, 9]"
        )

    # Shallower surface depths should resolve to lower layer indices (closer to surface)
    layer_shallow, _ = select_layer_for_depth(df, 2.0, relative_to="surface")
    layer_deep, _ = select_layer_for_depth(df, 20.0, relative_to="surface")
    assert layer_shallow <= layer_deep, (
        "2 m depth should resolve to a shallower (lower) layer than 20 m"
    )
