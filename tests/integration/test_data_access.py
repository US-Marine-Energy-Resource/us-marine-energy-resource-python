"""Integration tests for S3 data access and manifest spatial queries."""

import pandas as pd
import pytest

import us_marine_energy_resource.tidal_hindcast as tidal

from .conftest import COOK_INLET_BBOX, COOK_INLET_LINE, SITES

_SPEED_COLS = [f"vap_sea_water_speed_layer_{i}" for i in range(10)]
_POWER_COLS = [f"vap_sea_water_power_density_layer_{i}" for i in range(10)]

# Expected row counts for each region.
# Hourly regions (Cook Inlet 2005, Aleutian Islands 2010-06–2011-06) → 8 760 rows.
# Half-hourly regions (Puget Sound 2015, NH 2007, ME 2017)           → 17 520 rows.
_EXPECTED_ROWS: dict[str, int] = {
    "Cook Inlet, AK": 8_760,
    "Aleutian Islands, AK": 8_760,
    "Puget Sound, WA": 17_472,  # 364 days at 30 min — one day short of full year in this file
    "Piscataqua River, NH": 17_520,
    "Western Passage, ME": 17_520,
}

_HALF_HOURLY_REGIONS = {"Puget Sound, WA", "Piscataqua River, NH", "Western Passage, ME"}


@pytest.mark.integration
def test_cook_inlet_dataframe_structure(cook_inlet_df: pd.DataFrame) -> None:
    """get_data_at_point returns a fully-populated hourly-year DataFrame for Cook Inlet."""
    df = cook_inlet_df

    assert isinstance(df.index, pd.DatetimeIndex), "Index must be a DatetimeIndex"
    assert len(df) == 8_760, f"Expected 8 760 hourly rows; got {len(df)}"
    assert df.index[0].year == 2005, f"Cook Inlet data starts in 2005; got {df.index[0].year}"

    missing_speed = [c for c in _SPEED_COLS if c not in df.columns]
    assert not missing_speed, f"Missing speed columns: {missing_speed}"

    missing_power = [c for c in _POWER_COLS if c not in df.columns]
    assert not missing_power, f"Missing power density columns: {missing_power}"

    null_speed = df[_SPEED_COLS].isna().sum()
    assert null_speed.sum() == 0, f"Unexpected NaN in speed columns:\n{null_speed[null_speed > 0]}"

    assert (df[_SPEED_COLS] >= 0).all().all(), "Speed values must be non-negative"

    median_dt = df.index.to_series().diff().dropna().median()
    assert median_dt == pd.Timedelta(hours=1), f"Expected 1 h timestep; got {median_dt}"


@pytest.mark.integration
def test_all_regions_return_valid_dataframes(all_site_dfs: dict[str, pd.DataFrame]) -> None:
    """get_data_at_point returns valid DataFrames for all five dataset regions."""
    for name, df in all_site_dfs.items():
        assert isinstance(df, pd.DataFrame), f"{name}: expected DataFrame"
        assert isinstance(df.index, pd.DatetimeIndex), f"{name}: index must be DatetimeIndex"

        missing = [c for c in _SPEED_COLS if c not in df.columns]
        assert not missing, f"{name}: missing speed columns: {missing}"

        assert df[_SPEED_COLS].isna().sum().sum() == 0, f"{name}: NaN values in speed columns"
        assert (df[_SPEED_COLS] >= 0).all().all(), f"{name}: negative speed values found"

        assert len(df) == _EXPECTED_ROWS[name], (
            f"{name}: expected {_EXPECTED_ROWS[name]} rows; got {len(df)}"
        )

        expected_dt = (
            pd.Timedelta(minutes=30) if name in _HALF_HOURLY_REGIONS else pd.Timedelta(hours=1)
        )
        actual_dt = df.index.to_series().diff().dropna().median()
        assert actual_dt == expected_dt, (
            f"{name}: expected {expected_dt} timestep; got {actual_dt}"
        )


@pytest.mark.integration
def test_manifest_spatial_queries(cook_inlet_df: pd.DataFrame) -> None:
    """All three manifest query types return spatially accurate results in Cook Inlet."""
    assert tidal._state is not None
    query = tidal._state.query

    # Nearest-point query
    lat, lon = SITES["Cook Inlet, AK"]["lat"], SITES["Cook Inlet, AK"]["lon"]
    nearest = query.query_nearest_point(lat=lat, lon=lon)

    assert nearest is not None, "query_nearest_point returned None for a valid coordinate"
    assert nearest["distance_km"] < 1.0, (
        f"Nearest point is {nearest['distance_km']:.3f} km away; expected < 1 km"
    )
    assert abs(nearest["point"]["lat"] - lat) < 0.01
    assert abs(nearest["point"]["lon"] - lon) < 0.01
    assert "file_path" in nearest["point"], "Nearest result must include a file_path"

    # Rectangular area query
    lat_min, lat_max, lon_min, lon_max = COOK_INLET_BBOX
    area_faces = query.query_all_within_rectangular_area(lat_min, lat_max, lon_min, lon_max)
    assert len(area_faces) > 0, f"Area query returned 0 faces for bbox {COOK_INLET_BBOX}"

    # Line / transect query
    (lat1, lon1), (lat2, lon2) = COOK_INLET_LINE
    line_faces = query.query_all_on_line(lat1, lon1, lat2, lon2)
    assert len(line_faces) > 0, f"Line query returned 0 faces for transect {COOK_INLET_LINE}"
