"""Shared fixtures and test coordinates for integration tests."""

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

import us_marine_energy_resource.tidal_hindcast as tidal

# ---------------------------------------------------------------------------
# Test coordinates — one representative point per dataset region
# ---------------------------------------------------------------------------

SITES: dict[str, dict[str, float]] = {
    "Cook Inlet, AK": {"lat": 60.735016, "lon": -151.431396},
    "Aleutian Islands, AK": {"lat": 54.803799, "lon": -163.364441},
    "Puget Sound, WA": {"lat": 47.270191, "lon": -122.548172},
    "Piscataqua River, NH": {"lat": 43.079498, "lon": -70.752319},
    "Western Passage, ME": {"lat": 44.920837, "lon": -66.988762},
}

# Cook Inlet bounding box (lat_min, lat_max, lon_min, lon_max)
# Sized to match ~33 faces / ~118 MB — well under the 200 MB download budget.
COOK_INLET_BBOX = (60.731, 60.737, -151.437, -151.428)

# Cook Inlet transect — two waypoints forming a short cross-channel line
COOK_INLET_LINE = ((60.72, -151.43), (60.75, -151.44))

# ---------------------------------------------------------------------------
# Session-scoped fixtures — download each site once per test run
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def cook_inlet_df(tmp_path_factory: pytest.TempPathFactory) -> pd.DataFrame:
    """Download Cook Inlet data and initialise tidal._state for all tests."""
    cache_dir = tmp_path_factory.mktemp("tidal_cache")
    return tidal.get_data_at_point(
        lat=SITES["Cook Inlet, AK"]["lat"],
        lon=SITES["Cook Inlet, AK"]["lon"],
        cache_dir=cache_dir,
    )


@pytest.fixture(scope="session")
def all_site_dfs(cook_inlet_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Download one point per region; reuses the _state initialised by cook_inlet_df."""
    results: dict[str, pd.DataFrame] = {"Cook Inlet, AK": cook_inlet_df}
    for name, coords in SITES.items():
        if name == "Cook Inlet, AK":
            continue
        results[name] = tidal.get_data_at_point(coords["lat"], coords["lon"])
    return results
