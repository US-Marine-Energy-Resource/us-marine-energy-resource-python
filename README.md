# US Marine Energy Resource


## Overview

`us-marine-energy-resource` is a Python library for accessing the [U.S.
DOE H2O High Resolution Tidal
Hindcast](https://us-marine-energy-resource.github.io/tidal/high_resolution_hindcast/)
dataset, a high-resolution, 3D tidal current hindcast for five US
coastal regions, generated with the [Finite Volume Community Ocean Model
(FVCOM)](https://us-marine-energy-resource.github.io/tidal/high_resolution_hindcast/).

> [!IMPORTANT]
>
> This library is in early development and the API is subject to change.
> The core functionality of downloading and visualizing tidal hindcast
> data at specific points is stable, but additional features and
> datasets are still being added. Please reach out if you have questions
> or would like to contribute!

> [!NOTE]
>
> At this time this libary does not support for the U.S. DOE H20 wave
> energy hindcast dataset. The marine and hydrokinetic toolkit (MHKiT)
> can access using the `wave.io.hindcast` module showcased in this [wave
> hindcast
> example](https://mhkit-software.github.io/MHKiT/WPTO_hindcast_example.html)
> that leverages the [NLR Resource eXtraction tool
> (rex)](https://github.com/NatLabRockies/rex) to access wave hindcast
> data.

## Installation

``` bash
pip install git+https://github.com/US-Marine-Energy-Resource/us-marine-energy-resource-python.git@main
```

## Tidal Quick Start

`us_marine_energy_resource.tidal_hindcast.get_data_at_point` takes a
latitude and longitude as input and fetch a full year of tidal current
data at US coastal coordinate within the 5 locations listed above and
visualize current speed across all 10 depth layers over the entire
hindcast year.

The dataset covers a full year at each region at hourly or half-hourly
resolution across 10 terrain-following [sigma
layers](https://us-marine-energy-resource.github.io/tidal/high_resolution_hindcast/sigma-layers/)
from the sea surface to the seafloor.

| Region                          | Period    | Timestep    | Grid Points |
|---------------------------------|-----------|-------------|-------------|
| Cook Inlet, Alaska              | 2005      | hourly      | 392,002     |
| Aleutian Islands, Alaska        | 2010–2011 | hourly      | 797,978     |
| Puget Sound, Washington         | 2015      | half-hourly | 1,734,765   |
| Piscataqua River, New Hampshire | 2007      | half-hourly | 292,927     |
| Western Passage, Maine          | 2017      | half-hourly | 231,208     |

Full dataset documentation, variable definitions, methodology, and
validation are at
[us-marine-energy-resource.github.io/tidal/high_resolution_hindcast/](https://us-marine-energy-resource.github.io/tidal/high_resolution_hindcast/).

``` python
import matplotlib.pyplot as plt
import pandas as pd

from us_marine_energy_resource import tidal_hindcast as tidal
from us_marine_energy_resource.tidal_hindcast import PlotSettings

# Cook Inlet near Nikiski AK,
lat=60.735016
lon=-151.431396
location_name = "Cook Inlet, Near Nikiski, AK"

df = tidal.get_data_at_point(lat=lat, lon=lon)
```

`us_marine_energy_resource` has functions to plot the point data at the
10 uniform depths over time. The underlying data contains speed \[m/s\]
and direction \[deg cw from True North\] calculated from the underlying
model `u` and `v` variables at each sigma layer at each time step. To
convert this to a plot this library uses the calculated sigma depth and
uniform model specification to “extract” volume data, and convert the
data from a compacted format to a format usable for engineering
analysis.

The following visualization uses `plot_sigma_layers_speed` function with
the downloaded and extracted pandas DataFrame, `df` and a `PlotSettings`
object (custom class for this library to control plot styling) and
outputs a visualization of speed at each volume over time.

Each horizontal band is one of 10 sigma layers model results, expanded
to color an entire volume, spanning the full water column from the sea
surface (top) to the seabed (bottom). Color encodes current speed in
m/s. The tidal cycle and spring–neap modulation are immediately visible
across the full hindcast year.

``` python
settings=PlotSettings(
    title=f"Full Model 1 Year | Current Speed | {location_name}",
    fig_width=9,
    fig_height=2.5,
    caption=f"Latitude: {lat}, Longitude: {lon}",
    save_path="docs/images/quickstart-sigma-speed-year.png",
)

tidal.plot_sigma_layers_speed(df, settings=settings)
```

![Full year current speed across sigma layers — Cook Inlet near Nikiski,
AK](docs/images/quickstart-sigma-speed-year.png)

Additionally we can plot direction \[deg clockwise from true north\] at
all depths over time.

``` python
settings.title = settings.title.replace("Current Speed", "Direction [deg cw from True North]")
settings.save_path = "docs/images/quickstart-sigma-direction-year.png"

tidal.plot_sigma_layers_direction(df, settings=settings)
```

![Full year direction across sigma layers — Cook Inlet near Nikiski,
AK](docs/images/quickstart-sigma-direction-year.png)

It is also possible to zoom into specific start dates within the model
run. The simplest way to do this is to create time objects from the data
and use variables to control the offset from the start of the dataset
and the number of days visible.

``` python
n_days = 3
start_day_offset = 7
start_date=str((df.index[0] + pd.Timedelta(days=start_day_offset)).date())
end_date=str((df.index[0] + pd.Timedelta(days=start_day_offset + n_days)).date())

settings = PlotSettings(
    title=f"{n_days} Days | Current Speed | {location_name}",
    start_date=start_date,
    end_date=end_date,
    fig_width=8,
    fig_height=3,
    caption=f"Latitude: {lat}, Longitude: {lon}",
    save_path="docs/images/quickstart-sigma-speed-3day.png",
)

tidal.plot_sigma_layers_speed(df, settings=settings)
```

![3-day current speed across sigma layers — Cook Inlet near Nikiski,
AK](docs/images/quickstart-sigma-speed-3day.png)

``` python
settings.title = settings.title.replace("Current Speed", "Direction [deg cw from True North]")
settings.save_path = "docs/images/quickstart-sigma-direction-3day.png"

tidal.plot_sigma_layers_direction(df, settings=settings)
```

![3-day direction across sigma layers — Cook Inlet near Nikiski,
AK](docs/images/quickstart-sigma-direction-3day.png)

The same `df` can be used to visualize tidal joint probability
distributions (single sigma layer) and velocity exceedance curves
(multiple sigma layers):

``` python
tidal.generate_tidal_joint_probability(
    df,
    sigma_layer=4,
    settings=PlotSettings(
        title=f"Joint Probability Distribution\n{location_name}\nSigma Layer 4",
        fig_width=8,
        fig_height=8,
        save_path="docs/images/quickstart-jpd-layer-4.png",
    ),
)
```

![Joint probability distribution at sigma layer 4 — Cook Inlet near
Nikiski, AK](docs/images/quickstart-jpd-layer-4.png)

``` python
_, stats = tidal.plot_velocity_exceedance(
    df,
    settings=PlotSettings(
        title=f"Velocity Exceedance | {location_name}",
        fig_width=10,
        fig_height=5,
        caption=f"Latitude: {lat}, Longitude: {lon}",
        save_path="docs/images/quickstart-velocity-exceedance.png",
    ),
)
```

![Velocity exceedance across all sigma layers — Cook Inlet near Nikiski,
AK](docs/images/quickstart-velocity-exceedance.png)

`plot_velocity_profile_with_histograms` produces a five-panel diagnostic
overview of the full vertical structure of the tidal resource. From left
to right: a mean velocity profile with per-layer box plots showing
spread and whiskers; a depth vs. speed scatter colored by direction with
quadratic mean and maximum fit curves; and per-layer histograms of
current speed, current direction, and sigma-layer depth. Before
plotting, dry time steps and anomalously thin (“smushed”) sigma layers
are removed — the data-quality summary in the top-left corner reports
what was filtered.

``` python
tidal.plot_velocity_profile_with_histograms(
    df,
    settings=PlotSettings(
        title=f"Velocity and Direction Overview | {location_name}",
        fig_width=10,
        fig_height=16,
        caption=f"Latitude: {lat}, Longitude: {lon}",
        save_path="docs/images/quickstart-velocity-profile.png",
    ),
)
```

![Velocity profile with histograms — Cook Inlet near Nikiski,
AK](docs/images/quickstart-velocity-profile.png)

## Dataset Variables

The [full variable
reference](https://us-marine-energy-resource.github.io/tidal/high_resolution_hindcast/variables/)
documents every field in the dataset. The key summary variables
(available in the manifest for every grid point) are:

| Variable | Units | Description |
|----|----|----|
| [Mean Current Speed](https://us-marine-energy-resource.github.io/tidal/high_resolution_hindcast/variables/mean-current-speed/) | m/s | Annual average depth-averaged current speed |
| [95th Percentile Current Speed](https://us-marine-energy-resource.github.io/tidal/high_resolution_hindcast/variables/95th-percentile-current-speed/) | m/s | Extreme current speed, outlier-tolerant |
| [Mean Power Density](https://us-marine-energy-resource.github.io/tidal/high_resolution_hindcast/variables/mean-power-density/) | W/m² | Annual average depth-averaged kinetic energy flux |
| [95th Percentile Power Density](https://us-marine-energy-resource.github.io/tidal/high_resolution_hindcast/variables/95th-percentile-power-density/) | W/m² | Extreme power density, robust to cubic-velocity sensitivity |
| [Tidal Range](https://us-marine-energy-resource.github.io/tidal/high_resolution_hindcast/variables/tidal-range/) | m | Max − min sea surface elevation over the hindcast year |
| [Minimum Water Depth](https://us-marine-energy-resource.github.io/tidal/high_resolution_hindcast/variables/minimum-water-depth/) | m | Minimum depth over the hindcast year (navigation constraint) |
| [Full Year S3 URI](https://us-marine-energy-resource.github.io/tidal/high_resolution_hindcast/variables/full_year_s3_uri/) | — | Direct S3 link to the time-series parquet for each grid point |

The time-series parquet (one file per grid point) contains hourly or
half-hourly records for one year with columns for speed, direction,
power density, and sigma-layer depth bounds at all 10 vertical levels.

Using a downloaded dataframe as an example, the column names for speed,
direction, power density, and sigma-layer depth bounds at each of the 10
vertical levels are:

``` python
df
```

## Usage examples

The following sections walk through data retrieval and three core
visualizations — time series, velocity exceedance, and joint probability
distribution — for six candidate tidal energy sites across the US.

### Site definitions

``` python
from dataclasses import dataclass


@dataclass
class Site:
    """A tidal energy candidate site."""

    name: str
    lat: float
    lon: float
    region: str


SITES: list[Site] = [
    Site(
        "Upper Cook Inlet, AK",
        lat=60.735016,
        lon=-151.431396,
        region="Cook Inlet, Alaska",
    ),
    Site(
        "Tacoma Narrows, WA",
        lat=47.270191,
        lon=-122.548172,
        region="Puget Sound, Washington",
    ),
    Site(
        "Admiralty Inlet, WA",
        lat=48.173931,
        lon=-122.774963,
        region="Puget Sound, Washington",
    ),
    Site(
        "UNH Living Bridge, NH",
        lat=43.079498,
        lon=-70.752319,
        region="Piscataqua River, New Hampshire",
    ),
    Site(
        "Moose Island, Western Passage, ME",
        lat=44.920837,
        lon=-66.988762,
        region="Western Passage, Maine",
    ),
    Site(
        "False Pass, Aleutian Islands, AK",
        lat=54.803799,
        lon=-163.364441,
        region="Aleutian Islands, Alaska",
    ),
]

PRIMARY = SITES[0]
```

### Load data for all sites

`get_data_at_point` finds the nearest grid point in the manifest and
downloads (or returns from cache) the full-year time-series parquet. It
returns a `DataFrame` with a `DatetimeIndex` and columns for speed,
direction, power density, and sigma-layer depth bounds at all 10
vertical levels.

``` python
site_data: dict[str, pd.DataFrame] = {}

for site in SITES:
    print(f"Loading {site.name} …")
    site_data[site.name] = tidal.get_data_at_point(site.lat, site.lon)
    site_df = site_data[site.name]
    date_range = f"{site_df.index[0].date()} → {site_df.index[-1].date()}"
    print(f"  {len(site_df):,} timesteps  ({date_range})\n")
```

    Loading Upper Cook Inlet, AK …
      8,760 timesteps  (2005-01-01 → 2005-12-31)

    Loading Tacoma Narrows, WA …
      17,472 timesteps  (2015-01-01 → 2015-12-30)

    Loading Admiralty Inlet, WA …
      17,472 timesteps  (2015-01-01 → 2015-12-30)

    Loading UNH Living Bridge, NH …
      17,520 timesteps  (2007-01-01 → 2007-12-31)

    Loading Moose Island, Western Passage, ME …
      17,520 timesteps  (2017-01-01 → 2017-12-31)

    Loading False Pass, Aleutian Islands, AK …
      8,760 timesteps  (2010-06-03 → 2011-06-02)

## Primary site deep-dive — Upper Cook Inlet, AK

Cook Inlet has some of the most energetic tidal currents in North
America. Spring tidal ranges exceed 9 m and the geometry of the upper
inlet amplifies current speeds to 3–4 m/s — among the highest in the US.
It is a natural first site for demonstrating what this dataset reveals
about the vertical structure and variability of a tidal energy resource.

### Current speed across depth and time

The depth-time cross-section shows current speed across all 10 sigma
layers for the full hindcast year. Each band spans one sigma layer —
equal fractions of the water column from surface to seafloor. The
spring–neap cycle, the tidal asymmetry between flood and ebb, and the
vertical shear between the fast surface and slower near-bed layers are
all directly visible.

``` python
df_primary = site_data[PRIMARY.name]
lat, lon = PRIMARY.lat, PRIMARY.lon

tidal.plot_sigma_layers_speed(
    df_primary,
    settings=PlotSettings(
        title=f"Current Speed Across Sigma Layers — {PRIMARY.name}",
        fig_height=3,
        fig_width=8,
        caption=f"Latitude: {lat}, Longitude: {lon}",
        save_path="docs/images/cook-inlet-sigma-speed.png",
    ),
)
```

![Current speed across sigma layers — Upper Cook Inlet,
AK](docs/images/cook-inlet-sigma-speed.png)

### Velocity exceedance

Exceedance probability curves across all ten sigma layers. Annotated
percentiles show the current speed exceeded for 50%, 25%, 10%, … of the
hindcast record — the key inputs for turbine capacity-factor estimates.

``` python
_, exc_stats = tidal.plot_velocity_exceedance(
    df_primary,
    settings=PlotSettings(
        title=f"Velocity Exceedance — {PRIMARY.name}",
        fig_width=10,
        fig_height=5,
        caption=f"Latitude: {lat}, Longitude: {lon}",
        save_path="docs/images/cook-inlet-exceedance.png",
    ),
)
```

![Velocity exceedance across all sigma layers — Upper Cook Inlet,
AK](docs/images/cook-inlet-exceedance.png)

The returned `exc_stats` dict is keyed by `"Layer {i}"` and contains
per-layer exceedance speeds at each annotated percentile. We can create
a `pd.DataFrame` from this data to display in tabular form.

``` python
exc_stats = pd.DataFrame(exc_stats).T[["mean", "5%", "1%", "0.1%", "max"]]
exc_stats
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|         | mean     | 5%       | 1%       | 0.1%     | max      |
|---------|----------|----------|----------|----------|----------|
| Layer 0 | 1.845529 | 3.139129 | 3.393487 | 3.543426 | 3.601168 |
| Layer 1 | 1.795470 | 3.047897 | 3.293689 | 3.438518 | 3.494025 |
| Layer 2 | 1.746668 | 2.959500 | 3.196892 | 3.339104 | 3.391897 |
| Layer 3 | 1.695603 | 2.866398 | 3.096244 | 3.235751 | 3.286159 |
| Layer 4 | 1.640122 | 2.768487 | 2.988489 | 3.123703 | 3.172075 |
| Layer 5 | 1.577625 | 2.660134 | 2.868485 | 2.998511 | 3.044529 |
| Layer 6 | 1.503978 | 2.530808 | 2.727775 | 2.852564 | 2.895835 |
| Layer 7 | 1.411291 | 2.370027 | 2.554669 | 2.670877 | 2.711424 |
| Layer 8 | 1.280689 | 2.147953 | 2.313955 | 2.418449 | 2.455549 |
| Layer 9 | 1.041188 | 1.743161 | 1.877486 | 1.960242 | 1.990621 |

</div>

### Joint probability distribution

Polar histogram of current speed vs. direction at the mid-column sigma
layer (layer 4). Shows the dominant flow direction and how speed is
distributed across the tidal cycle.

``` python
tidal.generate_tidal_joint_probability(
    df_primary,
    sigma_layer=4,
    settings=PlotSettings(
        title=f"Joint Probability Distribution — {PRIMARY.name}",
        fig_width=8,
        fig_height=8,
        caption=f"Latitude: {lat}, Longitude: {lon}",
        save_path="docs/images/cook-inlet-jpd-layer-4.png",
    ),
)
```

![Joint probability distribution at sigma layer 4 — Upper Cook Inlet,
AK](docs/images/cook-inlet-jpd-layer-4.png)

------------------------------------------------------------------------

## All-sites comparison

The following three visualizations compare the same plot type across all
six candidate sites. A single `ANALYSIS_DEPTH` variable controls the
nominal depth used for layer selection, keeping the exceedance and JPD
panels on an apples-to-apples basis.

### Setup — depth and layer selection

``` python
import math

ANALYSIS_DEPTH = 10.0  # m — nominal analysis depth
ANALYSIS_DEPTH_RELATIVE_TO = "surface"  # "surface" or "sea_floor"

# Select the sigma layer closest to ANALYSIS_DEPTH for each site.
site_layers: dict[str, tuple[int, float]] = {}
for site in SITES:
    layer, actual_depth = tidal.select_layer_for_depth(
        site_data[site.name],
        ANALYSIS_DEPTH,
        relative_to=ANALYSIS_DEPTH_RELATIVE_TO,
    )
    site_layers[site.name] = (layer, actual_depth)
    print(
        f"{site.name:<40} layer {layer}  "
        f"(mean depth {actual_depth:.1f} m)"
    )
```

    Upper Cook Inlet, AK                     layer 2  (mean depth 8.4 m)
    Tacoma Narrows, WA                       layer 1  (mean depth 9.4 m)
    Admiralty Inlet, WA                      layer 1  (mean depth 8.8 m)
    UNH Living Bridge, NH                    layer 4  (mean depth 9.7 m)
    Moose Island, Western Passage, ME        layer 2  (mean depth 10.0 m)
    False Pass, Aleutian Islands, AK         layer 2  (mean depth 11.4 m)

### Current speed across depth and time — all sites

Each panel covers the full hindcast year at one site. The colorbar range
is shared across all six: `vmax` is the maximum speed in the dataset,
rounded up to the nearest 0.5 m/s.

``` python
# Compute shared "nice max" colorbar limit.
all_max_speeds = [
    float(
        site_data[site.name][
            [f"vap_sea_water_speed_layer_{i}" for i in range(10)]
        ].max().max()
    )
    for site in SITES
]
speed_vmax = math.ceil(max(all_max_speeds) / 0.5) * 0.5

for site in SITES:
    slug = site.name.lower().replace(", ", "-").replace(" ", "-").replace(".", "")
    site_df = site_data[site.name]
    tidal.plot_sigma_layers_speed(
        site_df,
        settings=PlotSettings(
            title=f"Current Speed Across Sigma Layers — {site.name}",
            caption=f"Lat: {site.lat}, Lon: {site.lon}",
            colorbar_max=speed_vmax,
            fig_width=9,
            fig_height=2.5,
            save_path=f"docs/images/sigma-speed-{slug}.png",
        ),
    )
```

![Current speed — Upper Cook Inlet,
AK](docs/images/sigma-speed-upper-cook-inlet-ak.png) ![Current speed —
Tacoma Narrows, WA](docs/images/sigma-speed-tacoma-narrows-wa.png)
![Current speed — Admiralty Inlet,
WA](docs/images/sigma-speed-admiralty-inlet-wa.png) ![Current speed —
UNH Living Bridge, NH](docs/images/sigma-speed-unh-living-bridge-nh.png)
![Current speed — Moose Island, Western Passage,
ME](docs/images/sigma-speed-moose-island-western-passage-me.png)
![Current speed — False Pass, Aleutian Islands,
AK](docs/images/sigma-speed-false-pass-aleutian-islands-ak.png)

### Velocity exceedance — all sites

All six sites on one figure at their respective analysis-depth layers.

``` python
TAB10 = plt.cm.tab10.colors  # type: ignore[attr-defined]
site_records_exc = [
    (site.name, site_data[site.name], site_layers[site.name][0], TAB10[i])
    for i, site in enumerate(SITES)
]

tidal.plot_multi_site_exceedance_overlay(
    site_records_exc,
    settings=PlotSettings(
        title=f"Velocity Exceedance — All Sites  (analysis depth ≈ {ANALYSIS_DEPTH} m {ANALYSIS_DEPTH_RELATIVE_TO})",
        fig_height=3,
        fig_width=8,
        save_path="docs/images/all-sites-exceedance-overlay.png",
    ),
)
```

![Velocity exceedance overlay — all six
sites](docs/images/all-sites-exceedance-overlay.png)

### Joint probability distribution — all sites

2 × 3 grid of JPD polar histograms with a shared color scale.

``` python
site_records_jpd = [
    (site.name, site_data[site.name], site_layers[site.name][0])
    for site in SITES
]

tidal.plot_jpd_comparison_grid(
    site_records_jpd,
    ncols=2,
    settings=PlotSettings(
        title=f"Joint Probability Distribution — All Sites  (analysis depth ≈ {ANALYSIS_DEPTH} m {ANALYSIS_DEPTH_RELATIVE_TO})",
        fig_width=10,
        fig_height=13,
        save_path="docs/images/all-sites-jpd-grid.png",
    ),
)
```

![Joint probability distribution grid — all six
sites](docs/images/all-sites-jpd-grid.png)

## Command Line Interface

Installing via pip includes the `us-tidal-query` CLI for quick spatial
lookups directly against the manifest — no Python required. The three
modes below are run against Cook Inlet, AK.

### Point query — nearest grid point

``` {bash}
#| output: true
us-tidal-query --lat 60.73 --lon -151.43 --info-only
```

### Area query — all grid points in a bounding box

``` {bash}
#| output: true
us-tidal-query --mode area --lat-min 60.7 --lat-max 60.8 \
               --lon-min -151.5 --lon-max -151.4
```

### Line query — grid points along a transect

``` {bash}
#| output: true
us-tidal-query --mode line --start-lat 60.7 --start-lon -151.4 \
               --end-lat 60.8 --end-lon -151.5 \
               --max-distance-from-line 0.01
```

## Direct Downloads using the `tidal_hindcast` API

Tidal hindcast data is accessible via multiple functions that can be
used independently of the plotting functions. This allows users to
access the underlying data at specific points, along lines, or within
rectangular areas. This downloads data to a local cache directory and
returns the path to the downloaded files, which can be loaded and
analyzed with the `load_parquet` and `prepare_dataframe` functions in
the `analysis` module.

`tidal._state` is initialized lazily on the first `get_data_at_point()`
call; call that once to populate the shared cache and manifest before
accessing `_state` directly.

``` python
from us_marine_energy_resource import tidal_hindcast as tidal
from us_marine_energy_resource.analysis import load_parquet, prepare_dataframe

# Initialize the shared cache and manifest (no-op if already done above).
tidal.get_data_at_point(lat=60.73, lon=-151.43)

# Access the underlying cache and manifest directly.
cache = tidal._state.cache
query = tidal._state.query   # TidalManifestQuery instance

# --- Spatial queries against the manifest -----------------------------------

# Single nearest point
point = query.query_nearest_point(lat=60.73, lon=-151.43)

# All grid centroids within a bounding box (load_details=False → manifest-only,
# no per-grid S3 requests — extremely fast)
area = query.query_all_within_rectangular_area(
    60.7, 60.8, -151.5, -151.4, load_details=False
)

# All grid centroids within 0.01° of a transect (load_details=False → fast)
line = query.query_all_on_line(
    60.7, -151.4, 60.8, -151.5, max_distance_deg=0.01, load_details=False
)

print(
    f"Nearest point : face {point['point']['face_id']}"
    f"  ({point['point']['lat']:.4f}, {point['point']['lon']:.4f})"
    f"  —  {point['distance_km']:.3f} km"
)
print(f"Area query    : {len(area)} grid centroids in bbox")
print(f"Line query    : {len(line)} grid centroids along transect")
```

    Nearest point : face 00126601  (60.7298, -151.4297)  —  0.023 km
    Area query    : 143 grid centroids in bbox
    Line query    : 83 grid centroids along transect

``` python
# Load a specific grid point's full-year time-series parquet.
local_path = cache.get(point["point"]["file_path"])
raw_df, file_meta, var_meta = load_parquet(local_path)
df = prepare_dataframe(raw_df, file_meta)
print(f"Loaded {len(df):,} timesteps, {df.shape[1]} columns")
```

    Loaded 8,760 timesteps, 136 columns
