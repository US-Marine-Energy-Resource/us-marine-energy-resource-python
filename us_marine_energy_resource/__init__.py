"""
US Marine Energy Resource — Python library for tidal energy data.

Download, cache, and query the H20 High Resolution Tidal Hindcast dataset
stored on AWS (marine-energy-data S3 bucket).

Quick start (high-level API)::

    from us_marine_energy_resource import tidal_hindcast as tidal

    df = tidal.get_data_at_point(lat=60.73, lon=-151.43)
    fig = tidal.plot_tidal_time_series(df)
    fig, stats = tidal.plot_velocity_exceedance(df)
    fig = tidal.generate_tidal_joint_probability(df, sigma_layer=4)

Low-level API::

    from us_marine_energy_resource.cache import S3CacheManager
    from us_marine_energy_resource.manifest import TidalManifestQuery

    cache = S3CacheManager(bucket="marine-energy-data", prefix="us-tidal")
    manifest_path, _ = find_latest_manifest_s3(cache)
    query = TidalManifestQuery(manifest_path, s3_cache=cache)
    result = query.query_nearest_point(lat=60.73, lon=-151.43)
"""

from . import tidal_hindcast
from .analysis.preprocessing import DepthMode
from .cache import S3CacheManager
from .manifest import TidalManifestQuery
from .query import find_latest_manifest_hpc, find_latest_manifest_s3

__all__ = [
    "DepthMode",
    "S3CacheManager",
    "TidalManifestQuery",
    "find_latest_manifest_hpc",
    "find_latest_manifest_s3",
    "tidal_hindcast",
]
