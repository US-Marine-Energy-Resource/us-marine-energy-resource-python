"""
Query library for tidal parquet partition manifests.

Provides DuckDB-backed spatial queries against the bundled geometry index, and
path reconstruction utilities for direct S3 parquet file access.

Features:
- DuckDB spatial indexing (ST_Contains / ST_Intersects) for exact mesh queries
- Support for manifest spec with versioning and self-documenting schema
- Path reconstruction for direct parquet file access
- Version resolution (latest version per location, or specific version)
- S3 cache integration for on-demand data file loading
"""

from __future__ import annotations

import contextlib
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .cache import S3CacheManager

from ._spatial import (
    _COORD_DECIMAL_PRECISION,
    _COORD_PRECISION_SCALE,
    AreaOutsideDomainError,
    OutsideDomainError,
    PointOutsideDomainError,
    TransectOutsideDomainError,
    find_faces_area,
    find_faces_line,
    find_faces_point,
)

# ---------------------------------------------------------------------------
# Manifest discovery utilities
# ---------------------------------------------------------------------------


def parse_semver(version_str: str) -> tuple[int, int, int]:
    """Parse a semantic version string into a comparable integer tuple."""
    m = re.match(r"(\d+)\.(\d+)\.(\d+)", version_str)
    if not m:
        raise ValueError(f"Invalid semver: {version_str!r}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def find_latest_manifest_hpc(base_path: str) -> Path | None:
    """Find the latest manifest file on the HPC filesystem.

    Parameters
    ----------
    base_path : str
        HPC dataset root (e.g. ``/projects/hindcastra/Tidal/datasets/…``).

    Returns
    -------
    Path or None
        Path to the newest ``manifest_*.json`` found, or ``None`` if not found.
    """
    manifests_dir = Path(base_path) / "manifest"
    if not manifests_dir.exists():
        return None

    version_dirs: list[tuple[Path, tuple[int, int, int]]] = []
    for d in manifests_dir.iterdir():
        if d.is_dir() and d.name.startswith("v"):
            with contextlib.suppress(ValueError):
                version_dirs.append((d, parse_semver(d.name[1:])))

    version_dirs.sort(key=lambda x: x[1], reverse=True)
    for version_dir, _ in version_dirs:
        manifest_files: list[tuple[Path, tuple[int, int, int]]] = []
        for f in version_dir.glob("manifest_*.json"):
            m = re.search(r"manifest_(\d+\.\d+\.\d+)\.json", f.name)
            if m:
                with contextlib.suppress(ValueError):
                    manifest_files.append((f, parse_semver(m.group(1))))
        manifest_files.sort(key=lambda x: x[1], reverse=True)
        if manifest_files:
            return manifest_files[0][0]
    return None


def find_latest_manifest_s3(
    s3_cache: S3CacheManager,
) -> tuple[Path, str] | None:
    """Find the latest manifest, checking the local cache before hitting S3.

    Parameters
    ----------
    s3_cache : S3CacheManager
        Configured cache manager pointing at the tidal dataset bucket.

    Returns
    -------
    tuple of (Path, str) or None
        ``(local path to cached manifest, version string)`` when found.
        Returns ``None`` when no manifest versions exist under the S3 prefix.
    """
    manifest_cache_dir = s3_cache.cache_dir / "manifest"
    if manifest_cache_dir.exists():
        version_dirs: list[tuple[Path, tuple[int, int, int]]] = []
        for d in manifest_cache_dir.iterdir():
            if d.is_dir() and d.name.startswith("v"):
                with contextlib.suppress(ValueError):
                    version_dirs.append((d, parse_semver(d.name[1:])))
        if version_dirs:
            version_dirs.sort(key=lambda x: x[1], reverse=True)
            latest_dir, latest_version_tuple = version_dirs[0]
            latest_version = ".".join(map(str, latest_version_tuple))
            manifest_file = latest_dir / f"manifest_{latest_version}.json"
            if manifest_file.exists():
                return manifest_file, latest_version

    manifest_prefix = f"{s3_cache.prefix}/manifest/"
    response = s3_cache.s3.list_objects_v2(
        Bucket=s3_cache.bucket,
        Prefix=manifest_prefix,
        Delimiter="/",
    )

    s3_version_dirs: list[tuple[str, tuple[int, int, int]]] = []
    for prefix_info in response.get("CommonPrefixes", []):
        dir_name = prefix_info["Prefix"].rstrip("/").split("/")[-1]  # pyright: ignore[reportTypedDictNotRequiredAccess]
        if dir_name.startswith("v"):
            with contextlib.suppress(ValueError):
                s3_version_dirs.append((dir_name, parse_semver(dir_name[1:])))

    if not s3_version_dirs:
        return None

    s3_version_dirs.sort(key=lambda x: x[1], reverse=True)
    latest_dir_name = s3_version_dirs[0][0]
    latest_version = ".".join(map(str, s3_version_dirs[0][1]))
    manifest_key = f"{manifest_prefix}{latest_dir_name}/manifest_{latest_version}.json"
    relative_path = manifest_key[len(s3_cache.prefix) + 1 :]
    local_path = s3_cache.get(relative_path)
    return local_path, latest_version


class TidalManifestQuery:
    """
    Query interface for tidal parquet partition manifests.

    Spatial query methods:
    - query_nearest_point: Find the mesh face containing a coordinate
    - query_all_within_rectangular_area: Find all faces inside a bounding box
    - query_all_on_line: Find all faces a transect passes through
    """

    def __init__(self, manifest_path: Path, s3_cache=None, verbose: bool = False):
        """
        Initialize the query interface by loading the manifest.

        Parameters
        ----------
        manifest_path : Path
            Path to ``manifest_{version}.json`` (spec v2.0.0 format).
        s3_cache : S3CacheManager, optional
            S3 cache manager for fetching data files on-demand.
        verbose : bool, optional
            If True, print manifest summary after loading.
        """
        self.manifest_path = Path(manifest_path)
        self.manifest_dir = self.manifest_path.parent
        self.grids_dir = self.manifest_dir / "grids"
        self.s3_cache = s3_cache
        self.verbose = verbose

        with open(self.manifest_path) as f:
            self.manifest = json.load(f)

        if "spec_version" not in self.manifest:
            raise ValueError(
                "Invalid manifest format: missing 'spec_version'. "
                "This query interface requires spec v2.0.0 or later."
            )

        self.spec_version = self.manifest["spec_version"]
        self.manifest_version = self.manifest["manifest_version"]
        self.dataset = self.manifest["dataset"]

        partition = self.manifest["partition"]
        self.decimal_places = partition["decimal_places"]
        self.grid_resolution_deg = partition["grid_resolution_deg"]
        self.data_level = partition["data_level"]
        self.index_max_digits = partition["index_max_digits"]

        self.path_template = self.manifest["path_template"]["template"]
        self.storage = self.manifest["storage"]

        self.total_grids = self.manifest["total_grids"]
        self.total_points = self.manifest["total_points"]

        if verbose:
            print(f"Loaded manifest (spec v{self.spec_version}, manifest v{self.manifest_version})")
            print(f"  Dataset: {self.dataset['label']}")
            print(f"  Total grids: {self.total_grids:,}")
            print(f"  Total points: {self.total_points:,}")
            print(f"  Grid resolution: {self.grid_resolution_deg}°")
            print(f"  Locations: {list(self.manifest['locations'].keys())}")
            print(f"  S3 base: {self.storage['s3_base_uri']}")

    # ---------------------------------------------------------------------------
    # Path reconstruction
    # ---------------------------------------------------------------------------

    def reconstruct_path(
        self,
        point: list[str],
        location_name: str,
        version: str | None = None,
    ) -> str:
        """
        Reconstruct the full parquet file path from point data.

        Parameters
        ----------
        point : list of str
            Point data as ``[lat_str, lon_str, face_id_str]``.
        location_name : str
            Location identifier (e.g., ``"AK_cook_inlet"``).
        version : str, optional
            Data version. If None, uses the location's ``latest_version``.

        Returns
        -------
        str
            Relative path to the parquet file (prefix-relative to S3 base URI).

        Examples
        --------
        >>> point = ["61.4657288", "-149.6356201", "002499"]
        >>> path = query.reconstruct_path(point, "AK_cook_inlet")
        """
        lat_str, lon_str, face_id_str = point
        lat = float(lat_str)
        lon = float(lon_str)

        # Re-format to the canonical precision used in S3 filenames.
        # str(float) silently drops trailing zeros (e.g. -122.5481720 → '-122.548172'),
        # which produces a path that doesn't match the actual S3 object key.
        lat_str = f"{lat:.{_COORD_DECIMAL_PRECISION}f}"
        lon_str = f"{lon:.{_COORD_DECIMAL_PRECISION}f}"

        multiplier = 10**self.decimal_places
        lat_deg = int(lat)
        lon_deg = int(lon)
        lat_dec = int(abs(lat * multiplier) % multiplier)
        lon_dec = int(abs(lon * multiplier) % multiplier)

        if location_name not in self.manifest["locations"]:
            raise ValueError(f"Unknown location: {location_name}")

        loc_meta = self.manifest["locations"][location_name]

        if version is None:
            version = loc_meta.get("latest_version", "1.0.0")

        face_id_padded = face_id_str.zfill(self.index_max_digits)

        return self.path_template.format(
            location=location_name,
            data_version=version,
            data_level=self.data_level,
            lat_deg=lat_deg,
            lon_deg=lon_deg,
            lat_dec=lat_dec,
            lon_dec=lon_dec,
            face_id=face_id_padded,
            lat=lat_str,
            lon=lon_str,
            temporal=loc_meta["temporal"],
            date=loc_meta["date"],
            time=loc_meta["time"],
        )

    # ---------------------------------------------------------------------------
    # Path helpers
    # ---------------------------------------------------------------------------

    def get_s3_uri(self, relative_path: str) -> str:
        """
        Convert a relative path to a full S3 URI.

        Parameters
        ----------
        relative_path : str
            Relative path from :meth:`reconstruct_path`.

        Returns
        -------
        str
            Full S3 URI (e.g., ``s3://marine-energy-data/us-tidal/AK_cook_inlet/...``).
        """
        if not self.storage:
            raise ValueError("Manifest does not contain storage configuration")
        return f"{self.storage['s3_base_uri']}/{relative_path}"

    def get_hpc_path(self, relative_path: str) -> str:
        """
        Convert a relative path to a full HPC filesystem path.

        Parameters
        ----------
        relative_path : str
            Relative path from :meth:`reconstruct_path`.

        Returns
        -------
        str
            Full HPC filesystem path.
        """
        if not self.storage.get("hpc_base_path"):
            raise ValueError("Manifest does not contain HPC base path configuration")
        return f"{self.storage['hpc_base_path']}/{relative_path}"

    def get_location_version_info(self, location_name: str) -> dict[str, Any]:
        """
        Get version information for a specific location.

        Parameters
        ----------
        location_name : str
            Location identifier (e.g., ``"AK_cook_inlet"``).

        Returns
        -------
        dict
            Keys: ``latest_version``, ``versions``.
        """
        if location_name not in self.manifest["locations"]:
            raise ValueError(f"Unknown location: {location_name}")

        loc_meta = self.manifest["locations"][location_name]
        return {
            "latest_version": loc_meta.get("latest_version", "unknown"),
            "versions": loc_meta.get("versions", {}),
        }

    # ---------------------------------------------------------------------------
    # Spatial queries — backed by DuckDB / bundled geometry parquets
    # ---------------------------------------------------------------------------

    def query_nearest_point(
        self,
        lat: float,
        lon: float,
    ) -> dict[str, Any] | None:
        """
        Find the mesh face containing (lat, lon).

        Returns the containing face (distance_km = 0.0).  If the point lies
        outside the mesh interior (e.g., on a boundary edge), the nearest face
        by exact ST_ClosestPoint distance is returned instead.

        Parameters
        ----------
        lat : float
            Query latitude in decimal degrees (WGS84).
        lon : float
            Query longitude in decimal degrees (WGS84).

        Returns
        -------
        dict or None
            Keys:

            - ``point`` : dict with ``face_id``, ``lat``, ``lon``, ``file_path``
            - ``distance_km`` : float — 0.0 for containing faces
            - ``location`` : str — dataset location name
            - ``grid_id`` : str — same as ``face_id``

            Returns None when (lat, lon) is outside all dataset domains.
        """
        try:
            df = find_faces_point(lat, lon)
        except PointOutsideDomainError:
            return None

        if df.empty:
            return None

        row = df.iloc[0]
        face_id = str(row["face_id"])
        lat_val = float(row["lat_fixed_precision"]) / _COORD_PRECISION_SCALE
        lon_val = float(row["lon_fixed_precision"]) / _COORD_PRECISION_SCALE
        location = str(row["location"])
        distance_km = float(row["distance_km"])

        file_path = self.reconstruct_path([str(lat_val), str(lon_val), face_id], location)

        return {
            "point": {
                "face_id": face_id,
                "lat": lat_val,
                "lon": lon_val,
                "file_path": file_path,
            },
            "distance_km": distance_km,
            "location": location,
            "grid_id": face_id,
        }

    def query_all_within_rectangular_area(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
    ) -> list[dict[str, Any]]:
        """
        Find all mesh faces that intersect a rectangular bounding box.

        Uses an exact ST_Intersects test against each triangle — no centroid
        approximation, no buffer.

        Parameters
        ----------
        lat_min, lat_max : float
            Latitude bounds in decimal degrees.
        lon_min, lon_max : float
            Longitude bounds in decimal degrees.

        Returns
        -------
        list of dict
            Each entry has: ``face_id``, ``centroid`` (lat, lon), ``location``,
            ``distance_km``, ``n_points`` (always 1).
            Returns ``[]`` when the bbox does not intersect any dataset domain.
        """
        return self.query_all_within_polygon(
            [
                (lat_min, lon_min),
                (lat_min, lon_max),
                (lat_max, lon_max),
                (lat_max, lon_min),
            ]
        )

    def query_all_on_line(
        self,
        start_lat: float,
        start_lon: float,
        end_lat: float,
        end_lon: float,
    ) -> list[dict[str, Any]]:
        """
        Find all mesh faces that the line segment (start → end) passes through.

        Only faces where the line geometrically intersects the triangle are
        returned.  For proximity queries (faces within a corridor around the
        line), build a buffered polygon and call
        :meth:`query_all_within_rectangular_area` instead.

        Parameters
        ----------
        start_lat, start_lon : float
            Start coordinate in decimal degrees (WGS84).
        end_lat, end_lon : float
            End coordinate in decimal degrees (WGS84).

        Returns
        -------
        list of dict
            Each entry has: ``face_id``, ``centroid`` (lat, lon), ``location``,
            ``frac_along`` (fractional position 0→1 along the line), ``n_points``
            (always 1).  Sorted by ``frac_along``.
            Returns ``[]`` when the line does not intersect any dataset domain.
        """
        return self.query_all_on_path([(start_lat, start_lon), (end_lat, end_lon)])

    def query_all_on_path(
        self,
        coords: list[tuple[float, float]],
    ) -> list[dict[str, Any]]:
        """Find all mesh faces intersected by a multi-vertex polyline.

        Parameters
        ----------
        coords : list of (lat, lon) tuples
            Polyline vertices in order (at least 2 points).

        Returns
        -------
        list of dict
            Each entry has: ``face_id``, ``centroid`` (lat, lon), ``location``,
            ``frac_along``, ``n_points`` (always 1).  Sorted by ``frac_along``.
            Returns ``[]`` when the line does not intersect any dataset domain.
        """
        try:
            df = find_faces_line(coords)
        except (TransectOutsideDomainError, OutsideDomainError):
            return []

        results: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            results.append(
                {
                    "face_id": str(row["face_id"]),
                    "centroid": (
                        float(row["lat_fixed_precision"]) / _COORD_PRECISION_SCALE,
                        float(row["lon_fixed_precision"]) / _COORD_PRECISION_SCALE,
                    ),
                    "location": str(row["location"]),
                    "frac_along": float(row["frac_along"]),
                    "n_points": 1,
                }
            )
        return results

    def query_all_within_polygon(
        self,
        coords: list[tuple[float, float]],
    ) -> list[dict[str, Any]]:
        """Find all mesh faces that intersect an arbitrary polygon.

        Parameters
        ----------
        coords : list of (lat, lon) tuples
            Ring coordinates defining the polygon.  At least 3 points required.
            Need not be closed (closing vertex is added automatically).

        Returns
        -------
        list of dict
            Each entry has: ``face_id``, ``centroid`` (lat, lon), ``location``,
            ``distance_km``, ``n_points`` (always 1).
            Returns ``[]`` when the polygon does not intersect any dataset domain.
        """
        try:
            df = find_faces_area(coords)
        except AreaOutsideDomainError:
            return []

        results: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            results.append(
                {
                    "face_id": str(row["face_id"]),
                    "centroid": (
                        float(row["lat_fixed_precision"]) / _COORD_PRECISION_SCALE,
                        float(row["lon_fixed_precision"]) / _COORD_PRECISION_SCALE,
                    ),
                    "location": str(row["location"]),
                    "distance_km": float(row["distance_km"]),
                    "n_points": 1,
                }
            )
        return results

    def get_file_path(self, face_result: dict[str, Any]) -> str:
        """Get the parquet file path for a multi-face query result dict.

        Parameters
        ----------
        face_result : dict
            A result dict from :meth:`query_all_on_path`, :meth:`query_all_on_line`,
            :meth:`query_all_within_rectangular_area`, or
            :meth:`query_all_within_polygon`.

        Returns
        -------
        str
            Relative parquet file path (prefix-relative to S3 base URI).
        """
        lat, lon = face_result["centroid"]
        return self.reconstruct_path(
            [str(lat), str(lon), face_result["face_id"]],
            face_result["location"],
        )
