"""
S3 Cache Manager with ETag-based cache validation.

Downloads S3 objects on first access, caches them locally, and uses ETags
to detect when objects change so stale files are re-downloaded automatically.
Cached files are stored in a directory tree mirroring the S3 key structure.

Usage::

    cache = S3CacheManager(bucket="marine-energy-data", prefix="us-tidal")

    # Downloads on first call; ETag-validated on every subsequent call.
    local_path = cache.get("manifest/v1.0.0/manifest_1.0.0.json")
"""

import contextlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError


class S3CacheManager:
    """
    Local cache for S3 objects with ETag-based freshness checking.

    Files are stored in a directory tree that mirrors the S3 key structure
    under ``cache_dir``.  On every access the cached file's ETag is compared
    against S3; the file is re-downloaded only when the ETag differs.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str,
        cache_dir: Path | None = None,
        aws_profile: str | None = None,
    ):
        """
        Initialize S3 cache manager.

        Parameters
        ----------
        bucket : str
            S3 bucket name
        prefix : str
            S3 prefix (e.g., 'us-tidal')
        cache_dir : Path, optional
            Local cache directory. Defaults to ./us_tidal_cache
        aws_profile : str, optional
            AWS profile name for S3 access
        """
        self.bucket = bucket
        self.prefix = prefix
        self.cache_dir = cache_dir or Path("./us_tidal_cache")
        self.aws_profile = aws_profile

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # ETag cache file (stores ETags for all cached files)
        self.etag_cache_file = self.cache_dir / ".etag_cache.json"
        self.etag_cache = self._load_etag_cache()

        # Initialize S3 client
        self._s3_client = None

        # Lock protecting etag_cache dict and .etag_cache.json writes
        self._etag_lock = threading.Lock()

    @property
    def s3(self):
        """Lazy-load S3 client.

        Uses the named AWS profile when ``aws_profile`` is set.  Otherwise
        uses anonymous (unsigned) access, which is correct for public buckets
        like ``marine-energy-data`` and avoids failures caused by stale
        credentials in the environment.
        """
        if self._s3_client is None:
            if self.aws_profile:
                session = boto3.Session(profile_name=self.aws_profile)
                self._s3_client = session.client("s3")
            else:
                from botocore import UNSIGNED
                from botocore.config import Config

                self._s3_client = boto3.client(
                    "s3",
                    config=Config(signature_version=UNSIGNED),
                )
        return self._s3_client

    def _load_etag_cache(self) -> dict[str, str]:
        """Load ETag cache from disk."""
        if self.etag_cache_file.exists():
            try:
                with open(self.etag_cache_file) as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError):
                return {}
        return {}

    def _save_etag_cache(self):
        """Save ETag cache to disk."""
        with open(self.etag_cache_file, "w") as f:
            json.dump(self.etag_cache, f, indent=2)

    def _get_s3_key(self, relative_path: str) -> str:
        """Convert relative path to full S3 key."""
        return f"{self.prefix}/{relative_path}"

    def _get_local_path(self, relative_path: str) -> Path:
        """Convert relative path to local cache path."""
        return self.cache_dir / relative_path

    def _get_s3_etag(self, s3_key: str) -> str | None:
        """
        Get ETag for an S3 object via ``HeadObject``.

        Returns
        -------
        str or None
            The ETag string, or ``None`` if the object does not exist (404)
            or the check is inaccessible (403 / network error), in which case
            the caller should trust the cached file.
        """
        try:
            response = self.s3.head_object(Bucket=self.bucket, Key=s3_key)
            return response.get("ETag", "").strip('"')
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("404", "NoSuchKey", "403", "AccessDenied"):
                return None
            raise
        except Exception:
            return None

    def _get_local_etag(self, relative_path: str) -> str | None:
        """Get cached ETag for a relative path."""
        cache_key = f"{self.bucket}/{self.prefix}/{relative_path}"
        return self.etag_cache.get(cache_key)

    def _set_local_etag(self, relative_path: str, etag: str):
        """Store ETag for a relative path (thread-safe)."""
        cache_key = f"{self.bucket}/{self.prefix}/{relative_path}"
        with self._etag_lock:
            self.etag_cache[cache_key] = etag
            self._save_etag_cache()

    def is_cached(self, relative_path: str) -> bool:
        """
        Check if a file is cached locally.

        Parameters
        ----------
        relative_path : str
            Path relative to S3 prefix

        Returns
        -------
        bool
            True if file exists in cache
        """
        local_path = self._get_local_path(relative_path)
        return local_path.exists()

    def is_valid(self, relative_path: str) -> bool:
        """
        Check if cached file matches S3 ETag.

        Parameters
        ----------
        relative_path : str
            Path relative to S3 prefix

        Returns
        -------
        bool
            True if cached file exists and ETag matches S3
        """
        if not self.is_cached(relative_path):
            return False

        s3_key = self._get_s3_key(relative_path)
        s3_etag = self._get_s3_etag(s3_key)
        local_etag = self._get_local_etag(relative_path)

        return s3_etag is not None and s3_etag == local_etag

    def get(
        self,
        relative_path: str,
        force_download: bool = False,
    ) -> Path:
        """
        Get a file from cache, downloading from S3 if necessary.

        If the file is cached locally, its ETag is checked against S3 before
        use.  Re-downloads automatically if the object has changed.  Pass
        ``force_download=True`` to skip the cache entirely.

        Parameters
        ----------
        relative_path : str
            Path relative to S3 prefix (e.g., "manifest/v1.0.0/manifest_1.0.0.json")
        force_download : bool, default False
            If True, always download from S3 even if cached.

        Returns
        -------
        Path
            Local path to the cached file

        Raises
        ------
        FileNotFoundError
            If file doesn't exist on S3
        """
        local_path = self._get_local_path(relative_path)
        s3_key = self._get_s3_key(relative_path)

        need_download = force_download or not local_path.exists()

        if not need_download:
            # Check ETag to detect stale cached files.
            # If HeadObject is inaccessible (403 / network error), s3_etag is
            # None — trust the cached file rather than failing.
            s3_etag = self._get_s3_etag(s3_key)
            if s3_etag is not None:
                local_etag = self._get_local_etag(relative_path)
                if s3_etag != local_etag:
                    need_download = True

        if need_download:
            local_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                self.s3.download_file(self.bucket, s3_key, str(local_path))
            except ClientError as e:
                if e.response.get("Error", {}).get("Code") == "404":
                    raise FileNotFoundError(
                        f"S3 object not found: s3://{self.bucket}/{s3_key}"
                    ) from e
                raise

            # Store ETag so next call can detect changes.
            s3_etag = self._get_s3_etag(s3_key)
            if s3_etag:
                self._set_local_etag(relative_path, s3_etag)

        return local_path

    def get_many(
        self,
        relative_paths: list[str],
        max_workers: int = 4,
        force_download: bool = False,
    ) -> dict[str, Path]:
        """
        Download multiple S3 objects in parallel.

        Uses a ``ThreadPoolExecutor`` to fetch up to *max_workers* files
        concurrently.  Already-cached files benefit from parallel ETag checks;
        missing files are downloaded concurrently.  The ETag cache is
        protected by a lock so concurrent writes are safe.

        Parameters
        ----------
        relative_paths : list[str]
            Paths relative to the S3 prefix to fetch.
        max_workers : int, default 4
            Number of parallel download threads.
        force_download : bool, default False
            If True, re-download every file even if cached.

        Returns
        -------
        dict[str, Path]
            Mapping of relative_path → local ``Path`` for every successfully
            fetched file.  Paths that raise ``FileNotFoundError`` are omitted;
            all other exceptions are re-raised.

        Raises
        ------
        Exception
            Any error from ``get()`` other than ``FileNotFoundError``.
        """
        results: dict[str, Path] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_path = {pool.submit(self.get, p, force_download): p for p in relative_paths}
            for future in as_completed(future_to_path):
                rel = future_to_path[future]
                with contextlib.suppress(FileNotFoundError):
                    results[rel] = future.result()
        return results

    def get_json(
        self,
        relative_path: str,
        force_download: bool = False,
    ) -> dict[str, Any]:
        """
        Get a JSON file from cache and parse it.

        Parameters
        ----------
        relative_path : str
            Path relative to S3 prefix
        force_download : bool, default False
            If True, always download from S3

        Returns
        -------
        dict
            Parsed JSON content
        """
        local_path = self.get(relative_path, force_download)
        with open(local_path) as f:
            return json.load(f)

    def clear_cache(self, relative_path: str | None = None):
        """
        Clear cached files.

        Parameters
        ----------
        relative_path : str, optional
            If specified, clear only this file. Otherwise clear entire cache.
        """
        if relative_path:
            local_path = self._get_local_path(relative_path)
            if local_path.exists():
                local_path.unlink()

            cache_key = f"{self.bucket}/{self.prefix}/{relative_path}"
            if cache_key in self.etag_cache:
                del self.etag_cache[cache_key]
                self._save_etag_cache()
        else:
            # Clear entire cache
            import shutil

            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.etag_cache = {}
            self._save_etag_cache()

    def cache_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns
        -------
        dict
            Cache statistics including file count, size, etc.
        """
        total_files = 0
        total_size = 0

        for root, _dirs, files in os.walk(self.cache_dir):
            for f in files:
                if f.startswith("."):
                    continue
                total_files += 1
                total_size += (Path(root) / f).stat().st_size

        return {
            "cache_dir": str(self.cache_dir),
            "total_files": total_files,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "etag_entries": len(self.etag_cache),
        }
