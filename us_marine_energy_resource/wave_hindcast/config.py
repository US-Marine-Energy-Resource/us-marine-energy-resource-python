"""Configuration for the wave hindcast package.

One place for the values that describe how the package talks to the world:
service URLs, environment variable names, the cache layout on disk, and the
default archive timeout. Everything here is a fact a user or operator might
care about. Tuning constants that shape one algorithm stay next to the code
they tune.

Stdlib-only at import and imports nothing from the package, so any module can
import it without cycles.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WaveHindcastConfig:
    """Settings shared across the wave hindcast package.

    Frozen so every module reads the same values and nothing mutates them at
    runtime.
    """

    # NLR developer download API.
    api_base_url: str = "https://developer.nlr.gov/api/wave/v2/wave"
    signup_url: str = "https://developer.nlr.gov/signup/"
    support_email: str = "marineresource@nlr.gov"

    # The public bucket holding the published .h5 files. Bare name here;
    # use s3_bucket_uri for the s3:// form.
    s3_bucket: str = "wpto-pds-us-wave"

    # Environment variable names. Values are read at call time, never here.
    api_key_env: str = "NLR_DEVELOPER_API_KEY"
    email_env: str = "NLR_DEVELOPER_EMAIL"
    cache_dir_env: str = "MER_WAVE_CACHE_DIR"
    index_dir_env: str = "MER_WAVE_INDEX_DIR"
    index_url_env: str = "MER_WAVE_INDEX_URL"

    # Layout of the download cache root.
    cache_dir_name: str = ".mer_wave_cache"
    manifest_filename: str = "requests.json"
    archives_dirname: str = "archives"
    chunks_dirname: str = "s3_chunks"

    # Default ceiling on the archive wait. Giving up early loses the wait,
    # not the archive (the URL stays in the manifest), so a retry resumes.
    default_timeout_s: int = 7200

    @property
    def s3_bucket_uri(self) -> str:
        """Return the bucket as an ``s3://`` URI.

        Returns
        -------
        str
            The bucket name with the ``s3://`` scheme.
        """
        return f"s3://{self.s3_bucket}"

    @property
    def non_site_dirnames(self) -> frozenset[str]:
        """Return the cache root directories that are not site data.

        Returns
        -------
        frozenset of str
            Directory names to skip when scanning the cache for sites.
        """
        return frozenset({self.archives_dirname, self.chunks_dirname})

    def default_cache_dir(self) -> Path:
        """Return the wave download cache root.

        The environment variable named by ``cache_dir_env`` overrides the
        default directory under the user's home.

        Returns
        -------
        Path
            The cache root directory.
        """
        override = os.environ.get(self.cache_dir_env)
        return Path(override) if override else Path.home() / self.cache_dir_name


CONFIG = WaveHindcastConfig()
