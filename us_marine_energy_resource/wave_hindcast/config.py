"""Configuration for the wave hindcast package.

One place for the facts a user or operator might care about: service URLs,
environment variable names, the cache layout on disk, the wave node index
layout, and the default archive timeout. Tuning constants that shape one algorithm stay next to the code they
tune. Stdlib-only at import, so any module can import it without cycles.
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

    # The wave node index. The repository (as "owner/name") is where the
    # published index files download from, the prefix names every index file
    # and directory, and the version tags one published generation of them.
    github_repo: str = "US-Marine-Energy-Resource/us-marine-energy-resource-python"
    index_prefix: str = "h2o_wave_hindcast_index"
    index_version: str = "v1"

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
    def index_subdir(self) -> str:
        """Return the relative directory holding one generation of the index.

        The same layout is used in the repository (under ``data/``), in the
        download URL, and in the local cache.

        Returns
        -------
        str
            ``{index_prefix}/{index_version}`` as a relative POSIX path.
        """
        return f"{self.index_prefix}/{self.index_version}"

    @property
    def index_base_url(self) -> str:
        """Return the default download URL for the wave node index files.

        GitHub serves the actual LFS content (not the pointer) at this host
        for public repositories. The registry checksums make repointing it
        at another host safe. The environment variable named by
        ``index_url_env`` overrides it at call time.

        Returns
        -------
        str
            Base URL the index files are downloaded from.
        """
        return (
            f"https://media.githubusercontent.com/media/{self.github_repo}/"
            f"main/data/{self.index_subdir}/"
        )

    @property
    def package_data_dir(self) -> Path:
        """Return the package's own ``data/`` directory.

        Returns
        -------
        Path
            Directory holding the data files that ship inside the package.
        """
        return Path(__file__).resolve().parent.parent / "data"

    @property
    def index_registry_file(self) -> Path:
        """Return the packaged checksum registry for the index files.

        Returns
        -------
        Path
            The ``{index_prefix}_registry_{index_version}.txt`` file inside
            the package.
        """
        return self.package_data_dir / f"{self.index_prefix}_registry_{self.index_version}.txt"

    @property
    def index_file(self) -> Path:
        """Return the packaged index description file.

        Returns
        -------
        Path
            The ``{index_prefix}_{index_version}.json`` file inside the
            package.
        """
        return self.package_data_dir / f"{self.index_prefix}_{self.index_version}.json"

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
