"""Configuration for the explore package.

One place for the values that describe how the package talks to the world:
the S3 endpoints, the settings file, the completion cache, and the transfer
policy defaults a user can override. Tuning constants that shape one
algorithm stay next to the code they tune.

Stdlib-only at import and imports nothing from this package, so any module
can import it without cycles. The wave bucket comes from the wave hindcast
configuration, which is the same kind of leaf.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..wave_hindcast.config import CONFIG as _WAVE_CONFIG


@dataclass(frozen=True)
class ExploreConfig:
    """Settings shared across the explore package.

    Frozen so every module reads the same values and nothing mutates them at
    runtime.
    """

    # The public bucket and root prefix holding the tidal hindcast data. The
    # wave endpoint reuses the bucket the wave hindcast configuration owns.
    tidal_bucket: str = "marine-energy-data"
    tidal_prefix: str = "us-tidal/"

    # Extensions that mark a path as a data file rather than a directory.
    data_extensions: tuple[str, ...] = (".h5", ".hdf5", ".nc", ".nc4", ".parquet", ".pq")

    # Settings file under the user's home; policy defaults live in its
    # [explore] table.
    settings_filename: str = ".us_tidal.toml"

    # Transfer policy defaults, in megabytes. User visible and overridable
    # from the settings file.
    max_transfer_mb: int = 100
    max_memory_mb: int = 512
    max_download_mb: int = 500
    confirm_above_mb: int = 25

    # Object size above which a remote file gets a heads-up before it is read.
    large_remote_bytes: int = 1 << 30

    # How long a cached completion listing stays fresh. Covers a burst of
    # TAB presses while new uploads still appear soon.
    completion_cache_ttl_s: int = 300

    @property
    def endpoints(self) -> dict[str, tuple[str, str]]:
        """Return the endpoint map: name to bucket and root prefix.

        Returns
        -------
        dict of str to tuple of (str, str)
            Endpoint name mapped to its bucket and root prefix. The wave root
            is the bucket itself.
        """
        return {
            "tidal": (self.tidal_bucket, self.tidal_prefix),
            "wave": (_WAVE_CONFIG.s3_bucket, ""),
        }

    def settings_path(self) -> Path:
        """Return the settings file path under the user's home.

        Returns
        -------
        Path
            The settings file location.
        """
        return Path.home() / self.settings_filename

    def completion_cache_path(self) -> Path:
        """Return the on-disk cache file for shell completion listings.

        Returns
        -------
        Path
            The completion cache location.
        """
        return Path.home() / ".cache" / "mer" / "completion.json"


CONFIG = ExploreConfig()
