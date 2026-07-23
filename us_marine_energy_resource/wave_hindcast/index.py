"""Locate the wave grid-node index files.

The small pieces of the index ship inside the package, so resolving which
domain covers a point never touches the network. The six per-domain node
parquets do not fit in a wheel, so they live in Git LFS and are fetched once
and cached. A file resolves in order through the ``MER_WAVE_INDEX_DIR``
override, the package's own ``data/`` directory, a repo checkout (skipping
Git LFS pointer files), the local cache, a checksum-verified download, and
as a last resort generation from the source .h5 on S3. Point
``MER_WAVE_INDEX_URL`` at another host to override the download location.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any

from ..explore.lazy import lazy_import
from . import errors
from .config import CONFIG

# A clone without git-lfs installed checks out the pointer text instead of the
# parquet, and feeding that to a reader produces a baffling parse error, so sniff.
_LFS_POINTER_PREFIX = b"version https://git-lfs"

_state: dict[str, Any] = {}


def load_index() -> dict[str, Any]:
    """Return the packaged index description (node files, coord scale, bounds).

    Returns
    -------
    dict
        Parsed contents of the packaged index JSON.

    Raises
    ------
    IndexMissingError
        The packaged JSON is missing (a broken installation).
    """
    if "index" not in _state:
        if not CONFIG.index_file.exists():
            raise errors.IndexMissingError(
                f"packaged index description missing: {CONFIG.index_file}"
            )
        _state["index"] = json.loads(CONFIG.index_file.read_text())
    return _state["index"]


def registry() -> dict[str, str]:
    """``{filename: sha256}`` for the downloadable index files.

    Returns
    -------
    dict
        Empty if the packaged registry file is absent.
    """
    if "registry" not in _state:
        entries: dict[str, str] = {}
        registry_file = CONFIG.index_registry_file
        if registry_file.exists():
            for line in registry_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    name, checksum = line.split()
                    entries[name] = checksum
        _state["registry"] = entries
    return _state["registry"]


def cache_dir() -> Path:
    """Return the local cache directory for fetched or generated index files.

    Returns
    -------
    Path
        Cache directory for this index version.
    """
    pooch = lazy_import("pooch", "locating the wave node index cache")
    return Path(pooch.os_cache("us-marine-energy-resource")) / CONFIG.index_subdir


def _is_real_file(path: Path) -> bool:
    """Report whether ``path`` exists and is real content, not a Git LFS pointer.

    Parameters
    ----------
    path : Path
        File to check.

    Returns
    -------
    bool
        True when the file exists and holds real content.
    """
    if not path.is_file():
        return False
    with open(path, "rb") as handle:
        return not handle.read(len(_LFS_POINTER_PREFIX)).startswith(_LFS_POINTER_PREFIX)


def _checkout_path(filename: str) -> Path | None:
    """Find the file inside a repo checkout's LFS directory, if usable.

    Parameters
    ----------
    filename : str
        Index file name to look for.

    Returns
    -------
    Path or None
        Path to the checked out file, or None when it is absent or a pointer.
    """
    # wave/index.py -> us_marine_energy_resource -> repo root. In an installed
    # package the directory simply does not exist and this returns None.
    repo_root = Path(__file__).resolve().parent.parent.parent
    candidate = repo_root / "data" / CONFIG.index_subdir / filename
    return candidate if _is_real_file(candidate) else None


def _fetch(filename: str, dest_dir: Path) -> Path:
    """Download one index file into ``dest_dir``, verifying its checksum.

    Parameters
    ----------
    filename : str
        Index file name to download.
    dest_dir : Path
        Directory the file is saved into.

    Returns
    -------
    Path
        Path to the downloaded file.
    """
    pooch = lazy_import("pooch", "downloading the wave node index")
    base_url = os.environ.get(CONFIG.index_url_env, CONFIG.index_base_url)
    fetched = pooch.retrieve(
        url=base_url.rstrip("/") + "/" + filename,
        known_hash=registry().get(filename),
        fname=filename,
        path=dest_dir,
        progressbar=False,
    )
    return Path(fetched)


def _generate(filename: str, dest_dir: Path) -> Path:
    """Build a node parquet from S3 as the last resort, marking it generated.

    Parameters
    ----------
    filename : str
        Node parquet file name to build.
    dest_dir : Path
        Directory the file is written into.

    Returns
    -------
    Path
        Path to the generated file.
    """
    index = load_index()
    domain_by_file = {name: domain for domain, name in index["node_files"].items()}
    if filename not in domain_by_file:
        raise errors.IndexMissingError(
            f"only the per-domain node files can be generated, and {filename} is not one of them"
        )
    domain = domain_by_file[filename]

    warnings.warn(
        f"wave node index {filename} could not be downloaded, so it is being built "
        "from the source data on S3 instead. This happens once.",
        stacklevel=3,
    )
    from . import index_build

    dest = dest_dir / filename
    index_build.build_domain_nodes(domain, dest, coord_scale=index["coord_scale"])
    # The sentinel records that this file did not come from the published
    # registry. Delete the parquet (or the whole cache dir) to retry a fetch.
    (dest_dir / f"{filename}.generated").touch()
    return dest


def data_path(filename: str) -> Path:
    """Absolute path to one index file, fetching or generating it if necessary.

    Parameters
    ----------
    filename : str
        Index file name, e.g. ``nodes_West_Coast_v1.parquet``.

    Returns
    -------
    Path
        A real local file (never an LFS pointer).

    Raises
    ------
    IndexMissingError
        Every resolution step failed; the message says which were tried.
    """
    override = os.environ.get(CONFIG.index_dir_env)
    if override:
        path = Path(override) / filename
        if _is_real_file(path):
            return path
        raise errors.IndexMissingError(
            f"{path} does not exist, but {CONFIG.index_dir_env} says it should"
        )

    bundled = CONFIG.package_data_dir / filename
    if _is_real_file(bundled):
        return bundled

    checkout = _checkout_path(filename)
    if checkout is not None:
        return checkout

    cache = cache_dir()
    cached = cache / filename
    if _is_real_file(cached):
        return cached

    try:
        return _fetch(filename, cache)
    except Exception as fetch_exc:
        try:
            return _generate(filename, cache)
        except errors.IndexMissingError:
            raise
        except Exception as gen_exc:
            raise errors.IndexMissingError(
                f"{filename} could not be found in {CONFIG.index_dir_env}, the package, "
                f"a repo checkout, or {cache}. The download failed ({fetch_exc}) "
                f"and building it from S3 also failed ({gen_exc})."
            ) from gen_exc
