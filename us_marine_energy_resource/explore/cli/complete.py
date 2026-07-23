"""Shell completion for path arguments.

Completes endpoint names (``tidal``, ``wave``), their S3 children one level at
a time, ``s3://`` prefixes, and local files. Wired to the shared ``UriArg`` in
:mod:`.options`, so ``mer ls``/``info``/``explore``/``download`` all get it
once shell completion is installed (``mer --install-completion``).

A completion callback runs on every TAB press in a fresh process, so it must
never raise, never print, and never hang: remote listings use a short-timeout
client and any failure completes to nothing. Results are also cached on disk
for a short while, which makes repeat presses instant and gives a failed
press a second chance to be answered from the cache.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..catalog import ENDPOINTS
from ..config import CONFIG

# Where listing results are remembered between TAB presses. Each press runs a
# new process, so an in-memory cache would remember nothing.
_CACHE_FILE = CONFIG.completion_cache_path()

# How long a cached listing stays fresh.
_CACHE_TTL_S = CONFIG.completion_cache_ttl_s


def _s3_children(bucket: str, prefix: str) -> tuple[list[str], list[str]]:
    """List one delimited level under a prefix, tuned for interactive latency.

    Parameters
    ----------
    bucket : str
        Bucket to list.
    prefix : str
        Key prefix ending in a slash, or empty for the bucket root.

    Returns
    -------
    (list of str, list of str)
        Directory prefixes and file keys, both as full keys from the bucket
        root.
    """
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    client = boto3.client(
        "s3",
        config=Config(
            signature_version=UNSIGNED,
            connect_timeout=2,
            read_timeout=3,
            retries={"max_attempts": 2},
        ),
    )
    page = client.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter="/", MaxKeys=200)
    dirs = [d for c in page.get("CommonPrefixes", []) if (d := c.get("Prefix"))]
    keys = [k for o in page.get("Contents", []) if (k := o.get("Key")) and k != prefix]
    return dirs, keys


def _cached_children(bucket: str, prefix: str) -> tuple[list[str], list[str]]:
    """Serve a listing from the on-disk cache, refreshing it when stale.

    Cache trouble of any kind (corrupt file, unwritable directory) falls back
    to a plain listing, so the cache can only ever make completion better.

    Parameters
    ----------
    bucket : str
        Bucket to list.
    prefix : str
        Key prefix ending in a slash, or empty for the bucket root.

    Returns
    -------
    (list of str, list of str)
        Directory prefixes and file keys, both as full keys from the bucket
        root.
    """
    key = f"{bucket}/{prefix}"
    now = time.time()
    cache: dict[str, dict] = {}
    try:
        cache = json.loads(_CACHE_FILE.read_text())
        entry = cache.get(key)
        if entry and now - entry["at"] < _CACHE_TTL_S:
            return entry["dirs"], entry["keys"]
    except Exception:
        cache = {}

    dirs, keys = _s3_children(bucket, prefix)

    try:
        import os

        cache = {k: v for k, v in cache.items() if now - v.get("at", 0) < _CACHE_TTL_S}
        cache[key] = {"at": now, "dirs": dirs, "keys": keys}
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        # Per-process temp file, since the prefetcher and a TAB press can
        # write at once and replace() is atomic.
        tmp = _CACHE_FILE.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(cache))
        tmp.replace(_CACHE_FILE)
    except Exception:
        pass
    return dirs, keys


# How many candidate directories to warm ahead of the next TAB press. Bounds
# the background work when a level has very many children.
_PREFETCH_LIMIT = 12


def _spawn_prefetch(bucket: str, prefixes: list[str]) -> None:
    """Warm the cache for likely next levels, in a detached background process.

    The completion process itself must exit immediately (the shell waits on
    its stdout), so the fetching happens in a child with every stream pointed
    at devnull. Prefixes already fresh in the cache are dropped first, so a
    fully warmed path spawns nothing.

    Parameters
    ----------
    bucket : str
        Bucket the prefixes live in.
    prefixes : list of str
        Key prefixes to warm.
    """
    try:
        fresh: dict[str, dict] = {}
        try:
            cache = json.loads(_CACHE_FILE.read_text())
            now = time.time()
            fresh = {k: v for k, v in cache.items() if now - v.get("at", 0) < _CACHE_TTL_S}
        except Exception:
            pass
        todo = [p for p in prefixes if f"{bucket}/{p}" not in fresh][:_PREFETCH_LIMIT]
        if not todo:
            return

        import subprocess
        import sys

        subprocess.Popen(
            [sys.executable, "-m", "us_marine_energy_resource.explore.cli.complete", bucket, *todo],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception:  # prefetching is an optimization, never a failure
        pass


def _complete_endpoint_child(incomplete: str) -> list[str]:
    """Complete ``endpoint/partial`` from a one-level S3 listing.

    Parameters
    ----------
    incomplete : str
        Text typed so far, starting with an endpoint name and a slash.

    Returns
    -------
    list of str
        Candidate completions.
    """
    head, _, tail = incomplete.partition("/")
    bucket, root = ENDPOINTS[head]
    parent = tail.rsplit("/", 1)[0] + "/" if "/" in tail else ""
    dirs, keys = _cached_children(bucket, root + parent)
    hit_dirs = [d for d in dirs if f"{head}/{d[len(root) :]}".startswith(incomplete)]
    _spawn_prefetch(bucket, hit_dirs)
    options = [f"{head}/{d[len(root) :]}" for d in hit_dirs]
    options += [o for k in keys if (o := f"{head}/{k[len(root) :]}").startswith(incomplete)]
    return sorted(options)


def _complete_s3_uri(incomplete: str) -> list[str]:
    """Complete an ``s3://bucket/...`` prefix one level at a time.

    Parameters
    ----------
    incomplete : str
        Text typed so far, starting with ``s3://``.

    Returns
    -------
    list of str
        Candidate completions.
    """
    rest = incomplete[len("s3://") :]
    if "/" not in rest:
        # Bucket names cannot be listed anonymously.
        return []
    bucket, _, key = rest.partition("/")
    parent = key.rsplit("/", 1)[0] + "/" if "/" in key else ""
    dirs, keys = _cached_children(bucket, parent)
    hit_dirs = [d for d in dirs if f"s3://{bucket}/{d}".startswith(incomplete)]
    _spawn_prefetch(bucket, hit_dirs)
    options = [f"s3://{bucket}/{d}" for d in hit_dirs]
    options += [o for k in keys if (o := f"s3://{bucket}/{k}").startswith(incomplete)]
    return sorted(options)


def _complete_local(incomplete: str) -> list[str]:
    """Complete local paths, marking directories with a trailing slash.

    Parameters
    ----------
    incomplete : str
        Text typed so far.

    Returns
    -------
    list of str
        Candidate completions.
    """
    directory, _, stem = incomplete.rpartition("/")
    root = Path(directory).expanduser() if directory else Path()
    if not root.is_dir():
        return []
    options = []
    for child in root.iterdir():
        if not child.name.startswith(stem) or child.name.startswith("."):
            continue
        text = f"{directory}/{child.name}" if directory else child.name
        options.append(text + "/" if child.is_dir() else text)
    return sorted(options)


def complete_path(incomplete: str) -> list[str]:
    """Complete one path argument for a TAB press.

    Parameters
    ----------
    incomplete : str
        Whatever the user has typed so far.

    Returns
    -------
    list of str
        Candidate completions. Directories end in ``/`` so the next TAB
        drills further.
    """
    try:
        if incomplete.startswith("s3://"):
            return _complete_s3_uri(incomplete)
        head = incomplete.partition("/")[0]
        if head in ENDPOINTS and (incomplete == head or "/" in incomplete):
            if "/" in incomplete:
                return _complete_endpoint_child(incomplete)
            bucket, root = ENDPOINTS[head]
            _spawn_prefetch(bucket, [root])
            return [head + "/"]
        matches = sorted(name for name in ENDPOINTS if name.startswith(incomplete))
        for name in matches:
            bucket, root = ENDPOINTS[name]
            _spawn_prefetch(bucket, [root])
        return [name + "/" for name in matches] + _complete_local(incomplete)
    except Exception:  # a completion must never raise, print, or hang
        return []


def _prefetch_main(argv: list[str]) -> None:
    """Fetch the given prefixes into the cache. Run detached by _spawn_prefetch.

    Parameters
    ----------
    argv : list of str
        Bucket name followed by the prefixes to fetch.
    """
    import contextlib

    bucket, *prefixes = argv
    for prefix in prefixes:
        # One unreachable level must not stop the rest from being warmed.
        with contextlib.suppress(Exception):
            _cached_children(bucket, prefix)


if __name__ == "__main__":
    import sys

    _prefetch_main(sys.argv[1:])
