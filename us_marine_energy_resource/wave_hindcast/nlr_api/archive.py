"""Wait for, download, and organize the archives the download API builds.

Downloads are asynchronous: a request returns a ``downloadUrl`` that 403s
until the server has assembled the archive (it also emails the link), so
fetching polls that URL, downloads the zip, and unpacks it into tidy per-site
CSVs plus a ``metadata.json``.
"""

from __future__ import annotations

import csv
import io
import json
import re
import time
import zipfile
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...explore.lazy import lazy_import
from .. import _store, errors
from .._store import _noop
from ..domains import MATH_TO_MET, domain_config

if TYPE_CHECKING:
    import pandas as pd

# How often the wait re-checks an archive. Build time varies hugely by domain.
POLL_INTERVAL_S = 20

# How often the wait reports a sign of life. Kept well under the poll interval
# so a long build never looks like a hang.
HEARTBEAT_S = 10

# When the wait starts to look like a failed build. A failed build keeps 403ing
# forever and the only failure signal is an email, so once the wait outlasts a
# typical build it tells the user to go check.
SLOW_AFTER_S = 600


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    """Read the requests manifest, empty when it does not exist yet.

    Parameters
    ----------
    manifest_path : Path
        The manifest file.

    Returns
    -------
    dict
        The parsed manifest, empty when the file is missing.
    """
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    return {}


def save_manifest(manifest_path: Path, manifest: dict[str, Any]) -> None:
    """Write the requests manifest, creating its directory.

    Parameters
    ----------
    manifest_path : Path
        The manifest file.
    manifest : dict
        The manifest content.
    """
    _store.write_json(manifest_path, manifest)


def archive_ready(url: str) -> bool:
    """Check whether the server has finished assembling the archive.

    S3 answers 403 rather than 404 for an object that does not exist yet, so
    either means not ready.

    Parameters
    ----------
    url : str
        The archive's download URL.

    Returns
    -------
    bool
        True when the URL answers 200.
    """
    requests = lazy_import("requests", "polling the NLR wave archive")
    try:
        return requests.head(url, timeout=30).status_code == 200
    except requests.RequestException:
        return False


def wait_for_archive(
    name: str,
    url: str,
    timeout_s: int,
    on_event: Callable[[str], None] = _noop,
) -> bool:
    """Block until the archive is downloadable, or the timeout passes.

    Parameters
    ----------
    name : str
        Site label, for progress messages.
    url : str
        The archive's download URL.
    timeout_s : int
        Give up after this long. The request stays in the manifest, so a
        retry resumes rather than re-requesting.
    on_event : callable
        Sink for progress messages, including a regular heartbeat so a long
        build never looks like a hang.

    Returns
    -------
    bool
        True once ready, False on timeout.
    """
    started = time.monotonic()
    deadline = started + timeout_s
    escalated = False

    while time.monotonic() < deadline:
        if archive_ready(url):
            on_event(f"{name}: archive ready")
            return True
        if not escalated and time.monotonic() - started > SLOW_AFTER_S:
            escalated = True
            on_event(
                f"note: still waiting after {SLOW_AFTER_S // 60} minutes, longer than "
                "a typical build. Check your email: a failure notice means the server "
                "gave up, so stop this wait and rerun with --force"
            )
        # Sleep in heartbeat-sized slices so the wait keeps talking between
        # polls.
        poll_at = time.monotonic() + POLL_INTERVAL_S
        while time.monotonic() < poll_at and time.monotonic() < deadline:
            elapsed = time.monotonic() - started
            on_event(
                f"[{elapsed // 60:.0f}m{elapsed % 60:02.0f}s] waiting on {name}, "
                f"next check in {max(poll_at - time.monotonic(), 0):.0f}s"
            )
            time.sleep(min(HEARTBEAT_S, max(poll_at - time.monotonic(), 0.1)))
    return archive_ready(url)


def download_archive(url: str, dest: Path, on_event: Callable[[str], None] = _noop) -> None:
    """Stream the archive zip to ``dest``.

    Parameters
    ----------
    url : str
        The archive's download URL.
    dest : Path
        Zip file to write. Parent directories are created; the write is
        atomic (a ``.part`` file renamed on completion).
    on_event : callable
        Sink for progress messages.

    Raises
    ------
    DownloadError
        The URL did not answer 200.
    """
    requests = lazy_import("requests", "downloading the NLR wave archive")
    try:
        response = requests.get(url, stream=True, timeout=300)
    except requests.RequestException as exc:
        raise errors.DownloadError(f"archive download failed: {exc}", url=url) from exc
    if response.status_code != 200:
        raise errors.DownloadError(
            f"archive not available ({response.status_code} {response.reason})", url=url
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    with open(part, "wb") as handle:
        for chunk in response.iter_content(chunk_size=1 << 20):
            handle.write(chunk)
    part.rename(dest)
    on_event(f"wrote {dest.name} ({dest.stat().st_size:,} bytes)")


# --------------------------------------------------------------------------- #
# organizing
# --------------------------------------------------------------------------- #

# Filenames inside an archive look like 479519_44.57_-124.23_1979.csv.
MEMBER_RE = re.compile(r"(?P<gid>\d+)_(?P<lat>[-\d.]+)_(?P<lon>[-\d.]+)_(?P<year>\d{4})\.csv$")

# Columns holding a compass bearing, which a domain-level rotation applies to.
DIRECTION_COLUMNS = (
    "Maximum Energy Direction",
    "Mean Wave Direction",
    "Peak Wave Direction",
    "Direction Of Maximum Directionally Resolved Wave Power",
)


def _slug(text: str) -> str:
    """Turn a column label into a snake_case token usable in a filename.

    Parameters
    ----------
    text : str
        Column label.

    Returns
    -------
    str
        Lowercase token with runs of other characters collapsed to ``_``.
    """
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _read_member(archive: zipfile.ZipFile, member: str) -> tuple[dict[str, str], pd.DataFrame]:
    """Split one CSV member into (metadata dict, DataFrame).

    The two-line preamble before the real header is lifted into metadata.json
    so the emitted CSVs have a single ordinary header row.

    Parameters
    ----------
    archive : zipfile.ZipFile
        The open archive.
    member : str
        Name of the CSV member to read.

    Returns
    -------
    tuple of (dict, pandas.DataFrame)
        The preamble entries and the records.
    """
    pd = lazy_import("pandas", "organizing a wave hindcast archive")
    raw = archive.read(member).decode("utf-8")
    handle = io.StringIO(raw)
    keys = next(csv.reader(handle))
    values = next(csv.reader(handle))
    preamble = dict(zip(keys, values, strict=False))
    frame = pd.read_csv(io.StringIO(handle.read()))
    return preamble, frame


def _coerce(value: str | None) -> int | float | str | None:
    """Type a preamble value best-effort, minus python bytes reprs.

    Parameters
    ----------
    value : str or None
        Raw preamble value.

    Returns
    -------
    int, float, str, or None
        The value as a number when it parses as one, else the cleaned text.
    """
    if value is None:
        return None
    text = value.strip()
    if text.startswith("b'") and text.endswith("'"):
        text = text[2:-1]
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:  # noqa: PERF203 -- two casts, not a hot loop
            pass
    return text


def _site_for_archive(path: Path, manifest: dict[str, Any]) -> tuple[str | None, str | None]:
    """Match an archive back to the site it was requested for.

    Script-named archives say so in the filename. Hand-downloaded ones are
    named for the request hash, which also appears in the manifest's download
    URL, so try that next.

    Parameters
    ----------
    path : Path
        The archive zip.
    manifest : dict
        The requests manifest.

    Returns
    -------
    tuple of (str or None, str or None)
        The site name and how it was matched, both None when nothing matches.
    """
    if path.stem in manifest:
        return path.stem, "filename"
    for name, entry in manifest.items():
        if path.stem in (entry.get("download_url") or ""):
            return name, "request hash"
    return None, None


def apply_direction_transform(
    frame: pd.DataFrame, transform: str | None
) -> tuple[pd.DataFrame, list[str]]:
    """Rotate direction columns into the meteorological convention.

    Returns the frame and the list of columns changed, so the correction can
    be recorded in metadata.json rather than silently baked into the numbers.

    Parameters
    ----------
    frame : pandas.DataFrame
        One year's records.
    transform : str or None
        The domain's ``direction_transform``; only ``"270-x"`` exists.

    Returns
    -------
    (pandas.DataFrame, list of str)
        The (possibly rotated) frame and the columns changed.
    """
    if not transform:
        return frame, []
    changed = [c for c in frame.columns if c in DIRECTION_COLUMNS]
    for column in changed:
        if transform == MATH_TO_MET:
            frame[column] = (270 - frame[column]) % 360
        else:
            raise ValueError(f"Unknown direction_transform: {transform}")
    return frame, changed


def _add_timestamp(frame: pd.DataFrame) -> pd.DataFrame:
    """Prepend an ISO ``timestamp`` column, leaving the five integer parts in place.

    Parameters
    ----------
    frame : pandas.DataFrame
        One year's records.

    Returns
    -------
    pandas.DataFrame
        A copy with the ``timestamp`` column first.
    """
    pd = lazy_import("pandas", "organizing a wave hindcast archive")
    stamp = pd.to_datetime(frame[["Year", "Month", "Day", "Hour", "Minute"]], utc=True).dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    out = frame.copy()
    out.insert(0, "timestamp", stamp)
    return out


def organize_archive(
    path: Path,
    manifest: dict[str, Any],
    data_dir: Path,
    on_event: Callable[[str], None] = _noop,
) -> Path:
    """Unpack one archive into ``<site>_<lat>_<lon>/`` with all four layouts.

    Parameters
    ----------
    path : Path
        The downloaded zip.
    manifest : dict
        The requests manifest, for matching and metadata.
    data_dir : Path
        The wave cache root the site directory is created under.
    on_event : callable
        Sink for progress messages.

    Returns
    -------
    Path
        The organized site directory.

    Raises
    ------
    ArchiveCorruptError
        The zip holds no year CSVs.
    ArchiveUnmatchedError
        The zip cannot be matched to a requested site.
    """
    pd = lazy_import("pandas", "organizing a wave hindcast archive")

    with zipfile.ZipFile(path) as archive:
        members = sorted(m for m in archive.namelist() if MEMBER_RE.search(m))
        if not members:
            raise errors.ArchiveCorruptError(f"{path.name}: contains no year CSVs")

        first_preamble, _ = _read_member(archive, members[0])
        site, how = _site_for_archive(path, manifest)
        if site is None:
            raise errors.ArchiveUnmatchedError(f"{path.name}: does not match any requested site")

        first_match = MEMBER_RE.search(members[0])
        assert first_match is not None
        info = first_match.groupdict()
        entry = manifest.get(site, {})
        site_domain = entry.get("domain")
        node_lat = _coerce(first_preamble.get("Latitude"))
        node_lon = _coerce(first_preamble.get("Longitude"))
        stem = _store.site_stem(site, node_lat, node_lon)
        out_dir = data_dir / stem
        on_event(f"{path.name}: {site} (matched by {how}) -> {stem}/")

        transform = domain_config(site_domain)["direction_transform"] if site_domain else None
        frames: dict[str, pd.DataFrame] = {}
        rotated: list[str] = []
        for member in members:
            member_match = MEMBER_RE.search(member)
            assert member_match is not None
            year = member_match.group("year")
            _, frame = _read_member(archive, member)
            frame, rotated = apply_direction_transform(frame, transform)
            frames[year] = _add_timestamp(frame)
        if rotated:
            on_event(f"direction correction {transform} applied to: {', '.join(rotated)}")

    combined = pd.concat(frames.values(), ignore_index=True)
    time_parts = ("timestamp", "Year", "Month", "Day", "Hour", "Minute")
    data_cols = [c for c in combined.columns if c not in time_parts]

    for sub in ("by_year", "by_month", "by_variable"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    for year, frame in frames.items():
        frame.to_csv(out_dir / "by_year" / f"{stem}_{year}.csv", index=False)

    for (year, month), chunk in combined.groupby(["Year", "Month"], sort=True):
        chunk.to_csv(out_dir / "by_month" / f"{stem}_{year}-{month:02d}.csv", index=False)

    # One file per variable, spanning the whole record.
    for column in data_cols:
        combined[["timestamp", column]].to_csv(
            out_dir / "by_variable" / f"{stem}_{_slug(column)}.csv", index=False
        )

    years = sorted(frames)
    combined.to_csv(out_dir / _store.combined_csv_name(stem, years[0], years[-1]), index=False)

    # Everything the two stripped preamble lines carried, plus request context.
    scalar_keys = (
        "Source",
        "Location ID",
        "Jurisdiction",
        "Latitude",
        "Longitude",
        "Time Zone",
        "Local Time Zone",
        "Distance to Shore",
        "Water Depth",
        "Version",
    )
    metadata: dict[str, Any] = {
        "site": site,
        "domain": site_domain,
        "requested_lat": entry.get("requested_lat"),
        "requested_lon": entry.get("requested_lon"),
        "node_lat": node_lat,
        "node_lon": node_lon,
        "gid": _coerce(info["gid"]),
        "years": [years[0], years[-1]],
        "interval_minutes": (int(domain_config(site_domain)["interval"]) if site_domain else None),
        "rows": len(combined),
        "variables": data_cols,
        # The preamble's second line gives units for the timeseries columns.
        "units": {c: first_preamble.get(c) for c in data_cols},
        "direction_transform": transform,
        "direction_columns_corrected": rotated,
        "source_archive": path.name,
        "requested_at": entry.get("requested_at"),
        "organized_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata.update(
        {_slug(k): _coerce(first_preamble.get(k)) for k in scalar_keys if k in first_preamble}
    )
    _store.write_json(out_dir / _store.METADATA_FILENAME, metadata)

    on_event(
        f"{len(frames)} years, {len(combined):,} rows, "
        f"{len(data_cols)} variables -> {out_dir.name}/"
    )
    return out_dir
