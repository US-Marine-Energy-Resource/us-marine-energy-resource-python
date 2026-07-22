"""The backend that carries one grid node through the download API."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .. import errors
from ..backend import BackendInfo
from ..config import CONFIG
from ..domains import domain_config
from ..nodes import WaveNode
from . import archive, client


class ApiBackend:
    """Fetch a grid node's record through the NLR developer download API."""

    def describe(self, node: WaveNode) -> BackendInfo:
        """Report what the download API serves for the node's domain.

        Parameters
        ----------
        node : WaveNode
            The resolved grid node.

        Returns
        -------
        BackendInfo
            Year range, interval, and direction handling.
        """
        config = domain_config(node.domain)
        return BackendInfo(
            endpoint=config["endpoint"],
            first_year=config["first_year"],
            last_year=config["last_year"],
            interval_minutes=int(config["interval"]),
            direction_transform=config["direction_transform"],
        )

    def fetch(
        self,
        node: WaveNode,
        name: str,
        *,
        requested_lat: float,
        requested_lon: float,
        force: bool,
        timeout_s: int,
        cache_dir: Path,
        on_event: Callable[[str], None],
        years: list[int] | None = None,
        variables: list[str] | None = None,
    ) -> None:
        """Carry one node from request to organized CSVs under ``cache_dir``.

        Blocks while the server builds the archive.

        Parameters
        ----------
        node : WaveNode
            The resolved grid node.
        name : str
            Site label; names the archive and the organized directory.
        requested_lat, requested_lon : float
            The coordinate originally asked for, kept in the metadata.
        force : bool
            Re-request and re-download even when already on disk.
        timeout_s : int
            Ceiling on the archive wait.
        cache_dir : Path
            The wave cache root.
        on_event : callable
            Sink for progress messages.
        years, variables : list, optional
            Narrow the request. ``None`` means everything the domain serves.

        Raises
        ------
        CredentialsMissingError, RequestError, DownloadError
            See :mod:`..errors`. ``ArchiveTimeoutError`` in particular means
            the request is saved and a retry resumes the wait.
        """
        api_key, email = client.credentials()
        manifest_path = cache_dir / CONFIG.manifest_filename
        manifest = archive.load_manifest(manifest_path)

        entry = manifest.get(name)
        if entry is None or force:
            on_event(f"requesting {name} ...")
            entry = client.request_node(
                node,
                requested_lat,
                requested_lon,
                api_key,
                email,
                on_event,
                years_subset=years,
                variables=variables,
            )
            manifest[name] = entry
            archive.save_manifest(manifest_path, manifest)
            if entry.get("message"):
                on_event(str(entry["message"]))

        url = entry.get("download_url")
        if not url:
            raise errors.DownloadError(f"{name}: the API returned no download URL", site=name)

        target = cache_dir / CONFIG.archives_dirname / f"{name}.zip"
        if force and target.exists():
            target.unlink()
        if not target.exists():
            on_event(f"waiting for the {name} archive to build")
            if not archive.wait_for_archive(name, url, timeout_s, on_event):
                raise errors.ArchiveTimeoutError(
                    f"the {name} archive was not ready after {timeout_s // 60} minutes. "
                    "The request is saved, so running again resumes the wait. The ready "
                    f"link also arrives by email, and a zip placed at {target} is picked "
                    "up on rerun. A failure email means the server gave up, so rerun "
                    "with --force.",
                    site=name,
                    url=url,
                )
            archive.download_archive(url, target, on_event)

        archive.organize_archive(target, manifest, cache_dir, on_event)
