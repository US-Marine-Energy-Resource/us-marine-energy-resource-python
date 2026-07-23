"""File locations: local disk, S3, and HTTP(S).

Each source yields a seekable binary handle and knows nothing about the file's
format. ``resolve_source`` picks one by URI scheme.
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO
from urllib.parse import urlparse

from .blockio import BlockCachedReader
from .errors import SourceError
from .lazy import lazy_import
from .model import ByteSize, SourceRef


class LocalSource:
    """A file on the local filesystem, including HPC paths like ``/projects/...``.

    Parameters
    ----------
    path : Path
        Location of the file on disk.

    Raises
    ------
    SourceError
        If the path is not a file.
    """

    def __init__(self, path: Path) -> None:
        """Record the path and stat its size."""
        self._path = path
        if not path.is_file():
            raise SourceError(f"not a file: {path}")
        size = ByteSize(path.stat().st_size)
        self.ref = SourceRef(uri=str(path), scheme="file", display=str(path), size=size)

    @contextmanager
    def open_binary(
        self, max_bytes: int | None = None, block_size: int | None = None
    ) -> Iterator[BinaryIO]:
        """Open the file for reading. Local reads move nothing, so limits are ignored.

        Parameters
        ----------
        max_bytes : int or None
            Ignored for local files.
        block_size : int or None
            Ignored for local files.

        Yields
        ------
        BinaryIO
            A seekable handle on the file.
        """
        with open(self._path, "rb") as handle:
            yield handle

    def peek(self, n: int) -> bytes:
        """Read the first ``n`` bytes.

        Parameters
        ----------
        n : int
            Number of bytes to read.

        Returns
        -------
        bytes
            The bytes read.
        """
        with open(self._path, "rb") as handle:
            return handle.read(n)


class S3Source:
    """An object in S3, read over range requests without downloading the whole file.

    Parameters
    ----------
    bucket : str
        Name of the bucket.
    key : str
        Key of the object within the bucket.
    aws_profile : str, optional
        AWS profile for signed access. Anonymous when omitted.

    Raises
    ------
    SourceError
        If the object cannot be reached.
    """

    def __init__(self, bucket: str, key: str, aws_profile: str | None = None) -> None:
        """Resolve the object's region and size."""
        self._bucket = bucket
        self._key = key
        self._profile = aws_profile
        self._s3_path = f"{bucket}/{key}"
        try:
            info = self._fs().get_file_info(self._s3_path)
        except Exception as exc:
            raise SourceError(f"cannot reach s3://{self._s3_path}: {exc}") from exc
        size = None if info.size is None else ByteSize(info.size)
        self.ref = SourceRef(
            uri=f"s3://{self._s3_path}",
            scheme="s3",
            display=f"s3://{self._s3_path}",
            size=size,
        )

    def _fs(self):
        """Build an S3 filesystem, anonymous unless a profile is set.

        Returns
        -------
        pyarrow.fs.S3FileSystem
            Filesystem bound to the bucket's region.
        """
        fs_mod = lazy_import("pyarrow.fs", "reading files from S3")
        region = fs_mod.resolve_s3_region(self._bucket)
        if self._profile:
            return fs_mod.S3FileSystem(profile=self._profile, region=region)
        return fs_mod.S3FileSystem(anonymous=True, region=region)

    @contextmanager
    def open_binary(
        self, max_bytes: int | None = None, block_size: int | None = None
    ) -> Iterator[BinaryIO]:
        """Open a block-cached, range-backed handle to the object.

        Parameters
        ----------
        max_bytes : int or None
            Cap on total bytes fetched, or no cap.
        block_size : int or None
            Bytes fetched per block, or the default.

        Yields
        ------
        BinaryIO
            A seekable buffered handle on the object.
        """
        size = self.ref.size.bytes if self.ref.size is not None else 0
        native = self._fs().open_input_file(self._s3_path)

        def fetch(offset: int, length: int) -> bytes:
            """Read one range from the object.

            Parameters
            ----------
            offset : int
                Start byte.
            length : int
                Number of bytes to read.

            Returns
            -------
            bytes
                The bytes read.
            """
            native.seek(offset)
            return native.read(length)

        reader = BlockCachedReader(size, fetch, block_size=block_size, max_bytes=max_bytes)
        try:
            yield io.BufferedReader(reader)
        finally:
            native.close()

    def peek(self, n: int) -> bytes:
        """Read the first ``n`` bytes with one range request.

        Parameters
        ----------
        n : int
            Number of bytes to read.

        Returns
        -------
        bytes
            The bytes read.
        """
        with self._fs().open_input_file(self._s3_path) as native:
            return native.read(n)


class HttpSource:
    """A file served over HTTP(S), read with range requests when the server allows.

    Parameters
    ----------
    url : str
        Address of the file.

    Raises
    ------
    SourceError
        If the URL cannot be reached.
    """

    def __init__(self, url: str) -> None:
        """Probe the URL for size and range support."""
        self._url = url
        requests = lazy_import("requests", "reading files over HTTP(S)")
        try:
            resp = requests.head(url, allow_redirects=True, timeout=30)
            resp.raise_for_status()
        except Exception as exc:
            raise SourceError(f"cannot reach {url}: {exc}") from exc
        self._accept_ranges = resp.headers.get("Accept-Ranges", "").lower() == "bytes"
        length = resp.headers.get("Content-Length")
        size = ByteSize(int(length)) if length is not None else None
        self.ref = SourceRef(uri=url, scheme="https", display=url, size=size)

    @contextmanager
    def open_binary(
        self, max_bytes: int | None = None, block_size: int | None = None
    ) -> Iterator[BinaryIO]:
        """Open a block-cached handle backed by HTTP range requests.

        Parameters
        ----------
        max_bytes : int or None
            Cap on total bytes fetched, or no cap.
        block_size : int or None
            Bytes fetched per block, or the default.

        Yields
        ------
        BinaryIO
            A seekable buffered handle on the file.

        Raises
        ------
        SourceError
            If the server does not support range requests.
        """
        if not self._accept_ranges or self.ref.size is None:
            raise SourceError(
                f"{self._url} does not support range requests. A full download is required"
            )
        requests = lazy_import("requests", "reading files over HTTP(S)")

        def fetch(offset: int, length: int) -> bytes:
            """Read one range over HTTP.

            Parameters
            ----------
            offset : int
                Start byte.
            length : int
                Number of bytes to read.

            Returns
            -------
            bytes
                The bytes read.
            """
            headers = {"Range": f"bytes={offset}-{offset + length - 1}"}
            resp = requests.get(self._url, headers=headers, timeout=60)
            resp.raise_for_status()
            return resp.content

        yield io.BufferedReader(
            BlockCachedReader(
                self.ref.size.bytes, fetch, block_size=block_size, max_bytes=max_bytes
            )
        )

    def peek(self, n: int) -> bytes:
        """Read the first ``n`` bytes with one range request.

        Parameters
        ----------
        n : int
            Number of bytes to read.

        Returns
        -------
        bytes
            The bytes read.
        """
        requests = lazy_import("requests", "reading files over HTTP(S)")
        resp = requests.get(self._url, headers={"Range": f"bytes=0-{n - 1}"}, timeout=30)
        resp.raise_for_status()
        return resp.content


def resolve_source(uri: str, aws_profile: str | None = None) -> LocalSource | S3Source | HttpSource:
    """Pick a source for a URI by its scheme.

    Parameters
    ----------
    uri : str
        A local path, ``s3://bucket/key``, or ``http(s)://...`` URL.
    aws_profile : str, optional
        AWS profile for signed S3 access. Anonymous when omitted.

    Returns
    -------
    LocalSource or S3Source or HttpSource
        A source matching the scheme.
    """
    parsed = urlparse(uri)
    if len(parsed.scheme) < 2:
        # No scheme, or a Windows drive letter that urlparse mistakes for one.
        return LocalSource(Path(uri))
    if parsed.scheme == "file":
        return LocalSource(Path(parsed.path))
    if parsed.scheme == "s3":
        return S3Source(parsed.netloc, parsed.path.lstrip("/"), aws_profile=aws_profile)
    if parsed.scheme in ("http", "https"):
        return HttpSource(uri)
    raise SourceError(f"unsupported URI scheme: {parsed.scheme!r}")
