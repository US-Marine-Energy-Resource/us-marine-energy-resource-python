"""A seekable reader that fetches remote files in aligned blocks.

HDF5 metadata reads are many small seeks. Fetched one range request each, that
is one network round-trip per seek. ``BlockCachedReader`` reads in aligned
blocks and caches them, so a metadata walk costs a handful of requests. It also
counts bytes fetched, which is the runtime fuse behind the transfer budget.
"""

from __future__ import annotations

import io
from collections import OrderedDict
from collections.abc import Callable

from .errors import TransferBudgetExceededError

_BLOCK = 4 * 1024 * 1024
_MAX_CACHED_BLOCKS = 128


class BlockCachedReader(io.RawIOBase):
    """Wrap a range-fetch function as a seekable, block-cached binary stream.

    Parameters
    ----------
    size : int
        Total length of the underlying file.
    fetch : Callable[[int, int], bytes]
        Reads ``length`` bytes at ``offset`` from the source.
    block_size : int, optional
        Alignment of cached blocks.
    max_bytes : int, optional
        Fuse: raise once more than this many bytes have been fetched.
    """

    def __init__(
        self,
        size: int,
        fetch: Callable[[int, int], bytes],
        block_size: int | None = None,
        max_bytes: int | None = None,
    ) -> None:
        super().__init__()
        self._size = size
        self._fetch = fetch
        self._block = block_size or _BLOCK
        self._max_bytes = max_bytes
        self._pos = 0
        self._blocks: OrderedDict[int, bytes] = OrderedDict()
        self.bytes_fetched = 0
        self.n_requests = 0

    def _load_block(self, index: int) -> bytes:
        """Return one aligned block, fetching and caching it on a miss.

        Parameters
        ----------
        index : int
            Block number counted from the start of the file.

        Returns
        -------
        bytes
            The block contents.
        """
        cached = self._blocks.get(index)
        if cached is not None:
            self._blocks.move_to_end(index)
            return cached
        offset = index * self._block
        length = min(self._block, self._size - offset)
        data = self._fetch(offset, length)
        self.bytes_fetched += len(data)
        self.n_requests += 1
        if self._max_bytes is not None and self.bytes_fetched > self._max_bytes:
            raise TransferBudgetExceededError(fetched=self.bytes_fetched, limit=self._max_bytes)
        self._blocks[index] = data
        if len(self._blocks) > _MAX_CACHED_BLOCKS:
            self._blocks.popitem(last=False)
        return data

    def readable(self) -> bool:
        """Return ``True``; the stream supports reading.

        Returns
        -------
        bool
            Always true.
        """
        return True

    def seekable(self) -> bool:
        """Return ``True``; the stream supports seeking.

        Returns
        -------
        bool
            Always true.
        """
        return True

    def tell(self) -> int:
        """Return the current position.

        Returns
        -------
        int
            Current position in bytes.
        """
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        """Move the read position and return it.

        Parameters
        ----------
        offset : int
            Position in bytes, relative to ``whence``.
        whence : int, optional
            One of ``io.SEEK_SET``, ``io.SEEK_CUR``, or ``io.SEEK_END``.

        Returns
        -------
        int
            The new position.

        Raises
        ------
        ValueError
            If ``whence`` is not a recognized value.
        """
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self._size + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        return self._pos

    def readinto(self, buffer) -> int:
        """Fill ``buffer`` from the current position, spanning cached blocks.

        Parameters
        ----------
        buffer : bytearray or memoryview
            Writable buffer to fill.

        Returns
        -------
        int
            Number of bytes written into ``buffer``.
        """
        if self._pos >= self._size:
            return 0
        want = min(len(buffer), self._size - self._pos)
        written = 0
        while written < want:
            pos = self._pos + written
            index = pos // self._block
            block = self._load_block(index)
            start = pos - index * self._block
            take = min(len(block) - start, want - written)
            buffer[written : written + take] = block[start : start + take]
            written += take
        self._pos += written
        return written
