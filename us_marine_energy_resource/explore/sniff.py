"""Identify a file's format from its leading bytes, not its name."""

from __future__ import annotations

from .errors import UnknownFormatError, UnsupportedFormatError
from .model import Format

_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
_PARQUET_MAGIC = b"PAR1"
_NETCDF3_MAGIC = (b"CDF\x01", b"CDF\x02", b"CDF\x05")


def sniff_format(head: bytes) -> Format:
    """Return the format of a file from its first bytes.

    Parameters
    ----------
    head : bytes
        At least the first 8 bytes of the file.

    Returns
    -------
    Format
        ``"hdf5"`` (covers ``.h5`` and netCDF-4) or ``"parquet"``.

    Raises
    ------
    UnsupportedFormatError
        For netCDF-3 classic, which this tool does not read.
    UnknownFormatError
        If the bytes match no known format.
    """
    if head.startswith(_HDF5_MAGIC):
        return "hdf5"
    if head.startswith(_PARQUET_MAGIC):
        return "parquet"
    if head.startswith(_NETCDF3_MAGIC):
        raise UnsupportedFormatError("netCDF-3 classic; mer reads netCDF-4/HDF5 and parquet")
    raise UnknownFormatError(f"unrecognized file header: {head[:8]!r}")
