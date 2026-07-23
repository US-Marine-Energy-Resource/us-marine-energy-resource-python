"""Format backend registry, resolved lazily so imports stay cheap.

Each format maps to a ``"module:attr"`` string. The backend module (and its
third-party dependency) loads only when a file of that format is opened.
"""

from __future__ import annotations

import importlib

from ..model import Format
from ..protocols import FormatBackend

_REGISTRY: dict[Format, str] = {
    "hdf5": "us_marine_energy_resource.explore.formats.hdf5:Hdf5Backend",
    "parquet": "us_marine_energy_resource.explore.formats.parquet:ParquetBackend",
}


def get_backend(fmt: Format) -> FormatBackend:
    """Load and instantiate the backend for a format.

    Parameters
    ----------
    fmt : Format
        The format to read.

    Returns
    -------
    FormatBackend
        A ready backend instance.
    """
    module_name, attr = _REGISTRY[fmt].split(":")
    module = importlib.import_module(module_name)
    backend_cls = getattr(module, attr)
    return backend_cls()
