"""On-demand import of backend dependencies."""

from __future__ import annotations

import importlib
from types import ModuleType

from .errors import DependencyError


def lazy_import(module: str, purpose: str) -> ModuleType:
    """Import a module by name, mapping a missing import to ``DependencyError``.

    Parameters
    ----------
    module : str
        Import name, e.g. ``"h5py"``.
    purpose : str
        What the module is needed for, shown to the user if it is missing.

    Returns
    -------
    types.ModuleType
        The imported module.

    Raises
    ------
    DependencyError
        If the module is not installed.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise DependencyError(module=module, purpose=purpose) from exc
