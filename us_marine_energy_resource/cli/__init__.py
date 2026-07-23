"""Tidal query CLI, mounted as ``mer tidal`` and the deprecated ``us-tidal``."""

from .app import app, tidal_app, us_tidal_entry

__all__ = ["app", "tidal_app", "us_tidal_entry"]
