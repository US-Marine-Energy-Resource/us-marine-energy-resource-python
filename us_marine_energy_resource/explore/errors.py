"""Error types for file exploration.

``ExploreError`` is the union a caller can catch to handle any expected failure
(bad format, missing dependency, over-budget read, missing node, unreachable
source). Programming errors (bad ``Selection``, negative ``ByteSize``) stay as
plain ``ValueError`` and are not part of this union.
"""

from __future__ import annotations


class ExploreError(Exception):
    """Base class for expected exploration failures."""


class UnsupportedFormatError(ExploreError):
    """The file is a format ``mer`` does not read."""


class UnknownFormatError(ExploreError):
    """The file's leading bytes match no known format."""


class DependencyError(ExploreError):
    """A backend's third-party dependency is not installed."""

    def __init__(self, module: str, purpose: str) -> None:
        """Record the missing module and what it was needed for."""
        self.module = module
        self.purpose = purpose
        super().__init__(
            f"{module!r} is required for {purpose}. Install it with: pip install {module}"
        )


class NodeNotFoundError(ExploreError):
    """No group, array, or column exists at the requested path."""


class SourceError(ExploreError):
    """The file could not be reached or read from its location."""


class TransferBudgetError(ExploreError):
    """A read was refused before starting because it would exceed a limit."""


class TransferBudgetExceededError(ExploreError):
    """A read crossed its transfer limit while running and was stopped."""

    def __init__(self, fetched: int, limit: int) -> None:
        """Record bytes fetched and the limit that was crossed."""
        self.fetched = fetched
        self.limit = limit
        super().__init__(f"fetched {fetched} bytes, over limit {limit}")
