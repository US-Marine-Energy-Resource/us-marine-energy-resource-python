"""What to read from an array.

``Selection`` is a union of the only two shapes a read can take: the first N
rows, or an explicit numpy-style slice. There is no "read everything" variant,
so an unbounded read cannot be built. ``resolve`` turns a ``Selection`` into
concrete slices, validating rank and bounds against the real array once.
"""

from __future__ import annotations

import dataclasses

from .model import ArrayInfo


@dataclasses.dataclass(frozen=True)
class FirstN:
    """The first ``n`` entries along axis 0."""

    n: int

    def __post_init__(self) -> None:
        """Reject non-positive counts."""
        if self.n < 1:
            raise ValueError(f"n must be >= 1: {self.n}")


@dataclasses.dataclass(frozen=True)
class Index:
    """A numpy-style slice string, e.g. ``0:5,::2``."""

    spec: str

    def __post_init__(self) -> None:
        """Reject an empty slice string."""
        if not self.spec.strip():
            raise ValueError("index spec is empty")


Selection = FirstN | Index


@dataclasses.dataclass(frozen=True)
class ResolvedSelection:
    """Concrete per-axis slices, valid for a specific array shape."""

    slices: tuple[slice, ...]
    text: str

    @property
    def n_elements(self) -> int:
        """Number of elements the slices select."""
        total = 1
        for s in self.slices:
            total *= len(range(s.start, s.stop, s.step or 1))
        return total


def _parse_axis(token: str, length: int) -> slice:
    """Parse one axis token (``5``, ``0:10``, ``::2``) against an axis length."""
    token = token.strip()
    if ":" not in token:
        idx = int(token)
        if idx < 0:
            idx += length
        if not 0 <= idx < length:
            raise ValueError(f"index {token} out of range for axis length {length}")
        return slice(idx, idx + 1)
    parts = token.split(":")
    if len(parts) > 3:
        raise ValueError(f"too many ':' in slice {token!r}")
    start = int(parts[0]) if parts[0].strip() else None
    stop = int(parts[1]) if parts[1].strip() else None
    step = int(parts[2]) if len(parts) == 3 and parts[2].strip() else None
    return slice(*slice(start, stop, step).indices(length))


def resolve(selection: Selection, array: ArrayInfo) -> ResolvedSelection:
    """Turn a selection into concrete slices for a specific array.

    Parameters
    ----------
    selection : Selection
        The requested read shape.
    array : ArrayInfo
        The array to read from; supplies the true shape.

    Returns
    -------
    ResolvedSelection
        Per-axis slices matching the array rank and bounds.

    Raises
    ------
    ValueError
        If the array is scalar, or the index rank or bounds do not fit.
    """
    shape = array.shape
    if not shape:
        raise ValueError("cannot slice a scalar array")

    if isinstance(selection, FirstN):
        n = min(selection.n, shape[0])
        slices = (slice(0, n), *(slice(0, d) for d in shape[1:]))
        return ResolvedSelection(slices=slices, text=f"0:{n}" + ",:" * (len(shape) - 1))

    tokens = selection.spec.split(",")
    if len(tokens) > len(shape):
        raise ValueError(f"index has {len(tokens)} axes but array has {len(shape)}")
    slices = tuple(_parse_axis(tok, shape[i]) for i, tok in enumerate(tokens))
    slices += tuple(slice(0, d) for d in shape[len(tokens) :])
    return ResolvedSelection(slices=slices, text=selection.spec)
