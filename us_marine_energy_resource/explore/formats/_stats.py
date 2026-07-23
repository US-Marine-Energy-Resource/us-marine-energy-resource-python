"""Shared statistics reduction for the format backends.

Both backends end a stats read with the same step: reduce whatever sample was
read to a ``StatsResult`` that reports honestly how much of the array it
covers. That reduction lives here so the two stay identical.
"""

from __future__ import annotations

from typing import Any

from ..model import NodePath, StatsResult, StatsSpec


def summarize(
    np: Any,
    data: Any,
    path: NodePath,
    total: int,
    read: int,
    method: str,
    spec: StatsSpec,
) -> StatsResult:
    """Reduce sampled data to a StatsResult with honest coverage fields.

    Parameters
    ----------
    np : module
        The numpy module, passed in so this module stays import free.
    data : numpy.ndarray
        The values that were read.
    path : NodePath
        The array the values came from.
    total : int
        Element count of the full array.
    read : int
        Element count actually read.
    method : str
        How the sample was taken, ``"full"`` or ``"chunk-strided"``.
    spec : StatsSpec
        Controls the NaN policy.

    Returns
    -------
    StatsResult
        Statistics plus the sampling that produced them.
    """
    numeric = data.dtype.kind in "iufc"
    sampled = read < total
    fraction = (read / total) if total else 1.0
    if not numeric or read == 0:
        return StatsResult(
            path=path,
            count=read,
            n_nan=0,
            mean=None,
            std=None,
            min=None,
            max=None,
            sampled=sampled,
            sample_fraction=fraction,
            sample_method=method,  # type: ignore[arg-type]
        )
    flat = data.astype("float64").ravel()
    n_nan = int(np.isnan(flat).sum())
    clean = flat[~np.isnan(flat)] if spec.nan_policy == "omit" else flat
    has = clean.size > 0
    return StatsResult(
        path=path,
        count=read,
        n_nan=n_nan,
        mean=float(np.mean(clean)) if has else None,
        std=float(np.std(clean)) if has else None,
        min=float(np.min(clean)) if has else None,
        max=float(np.max(clean)) if has else None,
        sampled=sampled,
        sample_fraction=fraction,
        sample_method=method,  # type: ignore[arg-type]
    )
