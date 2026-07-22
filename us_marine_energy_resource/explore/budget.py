"""Volume limits and the gate that enforces them.

A payload read needs an ``ApprovedRead``, and the only way to get one is
``TransferPolicy.approve``. There is no path from a path string to bytes that
skips the check. ``approve`` returns one of three outcomes the caller matches
on: ``ApprovedRead`` (go), ``NeedsConfirm`` (ask the user), or ``Refusal``
(stop, and here is a cheaper command).
"""

from __future__ import annotations

import dataclasses

from .model import ByteSize, NodeInfo
from .selection import ResolvedSelection

_MB = 1024 * 1024

# Default ceilings, as module constants so they are not re-evaluated as
# function-call dataclass defaults.
_DEFAULT_MAX_TRANSFER = ByteSize(100 * _MB)
_DEFAULT_MAX_MEMORY = ByteSize(512 * _MB)
_DEFAULT_MAX_DOWNLOAD = ByteSize(500 * _MB)
_DEFAULT_CONFIRM_ABOVE = ByteSize(25 * _MB)

# Only approve() may construct an ApprovedRead. Hand-construction needs this
# object, which is not exported, so bypassing the gate is deliberate, not casual.
_GATE_TOKEN = object()


@dataclasses.dataclass(frozen=True)
class ReadPlan:
    """What a read would cost, computed from metadata alone."""

    node: NodeInfo
    selection: ResolvedSelection
    logical: ByteSize
    transferred: ByteSize
    n_chunks: int

    @property
    def amplification(self) -> float:
        """Transferred bytes divided by requested bytes.

        Returns
        -------
        float
            The ratio, or one when nothing was requested.
        """
        if self.logical.bytes == 0:
            return 1.0
        return self.transferred.bytes / self.logical.bytes


@dataclasses.dataclass(frozen=True)
class TransferPolicy:
    """Per-invocation ceilings on how much a read may move or hold."""

    max_transfer: ByteSize = _DEFAULT_MAX_TRANSFER
    max_memory: ByteSize = _DEFAULT_MAX_MEMORY
    max_download: ByteSize = _DEFAULT_MAX_DOWNLOAD
    confirm_above: ByteSize = _DEFAULT_CONFIRM_ABOVE
    assume_yes: bool = False
    dry_run: bool = False

    def approve(self, plan: ReadPlan, *, remote: bool) -> ApprovedRead | NeedsConfirm | Refusal:
        """Check a plan against the limits.

        Parameters
        ----------
        plan : ReadPlan
            The proposed read.
        remote : bool
            Whether the source moves bytes over the network. Local reads skip
            the transfer limit but still face the memory limit.

        Returns
        -------
        ApprovedRead or NeedsConfirm or Refusal
            ``NeedsConfirm`` means prompt the user; ``Refusal`` means stop.
        """
        cause = _cause(plan)
        if plan.logical.bytes > self.max_memory.bytes:
            return Refusal(plan, self.max_memory, "memory", cause, _alternatives(plan))
        if remote and plan.transferred.bytes > self.max_transfer.bytes:
            return Refusal(plan, self.max_transfer, "transfer", cause, _alternatives(plan))

        gauge = plan.transferred if remote else plan.logical
        if gauge.bytes > self.confirm_above.bytes and not self.assume_yes:
            return NeedsConfirm(plan, gauge)
        return ApprovedRead(plan=plan, policy=self, _token=_GATE_TOKEN)


@dataclasses.dataclass(frozen=True)
class ApprovedRead:
    """A ``ReadPlan`` that passed ``TransferPolicy.approve``."""

    plan: ReadPlan
    policy: TransferPolicy
    _token: dataclasses.InitVar[object]

    def __post_init__(self, _token: object) -> None:
        """Reject construction outside ``TransferPolicy.approve``.

        Parameters
        ----------
        _token : object
            Proof that ``approve`` created this instance.
        """
        if _token is not _GATE_TOKEN:
            raise TypeError("ApprovedRead is created only by TransferPolicy.approve")


@dataclasses.dataclass(frozen=True)
class NeedsConfirm:
    """The read is within hard limits but big enough to warrant a prompt."""

    plan: ReadPlan
    size: ByteSize


@dataclasses.dataclass(frozen=True)
class Refusal:
    """A rejected read: why, and what to run instead."""

    plan: ReadPlan
    limit: ByteSize
    limit_kind: str
    cause: str
    alternatives: tuple[str, ...]

    def message(self) -> str:
        """One-line reason plus the suggested alternatives.

        Returns
        -------
        str
            The reason, followed by commands to try instead.
        """
        head = (
            f"Refused: read would use {self.plan.transferred} "
            f"({self.limit_kind} limit {self.limit}). {self.cause}"
        )
        if not self.alternatives:
            return head
        body = "\n".join(f"    {a}" for a in self.alternatives)
        return f"{head}\n  Try instead:\n{body}"


def _cause(plan: ReadPlan) -> str:
    """Describe why a read is as large as it is.

    Parameters
    ----------
    plan : ReadPlan
        The proposed read.

    Returns
    -------
    str
        A short description, empty when the node has no array.
    """
    arr = plan.node.array
    if arr is None:
        return ""
    parts = [f"array {arr.shape} {arr.dtype}"]
    if arr.storage.chunks is not None:
        parts.append(f"chunks {arr.storage.chunks}")
    if arr.storage.compression:
        parts.append(str(arr.storage.compression))
    tail = ""
    if plan.amplification >= 10:
        tail = f"; slice costs {plan.transferred} at {plan.amplification:.0f}x amplification"
    return ", ".join(parts) + tail


def _alternatives(plan: ReadPlan) -> tuple[str, ...]:
    """Suggest cheaper commands for an over-budget read.

    Parameters
    ----------
    plan : ReadPlan
        The refused read.

    Returns
    -------
    tuple of str
        Example commands to run instead.
    """
    path = plan.node.path
    return (
        f"mer explore stats <uri> --path {path}                # sampled, reports sample_fraction",
        f"mer explore head  <uri> --path {path} -n 5           # first rows only",
        f"mer explore stats <uri> --path {path} --exact --max-transfer-mb 5000",
    )
