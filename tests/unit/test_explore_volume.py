"""Volume guardrails: metadata stays cheap, the gate refuses, the fuse trips."""

from __future__ import annotations

import io
import os
from pathlib import Path

import pytest

from us_marine_energy_resource.explore import (
    ByteSize,
    FirstN,
    Index,
    NodePath,
    StatsSpec,
    TransferPolicy,
    open_file,
)
from us_marine_energy_resource.explore.blockio import BlockCachedReader
from us_marine_energy_resource.explore.budget import ApprovedRead, NeedsConfirm, Refusal
from us_marine_energy_resource.explore.errors import TransferBudgetExceededError


def test_structure_walk_reads_little(h5_file: Path) -> None:
    """tree/info/attrs must not read payload, whatever the array size."""
    data = h5_file.read_bytes()
    counter = {"n": 0}

    def fetch(offset: int, length: int) -> bytes:
        counter["n"] += 1
        return data[offset : offset + length]

    reader = BlockCachedReader(len(data), fetch, block_size=1 << 20)
    import h5py

    with h5py.File(io.BufferedReader(reader), "r") as f:
        # Walk every node's metadata, as summary() does.
        f.visititems(lambda name, obj: (obj.shape, obj.dtype) if hasattr(obj, "shape") else None)
        _ = dict(f.attrs)
    assert reader.bytes_fetched < 256 * 1024


def test_plan_reports_amplification(h5_file: Path) -> None:
    """Plan reports amplification."""
    with open_file(str(h5_file)) as f:
        plan = f.plan_read(NodePath("/significant_wave_height"), Index("0:1,0:1"))
    # A 2-byte logical read pulls a whole compressed chunk.
    assert plan.transferred.bytes > plan.logical.bytes
    assert plan.amplification > 10


def test_gate_refuses_over_transfer(h5_file: Path) -> None:
    """Gate refuses over transfer."""
    with open_file(str(h5_file)) as f:
        plan = f.plan_stats(NodePath("/significant_wave_height"), StatsSpec(exact=True))
    tiny = TransferPolicy(max_transfer=ByteSize(8))
    outcome = tiny.approve(plan, remote=True)
    assert isinstance(outcome, Refusal)
    assert "Try instead" in outcome.message()


def test_gate_refuses_over_memory_even_local(h5_file: Path) -> None:
    """Gate refuses over memory even local."""
    with open_file(str(h5_file)) as f:
        plan = f.plan_stats(NodePath("/significant_wave_height"), StatsSpec(exact=True))
    tiny_mem = TransferPolicy(max_memory=ByteSize(8))
    outcome = tiny_mem.approve(plan, remote=False)
    assert isinstance(outcome, Refusal)
    assert outcome.limit_kind == "memory"


def test_gate_prompts_in_the_middle(h5_file: Path) -> None:
    """Gate prompts in the middle."""
    with open_file(str(h5_file)) as f:
        plan = f.plan_stats(NodePath("/significant_wave_height"), StatsSpec(exact=True))
    policy = TransferPolicy(confirm_above=ByteSize(1))
    outcome = policy.approve(plan, remote=False)
    assert isinstance(outcome, NeedsConfirm)


def test_approvedread_cannot_be_forged(h5_file: Path) -> None:
    """Approvedread cannot be forged."""
    with open_file(str(h5_file)) as f:
        plan = f.plan_read(NodePath("/significant_wave_height"), FirstN(1))
    with pytest.raises(TypeError):
        ApprovedRead(plan=plan, policy=TransferPolicy(), _token=object())


def test_fuse_trips_mid_read() -> None:
    """Fuse trips mid read."""
    payload = os.urandom(8 * 1024 * 1024)

    def fetch(offset: int, length: int) -> bytes:
        return payload[offset : offset + length]

    reader = BlockCachedReader(len(payload), fetch, block_size=1 << 20, max_bytes=2 * 1024 * 1024)
    with pytest.raises(TransferBudgetExceededError):
        io.BufferedReader(reader).read()


def test_blockio_is_correct_and_cheap() -> None:
    """Blockio is correct and cheap."""
    payload = os.urandom(10 * 1024 * 1024)
    gets = {"n": 0}

    def fetch(offset: int, length: int) -> bytes:
        gets["n"] += 1
        return payload[offset : offset + length]

    reader = BlockCachedReader(len(payload), fetch, block_size=1 << 20)
    br = io.BufferedReader(reader)
    br.seek(5)
    assert br.read(10) == payload[5:15]
    br.seek(9_000_000)
    assert br.read(100) == payload[9_000_000:9_000_100]
    assert gets["n"] <= 3


def test_large_remote_file_warns() -> None:
    """A large remote object gets a heads-up; local or small ones do not."""
    from us_marine_energy_resource.explore.cli._shared import _large_warning
    from us_marine_energy_resource.explore.model import ByteSize, SourceRef

    big = SourceRef("s3://b/x.h5", "s3", "s3://b/x.h5", ByteSize(80 * 1024**3))
    msg = _large_warning(big)
    assert msg is not None and "80.0 GB" in msg and "header is quick" in msg

    small = SourceRef("s3://b/x.h5", "s3", "s3://b/x.h5", ByteSize(1000))
    local = SourceRef("/tmp/x.h5", "file", "/tmp/x.h5", ByteSize(80 * 1024**3))
    assert _large_warning(small) is None
    assert _large_warning(local) is None
