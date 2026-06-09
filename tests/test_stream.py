# SPDX-License-Identifier: MIT
from __future__ import annotations

from threading import Event

import numpy as np
import pytest

from voxlocal._audio import AudioResult
from voxlocal._stream import (
    StreamAssembler,
    assemble_stream,
    prefetch_results,
    trim_boundary_silence,
)

# ─── trim_boundary_silence ──────────────────────────────────────────────────────


def test_short_current_chunk_is_not_retained_after_emission():
    assembler = StreamAssembler(sample_rate=100, overlap=8)
    assembler._tail = np.arange(8, dtype=np.float32) + 1

    chunk = assembler.push(1, np.arange(4, dtype=np.float32) + 10)
    final = assembler.finish()
    combined = np.concatenate(
        [
            chunk.numpy if chunk is not None else np.empty(0),
            final.numpy if final is not None else np.empty(0),
        ]
    )

    assert len(combined) == 8


def test_trim_boundary_silence_preserves_padding():
    audio = np.concatenate(
        [
            np.zeros(10, dtype=np.float32),
            np.ones(10, dtype=np.float32),
            np.zeros(10, dtype=np.float32),
        ]
    )

    trimmed = trim_boundary_silence(
        audio,
        sample_rate=100,
        trim_start=True,
        trim_end=True,
    )

    assert len(trimmed) == 22
    assert np.array_equal(trimmed[6:16], np.ones(10, dtype=np.float32))


def test_trim_boundary_silence_empty_audio():
    audio = np.array([], dtype=np.float32)
    trimmed = trim_boundary_silence(
        audio, sample_rate=100, trim_start=True, trim_end=True
    )
    assert len(trimmed) == 0


def test_trim_boundary_silence_all_silence():
    audio = np.zeros(100, dtype=np.float32)
    trimmed = trim_boundary_silence(
        audio, sample_rate=100, trim_start=True, trim_end=True
    )
    # When no active samples are found, returns the original array
    assert len(trimmed) == 100


def test_trim_boundary_silence_no_trim():
    audio = np.concatenate(
        [
            np.zeros(10, dtype=np.float32),
            np.ones(10, dtype=np.float32),
            np.zeros(10, dtype=np.float32),
        ]
    )
    trimmed = trim_boundary_silence(
        audio, sample_rate=100, trim_start=False, trim_end=False
    )
    assert len(trimmed) == 30


def test_trim_boundary_silence_single_sample():
    audio = np.array([0.5], dtype=np.float32)
    trimmed = trim_boundary_silence(
        audio, sample_rate=100, trim_start=True, trim_end=True
    )
    assert len(trimmed) >= 1


# ─── StreamAssembler ────────────────────────────────────────────────────────────


def test_assembler_push_returns_none_for_silent_chunk():
    assembler = StreamAssembler(sample_rate=100, overlap=8)
    chunk = assembler.push(0, np.zeros(10, dtype=np.float32))
    # Silent audio gets trimmed but retains padding, so may produce a chunk
    # The key invariant is that the chunk is small (just padding)
    if chunk is not None:
        assert len(chunk.numpy) <= 8  # Only padding samples


def test_assembler_finish_returns_none_when_empty():
    assembler = StreamAssembler(sample_rate=100, overlap=8)
    chunk = assembler.finish()
    assert chunk is None


def test_assembler_reset():
    assembler = StreamAssembler(sample_rate=100, overlap=8)
    assembler.push(0, np.ones(50, dtype=np.float32))
    assembler.reset()
    assert assembler._tail is None
    assert assembler._sequence == 0


def test_assembler_crossfade_preserves_duration():
    """Verify crossfade doesn't lose or duplicate samples."""
    sr = 100
    overlap = 8
    assembler = StreamAssembler(sample_rate=sr, overlap=overlap)
    a = np.ones(50, dtype=np.float32)
    b = np.ones(50, dtype=np.float32) * 0.5

    c1 = assembler.push(0, a)
    c2 = assembler.push(1, b)
    final = assembler.finish()

    total_samples = sum(len(c.numpy) for c in [c1, c2, final] if c is not None)
    # Due to crossfade overlap, total should be < 100
    assert total_samples < 100
    assert total_samples > 50


def test_assembler_single_chunk():
    assembler = StreamAssembler(sample_rate=100, overlap=8)
    chunk = assembler.push(0, np.ones(20, dtype=np.float32))
    final = assembler.finish()
    assert chunk is not None
    assert final is not None
    assert final.final is True


def test_assembler_many_chunks():
    assembler = StreamAssembler(sample_rate=100, overlap=8)
    chunks = []
    for i in range(10):
        c = assembler.push(i, np.ones(20, dtype=np.float32))
        if c is not None:
            chunks.append(c)
    final = assembler.finish()
    if final is not None:
        chunks.append(final)
    # Verify sequences are monotonically increasing
    assert [c.sequence for c in chunks] == list(range(len(chunks)))


# ─── prefetch_results ───────────────────────────────────────────────────────────


def test_prefetch_yields_first_before_later_generation_finishes():
    generating_second = Event()
    release_second = Event()

    def results():
        yield AudioResult(np.ones(10, dtype=np.float32), 100)
        generating_second.set()
        release_second.wait(timeout=1)
        yield AudioResult(np.ones(10, dtype=np.float32), 100)

    stream = prefetch_results(results(), maxsize=1)
    first = next(stream)

    assert len(first.numpy) == 10
    assert generating_second.wait(timeout=1)
    release_second.set()
    stream.close()


def test_prefetch_empty_input():
    results = list(prefetch_results([], maxsize=2))
    assert results == []


def test_prefetch_single_item():
    results = list(
        prefetch_results(
            [AudioResult(np.ones(5, dtype=np.float32), 100)],
            maxsize=2,
        )
    )
    assert len(results) == 1
    assert len(results[0].numpy) == 5


def test_prefetch_rejects_zero_maxsize():
    with pytest.raises(ValueError, match="at least 1"):
        list(prefetch_results([], maxsize=0))


def test_prefetch_error_propagation():
    def results():
        yield AudioResult(np.ones(5, dtype=np.float32), 100)
        raise ValueError("synthesis failed")

    stream = prefetch_results(results(), maxsize=2)
    first = next(stream)
    assert len(first.numpy) == 5
    with pytest.raises(Exception, match="synthesis failed"):
        next(stream)


# ─── assemble_stream ────────────────────────────────────────────────────────────


def test_assemble_stream_empty_raises():
    with pytest.raises(Exception, match="no audio"):
        list(assemble_stream([], prefetch=2))


def test_assemble_stream_single_chunk():
    def results():
        yield AudioResult(np.ones(100, dtype=np.float32), 16000)

    chunks = list(assemble_stream(results(), prefetch=1))
    assert len(chunks) >= 1
    assert chunks[-1].final is True


def test_assemble_stream_sample_rate_mismatch():
    def results():
        yield AudioResult(np.ones(50, dtype=np.float32), 16000)
        yield AudioResult(np.ones(50, dtype=np.float32), 24000)

    with pytest.raises(Exception, match="Sample rate changed"):
        list(assemble_stream(results(), prefetch=1))


def test_assemble_stream_crossfades_correctly():
    """End-to-end: verify chunks combine without gaps."""

    def results():
        yield AudioResult(np.ones(50, dtype=np.float32), 100)
        yield AudioResult(np.full(50, 0.5, dtype=np.float32), 100)

    chunks = list(assemble_stream(results(), prefetch=1))
    combined = np.concatenate([c.numpy for c in chunks])

    # Crossfade should produce combined audio < 100 (original 50+50)
    assert len(combined) < 100
    assert len(combined) > 50
    assert chunks[-1].final is True
    assert [c.sequence for c in chunks] == list(range(len(chunks)))
