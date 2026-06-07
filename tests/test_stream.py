from threading import Event

import numpy as np

from voxlocal._audio import AudioResult
from voxlocal._stream import (
    StreamAssembler,
    prefetch_results,
    trim_boundary_silence,
)


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
