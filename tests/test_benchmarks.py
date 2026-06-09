# SPDX-License-Identifier: MIT
"""Performance benchmarks for VoxLocal components.

Run with: pytest tests/test_benchmarks.py --benchmark-enable
"""
from __future__ import annotations

import numpy as np
import pytest

from voxlocal._audio import AudioChunk, AudioResult
from voxlocal._stream import StreamAssembler, assemble_stream, trim_boundary_silence
from voxlocal.tts._supertonic import (
    _split_first_sentence,
    _split_progressive,
    _split_sentences,
    _split_text,
)

# ─── Audio construction benchmarks ─────────────────────────────────────────────


def test_benchmark_audio_result_creation(benchmark: pytest.BenchmarkFixture) -> None:
    """Benchmark AudioResult creation and validation."""
    data = np.ones(48000, dtype=np.float32) * 0.5

    def create_result() -> AudioResult:
        return AudioResult(numpy=data.copy(), sample_rate=24000)

    benchmark(create_result)


def test_benchmark_audio_result_bytes(benchmark: pytest.BenchmarkFixture) -> None:
    """Benchmark WAV encoding."""
    result = AudioResult(
        numpy=np.ones(48000, dtype=np.float32) * 0.5, sample_rate=24000
    )
    benchmark(lambda: result.bytes)


def test_benchmark_audio_result_pcm(benchmark: pytest.BenchmarkFixture) -> None:
    """Benchmark PCM16 encoding."""
    result = AudioResult(
        numpy=np.ones(48000, dtype=np.float32) * 0.5, sample_rate=24000
    )
    benchmark(lambda: result.pcm_s16le)


def test_benchmark_audio_chunk_wire(benchmark: pytest.BenchmarkFixture) -> None:
    """Benchmark wire format serialization."""
    chunk = AudioChunk(
        sequence=0,
        numpy=np.ones(4800, dtype=np.float32) * 0.5,
        sample_rate=24000,
    )
    benchmark(chunk.to_wire_dict)


# ─── Stream assembly benchmarks ────────────────────────────────────────────────


def test_benchmark_trim_silence_short(benchmark: pytest.BenchmarkFixture) -> None:
    """Benchmark silence trimming on short audio (100ms)."""
    audio = np.concatenate(
        [
            np.zeros(200, dtype=np.float32),
            np.ones(1600, dtype=np.float32) * 0.5,
            np.zeros(200, dtype=np.float32),
        ]
    )
    benchmark(trim_boundary_silence, audio, 16000, trim_start=True, trim_end=True)


def test_benchmark_trim_silence_long(benchmark: pytest.BenchmarkFixture) -> None:
    """Benchmark silence trimming on longer audio (5s)."""
    audio = np.concatenate(
        [
            np.zeros(4000, dtype=np.float32),
            np.ones(76000, dtype=np.float32) * 0.5,
            np.zeros(4000, dtype=np.float32),
        ]
    )
    benchmark(trim_boundary_silence, audio, 16000, trim_start=True, trim_end=True)


def test_benchmark_stream_assembler_push(
    benchmark: pytest.BenchmarkFixture,
) -> None:
    """Benchmark StreamAssembler.push with crossfade."""
    sr = 24000
    overlap = int(0.08 * sr)
    assembler = StreamAssembler(sample_rate=sr, overlap=overlap)
    audio = np.ones(2400, dtype=np.float32) * 0.5  # 100ms at 24kHz

    # Pre-fill tail to exercise crossfade path
    assembler.push(0, audio)

    def push_with_crossfade() -> AudioChunk | None:
        return assembler.push(1, audio.copy())

    benchmark(push_with_crossfade)


def test_benchmark_stream_assemble_10_chunks(
    benchmark: pytest.BenchmarkFixture,
) -> None:
    """Benchmark full assembly of 10 chunks."""
    sr = 24000

    def results():
        for _ in range(10):
            yield AudioResult(
                numpy=np.ones(4800, dtype=np.float32) * 0.5,
                sample_rate=sr,
            )

    benchmark(lambda: list(assemble_stream(results(), prefetch=2)))


# ─── Text splitting benchmarks ─────────────────────────────────────────────────


LONG_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump! "
    "The five boxing wizards jump quickly. "
    "Sphinx of black quartz, judge my vow. "
    "Two driven jacks help fax my big quiz. "
    "The jay, pig, fox, zebra, and my wolves quack! "
    "Sympathizing would fix Quaker objectives. "
    "A wizard's job is to vex chumps quickly in fog. "
    "Watch Jeopardy!, Alex Trebek's fun TV quiz game."
)


def test_benchmark_split_progressive_short(
    benchmark: pytest.BenchmarkFixture,
) -> None:
    """Benchmark progressive splitting on short text."""
    benchmark(_split_progressive, "Hello world, this is a test.", "en")


def test_benchmark_split_progressive_long(
    benchmark: pytest.BenchmarkFixture,
) -> None:
    """Benchmark progressive splitting on long text."""
    benchmark(_split_progressive, LONG_TEXT, "en")


def test_benchmark_split_sentences(benchmark: pytest.BenchmarkFixture) -> None:
    """Benchmark sentence splitting."""
    benchmark(_split_sentences, LONG_TEXT)


def test_benchmark_split_text_line(benchmark: pytest.BenchmarkFixture) -> None:
    """Benchmark line splitting."""
    lines = "\n".join(["Line number " + str(i) + "." for i in range(100)])
    benchmark(_split_text, lines, "line")


def test_benchmark_split_text_paragraph(benchmark: pytest.BenchmarkFixture) -> None:
    """Benchmark paragraph splitting."""
    paragraphs = "\n\n".join(
        ["Paragraph " + str(i) + ". " + LONG_TEXT for i in range(10)]
    )
    benchmark(_split_text, paragraphs, "paragraph")


# ─── Split first sentence benchmarks ───────────────────────────────────────────


def test_benchmark_split_first_sentence_short(
    benchmark: pytest.BenchmarkFixture,
) -> None:
    """Benchmark first sentence splitting with few words."""
    benchmark(_split_first_sentence, "Hello world.", "en")


def test_benchmark_split_first_sentence_long(
    benchmark: pytest.BenchmarkFixture,
) -> None:
    """Benchmark first sentence splitting with many words."""
    sentence = " ".join(["word"] * 50) + "."
    benchmark(_split_first_sentence, sentence, "en")


# ─── Crossfade computation benchmark ──────────────────────────────────────────


def test_benchmark_crossfade_join(benchmark: pytest.BenchmarkFixture) -> None:
    """Benchmark crossfade boundary computation."""
    sr = 24000
    overlap = int(0.08 * sr)
    assembler = StreamAssembler(sample_rate=sr, overlap=overlap)
    # Fill the tail
    assembler._tail = np.ones(overlap, dtype=np.float32) * 0.5
    current = np.ones(4800, dtype=np.float32) * 0.3

    benchmark(assembler._join_boundary, assembler._tail, current)
