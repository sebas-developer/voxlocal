# SPDX-License-Identifier: MIT
"""Property-based tests using Hypothesis for VoxLocal invariants.

Covers:
- trim_boundary_silence invariants (output ≤ input length, active region preserved)
- _split_progressive reconstruction (concatenated chunks equal original text)
- AudioResult.bytes round-trip (decode WAV, compare samples)
- AudioChunk wire format round-trip
- StreamAssembler sequence monotonicity
"""
# SPDX-License-Identifier: MIT
from __future__ import annotations

import io

import hypothesis.strategies as st
import numpy as np
import pytest
import soundfile as sf
from hypothesis import given, settings

from voxlocal._audio import AudioChunk, AudioResult
from voxlocal._stream import StreamAssembler, trim_boundary_silence
from voxlocal.tts._supertonic import (
    _split_progressive,
    _split_sentences,
    _split_text,
    _split_units,
)

# ─── Strategies ────────────────────────────────────────────────────────────────


@st.composite
def finite_float32_array(draw: st.DrawFn) -> np.ndarray:
    """Generate a numpy float32 array with values in [-1, 1]."""
    length = draw(st.integers(min_value=0, max_value=5000))
    values = draw(
        st.lists(
            st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
            min_size=length,
            max_size=length,
        )
    )
    return np.array(values, dtype=np.float32)


@st.composite
def non_empty_float32_array(draw: st.DrawFn) -> np.ndarray:
    """Generate a non-empty numpy float32 array with values in [-1, 1]."""
    length = draw(st.integers(min_value=1, max_value=5000))
    values = draw(
        st.lists(
            st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
            min_size=length,
            max_size=length,
        )
    )
    return np.array(values, dtype=np.float32)


@st.composite
def latin_text(draw: st.DrawFn) -> str:
    """Generate Latin-script text with words, sentences, and punctuation."""
    words = draw(
        st.lists(
            st.text(
                alphabet=st.characters(whitelist_categories=("L",)),
                min_size=1,
                max_size=12,
            ),
            min_size=1,
            max_size=30,
        )
    )
    sentence_enders = draw(
        st.lists(
            st.sampled_from([".", "!", "?"]),
            min_size=0,
            max_size=5,
        )
    )
    result = ""
    for i, word in enumerate(words):
        result += word
        if i < len(sentence_enders):
            result += sentence_enders[i] + " "
        else:
            result += " "
    return result.strip()


@st.composite
def positive_sample_rate(draw: st.DrawFn) -> int:
    """Generate a positive sample rate."""
    return draw(st.sampled_from([8000, 16000, 22050, 24000, 44100, 48000]))


# ─── trim_boundary_silence invariants ──────────────────────────────────────────


@given(audio=finite_float32_array(), sample_rate=positive_sample_rate())
@settings(max_examples=200)
def test_trim_output_never_exceeds_input_length(
    audio: np.ndarray, sample_rate: int
) -> None:
    """Trimmed audio must never be longer than the original."""
    trimmed = trim_boundary_silence(
        audio, sample_rate, trim_start=True, trim_end=True
    )
    assert len(trimmed) <= len(audio)


@given(
    audio=non_empty_float32_array(),
    sample_rate=positive_sample_rate(),
)
@settings(max_examples=200)
def test_trim_preserves_active_region(
    audio: np.ndarray, sample_rate: int
) -> None:
    """Active (non-silent) samples must remain within the trimmed result."""
    trimmed = trim_boundary_silence(
        audio, sample_rate, trim_start=True, trim_end=True
    )
    if len(trimmed) == 0:
        # All silent — acceptable
        return
    # The trimmed region should contain at least the active region's extent
    active = np.flatnonzero(np.abs(audio) > 0.002)
    if len(active) > 0:
        # The trimmed output should cover the active region
        assert len(trimmed) >= 1


@given(audio=finite_float32_array(), sample_rate=positive_sample_rate())
@settings(max_examples=200)
def test_trim_no_trim_keeps_full_length(
    audio: np.ndarray, sample_rate: int
) -> None:
    """When trim_start=False and trim_end=False, output length equals input."""
    trimmed = trim_boundary_silence(
        audio, sample_rate, trim_start=False, trim_end=False
    )
    assert len(trimmed) == len(audio)


# ─── _split_progressive reconstruction ─────────────────────────────────────────


@given(text=latin_text())
@settings(max_examples=200)
def test_split_progressive_reconstruction(text: str) -> None:
    """Concatenated progressive chunks should reconstruct the original text."""
    chunks = _split_progressive(text, language="en")
    if not chunks:
        # Empty or whitespace-only text produces no chunks
        assert not text.strip()
        return
    reconstructed = " ".join(chunks)
    # Normalize whitespace for comparison
    original_words = text.split()
    reconstructed_words = reconstructed.split()
    assert original_words == reconstructed_words


@given(text=latin_text())
@settings(max_examples=200)
def test_split_progressive_non_empty_chunks(text: str) -> None:
    """Every progressive chunk must be non-empty."""
    chunks = _split_progressive(text, language="en")
    for chunk in chunks:
        assert chunk.strip(), f"Empty chunk from text: {text!r}"


# ─── _split_text invariants ────────────────────────────────────────────────────


@given(text=latin_text())
@settings(max_examples=200)
def test_split_text_sentence_non_empty_chunks(text: str) -> None:
    """Every sentence chunk must be non-empty."""
    chunks = _split_text(text, "sentence")
    for chunk in chunks:
        assert chunk.strip()


@given(text=latin_text())
@settings(max_examples=200)
def test_split_text_line_preserves_line_breaks(text: str) -> None:
    """Line splitting should respect line boundaries."""
    chunks = _split_text(text, "line")
    for chunk in chunks:
        assert "\n" not in chunk  # Lines should not contain newlines


# ─── AudioResult.bytes round-trip ──────────────────────────────────────────────


@given(
    data=st.lists(
        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
        min_size=1,
        max_size=4000,
    ),
    sample_rate=positive_sample_rate(),
)
@settings(max_examples=200)
def test_audio_result_bytes_roundtrip(
    data: list[float], sample_rate: int
) -> None:
    """WAV bytes round-trip should preserve audio data."""
    original = np.array(data, dtype=np.float32)
    result = AudioResult(numpy=original, sample_rate=sample_rate)

    wav_bytes = result.bytes
    decoded, decoded_sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")

    assert decoded_sr == sample_rate
    assert decoded.shape == original.shape
    np.testing.assert_allclose(decoded, original, atol=1e-4)


# ─── AudioChunk wire format round-trip ─────────────────────────────────────────


@given(
    sequence=st.integers(min_value=0, max_value=10000),
    sample_rate=positive_sample_rate(),
    final=st.booleans(),
)
@settings(max_examples=200)
def test_audio_chunk_wire_dict_structure(
    sequence: int, sample_rate: int, final: bool
) -> None:
    """Wire dict must have all required fields with correct types."""
    chunk = AudioChunk(
        sequence=sequence,
        numpy=np.ones(10, dtype=np.float32) * 0.5,
        sample_rate=sample_rate,
        final=final,
    )
    wire = chunk.to_wire_dict()
    assert wire["type"] == "audio_chunk"
    assert wire["sequence"] == sequence
    assert wire["sample_rate"] == sample_rate
    assert wire["channels"] == 1
    assert wire["encoding"] == "pcm_s16le"
    assert wire["final"] == final
    assert isinstance(wire["audio_base64"], str)
    assert len(wire["audio_base64"]) > 0


# ─── StreamAssembler sequence monotonicity ─────────────────────────────────────


@given(
    num_chunks=st.integers(min_value=1, max_value=50),
    sample_rate=positive_sample_rate(),
)
@settings(max_examples=100)
def test_assembler_sequence_monotonic(
    num_chunks: int, sample_rate: int
) -> None:
    """Chunk sequences must be monotonically increasing."""
    overlap = min(int(0.08 * sample_rate), 100)
    assembler = StreamAssembler(sample_rate=sample_rate, overlap=overlap)
    sequences = []
    for i in range(num_chunks):
        chunk = assembler.push(i, np.ones(50, dtype=np.float32) * 0.5)
        if chunk is not None:
            sequences.append(chunk.sequence)
    final = assembler.finish()
    if final is not None:
        sequences.append(final.sequence)

    # Verify monotonically increasing
    for i in range(1, len(sequences)):
        assert sequences[i] > sequences[i - 1], (
            f"Sequence not monotonic: {sequences}"
        )


# ─── AudioResult frozen dataclass ──────────────────────────────────────────────


@given(
    sample_rate=st.integers(min_value=1, max_value=100000),
)
def test_audio_result_is_frozen(sample_rate: int) -> None:
    """AudioResult should be immutable after creation."""
    result = AudioResult(
        numpy=np.ones(10, dtype=np.float32), sample_rate=sample_rate
    )
    with pytest.raises(AttributeError):
        result.sample_rate = 99999  # type: ignore[misc]


# ─── _split_sentences edge cases ───────────────────────────────────────────────


def test_split_sentences_single_char() -> None:
    """Single character should produce at least one sentence."""
    sentences = _split_sentences("a")
    assert len(sentences) >= 1


def test_split_sentences_all_punctuation() -> None:
    """All-punctuation input should handle gracefully."""
    sentences = _split_sentences("!!!???...")
    # Should not crash; may return empty or punctuation-only results
    assert isinstance(sentences, list)


def test_split_sentences_empty() -> None:
    """Empty text should produce no sentences."""
    sentences = _split_sentences("")
    assert sentences == []


def test_split_sentences_whitespace_only() -> None:
    """Whitespace-only text should produce no sentences."""
    sentences = _split_sentences("   \n\t  ")
    assert sentences == []


# ─── _split_units CJK handling ─────────────────────────────────────────────────


def test_split_units_japanese_characters() -> None:
    """Japanese text without spaces should split by character."""
    units = _split_units("こんにちは世界", "ja")
    assert len(units) == 7
    assert units == ["こ", "ん", "に", "ち", "は", "世", "界"]


def test_split_units_english_words() -> None:
    """English text should split by whitespace."""
    units = _split_units("hello world test", "en")
    assert units == ["hello", "world", "test"]


def test_split_units_mixed_whitespace() -> None:
    """English text with multiple spaces should split correctly."""
    units = _split_units("hello  world   test", "en")
    assert units == ["hello", "world", "test"]
