# SPDX-License-Identifier: MIT
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from voxlocal._audio import AudioChunk, AudioResult


def test_audio_result_properties():
    data = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    result = AudioResult(numpy=data, sample_rate=24000)
    assert np.array_equal(result.numpy, data)
    assert result.sample_rate == 24000


def test_audio_result_duration_seconds():
    data = np.zeros(48000, dtype=np.float32)
    result = AudioResult(numpy=data, sample_rate=48000)
    assert result.duration_seconds == 1.0


def test_audio_result_bytes():
    data = np.array([0.0, 0.5, -0.5], dtype=np.float32)
    result = AudioResult(numpy=data, sample_rate=16000)
    b = result.bytes
    assert isinstance(b, bytes)
    assert len(b) > 0
    assert b[:4] == b"RIFF"  # WAV header


def test_audio_result_pcm_s16le():
    data = np.array([0.0, 0.5, -0.5], dtype=np.float32)
    result = AudioResult(numpy=data, sample_rate=16000)
    pcm = result.pcm_s16le
    assert isinstance(pcm, bytes)
    assert len(pcm) == 6  # 3 samples * 2 bytes


def test_audio_result_save():
    data = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    result = AudioResult(numpy=data, sample_rate=24000)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    try:
        result.save(path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
    finally:
        os.unlink(path)


def test_audio_result_rejects_multichannel_data():
    with pytest.raises(ValueError, match="mono"):
        AudioResult(np.zeros((2, 20), dtype=np.float32), 16000)


def test_audio_result_rejects_negative_sample_rate():
    with pytest.raises(ValueError, match="positive"):
        AudioResult(np.ones(10, dtype=np.float32), -1)


def test_audio_result_rejects_zero_sample_rate():
    with pytest.raises(ValueError, match="positive"):
        AudioResult(np.ones(10, dtype=np.float32), 0)


def test_audio_result_normalizes_to_mono():
    # Single-row 2D array should be squeezed to 1D
    data = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    result = AudioResult(numpy=data, sample_rate=16000)
    assert result.numpy.ndim == 1


def test_audio_result_contiguous():
    data = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    result = AudioResult(numpy=data, sample_rate=16000)
    assert result.numpy.flags["C_CONTIGUOUS"]


# ─── AudioChunk ─────────────────────────────────────────────────────────────────


def test_audio_chunk_wire_format_is_portable():
    chunk = AudioChunk(
        sequence=2,
        numpy=np.array([-1.0, 0.0, 1.0], dtype=np.float32),
        sample_rate=16000,
        final=True,
    )

    wire = chunk.to_wire_dict()

    assert wire["encoding"] == "pcm_s16le"
    assert wire["channels"] == 1
    assert wire["sample_rate"] == 16000
    assert wire["final"] is True
    assert isinstance(wire["audio_base64"], str)


def test_audio_chunk_duration_seconds():
    chunk = AudioChunk(
        sequence=0,
        numpy=np.zeros(48000, dtype=np.float32),
        sample_rate=48000,
    )
    assert chunk.duration_seconds == 1.0


def test_audio_chunk_pcm_s16le():
    chunk = AudioChunk(
        sequence=0,
        numpy=np.array([0.0, 0.5, -0.5], dtype=np.float32),
        sample_rate=16000,
    )
    pcm = chunk.pcm_s16le
    assert isinstance(pcm, bytes)
    assert len(pcm) == 6


def test_audio_chunk_rejects_negative_sequence():
    with pytest.raises(ValueError, match="non-negative"):
        AudioChunk(
            sequence=-1,
            numpy=np.ones(10, dtype=np.float32),
            sample_rate=16000,
        )


def test_audio_chunk_rejects_negative_sample_rate():
    with pytest.raises(ValueError, match="positive"):
        AudioChunk(
            sequence=0,
            numpy=np.ones(10, dtype=np.float32),
            sample_rate=-1,
        )


def test_audio_chunk_frozen():
    chunk = AudioChunk(
        sequence=0,
        numpy=np.ones(10, dtype=np.float32),
        sample_rate=16000,
    )
    with pytest.raises(AttributeError):
        chunk.sequence = 1  # type: ignore[misc]


def test_audio_result_frozen():
    result = AudioResult(
        numpy=np.ones(10, dtype=np.float32),
        sample_rate=16000,
    )
    with pytest.raises(AttributeError):
        result.sample_rate = 48000  # type: ignore[misc]
