import numpy as np
import os
import tempfile
from voxlocal._audio import AudioResult


def test_audio_result_properties():
    data = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    result = AudioResult(numpy=data, sample_rate=24000)
    assert np.array_equal(result.numpy, data)
    assert result.sample_rate == 24000


def test_audio_result_bytes():
    data = np.array([0.0, 0.5, -0.5], dtype=np.float32)
    result = AudioResult(numpy=data, sample_rate=16000)
    b = result.bytes
    assert isinstance(b, bytes)
    assert len(b) > 0
    assert b[:4] == b"RIFF"  # WAV header


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
