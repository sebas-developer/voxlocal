from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


def _mono_float32(audio: np.ndarray) -> np.ndarray:
    """Normalize generated audio to a contiguous mono float32 array."""
    normalized = np.asarray(audio, dtype=np.float32)
    if normalized.ndim == 2 and 1 in normalized.shape:
        normalized = normalized.reshape(-1)
    if normalized.ndim != 1:
        raise ValueError("Audio must be a mono one-dimensional array")
    return np.ascontiguousarray(normalized)


def _pcm_s16le(audio: np.ndarray) -> bytes:
    """Encode normalized floating-point audio as portable signed PCM16."""
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


@dataclass(frozen=True)
class AudioResult:
    """Result of TTS synthesis."""

    numpy: np.ndarray
    sample_rate: int

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        object.__setattr__(self, "numpy", _mono_float32(self.numpy))

    @property
    def duration_seconds(self) -> float:
        """Audio duration in seconds."""
        return len(self.numpy) / self.sample_rate

    @property
    def bytes(self) -> bytes:
        """Convert to WAV bytes."""
        buf = io.BytesIO()
        sf.write(buf, self.numpy, self.sample_rate, format="WAV")
        return buf.getvalue()

    @property
    def pcm_s16le(self) -> bytes:
        """Raw mono signed 16-bit little-endian PCM."""
        return _pcm_s16le(self.numpy)

    def save(self, path: str | Path) -> None:
        """Save as WAV file."""
        sf.write(path, self.numpy, self.sample_rate)


@dataclass(frozen=True)
class AudioChunk:
    """Portable streaming audio block independent of playback libraries."""

    sequence: int
    numpy: np.ndarray
    sample_rate: int
    final: bool = False
    source_chunk: int | None = None

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        object.__setattr__(self, "numpy", _mono_float32(self.numpy))

    @property
    def duration_seconds(self) -> float:
        """Audio duration in seconds."""
        return len(self.numpy) / self.sample_rate

    @property
    def pcm_s16le(self) -> bytes:
        """Raw mono signed 16-bit little-endian PCM."""
        return _pcm_s16le(self.numpy)

    def to_wire_dict(self) -> dict[str, Any]:
        """Serialize metadata and PCM for NDJSON or message transports."""
        return {
            "type": "audio_chunk",
            "sequence": self.sequence,
            "sample_rate": self.sample_rate,
            "channels": 1,
            "encoding": "pcm_s16le",
            "final": self.final,
            "source_chunk": self.source_chunk,
            "audio_base64": base64.b64encode(self.pcm_s16le).decode("ascii"),
        }
