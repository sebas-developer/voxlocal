from __future__ import annotations

import io
import numpy as np
import soundfile as sf


class AudioResult:
    """Result of TTS synthesis."""

    def __init__(self, numpy: np.ndarray, sample_rate: int):
        self.numpy = numpy
        self.sample_rate = sample_rate

    @property
    def bytes(self) -> bytes:
        """Convert to WAV bytes."""
        buf = io.BytesIO()
        sf.write(buf, self.numpy, self.sample_rate, format="WAV")
        return buf.getvalue()

    def save(self, path: str) -> None:
        """Save as WAV file."""
        sf.write(path, self.numpy, self.sample_rate)
