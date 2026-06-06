from __future__ import annotations

import re
import numpy as np
from typing import Iterator
from voxlocal._audio import AudioResult


def _split_text(text: str, chunk_by: str) -> list[str]:
    if chunk_by == "sentence":
        parts = re.split(r"(?<=[.!?])\s+", text)
    elif chunk_by == "line":
        parts = text.split("\n")
    elif chunk_by == "paragraph":
        parts = re.split(r"\n\s*\n", text)
    else:
        parts = [text]
    return [p for p in parts if p.strip()]


class SupertonicTTS:
    """Supertonic TTS engine."""

    def __init__(self, language: str = "en"):
        self.language = language
        self._tts = None

    def _ensure_model(self) -> None:
        if self._tts is None:
            from supertonic import TTS

            self._tts = TTS(lang=self.language)

    def speak(self, text: str) -> AudioResult:
        self._ensure_model()
        import soundfile as sf
        import io

        wav_bytes = self._tts.synthesize(text)
        data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        return AudioResult(numpy=data, sample_rate=sr)

    def speak_iter(
        self,
        text: str,
        chunk_by: str = "sentence",  # valid: "sentence", "line", "paragraph"
    ) -> Iterator[AudioResult]:
        chunks = _split_text(text, chunk_by)
        for chunk in chunks:
            result = self.speak(chunk)
            yield result
