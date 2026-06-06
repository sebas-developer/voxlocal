from __future__ import annotations

from typing import Protocol, Iterator
from voxlocal._audio import AudioResult


class TTSEngine(Protocol):
    """Protocol for TTS engines."""

    def speak(self, text: str) -> AudioResult:
        """Synthesize text to audio."""
        ...

    def speak_iter(
        self, text: str, chunk_by: str = "sentence"
    ) -> Iterator[AudioResult]:
        """Synthesize text to audio chunks."""
        ...


def resolve_tts_engine(engine_name: str, language: str) -> TTSEngine:
    """Resolve TTS engine by name."""
    if engine_name == "supertonic":
        from voxlocal.tts._supertonic import SupertonicTTS

        return SupertonicTTS(language=language)
    else:
        raise ValueError(f"Unknown TTS engine: {engine_name}")
