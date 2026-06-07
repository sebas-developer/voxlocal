from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

from voxlocal._audio import AudioResult


class TTSEngine(Protocol):
    """Protocol for TTS engines."""

    def speak(self, text: str) -> AudioResult:
        """Synthesize text to audio."""
        ...

    def speak_iter(
        self, text: str, chunk_by: str = "progressive"
    ) -> Iterator[AudioResult]:
        """Synthesize text to audio chunks."""
        ...

    def warmup(self) -> None:
        """Pre-initialize model and resources."""
        ...


def resolve_tts_engine(
    engine_name: str, language: str, model_dir: Path
) -> TTSEngine:
    """Resolve TTS engine by name."""
    if engine_name == "supertonic":
        from voxlocal.tts._supertonic import SupertonicTTS

        return SupertonicTTS(language=language, model_dir=model_dir)
    else:
        raise ValueError(f"Unknown TTS engine: {engine_name}")
