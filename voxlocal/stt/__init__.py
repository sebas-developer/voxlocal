from __future__ import annotations

from pathlib import Path
from typing import Protocol


class STTEngine(Protocol):
    """Protocol for STT engines."""

    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file to text."""
        ...

    def warmup(self) -> None:
        """Pre-initialize model and resources."""
        ...


def resolve_stt_engine(
    engine_name: str, language: str, model_dir: Path
) -> STTEngine:
    """Resolve STT engine by name."""
    if engine_name == "whisper":
        from voxlocal.stt._whisper import WhisperSTT

        return WhisperSTT(language=language, model_dir=model_dir)
    elif engine_name == "moonshine":
        from voxlocal.stt._moonshine import MoonshineSTT

        return MoonshineSTT(language=language, model_dir=model_dir)
    elif engine_name == "sensevoice":
        from voxlocal.stt._sensevoice import SenseVoiceSTT

        return SenseVoiceSTT(language=language, model_dir=model_dir)
    else:
        raise ValueError(f"Unknown STT engine: {engine_name}")
