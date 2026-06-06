from __future__ import annotations

from typing import Protocol


class STTEngine(Protocol):
    """Protocol for STT engines."""

    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file to text."""
        ...


def resolve_stt_engine(engine_name: str, language: str) -> STTEngine:
    """Resolve STT engine by name."""
    if engine_name == "whisper":
        from voxlocal.stt._whisper import WhisperSTT

        return WhisperSTT(language=language)
    elif engine_name == "moonshine":
        from voxlocal.stt._moonshine import MoonshineSTT

        return MoonshineSTT(language=language)
    elif engine_name == "sensevoice":
        from voxlocal.stt._sensevoice import SenseVoiceSTT

        return SenseVoiceSTT(language=language)
    else:
        raise ValueError(f"Unknown STT engine: {engine_name}")
