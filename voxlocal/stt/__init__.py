# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path
from typing import Protocol


class STTEngine(Protocol):
    """Protocol for speech-to-text engines.

    All STT engines must implement this interface to be compatible with
    the VoxLocal facade.
    """

    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file to text.

        Args:
            audio_path: Path to an audio file.

        Returns:
            Transcribed text.
        """
        ...

    def warmup(self) -> None:
        """Pre-initialize model and resources."""
        ...


def resolve_stt_engine(engine_name: str, language: str, model_dir: Path) -> STTEngine:
    """Resolve STT engine by name.

    Args:
        engine_name: Engine identifier (e.g. 'whisper', 'moonshine').
        language: ISO 639-1 language code.
        model_dir: Path to the model directory.

    Returns:
        Configured STT engine instance.

    Raises:
        ValueError: If engine_name is not recognized.
    """
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
