# SPDX-License-Identifier: MIT
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

from voxlocal._audio import AudioResult


class TTSEngine(Protocol):
    """Protocol for text-to-speech engines.

    All TTS engines must implement this interface to be compatible with
    the VoxLocal facade.
    """

    def speak(self, text: str) -> AudioResult:
        """Synthesize text to a complete audio result.

        Args:
            text: Text to synthesize.

        Returns:
            Complete audio result.
        """
        ...

    def speak_iter(
        self, text: str, chunk_by: str = "progressive"
    ) -> Iterator[AudioResult]:
        """Synthesize text to audio chunks.

        Args:
            text: Text to synthesize.
            chunk_by: Chunking strategy.

        Yields:
            AudioResult for each text chunk.
        """
        ...

    def warmup(self) -> None:
        """Pre-initialize model and resources."""
        ...


def resolve_tts_engine(engine_name: str, language: str, model_dir: Path) -> TTSEngine:
    """Resolve TTS engine by name.

    Args:
        engine_name: Engine identifier (e.g. 'supertonic').
        language: ISO 639-1 language code.
        model_dir: Path to the model directory.

    Returns:
        Configured TTS engine instance.

    Raises:
        ValueError: If engine_name is not recognized.
    """
    if engine_name == "supertonic":
        from voxlocal.tts._supertonic import SupertonicTTS

        return SupertonicTTS(language=language, model_dir=model_dir)
    else:
        raise ValueError(f"Unknown TTS engine: {engine_name}")
