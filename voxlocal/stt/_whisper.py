# SPDX-License-Identifier: MIT
from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock

from voxlocal._errors import DependencyMissingError

logger = logging.getLogger("voxlocal.stt.whisper")


class WhisperSTT:
    """Whisper-based STT engine.

    Thread-safe: model loading and transcription are protected by a lock.
    """

    def __init__(self, language: str = "auto", model_dir: str | Path | None = None):
        self.language = language
        self.model_dir = Path(model_dir).expanduser() if model_dir else None
        self._model: object = None
        self._model_lock = Lock()

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        with self._model_lock:
            # Double-check after acquiring lock
            if self._model is not None:
                return
            try:
                import whisper
            except ImportError as error:
                raise DependencyMissingError("openai-whisper", "whisper") from error

            logger.debug("Loading Whisper model...")
            self._model = whisper.load_model(
                "base",
                download_root=str(self.model_dir) if self.model_dir else None,
                in_memory=False,
            )
            logger.info("Whisper model loaded for language=%s", self.language)

    def warmup(self) -> None:
        """Pre-initialize model and resources."""
        self._ensure_model()

    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file to text.

        Args:
            audio_path: Path to an audio file.

        Returns:
            Transcribed text.

        Raises:
            DependencyMissingError: If openai-whisper is not installed.
        """
        self._ensure_model()
        assert self._model is not None
        result = self._model.transcribe(  # type: ignore[attr-defined]
            audio_path,
            language=self.language if self.language != "auto" else None,
        )
        return str(result["text"]).strip()
