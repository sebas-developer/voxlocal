# SPDX-License-Identifier: MIT
from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock

from voxlocal._errors import DependencyMissingError

logger = logging.getLogger("voxlocal.stt.moonshine")

MOONSHINE_ES_MODEL_PATH = Path("download.moonshine.ai/model/base-es/quantized/base-es")


class MoonshineSTT:
    """Moonshine-based STT engine.

    Thread-safe: model loading and transcription are protected by a lock.

    Raises:
        ValueError: When language is not 'es' (only supported language).
        DependencyMissingError: When moonshine-voice is not installed.
    """

    def __init__(self, language: str = "es", model_dir: str | Path | None = None):
        if language != "es":
            raise ValueError(
                f"Moonshine does not support language '{language}'. Supported: es"
            )
        self.language = language
        self.model_dir = Path(model_dir).expanduser() if model_dir else None
        self._transcriber: object = None
        self._model_lock = Lock()

    def _ensure_model(self) -> None:
        if self._transcriber is not None:
            return
        with self._model_lock:
            if self._transcriber is not None:
                return
            try:
                from moonshine_voice import ModelArch, Transcriber
            except ImportError as error:
                raise DependencyMissingError("moonshine-voice", "moonshine") from error

            if self.model_dir is None:
                raise RuntimeError("Moonshine model_dir is required")
            model_path = self.model_dir / MOONSHINE_ES_MODEL_PATH
            logger.debug("Loading Moonshine model...")
            self._transcriber = Transcriber(
                model_path=model_path, model_arch=ModelArch.BASE
            )
            logger.info("Moonshine model loaded for language=%s", self.language)

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
            DependencyMissingError: If moonshine-voice is not installed.
        """
        self._ensure_model()
        assert self._transcriber is not None
        from moonshine_voice import load_wav_file

        audio_data, sample_rate = load_wav_file(audio_path)
        transcript = self._transcriber.transcribe_without_streaming(  # type: ignore[attr-defined]
            audio_data, sample_rate
        )
        return (
            " ".join(line.text for line in transcript.lines) if transcript.lines else ""
        )
