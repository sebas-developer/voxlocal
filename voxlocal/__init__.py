from __future__ import annotations

from typing import Iterator
from pathlib import Path

from voxlocal._version import __version__
from voxlocal._errors import (
    VoxLocalError,
    LanguageNotSupportedError,
    ModelNotDownloadedError,
    TranscriptionError,
    SynthesisError,
)
from voxlocal._audio import AudioResult
from voxlocal._config import get_stt_config, get_tts_config, SUPPORTED_LANGUAGES
from voxlocal._download import DownloadManager, DownloadProgress


class VoxLocal:
    """Local STT/TTS with zero setup."""

    def __init__(
        self,
        language: str,
        stt_engine: str | None = None,
        tts_engine: str | None = None,
    ):
        if language not in SUPPORTED_LANGUAGES:
            raise LanguageNotSupportedError(language)

        self.language = language
        self._stt_engine_name = stt_engine
        self._tts_engine_name = tts_engine
        self._stt_config = get_stt_config(language)
        self._tts_config = get_tts_config(language)
        self._download_manager = DownloadManager()
        self._stt = None
        self._tts = None

    def setup(self) -> Iterator[DownloadProgress]:
        """Download all models for configured language."""
        yield from self._download_manager.download(self._stt_config["model_id"])
        yield from self._download_manager.download(self._tts_config["model_id"])

    def download_model(self, model_id: str) -> Iterator[DownloadProgress]:
        """Download a specific model."""
        yield from self._download_manager.download(model_id)

    def _ensure_stt(self) -> None:
        if self._stt is None:
            engine_name = self._stt_engine_name or self._stt_config["engine"]
            if not self._download_manager.is_downloaded(self._stt_config["model_id"]):
                raise ModelNotDownloadedError(self._stt_config["model_id"])
            from voxlocal.stt import resolve_stt_engine

            self._stt = resolve_stt_engine(engine_name, self.language)

    def _ensure_tts(self) -> None:
        if self._tts is None:
            engine_name = self._tts_engine_name or self._tts_config["engine"]
            if not self._download_manager.is_downloaded(self._tts_config["model_id"]):
                raise ModelNotDownloadedError(self._tts_config["model_id"])
            from voxlocal.tts import resolve_tts_engine

            self._tts = resolve_tts_engine(engine_name, self.language)

    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file to text."""
        self._ensure_stt()
        try:
            return self._stt.transcribe(audio_path)
        except Exception as e:
            raise TranscriptionError(str(e)) from e

    def speak(self, text: str) -> AudioResult:
        """Synthesize text to audio."""
        self._ensure_tts()
        try:
            return self._tts.speak(text)
        except Exception as e:
            raise SynthesisError(str(e)) from e

    def speak_iter(
        self, text: str, chunk_by: str = "sentence"
    ) -> Iterator[AudioResult]:
        """Synthesize text to audio chunks."""
        self._ensure_tts()
        try:
            yield from self._tts.speak_iter(text, chunk_by)
        except Exception as e:
            raise SynthesisError(str(e)) from e


__all__ = [
    "__version__",
    "VoxLocal",
    "AudioResult",
    "DownloadProgress",
    "VoxLocalError",
    "LanguageNotSupportedError",
    "ModelNotDownloadedError",
    "TranscriptionError",
    "SynthesisError",
    "SUPPORTED_LANGUAGES",
]
