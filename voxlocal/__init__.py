from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from voxlocal._audio import AudioChunk, AudioResult
from voxlocal._config import (
    SUPPORTED_LANGUAGES,
    SUPPORTED_STT_LANGUAGES,
    SUPPORTED_TTS_LANGUAGES,
    EngineConfig,
    get_stt_config,
    get_tts_config,
)
from voxlocal._download import DownloadManager, DownloadProgress
from voxlocal._errors import (
    DependencyMissingError,
    EngineNotSupportedError,
    LanguageNotSupportedError,
    ModelDownloadError,
    ModelNotDownloadedError,
    SynthesisError,
    TranscriptionError,
    VoxLocalError,
)
from voxlocal._stream import DEFAULT_PREFETCH, assemble_stream
from voxlocal._version import __version__

if TYPE_CHECKING:
    from voxlocal.playback import PlaybackEvent


class VoxLocal:
    """Local STT/TTS facade with explicit setup and portable streaming."""

    def __init__(
        self,
        language: str,
        stt_engine: str | None = None,
        tts_engine: str | None = None,
        *,
        cache_dir: str | Path | None = None,
    ):
        if language not in SUPPORTED_LANGUAGES:
            raise LanguageNotSupportedError(language)

        self.language = language
        self._stt_config = self._optional_config(
            get_stt_config, language, stt_engine
        )
        self._tts_config = self._optional_config(
            get_tts_config, language, tts_engine
        )
        self._download_manager = DownloadManager(cache_dir)
        self._stt = None
        self._tts = None

    @staticmethod
    def _optional_config(
        resolver: Callable[[str, str | None], EngineConfig],
        language: str,
        engine: str | None,
    ) -> EngineConfig | None:
        try:
            return resolver(language, engine)
        except LanguageNotSupportedError:
            if engine is not None:
                raise
            return None

    def setup(
        self,
        *,
        stt: bool = True,
        tts: bool = True,
        warmup_tts: bool = True,
    ) -> list[DownloadProgress]:
        """Eagerly download selected models and optionally warm TTS."""
        return list(
            self.setup_iter(stt=stt, tts=tts, warmup_tts=warmup_tts)
        )

    def setup_iter(
        self,
        *,
        stt: bool = True,
        tts: bool = True,
        warmup_tts: bool = True,
    ) -> Iterator[DownloadProgress]:
        """Yield model setup progress while performing setup."""
        selected: list[EngineConfig] = []
        if stt and self._stt_config is not None:
            selected.append(self._stt_config)
        if tts and self._tts_config is not None:
            selected.append(self._tts_config)

        seen: set[str] = set()
        for config in selected:
            if config.model_id in seen:
                continue
            seen.add(config.model_id)
            yield from self._download_manager.download(config.model_id)

        if tts and warmup_tts and self._tts_config is not None:
            self._ensure_tts()
            self._tts.warmup()

    def download_model(self, model_id: str) -> list[DownloadProgress]:
        """Eagerly download one model and return its progress records."""
        return list(self.download_model_iter(model_id))

    def download_model_iter(
        self, model_id: str
    ) -> Iterator[DownloadProgress]:
        """Yield progress while downloading one model."""
        yield from self._download_manager.download(model_id)

    def _require_capability(
        self, config: EngineConfig | None, capability: str
    ) -> EngineConfig:
        if config is None:
            raise LanguageNotSupportedError(self.language, capability)
        return config

    def _ensure_stt(self) -> None:
        if self._stt is not None:
            return
        config = self._require_capability(self._stt_config, "stt")
        model_dir = self._download_manager.require_downloaded(config.model_id)
        from voxlocal.stt import resolve_stt_engine

        self._stt = resolve_stt_engine(
            config.engine, self.language, model_dir
        )

    def _ensure_tts(self) -> None:
        if self._tts is not None:
            return
        config = self._require_capability(self._tts_config, "tts")
        model_dir = self._download_manager.require_downloaded(config.model_id)
        from voxlocal.tts import resolve_tts_engine

        self._tts = resolve_tts_engine(
            config.engine, self.language, model_dir
        )

    def transcribe(self, audio_path: str | Path) -> str:
        """Transcribe one audio file."""
        self._ensure_stt()
        try:
            return self._stt.transcribe(str(audio_path))
        except VoxLocalError:
            raise
        except Exception as error:
            raise TranscriptionError(str(error)) from error

    def speak(self, text: str) -> AudioResult:
        """Synthesize one complete higher-quality result."""
        self._ensure_tts()
        try:
            return self._tts.speak(text)
        except VoxLocalError:
            raise
        except Exception as error:
            raise SynthesisError(str(error)) from error

    def speak_iter(
        self, text: str, chunk_by: str = "progressive"
    ) -> Iterator[AudioResult]:
        """Yield raw engine results as each text chunk is generated."""
        self._ensure_tts()
        try:
            yield from self._tts.speak_iter(text, chunk_by)
        except VoxLocalError:
            raise
        except Exception as error:
            raise SynthesisError(str(error)) from error

    def stream(
        self,
        text: str,
        *,
        chunk_by: str = "progressive",
        prefetch: int = DEFAULT_PREFETCH,
    ) -> Iterator[AudioChunk]:
        """Yield portable, trimmed and crossfaded PCM-ready blocks."""
        return assemble_stream(
            self.speak_iter(text, chunk_by=chunk_by),
            prefetch=prefetch,
        )

    def play(
        self,
        text: str,
        *,
        on_event: Callable[[PlaybackEvent], None] | None = None,
        device: int | str | None = None,
        prefetch: int = DEFAULT_PREFETCH,
    ) -> None:
        """Convenience wrapper around the optional playback adapter."""
        from voxlocal.playback import play

        play(
            self.stream(text, prefetch=prefetch),
            on_event=on_event,
            device=device,
        )


__all__ = [
    "__version__",
    "AudioChunk",
    "AudioResult",
    "DependencyMissingError",
    "DownloadProgress",
    "EngineNotSupportedError",
    "LanguageNotSupportedError",
    "ModelDownloadError",
    "ModelNotDownloadedError",
    "SUPPORTED_LANGUAGES",
    "SUPPORTED_STT_LANGUAGES",
    "SUPPORTED_TTS_LANGUAGES",
    "SynthesisError",
    "TranscriptionError",
    "VoxLocal",
    "VoxLocalError",
]
