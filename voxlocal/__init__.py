# SPDX-License-Identifier: MIT
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from voxlocal._audio import AudioChunk, AudioResult
from voxlocal._config import (
    SUPPORTED_LANGUAGES,
    SUPPORTED_STT_LANGUAGES,
    SUPPORTED_TTS_LANGUAGES,
    ChunkBy,
    EngineConfig,
    VoxLocalConfig,
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
from voxlocal._metrics import (
    MetricsCollector,
    TimingContext,
    create_metrics,
)
from voxlocal._stream import DEFAULT_PREFETCH, assemble_stream
from voxlocal._version import __version__

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from voxlocal.playback import PlaybackEvent
    from voxlocal.stt import STTEngine
    from voxlocal.tts import TTSEngine

logger = logging.getLogger("voxlocal")


class VoxLocal:
    """Local STT/TTS facade with explicit setup and portable streaming.

    Args:
        language: ISO 639-1 language code (e.g. 'en', 'es', 'ja').
        stt_engine: Optional engine override for speech-to-text.
        tts_engine: Optional engine override for text-to-speech.
        cache_dir: Optional directory for model cache storage.
        config: Optional VoxLocalConfig for centralized settings.
        metrics: Optional MetricsCollector for observability callbacks.

    Example:
        >>> from voxlocal import VoxLocal
        >>> v = VoxLocal("es")
        >>> v.setup()
        >>> audio = v.speak("Hola mundo")

        Using metrics:
        >>> from voxlocal import create_metrics
        >>> def on_complete(duration, audio_len):
        ...     print(f"Synthesis took {duration:.2f}s for {audio_len:.2f}s audio")
        >>> metrics = create_metrics(on_synthesis_complete=on_complete)
        >>> v = VoxLocal("en", metrics=metrics)
    """

    def __init__(
        self,
        language: str,
        stt_engine: str | None = None,
        tts_engine: str | None = None,
        *,
        cache_dir: str | Path | None = None,
        config: VoxLocalConfig | None = None,
        metrics: MetricsCollector | None = None,
    ):
        if language not in SUPPORTED_LANGUAGES:
            raise LanguageNotSupportedError(language)

        self.language = language
        self._config = config or VoxLocalConfig()
        self._metrics = metrics
        self._stt_config = self._optional_config(get_stt_config, language, stt_engine)
        self._tts_config = self._optional_config(get_tts_config, language, tts_engine)
        self._download_manager = DownloadManager(
            cache_dir or self._config.cache_dir
        )
        self._stt: STTEngine | None = None
        self._tts: TTSEngine | None = None
        self._closed = False

    def __repr__(self) -> str:
        stt_eng = self._stt_config.engine if self._stt_config else None
        tts_eng = self._tts_config.engine if self._tts_config else None
        loaded: list[str] = []
        if self._stt is not None:
            loaded.append("stt")
        if self._tts is not None:
            loaded.append("tts")
        caps = ", ".join(loaded) if loaded else "none"
        return (
            f"VoxLocal(language={self.language!r}, "
            f"stt={stt_eng}, tts={tts_eng}, loaded={caps})"
        )

    def __enter__(self) -> VoxLocal:
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: Exception | None,
        exc_tb: object,
    ) -> None:
        self.close()

    async def __aenter__(self) -> VoxLocal:
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: type | None,
        exc_val: Exception | None,
        exc_tb: object,
    ) -> None:
        """Async context manager exit — releases model resources."""
        self.close()

    def close(self) -> None:
        """Release model resources."""
        if self._closed:
            return
        self._closed = True
        if self._stt is not None:
            # Release model reference to free memory
            self._stt = None
            logger.debug("Released STT engine for %s", self.language)
        if self._tts is not None:
            self._tts = None
            logger.debug("Released TTS engine for %s", self.language)

    def __del__(self) -> None:
        if getattr(self, "_closed", True):
            return
        self.close()

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
        return list(self.setup_iter(stt=stt, tts=tts, warmup_tts=warmup_tts))

    def setup_iter(
        self,
        *,
        stt: bool = True,
        tts: bool = True,
        warmup_tts: bool = True,
    ) -> Iterator[DownloadProgress]:
        """Yield model setup progress while performing setup.

        Downloads STT and TTS models concurrently when both are requested.

        Args:
            stt: Download STT model.
            tts: Download TTS model.
            warmup_tts: Pre-initialize TTS engine after download.

        Yields:
            DownloadProgress records for each lifecycle event.
        """
        import concurrent.futures

        selected: list[EngineConfig] = []
        if stt and self._stt_config is not None:
            selected.append(self._stt_config)
        if tts and self._tts_config is not None:
            selected.append(self._tts_config)

        # Collect unique model IDs to download
        model_ids: list[str] = []
        seen: set[str] = set()
        for config in selected:
            if config.model_id not in seen:
                seen.add(config.model_id)
                model_ids.append(config.model_id)

        if len(model_ids) <= 1:
            # Sequential download for single model
            for model_id in model_ids:
                yield from self._download_manager.download(model_id)
        else:
            # Concurrent download for multiple models
            futures: dict[concurrent.futures.Future[None], str] = {}
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(model_ids)
            ) as pool:
                for model_id in model_ids:
                    # Consume the iterator in the worker
                    def _download(mid: str) -> None:
                        list(self._download_manager.download(mid))

                    futures[pool.submit(_download, model_id)] = model_id

                for future in concurrent.futures.as_completed(futures):
                    model_id = futures[future]
                    try:
                        future.result()
                    except Exception as error:
                        logger.error(
                            "Failed to download model '%s': %s", model_id, error
                        )
                        raise

            # Yield completion progress for all models
            for model_id in model_ids:
                yield self._download_manager._progress(
                    model_id, 100, "download complete"
                )

        if tts and warmup_tts and self._tts_config is not None:
            self._ensure_tts()
            assert self._tts is not None
            self._tts.warmup()

    def download_model(self, model_id: str) -> list[DownloadProgress]:
        """Eagerly download one model and return its progress records."""
        return list(self.download_model_iter(model_id))

    def download_model_iter(self, model_id: str) -> Iterator[DownloadProgress]:
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

        try:
            self._stt = resolve_stt_engine(config.engine, self.language, model_dir)
        except Exception:
            # Reset to None so next call retries cleanly
            self._stt = None
            raise
        logger.debug("Loaded STT engine=%s lang=%s", config.engine, self.language)

    def _ensure_tts(self) -> None:
        if self._tts is not None:
            return
        config = self._require_capability(self._tts_config, "tts")
        model_dir = self._download_manager.require_downloaded(config.model_id)
        from voxlocal.tts import resolve_tts_engine

        try:
            self._tts = resolve_tts_engine(config.engine, self.language, model_dir)
        except Exception:
            # Reset to None so next call retries cleanly
            self._tts = None
            raise
        logger.debug("Loaded TTS engine=%s lang=%s", config.engine, self.language)

    def transcribe(
        self, audio_path: str | Path, *, timeout: float | None = None
    ) -> str:
        """Transcribe one audio file.

        Args:
            audio_path: Path to a WAV audio file.
            timeout: Optional timeout in seconds. None means no timeout.

        Returns:
            Transcribed text.

        Raises:
            ModelNotDownloadedError: If the model has not been downloaded.
            TranscriptionError: If transcription fails.
        """
        self._ensure_stt()
        assert self._stt is not None
        try:
            if timeout is not None:
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(self._stt.transcribe, str(audio_path))
                    try:
                        return future.result(timeout=timeout)
                    except concurrent.futures.TimeoutError as exc:
                        raise TranscriptionError(
                            f"Transcription timed out after {timeout}s"
                        ) from exc
            return self._stt.transcribe(str(audio_path))
        except VoxLocalError:
            raise
        except Exception as error:
            raise TranscriptionError(str(error)) from error

    def speak(self, text: str, *, timeout: float | None = None) -> AudioResult:
        """Synthesize one complete higher-quality result.

        Args:
            text: Text to synthesize.
            timeout: Optional timeout in seconds. None means no timeout.

        Returns:
            Complete audio result.

        Raises:
            SynthesisError: If synthesis fails.
        """
        self._ensure_tts()
        assert self._tts is not None
        if self._metrics is not None:
            assert self._tts_config is not None
            self._metrics.on_synthesis_start(
                len(text), self.language, self._tts_config.engine
            )
        try:
            with TimingContext() as timer:
                if timeout is not None:
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        future = pool.submit(self._tts.speak, text)
                        try:
                            result = future.result(timeout=timeout)
                        except concurrent.futures.TimeoutError as exc:
                            raise SynthesisError(
                                f"Synthesis timed out after {timeout}s"
                            ) from exc
                else:
                    result = self._tts.speak(text)
            if self._metrics is not None:
                self._metrics.on_synthesis_complete(
                    timer.duration_seconds, result.duration_seconds
                )
            return result
        except VoxLocalError:
            raise
        except Exception as error:
            raise SynthesisError(str(error)) from error

    def speak_iter(
        self,
        text: str,
        chunk_by: ChunkBy = "progressive",
    ) -> Iterator[AudioResult]:
        """Yield raw engine results as each text chunk is generated.

        Args:
            text: Text to synthesize.
            chunk_by: Chunking strategy — 'progressive', 'sentence', 'line',
                or 'paragraph'.

        Yields:
            AudioResult for each text chunk.

        Raises:
            SynthesisError: If synthesis fails.
        """
        self._ensure_tts()
        assert self._tts is not None
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
        chunk_by: ChunkBy = "progressive",
        prefetch: int = DEFAULT_PREFETCH,
    ) -> Iterator[AudioChunk]:
        """Yield portable, trimmed and crossfaded PCM-ready blocks.

        Args:
            text: Text to synthesize.
            chunk_by: Chunking strategy — 'progressive', 'sentence', 'line',
                or 'paragraph'.
            prefetch: Number of chunks to prefetch ahead.

        Yields:
            AudioChunk with trimmed and crossfaded audio.
        """
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
        """Convenience wrapper around the optional playback adapter.

        Args:
            text: Text to synthesize and play.
            on_event: Optional callback for playback progress events.
            device: Optional audio device identifier.
            prefetch: Number of chunks to prefetch ahead.
        """
        from voxlocal.playback import play

        play(
            self.stream(text, prefetch=prefetch),
            on_event=on_event,
            device=device,
        )


__all__ = [
    "SUPPORTED_LANGUAGES",
    "SUPPORTED_STT_LANGUAGES",
    "SUPPORTED_TTS_LANGUAGES",
    "AudioChunk",
    "AudioResult",
    "ChunkBy",
    "DependencyMissingError",
    "DownloadProgress",
    "EngineNotSupportedError",
    "LanguageNotSupportedError",
    "MetricsCollector",
    "ModelDownloadError",
    "ModelNotDownloadedError",
    "SynthesisError",
    "TimingContext",
    "TranscriptionError",
    "VoxLocal",
    "VoxLocalConfig",
    "VoxLocalError",
    "__version__",
    "create_metrics",
]
