# SPDX-License-Identifier: MIT
from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from voxlocal._errors import EngineNotSupportedError, LanguageNotSupportedError

logger = logging.getLogger("voxlocal.config")


@dataclass(frozen=True)
class VoxLocalConfig:
    """Centralized configuration for VoxLocal instances.

    Attributes:
        cache_dir: Directory for model storage. None uses platform default.
        default_language: Default language code when not specified.
        synthesis_timeout: Default timeout in seconds for TTS synthesis.
        transcription_timeout: Default timeout in seconds for STT transcription.
        max_text_length: Maximum allowed text length for synthesis.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
        prefetch_count: Number of chunks to prefetch during streaming.
        max_instances: Maximum cached instances in server mode.
    """

    cache_dir: str | Path | None = None
    default_language: str = "en"
    synthesis_timeout: float | None = 30.0
    transcription_timeout: float | None = 60.0
    max_text_length: int = 10_000
    log_level: str = "INFO"
    prefetch_count: int = 2
    max_instances: int = 64

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.default_language not in SUPPORTED_LANGUAGES:
            logger.warning(
                "default_language '%s' is not in SUPPORTED_LANGUAGES. "
                "This may cause errors at runtime.",
                self.default_language,
            )
        if self.synthesis_timeout is not None and self.synthesis_timeout <= 0:
            raise ValueError("synthesis_timeout must be positive or None")
        if self.transcription_timeout is not None and self.transcription_timeout <= 0:
            raise ValueError("transcription_timeout must be positive or None")
        if self.max_text_length <= 0:
            raise ValueError("max_text_length must be positive")
        if self.prefetch_count < 1:
            raise ValueError("prefetch_count must be at least 1")
        if self.max_instances < 1:
            raise ValueError("max_instances must be at least 1")

    @classmethod
    def from_env(cls) -> VoxLocalConfig:
        """Create configuration from environment variables.

        Environment variables:
            VOXLOCAL_CACHE_DIR: Model cache directory
            VOXLOCAL_DEFAULT_LANGUAGE: Default language code
            VOXLOCAL_SYNTHESIS_TIMEOUT: TTS timeout in seconds
            VOXLOCAL_TRANSCRIPTION_TIMEOUT: STT timeout in seconds
            VOXLOCAL_MAX_TEXT_LENGTH: Maximum text length
            VOXLOCAL_LOG_LEVEL: Logging level
            VOXLOCAL_PREFETCH_COUNT: Streaming prefetch count
            VOXLOCAL_MAX_INSTANCES: Maximum server instances

        Returns:
            Configured VoxLocalConfig instance.
        """
        return cls(
            cache_dir=os.environ.get("VOXLOCAL_CACHE_DIR"),
            default_language=os.environ.get("VOXLOCAL_DEFAULT_LANGUAGE", "en"),
            synthesis_timeout=
                float(os.environ["VOXLOCAL_SYNTHESIS_TIMEOUT"])
                if "VOXLOCAL_SYNTHESIS_TIMEOUT" in os.environ
                else 30.0,
            transcription_timeout=
                float(os.environ["VOXLOCAL_TRANSCRIPTION_TIMEOUT"])
                if "VOXLOCAL_TRANSCRIPTION_TIMEOUT" in os.environ
                else 60.0,
            max_text_length=int(os.environ.get("VOXLOCAL_MAX_TEXT_LENGTH", "10000")),
            log_level=os.environ.get("VOXLOCAL_LOG_LEVEL", "INFO"),
            prefetch_count=int(os.environ.get("VOXLOCAL_PREFETCH_COUNT", "2")),
            max_instances=int(os.environ.get("VOXLOCAL_MAX_INSTANCES", "64")),
        )


@dataclass(frozen=True)
class EngineConfig:
    """Resolved engine and model selection for one capability.

    Attributes:
        engine: Engine name (e.g. 'whisper', 'moonshine', 'supertonic').
        model_id: Model identifier for cache management.
        size_mb: Approximate model size in megabytes.
    """

    engine: str
    model_id: str
    size_mb: int

    def __repr__(self) -> str:
        return (
            f"EngineConfig(engine={self.engine!r}, "
            f"model_id={self.model_id!r}, size_mb={self.size_mb})"
        )

    def __getitem__(self, key: str) -> str | int:
        """Preserve the original mapping-style private API.

        .. deprecated:: 0.2.0
            Use attribute access instead (e.g., ``config.engine``).
        """
        warnings.warn(
            "EngineConfig.__getitem__ is deprecated. "
            "Use attribute access instead (e.g., config.engine).",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(self, key)  # type: ignore[no-any-return]


MODEL_REGISTRY: dict[str, dict[str, EngineConfig]] = {
    "stt": {
        # Default engine for all languages is whisper (base).
        # Other engines (moonshine, sensevoice) available via engine parameter.
        "es": EngineConfig("whisper", "whisper_base", 139),
        "en": EngineConfig("whisper", "whisper_base", 139),
        "ja": EngineConfig("whisper", "whisper_base", 139),
        "ko": EngineConfig("whisper", "whisper_base", 139),
        "fr": EngineConfig("whisper", "whisper_base", 139),
        "de": EngineConfig("whisper", "whisper_base", 139),
        "pt": EngineConfig("whisper", "whisper_base", 139),
        "auto": EngineConfig("whisper", "whisper_base", 139),
    },
    "tts": {
        "es": EngineConfig("supertonic", "supertonic_3", 99),
        "en": EngineConfig("supertonic", "supertonic_3", 99),
        "ja": EngineConfig("supertonic", "supertonic_3", 99),
        "ko": EngineConfig("supertonic", "supertonic_3", 99),
        "fr": EngineConfig("supertonic", "supertonic_3", 99),
        "de": EngineConfig("supertonic", "supertonic_3", 99),
        "pt": EngineConfig("supertonic", "supertonic_3", 99),
    },
}

# Lazy-computed engine-to-model mapping to avoid redundant computation at import time
_ENGINE_MODELS: dict[str, dict[str, dict[str, EngineConfig]]] | None = None


def _get_engine_models() -> dict[str, dict[str, dict[str, EngineConfig]]]:
    """Return lazy-computed engine models mapping."""
    global _ENGINE_MODELS
    if _ENGINE_MODELS is not None:
        return _ENGINE_MODELS
    _ENGINE_MODELS = {
        "stt": {
            "whisper": {
                language: EngineConfig("whisper", "whisper_base", 139)
                for language in MODEL_REGISTRY["stt"]
            },
            "moonshine": {
                "es": EngineConfig("moonshine", "moonshine_es", 60),
            },
            "sensevoice": {
                language: EngineConfig("sensevoice", "sensevoice_onnx", 230)
                for language in ("en", "ja", "ko")
            },
        },
        "tts": {
            "supertonic": {
                language: EngineConfig("supertonic", "supertonic_3", 99)
                for language in MODEL_REGISTRY["tts"]
            }
        },
    }
    return _ENGINE_MODELS


ENGINE_MODELS = _get_engine_models()

SUPPORTED_STT_LANGUAGES = tuple(sorted(MODEL_REGISTRY["stt"]))
SUPPORTED_TTS_LANGUAGES = tuple(sorted(MODEL_REGISTRY["tts"]))
SUPPORTED_LANGUAGES = sorted(
    set(MODEL_REGISTRY["stt"].keys()) | set(MODEL_REGISTRY["tts"].keys())
)


def _get_config(capability: str, language: str, engine: str | None) -> EngineConfig:
    if engine is None:
        try:
            return MODEL_REGISTRY[capability][language]
        except KeyError as error:
            raise LanguageNotSupportedError(language, capability) from error

    try:
        language_models = ENGINE_MODELS[capability][engine]
    except KeyError as error:
        raise EngineNotSupportedError(capability, engine, language) from error
    try:
        return language_models[language]
    except KeyError as error:
        raise EngineNotSupportedError(capability, engine, language) from error


def get_stt_config(language: str, engine: str | None = None) -> EngineConfig:
    """Resolve STT configuration, including explicit engine overrides."""
    return _get_config("stt", language, engine)


def get_tts_config(language: str, engine: str | None = None) -> EngineConfig:
    """Resolve TTS configuration, including explicit engine overrides."""
    return _get_config("tts", language, engine)


# Type alias for TTS chunking strategies
ChunkBy = Literal["progressive", "sentence", "line", "paragraph"]


__all__ = [
    "ENGINE_MODELS",
    "MODEL_REGISTRY",
    "SUPPORTED_LANGUAGES",
    "SUPPORTED_STT_LANGUAGES",
    "SUPPORTED_TTS_LANGUAGES",
    "ChunkBy",
    "EngineConfig",
    "VoxLocalConfig",
    "get_stt_config",
    "get_tts_config",
]
