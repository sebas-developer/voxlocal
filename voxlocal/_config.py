from __future__ import annotations

from dataclasses import dataclass

from voxlocal._errors import EngineNotSupportedError, LanguageNotSupportedError


@dataclass(frozen=True)
class EngineConfig:
    """Resolved engine and model selection for one capability."""

    engine: str
    model_id: str
    size_mb: int

    def __getitem__(self, key: str) -> str | int:
        """Preserve the original mapping-style private API."""
        return getattr(self, key)


MODEL_REGISTRY: dict[str, dict[str, EngineConfig]] = {
    "stt": {
        "es": EngineConfig("moonshine", "moonshine_es", 60),
        "en": EngineConfig("sensevoice", "sensevoice_onnx", 230),
        "ja": EngineConfig("sensevoice", "sensevoice_onnx", 230),
        "ko": EngineConfig("sensevoice", "sensevoice_onnx", 230),
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

ENGINE_MODELS: dict[str, dict[str, dict[str, EngineConfig]]] = {
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

SUPPORTED_STT_LANGUAGES = tuple(sorted(MODEL_REGISTRY["stt"]))
SUPPORTED_TTS_LANGUAGES = tuple(sorted(MODEL_REGISTRY["tts"]))
SUPPORTED_LANGUAGES = sorted(
    set(MODEL_REGISTRY["stt"].keys()) | set(MODEL_REGISTRY["tts"].keys())
)


def _get_config(
    capability: str, language: str, engine: str | None
) -> EngineConfig:
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
