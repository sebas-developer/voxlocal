from __future__ import annotations

MODEL_REGISTRY: dict = {
    "stt": {
        "es": {"engine": "moonshine", "model_id": "moonshine_es", "size_mb": 60},
        "en": {"engine": "sensevoice", "model_id": "sensevoice_en", "size_mb": 230},
        "ja": {"engine": "sensevoice", "model_id": "sensevoice_ja", "size_mb": 230},
        "ko": {"engine": "sensevoice", "model_id": "sensevoice_ko", "size_mb": 230},
        "fr": {"engine": "whisper", "model_id": "whisper_base", "size_mb": 139},
        "de": {"engine": "whisper", "model_id": "whisper_base", "size_mb": 139},
        "pt": {"engine": "whisper", "model_id": "whisper_base", "size_mb": 139},
        "auto": {"engine": "whisper", "model_id": "whisper_base", "size_mb": 139},
    },
    "tts": {
        "es": {"engine": "supertonic", "model_id": "supertonic_es", "size_mb": 99},
        "en": {"engine": "supertonic", "model_id": "supertonic_en", "size_mb": 99},
        "ja": {"engine": "supertonic", "model_id": "supertonic_ja", "size_mb": 99},
        "ko": {"engine": "supertonic", "model_id": "supertonic_ko", "size_mb": 99},
        "fr": {"engine": "supertonic", "model_id": "supertonic_fr", "size_mb": 99},
        "de": {"engine": "supertonic", "model_id": "supertonic_de", "size_mb": 99},
        "pt": {"engine": "supertonic", "model_id": "supertonic_pt", "size_mb": 99},
    },
}

SUPPORTED_LANGUAGES = sorted(
    set(MODEL_REGISTRY["stt"].keys()) | set(MODEL_REGISTRY["tts"].keys())
)


def get_stt_config(language: str) -> dict:
    if language not in MODEL_REGISTRY["stt"]:
        from voxlocal._errors import LanguageNotSupportedError

        raise LanguageNotSupportedError(language)
    return MODEL_REGISTRY["stt"][language]


def get_tts_config(language: str) -> dict:
    if language not in MODEL_REGISTRY["tts"]:
        from voxlocal._errors import LanguageNotSupportedError

        raise LanguageNotSupportedError(language)
    return MODEL_REGISTRY["tts"][language]
