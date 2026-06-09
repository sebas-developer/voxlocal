# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest

from voxlocal._config import (
    ENGINE_MODELS,
    MODEL_REGISTRY,
    SUPPORTED_LANGUAGES,
    SUPPORTED_STT_LANGUAGES,
    SUPPORTED_TTS_LANGUAGES,
    EngineConfig,
    get_stt_config,
    get_tts_config,
)


def test_registry_has_stt():
    assert "stt" in MODEL_REGISTRY
    assert "es" in MODEL_REGISTRY["stt"]
    assert MODEL_REGISTRY["stt"]["es"].engine == "whisper"


def test_registry_has_tts():
    assert "tts" in MODEL_REGISTRY
    assert "es" in MODEL_REGISTRY["tts"]
    assert MODEL_REGISTRY["tts"]["es"].engine == "supertonic"


def test_get_stt_config():
    config = get_stt_config("es")
    assert config.engine == "whisper"


def test_get_stt_config_auto():
    config = get_stt_config("auto")
    assert config.engine == "whisper"


def test_supported_languages():
    assert "es" in SUPPORTED_LANGUAGES
    assert "en" in SUPPORTED_LANGUAGES
    assert "auto" in SUPPORTED_LANGUAGES


def test_get_stt_config_unsupported():
    from voxlocal._errors import LanguageNotSupportedError

    with pytest.raises(LanguageNotSupportedError):
        get_stt_config("xx")


def test_get_tts_config():
    config = get_tts_config("es")
    assert config.engine == "supertonic"


def test_override_uses_override_model():
    config = get_stt_config("es", engine="moonshine")

    assert config.engine == "moonshine"
    assert config.model_id == "moonshine_es"


def test_auto_has_no_tts_config():
    from voxlocal._errors import LanguageNotSupportedError

    with pytest.raises(LanguageNotSupportedError, match="tts"):
        get_tts_config("auto")


def test_get_tts_config_unsupported():
    from voxlocal._errors import LanguageNotSupportedError

    with pytest.raises(LanguageNotSupportedError):
        get_tts_config("xx")


# ─── EngineConfig ───────────────────────────────────────────────────────────────


def test_engine_config_frozen():
    config = EngineConfig(engine="whisper", model_id="whisper_base", size_mb=139)
    with pytest.raises(AttributeError):
        config.engine = "moonshine"  # type: ignore[misc]


def test_engine_config_getitem():
    config = EngineConfig(engine="whisper", model_id="whisper_base", size_mb=139)
    assert config["engine"] == "whisper"
    assert config["model_id"] == "whisper_base"
    assert config["size_mb"] == 139


def test_engine_config_repr():
    config = EngineConfig(engine="whisper", model_id="whisper_base", size_mb=139)
    r = repr(config)
    assert "EngineConfig" in r
    assert "whisper" in r
    assert "whisper_base" in r
    assert "139" in r


# ─── Engine Override Validation ─────────────────────────────────────────────────


def test_invalid_engine_raises_engine_not_supported():
    from voxlocal._errors import EngineNotSupportedError

    with pytest.raises(EngineNotSupportedError):
        get_stt_config("es", engine="nonexistent")


def test_invalid_engine_for_language():
    from voxlocal._errors import EngineNotSupportedError

    # sensevoice does not support Spanish
    with pytest.raises(EngineNotSupportedError):
        get_stt_config("es", engine="sensevoice")


# ─── Supported Languages Tuples ─────────────────────────────────────────────────


def test_stt_languages_tuple():
    assert isinstance(SUPPORTED_STT_LANGUAGES, tuple)
    assert "es" in SUPPORTED_STT_LANGUAGES
    assert "en" in SUPPORTED_STT_LANGUAGES


def test_tts_languages_tuple():
    assert isinstance(SUPPORTED_TTS_LANGUAGES, tuple)
    assert "es" in SUPPORTED_TTS_LANGUAGES
    assert "en" in SUPPORTED_TTS_LANGUAGES


def test_supported_languages_sorted():
    assert sorted(SUPPORTED_LANGUAGES) == SUPPORTED_LANGUAGES


# ─── ENGINE_MODELS Structure ────────────────────────────────────────────────────


def test_engine_models_stt_whisper():
    assert "whisper" in ENGINE_MODELS["stt"]
    assert "es" in ENGINE_MODELS["stt"]["whisper"]


def test_engine_models_stt_moonshine():
    assert "moonshine" in ENGINE_MODELS["stt"]
    assert "es" in ENGINE_MODELS["stt"]["moonshine"]


def test_engine_models_stt_sensevoice():
    assert "sensevoice" in ENGINE_MODELS["stt"]
    assert "en" in ENGINE_MODELS["stt"]["sensevoice"]


def test_engine_models_tts_supertonic():
    assert "supertonic" in ENGINE_MODELS["tts"]
    assert "es" in ENGINE_MODELS["tts"]["supertonic"]
