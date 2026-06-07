import pytest

from voxlocal._config import (
    MODEL_REGISTRY,
    SUPPORTED_LANGUAGES,
    get_stt_config,
    get_tts_config,
)


def test_registry_has_stt():
    assert "stt" in MODEL_REGISTRY
    assert "es" in MODEL_REGISTRY["stt"]
    assert MODEL_REGISTRY["stt"]["es"].engine == "moonshine"


def test_registry_has_tts():
    assert "tts" in MODEL_REGISTRY
    assert "es" in MODEL_REGISTRY["tts"]
    assert MODEL_REGISTRY["tts"]["es"].engine == "supertonic"


def test_get_stt_config():
    config = get_stt_config("es")
    assert config.engine == "moonshine"


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
    config = get_stt_config("es", engine="whisper")

    assert config.engine == "whisper"
    assert config.model_id == "whisper_base"


def test_auto_has_no_tts_config():
    from voxlocal._errors import LanguageNotSupportedError

    with pytest.raises(LanguageNotSupportedError, match="tts"):
        get_tts_config("auto")


def test_get_tts_config_unsupported():
    from voxlocal._errors import LanguageNotSupportedError

    with pytest.raises(LanguageNotSupportedError):
        get_tts_config("xx")
