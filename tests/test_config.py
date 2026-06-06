from voxlocal._config import (
    MODEL_REGISTRY,
    get_stt_config,
    get_tts_config,
    SUPPORTED_LANGUAGES,
)


def test_registry_has_stt():
    assert "stt" in MODEL_REGISTRY
    assert "es" in MODEL_REGISTRY["stt"]
    assert MODEL_REGISTRY["stt"]["es"]["engine"] == "moonshine"


def test_registry_has_tts():
    assert "tts" in MODEL_REGISTRY
    assert "es" in MODEL_REGISTRY["tts"]
    assert MODEL_REGISTRY["tts"]["es"]["engine"] == "supertonic"


def test_get_stt_config():
    config = get_stt_config("es")
    assert config["engine"] == "moonshine"


def test_get_stt_config_auto():
    config = get_stt_config("auto")
    assert config["engine"] == "whisper"


def test_supported_languages():
    assert "es" in SUPPORTED_LANGUAGES
    assert "en" in SUPPORTED_LANGUAGES
    assert "auto" in SUPPORTED_LANGUAGES
