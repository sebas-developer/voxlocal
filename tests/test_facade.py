import pytest
from voxlocal import VoxLocal
from voxlocal._errors import LanguageNotSupportedError, ModelNotDownloadedError


def test_voxlocal_init():
    v = VoxLocal(language="es")
    assert v.language == "es"


def test_voxlocal_unsupported_language():
    with pytest.raises(LanguageNotSupportedError):
        VoxLocal(language="xx")


def test_voxlocal_stt_engine_override():
    v = VoxLocal(language="es", stt_engine="whisper")
    assert v._stt_engine_name == "whisper"


def test_voxlocal_transcribe_without_download():
    v = VoxLocal(language="es")
    with pytest.raises(ModelNotDownloadedError):
        v.transcribe("test.wav")
