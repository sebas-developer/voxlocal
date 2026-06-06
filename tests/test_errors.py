# tests/test_errors.py
import pytest
from voxlocal import (
    VoxLocalError,
    LanguageNotSupportedError,
    ModelNotDownloadedError,
    TranscriptionError,
    SynthesisError,
)


def test_language_not_supported():
    with pytest.raises(LanguageNotSupportedError) as exc_info:
        raise LanguageNotSupportedError("xx")
    assert exc_info.value.language == "xx"
    assert "xx" in str(exc_info.value)


def test_model_not_downloaded():
    with pytest.raises(ModelNotDownloadedError) as exc_info:
        raise ModelNotDownloadedError("moonshine_es")
    assert exc_info.value.model_id == "moonshine_es"
    assert "setup()" in exc_info.value.message


def test_errors_are_voxlocal_error():
    assert issubclass(LanguageNotSupportedError, VoxLocalError)
    assert issubclass(ModelNotDownloadedError, VoxLocalError)
    assert issubclass(TranscriptionError, VoxLocalError)
    assert issubclass(SynthesisError, VoxLocalError)
