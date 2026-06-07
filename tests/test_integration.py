from voxlocal import SUPPORTED_LANGUAGES


def test_supported_languages_not_empty():
    assert len(SUPPORTED_LANGUAGES) > 0


def test_all_stt_languages_in_supported():
    from voxlocal._config import MODEL_REGISTRY

    for lang in MODEL_REGISTRY["stt"]:
        assert lang in SUPPORTED_LANGUAGES


def test_all_tts_languages_in_supported():
    from voxlocal._config import MODEL_REGISTRY

    for lang in MODEL_REGISTRY["tts"]:
        assert lang in SUPPORTED_LANGUAGES


def test_download_manager_cache_dir():
    import tempfile
    from pathlib import Path

    from voxlocal._download import DownloadManager

    with tempfile.TemporaryDirectory() as directory:
        manager = DownloadManager(directory)
        assert manager.cache_dir == Path(directory)


def test_audio_result_roundtrip():
    import numpy as np

    from voxlocal._audio import AudioResult

    data = np.random.randn(16000).astype(np.float32)
    result = AudioResult(numpy=data, sample_rate=16000)
    b = result.bytes
    assert len(b) > 0
