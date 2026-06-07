from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from voxlocal import AudioResult, VoxLocal
from voxlocal._download import DownloadProgress
from voxlocal._errors import (
    EngineNotSupportedError,
    LanguageNotSupportedError,
    ModelNotDownloadedError,
)


def test_voxlocal_init_uses_default_engines(tmp_path: Path):
    vox = VoxLocal(language="es", cache_dir=tmp_path)

    assert vox.language == "es"
    assert vox._stt_config.engine == "moonshine"
    assert vox._tts_config.engine == "supertonic"


def test_auto_language_is_valid_for_stt_only(tmp_path: Path):
    vox = VoxLocal(language="auto", cache_dir=tmp_path)

    assert vox._stt_config.engine == "whisper"
    assert vox._tts_config is None
    with pytest.raises(LanguageNotSupportedError, match="tts"):
        vox.speak("hello")


def test_voxlocal_unsupported_language(tmp_path: Path):
    with pytest.raises(LanguageNotSupportedError):
        VoxLocal(language="xx", cache_dir=tmp_path)


def test_engine_override_resolves_its_own_model(tmp_path: Path):
    vox = VoxLocal(
        language="es", stt_engine="whisper", cache_dir=tmp_path
    )

    assert vox._stt_config.engine == "whisper"
    assert vox._stt_config.model_id == "whisper_base"


def test_invalid_engine_override_fails_immediately(tmp_path: Path):
    with pytest.raises(EngineNotSupportedError):
        VoxLocal(
            language="es", stt_engine="sensevoice", cache_dir=tmp_path
        )


def test_transcribe_without_download(tmp_path: Path):
    vox = VoxLocal(language="en", cache_dir=tmp_path)

    with pytest.raises(ModelNotDownloadedError):
        vox.transcribe("test.wav")


def test_setup_is_eager_and_warms_tts(tmp_path: Path):
    class DownloadManagerStub:
        def download(self, model_id: str):
            yield DownloadProgress(
                model_id=model_id,
                percent=100,
                downloaded="complete",
                total="unknown",
                description="done",
            )

    class TTSStub:
        warmed = False

        def warmup(self) -> None:
            self.warmed = True

    vox = VoxLocal(language="es", cache_dir=tmp_path)
    vox._download_manager = DownloadManagerStub()
    vox._tts = TTSStub()

    progress = vox.setup(stt=False)

    assert progress[0].model_id == "supertonic_3"
    assert vox._tts.warmed is True
    assert vox._stt is None


def test_stream_crossfades_without_repeating_tail(tmp_path: Path):
    class TTSStub:
        def speak_iter(self, text: str, chunk_by: str):
            del text, chunk_by
            yield AudioResult(np.ones(30, dtype=np.float32), 100)
            yield AudioResult(np.full(30, 0.5, dtype=np.float32), 100)

    vox = VoxLocal(language="es", cache_dir=tmp_path)
    vox._tts = TTSStub()

    chunks = list(vox.stream("ignored", prefetch=1))
    combined = np.concatenate([chunk.numpy for chunk in chunks])

    assert len(combined) == 52
    assert chunks[-1].final is True
    assert [chunk.sequence for chunk in chunks] == list(range(len(chunks)))
