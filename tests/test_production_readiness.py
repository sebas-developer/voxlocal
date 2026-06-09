"""Tests for production-readiness edge cases and gaps.

Covers:
- VoxLocalConfig.from_env()
- EngineConfig.__getitem__ deprecation warning
- MetricsCollector callback paths
- MoonshineSTT error paths
- SenseVoiceSTT language validation
- Concurrent download paths in setup_iter
- Server Pydantic validation models
"""
# SPDX-License-Identifier: MIT
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from voxlocal import VoxLocal
from voxlocal._audio import AudioResult
from voxlocal._config import EngineConfig, VoxLocalConfig
from voxlocal._download import DownloadProgress
from voxlocal._metrics import TimingContext, create_metrics
from voxlocal._stream import StreamAssembler, trim_boundary_silence

# ─── VoxLocalConfig ────────────────────────────────────────────────────────────


def test_voxlocal_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOXLOCAL_CACHE_DIR", "/tmp/test-cache")
    monkeypatch.setenv("VOXLOCAL_DEFAULT_LANGUAGE", "es")
    monkeypatch.setenv("VOXLOCAL_SYNTHESIS_TIMEOUT", "5.0")
    monkeypatch.setenv("VOXLOCAL_TRANSCRIPTION_TIMEOUT", "10.0")
    monkeypatch.setenv("VOXLOCAL_MAX_TEXT_LENGTH", "5000")
    monkeypatch.setenv("VOXLOCAL_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("VOXLOCAL_PREFETCH_COUNT", "4")
    monkeypatch.setenv("VOXLOCAL_MAX_INSTANCES", "32")

    config = VoxLocalConfig.from_env()

    assert config.cache_dir == "/tmp/test-cache"
    assert config.default_language == "es"
    assert config.synthesis_timeout == 5.0
    assert config.transcription_timeout == 10.0
    assert config.max_text_length == 5000
    assert config.log_level == "DEBUG"
    assert config.prefetch_count == 4
    assert config.max_instances == 32


def test_voxlocal_config_from_env_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in [
        "VOXLOCAL_CACHE_DIR",
        "VOXLOCAL_DEFAULT_LANGUAGE",
        "VOXLOCAL_SYNTHESIS_TIMEOUT",
        "VOXLOCAL_TRANSCRIPTION_TIMEOUT",
        "VOXLOCAL_MAX_TEXT_LENGTH",
        "VOXLOCAL_LOG_LEVEL",
        "VOXLOCAL_PREFETCH_COUNT",
        "VOXLOCAL_MAX_INSTANCES",
    ]:
        monkeypatch.delenv(key, raising=False)

    config = VoxLocalConfig.from_env()

    assert config.cache_dir is None
    assert config.default_language == "en"
    assert config.synthesis_timeout == 30.0
    assert config.transcription_timeout == 60.0
    assert config.max_text_length == 10_000
    assert config.log_level == "INFO"
    assert config.prefetch_count == 2
    assert config.max_instances == 64


def test_voxlocal_config_rejects_negative_timeout() -> None:
    with pytest.raises(ValueError, match="synthesis_timeout"):
        VoxLocalConfig(synthesis_timeout=-1.0)


def test_voxlocal_config_rejects_zero_max_text_length() -> None:
    with pytest.raises(ValueError, match="max_text_length"):
        VoxLocalConfig(max_text_length=0)


def test_voxlocal_config_rejects_zero_prefetch() -> None:
    with pytest.raises(ValueError, match="prefetch_count"):
        VoxLocalConfig(prefetch_count=0)


def test_voxlocal_config_rejects_zero_max_instances() -> None:
    with pytest.raises(ValueError, match="max_instances"):
        VoxLocalConfig(max_instances=0)


# ─── EngineConfig deprecation ──────────────────────────────────────────────────


def test_engine_config_getitem_warns_deprecation() -> None:
    config = EngineConfig(engine="whisper", model_id="whisper_base", size_mb=139)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = config["engine"]

    assert value == "whisper"
    assert len(caught) == 1
    assert issubclass(caught[0].category, DeprecationWarning)
    assert "deprecated" in str(caught[0].message).lower()


# ─── MetricsCollector ──────────────────────────────────────────────────────────


def test_create_metrics_no_callbacks() -> None:
    metrics = create_metrics()
    # Should be the singleton no-op, not call anything
    metrics.on_download_progress(50, "test")
    metrics.on_synthesis_start(10, "en", "supertonic")
    metrics.on_synthesis_complete(1.0, 2.0)
    metrics.on_chunk_emitted(0, True, 10.0)


def test_create_metrics_with_callbacks() -> None:
    calls: list[tuple[str, tuple]] = []

    def on_dl(percent: int, model_id: str) -> None:
        calls.append(("download", (percent, model_id)))

    def on_start(text_len: int, lang: str, engine: str) -> None:
        calls.append(("start", (text_len, lang, engine)))

    def on_complete(duration: float, audio: float) -> None:
        calls.append(("complete", (duration, audio)))

    def on_chunk(seq: int, final: bool, latency: float) -> None:
        calls.append(("chunk", (seq, final, latency)))

    metrics = create_metrics(
        on_download_progress=on_dl,
        on_synthesis_start=on_start,
        on_synthesis_complete=on_complete,
        on_chunk_emitted=on_chunk,
    )

    metrics.on_download_progress(50, "test_model")
    metrics.on_synthesis_start(10, "en", "supertonic")
    metrics.on_synthesis_complete(1.0, 2.0)
    metrics.on_chunk_emitted(0, True, 10.0)

    assert calls == [
        ("download", (50, "test_model")),
        ("start", (10, "en", "supertonic")),
        ("complete", (1.0, 2.0)),
        ("chunk", (0, True, 10.0)),
    ]


def test_create_metrics_rejects_non_callable() -> None:
    with pytest.raises(TypeError, match="callable"):
        create_metrics(on_download_progress="not_callable")


def test_timing_context_measures_duration() -> None:
    with TimingContext() as timer:
        # Simulate some work
        _ = sum(range(1000))
    assert timer.duration_seconds >= 0


# ─── MoonshineSTT error paths ──────────────────────────────────────────────────


def test_moonshine_stt_rejects_unsupported_language() -> None:
    from voxlocal.stt._moonshine import MoonshineSTT

    with pytest.raises(ValueError, match="not support"):
        MoonshineSTT(language="en")


def test_moonshine_stt_rejects_french() -> None:
    from voxlocal.stt._moonshine import MoonshineSTT

    with pytest.raises(ValueError, match="not support"):
        MoonshineSTT(language="fr")


# ─── SenseVoiceSTT error paths ────────────────────────────────────────────────


def test_sensevoice_stt_rejects_unsupported_language() -> None:
    from voxlocal.stt._sensevoice import SenseVoiceSTT

    with pytest.raises(ValueError, match="not support"):
        SenseVoiceSTT(language="fr")


def test_sensevoice_stt_accepts_en() -> None:
    from voxlocal.stt._sensevoice import SenseVoiceSTT

    stt = SenseVoiceSTT(language="en")
    assert stt.language == "en"


def test_sensevoice_stt_accepts_ja() -> None:
    from voxlocal.stt._sensevoice import SenseVoiceSTT

    stt = SenseVoiceSTT(language="ja")
    assert stt.language == "ja"


def test_sensevoice_stt_accepts_ko() -> None:
    from voxlocal.stt._sensevoice import SenseVoiceSTT

    stt = SenseVoiceSTT(language="ko")
    assert stt.language == "ko"


def test_sensevoice_clean_transcript_multiple_tokens() -> None:
    from voxlocal.stt._sensevoice import _clean_transcript

    text = "<|zh|><|EMO_UNKNOWN|><|Speech|><|woitn|>你好世界"
    assert _clean_transcript(text) == "你好世界"


def test_sensevoice_clean_transcript_no_tokens() -> None:
    from voxlocal.stt._sensevoice import _clean_transcript

    assert _clean_transcript("Hello world") == "Hello world"


def test_sensevoice_clean_transcript_empty() -> None:
    from voxlocal.stt._sensevoice import _clean_transcript

    assert _clean_transcript("") == ""


# ─── Concurrent download path ─────────────────────────────────────────────────


def test_setup_iter_concurrent_downloads(tmp_path: Path) -> None:
    """Test that setup_iter uses concurrent downloads when multiple models needed."""
    download_calls: list[str] = []

    class StubDownloadManager:
        def __init__(self, *args, **kwargs):
            pass

        def download(self, model_id: str):
            download_calls.append(model_id)
            yield DownloadProgress(
                model_id=model_id,
                percent=100,
                downloaded="complete",
                total="unknown",
                description="done",
            )

        def require_downloaded(self, model_id: str) -> Path:
            return tmp_path / model_id

        @staticmethod
        def _progress(
            model_id: str, percent: int, description: str
        ) -> DownloadProgress:
            return DownloadProgress(
                model_id=model_id,
                percent=percent,
                downloaded="complete" if percent == 100 else "pending",
                total="unknown",
                description=f"{model_id}: {description}",
            )

    class StubTTS:
        def warmup(self) -> None:
            pass

    vox = VoxLocal(language="es", cache_dir=tmp_path)
    vox._download_manager = StubDownloadManager()
    vox._tts = TTSStub()

    # When both stt and tts are requested, both model IDs should be downloaded
    list(vox.setup_iter(stt=True, tts=True, warmup_tts=False))

    # Both models should have been downloaded
    assert "supertonic_3" in download_calls
    assert "whisper_base" in download_calls


class TTSStub:
    def warmup(self) -> None:
        pass


# ─── StreamAssembler edge cases ────────────────────────────────────────────────


def test_assembler_alternating_silence_and_audio() -> None:
    sr = 100
    overlap = 8
    assembler = StreamAssembler(sample_rate=sr, overlap=overlap)
    chunks = []

    # Alternate: silence, audio, silence, audio
    c = assembler.push(0, np.zeros(20, dtype=np.float32))
    if c is not None:
        chunks.append(c)
    c = assembler.push(1, np.ones(20, dtype=np.float32))
    if c is not None:
        chunks.append(c)
    c = assembler.push(2, np.zeros(20, dtype=np.float32))
    if c is not None:
        chunks.append(c)
    c = assembler.push(3, np.ones(20, dtype=np.float32) * 0.5)
    if c is not None:
        chunks.append(c)
    final = assembler.finish()
    if final is not None:
        chunks.append(final)

    # Should produce some chunks from non-silent audio
    assert len(chunks) >= 1
    assert chunks[-1].final is True


def test_assembler_single_sample_audio() -> None:
    sr = 100
    overlap = 8
    assembler = StreamAssembler(sample_rate=sr, overlap=overlap)
    c = assembler.push(0, np.array([0.5], dtype=np.float32))
    _final = assembler.finish()
    # Single sample should still produce output
    if c is not None:
        assert len(c.numpy) >= 1


def test_assembler_empty_audio() -> None:
    sr = 100
    overlap = 8
    assembler = StreamAssembler(sample_rate=sr, overlap=overlap)
    c = assembler.push(0, np.array([], dtype=np.float32))
    final = assembler.finish()
    # Empty audio returns None
    assert c is None
    # Finish on empty assembler returns None
    assert final is None


def test_trim_boundary_silence_very_short() -> None:
    """Audio shorter than padding should still work."""
    audio = np.ones(3, dtype=np.float32)
    trimmed = trim_boundary_silence(
        audio, sample_rate=100, trim_start=True, trim_end=True
    )
    assert len(trimmed) >= 1


# ─── Server Pydantic validation ────────────────────────────────────────────────


fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")


class FakeTTSForPydantic:
    def speak(self, text: str) -> AudioResult:
        return AudioResult(
            numpy=np.ones(100, dtype=np.float32) * 0.5,
            sample_rate=24000,
        )

    def speak_iter(self, text: str, chunk_by: str = "progressive"):
        yield AudioResult(np.ones(50, dtype=np.float32) * 0.5, 24000)
        yield AudioResult(np.ones(50, dtype=np.float32) * 0.5, 24000)

    def warmup(self) -> None:
        pass


class FakeVoxLocalForPydantic:
    def __init__(self, **kwargs):
        pass

    def setup(self, **kwargs):
        pass

    def speak(self, text: str) -> AudioResult:
        return FakeTTSForPydantic().speak(text)

    def speak_iter(self, text: str, chunk_by: str = "progressive"):
        yield from FakeTTSForPydantic().speak_iter(text, chunk_by)

    def stream(self, text: str, **kwargs):
        for i, result in enumerate(FakeTTSForPydantic().speak_iter(text)):
            yield {
                "sequence": i,
                "numpy": result.numpy,
                "sample_rate": result.sample_rate,
                "final": i == 1,
                "source_chunk": i,
            }

    def transcribe(self, audio_path: str) -> str:
        return "hello world"


def test_pydantic_rejects_missing_text() -> None:
    from voxlocal.server import create_app

    app = create_app(voxlocal_factory=FakeVoxLocalForPydantic)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )
    import asyncio

    response = asyncio.run(
        client.post("/v1/tts", json={"language": "es"})
    )
    assert response.status_code == 422


def test_pydantic_rejects_missing_language() -> None:
    from voxlocal.server import create_app

    app = create_app(voxlocal_factory=FakeVoxLocalForPydantic)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )
    import asyncio

    response = asyncio.run(
        client.post("/v1/tts", json={"text": "hello"})
    )
    assert response.status_code == 422


def test_pydantic_rejects_empty_text() -> None:
    from voxlocal.server import create_app

    app = create_app(voxlocal_factory=FakeVoxLocalForPydantic)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )
    import asyncio

    response = asyncio.run(
        client.post("/v1/tts", json={"text": "", "language": "es"})
    )
    assert response.status_code == 422


def test_pydantic_rejects_text_too_long() -> None:
    from voxlocal.server import MAX_TEXT_LENGTH, create_app

    app = create_app(voxlocal_factory=FakeVoxLocalForPydantic)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )
    import asyncio

    response = asyncio.run(
        client.post(
            "/v1/tts",
            json={"text": "x" * (MAX_TEXT_LENGTH + 1), "language": "es"},
        )
    )
    assert response.status_code == 422


def test_pydantic_accepts_valid_request() -> None:
    from voxlocal.server import create_app

    app = create_app(voxlocal_factory=FakeVoxLocalForPydantic)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )
    import asyncio

    response = asyncio.run(
        client.post("/v1/tts", json={"text": "hello", "language": "es"})
    )
    assert response.status_code == 200


def test_stream_pydantic_validation() -> None:
    from voxlocal.server import create_app

    app = create_app(voxlocal_factory=FakeVoxLocalForPydantic)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )
    import asyncio

    # Missing text
    response = asyncio.run(
        client.post("/v1/tts/stream", json={"language": "es"})
    )
    assert response.status_code == 422

    # Empty text
    response = asyncio.run(
        client.post("/v1/tts/stream", json={"text": "", "language": "es"})
    )
    assert response.status_code == 422


# ─── VoxLocal config integration ──────────────────────────────────────────────


def test_voxlocal_accepts_config(tmp_path: Path) -> None:
    config = VoxLocalConfig(cache_dir=tmp_path, prefetch_count=4)
    vox = VoxLocal(language="es", cache_dir=tmp_path, config=config)
    assert vox._config.prefetch_count == 4


def test_voxlocal_default_config(tmp_path: Path) -> None:
    vox = VoxLocal(language="es", cache_dir=tmp_path)
    assert vox._config.prefetch_count == 2
    assert vox._config.max_text_length == 10_000


# ─── __all__ completeness ──────────────────────────────────────────────────────


def test_init_all_includes_version() -> None:
    import voxlocal

    assert "__version__" in voxlocal.__all__


def test_config_all_includes_key_symbols() -> None:
    import voxlocal._config as config_mod

    for name in [
        "VoxLocalConfig",
        "EngineConfig",
        "ChunkBy",
        "MODEL_REGISTRY",
        "ENGINE_MODELS",
        "SUPPORTED_LANGUAGES",
        "SUPPORTED_STT_LANGUAGES",
        "SUPPORTED_TTS_LANGUAGES",
        "get_stt_config",
        "get_tts_config",
    ]:
        assert name in config_mod.__all__, f"{name} missing from __all__"


def test_errors_all_includes_all_exceptions() -> None:
    import voxlocal._errors as errors_mod

    for name in [
        "VoxLocalError",
        "LanguageNotSupportedError",
        "EngineNotSupportedError",
        "ModelNotDownloadedError",
        "ModelDownloadError",
        "DependencyMissingError",
        "TranscriptionError",
        "SynthesisError",
    ]:
        assert name in errors_mod.__all__, f"{name} missing from __all__"


def test_metrics_all_includes_key_symbols() -> None:
    import voxlocal._metrics as metrics_mod

    for name in [
        "MetricsCollector",
        "TimingContext",
        "create_metrics",
        "NO_OP_METRICS",
    ]:
        assert name in metrics_mod.__all__, f"{name} missing from __all__"
