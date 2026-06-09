# SPDX-License-Identifier: MIT
from __future__ import annotations

import json

import numpy as np
import pytest

from voxlocal._audio import AudioChunk, AudioResult
from voxlocal.server import (
    MAX_TEXT_LENGTH,
    chunk_to_ndjson,
    create_app,
)

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")


# ─── Helpers ────────────────────────────────────────────────────────────────────


class FakeTTS:
    """Minimal TTS engine for server tests."""

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


class FakeSTT:
    """Minimal STT engine for server tests."""

    def transcribe(self, audio_path: str) -> str:
        return "hello world"

    def warmup(self) -> None:
        pass


class FakeVoxLocal:
    """Fake VoxLocal that returns stub engines."""

    def __init__(self, **kwargs):
        pass

    def setup(self, **kwargs):
        pass

    def speak(self, text: str) -> AudioResult:
        return FakeTTS().speak(text)

    def speak_iter(self, text: str, chunk_by: str = "progressive"):
        yield from FakeTTS().speak_iter(text, chunk_by)

    def stream(self, text: str, **kwargs):
        for i, result in enumerate(FakeTTS().speak_iter(text)):
            yield AudioChunk(
                sequence=i,
                numpy=result.numpy,
                sample_rate=result.sample_rate,
                final=(i == 1),
                source_chunk=i,
            )

    def transcribe(self, audio_path: str) -> str:
        return FakeSTT().transcribe("")


# ─── NDJSON Contract ────────────────────────────────────────────────────────────


def test_chunk_ndjson_contract():
    encoded = chunk_to_ndjson(
        AudioChunk(
            sequence=0,
            numpy=np.ones(10, dtype=np.float32),
            sample_rate=24000,
            final=True,
        )
    )

    record = json.loads(encoded)

    assert encoded.endswith(b"\n")
    assert record["type"] == "audio_chunk"
    assert record["encoding"] == "pcm_s16le"
    assert record["final"] is True


# ─── Route Registration ─────────────────────────────────────────────────────────


def test_server_exposes_versioned_routes():
    app = create_app()
    paths = {route.path for route in app.routes}

    assert {"/v1/health", "/v1/stt", "/v1/tts", "/v1/tts/stream"} <= paths


# ─── Health Endpoint ────────────────────────────────────────────────────────────


def test_health_returns_ok():
    app = create_app(voxlocal_factory=FakeVoxLocal)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )
    import asyncio

    response = asyncio.run(client.get("/v1/health"))
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "api_version" in data


def test_health_verbose_returns_instance_info():
    app = create_app(voxlocal_factory=FakeVoxLocal)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )
    import asyncio

    response = asyncio.run(client.get("/v1/health?verbose=true"))
    assert response.status_code == 200
    data = response.json()
    assert "loaded_instances" in data
    assert "cached_languages" in data


# ─── TTS Endpoint ───────────────────────────────────────────────────────────────


def test_tts_returns_wav():
    app = create_app(voxlocal_factory=FakeVoxLocal)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )
    import asyncio

    response = asyncio.run(
        client.post("/v1/tts", json={"text": "hello", "language": "es"})
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert len(response.content) > 0


def test_tts_rejects_empty_text():
    app = create_app(voxlocal_factory=FakeVoxLocal)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )
    import asyncio

    response = asyncio.run(client.post("/v1/tts", json={"text": "", "language": "es"}))
    assert response.status_code == 422


def test_tts_rejects_long_text():
    app = create_app(voxlocal_factory=FakeVoxLocal)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )
    import asyncio

    long_text = "x" * (MAX_TEXT_LENGTH + 1)
    response = asyncio.run(
        client.post("/v1/tts", json={"text": long_text, "language": "es"})
    )
    assert response.status_code == 422


def test_tts_rejects_invalid_language():
    app = create_app(voxlocal_factory=FakeVoxLocal)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )
    import asyncio

    response = asyncio.run(
        client.post("/v1/tts", json={"text": "hello", "language": "xx"})
    )
    assert response.status_code == 422


def test_tts_includes_request_id():
    app = create_app(voxlocal_factory=FakeVoxLocal)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )
    import asyncio

    response = asyncio.run(
        client.post("/v1/tts", json={"text": "hello", "language": "es"})
    )
    assert "X-Request-ID" in response.headers
    assert "X-API-Version" in response.headers


# ─── Stream Endpoint ────────────────────────────────────────────────────────────


def test_stream_returns_ndjson():
    app = create_app(voxlocal_factory=FakeVoxLocal)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )
    import asyncio

    response = asyncio.run(
        client.post("/v1/tts/stream", json={"text": "hello world", "language": "es"})
    )
    assert response.status_code == 200
    assert "ndjson" in response.headers["content-type"]
    lines = [line for line in response.text.strip().split("\n") if line]
    records = [json.loads(line) for line in lines]
    assert all(r["type"] == "audio_chunk" for r in records)
    assert records[-1]["final"] is True


def test_stream_rejects_empty_text():
    app = create_app(voxlocal_factory=FakeVoxLocal)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )
    import asyncio

    response = asyncio.run(
        client.post("/v1/tts/stream", json={"text": "", "language": "es"})
    )
    assert response.status_code == 422


def test_stream_includes_request_id():
    app = create_app(voxlocal_factory=FakeVoxLocal)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )
    import asyncio

    response = asyncio.run(
        client.post("/v1/tts/stream", json={"text": "hello", "language": "es"})
    )
    assert "X-Request-ID" in response.headers


# ─── STT Endpoint ───────────────────────────────────────────────────────────────


def test_stt_returns_transcription():
    app = create_app(voxlocal_factory=FakeVoxLocal)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )
    import asyncio

    response = asyncio.run(
        client.post(
            "/v1/stt?language=es",
            content=b"RIFF" + b"\x00" * 100,
            headers={"content-type": "audio/wav"},
        )
    )
    assert response.status_code == 200
    assert response.json()["text"] == "hello world"


def test_stt_rejects_empty_body():
    app = create_app(voxlocal_factory=FakeVoxLocal)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )
    import asyncio

    response = asyncio.run(
        client.post(
            "/v1/stt?language=es",
            content=b"",
            headers={"content-type": "audio/wav"},
        )
    )
    assert response.status_code == 422


def test_stt_rejects_oversized_body():
    app = create_app(voxlocal_factory=FakeVoxLocal, max_audio_body_bytes=100)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )
    import asyncio

    response = asyncio.run(
        client.post(
            "/v1/stt?language=es",
            content=b"\x00" * 200,
            headers={"content-type": "audio/wav"},
        )
    )
    assert response.status_code == 413


def test_stt_rejects_invalid_language():
    app = create_app(voxlocal_factory=FakeVoxLocal)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )
    import asyncio

    response = asyncio.run(
        client.post(
            "/v1/stt?language=xx",
            content=b"RIFF" + b"\x00" * 100,
            headers={"content-type": "audio/wav"},
        )
    )
    assert response.status_code == 422


# ─── CORS Configuration ─────────────────────────────────────────────────────────


def test_cors_middleware_added_when_origins_specified():
    app = create_app(cors_origins=("http://localhost:3000",))
    # Check that CORS middleware is in the user_middleware list
    middleware_classes = []
    for item in app.user_middleware:
        # Middleware objects have a .cls attribute
        if hasattr(item, "cls"):
            middleware_classes.append(item.cls.__name__)
        else:
            middleware_classes.append(type(item).__name__)
    assert any("CORS" in name.upper() for name in middleware_classes)


def test_no_cors_middleware_when_empty():
    app = create_app(cors_origins=())
    middleware_classes = []
    for item in app.user_middleware:
        if hasattr(item, "cls"):
            middleware_classes.append(item.cls.__name__)
        else:
            middleware_classes.append(type(item).__name__)
    assert not any("CORS" in name.upper() for name in middleware_classes)


# ─── Models Endpoint ──────────────────────────────────────────────────────────


def test_models_endpoint_returns_languages():
    app = create_app(voxlocal_factory=FakeVoxLocal)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )
    import asyncio

    response = asyncio.run(client.get("/v1/models"))
    assert response.status_code == 200
    data = response.json()
    assert "stt" in data
    assert "tts" in data
    assert "languages" in data["stt"]
    assert "engines" in data["stt"]
    assert "es" in data["stt"]["languages"]
    assert "en" in data["tts"]["languages"]


# ─── TTS Error Propagation ────────────────────────────────────────────────────


class FakeVoxLocalWithError:
    """Fake VoxLocal that raises on synthesis."""

    def __init__(self, **kwargs):
        pass

    def setup(self, **kwargs):
        pass

    def speak(self, text: str) -> AudioResult:
        from voxlocal._errors import SynthesisError

        raise SynthesisError("synthesis failed")

    def speak_iter(self, text: str, chunk_by: str = "progressive"):
        from voxlocal._errors import SynthesisError

        raise SynthesisError("synthesis failed")

    def stream(self, text: str, **kwargs):
        from voxlocal._errors import SynthesisError

        raise SynthesisError("stream failed")

    def transcribe(self, audio_path: str) -> str:
        return "hello world"


def test_tts_error_returns_400():
    app = create_app(voxlocal_factory=FakeVoxLocalWithError)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )
    import asyncio

    response = asyncio.run(
        client.post("/v1/tts", json={"text": "hello", "language": "es"})
    )
    assert response.status_code == 400
    assert "synthesis failed" in response.text


def test_stream_error_returns_ndjson_error():
    app = create_app(voxlocal_factory=FakeVoxLocalWithError)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )
    import asyncio

    response = asyncio.run(
        client.post("/v1/tts/stream", json={"text": "hello", "language": "es"})
    )
    assert response.status_code == 200
    # The error is yielded as an NDJSON record, not an HTTP error
    lines = [line for line in response.text.strip().split("\n") if line]
    records = [json.loads(line) for line in lines]
    error_records = [r for r in records if r.get("type") == "error"]
    assert len(error_records) >= 1
    assert "stream failed" in error_records[0]["error"]


# ─── STT Endpoint Additional Edge Cases ────────────────────────────────────────


def test_stt_rejects_invalid_language_in_query():
    app = create_app(voxlocal_factory=FakeVoxLocal)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )
    import asyncio

    response = asyncio.run(
        client.post(
            "/v1/stt?language=",
            content=b"RIFF" + b"\x00" * 100,
            headers={"content-type": "audio/wav"},
        )
    )
    assert response.status_code == 422
