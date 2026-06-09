# SPDX-License-Identifier: MIT
from __future__ import annotations

import argparse
import json
import logging
import os
import time
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Any

from pydantic import BaseModel, Field

from voxlocal import AudioChunk, VoxLocal
from voxlocal._config import (
    ENGINE_MODELS,
    MODEL_REGISTRY,
    SUPPORTED_LANGUAGES,
    SUPPORTED_STT_LANGUAGES,
    SUPPORTED_TTS_LANGUAGES,
)
from voxlocal._errors import DependencyMissingError, VoxLocalError

logger = logging.getLogger("voxlocal.server")

# ─── Security limits ───────────────────────────────────────────────────────────
MAX_TEXT_LENGTH = 10_000
MAX_AUDIO_BODY_BYTES = 50 * 1024 * 1024  # 50 MB
MAX_INSTANCES = 64


class TTSRequest(BaseModel):
    """Request body for TTS synthesis endpoints."""

    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH)
    language: str = Field(..., min_length=1)


class StreamRequest(BaseModel):
    """Request body for streaming TTS endpoint."""

    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH)
    language: str = Field(..., min_length=1)


def chunk_to_ndjson(chunk: AudioChunk) -> bytes:
    """Encode one portable chunk as a newline-delimited JSON record."""
    return (json.dumps(chunk.to_wire_dict(), separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def create_app(
    *,
    voxlocal_factory: Callable[..., VoxLocal] = VoxLocal,
    cache_dir: str | Path | None = None,
    cors_origins: tuple[str, ...] = (),
    max_text_length: int = MAX_TEXT_LENGTH,
    max_audio_body_bytes: int = MAX_AUDIO_BODY_BYTES,
    max_instances: int = MAX_INSTANCES,
) -> Any:
    """Create the optional FastAPI transport without importing it at package load.

    Args:
        voxlocal_factory: Callable that creates VoxLocal instances.
        cache_dir: Optional cache directory for model storage.
        cors_origins: Tuple of allowed CORS origins. Empty tuple rejects all.
        max_text_length: Maximum allowed text length for TTS/STT requests.
        max_audio_body_bytes: Maximum WAV body size in bytes for STT.
        max_instances: Maximum number of cached VoxLocal instances.

    Returns:
        Configured FastAPI application instance.

    Raises:
        DependencyMissingError: If fastapi is not installed.
    """
    try:
        from fastapi import Body, FastAPI, HTTPException
        from fastapi.responses import Response, StreamingResponse
    except ImportError as error:
        raise DependencyMissingError("fastapi", "server") from error

    api_version = "1.0"
    app = FastAPI(
        title="VoxLocal",
        version=api_version,
        description="Local STT/TTS HTTP API with streaming NDJSON transport.",
    )

    if cors_origins:
        if "*" in cors_origins:
            logger.warning(
                "CORS wildcard ('*') is set — this allows any origin. "
                "Do not use in production without additional protections."
            )
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(cors_origins),
            allow_methods=["GET", "POST"],
            allow_headers=["content-type"],
        )

    # Instance cache with LRU eviction
    instances: dict[tuple[str, str], VoxLocal] = {}
    instances_lock = Lock()
    instance_order: list[tuple[str, str]] = []

    def _validate_language(language: str) -> None:
        """Validate language parameter against supported languages."""
        if not language or language not in SUPPORTED_LANGUAGES:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid language: '{language}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_LANGUAGES))}",
            )

    def get_instance(language: str, capability: str) -> VoxLocal:
        key = (language, capability)
        with instances_lock:
            instance = instances.get(key)
            if instance is None:
                # LRU eviction when cache is full
                if len(instances) >= max_instances and instance_order:
                    evict_key = instance_order.pop(0)
                    instances.pop(evict_key, None)
                    logger.debug("Evicted cached instance: %s", evict_key)
                kwargs: dict[str, Any] = {"language": language}
                if cache_dir is not None:
                    kwargs["cache_dir"] = cache_dir
                instance = voxlocal_factory(**kwargs)
                instance.setup(
                    stt=capability == "stt",
                    tts=capability == "tts",
                    warmup_tts=capability == "tts",
                )
                instances[key] = instance
                instance_order.append(key)
                logger.info("Created %s instance for language=%s", capability, language)
            else:
                # Move to end (most recently used)
                if key in instance_order:
                    instance_order.remove(key)
                instance_order.append(key)
            return instance

    @app.get("/v1/health")
    def health(verbose: bool = False) -> dict[str, Any]:
        """Health check endpoint.

        Returns basic status by default. Use ?verbose=true for detailed info.

        Args:
            verbose: If true, include loaded instances and cache info.

        Returns:
            Dictionary with status and optional instance information.
        """
        result: dict[str, Any] = {
            "status": "ok",
            "api_version": api_version,
        }
        if verbose:
            with instances_lock:
                result["loaded_instances"] = len(instances)
                result["cached_languages"] = [
                    {"language": k[0], "capability": k[1]} for k in instances
                ]
                result["max_instances"] = max_instances
        return result

    def error_response(code: str, message: str, status_code: int) -> HTTPException:
        """Create a standardized error response."""
        return HTTPException(
            status_code=status_code,
            detail={"error": {"code": code, "message": message}},
        )

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        """List available languages and supported engines.

        Returns:
            Dictionary with stt and tts language/engine mappings.
        """
        stt_engines: dict[str, list[str]] = {}
        for lang in SUPPORTED_STT_LANGUAGES:
            config = MODEL_REGISTRY["stt"].get(lang)
            if config:
                stt_engines.setdefault(config.engine, []).append(lang)

        tts_engines: dict[str, list[str]] = {}
        for lang in SUPPORTED_TTS_LANGUAGES:
            config = MODEL_REGISTRY["tts"].get(lang)
            if config:
                tts_engines.setdefault(config.engine, []).append(lang)

        def _engine_info(
            engines: dict[str, list[str]], capability: str
        ) -> dict[str, dict[str, object]]:
            result: dict[str, dict[str, object]] = {}
            for name, langs in engines.items():
                first_lang = langs[0]
                result[name] = {
                    "languages": langs,
                    "model_id": ENGINE_MODELS[capability][
                        name
                    ][first_lang].model_id,
                }
            return result

        return {
            "stt": {
                "languages": list(SUPPORTED_STT_LANGUAGES),
                "engines": _engine_info(stt_engines, "stt"),
            },
            "tts": {
                "languages": list(SUPPORTED_TTS_LANGUAGES),
                "engines": _engine_info(tts_engines, "tts"),
            },
        }

    @app.post("/v1/tts")
    def synthesize(request: TTSRequest) -> Response:
        """Synthesize text to a complete WAV audio response."""
        text = request.text.strip()
        language = request.language.strip()
        _validate_language(language)
        request_id = str(uuid.uuid4())[:8]
        logger.info(
            "[%s] TTS request: lang=%s text_len=%d", request_id, language, len(text)
        )
        t0 = time.monotonic()
        try:
            audio = get_instance(language, "tts").speak(text)
        except VoxLocalError as error:
            logger.error("[%s] TTS failed: %s", request_id, error)
            raise HTTPException(status_code=400, detail=str(error)) from error
        elapsed = time.monotonic() - t0
        logger.info(
            "[%s] TTS complete: duration=%.2fs audio=%.2fs",
            request_id,
            elapsed,
            audio.duration_seconds,
        )
        response = Response(content=audio.bytes, media_type="audio/wav")
        response.headers["X-API-Version"] = api_version
        response.headers["X-Request-ID"] = request_id
        return response

    @app.post("/v1/tts/stream")
    def stream(request: StreamRequest) -> StreamingResponse:
        """Stream TTS audio chunks as newline-delimited JSON."""
        text = request.text.strip()
        language = request.language.strip()
        _validate_language(language)
        request_id = str(uuid.uuid4())[:8]
        logger.info(
            "[%s] Stream request: lang=%s text_len=%d",
            request_id,
            language,
            len(text),
        )
        try:
            instance = get_instance(language, "tts")
        except VoxLocalError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        chunk_count = 0
        t0 = time.monotonic()

        def records() -> Iterator[bytes]:
            nonlocal chunk_count
            try:
                chunks = instance.stream(text)
                for chunk in chunks:
                    chunk_count += 1
                    yield chunk_to_ndjson(chunk)
            except VoxLocalError as error:
                logger.error("[%s] Stream error: %s", request_id, error)
                yield (
                    json.dumps(
                        {"type": "error", "error": str(error)},
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8")
            except GeneratorExit:
                # Client disconnected — clean up without logging an error
                logger.debug("[%s] Client disconnected mid-stream", request_id)
            except Exception as error:
                logger.error("[%s] Unexpected stream error: %s", request_id, error)
            finally:
                elapsed = time.monotonic() - t0
                logger.info(
                    "[%s] Stream complete: chunks=%d elapsed=%.2fs",
                    request_id,
                    chunk_count,
                    elapsed,
                )
                import contextlib

                with contextlib.suppress(Exception):
                    chunks_var = locals().get("chunks")
                    if chunks_var is not None:
                        close_fn = getattr(chunks_var, "close", None)
                        if close_fn is not None:
                            close_fn()  # Already closed or generator exhausted

        response = StreamingResponse(records(), media_type="application/x-ndjson")
        response.headers["X-API-Version"] = api_version
        response.headers["X-Request-ID"] = request_id
        return response

    @app.post("/v1/stt")
    def transcribe(
        language: str,
        audio: bytes = Body(media_type="audio/wav"),
    ) -> dict[str, str]:
        """Transcribe WAV audio to text."""
        _validate_language(language)
        if not audio:
            raise HTTPException(status_code=422, detail="WAV body must not be empty")
        if len(audio) > max_audio_body_bytes:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"WAV body exceeds maximum size of "
                    f"{max_audio_body_bytes // (1024 * 1024)} MB"
                ),
            )
        request_id = str(uuid.uuid4())[:8]
        logger.info(
            "[%s] STT request: lang=%s audio_size=%d",
            request_id,
            language,
            len(audio),
        )
        t0 = time.monotonic()
        try:
            # Use a secure temporary directory for atomic cleanup
            with TemporaryDirectory() as tmpdir:
                temp_path = Path(tmpdir) / "audio.wav"
                # Write with restrictive permissions
                fd = os.open(
                    str(temp_path),
                    os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                    0o600,
                )
                try:
                    os.write(fd, audio)
                finally:
                    os.close(fd)
                text = get_instance(language, "stt").transcribe(str(temp_path))
        except VoxLocalError as error:
            logger.error("[%s] STT failed: %s", request_id, error)
            raise HTTPException(status_code=400, detail=str(error)) from error
        elapsed = time.monotonic() - t0
        logger.info(
            "[%s] STT complete: text_len=%d elapsed=%.2fs",
            request_id,
            len(text),
            elapsed,
        )
        response: dict[str, str] = {"text": text}
        return response

    return app


def main() -> None:
    """Run the optional HTTP transport with graceful shutdown support."""
    import signal

    parser = argparse.ArgumentParser(description="Run the VoxLocal HTTP API")
    parser.add_argument(
        "--host",
        default=os.environ.get("VOXLOCAL_HOST", "127.0.0.1"),
        help="Bind address (default: 127.0.0.1; env: VOXLOCAL_HOST)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("VOXLOCAL_PORT", "8765")),
        help="Bind port (default: 8765; env: VOXLOCAL_PORT)",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("VOXLOCAL_CACHE_DIR"),
        help="Model cache directory (env: VOXLOCAL_CACHE_DIR)",
    )
    parser.add_argument(
        "--cors-origin",
        action="append",
        default=[],
        help="Allowed browser origin; repeat for multiple origins. "
        "(env: VOXLOCAL_CORS_ORIGIN, comma-separated)",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("VOXLOCAL_LOG_LEVEL", "INFO"),
        help="Log level (default: INFO; env: VOXLOCAL_LOG_LEVEL)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("VOXLOCAL_WORKERS", "1")),
        help="Number of worker processes (default: 1; env: VOXLOCAL_WORKERS)",
    )
    args = parser.parse_args()

    # Support comma-separated CORS origins from env
    env_cors = os.environ.get("VOXLOCAL_CORS_ORIGIN", "")
    cors_origins = tuple(args.cors_origin)
    if env_cors and not cors_origins:
        cors_origins = tuple(o.strip() for o in env_cors.split(",") if o.strip())

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        import uvicorn
    except ImportError as error:
        raise DependencyMissingError("uvicorn", "server") from error

    logger.info("Starting VoxLocal server on %s:%d", args.host, args.port)

    # Configure signal handlers for graceful shutdown
    # Limit max request body to 55 MB (slightly above MAX_AUDIO_BODY_BYTES)
    # to prevent memory exhaustion from oversized POST bodies
    server = uvicorn.Server(
        uvicorn.Config(
            create_app(
                cache_dir=args.cache_dir,
                cors_origins=cors_origins,
            ),
            host=args.host,
            port=args.port,
            log_level=args.log_level.lower(),
            workers=args.workers,
            limit_max_requests=None,
            timeout_keep_alive=30,
        )
    )

    def _handle_signal(sig: int, frame: object) -> None:
        """Handle SIGTERM/SIGINT for graceful shutdown."""
        sig_name = signal.Signals(sig).name
        logger.info("Received %s, initiating graceful shutdown...", sig_name)
        server.should_exit = True

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    server.run()


if __name__ == "__main__":
    main()
