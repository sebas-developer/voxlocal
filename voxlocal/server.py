from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from typing import Any

from voxlocal import AudioChunk, VoxLocal
from voxlocal._errors import DependencyMissingError, VoxLocalError


def chunk_to_ndjson(chunk: AudioChunk) -> bytes:
    """Encode one portable chunk as a newline-delimited JSON record."""
    return (
        json.dumps(chunk.to_wire_dict(), separators=(",", ":")) + "\n"
    ).encode("utf-8")


def create_app(
    *,
    voxlocal_factory: Callable[..., VoxLocal] = VoxLocal,
    cache_dir: str | Path | None = None,
    cors_origins: tuple[str, ...] = (),
) -> Any:
    """Create the optional FastAPI transport without importing it at package load."""
    try:
        from fastapi import Body, FastAPI, HTTPException
        from fastapi.responses import Response, StreamingResponse
    except ImportError as error:
        raise DependencyMissingError("fastapi", "server") from error

    app = FastAPI(title="VoxLocal", version="1.0")
    if cors_origins:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(cors_origins),
            allow_methods=["GET", "POST"],
            allow_headers=["content-type"],
        )
    instances: dict[tuple[str, str], VoxLocal] = {}
    instances_lock = Lock()

    def get_instance(language: str, capability: str) -> VoxLocal:
        key = (language, capability)
        with instances_lock:
            instance = instances.get(key)
            if instance is None:
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
            return instance

    @app.get("/v1/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    def parse_request(request: dict[str, str]) -> tuple[str, str]:
        text = request.get("text", "").strip()
        language = request.get("language", "").strip()
        if not text or not language:
            raise HTTPException(
                status_code=422,
                detail="'text' and 'language' must be non-empty strings",
            )
        return text, language

    @app.post("/v1/tts")
    def synthesize(request: dict[str, str]) -> Response:
        text, language = parse_request(request)
        try:
            audio = get_instance(language, "tts").speak(text)
        except VoxLocalError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return Response(content=audio.bytes, media_type="audio/wav")

    @app.post("/v1/tts/stream")
    def stream(request: dict[str, str]) -> StreamingResponse:
        text, language = parse_request(request)
        try:
            instance = get_instance(language, "tts")
        except VoxLocalError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        def records() -> Iterator[bytes]:
            chunks = instance.stream(text)
            try:
                for chunk in chunks:
                    yield chunk_to_ndjson(chunk)
            except VoxLocalError as error:
                yield (
                    json.dumps(
                        {"type": "error", "error": str(error)},
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8")
            finally:
                chunks.close()

        return StreamingResponse(
            records(), media_type="application/x-ndjson"
        )

    @app.post("/v1/stt")
    def transcribe(
        language: str,
        audio: bytes = Body(media_type="audio/wav"),
    ) -> dict[str, str]:
        if not audio:
            raise HTTPException(status_code=422, detail="WAV body must not be empty")
        temporary_path: Path | None = None
        try:
            with NamedTemporaryFile(suffix=".wav", delete=False) as temporary:
                temporary.write(audio)
                temporary_path = Path(temporary.name)
            text = get_instance(language, "stt").transcribe(temporary_path)
        except VoxLocalError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return {"text": text}

    return app


def main() -> None:
    """Run the optional HTTP transport."""
    parser = argparse.ArgumentParser(description="Run the VoxLocal HTTP API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--cache-dir")
    parser.add_argument(
        "--cors-origin",
        action="append",
        default=[],
        help="Allowed browser origin; repeat for multiple origins",
    )
    args = parser.parse_args()
    try:
        import uvicorn
    except ImportError as error:
        raise DependencyMissingError("uvicorn", "server") from error

    uvicorn.run(
        create_app(
            cache_dir=args.cache_dir,
            cors_origins=tuple(args.cors_origin),
        ),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
