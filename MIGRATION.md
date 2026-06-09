# Migration Guide

## 0.1.x → 0.2.x

### Breaking Changes

None in this release. The public API remains backward-compatible.

### New Features

- **Context manager support**: `VoxLocal` now supports `with` statements for
  automatic resource cleanup.

- **Timeout parameters**: `transcribe()` and `speak()` accept an optional
  `timeout` parameter to prevent indefinite blocking.

- **Structured logging**: All modules now use Python's `logging` module under
  the `voxlocal` namespace. Configure logging to see operation details:

  ```python
  import logging
  logging.basicConfig(level=logging.INFO)
  ```

- **Verbose health check**: `GET /v1/health?verbose=true` returns loaded
  instance information.

- **Request tracing**: Server responses include `X-Request-ID` and
  `X-API-Version` headers.

- **Server environment variables**: All CLI arguments can be set via
  `VOXLOCAL_HOST`, `VOXLOCAL_PORT`, `VOXLOCAL_CACHE_DIR`,
  `VOXLOCAL_CORS_ORIGIN`, and `VOXLOCAL_LOG_LEVEL`.

- **Download retry**: Model downloads retry transient failures with
  exponential backoff (configurable via `DownloadManager`).

- **Instance eviction**: Server LRU-evicts cached instances when the limit
  is reached.

- **Model cleanup**: `DownloadManager.cleanup_old_models()` for cache
  maintenance.

- **Type annotations**: Complete type hints across all public APIs.

### Deprecations

- `EngineConfig.__getitem__()` — use attribute access instead.

### Internal Changes

- Thread-safe engine loading with double-checked locking.
- Temporary file cleanup uses `TemporaryDirectory` for atomic removal.
- CORS wildcard usage logs a warning in production.
- Expanded ruff linting rules (`S`, `A`, `C4`, `SIM`, `TCH`, `PT`, `RUF`).
- `py.typed` marker for type checker recognition.
- Docker support with non-root user and health checks.
- Server request body size limits via uvicorn `timeout_keep_alive`.
- `ENGINE_MODELS` is now lazily computed at first access instead of import time.
- `SENSEVOICE_PINNED_REVISION` replaces `SENSEVOICE_REVISION` for clarity.
- `PlaybackEvent` callback types are exported in `__all__`.
- SPDX license identifiers added to all source files.
- Pre-commit hooks configuration (ruff, mypy, pytest).
- Device name validation in `play()` before opening audio stream.
- TypeScript client improvements: `AbortSignal` support for all methods,
  configurable timeout, retry with exponential backoff, ESM/CJS dual output.

### Security

- `NamedTemporaryFile` replaced with `TemporaryDirectory` context manager for
  atomic cleanup of uploaded audio files.
- CORS wildcard (`*`) logs a warning when used in production.
- `max_text_length` enforced via Pydantic field validation (10,000 char limit).
- `max_audio_body_bytes` enforced in STT endpoint (50 MB limit).
