# VoxLocal — Production Readiness Refinement Prompt

> **Goal:** Take the current VoxLocal alpha codebase and refine every module until it is production-grade — secure, resilient, observable, well-tested, and deployment-ready — without changing the public API contract or breaking backward compatibility for existing consumers.

---

## Context

VoxLocal is a Python library (`voxlocal`) that provides explicit local STT/TTS with portable streaming. The current codebase is alpha-quality (v0.1.0) and covers:

- **Facade** (`__init__.py`): `VoxLocal` class orchestrating setup, transcription, synthesis, and streaming
- **Config** (`_config.py`): Model registry and engine resolution
- **Download** (`_download.py`): Model caching and download management
- **Stream** (`_stream.py`): Prefetch, silence trimming, crossfade assembly
- **Audio** (`_audio.py`): `AudioResult` and `AudioChunk` data classes
- **Errors** (`_errors.py`): Exception hierarchy
- **Playback** (`playback.py`): Optional sounddevice adapter
- **Server** (`server.py`): Optional FastAPI HTTP transport
- **STT Engines**: Whisper, Moonshine, SenseVoice adapters
- **TTS Engines**: Supertonic adapter
- **TypeScript Client**: Browser/Node NDJSON streaming client
- **Swift Client** (stub): iOS/macOS client
- **Tests**: Unit tests for config, audio, errors, facade, server, streaming, playback, and integration

---

## Production Readiness Checklist

Work through each section below. For every item, either apply the fix directly or explain why it is intentionally deferred.

---

### 1. Security Hardening

- [ ] **Input validation in `server.py`**: Add strict length limits for `text` (e.g., 10,000 characters max) and validate `language` against `SUPPORTED_LANGUAGES` before creating instances. Reject excessively large WAV bodies in `/v1/stt` with a configurable max (e.g., 50 MB).
- [ ] **Temporary file safety**: The STT endpoint writes WAV to `NamedTemporaryFile` — verify the file is created with restrictive permissions (mode `0o600`) and that race conditions between creation and deletion are impossible. Consider using `tempfile.TemporaryDirectory` as a context manager for atomic cleanup.
- [ ] **CORS defaults**: When `cors_origins` is empty, the server should explicitly reject cross-origin requests rather than relying on FastAPI's default behavior. Log a warning if `cors_origins=("*",)` is set in production.
- [ ] **Dependency pinning in `pyproject.toml`**: Verify all version ranges are appropriately tight. Add upper bounds where missing (e.g., `numpy>=1.26,<3` is good; ensure no dependency allows a future breaking major version silently).
- [ ] **No secrets in model IDs or paths**: Audit `MODEL_REGISTRY` and `DownloadManager` for any hardcoded tokens, API keys, or credentials. Confirm model downloads use HTTPS and verify checksums where available.
- [ ] **Server request body size limits**: Configure FastAPI/uvicorn with `--limit-max-request-size` or equivalent to prevent memory exhaustion from oversized POST bodies.

---

### 2. Error Handling & Resilience

- [ ] **Retry logic in `DownloadManager`**: Add configurable retry with exponential backoff for transient network failures during model downloads. The current code fails immediately on any download error.
- [ ] **Graceful degradation in `server.py`**: The streaming endpoint catches `VoxLocalError` inside `records()` and yields an error JSON — but if the client disconnects mid-stream, the generator's `finally` block should handle cleanup without logging spurious errors.
- [ ] **Timeout guards**: Add optional timeouts for STT transcription (Whisper can run indefinitely on large files) and TTS synthesis. Expose `timeout` as a parameter on `transcribe()` and `speak()`.
- [ ] **Model load failure recovery**: If `_ensure_model()` fails after partial initialization (e.g., ONNX session created but embedding load fails), the engine should reset to `None` so the next call retries cleanly rather than operating in a corrupt state.
- [ ] **Thread-safety audit**: `_synthesis_lock` in `SupertonicTTS` serializes access, but `WhisperSTT` and `SenseVoiceSTT` do not. Verify whether the underlying Whisper/ONNX runtimes are thread-safe and add locks if not.
- [ ] **Explicit resource cleanup**: Add `__del__` or context manager (`__enter__`/`__exit__`) support to `VoxLocal` for releasing model memory when the instance is discarded.

---

### 3. Logging & Observability

- [ ] **Structured logging**: Replace all implicit `print()` or silent behavior with Python `logging` module calls. Use a `voxlocal` logger namespace. Log at appropriate levels:
  - `DEBUG`: Model load, synthesis step counts, chunk sequence numbers
  - `INFO`: Setup completion, download progress milestones
  - `WARNING`: CORS wildcard usage, deprecated API patterns, model cache size approaching limits
  - `ERROR`: Download failures, synthesis failures, engine crashes
- [ ] **Request-level tracing in server**: Add a `X-Request-ID` header (or use FastAPI's middleware) to correlate log lines per HTTP request. Include language, text length, and chunk count in the log.
- [ ] **Metrics hooks**: Add optional callback/hook points for:
  - `on_download_progress(percent, model_id)`
  - `on_synthesis_start(text_length, language, engine)`
  - `on_synthesis_complete(duration_seconds, audio_seconds)`
  - `on_chunk_emitted(sequence, final, latency_ms)`
- [ ] **Health check depth**: The `/v1/health` endpoint returns `{"status": "ok"}` — extend it to report loaded engines, cache size, and memory usage when queried with `?verbose=true`.

---

### 4. Configuration Management

- [ ] **Environment variable support**: Allow all CLI arguments in `server.py` to be set via environment variables (`VOXLOCAL_HOST`, `VOXLOCAL_PORT`, `VOXLOCAL_CACHE_DIR`, `VOXLOCAL_CORS_ORIGIN`). Use `os.environ.get()` with CLI args taking precedence.
- [ ] **Runtime configuration object**: Create a `VoxLocalConfig` dataclass that centralizes:
  - `cache_dir`
  - `default_language`
  - `synthesis_timeout`
  - `max_text_length`
  - `log_level`
  - `prefetch_count`
  
  Accept this as an optional parameter to `VoxLocal.__init__()`.

- [ ] **Validation of `chunk_by` parameter**: The `_split_text()` function accepts `"progressive"`, `"sentence"`, `"line"`, `"paragraph"` — document these in the docstring and add a `Literal` type annotation.
- [ ] **Model version pinning**: The `DownloadManager` uses pinned revisions (e.g., `SENSEVOICE_REVISION`) — add a `MODEL_VERSIONS` dict that maps `model_id` to version metadata and expose it in the health endpoint.

---

### 5. Testing & Quality Assurance

- [ ] **Increase test coverage**: Current tests are good but incomplete. Add tests for:
  - `server.py` endpoints with actual FastAPI `TestClient` (request/response cycle)
  - `prefetch_results` error propagation from producer thread
  - `StreamAssembler` edge cases (empty audio, single-sample audio, alternating silence/audio)
  - `SupertonicTTS.speak_iter()` with all `chunk_by` strategies
  - `SenseVoiceSTT.transcribe()` with different sample rates and languages
  - `MoonshineSTT` error paths (wrong language, missing model files)
  - `VoxLocal.play()` integration with mocked playback
- [ ] **Property-based testing**: Use `hypothesis` for:
  - `trim_boundary_silence` invariants (output ≤ input length, active region preserved)
  - `_split_progressive` reconstruction (concatenated chunks equal original text)
  - `AudioResult.bytes` round-trip (decode WAV, compare samples)
- [ ] **Performance benchmarks**: Add `pytest-benchmark` tests for:
  - Supertonic synthesis latency per step count
  - Whisper transcription latency for 1s, 10s, 60s audio
  - Stream assembly throughput (chunks/second)
- [ ] **Negative test cases**: Ensure every public method has tests for:
  - Invalid input types
  - Boundary values (empty string, max-length string, zero-length audio)
  - Concurrent access patterns
- [ ] **Type checking**: Add `mypy` or `pyright` to the dev dependencies and CI. Fix any type errors. The codebase uses `from __future__ import annotations` throughout — verify all type hints are correct.

---

### 6. Performance Optimization

- [ ] **Lazy model loading audit**: `_ensure_model()` is called on every `transcribe()`/`speak()` call but only loads once — verify there is no hidden overhead from repeated condition checks in hot paths.
- [ ] **NumPy array memory**: `AudioResult` and `AudioChunk` store `numpy` arrays as frozen dataclass fields. Verify these are not accidentally copied. Consider `__slots__` on dataclasses for memory efficiency.
- [ ] **Streaming prefetch tuning**: `DEFAULT_PREFETCH = 2` is hardcoded. Make it configurable via `VoxLocalConfig` and benchmark different values.
- [ ] **Crossfade computation**: `_join_boundary()` creates temporary `linspace` arrays on every call. Pre-compute fade curves for known overlap sizes if the overlap is constant.
- [ ] **Base64 encoding in wire format**: `audio_base64` in `to_wire_dict()` encodes PCM as base64, which adds ~33% overhead. For high-throughput streaming, consider offering a binary WebSocket transport as an alternative.
- [ ] **Concurrent model downloads**: When `setup()` downloads both STT and TTS models, they execute sequentially. Use `concurrent.futures.ThreadPoolExecutor` to download in parallel.

---

### 7. API Design & Documentation

- [ ] **Docstring completeness**: Every public class and method should have a complete docstring with:
  - One-line summary
  - Args section with types and descriptions
  - Returns section with type and description
  - Raises section listing all exceptions
  - Example usage for non-trivial methods
- [ ] **Deprecation path**: The `__getitem__` method on `EngineConfig` is described as "backward-compatible private API" — either document it as public or mark it with `@typing.deprecated`.
- [ ] **API versioning**: The server uses `/v1/` prefix — add `X-API-Version` header to responses and document the versioning policy.
- [ ] **OpenAPI/Swagger**: FastAPI auto-generates OpenAPI docs — add a `description` to the `FastAPI` constructor with the full API contract, and add `response_model` to endpoints for automatic schema generation.
- [ ] **TypeScript client improvements**: Add:
  - AbortController support for cancellable requests
  - Configurable timeout per request
  - Retry logic for transient failures
  - ESM and CJS dual-package output
- [ ] **Error response schema**: Define a consistent error response format across all endpoints:
  ```json
  {
    "error": {
      "code": "MODEL_NOT_DOWNLOADED",
      "message": "...",
      "details": {}
    }
  }
  ```

---

### 8. Packaging & Distribution

- [ ] **Wheel contents audit**: Verify `pyproject.toml`'s `[tool.hatch.build]` includes all necessary files (`.py`, `py.typed` marker, `pyproject.toml`) and excludes test files, docs, demo scripts, and output WAV files.
- [ ] **`py.typed` marker**: Add an empty `voxlocal/py.typed` file so that type checkers recognize the package as typed.
- [ ] **Version automation**: Replace the hardcoded `__version__ = "0.1.0"` with `importlib.metadata.version("voxlocal")` at runtime, keeping `pyproject.toml` as the single source of truth via `[tool.hatch.version]`.
- [ ] **MANIFEST.in or build config**: Ensure source distributions include `README.md`, `LICENSE`, `NOTICE.md`, `CONTRIBUTING.md`, and `PATTERNS.md`.
- [ ] **Python 3.13 compatibility**: The `pyproject.toml` lists 3.10–3.13 — verify CI runs on all four versions and that no deprecated APIs are used.
- [ ] **Platform-specific wheels**: If any dependency has platform-specific wheels (e.g., `onnxruntime`, `sounddevice`), ensure the build matrix produces correct wheels for macOS (arm64+x86_64), Linux (x86_64+arm64), and Windows (x64).

---

### 9. Deployment & Operations

- [ ] **Docker support**: Add a `Dockerfile` for the server that:
  - Uses a slim Python base image
  - Installs only the `[server,tts]` extras
  - Pre-downloads models into the image layer
  - Runs with a non-root user
  - Exposes port 8765
- [ ] **Health check integration**: Document how to use `/v1/health` with container orchestrators (Kubernetes liveness/readiness probes).
- [ ] **Graceful shutdown**: Handle `SIGTERM`/`SIGINT` in `server.py` to drain in-flight requests before exiting.
- [ ] **Resource limits documentation**: Document expected memory usage per engine (e.g., Whisper base ~1 GB, Supertonic ~200 MB, SenseVoice ~300 MB) and recommend minimum system requirements.
- [ ] **Log aggregation**: Document recommended log format for production (JSON structured logging) and how to integrate with ELK/Datadog/Grafana.

---

### 10. Code Quality & Consistency

- [ ] **Ruff rule expansion**: The current ruff config selects `["E", "F", "I", "UP", "B"]`. Add `"S"` (security), `"A"` (builtins), `"C4"` (comprehensions), `"SIM"` (simplify), and `"TCH"` (type-checking imports) for stricter linting.
- [ ] **Pre-commit hooks**: Add `.pre-commit-config.yaml` with `ruff`, `mypy`, and `pytest` hooks.
- [ ] **Dead code removal**: The `demo.py`, `test_sensevoice.py`, `test_tts.py`, and `transcribe.py` files are excluded from the gitignore — either integrate them into the test/docs suite or remove them.
- [ ] **Naming consistency**: The `SENSEVOICE_REVISION` is a full SHA — rename to `SENSEVOICE_PINNED_REVISION` or `SENSEVOICE_COMMIT` for clarity.
- [ ] **Docstring style**: Enforce a consistent docstring format (Google style or NumPy style) across all modules. The codebase currently mixes brief one-liners with detailed multi-line docstrings.
- [ ] **`__all__` completeness**: Verify every module's `__all__` includes all public symbols and excludes internal helpers.

---

### 11. Backward Compatibility & Migration

- [ ] **API stability contract**: Document which APIs are stable, which are experimental, and which may break. The current README says "Alpha API: pin the version" — formalize this.
- [ ] **Deprecation warnings**: If any internal APIs are renamed or removed, emit `DeprecationWarning` for one minor version before removal.
- [ ] **Migration guide**: Add a `MIGRATION.md` that documents any breaking changes from 0.1.x to 0.2.x.
- [ ] **Semantic versioning**: Ensure the version number follows SemVer strictly. Pre-1.0, minor versions may contain breaking changes — document this clearly.

---

### 12. License Compliance

- [ ] **NOTICE.md completeness**: Verify all third-party model licenses are listed. The Moonshine Community License restriction is noted — ensure the legal language is accurate and review annually.
- [ ] **SPDX identifiers**: Add SPDX license identifiers to all source files for automated license scanning.
- [ ] **SBOM generation**: Add a Software Bill of Materials (SBOM) to the release artifacts for supply chain transparency.

---

### 13. Specific Module Fixes

#### `__init__.py` (Facade)
- [ ] Add `__repr__` to `VoxLocal` showing language and loaded capabilities
- [ ] Add type hints for the `play()` convenience method's `on_event` parameter
- [ ] Consider adding `__aenter__`/`__aexit__` for async contexts (future-proofing)

#### `_config.py`
- [ ] Add a `@dataclass(frozen=True)` for `EngineConfig` — currently done correctly, verify immutability is enforced
- [ ] Add `__repr__` to `EngineConfig` for debugging
- [ ] Make `ENGINE_MODELS` a computed property or lazy dict to avoid redundant computation at import time

#### `_download.py`
- [ ] Add `cache_size_bytes()` method to `DownloadManager` for monitoring disk usage
- [ ] Add `cleanup_old_models(keep_last_n)` for cache maintenance
- [ ] Add download progress callback support (percentage-based)
- [ ] Verify `snapshot_download` for SenseVoice uses the correct revision pinning

#### `_stream.py`
- [ ] Add type hints for the `produce()` inner function's return type
- [ ] Document the crossfade algorithm and its parameters in a module-level docstring
- [ ] Add a `StreamAssembler.reset()` method for reuse scenarios
- [ ] Verify `prefetch_results` properly handles `GeneratorExit` when the consumer stops early

#### `server.py`
- [ ] Add request validation with Pydantic models instead of raw `dict[str, str]`
- [ ] Add request/response logging middleware
- [ ] Add rate limiting (per-language or global) to prevent abuse
- [ ] Add CORS preflight handling documentation
- [ ] The `instances` dict grows unboundedly — add LRU eviction or instance pooling with max size
- [ ] Add `/v1/models` endpoint listing available languages and engines

#### `playback.py`
- [ ] Add `device` parameter validation (check if device exists before opening stream)
- [ ] Add volume/gain control parameter
- [ ] Add buffer underrun detection and reporting

#### STT Engines
- [ ] Add `language` validation in each engine's `__init__`
- [ ] Add model size/shape validation after loading
- [ ] Add transcription confidence scores (where supported by the engine)
- [ ] Add batch transcription support for multiple files

#### TTS Engine
- [ ] Add voice style validation against available styles
- [ ] Add SSML-like tag support documentation
- [ ] Add prosody control parameters (speed, pitch, volume) to the public API
- [ ] Verify `_split_sentences` handles edge cases (single character, all punctuation, mixed scripts)

#### TypeScript Client
- [ ] Add comprehensive JSDoc comments
- [ ] Add input validation (empty text, invalid URL)
- [ ] Add `AbortSignal` support for request cancellation
- [ ] Add connection health check before streaming
- [ ] Add TypeScript strict mode verification
- [ ] Export types for custom type narrowing

---

## Implementation Strategy

1. **Phase 1 — Safety & Correctness** (Security, Error Handling, Thread Safety)
2. **Phase 2 — Observability** (Logging, Metrics, Health Checks)
3. **Phase 3 — Quality** (Tests, Type Checking, Linting)
4. **Phase 4 — Performance** (Profiling, Optimization, Caching)
5. **Phase 5 — Distribution** (Packaging, Docker, Documentation)
6. **Phase 6 — Polish** (API Design, Deprecation, Migration Guides)

Each phase should be a separate commit (or PR) to keep changes reviewable.

---

## Success Criteria

After all items are addressed:

- [ ] `ruff check .` passes with zero warnings on the expanded rule set
- [ ] `mypy voxlocal/ --strict` passes with zero errors
- [ ] `pytest --cov=voclocal --cov-report=term-missing` shows ≥ 90% line coverage
- [ ] All public methods have complete docstrings
- [ ] `python -m build` produces a clean sdist and wheel
- [ ] Server starts and passes `/v1/health` check
- [ ] TypeScript client compiles with `--strict` and passes type checks
- [ ] README includes production deployment guidance
- [ ] No `print()` statements remain in library code (only in `demo.py` and scripts)
- [ ] All `TODO`/`FIXME`/`HACK` comments are resolved or tracked as issues
