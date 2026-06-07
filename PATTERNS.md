# VoxLocal Engineering Patterns

## Boundaries

- `VoxLocal` resolves capabilities, model state, and engine calls.
- `_download.py` exclusively owns cache paths and downloads.
- TTS engines yield raw `AudioResult` objects.
- `_stream.py` exclusively owns prefetch, silence trimming, and crossfade.
- `playback.py` exclusively owns local audio-device output.
- `server.py` translates portable `AudioChunk` records to HTTP.

Do not move device, HTTP, or model-download behavior into an engine wrapper.

## Failure Semantics

- Missing models raise `ModelNotDownloadedError`.
- Missing optional packages raise `DependencyMissingError` with the required
  extra.
- Unsupported language/engine combinations fail during construction.
- Unknown chunk strategies fail explicitly; they never silently become a
  one-chunk request.

## Streaming Contract

- The producer queue is bounded.
- Every sample is emitted once.
- The final retained crossfade tail is marked with `final=True`.
- Wire audio is mono signed PCM16 little-endian with an explicit sample rate.
- Consumers must process chunks in sequence order.

## Testing

- Tests use temporary caches and do not inspect the developer's home cache.
- Network and audio hardware are excluded from unit tests.
- Public contracts require tests for unsupported inputs and failure behavior.
