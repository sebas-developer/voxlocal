# VoxLocal

VoxLocal is an alpha Python library for explicit local model setup, speech-to-text,
direct text-to-speech, and low-latency streaming text-to-speech.

The Python stream emits portable mono PCM chunks. Local playback is an optional
adapter, and the optional HTTP server exposes the same chunks as NDJSON for
TypeScript and other runtimes.

## Status

- Python 3.10-3.13
- Core package tests are configured for macOS, Linux, and Windows CI
- CPU inference
- Alpha API: pin the package version
- iOS, Android, and browser-side inference are not implemented

## Install

Install only the engines you use:

```bash
pip install "voxlocal[tts,playback]"
pip install "voxlocal[whisper]"
pip install "voxlocal[moonshine]"
pip install "voxlocal[sensevoice]"
```

Install every Python adapter:

```bash
pip install "voxlocal[all]"
```

## Streaming TTS

```python
from voxlocal import VoxLocal
from voxlocal.playback import play

voice = VoxLocal(language="es")
voice.setup(stt=False, tts=True)

play(voice.stream(
    "Hola, soy Sebastian. El primer bloque es pequeño. "
    "Los siguientes crecen mientras el audio se reproduce."
))
```

`stream()` uses six synthesis steps, starts with a small first text block, then
grows toward complete and grouped sentences. It trims generated edge silence and
crossfades boundaries without replaying samples.

Direct generation uses eight steps:

```python
audio = voice.speak("Hola mundo")
audio.save("output.wav")
```

## Speech To Text

```python
from voxlocal import VoxLocal

speech = VoxLocal(language="auto")
speech.setup(stt=True, tts=False)
print(speech.transcribe("recording.wav"))
```

`language="auto"` is STT-only and uses Whisper. Calling TTS on that instance
raises `LanguageNotSupportedError`.

## Setup Progress

`setup()` is eager and returns all progress records:

```python
progress = voice.setup(stt=False)
for item in progress:
    print(item.percent, item.description)
```

Use `setup_iter()` when progress must be consumed incrementally:

```python
for item in voice.setup_iter(stt=False):
    print(item.percent, item.description)
```

Models are stored under the platform user cache directory. Pass `cache_dir=` to
`VoxLocal` for an explicit location.

## HTTP And TypeScript

Run the optional transport:

```bash
pip install "voxlocal[tts,server]"
# Add Whisper for the language=auto STT endpoint:
pip install "voxlocal[tts,whisper,server]"
voxlocal-server --host 127.0.0.1 --port 8765
```

For a browser frontend on another origin, explicitly allow it:

```bash
voxlocal-server --cors-origin http://localhost:3000
```

The endpoints are:

- `POST /v1/stt?language=auto`: accepts an `audio/wav` body and returns JSON
- `POST /v1/tts`: returns `audio/wav`
- `POST /v1/tts/stream`: returns newline-delimited JSON with base64 PCM16 chunks
- `GET /v1/health`: health check

The TypeScript client is in [`clients/typescript`](clients/typescript):

```ts
import { stream, decodePcm16, transcribe } from "@voxlocal/client";

const text = await transcribe(
  "http://127.0.0.1:8765",
  await recordingFile.arrayBuffer(),
);

for await (const chunk of stream("http://127.0.0.1:8765", {
  language: "es",
  text: "Hola desde TypeScript.",
})) {
  const pcm = decodePcm16(chunk);
  console.log(chunk.sequence, pcm.length, chunk.final);
}
```

See [`docs/transport.md`](docs/transport.md) for the language-neutral contract.

## Engines

| Language | Default STT | TTS |
|---|---|---|
| `es` | Moonshine | Supertonic 3 |
| `en` | SenseVoice | Supertonic 3 |
| `ja` | SenseVoice | Supertonic 3 |
| `ko` | SenseVoice | Supertonic 3 |
| `fr` | Whisper base | Supertonic 3 |
| `de` | Whisper base | Supertonic 3 |
| `pt` | Whisper base | Supertonic 3 |
| `auto` | Whisper base | Not available |

Engine overrides are validated against the requested language and resolve their
own model. For example:

```python
speech = VoxLocal(language="es", stt_engine="whisper")
```

## Licensing

VoxLocal source code is MIT licensed. Models and engine packages have their own
licenses. In particular, Moonshine's non-English speech model is distributed
under the Moonshine Community License and may restrict commercial use. Review
[`NOTICE.md`](NOTICE.md) before distributing an application.

## Development

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest
python -m build
```

Architecture rules are documented in [`PATTERNS.md`](PATTERNS.md).
