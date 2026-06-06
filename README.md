# VoxLocal

Local speech-to-text and text-to-speech with zero setup.

## Install

```bash
pip install voxlocal
```

## Quick Start

```python
from voxlocal import VoxLocal

v = VoxLocal(language="es")

# Download models (first time only)
for progress in v.setup():
    print(f"{progress.percent}% - {progress.description}")

# Transcribe audio
text = v.transcribe("audio.wav")

# Text to speech
audio = v.speak("Hola mundo")

# Streaming TTS
for chunk in v.speak_iter("Texto largo", chunk_by="sentence"):
    play(chunk.numpy)
```

## Supported Languages

| Language | STT Engine | TTS Engine |
|----------|------------|------------|
| es | Moonshine | Supertonic |
| en | SenseVoice | Supertonic |
| ja | SenseVoice | Supertonic |
| ko | SenseVoice | Supertonic |
| fr | Whisper | Supertonic |
| de | Whisper | Supertonic |
| pt | Whisper | Supertonic |
| auto | Whisper | Supertonic |

## License

MIT