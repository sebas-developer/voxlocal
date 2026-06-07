# VoxLocal Streaming Transport

## Speech To Text

`POST /v1/stt?language=auto` accepts a WAV file as the raw request body with
`content-type: audio/wav` and returns:

```json
{"text":"transcribed text"}
```

## Streaming Text To Speech

`POST /v1/tts/stream` accepts:

```json
{"language":"es","text":"Hola mundo"}
```

The response content type is `application/x-ndjson`. Each line is one JSON
record:

```json
{
  "type": "audio_chunk",
  "sequence": 0,
  "sample_rate": 44100,
  "channels": 1,
  "encoding": "pcm_s16le",
  "final": false,
  "source_chunk": 0,
  "audio_base64": "..."
}
```

Rules:

1. Decode `audio_base64` to signed 16-bit little-endian PCM.
2. Play or concatenate records in ascending `sequence` order.
3. Configure the output device using `sample_rate` and one channel.
4. The record with `final=true` terminates the audio stream.
5. A record with `type="error"` terminates the stream with an error.

The NDJSON representation prioritizes portability and debuggability. A future
binary transport can preserve the same metadata contract.
