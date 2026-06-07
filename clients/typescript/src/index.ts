export interface TTSRequest {
  text: string;
  language: string;
}

export interface AudioChunk {
  type: "audio_chunk";
  sequence: number;
  sample_rate: number;
  channels: 1;
  encoding: "pcm_s16le";
  final: boolean;
  source_chunk: number | null;
  audio_base64: string;
}

export interface TranscriptionResponse {
  text: string;
}

interface ErrorRecord {
  type: "error";
  error: string;
}

export function decodePcm16(chunk: AudioChunk): Int16Array {
  const binary = atob(chunk.audio_base64);
  const bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0));
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const samples = new Int16Array(bytes.byteLength / 2);
  for (let index = 0; index < samples.length; index += 1) {
    samples[index] = view.getInt16(index * 2, true);
  }
  return samples;
}

export async function synthesize(
  baseUrl: string,
  request: TTSRequest,
): Promise<ArrayBuffer> {
  const response = await fetch(`${baseUrl}/v1/tts`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!response.ok) {
    throw new Error(`VoxLocal request failed: ${await response.text()}`);
  }
  return response.arrayBuffer();
}

export async function transcribe(
  baseUrl: string,
  audio: Blob | ArrayBuffer,
  language = "auto",
): Promise<string> {
  const response = await fetch(
    `${baseUrl}/v1/stt?language=${encodeURIComponent(language)}`,
    {
      method: "POST",
      headers: { "content-type": "audio/wav" },
      body: audio,
    },
  );
  if (!response.ok) {
    throw new Error(`VoxLocal request failed: ${await response.text()}`);
  }
  const result = (await response.json()) as TranscriptionResponse;
  return result.text;
}

export async function* stream(
  baseUrl: string,
  request: TTSRequest,
): AsyncGenerator<AudioChunk> {
  const response = await fetch(`${baseUrl}/v1/tts/stream`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!response.ok || response.body === null) {
    throw new Error(`VoxLocal request failed: ${await response.text()}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let pending = "";
  while (true) {
    const { done, value } = await reader.read();
    pending += decoder.decode(value, { stream: !done });
    const lines = pending.split("\n");
    pending = lines.pop() ?? "";
    for (const line of lines) {
      if (!line) continue;
      const record = JSON.parse(line) as AudioChunk | ErrorRecord;
      if (record.type === "error") {
        throw new Error(record.error);
      }
      yield record;
    }
    if (done) break;
  }
  if (pending.trim()) {
    const record = JSON.parse(pending) as AudioChunk | ErrorRecord;
    if (record.type === "error") throw new Error(record.error);
    yield record;
  }
}
