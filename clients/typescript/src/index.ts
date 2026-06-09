// SPDX-License-Identifier: MIT

/**
 * VoxLocal TypeScript Client
 *
 * Client library for the VoxLocal local STT/TTS HTTP API.
 * Supports streaming NDJSON transport, AbortController cancellation,
 * configurable timeouts, and automatic retry for transient failures.
 *
 * @example
 * ```typescript
 * import { VoxLocalClient } from "voxlocal";
 *
 * const client = new VoxLocalClient("http://localhost:8765");
 *
 * // Simple synthesis
 * const audio = await client.synthesize({ text: "Hello world", language: "en" });
 *
 * // Streaming with cancellation
 * const controller = new AbortController();
 * for await (const chunk of client.stream(
 *   { text: "Hello world", language: "en" },
 *   { signal: controller.signal }
 * )) {
 *   console.log(`Chunk ${chunk.sequence}: ${chunk.audio_base64.length} chars`);
 * }
 * controller.abort();
 * ```
 */

/** Request body for TTS synthesis. */
export interface TTSRequest {
  /** Text to synthesize. */
  text: string;
  /** ISO 639-1 language code (e.g., 'en', 'es', 'ja'). */
  language: string;
}

/** Streaming audio chunk from NDJSON transport. */
export interface AudioChunk {
  /** Record type identifier. */
  type: "audio_chunk";
  /** Monotonically increasing chunk sequence number. */
  sequence: number;
  /** Audio sample rate in Hz. */
  sample_rate: number;
  /** Number of audio channels (always 1 for mono). */
  channels: 1;
  /** Audio encoding format (always 'pcm_s16le'). */
  encoding: "pcm_s16le";
  /** True if this is the last chunk in the stream. */
  final: boolean;
  /** Index of the source chunk that produced this block. */
  source_chunk: number | null;
  /** Base64-encoded PCM audio data. */
  audio_base64: string;
}

/** Response from the STT transcription endpoint. */
export interface TranscriptionResponse {
  /** Transcribed text. */
  text: string;
}

/** Error record from streaming endpoint. */
interface ErrorRecord {
  type: "error";
  error: string;
}

/** Health check response. */
export interface HealthResponse {
  status: "ok" | "error";
  api_version: string;
  loaded_instances?: number;
  cached_languages?: Array<{ language: string; capability: string }>;
  max_instances?: number;
}

/** Available languages and engines. */
export interface ModelsResponse {
  stt: {
    languages: string[];
    engines: Record<
      string,
      { languages: string[]; model_id: string }
    >;
  };
  tts: {
    languages: string[];
    engines: Record<
      string,
      { languages: string[]; model_id: string }
    >;
  };
}

/** Client configuration options. */
export interface VoxLocalClientOptions {
  /** Request timeout in milliseconds. Default: 30000 (30s). */
  timeoutMs?: number;
  /** Number of retry attempts for transient failures. Default: 3. */
  retries?: number;
  /** Base delay in milliseconds for exponential backoff. Default: 1000. */
  retryDelayMs?: number;
}

/**
 * Error thrown when a VoxLocal request fails.
 */
export class VoxLocalError extends Error {
  constructor(
    message: string,
    public readonly statusCode?: number,
    public readonly details?: unknown,
  ) {
    super(message);
    this.name = "VoxLocalError";
  }
}

/**
 * Decode PCM16 base64-encoded audio to Int16Array.
 *
 * @param chunk - Audio chunk with base64-encoded PCM data.
 * @returns Decoded Int16Array of audio samples.
 *
 * @example
 * ```typescript
 * const samples = decodePcm16(chunk);
 * console.log(`${samples.length} samples at ${chunk.sample_rate}Hz`);
 * ```
 */
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

/** Internal helper for retry logic with exponential backoff. */
async function withRetry<T>(
  fn: () => Promise<T>,
  retries: number,
  delayMs: number,
): Promise<T> {
  let lastError: Error | undefined;
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));
      if (attempt < retries) {
        const delay = delayMs * Math.pow(2, attempt);
        await new Promise((resolve) => setTimeout(resolve, delay));
      }
    }
  }
  throw lastError;
}

/**
 * VoxLocal HTTP client for STT/TTS operations.
 *
 * Provides methods for synthesis, transcription, and streaming
 * with built-in timeout, retry, and cancellation support.
 */
export class VoxLocalClient {
  private readonly baseUrl: string;
  private readonly timeoutMs: number;
  private readonly retries: number;
  private readonly retryDelayMs: number;

  /**
   * Create a new VoxLocal client.
   *
   * @param baseUrl - Base URL of the VoxLocal server (e.g., "http://localhost:8765").
   * @param options - Optional configuration for timeouts and retries.
   */
  constructor(baseUrl: string, options?: VoxLocalClientOptions) {
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.timeoutMs = options?.timeoutMs ?? 30_000;
    this.retries = options?.retries ?? 3;
    this.retryDelayMs = options?.retryDelayMs ?? 1_000;
  }

  /**
   * Check server health status.
   *
   * @param verbose - If true, include loaded instances and cache info.
   * @param options - Optional fetch options (e.g., signal for cancellation).
   * @returns Health check response.
   */
  async health(
    verbose = false,
    options?: { signal?: AbortSignal },
  ): Promise<HealthResponse> {
    const url = verbose
      ? `${this.baseUrl}/v1/health?verbose=true`
      : `${this.baseUrl}/v1/health`;
    const response = await fetch(url, {
      signal: options?.signal,
    });
    if (!response.ok) {
      throw new VoxLocalError(
        `Health check failed: ${response.status}`,
        response.status,
      );
    }
    return response.json() as Promise<HealthResponse>;
  }

  /**
   * List available languages and engines.
   *
   * @param options - Optional fetch options.
   * @returns Available models and engines.
   */
  async listModels(options?: { signal?: AbortSignal }): Promise<ModelsResponse> {
    const response = await fetch(`${this.baseUrl}/v1/models`, {
      signal: options?.signal,
    });
    if (!response.ok) {
      throw new VoxLocalError(
        `Failed to list models: ${response.status}`,
        response.status,
      );
    }
    return response.json() as Promise<ModelsResponse>;
  }

  /**
   * Synthesize text to a complete WAV audio response.
   *
   * @param request - TTS request with text and language.
   * @param options - Optional fetch options including signal for cancellation.
   * @returns ArrayBuffer containing WAV audio data.
   * @throws {VoxLocalError} If synthesis fails.
   *
   * @example
   * ```typescript
   * const audio = await client.synthesize({ text: "Hello", language: "en" });
   * const blob = new Blob([audio], { type: "audio/wav" });
   * ```
   */
  async synthesize(
    request: TTSRequest,
    options?: { signal?: AbortSignal },
  ): Promise<ArrayBuffer> {
    return withRetry(
      async () => {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), this.timeoutMs);

        // Combine user signal with timeout
        if (options?.signal) {
          options.signal.addEventListener("abort", () => controller.abort());
        }

        try {
          const response = await fetch(`${this.baseUrl}/v1/tts`, {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify(request),
            signal: controller.signal,
          });
          if (!response.ok) {
            const body = await response.text();
            throw new VoxLocalError(
              `TTS request failed: ${body}`,
              response.status,
            );
          }
          return response.arrayBuffer();
        } finally {
          clearTimeout(timeoutId);
        }
      },
      this.retries,
      this.retryDelayMs,
    );
  }

  /**
   * Transcribe WAV audio to text.
   *
   * @param audio - Audio data as Blob or ArrayBuffer.
   * @param language - ISO 639-1 language code. Default: "auto".
   * @param options - Optional fetch options.
   * @returns Transcribed text.
   * @throws {VoxLocalError} If transcription fails.
   *
   * @example
   * ```typescript
   * const text = await client.transcribe(audioBlob, "en");
   * console.log(text);
   * ```
   */
  async transcribe(
    audio: Blob | ArrayBuffer,
    language = "auto",
    options?: { signal?: AbortSignal },
  ): Promise<string> {
    return withRetry(
      async () => {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), this.timeoutMs);

        if (options?.signal) {
          options.signal.addEventListener("abort", () => controller.abort());
        }

        try {
          const response = await fetch(
            `${this.baseUrl}/v1/stt?language=${encodeURIComponent(language)}`,
            {
              method: "POST",
              headers: { "content-type": "audio/wav" },
              body: audio,
              signal: controller.signal,
            },
          );
          if (!response.ok) {
            const body = await response.text();
            throw new VoxLocalError(
              `STT request failed: ${body}`,
              response.status,
            );
          }
          const result = (await response.json()) as TranscriptionResponse;
          return result.text;
        } finally {
          clearTimeout(timeoutId);
        }
      },
      this.retries,
      this.retryDelayMs,
    );
  }

  /**
   * Stream TTS audio chunks as an async generator.
   *
   * Yields AudioChunk records from the NDJSON streaming endpoint.
   * Supports AbortController for cancellation.
   *
   * @param request - TTS request with text and language.
   * @param options - Optional fetch options including signal for cancellation.
   * @yields AudioChunk records.
   * @throws {VoxLocalError} If the request fails or server returns an error.
   *
   * @example
   * ```typescript
   * const controller = new AbortController();
   * for await (const chunk of client.stream(
   *   { text: "Hello world", language: "en" },
   *   { signal: controller.signal }
   * )) {
   *   const samples = decodePcm16(chunk);
   *   // Process audio samples...
   * }
   * ```
   */
  async *stream(
    request: TTSRequest,
    options?: { signal?: AbortSignal },
  ): AsyncGenerator<AudioChunk> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeoutMs);

    if (options?.signal) {
      options.signal.addEventListener("abort", () => controller.abort());
    }

    try {
      const response = await fetch(`${this.baseUrl}/v1/tts/stream`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(request),
        signal: controller.signal,
      });
      if (!response.ok || response.body === null) {
        const body = await response.text();
        throw new VoxLocalError(
          `Stream request failed: ${body}`,
          response.status,
        );
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
            throw new VoxLocalError(record.error);
          }
          yield record;
        }
        if (done) break;
      }

      if (pending.trim()) {
        const record = JSON.parse(pending) as AudioChunk | ErrorRecord;
        if (record.type === "error") {
          throw new VoxLocalError(record.error);
        }
        yield record;
      }
    } finally {
      clearTimeout(timeoutId);
    }
  }
}
