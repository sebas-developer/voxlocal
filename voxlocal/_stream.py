from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from queue import Full, Queue
from threading import Event, Thread

import numpy as np

from voxlocal._audio import AudioChunk, AudioResult
from voxlocal._errors import SynthesisError

CROSSFADE_SECONDS = 0.08
BOUNDARY_PADDING_SECONDS = 0.06
SILENCE_THRESHOLD = 0.002
DEFAULT_PREFETCH = 2


def trim_boundary_silence(
    audio: np.ndarray,
    sample_rate: int,
    *,
    trim_start: bool,
    trim_end: bool,
) -> np.ndarray:
    """Remove generated edge silence while retaining a natural short pause."""
    normalized = np.asarray(audio, dtype=np.float32)
    active = np.flatnonzero(np.abs(normalized) > SILENCE_THRESHOLD)
    if len(active) == 0:
        return normalized

    padding = int(BOUNDARY_PADDING_SECONDS * sample_rate)
    start = max(0, int(active[0]) - padding) if trim_start else 0
    end = (
        min(len(normalized), int(active[-1]) + padding + 1)
        if trim_end
        else len(normalized)
    )
    return normalized[start:end]


@dataclass
class StreamAssembler:
    """Convert raw synthesis results into non-repeating gapless blocks."""

    sample_rate: int
    overlap: int
    _tail: np.ndarray | None = None
    _sequence: int = 0

    def push(self, source_index: int, audio: np.ndarray) -> AudioChunk | None:
        """Add one source chunk and return its immediately playable block."""
        current = trim_boundary_silence(
            audio,
            self.sample_rate,
            trim_start=source_index > 0,
            trim_end=True,
        )
        if len(current) == 0:
            return None

        if self._tail is None:
            body, self._tail = self._split_tail(current)
        else:
            body, self._tail = self._join_boundary(self._tail, current)

        if len(body) == 0:
            return None
        chunk = AudioChunk(
            sequence=self._sequence,
            numpy=body,
            sample_rate=self.sample_rate,
            source_chunk=source_index,
        )
        self._sequence += 1
        return chunk

    def finish(self) -> AudioChunk | None:
        """Flush the one retained tail exactly once."""
        if self._tail is None or len(self._tail) == 0:
            return None
        chunk = AudioChunk(
            sequence=self._sequence,
            numpy=self._tail,
            sample_rate=self.sample_rate,
            final=True,
        )
        self._tail = None
        self._sequence += 1
        return chunk

    def _split_tail(self, current: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.overlap <= 0 or len(current) <= self.overlap:
            return np.empty(0, dtype=np.float32), current
        return current[:-self.overlap], current[-self.overlap :]

    def _join_boundary(
        self, previous_tail: np.ndarray, current: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        overlap = min(self.overlap, len(previous_tail), len(current))
        if overlap <= 0:
            combined = np.concatenate([previous_tail, current])
            return self._split_tail(combined)

        fade_out = np.linspace(1.0, 0.0, overlap, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
        boundary = (
            previous_tail[-overlap:] * fade_out + current[:overlap] * fade_in
        )
        prefix = previous_tail[:-overlap]
        remainder = current[overlap:]
        combined = np.concatenate([prefix, boundary, remainder])
        return self._split_tail(combined)


def prefetch_results(
    results: Iterable[AudioResult], *, maxsize: int = DEFAULT_PREFETCH
) -> Iterator[AudioResult]:
    """Yield chunk zero promptly, then prefetch later chunks on one worker."""
    if maxsize < 1:
        raise ValueError("prefetch must be at least 1")

    iterator = iter(results)
    try:
        first = next(iterator)
    except StopIteration:
        return

    queue: Queue[AudioResult | Exception | None] = Queue(maxsize=maxsize)
    cancelled = Event()

    def put(item: AudioResult | Exception | None) -> bool:
        while not cancelled.is_set():
            try:
                queue.put(item, timeout=0.1)
                return True
            except Full:
                continue
        return False

    def produce() -> None:
        try:
            for result in iterator:
                if cancelled.is_set() or not put(result):
                    return
        except Exception as error:
            put(error)
        finally:
            put(None)

    producer = Thread(target=produce, name="voxlocal-stream-producer", daemon=True)
    producer.start()
    try:
        yield first
        while True:
            item = queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise SynthesisError(str(item)) from item
            yield item
    finally:
        cancelled.set()
        producer.join(timeout=1.0)


def assemble_stream(
    results: Iterable[AudioResult], *, prefetch: int = DEFAULT_PREFETCH
) -> Iterator[AudioChunk]:
    """Prefetch raw synthesis and emit trimmed, crossfaded portable chunks."""
    assembler: StreamAssembler | None = None
    for source_index, result in enumerate(
        prefetch_results(results, maxsize=prefetch)
    ):
        if assembler is None:
            assembler = StreamAssembler(
                sample_rate=result.sample_rate,
                overlap=int(CROSSFADE_SECONDS * result.sample_rate),
            )
        elif result.sample_rate != assembler.sample_rate:
            raise SynthesisError(
                f"Sample rate changed from {assembler.sample_rate} "
                f"to {result.sample_rate}"
            )

        chunk = assembler.push(source_index, result.numpy)
        if chunk is not None:
            yield chunk

    if assembler is None:
        raise SynthesisError("TTS produced no audio chunks")
    final = assembler.finish()
    if final is not None:
        yield final
