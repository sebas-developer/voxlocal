from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import chain
from time import perf_counter
from typing import Literal

from voxlocal._audio import AudioChunk
from voxlocal._errors import DependencyMissingError, SynthesisError


@dataclass(frozen=True)
class PlaybackEvent:
    """Progress emitted immediately before a block is submitted to the device."""

    kind: Literal["playing"]
    sequence: int
    elapsed_seconds: float
    audio_seconds: float
    final: bool


def play(
    chunks: Iterable[AudioChunk],
    *,
    on_event: Callable[[PlaybackEvent], None] | None = None,
    device: int | str | None = None,
) -> None:
    """Play a portable chunk stream through the optional sounddevice adapter."""
    try:
        import sounddevice as sd
    except ImportError as error:
        raise DependencyMissingError("sounddevice", "playback") from error

    iterator = iter(chunks)
    try:
        first = next(iterator)
    except StopIteration as error:
        raise SynthesisError("Audio stream produced no chunks") from error

    started_at = perf_counter()
    try:
        with sd.OutputStream(
            samplerate=first.sample_rate,
            channels=1,
            dtype="float32",
            device=device,
        ) as stream:
            for chunk in chain((first,), iterator):
                if chunk.sample_rate != first.sample_rate:
                    raise SynthesisError(
                        f"Sample rate changed from {first.sample_rate} "
                        f"to {chunk.sample_rate}"
                    )
                if on_event is not None:
                    on_event(
                        PlaybackEvent(
                            kind="playing",
                            sequence=chunk.sequence,
                            elapsed_seconds=perf_counter() - started_at,
                            audio_seconds=chunk.duration_seconds,
                            final=chunk.final,
                        )
                    )
                stream.write(chunk.numpy)
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()
