# SPDX-License-Identifier: MIT
from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import chain
from time import perf_counter
from typing import Literal

from voxlocal._audio import AudioChunk
from voxlocal._errors import DependencyMissingError, SynthesisError

logger = logging.getLogger("voxlocal.playback")


@dataclass(frozen=True)
class PlaybackEvent:
    """Progress emitted immediately before a block is submitted to the device.

    Attributes:
        kind: Event type (currently always 'playing').
        sequence: Chunk sequence number.
        elapsed_seconds: Wall-clock time since playback started.
        audio_seconds: Duration of this chunk in seconds.
        final: True if this is the last chunk.
    """

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
    """Play a portable chunk stream through the optional sounddevice adapter.

    Args:
        chunks: Iterable of AudioChunk blocks to play.
        on_event: Optional callback invoked before each chunk is played.
        device: Optional audio device identifier (int index or string name).

    Raises:
        DependencyMissingError: If sounddevice is not installed.
        SynthesisError: If the stream is empty or sample rates are inconsistent.
    """
    try:
        import sounddevice as sd
    except ImportError as error:
        raise DependencyMissingError("sounddevice", "playback") from error

    # Validate device if specified
    if device is not None:
        try:
            sd.query_devices(device=device)
        except (ValueError, TypeError) as error:
            raise ValueError(
                f"Audio device {device!r} not found. "
                f"Available devices: {[d['name'] for d in sd.query_devices()]}"
            ) from error

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


__all__ = [
    "PlaybackEvent",
    "play",
]
