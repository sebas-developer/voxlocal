# SPDX-License-Identifier: MIT
"""Metrics and callback hooks for VoxLocal observability.

This module provides optional callback mechanisms for monitoring
download progress, synthesis performance, and streaming metrics.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@runtime_checkable
class MetricsCollector(Protocol):
    """Protocol for metrics collection callbacks.

    Implement this protocol to receive callbacks for various
    VoxLocal operations. All methods are optional — only implement
    the callbacks you need.
    """

    def on_download_progress(self, percent: int, model_id: str) -> None:
        """Called when model download progress updates.

        Args:
            percent: Download completion percentage (0-100).
            model_id: Identifier of the model being downloaded.
        """
        ...

    def on_synthesis_start(
        self, text_length: int, language: str, engine: str
    ) -> None:
        """Called when synthesis begins.

        Args:
            text_length: Number of characters in the input text.
            language: Language code being synthesized.
            engine: Engine name being used.
        """
        ...

    def on_synthesis_complete(
        self, duration_seconds: float, audio_seconds: float
    ) -> None:
        """Called when synthesis completes.

        Args:
            duration_seconds: Wall-clock time for synthesis.
            audio_seconds: Duration of generated audio.
        """
        ...

    def on_chunk_emitted(
        self, sequence: int, final: bool, latency_ms: float
    ) -> None:
        """Called when a streaming chunk is emitted.

        Args:
            sequence: Chunk sequence number.
            final: True if this is the last chunk.
            latency_ms: Time since synthesis started for this chunk.
        """
        ...


@dataclass
class _NoOpMetrics:
    """Default no-op metrics collector that discards all events."""

    def on_download_progress(self, percent: int, model_id: str) -> None:
        pass

    def on_synthesis_start(
        self, text_length: int, language: str, engine: str
    ) -> None:
        pass

    def on_synthesis_complete(
        self, duration_seconds: float, audio_seconds: float
    ) -> None:
        pass

    def on_chunk_emitted(
        self, sequence: int, final: bool, latency_ms: float
    ) -> None:
        pass


@dataclass
class _CallbackMetrics:
    """Metrics collector that delegates to user-provided callbacks.

    Attributes:
        download_progress: Callback for download progress events.
        synthesis_start: Callback for synthesis start events.
        synthesis_complete: Callback for synthesis completion events.
        chunk_emitted: Callback for chunk emission events.
    """

    download_progress: Callable[..., None] | None = None
    synthesis_start: Callable[..., None] | None = None
    synthesis_complete: Callable[..., None] | None = None
    chunk_emitted: Callable[..., None] | None = None

    def __post_init__(self) -> None:
        if self.download_progress is not None and not callable(
            self.download_progress
        ):
            raise TypeError("download_progress must be callable or None")
        if self.synthesis_start is not None and not callable(
            self.synthesis_start
        ):
            raise TypeError("synthesis_start must be callable or None")
        if self.synthesis_complete is not None and not callable(
            self.synthesis_complete
        ):
            raise TypeError("synthesis_complete must be callable or None")
        if self.chunk_emitted is not None and not callable(
            self.chunk_emitted
        ):
            raise TypeError("chunk_emitted must be callable or None")

    def on_download_progress(self, percent: int, model_id: str) -> None:
        if self.download_progress is not None:
            self.download_progress(percent, model_id)

    def on_synthesis_start(
        self, text_length: int, language: str, engine: str
    ) -> None:
        if self.synthesis_start is not None:
            self.synthesis_start(text_length, language, engine)

    def on_synthesis_complete(
        self, duration_seconds: float, audio_seconds: float
    ) -> None:
        if self.synthesis_complete is not None:
            self.synthesis_complete(duration_seconds, audio_seconds)

    def on_chunk_emitted(
        self, sequence: int, final: bool, latency_ms: float
    ) -> None:
        if self.chunk_emitted is not None:
            self.chunk_emitted(sequence, final, latency_ms)


# Singleton no-op collector
NO_OP_METRICS = _NoOpMetrics()


def create_metrics(
    *,
    on_download_progress: Callable[..., None] | None = None,
    on_synthesis_start: Callable[..., None] | None = None,
    on_synthesis_complete: Callable[..., None] | None = None,
    on_chunk_emitted: Callable[..., None] | None = None,
) -> MetricsCollector:
    """Create a metrics collector from optional callbacks.

    Args:
        on_download_progress: Callback(percent, model_id) for download progress.
        on_synthesis_start: Callback(text_length, language, engine) for synthesis start.
        on_synthesis_complete: Callback(duration_seconds, audio_seconds) for completion.
        on_chunk_emitted: Callback(sequence, final, latency_ms) for chunk emission.

    Returns:
        A MetricsCollector that delegates to the provided callbacks.
    """
    has_any = any(
        [
            on_download_progress,
            on_synthesis_start,
            on_synthesis_complete,
            on_chunk_emitted,
        ]
    )
    if not has_any:
        return NO_OP_METRICS
    return _CallbackMetrics(
        download_progress=on_download_progress,
        synthesis_start=on_synthesis_start,
        synthesis_complete=on_synthesis_complete,
        chunk_emitted=on_chunk_emitted,
    )


@dataclass
class TimingContext:
    """Context manager for measuring operation duration.

    Usage:
        with TimingContext() as timer:
            # ... perform operation ...
            pass
        print(timer.duration_seconds)
    """

    _start: float = field(default_factory=time.monotonic, init=False, repr=False)
    duration_seconds: float = 0.0

    def __enter__(self) -> TimingContext:
        return self

    def __exit__(self, *args: object) -> None:
        self.duration_seconds = time.monotonic() - self._start


__all__ = [
    "NO_OP_METRICS",
    "MetricsCollector",
    "TimingContext",
    "create_metrics",
]
