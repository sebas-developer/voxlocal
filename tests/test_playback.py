# SPDX-License-Identifier: MIT
from __future__ import annotations

import sys
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from voxlocal._audio import AudioChunk
from voxlocal.playback import play


def test_playback_submits_chunks_in_order(monkeypatch):
    writes: list[np.ndarray] = []
    events = []

    class OutputStreamStub:
        def __init__(self, **kwargs):
            assert kwargs["samplerate"] == 100

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def write(self, audio):
            writes.append(audio.copy())

    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        SimpleNamespace(OutputStream=OutputStreamStub),
    )
    chunks = [
        AudioChunk(0, np.ones(5, dtype=np.float32), 100),
        AudioChunk(1, np.full(5, 0.5, dtype=np.float32), 100, final=True),
    ]

    play(chunks, on_event=events.append)

    assert len(writes) == 2
    assert [event.sequence for event in events] == [0, 1]
    assert events[-1].final is True


def test_playback_closes_stream_generator_on_device_error(monkeypatch):
    closed = Event()

    class OutputStreamStub:
        def __init__(self, **kwargs):
            del kwargs

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def write(self, audio):
            del audio
            raise RuntimeError("device failed")

    def chunks():
        try:
            yield AudioChunk(0, np.ones(5, dtype=np.float32), 100)
            yield AudioChunk(1, np.ones(5, dtype=np.float32), 100, final=True)
        finally:
            closed.set()

    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        SimpleNamespace(OutputStream=OutputStreamStub),
    )

    with pytest.raises(RuntimeError, match="device failed"):
        play(chunks())

    assert closed.is_set()
