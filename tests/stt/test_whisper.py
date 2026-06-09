# SPDX-License-Identifier: MIT
from __future__ import annotations

from voxlocal.stt._whisper import WhisperSTT


def test_whisper_stt_has_transcribe():
    stt = WhisperSTT.__new__(WhisperSTT)
    assert hasattr(stt, "transcribe")


def test_whisper_stt_language():
    stt = WhisperSTT.__new__(WhisperSTT)
    stt.language = "en"
    assert stt.language == "en"
