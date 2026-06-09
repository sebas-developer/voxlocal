# SPDX-License-Identifier: MIT
from __future__ import annotations

from voxlocal.stt._sensevoice import SenseVoiceSTT, _clean_transcript


def test_sensevoice_stt_has_transcribe():
    stt = SenseVoiceSTT.__new__(SenseVoiceSTT)
    assert hasattr(stt, "transcribe")


def test_sensevoice_stt_language():
    stt = SenseVoiceSTT.__new__(SenseVoiceSTT)
    stt.language = "ja"
    assert stt.language == "ja"


def test_clean_transcript_removes_engine_metadata():
    transcript = "<|en|><|EMO_UNKNOWN|><|Speech|><|withitn|>Hello world."

    assert _clean_transcript(transcript) == "Hello world."
