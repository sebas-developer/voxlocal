# SPDX-License-Identifier: MIT
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from voxlocal.stt._moonshine import MoonshineSTT


def test_moonshine_stt_has_transcribe():
    stt = MoonshineSTT.__new__(MoonshineSTT)
    assert hasattr(stt, "transcribe")


def test_moonshine_stt_language():
    stt = MoonshineSTT.__new__(MoonshineSTT)
    stt.language = "es"
    assert stt.language == "es"


def test_warmup_loads_verified_model_without_downloader(monkeypatch, tmp_path):
    calls = []
    fake_module = SimpleNamespace(
        ModelArch=SimpleNamespace(BASE="base"),
        Transcriber=lambda **kwargs: calls.append(kwargs) or object(),
    )
    monkeypatch.setitem(sys.modules, "moonshine_voice", fake_module)

    stt = MoonshineSTT(language="es", model_dir=tmp_path)
    stt.warmup()

    assert calls == [
        {
            "model_path": tmp_path
            / Path("download.moonshine.ai/model/base-es/quantized/base-es"),
            "model_arch": "base",
        }
    ]
