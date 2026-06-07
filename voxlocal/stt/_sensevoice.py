from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from voxlocal._errors import DependencyMissingError

LANG_MAP = {"en": 4, "ja": 11, "ko": 12, "zh": 3, "yue": 7}
CONTROL_TOKEN_PATTERN = re.compile(r"<\|[^|>]+\|>")


def _clean_transcript(text: str) -> str:
    """Remove SenseVoice metadata tokens from public transcript text."""
    return CONTROL_TOKEN_PATTERN.sub("", text).strip()


class SenseVoiceSTT:
    """SenseVoice ONNX-based STT engine."""

    def __init__(
        self, language: str = "en", model_dir: str | Path | None = None
    ):
        self.language = language
        self.model_dir = Path(model_dir).expanduser() if model_dir else None
        self._session = None
        self._embedding = None
        self._sp = None
        self._model_dir = None

    def warmup(self) -> None:
        self._ensure_model()

    def _ensure_model(self) -> None:
        if self._session is not None:
            return
        try:
            import sentencepiece as spm
            from onnxruntime import (
                GraphOptimizationLevel,
                InferenceSession,
                SessionOptions,
            )
        except ImportError as error:
            raise DependencyMissingError(
                error.name or "sensevoice dependencies", "sensevoice"
            ) from error

        if self.model_dir is None:
            raise RuntimeError("SenseVoice model_dir is required")
        model_dir = self.model_dir

        sess_opt = SessionOptions()
        sess_opt.log_severity_level = 4
        sess_opt.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = InferenceSession(
            str(model_dir / "sense-voice-encoder-int8.onnx"),
            sess_options=sess_opt,
            providers=[("CPUExecutionProvider", {})],
        )
        self._embedding = np.load(str(model_dir / "embedding.npy"))
        self._sp = spm.SentencePieceProcessor()
        self._sp.load(str(model_dir / "chn_jpn_yue_eng_ko_spectok.bpe.model"))
        self._model_dir = model_dir

    def transcribe(self, audio_path: str) -> str:
        self._ensure_model()
        import soundfile as sf
        try:
            from scipy.signal import resample_poly
        except ImportError as error:
            raise DependencyMissingError("scipy", "sensevoice") from error

        from voxlocal.stt._sensevoice_frontend import extract_features

        data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
        audio = data.mean(axis=1, dtype=np.float32)
        if sr != 16000:
            audio = resample_poly(audio, 16000, sr).astype(np.float32)

        feats = extract_features(audio, str(self._model_dir / "am.mvn"))
        feats = feats[None, ...]

        lid = LANG_MAP.get(self.language, 4)
        le = self._embedding[[[lid]]]
        tn = self._embedding[[[14]]]
        ev = self._embedding[[[1, 2]]]
        inp = np.concatenate([le, ev, tn, feats], axis=1).astype(np.float32)
        il = np.array([inp.shape[1]], dtype=np.int64)

        out = self._session.run(None, {"speech": inp, "speech_lengths": il})[0]
        tokens = out[0].argmax(axis=-1)
        mask = np.append([True], tokens[1:] != tokens[:-1])
        tokens = tokens[mask]
        tokens = tokens[tokens != 0]
        return _clean_transcript(self._sp.DecodeIds(tokens.tolist()))
