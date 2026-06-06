from __future__ import annotations

import numpy as np
from pathlib import Path


LANG_MAP = {"en": 4, "ja": 11, "ko": 12, "zh": 3, "yue": 7}


class SenseVoiceSTT:
    """SenseVoice ONNX-based STT engine."""

    def __init__(self, language: str = "en"):
        self.language = language
        self._session = None
        self._embedding = None
        self._sp = None
        self._model_dir = None

    def _ensure_model(self):
        if self._session is not None:
            return
        from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel
        from huggingface_hub import snapshot_download
        import sentencepiece as spm

        model_dir = Path.home() / ".cache" / "voxlocal" / "sensevoice_onnx"
        if not model_dir.exists():
            snapshot_download("lovemefan/SenseVoice-onnx", local_dir=str(model_dir))

        sess_opt = SessionOptions()
        sess_opt.intra_op_num_threads = 4
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
        from scipy.signal import resample as scipy_resample
        from voxlocal.stt._sensevoice_frontend import extract_features

        data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
        audio = data[:, 0]
        if sr != 16000:
            audio = scipy_resample(audio, int(len(audio) * 16000 / sr))

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
        return self._sp.DecodeIds(tokens.tolist())
