from __future__ import annotations


class WhisperSTT:
    """Whisper-based STT engine."""

    def __init__(self, language: str = "auto"):
        self.language = language
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            import whisper

            self._model = whisper.load_model("base")

    def transcribe(self, audio_path: str) -> str:
        self._ensure_model()
        result = self._model.transcribe(
            audio_path,
            language=self.language if self.language != "auto" else None,
        )
        return result["text"].strip()
