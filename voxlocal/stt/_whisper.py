from __future__ import annotations

from pathlib import Path

from voxlocal._errors import DependencyMissingError


class WhisperSTT:
    """Whisper-based STT engine."""

    def __init__(
        self, language: str = "auto", model_dir: str | Path | None = None
    ):
        self.language = language
        self.model_dir = Path(model_dir).expanduser() if model_dir else None
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is None:
            try:
                import whisper
            except ImportError as error:
                raise DependencyMissingError(
                    "openai-whisper", "whisper"
                ) from error

            self._model = whisper.load_model(
                "base",
                download_root=str(self.model_dir) if self.model_dir else None,
                in_memory=False,
            )

    def warmup(self) -> None:
        self._ensure_model()

    def transcribe(self, audio_path: str) -> str:
        self._ensure_model()
        result = self._model.transcribe(
            audio_path,
            language=self.language if self.language != "auto" else None,
        )
        return result["text"].strip()
