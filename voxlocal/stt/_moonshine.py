from __future__ import annotations

from pathlib import Path

from voxlocal._errors import DependencyMissingError

MOONSHINE_ES_MODEL_PATH = Path(
    "download.moonshine.ai/model/base-es/quantized/base-es"
)


class MoonshineSTT:
    """Moonshine-based STT engine."""

    def __init__(
        self, language: str = "es", model_dir: str | Path | None = None
    ):
        self.language = language
        self.model_dir = Path(model_dir).expanduser() if model_dir else None
        self._transcriber = None

    def _ensure_model(self) -> None:
        if self._transcriber is None:
            try:
                from moonshine_voice import ModelArch, Transcriber
            except ImportError as error:
                raise DependencyMissingError(
                    "moonshine-voice", "moonshine"
                ) from error

            if self.model_dir is None:
                raise RuntimeError("Moonshine model_dir is required")
            if self.language != "es":
                raise ValueError(
                    f"Moonshine does not support language '{self.language}'"
                )
            model_path = self.model_dir / MOONSHINE_ES_MODEL_PATH
            self._transcriber = Transcriber(
                model_path=model_path, model_arch=ModelArch.BASE
            )

    def warmup(self) -> None:
        self._ensure_model()

    def transcribe(self, audio_path: str) -> str:
        self._ensure_model()
        from moonshine_voice import load_wav_file

        audio_data, sample_rate = load_wav_file(audio_path)
        transcript = self._transcriber.transcribe_without_streaming(
            audio_data, sample_rate
        )
        return (
            " ".join(line.text for line in transcript.lines) if transcript.lines else ""
        )
