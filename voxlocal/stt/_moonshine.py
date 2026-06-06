from __future__ import annotations


class MoonshineSTT:
    """Moonshine-based STT engine."""

    def __init__(self, language: str = "es"):
        self.language = language
        self._transcriber = None

    def _ensure_model(self):
        if self._transcriber is None:
            from moonshine_voice import Transcriber, get_model_for_language

            model_path, model_arch = get_model_for_language(self.language)
            self._transcriber = Transcriber(
                model_path=model_path, model_arch=model_arch
            )

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
