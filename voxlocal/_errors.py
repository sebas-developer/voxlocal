class VoxLocalError(Exception):
    """Base exception for VoxLocal."""


class LanguageNotSupportedError(VoxLocalError):
    def __init__(self, language: str):
        self.language = language
        super().__init__(f"Language not supported: {language}")


class ModelNotDownloadedError(VoxLocalError):
    def __init__(self, model_id: str, message: str | None = None):
        self.model_id = model_id
        self.message = (
            message
            or f"Model not downloaded. Run v.setup() or v.download_model('{model_id}') first."
        )
        super().__init__(self.message)


class TranscriptionError(VoxLocalError):
    """Error during transcription."""


class SynthesisError(VoxLocalError):
    """Error during synthesis."""
