class VoxLocalError(Exception):
    """Base exception for VoxLocal."""


class LanguageNotSupportedError(VoxLocalError):
    def __init__(self, language: str, capability: str | None = None):
        self.language = language
        self.capability = capability
        detail = f" for {capability}" if capability else ""
        super().__init__(f"Language not supported{detail}: {language}")


class EngineNotSupportedError(VoxLocalError):
    """Raised when an engine cannot serve the requested language."""

    def __init__(self, capability: str, engine: str, language: str):
        self.capability = capability
        self.engine = engine
        self.language = language
        super().__init__(
            f"{capability.upper()} engine '{engine}' does not support language "
            f"'{language}'"
        )


class ModelNotDownloadedError(VoxLocalError):
    def __init__(self, model_id: str, message: str | None = None):
        self.model_id = model_id
        self.message = (
            message
            or "Model not downloaded. Run v.setup() or "
            f"v.download_model('{model_id}') first."
        )
        super().__init__(self.message)


class ModelDownloadError(VoxLocalError):
    """Raised when model setup fails."""


class DependencyMissingError(VoxLocalError):
    """Raised when an optional feature dependency is unavailable."""

    def __init__(self, dependency: str, extra: str):
        self.dependency = dependency
        self.extra = extra
        super().__init__(
            f"Missing optional dependency '{dependency}'. "
            f"Install it with: pip install 'voxlocal[{extra}]'"
        )


class TranscriptionError(VoxLocalError):
    """Error during transcription."""


class SynthesisError(VoxLocalError):
    """Error during synthesis."""
