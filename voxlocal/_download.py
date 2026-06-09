# SPDX-License-Identifier: MIT
from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from platformdirs import user_cache_dir

from voxlocal._errors import (
    DependencyMissingError,
    ModelDownloadError,
    ModelNotDownloadedError,
)

logger = logging.getLogger("voxlocal.download")

DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_BASE = 1.5

SENSEVOICE_REPO = "lovemefan/SenseVoice-onnx"
SENSEVOICE_PINNED_REVISION = "8a5ee5b014950890a07246bc590a4f77b3ef67a4"
SENSEVOICE_FILES = (
    "sense-voice-encoder-int8.onnx",
    "embedding.npy",
    "chn_jpn_yue_eng_ko_spectok.bpe.model",
    "am.mvn",
)
SUPERTONIC_FILES = (
    "onnx/tts.json",
    "onnx/unicode_indexer.json",
    "onnx/duration_predictor.onnx",
    "onnx/text_encoder.onnx",
    "onnx/vector_estimator.onnx",
    "onnx/vocoder.onnx",
    "voice_styles/M1.json",
)
MOONSHINE_ES_FILES = (
    "download.moonshine.ai/model/base-es/quantized/base-es/encoder_model.ort",
    "download.moonshine.ai/model/base-es/quantized/base-es/decoder_model_merged.ort",
    "download.moonshine.ai/model/base-es/quantized/base-es/tokenizer.bin",
)


@dataclass(frozen=True)
class DownloadProgress:
    """Observable model setup state."""

    model_id: str
    percent: int
    downloaded: str
    total: str
    description: str


class DownloadManager:
    """Own all model paths, availability checks, and explicit downloads.

    Args:
        cache_dir: Root directory for model storage. Uses platform user cache
            when not provided.
        max_retries: Maximum number of retry attempts for transient failures.
        retry_backoff_base: Base multiplier for exponential backoff.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        *,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_backoff_base: float = DEFAULT_RETRY_BACKOFF_BASE,
    ):
        root = cache_dir or user_cache_dir("voxlocal")
        self.cache_dir = Path(root).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max(0, max_retries)
        self.retry_backoff_base = max(1.0, retry_backoff_base)

    def is_downloaded(self, model_id: str) -> bool:
        """Return true only when required model files exist on disk."""
        if model_id == "supertonic_3":
            return self._all_exist(self.model_dir(model_id), SUPERTONIC_FILES)
        if model_id == "whisper_base":
            return (self.model_dir(model_id) / "base.pt").is_file()
        if model_id == "sensevoice_onnx":
            return self._all_exist(self.model_dir(model_id), SENSEVOICE_FILES)
        if model_id == "moonshine_es":
            return self._all_exist(self.model_dir(model_id), MOONSHINE_ES_FILES)
        raise ValueError(f"Unknown model: {model_id}")

    def require_downloaded(self, model_id: str) -> Path:
        """Return the model directory or raise the public setup error."""
        if not self.is_downloaded(model_id):
            raise ModelNotDownloadedError(model_id)
        return self.model_dir(model_id)

    def download(self, model_id: str) -> Iterator[DownloadProgress]:
        """Download one known model with explicit start and completion records.

        Retries transient failures with exponential backoff up to max_retries.

        Args:
            model_id: Identifier of the model to download.

        Yields:
            DownloadProgress records for each lifecycle event.

        Raises:
            ModelDownloadError: When download fails after all retries.
            DependencyMissingError: When required optional packages are missing.
            ValueError: When model_id is unknown.
        """
        if self.is_downloaded(model_id):
            yield self._progress(model_id, 100, "already cached")
            return

        yield self._progress(model_id, 0, "download started")
        last_error: Exception | None = None
        for attempt in range(1 + self.max_retries):
            try:
                if model_id == "supertonic_3":
                    self._download_supertonic()
                elif model_id == "whisper_base":
                    self._download_whisper()
                elif model_id == "sensevoice_onnx":
                    self._download_sensevoice()
                elif model_id == "moonshine_es":
                    self._download_moonshine()
                else:
                    raise ValueError(f"Unknown model: {model_id}")
                # Non-retryable errors
                break
            except (DependencyMissingError, ValueError):
                raise
            except Exception as error:
                last_error = error
                if attempt < self.max_retries:
                    backoff = self.retry_backoff_base**attempt
                    logger.warning(
                        "Download '%s' attempt %d/%d failed: %s. Retrying in %.1fs...",
                        model_id,
                        attempt + 1,
                        1 + self.max_retries,
                        error,
                        backoff,
                    )
                    time.sleep(backoff)
                else:
                    logger.error(
                        "Download '%s' failed after %d attempts: %s",
                        model_id,
                        1 + self.max_retries,
                        error,
                    )

        if last_error is not None and not self.is_downloaded(model_id):
            raise ModelDownloadError(
                f"Failed to download model '{model_id}' after "
                f"{1 + self.max_retries} attempts: {last_error}"
            ) from last_error

        if not self.is_downloaded(model_id):
            raise ModelDownloadError(
                f"Download for '{model_id}' completed without all required files"
            )
        yield self._progress(model_id, 100, "download complete")

    def model_dir(self, model_id: str) -> Path:
        """Return the engine-specific directory for a model."""
        directories = {
            "supertonic_3": self.cache_dir / "supertonic-3",
            "whisper_base": self.cache_dir / "whisper",
            "sensevoice_onnx": self.cache_dir / "sensevoice-onnx",
            "moonshine_es": self.cache_dir / "moonshine",
        }
        try:
            return directories[model_id]
        except KeyError as error:
            raise ValueError(f"Unknown model: {model_id}") from error

    def get_model_path(self, model_id: str) -> Path:
        """Backward-compatible alias for model_dir()."""
        return self.model_dir(model_id)

    @staticmethod
    def _all_exist(root: Path, relative_paths: tuple[str, ...]) -> bool:
        return all((root / relative_path).is_file() for relative_path in relative_paths)

    def cache_size_bytes(self) -> int:
        """Return total size of the model cache directory in bytes."""
        total = 0
        if self.cache_dir.is_dir():
            for path in self.cache_dir.rglob("*"):
                if path.is_file():
                    total += path.stat().st_size
        return total

    def cleanup_old_models(self, keep_last_n: int = 2) -> list[str]:
        """Remove model directories, keeping the most recently accessed.

        Args:
            keep_last_n: Number of most recently accessed models to retain.

        Returns:
            List of model IDs that were removed.
        """
        model_ids = list(self._model_directories().keys())
        if len(model_ids) <= keep_last_n:
            return []

        # Sort by access time (oldest first)
        model_dirs = [(mid, self.model_dir(mid)) for mid in model_ids]
        model_dirs.sort(key=lambda x: x[1].stat().st_atime if x[1].is_dir() else 0)

        removed: list[str] = []
        for mid, mdir in model_dirs[:-keep_last_n]:
            if mdir.is_dir():
                import shutil

                shutil.rmtree(mdir)
                removed.append(mid)
                logger.info("Cleaned up old model: %s", mid)
        return removed

    def _model_directories(self) -> dict[str, Path]:
        """Return mapping of model IDs to their directories."""
        directories = {
            "supertonic_3": self.cache_dir / "supertonic-3",
            "whisper_base": self.cache_dir / "whisper",
            "sensevoice_onnx": self.cache_dir / "sensevoice-onnx",
            "moonshine_es": self.cache_dir / "moonshine",
        }
        return directories

    @staticmethod
    def _progress(model_id: str, percent: int, description: str) -> DownloadProgress:
        return DownloadProgress(
            model_id=model_id,
            percent=percent,
            downloaded="complete" if percent == 100 else "pending",
            total="unknown",
            description=f"{model_id}: {description}",
        )

    def _download_supertonic(self) -> None:
        try:
            from supertonic.loader import download_model
        except ImportError as error:
            raise DependencyMissingError("supertonic", "tts") from error

        model_dir = self.model_dir("supertonic_3")
        model_dir.parent.mkdir(parents=True, exist_ok=True)
        download_model(model_dir, model_name="supertonic-3")

    def _download_whisper(self) -> None:
        try:
            import whisper
        except ImportError as error:
            raise DependencyMissingError("openai-whisper", "whisper") from error

        model = whisper.load_model(
            "base", download_root=str(self.model_dir("whisper_base"))
        )
        del model

    def _download_sensevoice(self) -> None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as error:
            raise DependencyMissingError("huggingface-hub", "sensevoice") from error

        snapshot_download(
            repo_id=SENSEVOICE_REPO,
            revision=SENSEVOICE_PINNED_REVISION,
            local_dir=str(self.model_dir("sensevoice_onnx")),
            allow_patterns=list(SENSEVOICE_FILES),
        )

    def _download_moonshine(self) -> None:
        try:
            from moonshine_voice import get_model_for_language
        except ImportError as error:
            raise DependencyMissingError("moonshine-voice", "moonshine") from error

        get_model_for_language("es", cache_root=self.model_dir("moonshine_es"))


__all__ = [
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_RETRY_BACKOFF_BASE",
    "DownloadManager",
    "DownloadProgress",
]
