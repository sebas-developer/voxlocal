# voxlocal/_download.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class DownloadProgress:
    model_id: str
    percent: int
    downloaded: str
    total: str
    description: str


class DownloadManager:
    """Manages model downloads with progress tracking."""

    def __init__(self, cache_dir: str | Path | None = None):
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "voxlocal"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def is_downloaded(self, model_id: str) -> bool:
        """Check if a model is already downloaded."""
        model_dir = self.cache_dir / model_id
        return model_dir.exists() and any(model_dir.iterdir())

    def download(self, model_id: str) -> Iterator[DownloadProgress]:
        """Download a model, yielding progress updates."""
        if self.is_downloaded(model_id):
            yield DownloadProgress(
                model_id=model_id,
                percent=100,
                downloaded="0MB",
                total="0MB",
                description=f"Model {model_id} already cached",
            )
            return

        # Placeholder: actual download logic per engine
        # Each engine wrapper will implement its own download
        yield DownloadProgress(
            model_id=model_id,
            percent=0,
            downloaded="0MB",
            total="0MB",
            description=f"Download not implemented for {model_id}",
        )

    def get_model_path(self, model_id: str) -> Path:
        """Get path to cached model directory."""
        return self.cache_dir / model_id
