# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from voxlocal._download import (
    SENSEVOICE_FILES,
    SUPERTONIC_FILES,
    DownloadManager,
    DownloadProgress,
)

if TYPE_CHECKING:
    from pathlib import Path


def _touch_all(root: Path, relative_paths: tuple[str, ...]) -> None:
    for relative_path in relative_paths:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def test_download_progress_fields():
    progress = DownloadProgress(
        model_id="test_model",
        percent=50,
        downloaded="12MB",
        total="24MB",
        description="Downloading...",
    )

    assert progress.model_id == "test_model"
    assert progress.percent == 50


def test_model_checks_are_scoped_to_configured_cache(tmp_path: Path):
    manager = DownloadManager(tmp_path)
    _touch_all(manager.model_dir("supertonic_3"), SUPERTONIC_FILES)
    _touch_all(manager.model_dir("sensevoice_onnx"), SENSEVOICE_FILES)

    assert manager.is_downloaded("supertonic_3")
    assert manager.is_downloaded("sensevoice_onnx")
    assert not manager.is_downloaded("whisper_base")


def test_cached_download_reports_completion_without_network(tmp_path: Path):
    manager = DownloadManager(tmp_path)
    model = manager.model_dir("whisper_base") / "base.pt"
    model.parent.mkdir(parents=True)
    model.touch()

    progress = list(manager.download("whisper_base"))

    assert len(progress) == 1
    assert progress[0].percent == 100
    assert "already cached" in progress[0].description


def test_unknown_model_is_an_explicit_error(tmp_path: Path):
    manager = DownloadManager(tmp_path)

    with pytest.raises(ValueError, match="Unknown model"):
        manager.is_downloaded("unknown")


def test_require_downloaded_raises_when_missing(tmp_path: Path):
    manager = DownloadManager(tmp_path)

    with pytest.raises(Exception, match="not downloaded"):
        manager.require_downloaded("whisper_base")


def test_require_downloaded_returns_path(tmp_path: Path):
    manager = DownloadManager(tmp_path)
    model = manager.model_dir("whisper_base") / "base.pt"
    model.parent.mkdir(parents=True)
    model.touch()

    path = manager.require_downloaded("whisper_base")
    assert path == manager.model_dir("whisper_base")


def test_cache_size_bytes(tmp_path: Path):
    manager = DownloadManager(tmp_path)
    _touch_all(manager.model_dir("supertonic_3"), SUPERTONIC_FILES)

    size = manager.cache_size_bytes()
    assert size == 0  # Empty files


def test_cache_size_bytes_with_content(tmp_path: Path):
    manager = DownloadManager(tmp_path)
    model_dir = manager.model_dir("supertonic_3")
    model_dir.mkdir(parents=True)
    (model_dir / "test.txt").write_bytes(b"hello world")

    size = manager.cache_size_bytes()
    assert size == 11


def test_cleanup_old_models(tmp_path: Path):
    manager = DownloadManager(tmp_path)
    # Create two model directories
    d1 = manager.model_dir("supertonic_3")
    d2 = manager.model_dir("whisper_base")
    d1.mkdir(parents=True)
    d2.mkdir(parents=True)
    (d1 / "a.txt").write_bytes(b"a")
    (d2 / "b.txt").write_bytes(b"b")

    # Make d1 older
    import os
    import time

    old_time = time.time() - 1000
    os.utime(str(d1 / "a.txt"), (old_time, old_time))

    removed = manager.cleanup_old_models(keep_last_n=1)
    assert len(removed) == 1
    assert "supertonic_3" in removed


def test_cleanup_keeps_all_when_under_limit(tmp_path: Path):
    manager = DownloadManager(tmp_path)
    d1 = manager.model_dir("supertonic_3")
    d1.mkdir(parents=True)
    (d1 / "a.txt").write_bytes(b"a")

    removed = manager.cleanup_old_models(keep_last_n=5)
    assert removed == []


def test_download_retry_configurable(tmp_path: Path):
    manager = DownloadManager(tmp_path, max_retries=5, retry_backoff_base=2.0)
    assert manager.max_retries == 5
    assert manager.retry_backoff_base == 2.0


def test_download_retry_clamps_to_minimum(tmp_path: Path):
    manager = DownloadManager(tmp_path, max_retries=-1, retry_backoff_base=0.5)
    assert manager.max_retries == 0
    assert manager.retry_backoff_base == 1.0


def test_model_dir_all_models(tmp_path: Path):
    manager = DownloadManager(tmp_path)
    for model_id in [
        "supertonic_3",
        "whisper_base",
        "sensevoice_onnx",
        "moonshine_es",
    ]:
        d = manager.model_dir(model_id)
        assert d.is_absolute()


def test_model_dir_unknown_raises(tmp_path: Path):
    manager = DownloadManager(tmp_path)
    with pytest.raises(ValueError, match="Unknown model"):
        manager.model_dir("nonexistent")


def test_get_model_path_alias(tmp_path: Path):
    manager = DownloadManager(tmp_path)
    assert manager.get_model_path("whisper_base") == manager.model_dir("whisper_base")
