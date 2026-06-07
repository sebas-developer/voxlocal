from pathlib import Path

import pytest

from voxlocal._download import (
    SENSEVOICE_FILES,
    SUPERTONIC_FILES,
    DownloadManager,
    DownloadProgress,
)


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
