# tests/test_download.py
from voxlocal._download import DownloadManager, DownloadProgress


def test_download_progress_fields():
    p = DownloadProgress(
        model_id="test_model",
        percent=50,
        downloaded="12MB",
        total="24MB",
        description="Downloading...",
    )
    assert p.model_id == "test_model"
    assert p.percent == 50
    assert p.downloaded == "12MB"
    assert p.total == "24MB"
    assert p.description == "Downloading..."


def test_download_manager_has_download_method():
    dm = DownloadManager()
    assert hasattr(dm, "download")
    assert callable(dm.download)


def test_download_manager_has_is_downloaded():
    dm = DownloadManager()
    assert hasattr(dm, "is_downloaded")
    assert callable(dm.is_downloaded)
