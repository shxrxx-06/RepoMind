import stat
import zipfile

import pytest

from backend.app.ingestion.github import validate_github_url
from backend.app.ingestion.scanner import scan_repository
from backend.app.ingestion.zip_handler import extract_repository_zip


def test_scanner_detects_languages(tmp_path):
    (tmp_path / "main.py").write_text("print('Hello')", encoding="utf-8")
    (tmp_path / "app.js").write_text("console.log('Hello')", encoding="utf-8")

    result = scan_repository(tmp_path)

    assert result["total_files"] == 2
    assert result["language_distribution"] == {"JavaScript": 1, "Python": 1}
    assert result["files"][0]["path"] == "app.js"


def test_scanner_ignores_dependencies_and_environment_files(tmp_path):
    ignored = tmp_path / "node_modules"
    ignored.mkdir()
    (ignored / "package.js").write_text("ignored", encoding="utf-8")
    (tmp_path / ".env.production").write_text("SECRET=x", encoding="utf-8")
    (tmp_path / "main.py").write_text("print('test')", encoding="utf-8")

    result = scan_repository(tmp_path)

    assert result["total_files"] == 1
    assert result["skipped_files"] == 1


def test_github_url_validation():
    assert validate_github_url("https://github.com/psf/requests") == (
        "https://github.com/psf/requests.git"
    )
    assert validate_github_url("https://github.com/psf/requests.git") == (
        "https://github.com/psf/requests.git"
    )


@pytest.mark.parametrize(
    "url",
    [
        "http://github.com/psf/requests",
        "https://github.com.evil.test/psf/requests",
        "https://user:pass@github.com/psf/requests",
        "https://github.com/psf/requests/tree/main",
    ],
)
def test_github_url_rejects_unsupported_urls(url):
    with pytest.raises(ValueError):
        validate_github_url(url)


def test_zip_extraction_rejects_path_traversal(tmp_path, monkeypatch):
    from backend.app.core.config import settings

    monkeypatch.setattr(settings, "REPOSITORIES_DIR", tmp_path / "repos")
    settings.REPOSITORIES_DIR.mkdir()
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../outside.txt", "unsafe")

    with pytest.raises(ValueError, match="unsafe path"):
        extract_repository_zip(archive_path)
    assert not (tmp_path / "outside.txt").exists()


def test_zip_extraction_rejects_symlinks(tmp_path, monkeypatch):
    from backend.app.core.config import settings

    monkeypatch.setattr(settings, "REPOSITORIES_DIR", tmp_path / "repos")
    settings.REPOSITORIES_DIR.mkdir()
    archive_path = tmp_path / "symlink.zip"
    info = zipfile.ZipInfo("link")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(info, "target")

    with pytest.raises(ValueError, match="symbolic link"):
        extract_repository_zip(archive_path)
