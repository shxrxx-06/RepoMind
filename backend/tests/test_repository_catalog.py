import json
from uuid import uuid4

from backend.app.api import routes
from backend.app.services.repository_catalog import (
    load_repository_summary,
    save_repository_summary,
)


def test_repository_summary_persists_only_valid_listing_fields(tmp_path, monkeypatch):
    repository_id = str(uuid4())
    monkeypatch.setattr(routes.settings, "DATA_DIR", tmp_path)

    save_repository_summary(
        repository_id,
        {
            "total_files": 4,
            "language_distribution": {"Python": 3, "JavaScript": 1},
            "files": [{"path": "untrusted/ignored.py"}],
        },
    )

    assert load_repository_summary(repository_id) == {
        "total_files": 4,
        "language_distribution": {"JavaScript": 1, "Python": 3},
    }
    stored = json.loads(
        (tmp_path / "repository_metadata" / f"{repository_id}.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(stored) == {"total_files", "language_distribution"}


def test_repository_summary_rejects_malformed_or_missing_metadata(tmp_path, monkeypatch):
    repository_id = str(uuid4())
    monkeypatch.setattr(routes.settings, "DATA_DIR", tmp_path)
    assert load_repository_summary(repository_id) is None

    metadata_file = tmp_path / "repository_metadata" / f"{repository_id}.json"
    metadata_file.parent.mkdir(parents=True)
    metadata_file.write_text('{"total_files": "../../outside"}', encoding="utf-8")

    assert load_repository_summary(repository_id) is None


def test_repository_listing_uses_persisted_summary_without_scanning(tmp_path, monkeypatch):
    repository_id = str(uuid4())
    (tmp_path / "repos" / repository_id).mkdir(parents=True)
    monkeypatch.setattr(routes.settings, "REPOSITORIES_DIR", tmp_path / "repos")
    monkeypatch.setattr(routes.settings, "DATA_DIR", tmp_path / "data")
    save_repository_summary(
        repository_id,
        {"total_files": 12, "language_distribution": {"Python": 8}},
    )
    monkeypatch.setattr(
        routes,
        "scan_repository",
        lambda path: (_ for _ in ()).throw(AssertionError("should use persisted summary")),
    )

    listed = routes.list_repositories()

    assert len(listed) == 1
    assert listed[0].total_files == 12
    assert listed[0].language_distribution == {"Python": 8}


def test_repository_listing_scans_once_for_missing_or_malformed_metadata(
    tmp_path, monkeypatch
):
    repository_id = str(uuid4())
    repository = tmp_path / "repos" / repository_id
    repository.mkdir(parents=True)
    metadata_root = tmp_path / "data" / "repository_metadata"
    metadata_root.mkdir(parents=True)
    (metadata_root / f"{repository_id}.json").write_text("not json", encoding="utf-8")
    monkeypatch.setattr(routes.settings, "REPOSITORIES_DIR", tmp_path / "repos")
    monkeypatch.setattr(routes.settings, "DATA_DIR", tmp_path / "data")
    scans = 0

    def scan(path):
        nonlocal scans
        scans += 1
        assert path == repository
        return {
            "total_files": 2,
            "language_distribution": {"Python": 2},
            "files": [{"path": "source.py"}],
        }

    monkeypatch.setattr(routes, "scan_repository", scan)

    first = routes.list_repositories()
    second = routes.list_repositories()

    assert scans == 1
    assert first == second
    assert first[0].total_files == 2
    assert load_repository_summary(repository_id) == {
        "total_files": 2,
        "language_distribution": {"Python": 2},
    }
