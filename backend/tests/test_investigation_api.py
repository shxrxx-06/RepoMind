from uuid import uuid4

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from backend.app.api import routes
from backend.app.investigation.models import (
    InvestigationRequest,
    InvestigationResponse,
    InvestigationTrace,
)
from backend.app.rag.models import EvidenceItem
from backend.app.services.investigation_runtime import InvestigationRuntimeError


def _response() -> InvestigationResponse:
    return InvestigationResponse(
        question="Where is the scanner?",
        answer="One finding is verified.",
        trace=InvestigationTrace(steps=[], stop_reason="No connected graph nodes."),
        context_budget_tokens=8_000,
        evidence=[
            EvidenceItem(
                evidence_id="E-0123456789abcdef01234567",
                document_id="doc-1",
                qualified_name="scan_repository",
                source_file_path="backend/scanner.py",
                start_line=4,
                end_line=8,
                source_content="def scan_repository(path):\n    return path",
                source_content_missing=False,
                indexed_text="scanner source",
                hybrid_score=0.02,
                hybrid_rank=1,
            )
        ],
    )


class FakeService:
    def __init__(self, result=None, error=None):
        self.result = result or _response()
        self.error = error
        self.requests = []

    def investigate(self, request):
        self.requests.append(request)
        if self.error:
            raise self.error
        return self.result


class FakeRuntime:
    def __init__(self, service=None, error=None):
        self.service = service or FakeService()
        self.error = error
        self.calls = []

    def for_repository(self, repository_id, repository_path):
        self.calls.append((repository_id, repository_path))
        if self.error:
            raise self.error
        return self.service


def test_investigation_endpoint_invokes_agent_and_serializes_evidence(tmp_path, monkeypatch):
    repository_id = str(uuid4())
    repository_path = tmp_path / repository_id
    repository_path.mkdir()
    runtime = FakeRuntime()
    monkeypatch.setattr(routes.settings, "REPOSITORIES_DIR", tmp_path)
    monkeypatch.setattr(routes, "investigation_runtime", runtime)

    response = routes.investigate_repository(
        repository_id,
        InvestigationRequest(question="Where is the scanner?"),
    )

    assert runtime.calls == [(repository_id, repository_path.resolve())]
    assert runtime.service.requests[0].question == "Where is the scanner?"
    payload = response.model_dump(mode="json")
    assert payload["evidence"][0]["source_file_path"] == "backend/scanner.py"
    assert payload["evidence"][0]["start_line"] == 4


def test_investigation_endpoint_rejects_invalid_and_missing_repository_ids(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(routes.settings, "REPOSITORIES_DIR", tmp_path)
    runtime = FakeRuntime()
    monkeypatch.setattr(routes, "investigation_runtime", runtime)

    with pytest.raises(HTTPException) as invalid:
        routes.investigate_repository("not-an-id", InvestigationRequest(question="question"))
    with pytest.raises(HTTPException) as missing:
        routes.investigate_repository(
            str(uuid4()), InvestigationRequest(question="question")
        )

    assert invalid.value.status_code == 404
    assert missing.value.status_code == 404
    assert runtime.calls == []


def test_investigation_request_rejects_empty_questions():
    with pytest.raises(ValidationError):
        InvestigationRequest(question="  ")


def test_investigation_endpoint_hides_runtime_failure_details(tmp_path, monkeypatch):
    repository_id = str(uuid4())
    (tmp_path / repository_id).mkdir()
    monkeypatch.setattr(routes.settings, "REPOSITORIES_DIR", tmp_path)
    monkeypatch.setattr(
        routes,
        "investigation_runtime",
        FakeRuntime(error=InvestigationRuntimeError("private filesystem detail")),
    )

    with pytest.raises(HTTPException) as failure:
        routes.investigate_repository(repository_id, InvestigationRequest(question="question"))

    assert failure.value.status_code == 503
    assert failure.value.detail == "Repository intelligence is unavailable for investigation"
    assert "private filesystem detail" not in failure.value.detail


def test_repository_list_only_returns_ingested_uuid_directories(tmp_path, monkeypatch):
    repository_id = str(uuid4())
    repository = tmp_path / repository_id
    repository.mkdir()
    (repository / "main.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "not-a-repository").mkdir()
    monkeypatch.setattr(routes.settings, "REPOSITORIES_DIR", tmp_path)
    monkeypatch.setattr(routes.settings, "DATA_DIR", tmp_path / "data")

    repositories = routes.list_repositories()

    assert len(repositories) == 1
    assert repositories[0].repository_id == repository_id
    assert repositories[0].total_files == 1
    assert repositories[0].language_distribution == {"Python": 1}
