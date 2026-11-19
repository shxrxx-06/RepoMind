from uuid import uuid4

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from backend.app.api import routes
from backend.app.graph.models import (
    GraphNode,
    GraphRelationship,
    RepositoryAnalysisResult,
)
from backend.app.intelligence.impact_models import ImpactAnalysisRequest
from backend.app.investigation.models import (
    InvestigationRequest,
    InvestigationResponse,
    InvestigationTrace,
)
from backend.app.parsing.models import ParseError
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
        self.graph_calls = []

    @staticmethod
    def repository_fingerprint(repository_path):
        return "|".join(
            f"{path.relative_to(repository_path).as_posix()}:{path.read_bytes().hex()}"
            for path in sorted(repository_path.rglob("*.py"))
        )

    def for_repository(self, repository_id, repository_path):
        self.calls.append((repository_id, repository_path))
        if self.error:
            raise self.error
        return self.service

    def analysis_for_repository(self, repository_id, repository_path):
        self.graph_calls.append((repository_id, repository_path))
        if self.error:
            raise self.error
        return RepositoryAnalysisResult(
            nodes=[
                GraphNode(
                    node_id="file:src/module.py",
                    node_type="File",
                    file_path="src/module.py",
                ),
                GraphNode(
                    node_id="symbol:src/module.py:run",
                    node_type="Function",
                    qualified_name="run",
                    symbol_type="function",
                    file_path="src/module.py",
                    start_line=3,
                    end_line=5,
                ),
                GraphNode(
                    node_id="file:src/caller.py",
                    node_type="File",
                    file_path="src/caller.py",
                ),
            ],
            relationships=[
                GraphRelationship(
                    source="file:src/module.py",
                    target="symbol:src/module.py:run",
                    relationship_type="DEFINES",
                    source_lines=[],
                    resolution_status="resolved",
                ),
                GraphRelationship(
                    source="file:src/caller.py",
                    target="file:src/module.py",
                    relationship_type="IMPORTS",
                    source_lines=[9],
                    resolution_status="internal",
                ),
            ],
        )


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


def test_graph_endpoint_serializes_existing_nodes_and_relationships(tmp_path, monkeypatch):
    repository_id = str(uuid4())
    repository_path = tmp_path / repository_id
    repository_path.mkdir()
    runtime = FakeRuntime()
    monkeypatch.setattr(routes.settings, "REPOSITORIES_DIR", tmp_path)
    monkeypatch.setattr(routes, "investigation_runtime", runtime)

    response = routes.get_repository_graph(repository_id)

    assert runtime.graph_calls == [(repository_id, repository_path.resolve())]
    payload = response.model_dump(mode="json")
    assert payload["nodes"][1] == {
        "node_id": "symbol:src/module.py:run",
        "node_type": "Function",
        "qualified_name": "run",
        "file_path": "src/module.py",
        "symbol_type": "function",
        "start_line": 3,
        "end_line": 5,
    }
    assert payload["relationships"][0] == {
        "source": "file:src/module.py",
        "target": "symbol:src/module.py:run",
        "relationship_type": "DEFINES",
        "source_lines": [],
        "resolution_status": "resolved",
    }
    assert payload["relationships"][1]["relationship_type"] == "IMPORTS"


def test_graph_endpoint_rejects_invalid_and_missing_repository_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(routes.settings, "REPOSITORIES_DIR", tmp_path)
    runtime = FakeRuntime()
    monkeypatch.setattr(routes, "investigation_runtime", runtime)

    with pytest.raises(HTTPException) as invalid:
        routes.get_repository_graph("invalid-id")
    with pytest.raises(HTTPException) as missing:
        routes.get_repository_graph(str(uuid4()))

    assert invalid.value.status_code == 404
    assert missing.value.status_code == 404
    assert runtime.graph_calls == []


def test_graph_endpoint_hides_graph_generation_failure_details(tmp_path, monkeypatch):
    repository_id = str(uuid4())
    (tmp_path / repository_id).mkdir()
    monkeypatch.setattr(routes.settings, "REPOSITORIES_DIR", tmp_path)
    monkeypatch.setattr(
        routes,
        "investigation_runtime",
        FakeRuntime(error=InvestigationRuntimeError("local path must remain private")),
    )

    with pytest.raises(HTTPException) as failure:
        routes.get_repository_graph(repository_id)

    assert failure.value.status_code == 503
    assert failure.value.detail == "Repository graph is unavailable"
    assert "local path must remain private" not in failure.value.detail


def test_graph_response_sanitizes_local_filesystem_paths(tmp_path):
    repository_path = tmp_path / "repository-id"
    result = routes._repository_graph_response(
        RepositoryAnalysisResult(errors=[
            ParseError(
                message=f"Unable to read {repository_path}\\src\\broken.py",
                error_type="OSError",
            )
        ]),
        repository_path,
    )

    assert result.errors[0].message == "Python source could not be parsed or read."
    assert str(repository_path) not in result.errors[0].message


def test_impact_endpoint_returns_serialized_incoming_impact_report(tmp_path, monkeypatch):
    repository_id = str(uuid4())
    repository = tmp_path / repository_id
    (repository / "src").mkdir(parents=True)
    (repository / "src" / "module.py").write_text("def run(): pass\n", encoding="utf-8")
    (repository / "src" / "caller.py").write_text("import src.module\n", encoding="utf-8")
    runtime = FakeRuntime()
    monkeypatch.setattr(routes.settings, "REPOSITORIES_DIR", tmp_path)
    monkeypatch.setattr(routes, "investigation_runtime", runtime)

    response = routes.analyze_repository_impact(
        repository_id,
        ImpactAnalysisRequest(changed_file_path="src/module.py"),
    )

    payload = response.model_dump(mode="json")
    assert runtime.graph_calls == [(repository_id, repository.resolve())]
    assert payload["changed_component"]["node_id"] == "file:src/module.py"
    assert payload["directly_affected"][0]["node"]["file_path"] == "src/caller.py"
    assert payload["directly_affected"][0]["paths"][0]["relationships"][0]["source_lines"] == [9]


def test_impact_endpoint_validates_repository_file_and_symbol_inputs(tmp_path, monkeypatch):
    repository_id = str(uuid4())
    repository = tmp_path / repository_id
    (repository / "src").mkdir(parents=True)
    (repository / "src" / "module.py").write_text("def run(): pass\n", encoding="utf-8")
    monkeypatch.setattr(routes.settings, "REPOSITORIES_DIR", tmp_path)
    runtime = FakeRuntime()
    monkeypatch.setattr(routes, "investigation_runtime", runtime)

    with pytest.raises(HTTPException) as invalid_id:
        routes.analyze_repository_impact(
            "bad-id", ImpactAnalysisRequest(changed_file_path="src/module.py")
        )
    with pytest.raises(HTTPException) as missing_file:
        routes.analyze_repository_impact(
            repository_id, ImpactAnalysisRequest(changed_file_path="src/missing.py")
        )
    with pytest.raises(HTTPException) as missing_symbol:
        routes.analyze_repository_impact(
            repository_id,
            ImpactAnalysisRequest(
                changed_file_path="src/module.py", qualified_symbol_name="absent"
            ),
        )

    assert invalid_id.value.status_code == 404
    assert missing_file.value.status_code == 404
    assert missing_symbol.value.status_code == 404


def test_impact_endpoint_rejects_repository_changes_during_analysis(tmp_path, monkeypatch):
    repository_id = str(uuid4())
    repository = tmp_path / repository_id
    (repository / "src").mkdir(parents=True)
    source = repository / "src" / "module.py"
    source.write_text("def run(): pass\n", encoding="utf-8")
    monkeypatch.setattr(routes.settings, "REPOSITORIES_DIR", tmp_path)
    monkeypatch.setattr(routes, "investigation_runtime", FakeRuntime())

    original_analyzer = routes.ChangeImpactAnalyzer

    class MutatingAnalyzer:
        def analyze(self, *args, **kwargs):
            result = original_analyzer().analyze(*args, **kwargs)
            source.write_text("def run(): return True\n", encoding="utf-8")
            return result

    monkeypatch.setattr(routes, "ChangeImpactAnalyzer", MutatingAnalyzer)

    with pytest.raises(HTTPException) as changed:
        routes.analyze_repository_impact(
            repository_id,
            ImpactAnalysisRequest(changed_file_path="src/module.py"),
        )

    assert changed.value.status_code == 409
    assert changed.value.detail == (
        "Repository changed during impact analysis; retry against the updated snapshot"
    )
