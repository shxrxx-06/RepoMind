from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.app.intelligence.repository_analyzer import RepositoryAnalyzer
from backend.app.intelligence.test_intelligence import (
    TestIntelligenceAnalyzer as TestAnalyzer,
)
from backend.app.intelligence.test_intelligence_models import (
    TestCodeMapping as CodeMapping,
)
from backend.app.intelligence.test_intelligence_models import (
    TestIntelligenceRequest as IntelligenceRequest,
)
from backend.app.intelligence.test_intelligence_models import (
    TestIntelligenceResponse as IntelligenceResponse,
)
from backend.app.intelligence.test_intelligence_models import (
    TestMappingEvidence as MappingEvidence,
)
from backend.app.parsing.python_parser import PythonParser


def _analyze(root: Path):
    parser = PythonParser()
    parsed = []
    for source_path in sorted(root.rglob("*.py")):
        result = parser.parse_file(source_path)
        result.file_path = source_path.relative_to(root).as_posix()
        parsed.append(result)
    analysis = RepositoryAnalyzer().analyze(root, parsed)
    return parsed, analysis


def _run(root: Path, request=None):
    parsed, analysis = _analyze(root)
    return TestAnalyzer().analyze(
        "repo-id",
        root,
        parsed,
        analysis,
        request or IntelligenceRequest(),
    )


def _run_with_unresolved_calls(root: Path):
    parsed, analysis = _analyze(root)
    unresolved_calls = [
        call.model_copy(
            update={
                "resolution_status": "unresolved",
                "target_file": None,
                "target_qualified_name": None,
            }
        )
        for call in analysis.function_calls
    ]
    analysis = analysis.model_copy(update={"function_calls": unresolved_calls})
    return TestAnalyzer().analyze(
        "repo-id", root, parsed, analysis, IntelligenceRequest()
    )


def test_discovers_test_conventions_cases_imports_and_syntax_errors(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "pkg" / "service.py").write_text(
        "def save(value):\n    return value\n", encoding="utf-8"
    )
    (tmp_path / "tests" / "test_service.py").write_text(
        "from pkg.service import save as save_record\n\n"
        "class TestService:\n"
        "    def test_save(self):\n"
        "        save_record(1)\n",
        encoding="utf-8",
    )
    (tmp_path / "tests" / "service_test.py").write_text(
        "def test_broken(:\n    pass\n", encoding="utf-8"
    )

    response = _run(tmp_path)

    assert [item.file_path for item in response.discovered_test_files] == [
        "tests/service_test.py",
        "tests/test_service.py",
    ]
    broken = response.discovered_test_files[0]
    assert broken.parse_errors[0].error_type == "SyntaxError"
    valid = response.discovered_test_files[1]
    assert valid.imports[0].alias == "save_record"
    assert [(item.name, item.kind, item.start_line, item.end_line) for item in valid.tests] == [
        ("TestService", "class", 3, 5),
        ("test_save", "method", 4, 5),
    ]
    assert any("could not be fully parsed" in warning for warning in response.warnings)


def test_maps_resolved_test_calls_with_source_evidence_and_no_runtime_claim(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "pkg" / "service.py").write_text(
        "def save(value):\n    return value\n", encoding="utf-8"
    )
    (tmp_path / "tests" / "test_service.py").write_text(
        "from pkg.service import save as save_record\n\n"
        "def test_save():\n    assert save_record(1) == 1\n",
        encoding="utf-8",
    )

    response = _run(tmp_path)

    mapping = next(
        item
        for item in response.mappings
        if item.source_component.qualified_name == "save"
        and item.test_qualified_name == "test_save"
        and item.evidence_classification == "direct_resolved_call"
    )
    assert mapping.test_file_path == "tests/test_service.py"
    assert mapping.test_qualified_name == "test_save"
    assert mapping.certainty == "high"
    call_evidence = next(
        evidence for evidence in mapping.evidence if evidence.evidence_type == "resolved_call"
    )
    assert call_evidence.line_number == 4
    assert "does not prove" in call_evidence.description
    import_mapping = next(
        item
        for item in response.mappings
        if item.source_component.qualified_name == "save"
        and item.evidence_classification == "import_only_reference"
    )
    assert import_mapping.certainty == "medium"
    assert {item.evidence_type for item in import_mapping.evidence} == {"imported_symbol"}
    assert any("no test execution" in warning for warning in response.warnings)
    assert all(item.source_component.qualified_name != "save" for item in response.uncovered_components)


def test_import_without_call_remains_import_only_and_is_recommended_for_review(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "app" / "user.py").write_text(
        "def authenticate(token):\n    return token\n", encoding="utf-8"
    )
    (tmp_path / "tests" / "test_user.py").write_text(
        "from app.user import authenticate\n\n"
        "def test_user_creation():\n    assert True\n",
        encoding="utf-8",
    )

    response = _run(
        tmp_path,
        IntelligenceRequest(
            changed_file_path="app/user.py",
            qualified_symbol_name="authenticate",
        ),
    )

    mapping = next(item for item in response.mappings if item.source_component.qualified_name == "authenticate")
    assert mapping.evidence_classification == "import_only_reference"
    assert mapping.certainty == "medium"
    assert mapping.evidence[0].evidence_type == "imported_symbol"
    uncovered = next(item for item in response.uncovered_components if item.source_component.qualified_name == "authenticate")
    assert uncovered.evidence_classification == "import_only_reference"
    assert "no resolved call" in uncovered.explanation
    suggestion = next(item for item in response.inferred_recommendations if item.source_component.qualified_name == "authenticate")
    assert suggestion.evidence_classification == "import_only_reference"
    assert "not an existing test" in suggestion.explanation


def test_function_local_import_does_not_leak_into_another_test_case(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "app" / "service.py").write_text(
        "def authenticate():\n    return True\n", encoding="utf-8"
    )
    (tmp_path / "tests" / "test_auth.py").write_text(
        "def test_a():\n"
        "    from app.service import authenticate\n"
        "    authenticate()\n\n"
        "def test_b():\n"
        "    authenticate()\n",
        encoding="utf-8",
    )

    response = _run(tmp_path)

    assert any(
        mapping.test_qualified_name == "test_a"
        and mapping.evidence_classification == "direct_resolved_call"
        for mapping in response.mappings
    )
    test_b_mapping = next(
        mapping
        for mapping in response.mappings
        if mapping.test_qualified_name == "test_b"
    )
    assert test_b_mapping.evidence_classification == "uncertain_candidate"
    assert test_b_mapping.certainty == "low"
    assert test_b_mapping.evidence[0].expression == "authenticate()"
    assert "import:" not in test_b_mapping.evidence[0].expression


def test_unrelated_local_import_does_not_suppress_name_only_candidate(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "app" / "service.py").write_text(
        "def authenticate():\n    return True\n\n"
        "def persist():\n    return True\n",
        encoding="utf-8",
    )
    (tmp_path / "tests" / "test_service.py").write_text(
        "def test_a():\n"
        "    from app.service import authenticate\n"
        "    assert authenticate()\n\n"
        "def test_b():\n"
        "    persist()\n",
        encoding="utf-8",
    )

    response = _run_with_unresolved_calls(tmp_path)
    candidates = [
        mapping
        for mapping in response.mappings
        if mapping.test_qualified_name == "test_b"
        and mapping.source_component.qualified_name == "persist"
    ]

    assert len(candidates) == 1
    assert candidates[0].evidence_classification == "uncertain_candidate"
    assert candidates[0].certainty == "low"
    evidence = candidates[0].evidence[0]
    assert evidence.evidence_type == "unresolved_call_candidate"
    assert evidence.expression == "persist()"
    assert "import:" not in evidence.expression


@pytest.mark.parametrize(
    ("test_source", "test_name"),
    [
        (
            (
                "def test_outer():\n"
                "    def helper():\n"
                "        from app.service import authenticate\n"
                "        authenticate()\n"
                "    authenticate()\n"
            ),
            "test_outer",
        ),
        (
            (
                "class TestBindings:\n"
                "    from app.service import authenticate\n\n"
                "def test_unrelated():\n    authenticate()\n"
            ),
            "test_unrelated",
        ),
    ],
)
def test_nested_or_class_imports_do_not_become_test_bindings(
    tmp_path, test_source, test_name
):
    (tmp_path / "app").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "app" / "service.py").write_text(
        "def authenticate():\n    return True\n", encoding="utf-8"
    )
    (tmp_path / "tests" / "test_service.py").write_text(
        test_source, encoding="utf-8"
    )

    response = _run(tmp_path)
    candidates = [
        mapping
        for mapping in response.mappings
        if mapping.test_qualified_name == test_name
        and mapping.source_component.qualified_name == "authenticate"
    ]

    assert len(candidates) == 1
    assert candidates[0].evidence_classification == "uncertain_candidate"
    assert candidates[0].certainty == "low"
    assert candidates[0].evidence[0].evidence_type == "unresolved_call_candidate"
    assert candidates[0].evidence[0].expression == "authenticate()"


@pytest.mark.parametrize(
    ("test_source", "call"),
    [
        (
            (
                "from app.service import authenticate as verify\n\n"
                "def test_authentication():\n    verify()\n"
            ),
            "verify",
        ),
        (
            (
                "def test_authentication():\n"
                "    from app.service import authenticate as verify\n"
                "    verify()\n"
            ),
            "verify",
        ),
        (
            (
                "import app.service as svc\n\n"
                "def test_authentication():\n    svc.authenticate()\n"
            ),
            "svc.authenticate",
        ),
    ],
)
def test_unresolved_aliased_imports_remain_uncertain_candidates(
    tmp_path, test_source, call
):
    (tmp_path / "app").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "app" / "service.py").write_text(
        "def authenticate():\n    return True\n", encoding="utf-8"
    )
    (tmp_path / "tests" / "test_auth.py").write_text(test_source, encoding="utf-8")

    response = _run_with_unresolved_calls(tmp_path)
    candidates = [
        mapping
        for mapping in response.mappings
        if mapping.test_qualified_name == "test_authentication"
        and mapping.evidence_classification == "uncertain_candidate"
    ]

    assert len(candidates) == 1
    assert candidates[0].certainty == "low"
    assert candidates[0].evidence[0].evidence_type == "unresolved_call_candidate"
    assert call in candidates[0].evidence[0].expression
    assert "import" in candidates[0].evidence[0].expression
    assert "authenticate" in candidates[0].evidence[0].expression


def test_duplicate_import_aliases_remain_ambiguous_candidates(tmp_path):
    for module in ("alpha", "beta"):
        (tmp_path / module).mkdir()
        (tmp_path / module / "ops.py").write_text(
            "def execute():\n    return True\n", encoding="utf-8"
        )
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_ops.py").write_text(
        "from alpha.ops import execute as run\n"
        "from beta.ops import execute as run\n\n"
        "def test_both():\n    run()\n",
        encoding="utf-8",
    )

    response = _run_with_unresolved_calls(tmp_path)
    candidates = [
        mapping
        for mapping in response.mappings
        if mapping.test_qualified_name == "test_both"
        and mapping.evidence_classification == "uncertain_candidate"
    ]

    assert len(candidates) == 2
    assert {item.source_component.file_path for item in candidates} == {
        "alpha/ops.py",
        "beta/ops.py",
    }
    assert all(item.certainty == "low" for item in candidates)
    assert all(" as run" in item.evidence[0].expression for item in candidates)


def test_resolved_test_call_maps_the_exact_source_method(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "app" / "service.py").write_text(
        "class UserService:\n"
        "    def authenticate(self, token):\n"
        "        return token\n",
        encoding="utf-8",
    )
    (tmp_path / "tests" / "test_service.py").write_text(
        "from app.service import UserService\n\n"
        "def test_authenticate():\n"
        "    UserService.authenticate(None, 'token')\n",
        encoding="utf-8",
    )

    response = _run(tmp_path)

    mapping = next(
        item
        for item in response.mappings
        if item.source_component.qualified_name == "UserService.authenticate"
        and item.evidence_classification == "direct_resolved_call"
    )
    assert mapping.source_component.file_path == "app/service.py"
    assert mapping.source_component.start_line == 2
    assert mapping.evidence[0].line_number == 4


def test_resolved_call_outside_test_case_is_not_counted_as_a_direct_test_call(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "app" / "service.py").write_text(
        "def run():\n    return True\n", encoding="utf-8"
    )
    (tmp_path / "tests" / "test_service.py").write_text(
        "from app.service import run\nrun()\n", encoding="utf-8"
    )

    response = _run(tmp_path)

    source_mappings = [
        item for item in response.mappings if item.source_component.qualified_name == "run"
    ]
    assert not any(item.evidence_classification == "direct_resolved_call" for item in source_mappings)
    assert any(item.evidence_classification == "import_only_reference" for item in source_mappings)
    assert any(item.evidence_classification == "uncertain_candidate" for item in source_mappings)
    assert any(
        "outside an identified test case" in evidence.description
        for mapping in source_mappings
        for evidence in mapping.evidence
    )


def test_same_name_functions_from_different_modules_remain_ambiguous_candidates(tmp_path):
    for package in ("alpha", "beta", "tests"):
        (tmp_path / package).mkdir()
    for package in ("alpha", "beta"):
        (tmp_path / package / "ops.py").write_text(
            "def execute():\n    return True\n", encoding="utf-8"
        )
    (tmp_path / "tests" / "test_ops.py").write_text(
        "from alpha.ops import execute\n"
        "from beta.ops import execute\n\n"
        "def test_execute():\n    execute()\n",
        encoding="utf-8",
    )

    response = _run(tmp_path)

    candidates = [
        item for item in response.mappings
        if item.evidence_classification == "uncertain_candidate"
    ]
    assert {item.source_component.file_path for item in candidates} == {
        "alpha/ops.py",
        "beta/ops.py",
    }
    assert all(item.certainty == "low" for item in candidates)
    assert not any(
        item.evidence_classification == "direct_resolved_call"
        for item in response.mappings
        if item.source_component.qualified_name == "execute"
    )


def test_symbol_scoped_intelligence_excludes_other_symbols_in_same_file(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "users.py").write_text(
        "def authenticate():\n    return True\n\n"
        "def deactivate():\n    return False\n",
        encoding="utf-8",
    )
    (tmp_path / "tests" / "test_users.py").write_text(
        "from src.users import authenticate, deactivate\n\n"
        "def test_authenticate():\n    authenticate()\n",
        encoding="utf-8",
    )

    response = _run(
        tmp_path,
        IntelligenceRequest(
            changed_file_path="src/users.py",
            qualified_symbol_name="authenticate",
        ),
    )

    assert {node.qualified_name for node in response.impact_report.changed_targets} == {
        "authenticate"
    }
    assert {item.source_component.qualified_name for item in response.mappings} == {
        "authenticate"
    }
    assert all(
        item.source_component.qualified_name != "deactivate"
        for item in response.uncovered_components
    )


def test_unresolved_matching_call_is_only_a_low_certainty_candidate(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "service.py").write_text(
        "def persist():\n    return True\n", encoding="utf-8"
    )
    (tmp_path / "tests" / "test_service.py").write_text(
        "def test_persist():\n    persist()\n", encoding="utf-8"
    )

    response = _run(tmp_path)

    candidates = [
        item
        for item in response.mappings
        if item.evidence_classification == "uncertain_candidate"
    ]
    assert len(candidates) == 1
    assert candidates[0].source_component.qualified_name == "persist"
    assert candidates[0].evidence[0].evidence_type == "unresolved_call_candidate"
    assert "name only" in candidates[0].evidence[0].description
    assert any("uncertain candidates" in warning for warning in response.warnings)
    assert any(
        item.source_component.qualified_name == "persist"
        and item.evidence_classification == "uncertain_candidate"
        for item in response.uncovered_components
    )


def test_filename_similarity_does_not_create_a_test_mapping(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "widget.py").write_text(
        "def render():\n    return None\n", encoding="utf-8"
    )
    (tmp_path / "tests" / "test_widget.py").write_text(
        "def test_placeholder():\n    assert True\n", encoding="utf-8"
    )

    response = _run(tmp_path)

    assert not any(
        item.source_component.file_path == "src/widget.py"
        for item in response.mappings
    )
    assert any(
        item.source_component.qualified_name == "render"
        for item in response.uncovered_components
    )
    assert next(
        item
        for item in response.uncovered_components
        if item.source_component.qualified_name == "render"
    ).evidence_classification == "no_discoverable_test_reference"
    assert any(
        item.source_component.qualified_name == "render"
        for item in response.inferred_recommendations
    )
    assert all(item.recommendation_type == "inferred" for item in response.inferred_recommendations)


def test_impact_scope_associates_tests_and_separates_uncovered_code(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "core.py").write_text(
        "def core():\n    return True\n", encoding="utf-8"
    )
    (tmp_path / "src" / "service.py").write_text(
        "from src.core import core\n\n"
        "def service():\n    return core()\n\n"
        "def orphan():\n    return core()\n",
        encoding="utf-8",
    )
    (tmp_path / "tests" / "test_service.py").write_text(
        "from src.service import service\n\n"
        "def test_service():\n    assert service()\n",
        encoding="utf-8",
    )

    response = _run(
        tmp_path,
        IntelligenceRequest(changed_file_path="src/core.py"),
    )

    assert response.impact_report is not None
    assert response.changed_file_path == "src/core.py"
    assert any(
        mapping.source_component.qualified_name == "service"
        for mapping in response.mappings
    )
    assert any(
        item.source_component.qualified_name == "orphan"
        for item in response.uncovered_components
    )
    assert any(
        item.source_component.qualified_name == "orphan"
        for item in response.inferred_recommendations
    )
    assert all(
        item.source_component.file_path != "tests/test_service.py"
        for item in response.uncovered_components
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"changed_file_path": "../outside.py"},
        {"changed_file_path": "/absolute.py"},
        {"changed_file_path": "src/core.py", "qualified_symbol_name": "bad-name"},
        {"changed_file_path": "src/core.py", "max_depth": 11},
        {"changed_file_path": "src/core.py", "max_results": 1_001},
        {"qualified_symbol_name": "Service.run"},
    ],
)
def test_test_intelligence_request_rejects_malformed_scope(payload):
    with pytest.raises(ValidationError):
        IntelligenceRequest.model_validate(payload)


def test_empty_repository_returns_empty_inventory_with_warning(tmp_path):
    response = _run(tmp_path, request=IntelligenceRequest())

    assert response.discovered_test_files == []
    assert response.mappings == []
    assert response.uncovered_components == []
    assert any("No Python test files" in warning for warning in response.warnings)


def test_repository_wide_results_report_component_limit(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "features.py").write_text(
        "def alpha(): return 1\n"
        "def beta(): return 2\n",
        encoding="utf-8",
    )

    response = _run(tmp_path, IntelligenceRequest(max_results=1))

    assert response.analyzed_component_count == 1
    assert response.max_results == 1
    assert response.result_limit_reached
    assert any("first max_results" in warning for warning in response.warnings)


def test_syntax_errors_are_reported_without_crashing_source_recommendations(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "service.py").write_text(
        "def run():\n    return True\n", encoding="utf-8"
    )
    (tmp_path / "tests" / "test_service.py").write_text(
        "def test_run(:\n    run()\n", encoding="utf-8"
    )

    response = _run(tmp_path)

    assert response.discovered_test_files[0].parse_errors
    assert response.uncovered_components
    assert response.inferred_recommendations


def test_mapping_models_reject_invalid_nodes_certainty_and_source_lines(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "app" / "user.py").write_text(
        "def authenticate():\n    return True\n", encoding="utf-8"
    )
    (tmp_path / "tests" / "test_user.py").write_text(
        "from app.user import authenticate\n\n"
        "def test_user():\n    assert True\n",
        encoding="utf-8",
    )
    response = _run(
        tmp_path,
        IntelligenceRequest(changed_file_path="app/user.py", qualified_symbol_name="authenticate"),
    )
    payload = response.model_dump(mode="python")

    invalid_category = response.mappings[0].model_dump(mode="python")
    invalid_category["evidence_classification"] = "direct_resolved_call"
    invalid_category["certainty"] = "high"
    with pytest.raises(ValidationError, match="resolved-call evidence"):
        CodeMapping.model_validate(invalid_category)

    invalid_node = response.mappings[0].model_dump(mode="python")
    invalid_node["source_component"]["node_id"] = "symbol:elsewhere.py:authenticate"
    with pytest.raises(ValidationError, match="valid graph node identifiers"):
        CodeMapping.model_validate(invalid_node)

    with pytest.raises(ValidationError):
        MappingEvidence(
            evidence_type="resolved_call",
            description="invalid line",
            line_number=0,
        )

    invalid_report = dict(payload)
    invalid_report["inferred_recommendations"] = []
    with pytest.raises(ValidationError, match="correspond exactly"):
        IntelligenceResponse.model_validate(invalid_report)
