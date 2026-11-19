from tempfile import TemporaryDirectory

import pytest

from backend.app.graph.models import (
    CallResolution,
    DependencyInfo,
    DependencyResolution,
    FunctionCallInfo,
    GraphNode,
    GraphRelationship,
    RepositoryAnalysisResult,
)
from backend.app.graph.repository_graph import file_node_id, symbol_node_id
from backend.app.intelligence.impact_analyzer import (
    ChangeImpactAnalyzer,
    ImpactFileNotInGraph,
    ImpactSymbolNotFound,
)
from backend.app.intelligence.impact_models import (
    ChangeImpactReport,
    ImpactAnalysisRequest,
)


def _file(path):
    return GraphNode(
        node_id=file_node_id(path),
        node_type="File",
        file_path=path,
    )


def _symbol(path, qualified_name, symbol_type="function", start=2, end=4):
    return GraphNode(
        node_id=symbol_node_id(path, qualified_name),
        node_type="Function" if symbol_type == "function" else "Method",
        qualified_name=qualified_name,
        file_path=path,
        symbol_type=symbol_type,
        start_line=start,
        end_line=end,
    )


def _analysis(files, relationships=(), symbols=(), dependencies=(), calls=()):
    return RepositoryAnalysisResult(
        nodes=[*[_file(path) for path in files], *symbols],
        relationships=list(relationships),
        dependencies=list(dependencies),
        function_calls=list(calls),
    )


def _import(source, target, line=1, status="internal"):
    return GraphRelationship(
        source=file_node_id(source),
        target=file_node_id(target),
        relationship_type="IMPORTS",
        source_lines=[line],
        resolution_status=status,
    )


def _run(analysis, question="src/base.py", **options):
    request = ImpactAnalysisRequest(changed_file_path=question, **options)
    with TemporaryDirectory() as temporary_root:
        return ChangeImpactAnalyzer().analyze(
            "repo-1",
            temporary_root,
            analysis,
            request,
        )


def test_finds_direct_and_multihop_dependents_with_actual_paths():
    analysis = _analysis(
        ["src/base.py", "src/service.py", "src/app.py", "src/other.py"],
        [
            _import("src/service.py", "src/base.py", 5),
            _import("src/app.py", "src/service.py", 8),
            _import("src/base.py", "src/other.py", 11),
        ],
    )

    report = _run(analysis)

    assert [item.node.file_path for item in report.directly_affected] == ["src/service.py"]
    assert [item.node.file_path for item in report.indirectly_affected] == ["src/app.py"]
    indirect_path = report.indirectly_affected[0].paths[0]
    assert indirect_path.node_ids == [
        file_node_id("src/app.py"),
        file_node_id("src/service.py"),
        file_node_id("src/base.py"),
    ]
    assert [(edge.source_node_id, edge.target_node_id) for edge in indirect_path.relationships] == [
        (file_node_id("src/app.py"), file_node_id("src/service.py")),
        (file_node_id("src/service.py"), file_node_id("src/base.py")),
    ]
    assert report.affected_files[0].file_path == "src/app.py"


def test_only_follows_incoming_edges_and_respects_depth_limit():
    analysis = _analysis(
        ["src/base.py", "src/direct.py", "src/indirect.py", "src/outgoing.py"],
        [
            _import("src/direct.py", "src/base.py"),
            _import("src/indirect.py", "src/direct.py"),
            _import("src/base.py", "src/outgoing.py"),
        ],
    )

    report = _run(analysis, max_depth=1)

    assert [item.node.file_path for item in report.directly_affected] == ["src/direct.py"]
    assert report.indirectly_affected == []
    assert "src/outgoing.py" not in {item.file_path for item in report.affected_files}


def test_cycles_do_not_revisit_changed_nodes_or_duplicate_impacts():
    analysis = _analysis(
        ["src/base.py", "src/caller.py"],
        [
            _import("src/caller.py", "src/base.py"),
            _import("src/base.py", "src/caller.py"),
        ],
    )

    report = _run(analysis)

    assert len(report.directly_affected) == 1
    assert report.directly_affected[0].node.file_path == "src/caller.py"
    assert report.indirectly_affected == []


def test_missing_file_graph_node_and_missing_symbol_are_reported():
    analysis = _analysis(["src/base.py"], symbols=[_symbol("src/base.py", "known")])

    with pytest.raises(ImpactFileNotInGraph):
        _run(analysis, question="src/missing.py")
    with pytest.raises(ImpactSymbolNotFound):
        _run(analysis, qualified_symbol_name="missing")


def test_symbol_scope_limits_changed_component_to_that_symbol():
    symbols = [
        _symbol("src/base.py", "Target.changed"),
        _symbol("src/base.py", "Target.stable", symbol_type="method"),
        _symbol("src/caller.py", "use_changed"),
        _symbol("src/caller.py", "use_stable"),
    ]
    calls = [
        FunctionCallInfo(
            caller_qualified_name="use_changed",
            called_name="Target.changed",
            source_file="src/caller.py",
            line_number=6,
            call_expression="Target.changed()",
            resolution_status=CallResolution.RESOLVED,
            target_qualified_name="Target.changed",
            target_file="src/base.py",
        ),
        FunctionCallInfo(
            caller_qualified_name="use_stable",
            called_name="Target.stable",
            source_file="src/caller.py",
            line_number=10,
            call_expression="Target.stable()",
            resolution_status=CallResolution.RESOLVED,
            target_qualified_name="Target.stable",
            target_file="src/base.py",
        ),
    ]
    analysis = _analysis(
        ["src/base.py", "src/caller.py"],
        [GraphRelationship(
            source=symbol_node_id("src/caller.py", "use_changed"),
            target=symbol_node_id("src/base.py", "Target.changed"),
            relationship_type="CALLS",
            source_lines=[6],
            resolution_status="resolved",
        ), GraphRelationship(
            source=symbol_node_id("src/caller.py", "use_stable"),
            target=symbol_node_id("src/base.py", "Target.stable"),
            relationship_type="CALLS",
            source_lines=[10],
            resolution_status="resolved",
        )],
        symbols=symbols,
        calls=calls,
    )

    report = _run(analysis, qualified_symbol_name="Target.changed")

    assert report.changed_component.qualified_name == "Target.changed"
    assert [item.node.qualified_name for item in report.directly_affected] == ["use_changed"]


def test_file_scope_call_path_preserves_the_exact_changed_symbol_target():
    changed = _symbol("src/base.py", "run")
    caller = _symbol("src/caller.py", "invoke")
    analysis = _analysis(
        ["src/base.py", "src/caller.py"],
        [GraphRelationship(
            source=caller.node_id,
            target=changed.node_id,
            relationship_type="CALLS",
            source_lines=[7],
            resolution_status="resolved",
        )],
        symbols=[changed, caller],
    )

    report = _run(analysis)

    assert report.changed_component.node_type == "File"
    assert changed.node_id in {node.node_id for node in report.changed_targets}
    assert report.directly_affected[0].paths[0].node_ids[-1] == changed.node_id
    assert ChangeImpactReport.model_validate(report.model_dump())


def test_symbol_scope_does_not_confirm_file_imports_as_symbol_usage():
    target_path = "src/base.py"
    importer_path = "src/importer.py"
    changed_symbol = _symbol(target_path, "Target.changed")
    analysis = _analysis(
        [target_path, importer_path],
        [_import(importer_path, target_path, line=12)],
        symbols=[changed_symbol],
    )

    report = _run(
        analysis,
        question=target_path,
        qualified_symbol_name="Target.changed",
    )

    assert report.directly_affected == []
    assert report.indirectly_affected == []


def test_unresolved_import_remains_an_uncertain_file_level_candidate():
    target_path = "src/base.py"
    importer_path = "src/importer.py"
    analysis = _analysis(
        [target_path, importer_path],
        dependencies=[DependencyInfo(
            source_file=importer_path,
            module_name="src.base",
            line_number=12,
            resolution=DependencyResolution.UNRESOLVED,
        )],
    )

    report = _run(analysis, question=target_path)

    assert [item.node.file_path for item in report.directly_affected] == [importer_path]
    item = report.directly_affected[0]
    assert item.certainty == "low"
    assert item.paths[0].relationships[0].uncertain_candidate
    assert "unresolved" in item.explanation


def test_unresolved_import_and_ambiguous_call_are_marked_as_uncertain_candidates():
    importer = "src/importer.py"
    caller = "src/caller.py"
    target_path = "src/base.py"
    caller_symbol = _symbol(caller, "caller")
    changed_symbol = _symbol(target_path, "run")
    analysis = _analysis(
        [target_path, importer, caller],
        symbols=[caller_symbol, changed_symbol],
        dependencies=[DependencyInfo(
            source_file=importer,
            module_name="src.base",
            line_number=2,
            resolution=DependencyResolution.UNRESOLVED,
        )],
        calls=[FunctionCallInfo(
            caller_qualified_name="caller",
            called_name="run",
            source_file=caller,
            line_number=7,
            call_expression="run()",
            resolution_status=CallResolution.AMBIGUOUS,
        )],
    )

    report = _run(analysis, question=target_path, qualified_symbol_name="run")

    uncertain = [item for item in report.directly_affected if item.certainty == "low"]
    assert {item.node.file_path for item in uncertain} == {caller}
    assert all(
        path.relationships[0].uncertain_candidate
        for item in uncertain
        for path in item.paths
    )
    assert any("unresolved or ambiguous" in warning for warning in report.warnings)


def test_test_discovery_is_separate_from_inferred_recommendations(tmp_path):
    target = "src/base.py"
    affected = "src/caller.py"
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_caller.py").write_text("# caller test\n", encoding="utf-8")
    analysis = _analysis(
        [target, affected],
        [_import(affected, target)],
        symbols=[_symbol(affected, "caller")],
    )

    report = ChangeImpactAnalyzer().analyze(
        "repo-1",
        tmp_path,
        analysis,
        ImpactAnalysisRequest(changed_file_path=target),
    )

    assert [item.test_file_path for item in report.discovered_test_files] == [
        "tests/test_caller.py"
    ]
    assert report.inferred_test_recommendations[0].affected_file_path == affected
    assert "not a claim" in report.inferred_test_recommendations[0].explanation


def test_result_limit_is_bounded_and_reported():
    affected_paths = [f"src/caller_{index}.py" for index in range(4)]
    analysis = _analysis(
        ["src/base.py", *affected_paths],
        [_import(path, "src/base.py") for path in affected_paths],
    )

    report = _run(analysis, max_results=2)

    assert len(report.directly_affected) == 2
    assert report.result_limit_reached
    assert report.warnings


def test_report_validation_rejects_unsafe_component_paths_and_malformed_edges():
    analysis = _analysis(
        ["src/base.py", "src/caller.py"],
        [_import("src/caller.py", "src/base.py", line=8)],
    )
    payload = _run(analysis).model_dump(mode="python")

    unsafe_path = ChangeImpactReport.model_validate(payload).model_dump(mode="python")
    unsafe_path["directly_affected"][0]["node"]["file_path"] = "../outside.py"
    with pytest.raises(ValueError, match="repository-relative path"):
        ChangeImpactReport.model_validate(unsafe_path)

    malformed_edge = ChangeImpactReport.model_validate(payload).model_dump(mode="python")
    malformed_edge["directly_affected"][0]["paths"][0]["relationships"][0][
        "source_node_id"
    ] = "file:unrelated.py"
    with pytest.raises(ValueError, match="adjacent path nodes"):
        ChangeImpactReport.model_validate(malformed_edge)


def test_report_validation_rejects_unreported_path_nodes_and_unsafe_test_paths():
    analysis = _analysis(
        ["src/base.py", "src/caller.py", "src/app.py"],
        [
            _import("src/caller.py", "src/base.py", line=8),
            _import("src/app.py", "src/caller.py", line=12),
        ],
    )
    payload = _run(analysis).model_dump(mode="python")

    unreported_node = ChangeImpactReport.model_validate(payload).model_dump(mode="python")
    indirect_path = unreported_node["indirectly_affected"][0]["paths"][0]
    indirect_path["node_ids"][1] = "file:src/unknown.py"
    indirect_path["relationships"][0]["target_node_id"] = "file:src/unknown.py"
    indirect_path["relationships"][1]["source_node_id"] = "file:src/unknown.py"
    with pytest.raises(ValueError, match="absent from the report"):
        ChangeImpactReport.model_validate(unreported_node)

    payload["discovered_test_files"] = [
        {"test_file_path": "../outside_test.py", "reason": "matched"}
    ]
    with pytest.raises(ValueError, match="safe repository-relative"):
        ChangeImpactReport.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [("max_depth", -1), ("max_depth", 11), ("max_results", 0), ("max_results", 1_001)],
)
def test_report_validation_rejects_out_of_range_limits(field, value):
    analysis = _analysis(["src/base.py"])
    payload = _run(analysis).model_dump(mode="python")
    payload[field] = value

    with pytest.raises(ValueError):
        ChangeImpactReport.model_validate(payload)


def test_report_validation_rejects_component_depth_over_max_depth():
    analysis = _analysis(
        ["src/base.py", "src/caller.py"],
        [_import("src/caller.py", "src/base.py")],
    )
    payload = _run(analysis).model_dump(mode="python")
    payload["max_depth"] = 0

    with pytest.raises(ValueError, match="cannot exceed max_depth"):
        ChangeImpactReport.model_validate(payload)


def test_report_validation_rejects_component_count_over_max_results():
    callers = ["src/first.py", "src/second.py"]
    analysis = _analysis(
        ["src/base.py", *callers],
        [_import(path, "src/base.py") for path in callers],
    )
    payload = _run(analysis, max_results=2).model_dump(mode="python")
    payload["max_results"] = 1

    with pytest.raises(ValueError, match="component count cannot exceed max_results"):
        ChangeImpactReport.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("minimum_depth", 2, "minimum_depth"),
        ("impact_type", "indirect", "impact_type"),
        ("certainty", "low", "certainty"),
    ],
)
def test_report_validation_rejects_inconsistent_affected_file_metadata(
    field, value, message
):
    analysis = _analysis(
        ["src/base.py", "src/caller.py"],
        [_import("src/caller.py", "src/base.py")],
    )
    payload = _run(analysis).model_dump(mode="python")
    payload["affected_files"][0][field] = value

    with pytest.raises(ValueError, match=message):
        ChangeImpactReport.model_validate(payload)


@pytest.mark.parametrize("path", ["../outside.py", "/absolute.py", "C:\\outside.py", ""])
def test_impact_request_rejects_non_repository_relative_paths(path):
    with pytest.raises(ValueError):
        ImpactAnalysisRequest(changed_file_path=path)


def test_impact_request_rejects_invalid_depth_and_symbol_name():
    with pytest.raises(ValueError):
        ImpactAnalysisRequest(changed_file_path="src/base.py", max_depth=11)
    with pytest.raises(ValueError):
        ImpactAnalysisRequest(changed_file_path="src/base.py", qualified_symbol_name="bad-name")
