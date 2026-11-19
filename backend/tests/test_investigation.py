import hashlib

import networkx as nx
import pytest
from pydantic import ValidationError

from backend.app.core.config import Settings
from backend.app.graph.models import RepositoryAnalysisResult
from backend.app.investigation import (
    ClaimVerifier,
    GraphExplorer,
    InvestigationContextBuilder,
    InvestigationPlanner,
    InvestigationRequest,
    InvestigationService,
    TestRecommender,
)
from backend.app.rag import (
    EvidenceBuilder,
    MockLLMProvider,
    PromptBuilder,
    configured_provider,
)
from backend.app.retrieval.models import RetrievalResponse, RetrievalResult


def _result(
    document_id="doc-one",
    *,
    name="Service.run",
    path="src/service.py",
    source="return verify(token)",
    rank=1,
):
    return RetrievalResult(
        document_id=document_id,
        symbol_name=name.rsplit(".", 1)[-1],
        qualified_name=name,
        source_file_path=path,
        symbol_type="method",
        start_line=5,
        end_line=7,
        source_content=source,
        text=f"{name}\n{source or ''}",
        semantic_score=0.8,
        semantic_rank=rank,
        keyword_score=1.0,
        keyword_rank=rank,
        hybrid_score=0.03,
        hybrid_rank=rank,
        ranking_explanation="test result",
    )


def _evidence_id(document_id):
    return "E-" + hashlib.sha256(document_id.encode()).hexdigest()[:24]


class FakeRetriever:
    def __init__(self, initial=None, followups=None, *, fail=False):
        self.initial = initial or []
        self.followups = followups or {}
        self.fail = fail
        self.calls = []

    def search(self, query):
        self.calls.append(query)
        if self.fail:
            raise RuntimeError("retrieval unavailable")
        results = self.initial if len(self.calls) == 1 else self.followups.get(query.query, [])
        return RetrievalResponse(
            query=query.query,
            results=results,
            total_documents=len(results),
            filtered_documents=len(results),
        )


def _graph():
    graph = nx.DiGraph()
    graph.add_node(
        "symbol:src/service.py:Service.run",
        node_type="Method",
        qualified_name="Service.run",
        file_path="src/service.py",
        start_line=5,
        end_line=7,
    )
    graph.add_node(
        "symbol:src/db.py:Database.connect",
        node_type="Method",
        qualified_name="Database.connect",
        file_path="src/db.py",
        start_line=2,
        end_line=3,
    )
    graph.add_edge(
        "symbol:src/service.py:Service.run",
        "symbol:src/db.py:Database.connect",
        relationships=[{"relationship_type": "CALLS", "source_lines": [6]}],
    )
    return graph


def _analysis(graph):
    nodes = []
    relationships = []
    for node_id, attrs in graph.nodes(data=True):
        node = {"node_id": node_id, **attrs}
        node.setdefault("node_type", str(node.get("symbol_type", "File")).title())
        nodes.append(node)
    for source, target, attrs in graph.edges(data=True):
        for relationship in attrs["relationships"]:
            relationships.append({"source": source, "target": target, **relationship})
    return RepositoryAnalysisResult(nodes=nodes, relationships=relationships)


def test_investigation_request_validates_question_and_limits():
    with pytest.raises(ValidationError):
        InvestigationRequest(question="  ")
    with pytest.raises(ValidationError):
        InvestigationRequest(question="Find auth", max_steps=0)


def test_graph_explorer_handles_cycles_depth_and_duplicate_nodes():
    graph = nx.DiGraph()
    graph.add_edge("a", "b", relationships=[{"relationship_type": "CALLS"}])
    graph.add_edge("b", "c", relationships=[{"relationship_type": "IMPORTS"}])
    graph.add_edge("c", "a", relationships=[{"relationship_type": "CONTAINS"}])
    chain = nx.DiGraph()
    chain.add_edge("a", "b", relationships=[{"relationship_type": "CALLS"}])
    chain.add_edge("b", "c", relationships=[{"relationship_type": "IMPORTS"}])

    depth_one = GraphExplorer().explore(chain, ["a"], max_depth=1)
    full = GraphExplorer().explore(graph, ["a"], max_depth=5)

    assert [item["node_id"] for item in depth_one] == ["b"]
    assert {item["node_id"] for item in full} == {"b", "c"}
    assert len(full) == 2
    assert full[0]["path"] == ["a", "b"]
    assert {(edge["source_node_id"], edge["target_node_id"]) for edge in full.relationships} == {
        ("a", "b"), ("b", "c"), ("c", "a")
    }
    assert len(full.relationships) == 3


def test_graph_explorer_preserves_both_paths_in_diamond_graph():
    graph = nx.DiGraph()
    graph.add_edge("A", "B", relationships=[{"relationship_type": "CALLS", "resolution_status": "resolved"}])
    graph.add_edge("A", "C", relationships=[{"relationship_type": "IMPORTS", "resolution_status": "internal"}])
    graph.add_edge("B", "D", relationships=[{"relationship_type": "CALLS", "resolution_status": "resolved"}])
    graph.add_edge("C", "D", relationships=[{"relationship_type": "CONTAINS", "resolution_status": "resolved"}])

    result = GraphExplorer().explore(graph, ["A"], max_depth=2)

    assert [item["node_id"] for item in result].count("D") == 1
    assert {(edge["source_node_id"], edge["target_node_id"]) for edge in result.relationships} == {
        ("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")
    }
    assert len(result.relationships) == 4
    d_paths = {
        tuple(edge["traversal_path"])
        for edge in result.relationships
        if edge["target_node_id"] == "D"
    }
    assert d_paths == {("A", "B", "D"), ("A", "C", "D")}
    assert all(graph.has_edge(edge["source_node_id"], edge["target_node_id"]) for edge in result.relationships)

    limited = GraphExplorer().explore(graph, ["A"], max_depth=3, max_nodes=3)
    assert {item["node_id"] for item in limited} == {"B", "C"}
    assert {(edge["source_node_id"], edge["target_node_id"]) for edge in limited.relationships} == {
        ("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")
    }


def test_graph_explorer_preserves_edges_between_multiple_start_nodes():
    graph = nx.DiGraph()
    graph.add_edge("A", "B", relationships=[{"relationship_type": "IMPORTS", "resolution_status": "internal"}])
    graph.add_edge("B", "C", relationships=[{"relationship_type": "CALLS", "resolution_status": "resolved"}])

    result = GraphExplorer().explore(graph, ["B", "A"], max_depth=1)

    assert {"A", "B"}.isdisjoint({item["node_id"] for item in result})
    assert {(edge["source_node_id"], edge["target_node_id"]) for edge in result.relationships} == {
        ("A", "B"), ("B", "C")
    }
    assert all(graph.has_edge(edge["source_node_id"], edge["target_node_id"]) for edge in result.relationships)


def test_graph_explorer_deduplicates_relationships_in_cycles():
    graph = nx.DiGraph()
    graph.add_edge(
        "A",
        "B",
        relationships=[
            {"relationship_type": "CALLS", "resolution_status": "resolved", "source_lines": [4]},
            {"relationship_type": "CALLS", "resolution_status": "resolved", "source_lines": [4]},
        ],
    )
    graph.add_edge("B", "A", relationships=[{"relationship_type": "IMPORTS", "resolution_status": "internal"}])

    result = GraphExplorer().explore(graph, ["A"], max_depth=5)
    keys = [
        (edge["source_node_id"], edge["target_node_id"], edge["relationship_type"], tuple(edge["source_lines"]))
        for edge in result.relationships
    ]

    assert len(result) == 1
    assert len(keys) == len(set(keys)) == 2
    assert all(graph.has_edge(edge["source_node_id"], edge["target_node_id"]) for edge in result.relationships)


def test_multihop_trace_records_each_actual_relationship_edge():
    graph = nx.DiGraph()
    graph.add_node("symbol:src/a.py:Service.run", qualified_name="Service.run", file_path="src/a.py", symbol_type="method")
    graph.add_node("symbol:src/b.py:Database.connect", qualified_name="Database.connect", file_path="src/b.py", symbol_type="method")
    graph.add_node("symbol:src/c.py:Config.load", qualified_name="Config.load", file_path="src/c.py", symbol_type="function")
    graph.add_edge(
        "symbol:src/a.py:Service.run",
        "symbol:src/b.py:Database.connect",
        relationships=[{"relationship_type": "CALLS", "resolution_status": "resolved", "source_lines": [8]}],
    )
    graph.add_edge(
        "symbol:src/b.py:Database.connect",
        "symbol:src/c.py:Config.load",
        relationships=[{"relationship_type": "IMPORTS", "resolution_status": "internal", "source_lines": [2]}],
    )

    explored = GraphExplorer().explore(graph, ["symbol:src/a.py:Service.run"], max_depth=3)
    relationships = [edge for node in explored for edge in node["relationships"]]
    by_type = {edge["relationship_type"]: edge for edge in relationships}

    assert by_type["CALLS"]["source_node_id"] == "symbol:src/a.py:Service.run"
    assert by_type["CALLS"]["target_node_id"] == "symbol:src/b.py:Database.connect"
    assert by_type["IMPORTS"]["source_node_id"] == "symbol:src/b.py:Database.connect"
    assert by_type["IMPORTS"]["target_node_id"] == "symbol:src/c.py:Config.load"
    assert by_type["IMPORTS"]["traversal_path"] == [
        "symbol:src/a.py:Service.run",
        "symbol:src/b.py:Database.connect",
        "symbol:src/c.py:Config.load",
    ]
    for relationship in relationships:
        actual = graph.get_edge_data(
            relationship["source_node_id"], relationship["target_node_id"]
        )
        assert actual is not None
        assert relationship["relationship_type"] in {
            item["relationship_type"] for item in actual["relationships"]
        }



def test_planner_respects_step_limit_and_collects_multifile_evidence():
    first = _result()
    second = _result(
        "doc-two", name="Database.connect", path="src/db.py", source="return open_connection()"
    )
    query = "Database.connect in src/db.py"
    retriever = FakeRetriever([first], {query: [second]})
    service = InvestigationService(
        retriever,
        _analysis(_graph()),
        provider=MockLLMProvider(f"return verify(token) [{_evidence_id('doc-one')}]."),
    )
    response = service.investigate(
        InvestigationRequest(question="What does Service.run do?", max_steps=3)
    )

    assert len(retriever.calls) <= 3
    assert {item.source_file_path for item in response.evidence} == {"src/service.py", "src/db.py"}
    assert response.trace.steps[0].action == "hybrid_retrieval"
    assert any(step.action == "graph_exploration" for step in response.trace.steps)
    assert response.verified_findings[0].verification_method == "literal_source_match"
    assert response.investigated_claims[0].claim == response.verified_findings[0].claim


def test_planner_respects_one_step_limit():
    service = InvestigationService(FakeRetriever([_result()]), _analysis(_graph()))
    response = service.investigate(
        InvestigationRequest(question="Inspect run", max_steps=1)
    )
    assert len(response.trace.steps) == 1


def test_graph_explorer_failure_retains_initial_evidence_and_stops_followups():
    class RaisingGraphExplorer:
        def explore(self, *args, **kwargs):
            raise RuntimeError("synthetic graph failure")

    retriever = FakeRetriever([_result()])
    planner = InvestigationPlanner(graph_explorer=RaisingGraphExplorer())
    response = InvestigationService(
        retriever,
        _analysis(_graph()),
        planner=planner,
    ).investigate(InvestigationRequest(question="Trace run"))

    assert [item.document_id for item in response.evidence] == ["doc-one"]
    assert "Repository graph exploration failed; retrieval evidence was retained." in response.warnings
    assert len(retriever.calls) == 1


def test_planner_trace_edges_are_adjacent_and_keep_resolution_metadata():
    graph = nx.DiGraph()
    graph.add_node("symbol:src/service.py:Service.run", qualified_name="Service.run", file_path="src/service.py", symbol_type="method")
    graph.add_node("symbol:src/db.py:Database.connect", qualified_name="Database.connect", file_path="src/db.py", symbol_type="method")
    graph.add_node("symbol:src/config.py:Config.load", qualified_name="Config.load", file_path="src/config.py", symbol_type="function")
    graph.add_edge("symbol:src/service.py:Service.run", "symbol:src/db.py:Database.connect", relationships=[{"relationship_type": "CALLS", "resolution_status": "resolved", "source_lines": [6]}])
    graph.add_edge("symbol:src/db.py:Database.connect", "symbol:src/config.py:Config.load", relationships=[{"relationship_type": "IMPORTS", "resolution_status": "internal", "source_lines": [1]}])
    response = InvestigationService(
        FakeRetriever([_result()]), _analysis(graph)
    ).investigate(InvestigationRequest(question="Inspect run", max_graph_depth=2))
    graph_trace = next(step for step in response.trace.steps if step.action == "graph_exploration")

    imported = next(edge for edge in graph_trace.relationships_traversed if edge.relationship_type == "IMPORTS")
    assert imported.source_node_id == "symbol:src/db.py:Database.connect"
    assert imported.target_node_id == "symbol:src/config.py:Config.load"
    assert imported.resolution_status == "internal"
    assert imported.source_lines == [1]
    assert imported.traversal_path == [
        "symbol:src/service.py:Service.run",
        "symbol:src/db.py:Database.connect",
        "symbol:src/config.py:Config.load",
    ]
    for edge in graph_trace.relationships_traversed:
        assert graph.has_edge(edge.source_node_id, edge.target_node_id)
        graph_relationships = graph[edge.source_node_id][edge.target_node_id]["relationships"]
        assert edge.relationship_type in {item["relationship_type"] for item in graph_relationships}


def test_followup_retrieval_preserves_user_path_prefix_and_symbol_type():
    start = _result(name="Service.run", path="src/pkg/start.py")
    graph = nx.DiGraph()
    graph.add_node("symbol:src/pkg/start.py:Service.run", qualified_name="Service.run", file_path="src/pkg/start.py", symbol_type="method")
    graph.add_node("symbol:src/pkg/db.py:Database.connect", qualified_name="Database.connect", file_path="src/pkg/db.py", symbol_type="method")
    graph.add_edge("symbol:src/pkg/start.py:Service.run", "symbol:src/pkg/db.py:Database.connect", relationships=[{"relationship_type": "CALLS", "resolution_status": "resolved"}])
    query = "Database.connect in src/pkg/db.py"
    out_of_scope = _result("outside", name="External.open", path="src/other.py")
    retriever = FakeRetriever([start], {query: [out_of_scope]})
    request = InvestigationRequest(
        question="Trace run", filters={"path_prefix": "src/pkg", "symbol_type": "method"}
    )

    response = InvestigationService(retriever, _analysis(graph)).investigate(request)

    followup = retriever.calls[1]
    assert followup.filters.source_file_path == "src/pkg/db.py"
    assert followup.filters.path_prefix == "src/pkg"
    assert followup.filters.symbol_type == "method"
    assert followup.filters.qualified_name == "Database.connect"
    assert {item.document_id for item in response.evidence} == {"doc-one"}


def test_followup_retrieval_preserves_user_qualified_name():
    name = "Worker.run"
    start = _result(name=name, path="src/pkg/a.py")
    neighbor = _result("neighbor", name=name, path="src/pkg/b.py")
    graph = nx.DiGraph()
    graph.add_node("symbol:src/pkg/a.py:Worker.run", qualified_name=name, file_path="src/pkg/a.py", symbol_type="method")
    graph.add_node("symbol:src/pkg/b.py:Worker.run", qualified_name=name, file_path="src/pkg/b.py", symbol_type="method")
    graph.add_edge("symbol:src/pkg/a.py:Worker.run", "symbol:src/pkg/b.py:Worker.run", relationships=[{"relationship_type": "CALLS"}])
    query = "Worker.run in src/pkg/b.py"
    retriever = FakeRetriever([start], {query: [neighbor]})

    response = InvestigationService(retriever, _analysis(graph)).investigate(
        InvestigationRequest(
            question="Trace worker",
            filters={"path_prefix": "src/pkg", "qualified_name": name},
        )
    )

    followup_filters = retriever.calls[1].filters
    assert followup_filters.qualified_name == name
    assert followup_filters.path_prefix == "src/pkg"
    assert followup_filters.source_file_path == "src/pkg/b.py"
    assert len(response.evidence) == 2


def test_followup_retrieval_preserves_exact_path_and_skips_conflicting_neighbors():
    start = _result(name="Service.run", path="src/pkg/module.py")
    same_path_graph = nx.DiGraph()
    same_path_graph.add_node("symbol:src/pkg/module.py:Service.run", qualified_name="Service.run", file_path="src/pkg/module.py", symbol_type="method")
    same_path_graph.add_node("symbol:src/pkg/module.py:Service.save", qualified_name="Service.save", file_path="src/pkg/module.py", symbol_type="method")
    same_path_graph.add_edge("symbol:src/pkg/module.py:Service.run", "symbol:src/pkg/module.py:Service.save", relationships=[{"relationship_type": "CALLS", "resolution_status": "resolved"}])
    retriever = FakeRetriever([start])
    InvestigationService(retriever, _analysis(same_path_graph)).investigate(
        InvestigationRequest(question="Trace run", filters={"source_file_path": "src/pkg/module.py"})
    )
    assert retriever.calls[1].filters.source_file_path == "src/pkg/module.py"

    outside_graph = nx.DiGraph()
    outside_graph.add_node("symbol:src/pkg/start.py:Service.run", qualified_name="Service.run", file_path="src/pkg/start.py", symbol_type="method")
    outside_graph.add_node("symbol:src/outside.py:External.work", qualified_name="External.work", file_path="src/outside.py", symbol_type="method")
    outside_graph.add_edge("symbol:src/pkg/start.py:Service.run", "symbol:src/outside.py:External.work", relationships=[{"relationship_type": "CALLS"}])
    scoped_retriever = FakeRetriever([_result(path="src/pkg/start.py")])
    scoped_response = InvestigationService(scoped_retriever, _analysis(outside_graph)).investigate(
        InvestigationRequest(question="Trace run", filters={"path_prefix": "src/pkg"})
    )
    assert len(scoped_retriever.calls) == 1
    assert any("conflicts with user retrieval filters" in warning for warning in scoped_response.warnings)

    exact_path_retriever = FakeRetriever([_result(path="src/pkg/start.py")])
    exact_path_response = InvestigationService(
        exact_path_retriever, _analysis(outside_graph)
    ).investigate(
        InvestigationRequest(
            question="Trace run",
            filters={"source_file_path": "src/pkg/start.py"},
        )
    )
    assert len(exact_path_retriever.calls) == 1
    assert any("conflicts with user retrieval filters" in warning for warning in exact_path_response.warnings)


@pytest.mark.parametrize(
    ("filter_name", "filter_value", "neighbor_type", "neighbor_name"),
    [
        ("symbol_type", "function", "method", "Worker.run"),
        ("qualified_name", "Service.save", "method", "Worker.run"),
    ],
)
def test_conflicting_symbol_filters_skip_graph_neighbors(filter_name, filter_value, neighbor_type, neighbor_name):
    start = _result(name="Service.run", path="src/service.py")
    graph = nx.DiGraph()
    graph.add_node("symbol:src/service.py:Service.run", qualified_name="Service.run", file_path="src/service.py", symbol_type="method")
    graph.add_node(f"symbol:src/other.py:{neighbor_name}", qualified_name=neighbor_name, file_path="src/other.py", symbol_type=neighbor_type)
    graph.add_edge("symbol:src/service.py:Service.run", f"symbol:src/other.py:{neighbor_name}", relationships=[{"relationship_type": "CALLS"}])
    retriever = FakeRetriever([start])

    response = InvestigationService(retriever, _analysis(graph)).investigate(
        InvestigationRequest(question="Trace run", filters={filter_name: filter_value})
    )

    assert len(retriever.calls) == 1
    assert any(neighbor_name in warning for warning in response.warnings)


def test_claim_verifier_requires_direct_source_text_and_supports_partial():
    evidence = EvidenceBuilder().build([_result()]).items
    exact = ClaimVerifier().verify("return verify(token)", evidence)
    partial = ClaimVerifier().verify("return verify(token) and emit audit", evidence)
    unsupported = ClaimVerifier().verify("The method always authenticates users", evidence)

    assert exact.verification == "verified"
    assert exact.verification_method == "literal_source_match"
    assert "does not establish" in exact.explanation
    assert exact.evidence[0].source_file_path == "src/service.py"
    assert partial.verification == "partially_supported"
    assert unsupported.status == "unknown"


@pytest.mark.parametrize(
    "source",
    [
        "# return verify(token)",
        'message = "return verify(token)"',
    ],
)
def test_literal_matches_in_comments_and_strings_are_not_runtime_proof(source):
    evidence = EvidenceBuilder().build([_result(source=source)]).items
    finding = ClaimVerifier().verify("return verify(token)", evidence)

    assert finding.verification == "verified"
    assert finding.verification_method == "literal_source_match"
    assert "does not establish semantic entailment" in finding.explanation


def test_conflicting_literal_evidence_remains_unverified():
    evidence = EvidenceBuilder().build(
        [
            _result("positive", source="# cache enabled"),
            _result("negative", name="Cache.disable", source="# cache not enabled", rank=2),
        ]
    ).items

    finding = ClaimVerifier().verify("enabled", evidence)

    assert finding.status == "unknown"
    assert "ambiguous" in finding.explanation


def test_service_rejects_cited_but_unsupported_claim():
    provider = MockLLMProvider(
        f"The method always authenticates users [{_evidence_id('doc-one')}]."
    )
    response = InvestigationService(FakeRetriever([_result()]), _analysis(_graph()), provider=provider).investigate(
        InvestigationRequest(question="What does run guarantee?")
    )

    assert not response.verified_findings
    assert response.unverified_findings[0].status == "unknown"
    assert "unverified" in response.warnings[-1].lower()
    assert response.investigated_claims[0].claim == response.unverified_findings[0].claim
    assert response.investigated_claims[0].claim_id.startswith("H-")
    assert response.hypotheses == response.investigated_claims


def test_missing_evidence_and_retrieval_failures_are_safe():
    no_results_provider = MockLLMProvider("must not run")
    no_results = InvestigationService(
        FakeRetriever(), _analysis(nx.DiGraph()), provider=no_results_provider
    ).investigate(InvestigationRequest(question="What happens?"))
    failed_provider = MockLLMProvider("must not run")
    failed = InvestigationService(
        FakeRetriever(fail=True), _analysis(nx.DiGraph()), provider=failed_provider
    ).investigate(InvestigationRequest(question="What happens?"))

    assert no_results.unverified_findings
    assert no_results_provider.prompts == []
    assert "retrieval failed" in failed.warnings[0].lower()
    assert failed_provider.prompts == []


def test_provider_errors_are_reported_without_leaking_exception_text():
    class FailingProvider:
        def generate(self, prompt):
            raise RuntimeError("secret-token-123")

    response = InvestigationService(
        FakeRetriever([_result()]), _analysis(_graph()), provider=FailingProvider()
    ).investigate(InvestigationRequest(question="What does run do?"))

    assert "provider failed" in response.warnings[-1].lower()
    assert "secret-token-123" not in response.model_dump_json()
    assert response.trace.stop_reason == "Stopped because the provider failed."


def test_investigation_fails_safely_when_production_provider_is_unconfigured():
    provider = configured_provider(
        Settings(
            _env_file=None,
            LLM_PROVIDER="openai",
            OPENAI_API_KEY="",
            OPENAI_MODEL="",
        )
    )
    response = InvestigationService(
        FakeRetriever([_result()]), _analysis(_graph()), provider=provider
    ).investigate(InvestigationRequest(question="What does run do?"))

    assert response.trace.stop_reason == "Stopped because the provider failed."
    assert any("provider failed" in warning.lower() for warning in response.warnings)
    assert "API key" not in response.model_dump_json()


def test_metadata_only_evidence_does_not_call_provider():
    provider = MockLLMProvider("unusable")
    response = InvestigationService(
        FakeRetriever([_result(source=None)]), _analysis(nx.DiGraph()), provider=provider
    ).investigate(InvestigationRequest(question="What happens?"))

    assert response.evidence[0].source_content_missing is True
    assert provider.prompts == []


def test_test_recommendations_only_return_existing_matching_files(tmp_path):
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_service.py").write_text("def test_run(): Service.run()", encoding="utf-8")

    found = TestRecommender().recommend(tmp_path, {"run"})
    missing = TestRecommender().recommend(tmp_path, {"connect"})

    assert found[0].test_file_path == "tests/test_service.py"
    assert found[0].matched_symbols == ["run"]
    assert missing[0].test_file_path is None
    assert "no matching tests" in missing[0].reason.lower()
    assert "no matching tests" in TestRecommender().recommend(None, {"run"})[0].reason.lower()


def test_context_budget_counts_overhead_and_never_truncates_source():
    small = _result("small", source="return True", rank=1)
    huge = _result("huge", source="x = '" + ("large evidence " * 500) + "'", rank=2)
    evidence = EvidenceBuilder(max_context_chars=1_000_000).build([small, huge]).items
    prepared = InvestigationContextBuilder().build(
        "Inspect this",
        "step 1: retrieval found Service.run",
        evidence,
        token_budget=512,
        max_items=5,
        prompt_builder=PromptBuilder(),
    )
    rendered_characters = len(prepared.prompt.system_instructions) + len(prepared.prompt.user_message)

    assert [item.document_id for item in prepared.items] == ["small"]
    assert prepared.character_count == rendered_characters
    assert rendered_characters <= 512 * 4
    assert prepared.token_estimate == (rendered_characters + 3) // 4
    assert "step 1: retrieval found Service.run" in prepared.prompt.user_message
    assert "Question:\nInspect this" in prepared.prompt.user_message
    assert "<repository_evidence_json>" in prepared.prompt.user_message
    assert small.source_content in prepared.prompt.user_message
    assert huge.source_content not in prepared.prompt.user_message
    assert huge.source_content == next(item for item in evidence if item.document_id == "huge").source_content
    assert any("whole item" in warning for warning in prepared.warnings)


def test_oversized_question_does_not_call_provider():
    provider = MockLLMProvider("must not run")
    request = InvestigationRequest(question="Question " * 1_000, context_token_budget=256)
    response = InvestigationService(
        FakeRetriever([_result()]), _analysis(nx.DiGraph()), provider=provider
    ).investigate(request)

    assert provider.prompts == []
    assert any("exceed the context budget" in warning for warning in response.warnings)


def test_final_provider_prompt_is_fully_counted_and_includes_trace():
    provider = MockLLMProvider("return verify(token) [" + _evidence_id("doc-one") + "]")
    response = InvestigationService(
        FakeRetriever([_result()]), _analysis(nx.DiGraph()), provider=provider
    ).investigate(
        InvestigationRequest(question="What does run do?", context_token_budget=512)
    )

    prompt = provider.prompts[0]
    combined_characters = len(prompt.system_instructions) + len(prompt.user_message)
    assert "Investigation trace:" in prompt.user_message
    assert "hybrid_retrieval:" in prompt.user_message
    assert combined_characters <= response.context_budget_tokens * 4
    assert response.context_token_estimate == (combined_characters + 3) // 4


def test_empty_graph_is_safe_and_serialization_is_deterministic():
    explorer = GraphExplorer()
    graph = nx.DiGraph()
    assert explorer.explore(graph, ["missing"], max_depth=3) == []
    request = InvestigationRequest(question="Inspect repository")
    first = InvestigationService(FakeRetriever([_result()]), _analysis(graph)).investigate(request)
    second = InvestigationService(FakeRetriever([_result()]), _analysis(graph)).investigate(request)
    assert first.model_dump_json() == second.model_dump_json()
