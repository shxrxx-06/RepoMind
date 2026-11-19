"""Deterministic bounded retrieval and graph-guided investigation planning."""

import hashlib
from dataclasses import dataclass, field

from backend.app.graph.models import RepositoryAnalysisResult
from backend.app.graph.repository_graph import file_node_id, symbol_node_id
from backend.app.investigation.graph_explorer import GraphExplorationResult, GraphExplorer
from backend.app.investigation.models import (
    InvestigationRequest,
    InvestigationStep,
    TraversedRelationship,
)
from backend.app.retrieval.models import RetrievalFilters, RetrievalQuery, RetrievalResult


@dataclass
class PlanOutput:
    results: list[RetrievalResult] = field(default_factory=list)
    steps: list[InvestigationStep] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stop_reason: str = "Configured investigation step limit reached."


class InvestigationPlanner:
    """Retrieve initial symbols, traverse actual graph edges, and query neighbors."""

    def __init__(self, graph_explorer: GraphExplorer | None = None) -> None:
        self.graph_explorer = graph_explorer or GraphExplorer()

    def plan(self, request: InvestigationRequest, retriever, graph) -> PlanOutput:
        output = PlanOutput()
        try:
            initial = retriever.search(
                RetrievalQuery(
                    query=request.question,
                    top_k=min(request.top_k, request.max_evidence_items),
                    filters=request.filters,
                )
            )
        except Exception:  # noqa: BLE001 - retrieval implementations expose varied errors.
            output.warnings.append("Initial repository retrieval failed.")
            output.stop_reason = "Stopped because initial retrieval failed."
            return output
        output.warnings.extend(initial.warnings)
        output.results.extend(initial.results)
        output.steps.append(
            InvestigationStep(
                step_number=1,
                action="hybrid_retrieval",
                symbols_examined=[item.qualified_name for item in initial.results],
                files_examined=sorted({item.source_file_path for item in initial.results}),
                evidence_ids_collected=self._evidence_ids(initial.results),
                result=f"Retrieved {len(initial.results)} starting code symbols.",
                continuation_reason=(
                    "Explore graph relationships to find connected callers, dependencies, and definitions."
                    if initial.results and request.max_graph_depth > 0
                    else "No graph exploration is needed or permitted by the configured depth."
                ),
            )
        )
        if not initial.results or request.max_steps <= 1 or request.max_graph_depth == 0:
            output.stop_reason = (
                "No retrieval results were available."
                if not initial.results
                else "Stopped at the configured investigation step limit."
            )
            return output

        graph_obj = graph.to_networkx() if isinstance(graph, RepositoryAnalysisResult) else graph
        starts = []
        for result in initial.results:
            path = result.source_file_path.replace("\\", "/").strip("/")
            sid = symbol_node_id(path, result.qualified_name)
            fid = file_node_id(path)
            if sid in graph_obj:
                starts.append(sid)
            elif fid in graph_obj:
                starts.append(fid)
        try:
            explored = self.graph_explorer.explore(
                graph_obj,
                starts,
                max_depth=request.max_graph_depth,
                max_nodes=max(request.max_evidence_items * 2, 1),
            )
        except Exception:  # noqa: BLE001 - incomplete graphs should not break investigation.
            explored = GraphExplorationResult(discoveries=[], relationships=[])
            output.warnings.append("Repository graph exploration failed; retrieval evidence was retained.")
        neighbor_symbols = []
        skipped_neighbors: list[str] = []
        for item in explored:
            node = item["node"]
            name = node.get("qualified_name")
            path = node.get("file_path")
            if name and path:
                name = str(name)
                path = str(path)
                if self._matches_user_filters(name, path, node, request.filters):
                    neighbor_symbols.append((name, path))
                else:
                    skipped_neighbors.append(f"{name} in {path}")
                    output.warnings.append(
                        f"Skipped graph neighbor {name} in {path} because it conflicts with user retrieval filters."
                    )
        traversed_relationships: list[TraversedRelationship] = [
            TraversedRelationship.model_validate(relationship)
            for relationship in explored.relationships
        ]
        traversed_relationships.sort(
            key=lambda edge: (
                edge.traversal_path,
                edge.source_node_id,
                edge.target_node_id,
                edge.relationship_type,
                edge.resolution_status or "",
            )
        )
        output.steps.append(
            InvestigationStep(
                step_number=2,
                action="graph_exploration",
                symbols_examined=sorted({name for name, _ in neighbor_symbols}),
                files_examined=sorted({path for _, path in neighbor_symbols}),
                relationships_traversed=traversed_relationships,
                result=f"Explored {len(explored)} connected graph nodes.",
                continuation_reason=(
                    "Retrieve source snippets for connected symbols." if neighbor_symbols
                    else (
                        "Graph neighbors were skipped because they conflict with user filters."
                        if skipped_neighbors
                        else "No connected graph symbols were available to retrieve."
                    )
                ),
            )
        )
        seen_queries = {request.question.casefold()}
        for name, path in sorted(set(neighbor_symbols)):
            if len(output.steps) >= request.max_steps:
                output.stop_reason = "Stopped at the configured investigation step limit."
                break
            if len(output.results) >= request.max_evidence_items:
                output.stop_reason = "Stopped at the configured evidence item limit."
                break
            query = f"{name} in {path}"
            if query.casefold() in seen_queries:
                continue
            seen_queries.add(query.casefold())
            try:
                response = retriever.search(
                    RetrievalQuery(
                        query=query,
                        top_k=min(request.top_k, request.max_evidence_items - len(output.results)),
                        filters=self._followup_filters(name, path, request),
                    )
                )
            except Exception:  # noqa: BLE001 - a failed follow-up does not erase prior evidence.
                output.warnings.append(f"Follow-up retrieval failed for {name}.")
                continue
            output.warnings.extend(response.warnings)
            new_results = [
                result for result in response.results
                if result.document_id not in {existing.document_id for existing in output.results}
                and self._result_matches_followup(result, name, path, request)
            ][: request.max_evidence_items - len(output.results)]
            output.results.extend(new_results)
            output.steps.append(
                InvestigationStep(
                    step_number=len(output.steps) + 1,
                    action="graph_guided_retrieval",
                    symbols_examined=[name],
                    files_examined=[path],
                    relationships_traversed=[],
                    evidence_ids_collected=self._evidence_ids(new_results),
                    result=f"Follow-up retrieval added {len(new_results)} source documents.",
                    continuation_reason=(
                        "Continue while step and evidence limits allow."
                        if new_results else "No additional evidence was found for this graph neighbor."
                    ),
                )
            )
        output.results = output.results[: request.max_evidence_items]
        if len(output.steps) < request.max_steps and output.stop_reason == "Configured investigation step limit reached.":
            output.stop_reason = "Stopped after bounded graph-guided retrieval was exhausted."
        return output

    @classmethod
    def _matches_user_filters(cls, name, path, node, filters) -> bool:
        if filters is None:
            return True
        normalized_path = cls._normalize_path(path)
        if filters.source_file_path and normalized_path != cls._normalize_path(filters.source_file_path):
            return False
        if filters.path_prefix:
            prefix = cls._normalize_path(filters.path_prefix).strip("/")
            if normalized_path != prefix and not normalized_path.startswith(f"{prefix}/"):
                return False
        if filters.symbol_type:
            actual_type = str(node.get("symbol_type") or node.get("node_type") or "").casefold()
            if actual_type != filters.symbol_type.casefold():
                return False
        if filters.qualified_name and name.casefold() != filters.qualified_name.casefold():
            return False
        return True

    @classmethod
    def _followup_filters(cls, name, path, request):
        filters = request.filters
        return RetrievalFilters(
            source_file_path=filters.source_file_path if filters and filters.source_file_path else path,
            path_prefix=filters.path_prefix if filters else None,
            symbol_type=filters.symbol_type if filters else None,
            qualified_name=filters.qualified_name if filters and filters.qualified_name else name,
        )

    @classmethod
    def _result_matches_followup(cls, result, name, path, request) -> bool:
        if cls._normalize_path(result.source_file_path) != cls._normalize_path(path):
            return False
        if result.qualified_name.casefold() != name.casefold():
            return False
        return cls._matches_user_filters(
            result.qualified_name,
            result.source_file_path,
            {"symbol_type": result.symbol_type},
            request.filters,
        )

    @staticmethod
    def _normalize_path(path: str) -> str:
        return path.replace("\\", "/").strip("/").casefold()

    @staticmethod
    def _evidence_ids(results: list[RetrievalResult]) -> list[str]:
        return [
            "E-" + hashlib.sha256(item.document_id.encode("utf-8")).hexdigest()[:24]
            for item in results
        ]
