"""Analyze potential change impact by following incoming repository dependencies."""

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from backend.app.graph.models import GraphNode, RepositoryAnalysisResult
from backend.app.graph.repository_graph import file_node_id, symbol_node_id
from backend.app.intelligence.impact_models import (
    AffectedComponent,
    AffectedFile,
    ChangeImpactReport,
    ImpactAnalysisRequest,
    ImpactPath,
    ImpactRelationship,
    InferredTestRecommendation,
)
from backend.app.investigation.recommendations import TestRecommender

_DEPENDENCY_TYPES = {"IMPORTS", "CALLS"}
_MAX_PATHS_PER_COMPONENT = 3


@dataclass(frozen=True)
class _Edge:
    source: str
    target: str
    relationship_type: str
    resolution_status: str | None
    source_lines: tuple[int, ...]
    uncertain_candidate: bool = False
    evidence_expression: str | None = None

    def key(self) -> tuple[object, ...]:
        return (
            self.source,
            self.target,
            self.relationship_type,
            self.resolution_status,
            self.source_lines,
            self.uncertain_candidate,
            self.evidence_expression,
        )

    def to_model(self) -> ImpactRelationship:
        return ImpactRelationship(
            source_node_id=self.source,
            target_node_id=self.target,
            relationship_type=self.relationship_type,
            resolution_status=self.resolution_status,
            source_lines=list(self.source_lines),
            uncertain_candidate=self.uncertain_candidate,
            evidence_expression=self.evidence_expression,
        )


class ImpactAnalysisError(ValueError):
    """Raised when the changed file or symbol is not present in the graph."""


class ImpactFileNotInGraph(ImpactAnalysisError):
    """Raised when an existing repository file has no graph node."""


class ImpactSymbolNotFound(ImpactAnalysisError):
    """Raised when the requested qualified symbol is absent from its file."""


class ChangeImpactAnalyzer:
    """Return bounded, deterministic incoming-dependency impact paths."""

    def __init__(self, test_recommender: TestRecommender | None = None) -> None:
        self.test_recommender = test_recommender or TestRecommender()

    def analyze(
        self,
        repository_id: str,
        repository_root: str | Path,
        analysis: RepositoryAnalysisResult,
        request: ImpactAnalysisRequest,
    ) -> ChangeImpactReport:
        root = Path(repository_root).resolve()
        target_path = PurePosixPath(request.changed_file_path).as_posix()
        nodes_by_id = {node.node_id: node for node in analysis.nodes}
        file_node = next(
            (
                node
                for node in analysis.nodes
                if node.node_type == "File" and node.file_path == target_path
            ),
            None,
        )
        if file_node is None:
            raise ImpactFileNotInGraph("Changed file is not represented in the repository graph")

        file_symbols = sorted(
            (
                node
                for node in analysis.nodes
                if node.file_path == target_path and node.node_type != "File"
            ),
            key=lambda node: (node.qualified_name or "", node.node_id),
        )
        if request.qualified_symbol_name:
            changed_component = next(
                (
                    node
                    for node in file_symbols
                    if node.qualified_name == request.qualified_symbol_name
                ),
                None,
            )
            if changed_component is None:
                raise ImpactSymbolNotFound("Changed symbol was not found in the specified file")
            target_nodes = [changed_component]
        else:
            changed_component = file_node
            target_nodes = [file_node, *file_symbols]

        targets = {node.node_id for node in target_nodes}
        incoming = self._incoming_edges(
            analysis,
            target_nodes,
            file_node,
            symbol_scope=request.qualified_symbol_name is not None,
        )
        best_depth: dict[str, int] = {node_id: 0 for node_id in targets}
        paths_by_node: dict[
            str,
            list[tuple[tuple[str, ...], tuple[_Edge, ...]]],
        ] = defaultdict(list)
        queue = deque(
            (node_id, (node_id,), ())
            for node_id in sorted(targets)
        )
        while queue:
            dependency_node_id, reverse_path, reverse_edges = queue.popleft()
            depth = len(reverse_edges)
            if depth >= request.max_depth:
                continue
            for edge in incoming.get(dependency_node_id, []):
                dependent = edge.source
                next_depth = depth + 1
                if dependent in reverse_path or dependent in targets:
                    continue
                if next_depth > request.max_depth:
                    continue
                if dependent not in nodes_by_id:
                    continue
                known_depth = best_depth.get(dependent)
                if known_depth is not None and next_depth > known_depth:
                    continue
                if known_depth is None or next_depth < known_depth:
                    best_depth[dependent] = next_depth
                    paths_by_node[dependent] = []
                next_reverse_path = (*reverse_path, dependent)
                next_reverse_edges = (*reverse_edges, edge)
                path_key = (next_reverse_path, tuple(item.key() for item in next_reverse_edges))
                if any(
                    (path_nodes, tuple(item.key() for item in path_edges)) == path_key
                    for path_nodes, path_edges in paths_by_node[dependent]
                ):
                    continue
                if len(paths_by_node[dependent]) >= _MAX_PATHS_PER_COMPONENT:
                    continue
                paths_by_node[dependent].append((next_reverse_path, next_reverse_edges))
                queue.append((dependent, next_reverse_path, next_reverse_edges))

        ordered_impacts = sorted(best_depth, key=lambda node_id: (best_depth[node_id], node_id))
        ordered_impacts = [node_id for node_id in ordered_impacts if best_depth[node_id] > 0]
        result_limit_reached = len(ordered_impacts) > request.max_results
        selected_impacts = ordered_impacts[: request.max_results]
        components = [
            self._component(nodes_by_id[node_id], best_depth[node_id], paths_by_node[node_id])
            for node_id in selected_impacts
        ]
        direct = [item for item in components if item.impact_type == "direct"]
        indirect = [item for item in components if item.impact_type == "indirect"]
        affected_files = self._affected_files(components, target_path)
        symbol_names = {
            node.qualified_name.rsplit(".", 1)[-1]
            for node in (item.node for item in components)
            if node.qualified_name
        }
        test_results = self.test_recommender.recommend(
            root,
            symbol_names,
            file_paths={item.file_path for item in affected_files},
        )
        discovered_tests = [item for item in test_results if item.test_file_path]
        inferred_tests = [
            InferredTestRecommendation(
                affected_file_path=item.file_path,
                explanation=(
                    "Inferred suggestion: review existing tests or consider adding tests "
                    "for this potentially affected file. This is not a claim that a test "
                    "currently covers the affected components."
                ),
            )
            for item in affected_files
        ]
        warnings = []
        if result_limit_reached:
            warnings.append("The impact result limit was reached; increase max_results to inspect more components.")
        if any(
            relationship.uncertain_candidate
            for component in components
            for path in component.paths
            for relationship in path.relationships
        ):
            warnings.append(
                "Some potential impacts rely on unresolved or ambiguous source references and need review."
            )
        if not components:
            warnings.append("No incoming dependency impacts were found within the requested depth.")

        return ChangeImpactReport(
            repository_id=repository_id,
            changed_component=changed_component,
            changed_file_path=target_path,
            directly_affected=direct,
            indirectly_affected=indirect,
            affected_files=affected_files,
            discovered_test_files=discovered_tests,
            inferred_test_recommendations=inferred_tests,
            max_depth=request.max_depth,
            max_results=request.max_results,
            result_limit_reached=result_limit_reached,
            warnings=warnings,
        )

    @staticmethod
    def _incoming_edges(
        analysis: RepositoryAnalysisResult,
        target_nodes: list[GraphNode],
        changed_file_node: GraphNode,
        *,
        symbol_scope: bool = False,
    ) -> dict[str, list[_Edge]]:
        incoming: dict[str, list[_Edge]] = defaultdict(list)
        graph = analysis.to_networkx()
        allowed_relationships = {"CALLS"} if symbol_scope else _DEPENDENCY_TYPES
        for source, target, data in graph.edges(data=True):
            for relationship in data.get("relationships", []):
                kind = str(relationship.get("relationship_type", ""))
                if kind not in allowed_relationships:
                    continue
                expected_resolution = "resolved" if kind == "CALLS" else "internal"
                if relationship.get("resolution_status") != expected_resolution:
                    # Unresolved and ambiguous records are added below from their
                    # source analysis records and marked as candidates.
                    continue
                edge = _Edge(
                    source=source,
                    target=target,
                    relationship_type=kind,
                    resolution_status=relationship.get("resolution_status"),
                    source_lines=tuple(sorted(relationship.get("source_lines") or [])),
                )
                incoming[target].append(edge)

        target_symbols = [node for node in target_nodes if node.node_type != "File"]
        target_file = changed_file_node.file_path or ""
        target_module = ChangeImpactAnalyzer._module_for_path(target_file)
        for dependency in analysis.dependencies if not symbol_scope else ():
            if str(dependency.resolution) != "unresolved":
                continue
            candidate_modules = {dependency.module_name or ""}
            if dependency.symbol_name:
                candidate_modules.add(
                    ".".join(filter(None, (dependency.module_name, dependency.symbol_name)))
                )
            if target_module not in candidate_modules:
                continue
            source = file_node_id(dependency.source_file)
            if source == changed_file_node.node_id:
                continue
            incoming[changed_file_node.node_id].append(
                _Edge(
                    source=source,
                    target=changed_file_node.node_id,
                    relationship_type="IMPORTS",
                    resolution_status="unresolved",
                    source_lines=(dependency.line_number,),
                    uncertain_candidate=True,
                    evidence_expression=dependency.module_name,
                )
            )

        target_names: dict[str, list] = defaultdict(list)
        for node in target_symbols:
            if node.qualified_name:
                target_names[node.qualified_name.rsplit(".", 1)[-1]].append(node)
        if target_names:
            graph_nodes = {node.node_id for node in analysis.nodes}
            for call in analysis.function_calls:
                status = str(call.resolution_status)
                if status not in {"unresolved", "ambiguous"}:
                    continue
                called_name = call.called_name.rsplit(".", 1)[-1]
                targets = target_names.get(called_name, [])
                if not targets:
                    continue
                source = (
                    file_node_id(call.source_file)
                    if call.caller_qualified_name == "<module>"
                    else symbol_node_id(call.source_file, call.caller_qualified_name)
                )
                if source not in graph_nodes:
                    continue
                for target in targets:
                    incoming[target.node_id].append(
                        _Edge(
                            source=source,
                            target=target.node_id,
                            relationship_type="CALLS",
                            resolution_status=status,
                            source_lines=(call.line_number,),
                            uncertain_candidate=True,
                            evidence_expression=call.call_expression,
                        )
                    )
        for edges in incoming.values():
            edges.sort(key=lambda edge: edge.key())
        return incoming

    @staticmethod
    def _module_for_path(path: str) -> str:
        pure = PurePosixPath(path)
        if pure.suffix != ".py":
            return ""
        parts = list(pure.with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts.pop()
        return ".".join(parts)

    @staticmethod
    def _component(
        node: GraphNode,
        depth: int,
        reverse_paths: list[tuple[tuple[str, ...], tuple[_Edge, ...]]],
    ) -> AffectedComponent:
        paths = []
        for reverse_nodes, reverse_edges in reverse_paths:
            paths.append(
                ImpactPath(
                    node_ids=list(reversed(reverse_nodes)),
                    relationships=[edge.to_model() for edge in reversed(reverse_edges)],
                )
            )
        uncertain = any(edge.uncertain_candidate for path in paths for edge in path.relationships)
        certainty = "low" if uncertain else "high"
        primary_relationships = paths[0].relationships if paths else []
        relationship_summary = " → ".join(
            relationship.relationship_type for relationship in primary_relationships
        )
        source_locations = [
            f"line {line}"
            for relationship in primary_relationships
            for line in relationship.source_lines
        ]
        location_summary = (
            f" at {', '.join(source_locations)}" if source_locations else ""
        )
        scope = "Directly" if depth == 1 else f"Transitively ({depth} edges)"
        if uncertain:
            explanation = (
                f"{scope} potentially affected through an unresolved or ambiguous "
                f"candidate relationship ({relationship_summary}){location_summary}; "
                "the target usage is not established and needs manual review."
            )
        else:
            explanation = (
                f"{scope} potentially affected through recorded {relationship_summary} "
                f"relationship(s){location_summary}. This static dependency does not "
                "establish that the component is broken."
            )
        return AffectedComponent(
            node=node,
            depth=depth,
            impact_type="direct" if depth == 1 else "indirect",
            certainty=certainty,
            paths=paths,
            explanation=explanation,
        )

    @staticmethod
    def _affected_files(
        components: list[AffectedComponent],
        changed_file_path: str,
    ) -> list[AffectedFile]:
        grouped: dict[str, list[AffectedComponent]] = defaultdict(list)
        for component in components:
            path = component.node.file_path
            if path and path != changed_file_path:
                grouped[path].append(component)
        results = []
        for path, items in sorted(grouped.items()):
            depth = min(item.depth for item in items)
            certainty = "high" if any(item.certainty == "high" for item in items) else "low"
            results.append(
                AffectedFile(
                    file_path=path,
                    minimum_depth=depth,
                    impact_type="direct" if depth == 1 else "indirect",
                    certainty=certainty,
                    affected_node_ids=sorted(item.node.node_id for item in items),
                )
            )
        return results
