"""Build deterministic NetworkX graphs from repository analysis records."""

import networkx as nx

from backend.app.graph.models import (
    CallResolution,
    DependencyInfo,
    DependencyResolution,
    FunctionCallInfo,
    GraphNode,
    GraphNodeType,
    GraphRelationship,
    RelationshipType,
    RepositoryAnalysisResult,
)
from backend.app.intelligence.dependency_analyzer import normalize_repo_path
from backend.app.parsing.models import ParsedFile, ParseError, SymbolType


def file_node_id(path: str) -> str:
    return f"file:{normalize_repo_path(path)}"


def symbol_node_id(path: str, qualified_name: str) -> str:
    return f"symbol:{normalize_repo_path(path)}:{qualified_name}"


class RepositoryGraphBuilder:
    """Create File/Class/Function/Method nodes and typed directed relationships."""

    def build(
        self,
        parsed_files: list[ParsedFile],
        dependencies: list[DependencyInfo],
        function_calls: list[FunctionCallInfo],
    ) -> nx.DiGraph:
        graph = nx.DiGraph()
        symbols_by_file: dict[str, dict[str, object]] = {}
        for parsed in sorted(parsed_files, key=lambda item: normalize_repo_path(item.file_path)):
            path = normalize_repo_path(parsed.file_path)
            file_id = file_node_id(path)
            graph.add_node(
                file_id,
                node_type=GraphNodeType.FILE.value,
                qualified_name=None,
                file_path=path,
                symbol_type=None,
                start_line=None,
                end_line=None,
            )
            symbol_nodes: dict[str, object] = {}
            for symbol in parsed.symbols:
                node_id = symbol_node_id(path, symbol.qualified_name)
                node_type = {
                    SymbolType.CLASS: GraphNodeType.CLASS,
                    SymbolType.FUNCTION: GraphNodeType.FUNCTION,
                    SymbolType.METHOD: GraphNodeType.METHOD,
                }[SymbolType(symbol.symbol_type)]
                graph.add_node(
                    node_id,
                    node_type=node_type.value,
                    qualified_name=symbol.qualified_name,
                    file_path=path,
                    symbol_type=str(symbol.symbol_type),
                    start_line=symbol.start_line,
                    end_line=symbol.end_line,
                )
                symbol_nodes[symbol.qualified_name] = node_id
            symbols_by_file[path] = symbol_nodes

            for qualified_name, node_id in symbol_nodes.items():
                parent_name = qualified_name.rpartition(".")[0]
                parent_id = symbol_nodes.get(parent_name) if parent_name else None
                if parent_id:
                    self._add_edge(
                        graph,
                        parent_id,
                        node_id,
                        RelationshipType.CONTAINS,
                        None,
                        "resolved",
                    )
                else:
                    self._add_edge(
                        graph,
                        file_id,
                        node_id,
                        RelationshipType.DEFINES,
                        None,
                        "resolved",
                    )

        for dependency in dependencies:
            if (
                dependency.resolution != DependencyResolution.INTERNAL
                or not dependency.imported_file
            ):
                continue
            source = file_node_id(dependency.source_file)
            target = file_node_id(dependency.imported_file)
            if source in graph and target in graph:
                self._add_edge(
                    graph,
                    source,
                    target,
                    RelationshipType.IMPORTS,
                    dependency.line_number,
                    DependencyResolution.INTERNAL.value,
                )

        for call in function_calls:
            if (
                call.resolution_status != CallResolution.RESOLVED
                or not call.target_file
                or not call.target_qualified_name
            ):
                continue
            source = (
                file_node_id(call.source_file)
                if call.caller_qualified_name == "<module>"
                else symbol_node_id(call.source_file, call.caller_qualified_name)
            )
            target = symbol_node_id(call.target_file, call.target_qualified_name)
            if source in graph and target in graph:
                self._add_edge(
                    graph,
                    source,
                    target,
                    RelationshipType.CALLS,
                    call.line_number,
                    CallResolution.RESOLVED.value,
                )
        return graph

    @staticmethod
    def _add_edge(
        graph: nx.DiGraph,
        source: str,
        target: str,
        relationship_type: RelationshipType,
        line_number: int | None,
        resolution_status: str,
    ) -> None:
        if not graph.has_edge(source, target):
            graph.add_edge(source, target, relationships=[])
        edge = graph[source][target]
        record = next(
            (
                item
                for item in edge["relationships"]
                if item["relationship_type"] == relationship_type.value
                and item["resolution_status"] == resolution_status
            ),
            None,
        )
        if record is None:
            record = {
                "relationship_type": relationship_type.value,
                "source_lines": [],
                "resolution_status": resolution_status,
            }
            edge["relationships"].append(record)
        if line_number is not None and line_number not in record["source_lines"]:
            record["source_lines"].append(line_number)
            record["source_lines"].sort()

    def to_result(
        self,
        graph: nx.DiGraph,
        dependencies: list[DependencyInfo],
        function_calls: list[FunctionCallInfo],
        errors: list[ParseError],
    ) -> RepositoryAnalysisResult:
        """Convert a NetworkX graph plus analysis records to Pydantic output."""
        nodes = [
            GraphNode(node_id=node_id, **attributes)
            for node_id, attributes in sorted(graph.nodes(data=True))
        ]
        relationships = []
        for source, target, attributes in sorted(
            graph.edges(data=True), key=lambda edge: (edge[0], edge[1])
        ):
            for relationship in attributes.get("relationships", []):
                relationships.append(
                    GraphRelationship(
                        source=source,
                        target=target,
                        relationship_type=relationship["relationship_type"],
                        source_lines=relationship["source_lines"],
                        resolution_status=relationship["resolution_status"],
                    )
                )
        relationships.sort(
            key=lambda item: (item.source, item.target, item.relationship_type)
        )
        return RepositoryAnalysisResult(
            dependencies=dependencies,
            function_calls=function_calls,
            nodes=nodes,
            relationships=relationships,
            errors=errors,
        )
