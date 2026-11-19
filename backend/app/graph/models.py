"""Structured, JSON-serializable repository graph and analysis models."""

from enum import StrEnum
from typing import Any

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field

from backend.app.parsing.models import ParseError


class DependencyResolution(StrEnum):
    INTERNAL = "internal"
    EXTERNAL = "external"
    UNRESOLVED = "unresolved"


class CallResolution(StrEnum):
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    AMBIGUOUS = "ambiguous"


class GraphNodeType(StrEnum):
    FILE = "File"
    CLASS = "Class"
    FUNCTION = "Function"
    METHOD = "Method"


class RelationshipType(StrEnum):
    IMPORTS = "IMPORTS"
    DEFINES = "DEFINES"
    CONTAINS = "CONTAINS"
    CALLS = "CALLS"


class DependencyInfo(BaseModel):
    """One import binding and its repository-level resolution, if any."""

    source_file: str
    imported_file: str | None = None
    module_name: str | None = None
    symbol_name: str | None = None
    alias: str | None = None
    line_number: int
    resolution: DependencyResolution

    model_config = ConfigDict(use_enum_values=True)


class FunctionCallInfo(BaseModel):
    """One statically observed call expression and conservative resolution."""

    caller_qualified_name: str
    called_name: str
    source_file: str
    line_number: int
    call_expression: str
    resolution_status: CallResolution
    target_qualified_name: str | None = None
    target_file: str | None = None

    model_config = ConfigDict(use_enum_values=True)


class GraphNode(BaseModel):
    node_id: str
    node_type: GraphNodeType
    qualified_name: str | None = None
    file_path: str | None = None
    symbol_type: str | None = None
    start_line: int | None = None
    end_line: int | None = None

    model_config = ConfigDict(use_enum_values=True)


class GraphRelationship(BaseModel):
    source: str
    target: str
    relationship_type: RelationshipType
    source_lines: list[int] = Field(default_factory=list)
    resolution_status: str | None = None

    model_config = ConfigDict(use_enum_values=True)


class RepositoryAnalysisResult(BaseModel):
    """Serializable analysis plus a helper to reconstruct its NetworkX graph."""

    dependencies: list[DependencyInfo] = Field(default_factory=list)
    function_calls: list[FunctionCallInfo] = Field(default_factory=list)
    nodes: list[GraphNode] = Field(default_factory=list)
    relationships: list[GraphRelationship] = Field(default_factory=list)
    errors: list[ParseError] = Field(default_factory=list)

    model_config = ConfigDict(use_enum_values=True)

    def to_networkx(self) -> nx.DiGraph:
        """Reconstruct the directed graph represented by this result."""
        graph = nx.DiGraph()
        for node in self.nodes:
            attributes: dict[str, Any] = node.model_dump(exclude={"node_id"})
            graph.add_node(node.node_id, **attributes)
        for relationship in self.relationships:
            edge = graph.get_edge_data(relationship.source, relationship.target)
            if edge is None:
                edge = {"relationships": []}
                graph.add_edge(relationship.source, relationship.target, **edge)
                edge = graph[relationship.source][relationship.target]
            edge["relationships"].append(
                relationship.model_dump(
                    exclude={"source", "target"}, mode="json"
                )
            )
        return graph
