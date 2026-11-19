"""Bounded deterministic exploration of the existing repository graph."""

from collections import deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import networkx as nx

from backend.app.graph.models import RepositoryAnalysisResult


@dataclass(frozen=True, eq=False)
class GraphExplorationResult(Sequence[dict[str, object]]):
    """Unique node discoveries and all unique relationships encountered.

    The sequence interface preserves compatibility with callers that previously
    iterated over the discovery list; ``relationships`` also includes edges to
    already visited nodes and edges between starting nodes.
    """

    discoveries: list[dict[str, object]]
    relationships: list[dict[str, object]]

    def __getitem__(self, index):
        return self.discoveries[index]

    def __len__(self) -> int:
        return len(self.discoveries)

    def __iter__(self) -> Iterator[dict[str, object]]:
        return iter(self.discoveries)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, GraphExplorationResult):
            return (
                self.discoveries == other.discoveries
                and self.relationships == other.relationships
            )
        if isinstance(other, Sequence):
            return self.discoveries == list(other)
        return NotImplemented


class GraphExplorer:
    """Traverse actual graph edges without inferring missing relationships."""

    def explore(
        self,
        graph: nx.DiGraph | RepositoryAnalysisResult,
        start_nodes: list[str],
        *,
        max_depth: int = 2,
        max_nodes: int = 50,
    ) -> GraphExplorationResult:
        """Return unique discoveries and edge records with their actual endpoints."""
        if max_depth < 0 or max_nodes < 1:
            raise ValueError("max_depth must be non-negative and max_nodes positive")
        if isinstance(graph, RepositoryAnalysisResult):
            graph = graph.to_networkx()

        valid_starts = sorted(set(start_nodes) & set(graph.nodes))[:max_nodes]
        queue = deque((node, 0, [node]) for node in valid_starts)
        scheduled = set(valid_starts)
        discoveries: list[dict[str, object]] = []
        relationships: list[dict[str, object]] = []
        recorded_edges: set[tuple[object, ...]] = set()

        while queue:
            node, depth, path = queue.popleft()
            if depth >= max_depth:
                continue

            neighbors: list[tuple[str, dict[str, object], str]] = []
            neighbors.extend((target, graph[node][target], "outgoing") for target in graph.successors(node))
            neighbors.extend((source, graph[source][node], "incoming") for source in graph.predecessors(node))
            for neighbor, edge_data, direction in sorted(neighbors, key=lambda row: (row[0], row[2])):
                edge_source, edge_target = (
                    (node, neighbor) if direction == "outgoing" else (neighbor, node)
                )
                edge_records = sorted(
                    edge_data.get("relationships", []),
                    key=lambda record: (
                        str(record.get("relationship_type", "")),
                        str(record.get("resolution_status", "")),
                        tuple(record.get("source_lines") or []),
                    ),
                )
                newly_recorded: list[dict[str, object]] = []
                for record in edge_records:
                    relationship = {
                        "source_node_id": edge_source,
                        "target_node_id": edge_target,
                        "relationship_type": record.get("relationship_type", "UNKNOWN"),
                        "resolution_status": record.get("resolution_status"),
                        "source_lines": sorted(record.get("source_lines") or []),
                        "direction": direction,
                        "traversal_path": [*path, neighbor],
                    }
                    edge_key = (
                        edge_source,
                        edge_target,
                        relationship["relationship_type"],
                        relationship["resolution_status"],
                        tuple(relationship["source_lines"]),
                    )
                    if edge_key not in recorded_edges:
                        recorded_edges.add(edge_key)
                        relationships.append(relationship)
                        newly_recorded.append(relationship)

                if neighbor not in scheduled and len(scheduled) < max_nodes:
                    scheduled.add(neighbor)
                    discovery = {
                        "node_id": neighbor,
                        "node": dict(graph.nodes[neighbor]),
                        "depth": depth + 1,
                        "direction": direction,
                        "path": [*path, neighbor],
                        "relationships": newly_recorded.copy(),
                    }
                    discoveries.append(discovery)
                    queue.append((neighbor, depth + 1, [*path, neighbor]))
                elif neighbor in scheduled and newly_recorded:
                    existing = next(
                        (item for item in discoveries if item["node_id"] == neighbor),
                        None,
                    )
                    if existing is not None:
                        existing["relationships"].extend(newly_recorded)

        relationships.sort(
            key=lambda edge: (
                len(edge["traversal_path"]),
                edge["traversal_path"],
                edge["source_node_id"],
                edge["target_node_id"],
                str(edge["relationship_type"]),
                str(edge["resolution_status"] or ""),
                edge["source_lines"],
            )
        )
        for discovery in discoveries:
            discovery["relationships"].sort(
                key=lambda edge: (
                    edge["source_node_id"],
                    edge["target_node_id"],
                    str(edge["relationship_type"]),
                )
            )
        return GraphExplorationResult(discoveries, relationships)
