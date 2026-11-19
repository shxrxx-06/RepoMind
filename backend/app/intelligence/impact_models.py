"""Request and result models for uncertainty-aware change impact reports."""

from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from backend.app.graph.models import GraphNode
from backend.app.investigation.models import TestRecommendation


def _safe_repository_path(value: str) -> bool:
    normalized = value.replace("\\", "/")
    parts = normalized.split("/")
    return bool(
        normalized
        and not normalized.startswith("/")
        and not (len(normalized) >= 2 and normalized[1] == ":")
        and all(part not in {"", ".", ".."} for part in parts)
        and PurePosixPath(normalized).as_posix() == normalized
    )


class ImpactAnalysisRequest(BaseModel):
    changed_file_path: str = Field(min_length=1, max_length=1_000)
    qualified_symbol_name: str | None = Field(default=None, max_length=500)
    max_depth: int = Field(default=3, ge=0, le=10)
    max_results: int = Field(default=200, ge=1, le=1_000)

    @field_validator("changed_file_path")
    @classmethod
    def normalize_changed_path(cls, value: str) -> str:
        normalized = value.strip().replace("\\", "/")
        parts = normalized.split("/")
        if (
            not normalized
            or normalized.startswith("/")
            or any(part in {"", ".", ".."} for part in parts)
            or (len(normalized) >= 2 and normalized[1] == ":")
        ):
            raise ValueError("changed_file_path must be a repository-relative file path")
        return "/".join(parts)

    @field_validator("qualified_symbol_name")
    @classmethod
    def validate_symbol_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        parts = normalized.split(".")
        if not normalized or any(not part.isidentifier() for part in parts):
            raise ValueError("qualified_symbol_name must be a qualified Python identifier")
        return normalized


class ImpactRelationship(BaseModel):
    source_node_id: str
    target_node_id: str
    relationship_type: Literal["IMPORTS", "CALLS"]
    resolution_status: Literal["internal", "resolved", "unresolved", "ambiguous"]
    source_lines: list[int] = Field(default_factory=list)
    uncertain_candidate: bool = False
    evidence_expression: str | None = None

    @field_validator("source_lines")
    @classmethod
    def validate_source_lines(cls, value: list[int]) -> list[int]:
        if any(line < 1 for line in value) or value != sorted(set(value)):
            raise ValueError("source_lines must be unique, sorted, positive line numbers")
        return value

    @model_validator(mode="after")
    def validate_resolution(self) -> "ImpactRelationship":
        uncertain_statuses = {"unresolved", "ambiguous"}
        if self.uncertain_candidate and self.resolution_status not in uncertain_statuses:
            raise ValueError("uncertain candidates must retain unresolved or ambiguous status")
        if not self.uncertain_candidate and self.resolution_status not in {"internal", "resolved"}:
            raise ValueError("confirmed relationships must have resolved status")
        return self


class ImpactPath(BaseModel):
    node_ids: list[str] = Field(min_length=2)
    relationships: list[ImpactRelationship] = Field(min_length=1)


class AffectedComponent(BaseModel):
    node: GraphNode
    depth: int = Field(ge=1)
    impact_type: Literal["direct", "indirect"]
    certainty: Literal["high", "medium", "low"]
    paths: list[ImpactPath] = Field(min_length=1)
    explanation: str


class AffectedFile(BaseModel):
    file_path: str
    minimum_depth: int = Field(ge=1)
    impact_type: Literal["direct", "indirect"]
    certainty: Literal["high", "medium", "low"]
    affected_node_ids: list[str] = Field(default_factory=list)


class InferredTestRecommendation(BaseModel):
    recommendation_type: Literal["inferred"] = "inferred"
    affected_file_path: str
    explanation: str


class ChangeImpactReport(BaseModel):
    repository_id: str
    changed_component: GraphNode
    changed_targets: list[GraphNode] = Field(default_factory=list)
    changed_file_path: str
    directly_affected: list[AffectedComponent] = Field(default_factory=list)
    indirectly_affected: list[AffectedComponent] = Field(default_factory=list)
    affected_files: list[AffectedFile] = Field(default_factory=list)
    discovered_test_files: list[TestRecommendation] = Field(default_factory=list)
    inferred_test_recommendations: list[InferredTestRecommendation] = Field(
        default_factory=list
    )
    max_depth: int = Field(ge=0, le=10)
    max_results: int = Field(ge=1, le=1_000)
    result_limit_reached: bool = False
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_references(self) -> "ChangeImpactReport":
        if not _safe_repository_path(self.changed_file_path):
            raise ValueError("changed_file_path must be repository-relative")
        if self.changed_component.file_path != self.changed_file_path:
            raise ValueError("changed component path must match changed_file_path")

        components = [*self.directly_affected, *self.indirectly_affected]
        if len(components) > self.max_results:
            raise ValueError("affected component count cannot exceed max_results")
        changed_targets = self.changed_targets or [self.changed_component]
        changed_nodes_by_id = {node.node_id: node for node in changed_targets}
        target_ids = set(changed_nodes_by_id)
        if len(target_ids) != len(changed_targets) or self.changed_component.node_id not in target_ids:
            raise ValueError("changed_targets must be unique and include changed_component")
        component_nodes_by_id = {item.node.node_id: item.node for item in components}
        if (
            len(component_nodes_by_id) != len(components)
            or target_ids.intersection(component_nodes_by_id)
        ):
            raise ValueError("changed target and affected component node IDs must be unique")
        known_nodes = {**changed_nodes_by_id, **component_nodes_by_id}
        for node in known_nodes.values():
            if not node.file_path or not _safe_repository_path(node.file_path):
                raise ValueError("graph node has an invalid repository-relative path")
            if node.file_path != self.changed_file_path and node in changed_targets:
                raise ValueError("changed target nodes must belong to changed_file_path")
            expected_id = (
                f"file:{node.file_path}"
                if node.node_type == "File"
                else f"symbol:{node.file_path}:{node.qualified_name}"
            )
            if node.node_id != expected_id:
                raise ValueError("graph node ID does not match its type and repository path")
        for component in components:
            node = component.node
            if component.depth > self.max_depth:
                raise ValueError("affected component depth cannot exceed max_depth")
            expected_impact = "direct" if component.depth == 1 else "indirect"
            if component.impact_type != expected_impact:
                raise ValueError("impact type must agree with component depth")
            if not component.explanation.strip():
                raise ValueError("affected components must include an explanation")
            for path in component.paths:
                if len(path.node_ids) != component.depth + 1:
                    raise ValueError("dependency path length must agree with impact depth")
                if path.node_ids[0] != node.node_id:
                    raise ValueError("dependency path must start at its affected component")
                if path.node_ids[-1] not in target_ids:
                    raise ValueError("dependency path must end at a changed target")
                if len(path.relationships) != len(path.node_ids) - 1:
                    raise ValueError("dependency path must have one relationship per adjacent node pair")
                if any(node_id not in known_nodes for node_id in path.node_ids):
                    raise ValueError("dependency path references a node absent from the report")
                for index, relationship in enumerate(path.relationships):
                    if (
                        relationship.source_node_id != path.node_ids[index]
                        or relationship.target_node_id != path.node_ids[index + 1]
                    ):
                        raise ValueError("relationship references do not match adjacent path nodes")

        components_by_id = {component.node.node_id: component for component in components}
        for path in self.affected_files:
            if not _safe_repository_path(path.file_path):
                raise ValueError("affected file path must be repository-relative")
            if path.file_path == self.changed_file_path:
                raise ValueError("changed file cannot be listed as an affected file")
            if not path.affected_node_ids or any(
                node_id not in known_nodes
                or known_nodes[node_id].file_path != path.file_path
                for node_id in path.affected_node_ids
            ):
                raise ValueError("affected file references must identify reported components in that file")
            if len(set(path.affected_node_ids)) != len(path.affected_node_ids):
                raise ValueError("affected file node references must be unique")
            referenced = [components_by_id[node_id] for node_id in path.affected_node_ids]
            expected_depth = min(component.depth for component in referenced)
            if path.minimum_depth != expected_depth:
                raise ValueError("affected file minimum_depth must match its referenced components")
            expected_impact = "direct" if expected_depth == 1 else "indirect"
            if path.impact_type != expected_impact:
                raise ValueError("affected file impact_type must agree with minimum_depth")
            certainty_rank = {"low": 0, "medium": 1, "high": 2}
            expected_certainty = max(
                (component.certainty for component in referenced),
                key=certainty_rank.__getitem__,
            )
            if path.certainty != expected_certainty:
                raise ValueError("affected file certainty must match its referenced components")

        for test in self.discovered_test_files:
            if not test.test_file_path or not _safe_repository_path(test.test_file_path):
                raise ValueError("discovered test references must use safe repository-relative paths")
        for suggestion in self.inferred_test_recommendations:
            if not _safe_repository_path(suggestion.affected_file_path):
                raise ValueError("inferred test paths must be repository-relative")
            if suggestion.affected_file_path not in {item.file_path for item in self.affected_files}:
                raise ValueError("inferred test suggestions must reference an affected file")
        return self
