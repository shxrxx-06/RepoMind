"""Structured inputs and evidence-grounded outputs for test intelligence."""

from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from backend.app.graph.models import GraphNode
from backend.app.intelligence.impact_models import (
    ChangeImpactReport,
    ImpactAnalysisRequest,
)
from backend.app.parsing.models import ImportInfo, ParseError

EvidenceClassification = Literal[
    "direct_resolved_call",
    "import_only_reference",
    "uncertain_candidate",
    "no_discoverable_test_reference",
]


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


def _valid_graph_node(node: GraphNode) -> bool:
    if not node.file_path or not _safe_repository_path(node.file_path):
        return False
    expected_id = (
        f"file:{node.file_path}"
        if node.node_type == "File"
        else f"symbol:{node.file_path}:{node.qualified_name}"
    )
    return (
        node.node_id == expected_id
        and (node.start_line is None or node.start_line >= 1)
        and (node.end_line is None or node.end_line >= 1)
        and (
            node.start_line is None
            or node.end_line is None
            or node.end_line >= node.start_line
        )
    )


class TestIntelligenceRequest(BaseModel):
    """Analyze repository test references, optionally scoped to a change impact."""

    changed_file_path: str | None = Field(default=None, min_length=1, max_length=1_000)
    qualified_symbol_name: str | None = Field(default=None, max_length=500)
    max_depth: int = Field(default=3, ge=0, le=10)
    max_results: int = Field(default=200, ge=1, le=1_000)

    @field_validator("changed_file_path")
    @classmethod
    def validate_repository_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return ImpactAnalysisRequest(changed_file_path=value).changed_file_path

    @field_validator("qualified_symbol_name")
    @classmethod
    def validate_qualified_symbol(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return ImpactAnalysisRequest(
            changed_file_path="placeholder.py",
            qualified_symbol_name=value,
        ).qualified_symbol_name

    @model_validator(mode="after")
    def validate_scope(self) -> "TestIntelligenceRequest":
        if self.qualified_symbol_name and not self.changed_file_path:
            raise ValueError("changed_file_path is required when qualified_symbol_name is set")
        return self


class TestCaseInfo(BaseModel):
    """A test class, function, or test method located by Python AST parsing."""

    name: str
    qualified_name: str
    kind: Literal["class", "function", "method"]
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_line_range(self) -> "TestCaseInfo":
        if self.end_line < self.start_line:
            raise ValueError("test end_line cannot precede start_line")
        return self


class DiscoveredTestFile(BaseModel):
    """A convention-matched Python test file, including safe parse output."""

    file_path: str
    tests: list[TestCaseInfo] = Field(default_factory=list)
    imports: list[ImportInfo] = Field(default_factory=list)
    parse_errors: list[ParseError] = Field(default_factory=list)

    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, value: str) -> str:
        if not _safe_repository_path(value):
            raise ValueError("test file path must be repository-relative")
        return value


class TestMappingEvidence(BaseModel):
    """One source-level fact supporting a test-to-code reference mapping."""

    evidence_type: Literal[
        "resolved_call",
        "resolved_call_outside_test",
        "imported_symbol",
        "imported_module",
        "unresolved_call_candidate",
    ]
    description: str
    line_number: int = Field(ge=1)
    expression: str | None = None


class TestCodeMapping(BaseModel):
    """A static test-to-code reference with explicit certainty and evidence."""

    test_file_path: str
    test_qualified_name: str
    source_component: GraphNode
    evidence_classification: Literal[
        "direct_resolved_call", "import_only_reference", "uncertain_candidate"
    ]
    certainty: Literal["high", "medium", "low"]
    evidence: list[TestMappingEvidence] = Field(min_length=1)
    explanation: str

    @model_validator(mode="after")
    def validate_evidence_classification(self) -> "TestCodeMapping":
        evidence_types = {item.evidence_type for item in self.evidence}
        if self.evidence_classification == "direct_resolved_call":
            if evidence_types != {"resolved_call"} or self.certainty != "high":
                raise ValueError("direct resolved-call mappings require only resolved-call evidence and high certainty")
        elif self.evidence_classification == "import_only_reference":
            if not evidence_types or not evidence_types.issubset({"imported_symbol", "imported_module"}):
                raise ValueError("import-only mappings may contain only import evidence")
            expected_certainty = "medium" if "imported_symbol" in evidence_types else "low"
            if self.certainty != expected_certainty:
                raise ValueError("import-only mapping certainty must match its import evidence")
        elif (
            not evidence_types.issubset(
                {"unresolved_call_candidate", "resolved_call_outside_test"}
            )
            or not evidence_types
            or self.certainty != "low"
        ):
            raise ValueError("uncertain candidates require low certainty and candidate or unattributed-call evidence")
        if not _safe_repository_path(self.test_file_path) or not _valid_graph_node(self.source_component):
            raise ValueError("test mappings must reference safe paths and valid graph node identifiers")
        return self


class UncoveredComponent(BaseModel):
    """A code component with no discovered test reference in this analysis."""

    source_component: GraphNode
    evidence_classification: Literal[
        "import_only_reference",
        "uncertain_candidate",
        "no_discoverable_test_reference",
    ]
    explanation: str


class InferredTestSuggestion(BaseModel):
    """A recommendation to add or review tests, not an existing test record."""

    recommendation_type: Literal["inferred"] = "inferred"
    source_component: GraphNode
    evidence_classification: Literal[
        "import_only_reference",
        "uncertain_candidate",
        "no_discoverable_test_reference",
    ]
    explanation: str


class TestIntelligenceResponse(BaseModel):
    """Repository test inventory, reference mappings, and inferred gaps."""

    repository_id: str
    changed_file_path: str | None = None
    qualified_symbol_name: str | None = None
    discovered_test_files: list[DiscoveredTestFile] = Field(default_factory=list)
    mappings: list[TestCodeMapping] = Field(default_factory=list)
    uncovered_components: list[UncoveredComponent] = Field(default_factory=list)
    inferred_recommendations: list[InferredTestSuggestion] = Field(default_factory=list)
    impact_report: ChangeImpactReport | None = None
    max_results: int = Field(default=200, ge=1, le=1_000)
    analyzed_component_count: int = Field(default=0, ge=0)
    result_limit_reached: bool = False
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_report_consistency(self) -> "TestIntelligenceResponse":
        if self.analyzed_component_count > self.max_results:
            raise ValueError("analyzed component count cannot exceed max_results")
        if self.changed_file_path is not None and not _safe_repository_path(self.changed_file_path):
            raise ValueError("changed_file_path must be repository-relative")
        if self.qualified_symbol_name and not self.changed_file_path:
            raise ValueError("qualified_symbol_name requires changed_file_path")
        if self.impact_report and (
            self.impact_report.changed_file_path != self.changed_file_path
            or self.impact_report.changed_component.qualified_name != self.qualified_symbol_name
        ):
            raise ValueError("impact report scope must match the test intelligence request")
        files = {item.file_path: item for item in self.discovered_test_files}
        if len(files) != len(self.discovered_test_files):
            raise ValueError("discovered test file paths must be unique")
        cases_by_file = {
            path: {case.qualified_name: case for case in item.tests}
            for path, item in files.items()
        }

        mapped_by_source: dict[str, set[str]] = {}
        for mapping in self.mappings:
            if mapping.test_file_path not in files:
                raise ValueError("test mapping references an undiscovered test file")
            if mapping.test_qualified_name != "<module>":
                case = cases_by_file[mapping.test_file_path].get(mapping.test_qualified_name)
                if case is None:
                    raise ValueError("test mapping references an undiscovered test case")
                if any(
                    not case.start_line <= evidence.line_number <= case.end_line
                    for evidence in mapping.evidence
                ):
                    raise ValueError("test evidence line must fall within its mapped test case")
            mapped_by_source.setdefault(mapping.source_component.node_id, set()).add(
                mapping.evidence_classification
            )

        if len({(item.test_file_path, item.test_qualified_name, item.source_component.node_id, item.evidence_classification) for item in self.mappings}) != len(self.mappings):
            raise ValueError("test mappings must be unique by test, source component, and evidence class")

        uncovered_by_id = {
            item.source_component.node_id: item for item in self.uncovered_components
        }
        if len(uncovered_by_id) != len(self.uncovered_components):
            raise ValueError("uncovered component identifiers must be unique")
        for node_id, item in uncovered_by_id.items():
            if not _valid_graph_node(item.source_component):
                raise ValueError("uncovered components must reference valid graph node identifiers")
            classifications = mapped_by_source.get(node_id, set())
            expected = item.evidence_classification
            if expected == "no_discoverable_test_reference" and classifications:
                raise ValueError("components with no discoverable references cannot have mappings")
            if expected == "import_only_reference" and classifications != {"import_only_reference"}:
                raise ValueError("import-only component classification contradicts its mappings")
            if expected == "uncertain_candidate" and not (
                "uncertain_candidate" in classifications
                and "direct_resolved_call" not in classifications
            ):
                raise ValueError("uncertain component classification contradicts its mappings")

        for node_id, classifications in mapped_by_source.items():
            if "direct_resolved_call" in classifications and node_id in uncovered_by_id:
                raise ValueError("directly called components cannot be listed as uncovered")
            if node_id not in uncovered_by_id and "direct_resolved_call" not in classifications:
                raise ValueError("non-direct mappings must have an uncovered evidence classification")

        classified_component_ids = set(mapped_by_source) | set(uncovered_by_id)
        if len(classified_component_ids) != self.analyzed_component_count:
            raise ValueError("every analyzed component must have exactly one evidence classification")

        recommendation_ids = {
            item.source_component.node_id: item for item in self.inferred_recommendations
        }
        if len(recommendation_ids) != len(self.inferred_recommendations):
            raise ValueError("inferred recommendations must have unique component identifiers")
        if set(recommendation_ids) != set(uncovered_by_id):
            raise ValueError("inferred recommendations must correspond exactly to uncovered components")
        for node_id, recommendation in recommendation_ids.items():
            if not _valid_graph_node(recommendation.source_component):
                raise ValueError("inferred recommendations must reference valid graph node identifiers")
            if recommendation.evidence_classification != uncovered_by_id[node_id].evidence_classification:
                raise ValueError("recommendation classification must match its uncovered component")
        return self
