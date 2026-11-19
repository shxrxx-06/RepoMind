"""Structured inputs and outputs for bounded repository investigations."""

from typing import Literal

from pydantic import AliasChoices, BaseModel, Field, field_validator

from backend.app.rag.models import EvidenceItem
from backend.app.retrieval.models import RetrievalFilters


class InvestigationRequest(BaseModel):
    question: str = Field(min_length=1, max_length=20_000)
    top_k: int = Field(default=5, ge=1, le=100)
    max_steps: int = Field(default=5, ge=1, le=20)
    max_graph_depth: int = Field(default=2, ge=0, le=10)
    max_evidence_items: int = Field(default=20, ge=1, le=200)
    context_token_budget: int = Field(default=8_000, ge=256, le=100_000)
    filters: RetrievalFilters | None = None

    @field_validator("question")
    @classmethod
    def question_is_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


class EvidenceReference(BaseModel):
    evidence_id: str
    source_file_path: str
    qualified_name: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)


class TraversedRelationship(BaseModel):
    """One actual graph edge traversed, with its complete start-to-node path."""

    source_node_id: str
    target_node_id: str
    relationship_type: str
    resolution_status: str | None = None
    source_lines: list[int] = Field(default_factory=list)
    direction: Literal["incoming", "outgoing"]
    traversal_path: list[str] = Field(min_length=2)


class InvestigationStep(BaseModel):
    step_number: int = Field(ge=1)
    action: str
    symbols_examined: list[str] = Field(default_factory=list)
    files_examined: list[str] = Field(default_factory=list)
    relationships_traversed: list[TraversedRelationship] = Field(default_factory=list)
    evidence_ids_collected: list[str] = Field(default_factory=list)
    result: str
    continuation_reason: str


class InvestigationClaim(BaseModel):
    """A claim produced during investigation, whether supported or not."""

    claim_id: str = Field(validation_alias=AliasChoices("claim_id", "hypothesis_id"))
    claim: str
    evidence_ids: list[str] = Field(default_factory=list)


# Kept as a source-compatible name for clients written during Day 7.
InvestigationHypothesis = InvestigationClaim


class VerifiedFinding(BaseModel):
    """A literal textual observation, not proof that the code runs or entails behavior."""

    claim: str
    verification: Literal["verified", "partially_supported"]
    verification_method: Literal[
        "literal_source_match", "partial_literal_source_match"
    ] = "literal_source_match"
    evidence: list[EvidenceReference]
    explanation: str


class UnverifiedFinding(BaseModel):
    claim: str
    status: Literal["hypothesis", "unknown"] = "unknown"
    verification_method: Literal["not_verified"] = "not_verified"
    evidence_ids: list[str] = Field(default_factory=list)
    explanation: str


class InvestigationTrace(BaseModel):
    steps: list[InvestigationStep] = Field(default_factory=list)
    stop_reason: str


class TestRecommendation(BaseModel):
    test_file_path: str | None = None
    matched_symbols: list[str] = Field(default_factory=list)
    reason: str


class InvestigationResponse(BaseModel):
    question: str
    answer: str
    investigated_claims: list[InvestigationClaim] = Field(
        default_factory=list,
        validation_alias=AliasChoices("investigated_claims", "hypotheses"),
    )
    verified_findings: list[VerifiedFinding] = Field(default_factory=list)
    unverified_findings: list[UnverifiedFinding] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    trace: InvestigationTrace
    test_recommendations: list[TestRecommendation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    context_token_estimate: int = Field(default=0, ge=0)
    context_budget_tokens: int = Field(ge=1)

    @property
    def hypotheses(self) -> list[InvestigationClaim]:
        """Compatibility accessor for Day 7 clients; includes all investigated claims."""
        return self.investigated_claims
