"""Structured request, evidence, prompt, and response models for grounded RAG."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.app.retrieval.models import RetrievalFilters


class RAGRequest(BaseModel):
    question: str = Field(min_length=1, max_length=20_000)
    top_k: int = Field(default=5, ge=1, le=100)
    filters: RetrievalFilters | None = None

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


class EvidenceItem(BaseModel):
    evidence_id: str
    document_id: str
    qualified_name: str
    source_file_path: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    source_content: str | None = None
    source_content_missing: bool
    indexed_text: str
    semantic_score: float | None = None
    semantic_rank: int | None = None
    keyword_score: float | None = None
    keyword_rank: int | None = None
    hybrid_score: float
    hybrid_rank: int = Field(ge=1)

    @model_validator(mode="after")
    def line_range_is_valid(self) -> "EvidenceItem":
        if self.end_line < self.start_line:
            raise ValueError("end_line must be greater than or equal to start_line")
        if self.source_content_missing != (not bool(self.source_content and self.source_content.strip())):
            raise ValueError("source_content_missing must reflect source_content availability")
        return self

    model_config = ConfigDict(use_enum_values=True)


class EvidenceContext(BaseModel):
    items: list[EvidenceItem] = Field(default_factory=list)
    context_text: str = "[]"
    warnings: list[str] = Field(default_factory=list)
    context_characters: int = 2


class GroundedPrompt(BaseModel):
    system_instructions: str
    user_message: str


class RAGResponse(BaseModel):
    question: str
    answer: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    uncertainty: Literal["low", "medium", "high"] = "high"
    insufficient_evidence: bool = False
    warnings: list[str] = Field(default_factory=list)

