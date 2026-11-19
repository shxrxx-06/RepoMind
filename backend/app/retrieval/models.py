"""Structured query, filter, result, and response models for retrieval."""

from math import isfinite

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RetrievalFilters(BaseModel):
    """Optional exact or prefix filters over existing embedding metadata."""

    source_file_path: str | None = None
    path_prefix: str | None = None
    symbol_type: str | None = None
    qualified_name: str | None = None


class RetrievalQuery(BaseModel):
    """Natural-language query and configurable hybrid retrieval controls."""

    query: str
    top_k: int = Field(default=10, ge=1, le=1_000)
    semantic_limit: int | None = Field(default=None, ge=1, le=10_000)
    keyword_limit: int | None = Field(default=None, ge=1, le=10_000)
    semantic_weight: float = Field(default=1.0, ge=0.0)
    keyword_weight: float = Field(default=1.0, ge=0.0)
    rrf_k: int = Field(default=60, ge=1)
    filters: RetrievalFilters | None = None

    @model_validator(mode="after")
    def at_least_one_channel_enabled(self) -> "RetrievalQuery":
        if not isfinite(self.semantic_weight) or not isfinite(self.keyword_weight):
            raise ValueError("Retrieval weights must be finite")
        if self.semantic_weight == 0 and self.keyword_weight == 0:
            raise ValueError("At least one retrieval channel must have positive weight")
        return self


class RetrievalResult(BaseModel):
    """Ranked code symbol with channel scores and original source metadata."""

    document_id: str
    symbol_name: str
    qualified_name: str
    source_file_path: str
    symbol_type: str
    start_line: int
    end_line: int
    source_content: str | None = None
    text: str
    semantic_score: float | None = None
    semantic_rank: int | None = None
    keyword_score: float | None = None
    keyword_rank: int | None = None
    hybrid_score: float
    hybrid_rank: int
    ranking_explanation: str

    model_config = ConfigDict(use_enum_values=True)


class RetrievalResponse(BaseModel):
    """Ranked result set with filtering counts and non-fatal warnings."""

    query: str
    results: list[RetrievalResult] = Field(default_factory=list)
    total_documents: int = 0
    filtered_documents: int = 0
    ranking_strategy: str = "weighted reciprocal rank fusion (RRF)"
    warnings: list[str] = Field(default_factory=list)

