"""Pydantic models for prepared code documents and their vectors."""

from math import isfinite

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.app.parsing.models import ParameterInfo


class CodeDocument(BaseModel):
    """One symbol-level text item with source metadata kept separately."""

    document_id: str
    text: str
    source_file_path: str
    qualified_symbol_name: str
    symbol_type: str
    start_line: int
    end_line: int
    source_snippet: str | None = None
    docstring: str | None = None
    decorators: list[str] = Field(default_factory=list)
    bases: list[str] = Field(default_factory=list)
    parameters: list[ParameterInfo] = Field(default_factory=list)
    return_annotation: str | None = None
    method_type: str | None = None
    is_async: bool = False


class CodeEmbedding(CodeDocument):
    """A code document with its normalized vector and model metadata."""

    vector: list[float]
    model_name: str
    embedding_dimension: int

    @field_validator("vector")
    @classmethod
    def validate_finite_vector(cls, value: list[float]) -> list[float]:
        if not all(isfinite(item) for item in value):
            raise ValueError("Embedding vector values must be finite")
        return value

    @model_validator(mode="after")
    def validate_dimension(self) -> "CodeEmbedding":
        if self.embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be positive")
        if len(self.vector) != self.embedding_dimension:
            raise ValueError("Vector length does not match embedding_dimension")
        return self


class EmbeddingBatch(BaseModel):
    """Embeddings created or loaded together, with optional operation metadata."""

    embeddings: list[CodeEmbedding] = Field(default_factory=list)
    model_name: str
    embedding_dimension: int | None = None
    processing_time_seconds: float | None = None
    storage_error: str | None = None

    model_config = ConfigDict(use_enum_values=True)
