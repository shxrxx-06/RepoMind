"""Embedding data models and lazy model-backed service."""

from backend.app.embeddings.models import CodeDocument, CodeEmbedding, EmbeddingBatch
from backend.app.embeddings.service import (
    DEFAULT_MODEL_NAME,
    EmbeddingError,
    EmbeddingGenerationError,
    EmbeddingModelLoadError,
    EmbeddingService,
)

__all__ = [
    "DEFAULT_MODEL_NAME",
    "CodeDocument",
    "CodeEmbedding",
    "EmbeddingBatch",
    "EmbeddingError",
    "EmbeddingGenerationError",
    "EmbeddingModelLoadError",
    "EmbeddingService",
]
