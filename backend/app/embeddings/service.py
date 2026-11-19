"""Lazy Sentence Transformers service for normalized code embeddings."""

from collections.abc import Callable, Sequence
from threading import RLock
from time import perf_counter
from typing import Any, ClassVar

import numpy as np

from backend.app.embeddings.models import CodeDocument, CodeEmbedding, EmbeddingBatch

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingError(RuntimeError):
    """Base exception for model loading or vector generation failures."""


class EmbeddingModelLoadError(EmbeddingError):
    """Raised when the configured Sentence Transformer cannot be loaded."""


class EmbeddingGenerationError(EmbeddingError):
    """Raised when the model returns malformed or inconsistent vectors."""


class EmbeddingService:
    """Generate normalized embeddings, loading and reusing models on demand."""

    _model_cache: ClassVar[dict[tuple[str, str | None], Any]] = {}
    _cache_lock: ClassVar[RLock] = RLock()

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | None = "cpu",
        model_loader: Callable[[str, str | None], Any] | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model_loader = model_loader
        self._model: Any | None = None
        self._dimension: int | None = None

    @property
    def embedding_dimension(self) -> int:
        """Return the loaded model's dimension, loading it lazily if needed."""
        model = self._get_model()
        dimension = self._read_dimension(model)
        if dimension is None or dimension <= 0:
            raise EmbeddingModelLoadError(
                f"Model {self.model_name!r} does not report a valid embedding dimension"
            )
        self._dimension = dimension
        return dimension

    def embed_document(self, document: CodeDocument) -> CodeEmbedding:
        """Embed one document using the same batching and validation path."""
        return self.embed_documents([document]).embeddings[0]

    def embed_documents(
        self,
        documents: Sequence[CodeDocument],
        batch_size: int = 32,
    ) -> EmbeddingBatch:
        """Encode a document batch with normalized, dimension-checked vectors."""
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not documents:
            return EmbeddingBatch(
                embeddings=[],
                model_name=self.model_name,
                embedding_dimension=self._dimension,
                processing_time_seconds=0.0,
            )

        started = perf_counter()
        model = self._get_model()
        expected_dimension = self._read_dimension(model)
        texts = [document.text for document in documents]
        try:
            raw_vectors = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            vectors = np.asarray(raw_vectors, dtype=np.float32)
        except Exception as exc:
            raise EmbeddingGenerationError(
                f"Failed to generate embeddings with model {self.model_name!r}: {exc}"
            ) from exc

        if vectors.ndim == 1 and len(documents) == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.ndim != 2 or vectors.shape[0] != len(documents):
            raise EmbeddingGenerationError(
                "Model returned a vector matrix that does not match the document batch"
            )
        dimension = int(vectors.shape[1])
        if dimension <= 0 or (expected_dimension is not None and dimension != expected_dimension):
            raise EmbeddingGenerationError(
                f"Model returned dimension {dimension}; expected {expected_dimension}"
            )
        if not np.isfinite(vectors).all():
            raise EmbeddingGenerationError("Model returned non-finite vector values")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise EmbeddingGenerationError("Model returned a zero-length vector")
        vectors = vectors / norms
        self._dimension = dimension

        embedded = [
            CodeEmbedding(
                **document.model_dump(),
                vector=vectors[index].astype(float).tolist(),
                model_name=self.model_name,
                embedding_dimension=dimension,
            )
            for index, document in enumerate(documents)
        ]
        return EmbeddingBatch(
            embeddings=embedded,
            model_name=self.model_name,
            embedding_dimension=dimension,
            processing_time_seconds=perf_counter() - started,
        )

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        cache_key = (self.model_name, self.device)
        if self._model_loader is None:
            with self._cache_lock:
                cached = self._model_cache.get(cache_key)
                if cached is not None:
                    self._model = cached
                    return cached
                try:
                    from sentence_transformers import SentenceTransformer

                    model = SentenceTransformer(self.model_name, device=self.device)
                except Exception as exc:
                    raise EmbeddingModelLoadError(
                        f"Unable to load Sentence Transformer model {self.model_name!r} "
                        "(install sentence-transformers and check model availability): "
                        f"{exc}"
                    ) from exc
                self._model_cache[cache_key] = model
                self._model = model
                return model

        try:
            self._model = self._model_loader(self.model_name, self.device)
        except Exception as exc:
            raise EmbeddingModelLoadError(
                f"Unable to load Sentence Transformer model {self.model_name!r}: {exc}"
            ) from exc
        return self._model

    @staticmethod
    def _read_dimension(model: Any) -> int | None:
        try:
            dimension = model.get_sentence_embedding_dimension()
        except (AttributeError, TypeError, RuntimeError):
            return None
        return int(dimension) if dimension is not None else None
