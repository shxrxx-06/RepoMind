"""Atomic local persistence for embedding matrices and aligned metadata."""

import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import ValidationError

from backend.app.embeddings.models import CodeEmbedding, EmbeddingBatch


class LocalEmbeddingStore:
    """Save/load compressed NumPy vectors alongside JSON-encoded metadata."""

    def save(self, batch: EmbeddingBatch, path: str | Path) -> None:
        """Write one aligned collection atomically; never use pickle."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        dimension = batch.embedding_dimension
        if batch.embeddings:
            dimensions = {len(item.vector) for item in batch.embeddings}
            models = {item.model_name for item in batch.embeddings}
            if len(dimensions) != 1 or len(models) != 1:
                raise ValueError("Embedding batch contains inconsistent models or dimensions")
            actual_dimension = dimensions.pop()
            if dimension is not None and dimension != actual_dimension:
                raise ValueError("Embedding batch dimension does not match its vectors")
            dimension = actual_dimension
            if batch.model_name not in models:
                raise ValueError("Embedding batch model name does not match its vectors")
            vectors = np.asarray([item.vector for item in batch.embeddings], dtype=np.float32)
        else:
            vectors = np.empty((0, dimension or 0), dtype=np.float32)

        metadata = [
            item.model_dump(
                exclude={"vector", "model_name", "embedding_dimension"}, mode="json"
            )
            for item in batch.embeddings
        ]
        handle, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        try:
            with os.fdopen(handle, "wb") as output:
                np.savez_compressed(
                    output,
                    vectors=vectors,
                    metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
                    model_name=np.asarray(batch.model_name),
                    embedding_dimension=np.asarray(dimension if dimension is not None else -1),
                )
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_name, target)
        except Exception:
            Path(temporary_name).unlink(missing_ok=True)
            raise

    def load(self, path: str | Path) -> EmbeddingBatch:
        """Load vectors and metadata; return an empty result with an error on failure."""
        target = Path(path)
        if not target.is_file():
            return EmbeddingBatch(
                embeddings=[], model_name="", storage_error=f"Embedding store not found: {target}"
            )
        try:
            with np.load(target, allow_pickle=False) as archive:
                vectors = np.asarray(archive["vectors"], dtype=np.float32)
                metadata: Any = json.loads(str(archive["metadata"].item()))
                model_name = str(archive["model_name"].item())
                stored_dimension = int(archive["embedding_dimension"].item())
            dimension = stored_dimension if stored_dimension > 0 else None
            if vectors.ndim != 2:
                raise ValueError("Stored vectors must be a two-dimensional array")
            if len(metadata) != vectors.shape[0]:
                raise ValueError("Stored metadata and vectors are misaligned")
            if dimension is not None and vectors.shape[1] != dimension:
                raise ValueError("Stored vector dimension does not match metadata")
            if len(metadata) and dimension is None:
                raise ValueError("Non-empty embedding store has no vector dimension")
            if not np.isfinite(vectors).all():
                raise ValueError("Stored vectors contain non-finite values")
            embeddings = [
                CodeEmbedding(
                    **item,
                    vector=vectors[index].astype(float).tolist(),
                    model_name=model_name,
                    embedding_dimension=dimension or 0,
                )
                for index, item in enumerate(metadata)
            ]
            return EmbeddingBatch(
                embeddings=embeddings,
                model_name=model_name,
                embedding_dimension=dimension,
            )
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            EOFError,
            zipfile.BadZipFile,
            ValidationError,
        ) as exc:
            return EmbeddingBatch(
                embeddings=[],
                model_name="",
                storage_error=f"Unable to load embedding store {target}: {exc}",
            )
