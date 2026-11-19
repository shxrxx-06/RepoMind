"""Local semantic, keyword, and weighted RRF hybrid code retrieval."""

import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from backend.app.embeddings.models import CodeEmbedding
from backend.app.embeddings.service import EmbeddingError, EmbeddingService
from backend.app.retrieval.models import (
    RetrievalFilters,
    RetrievalQuery,
    RetrievalResponse,
    RetrievalResult,
)
from backend.app.storage.local_embedding_store import LocalEmbeddingStore

_TOKEN_PATTERN = re.compile(r"[\w]+", re.UNICODE)
_IDENTIFIER_NORMALIZER = re.compile(r"[^\w]+", re.UNICODE)


class HybridRetriever:
    """Search a persisted embedding collection with semantic, lexical, and RRF ranking."""

    def __init__(
        self,
        store_path: str | Path,
        embedding_service: EmbeddingService,
        store: LocalEmbeddingStore | None = None,
        semantic_candidate_limit: int = 100,
        keyword_candidate_limit: int = 100,
    ) -> None:
        if semantic_candidate_limit < 1 or keyword_candidate_limit < 1:
            raise ValueError("Candidate limits must be positive")
        self.store_path = Path(store_path)
        self.embedding_service = embedding_service
        self.store = store or LocalEmbeddingStore()
        self.semantic_candidate_limit = semantic_candidate_limit
        self.keyword_candidate_limit = keyword_candidate_limit

    def search(self, request: RetrievalQuery) -> RetrievalResponse:
        """Return deterministic ranked results; query failures degrade to keyword search."""
        query_text = request.query.strip()
        if not query_text:
            return RetrievalResponse(
                query=request.query,
                warnings=["Query is empty; no search was performed."],
            )

        try:
            batch = self.store.load(self.store_path)
        except (OSError, ValueError, TypeError, RuntimeError) as exc:
            return RetrievalResponse(
                query=request.query,
                warnings=[f"Unable to load embedding store: {exc}"],
            )
        warnings = [batch.storage_error] if batch.storage_error else []
        if not batch.embeddings:
            if not warnings:
                warnings.append("Embedding store contains no code documents.")
            return RetrievalResponse(query=request.query, warnings=warnings)

        documents = self._unique_documents(batch.embeddings)
        filtered = [
            document
            for document in documents
            if self._matches_filters(document, request.filters)
        ]
        if not filtered:
            return RetrievalResponse(
                query=request.query,
                total_documents=len(batch.embeddings),
                filtered_documents=0,
                warnings=warnings,
            )

        semantic_limit = request.semantic_limit or self.semantic_candidate_limit
        keyword_limit = request.keyword_limit or self.keyword_candidate_limit
        semantic_ranked: list[tuple[CodeEmbedding, float]] = []
        if request.semantic_weight > 0:
            semantic_ranked, semantic_warning = self._semantic_search(
                query_text, filtered, semantic_limit
            )
            if semantic_warning:
                warnings.append(semantic_warning)

        keyword_ranked: list[tuple[CodeEmbedding, float, str]] = []
        if request.keyword_weight > 0:
            keyword_ranked = self._keyword_search(query_text, filtered, keyword_limit)

        results = self._fuse(
            request,
            semantic_ranked,
            keyword_ranked,
        )[: request.top_k]
        return RetrievalResponse(
            query=request.query,
            results=results,
            total_documents=len(batch.embeddings),
            filtered_documents=len(filtered),
            warnings=warnings,
        )

    @staticmethod
    def _unique_documents(documents: Sequence[CodeEmbedding]) -> list[CodeEmbedding]:
        by_id: dict[str, CodeEmbedding] = {}
        for document in documents:
            by_id.setdefault(document.document_id, document)
        return [by_id[key] for key in sorted(by_id)]

    @staticmethod
    def _matches_filters(
        document: CodeEmbedding,
        filters: RetrievalFilters | None,
    ) -> bool:
        if filters is None:
            return True
        path = document.source_file_path.replace("\\", "/").casefold()
        if filters.source_file_path and path != filters.source_file_path.replace("\\", "/").casefold():
            return False
        if filters.path_prefix:
            prefix = filters.path_prefix.replace("\\", "/").strip("/").casefold()
            if path != prefix and not path.startswith(f"{prefix}/"):
                return False
        if filters.symbol_type and document.symbol_type.casefold() != filters.symbol_type.casefold():
            return False
        return not (
            filters.qualified_name
            and document.qualified_symbol_name.casefold()
            != filters.qualified_name.casefold()
        )

    def _semantic_search(
        self,
        query: str,
        documents: list[CodeEmbedding],
        limit: int,
    ) -> tuple[list[tuple[CodeEmbedding, float]], str | None]:
        model_names = {document.model_name for document in documents}
        if model_names != {self.embedding_service.model_name}:
            return [], (
                "Semantic search skipped because stored vectors use a different or mixed "
                "embedding model."
            )
        dimensions = {document.embedding_dimension for document in documents}
        if len(dimensions) != 1:
            return [], "Semantic search skipped because stored vector dimensions are inconsistent."
        try:
            query_vector = np.asarray(self.embedding_service.embed_text(query), dtype=np.float32)
        except (EmbeddingError, ValueError, RuntimeError) as exc:
            return [], f"Semantic search unavailable: {exc}"
        if query_vector.ndim != 1 or query_vector.shape[0] != next(iter(dimensions)):
            return [], "Semantic search skipped because the query vector dimension does not match the store."
        query_norm = float(np.linalg.norm(query_vector))
        if not np.isfinite(query_norm) or query_norm == 0:
            return [], "Semantic search skipped because the query vector is invalid."
        query_vector = query_vector / query_norm

        scored: list[tuple[CodeEmbedding, float]] = []
        for document in documents:
            vector = np.asarray(document.vector, dtype=np.float32)
            norm = float(np.linalg.norm(vector))
            if vector.ndim != 1 or vector.shape[0] != query_vector.shape[0] or norm == 0:
                continue
            score = float(np.dot(query_vector, vector) / norm)
            if np.isfinite(score):
                scored.append((document, score))
        scored.sort(key=lambda item: (-item[1], item[0].document_id))
        return scored[:limit], None

    @classmethod
    def _keyword_search(
        cls,
        query: str,
        documents: list[CodeEmbedding],
        limit: int,
    ) -> list[tuple[CodeEmbedding, float, str]]:
        query_folded = query.casefold().strip()
        query_identifier = cls._normalize_identifier(query_folded)
        query_tokens = set(_TOKEN_PATTERN.findall(query_folded))
        scored: list[tuple[CodeEmbedding, float, str]] = []
        for document in documents:
            score, reason = cls._keyword_score(
                document,
                query_folded,
                query_identifier,
                query_tokens,
            )
            if score > 0:
                scored.append((document, score, reason))
        scored.sort(key=lambda item: (-item[1], item[0].document_id))
        return scored[:limit]

    @classmethod
    def _keyword_score(
        cls,
        document: CodeEmbedding,
        query: str,
        query_identifier: str,
        query_tokens: set[str],
    ) -> tuple[float, str]:
        qualified = document.qualified_symbol_name.casefold()
        symbol_name = qualified.rsplit(".", 1)[-1]
        qualified_identifier = cls._normalize_identifier(qualified)
        symbol_identifier = cls._normalize_identifier(symbol_name)
        if qualified == query or qualified_identifier == query_identifier:
            return 1_000.0, "exact qualified-name match"
        if symbol_name == query or symbol_identifier == query_identifier:
            return 900.0, "exact symbol-name match"

        score = 0.0
        reasons: list[str] = []
        if query in qualified:
            score += 400.0
            reasons.append("query appears in qualified name")
        qualified_tokens = set(_TOKEN_PATTERN.findall(qualified))
        name_overlap = len(query_tokens & qualified_tokens)
        if query_tokens and name_overlap == len(query_tokens):
            score += 200.0
            reasons.append("all query terms appear in symbol name")
        elif name_overlap:
            score += 40.0 * name_overlap / len(query_tokens)
            reasons.append("partial symbol-name term match")

        docstring = (document.docstring or "").casefold()
        source = (document.source_snippet or "").casefold()
        if query in docstring:
            score += 30.0
            reasons.append("phrase match in docstring")
        doc_overlap = len(query_tokens & set(_TOKEN_PATTERN.findall(docstring)))
        if doc_overlap:
            score += 8.0 * doc_overlap / max(len(query_tokens), 1)
        if query in source:
            score += 15.0
            reasons.append("phrase match in source")
        source_overlap = len(query_tokens & set(_TOKEN_PATTERN.findall(source)))
        if source_overlap:
            score += 3.0 * source_overlap / max(len(query_tokens), 1)

        prepared_text = document.text.casefold()
        if query in prepared_text:
            score += 5.0
            reasons.append("phrase match in prepared code document")
        prepared_overlap = len(query_tokens & set(_TOKEN_PATTERN.findall(prepared_text)))
        if prepared_overlap:
            score += 1.0 * prepared_overlap / max(len(query_tokens), 1)

        metadata_text = " ".join(
            [
                document.source_file_path,
                document.symbol_type,
                *document.decorators,
                *document.bases,
                *[
                    value
                    for parameter in document.parameters
                    for value in (
                        parameter.name,
                        parameter.annotation or "",
                        parameter.default or "",
                    )
                ],
            ]
        ).casefold()
        metadata_overlap = len(query_tokens & set(_TOKEN_PATTERN.findall(metadata_text)))
        if metadata_overlap:
            score += 2.0 * metadata_overlap / max(len(query_tokens), 1)
            reasons.append("match in symbol metadata")
        return score, "; ".join(reasons) or "keyword term match"

    @staticmethod
    def _normalize_identifier(value: str) -> str:
        return _IDENTIFIER_NORMALIZER.sub("", value.casefold())

    @classmethod
    def _fuse(
        cls,
        request: RetrievalQuery,
        semantic_ranked: list[tuple[CodeEmbedding, float]],
        keyword_ranked: list[tuple[CodeEmbedding, float, str]],
    ) -> list[RetrievalResult]:
        candidates: dict[str, dict[str, object]] = {}
        for rank, (document, score) in enumerate(semantic_ranked, start=1):
            candidate = candidates.setdefault(document.document_id, {"document": document})
            candidate["semantic_score"] = score
            candidate["semantic_rank"] = rank
            candidate["hybrid_score"] = float(candidate.get("hybrid_score", 0.0)) + (
                request.semantic_weight / (request.rrf_k + rank)
            )
        for rank, (document, score, reason) in enumerate(keyword_ranked, start=1):
            candidate = candidates.setdefault(document.document_id, {"document": document})
            candidate["keyword_score"] = score
            candidate["keyword_rank"] = rank
            candidate["keyword_reason"] = reason
            candidate["hybrid_score"] = float(candidate.get("hybrid_score", 0.0)) + (
                request.keyword_weight / (request.rrf_k + rank)
            )

        ordered = sorted(
            candidates.values(),
            key=lambda item: (
                -float(item["hybrid_score"]),
                int(item.get("semantic_rank", 10**9)),
                int(item.get("keyword_rank", 10**9)),
                item["document"].document_id,
            ),
        )
        results: list[RetrievalResult] = []
        for rank, candidate in enumerate(ordered, start=1):
            document = candidate["document"]
            semantic_rank = candidate.get("semantic_rank")
            keyword_rank = candidate.get("keyword_rank")
            contributions = []
            if semantic_rank is not None:
                contributions.append(
                    f"semantic rank {semantic_rank} contributes "
                    f"{request.semantic_weight:g}/({request.rrf_k}+{semantic_rank})"
                )
            if keyword_rank is not None:
                contributions.append(
                    f"keyword rank {keyword_rank} contributes "
                    f"{request.keyword_weight:g}/({request.rrf_k}+{keyword_rank})"
                )
            reason = candidate.get("keyword_reason")
            if reason:
                contributions.append(str(reason))
            explanation = "Weighted reciprocal rank fusion: " + "; ".join(contributions)
            results.append(
                RetrievalResult(
                    document_id=document.document_id,
                    symbol_name=document.qualified_symbol_name.rsplit(".", 1)[-1],
                    qualified_name=document.qualified_symbol_name,
                    source_file_path=document.source_file_path,
                    symbol_type=document.symbol_type,
                    start_line=document.start_line,
                    end_line=document.end_line,
                    source_content=document.source_snippet,
                    text=document.text,
                    semantic_score=candidate.get("semantic_score"),
                    semantic_rank=semantic_rank,
                    keyword_score=candidate.get("keyword_score"),
                    keyword_rank=keyword_rank,
                    hybrid_score=float(candidate["hybrid_score"]),
                    hybrid_rank=rank,
                    ranking_explanation=explanation,
                )
            )
        return results
