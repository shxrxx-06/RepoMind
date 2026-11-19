"""Build bounded, traceable evidence items from hybrid retrieval results."""

import hashlib
import json

from backend.app.rag.models import EvidenceContext, EvidenceItem
from backend.app.retrieval.models import RetrievalResult


class EvidenceBuilder:
    """Convert retrieval hits into source-preserving evidence within a char budget."""

    def __init__(self, max_context_chars: int = 20_000) -> None:
        if max_context_chars < 2:
            raise ValueError("max_context_chars must be at least 2")
        self.max_context_chars = max_context_chars

    def build(self, results: list[RetrievalResult]) -> EvidenceContext:
        """Keep whole evidence records that fit; omit oversized items without truncating."""
        items: list[EvidenceItem] = []
        warnings: list[str] = []
        seen_documents: set[str] = set()
        for result in sorted(results, key=lambda item: (item.hybrid_rank, item.document_id)):
            if result.document_id in seen_documents:
                continue
            seen_documents.add(result.document_id)
            snippet = result.source_content
            snippet_available = bool(snippet and snippet.strip())
            evidence = EvidenceItem(
                evidence_id=(
                    "E-" + hashlib.sha256(result.document_id.encode("utf-8")).hexdigest()[:24]
                ),
                document_id=result.document_id,
                qualified_name=result.qualified_name,
                source_file_path=result.source_file_path,
                start_line=result.start_line,
                end_line=result.end_line,
                source_content=snippet if snippet_available else None,
                source_content_missing=not snippet_available,
                indexed_text=result.text,
                semantic_score=result.semantic_score,
                semantic_rank=result.semantic_rank,
                keyword_score=result.keyword_score,
                keyword_rank=result.keyword_rank,
                hybrid_score=result.hybrid_score,
                hybrid_rank=result.hybrid_rank,
            )
            candidate = [*items, evidence]
            serialized = self._serialize(candidate)
            if len(serialized) > self.max_context_chars:
                warnings.append(
                    f"Evidence {evidence.evidence_id} was omitted because the complete item "
                    f"would exceed the {self.max_context_chars}-character context limit."
                )
                continue
            items.append(evidence)
            if evidence.source_content_missing:
                warnings.append(
                    f"Evidence {evidence.evidence_id} has no stored source snippet; "
                    "its metadata and indexed text are retained, but no source excerpt is cited."
                )

        return EvidenceContext(
            items=items,
            context_text=self._serialize(items),
            warnings=warnings,
            context_characters=len(self._serialize(items)),
        )

    @staticmethod
    def _serialize(items: list[EvidenceItem]) -> str:
        return json.dumps(
            [item.model_dump(mode="json") for item in items],
            ensure_ascii=False,
            separators=(",", ":"),
        )
