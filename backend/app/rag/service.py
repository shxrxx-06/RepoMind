"""Evidence-grounded RAG orchestration over RepoMind's hybrid retriever."""

import re
from collections.abc import Sequence

from backend.app.rag.evidence_builder import EvidenceBuilder
from backend.app.rag.models import EvidenceItem, RAGRequest, RAGResponse
from backend.app.rag.prompt_builder import PromptBuilder
from backend.app.rag.provider import LLMProvider
from backend.app.retrieval.hybrid_retriever import HybridRetriever
from backend.app.retrieval.models import RetrievalQuery

_EVIDENCE_CITATION = re.compile(r"\bE-[0-9a-f]{24}\b", re.IGNORECASE)
_NO_EVIDENCE_ANSWER = "I don't have enough source-backed repository evidence to answer that question."
_PROVIDER_FAILURE_ANSWER = "I couldn't generate a grounded answer at this time."


class RAGService:
    """Retrieve source-backed evidence, prompt an injected provider, and validate citations."""

    def __init__(
        self,
        retriever: HybridRetriever,
        provider: LLMProvider,
        evidence_builder: EvidenceBuilder | None = None,
        prompt_builder: PromptBuilder | None = None,
    ) -> None:
        self.retriever = retriever
        self.provider = provider
        self.evidence_builder = evidence_builder or EvidenceBuilder()
        self.prompt_builder = prompt_builder or PromptBuilder()

    def answer(self, request: RAGRequest) -> RAGResponse:
        """Run retrieval and answer only when bounded source evidence is available."""
        try:
            retrieval = self.retriever.search(
                RetrievalQuery(
                    query=request.question,
                    top_k=request.top_k,
                    filters=request.filters,
                )
            )
        except Exception:  # noqa: BLE001 - safe failure at the orchestration boundary.
            return RAGResponse(
                question=request.question,
                answer=_NO_EVIDENCE_ANSWER,
                uncertainty="high",
                insufficient_evidence=True,
                warnings=["Repository retrieval failed; no answer was generated."],
            )

        context = self.evidence_builder.build(retrieval.results)
        warnings = [*retrieval.warnings, *context.warnings]
        has_source_evidence = any(
            item.source_content and item.source_content.strip() for item in context.items
        )
        if not retrieval.results or not context.items or not has_source_evidence:
            warnings.append("No usable source-backed evidence was available for generation.")
            return RAGResponse(
                question=request.question,
                answer=_NO_EVIDENCE_ANSWER,
                evidence=context.items,
                uncertainty="high",
                insufficient_evidence=True,
                warnings=warnings,
            )

        prompt = self.prompt_builder.build(request.question, context)
        try:
            generated = self.provider.generate(prompt)
        except Exception:  # noqa: BLE001 - provider SDKs expose varied exception classes.
            return RAGResponse(
                question=request.question,
                answer=_PROVIDER_FAILURE_ANSWER,
                evidence=context.items,
                uncertainty="high",
                warnings=[*warnings, "The answer provider failed; its response was not returned."],
            )

        if not isinstance(generated, str) or not generated.strip():
            return RAGResponse(
                question=request.question,
                answer=_PROVIDER_FAILURE_ANSWER,
                evidence=context.items,
                uncertainty="high",
                warnings=[*warnings, "The answer provider returned no usable response."],
            )

        answer = generated.strip()
        citations = {citation.lower() for citation in _EVIDENCE_CITATION.findall(answer)}
        valid_ids = {
            item.evidence_id.lower()
            for item in context.items
            if item.source_content and item.source_content.strip()
        }
        unknown_ids = sorted(citations - valid_ids)
        if unknown_ids:
            metadata_only_ids = {
                item.evidence_id.lower()
                for item in context.items
                if not item.source_content or not item.source_content.strip()
            }
            if citations & metadata_only_ids:
                warning = (
                    "The provider response cited evidence without a source snippet; "
                    "the answer was withheld."
                )
            else:
                warning = (
                    "The provider response cited evidence identifiers that were not "
                    "retrieved with source content; the answer was withheld."
                )
            return self._rejected_response(
                request,
                context.items,
                warnings,
                warning,
            )
        if not citations:
            return self._rejected_response(
                request,
                context.items,
                warnings,
                "The provider response did not cite retrieved evidence; the answer was withheld.",
            )

        return RAGResponse(
            question=request.question,
            answer=answer,
            evidence=context.items,
            uncertainty="medium",
            insufficient_evidence=False,
            warnings=warnings,
        )

    @staticmethod
    def _rejected_response(
        request: RAGRequest,
        evidence: Sequence[EvidenceItem],
        warnings: list[str],
        warning: str,
    ) -> RAGResponse:
        return RAGResponse(
            question=request.question,
            answer="The generated response could not be validated against retrieved evidence.",
            evidence=evidence,
            uncertainty="high",
            insufficient_evidence=False,
            warnings=[*warnings, warning],
        )
