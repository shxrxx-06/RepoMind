"""Coordinate retrieval, graph exploration, conservative verification, and reporting."""

import re
from hashlib import sha256

import networkx as nx

from backend.app.graph.models import RepositoryAnalysisResult
from backend.app.investigation.context import InvestigationContextBuilder
from backend.app.investigation.models import (
    InvestigationClaim,
    InvestigationRequest,
    InvestigationResponse,
    InvestigationTrace,
    UnverifiedFinding,
)
from backend.app.investigation.planner import InvestigationPlanner
from backend.app.investigation.recommendations import TestRecommender
from backend.app.investigation.verifier import ClaimVerifier
from backend.app.rag.evidence_builder import EvidenceBuilder
from backend.app.rag.prompt_builder import PromptBuilder
from backend.app.rag.provider import LLMProvider
from backend.app.retrieval.hybrid_retriever import HybridRetriever

_CITATION = re.compile(r"\bE-[0-9a-f]{24}\b", re.IGNORECASE)
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


class InvestigationService:
    """Investigate a repository using injected retrieval, graph, and optional text provider."""

    def __init__(
        self,
        retriever: HybridRetriever,
        graph: RepositoryAnalysisResult | nx.DiGraph,
        *,
        repository_root: str | None = None,
        provider: LLMProvider | None = None,
        planner: InvestigationPlanner | None = None,
        verifier: ClaimVerifier | None = None,
        context_builder: InvestigationContextBuilder | None = None,
        test_recommender: TestRecommender | None = None,
        prompt_builder: PromptBuilder | None = None,
    ) -> None:
        self.retriever = retriever
        self.graph = graph
        self.repository_root = repository_root
        self.provider = provider
        self.planner = planner or InvestigationPlanner()
        self.verifier = verifier or ClaimVerifier()
        self.context_builder = context_builder or InvestigationContextBuilder()
        self.test_recommender = test_recommender or TestRecommender()
        self.prompt_builder = prompt_builder or PromptBuilder()

    def investigate(self, request: InvestigationRequest) -> InvestigationResponse:
        """Run a bounded investigation and separate verified from unresolved claims."""
        plan = self.planner.plan(request, self.retriever, self.graph)
        evidence_capacity = sum(
            len(result.model_dump_json()) * 6 + 1_000 for result in plan.results
        )
        builder = EvidenceBuilder(max_context_chars=max(2, evidence_capacity))
        evidence_context = builder.build(plan.results)
        steps = plan.steps
        trace_text = " ".join(
            f"{step.action}: {step.result}" for step in steps
        )
        prepared = self.context_builder.build(
            request.question,
            trace_text,
            evidence_context.items,
            request.context_token_budget,
            max_items=request.max_evidence_items,
            prompt_builder=self.prompt_builder,
        )
        selected = prepared.items
        token_estimate = prepared.token_estimate
        warnings = [*plan.warnings, *evidence_context.warnings, *prepared.warnings]
        if len(selected) < len(evidence_context.items):
            warnings.append("Some evidence was excluded by the investigation context budget.")
        recommendations = self.test_recommender.recommend(
            self.repository_root,
            {item.qualified_name.rsplit(".", 1)[-1] for item in selected},
        )
        if prepared.exceeds_budget:
            warnings.append(
                "The question and required prompt instructions exceed the context budget; "
                "the provider was not called."
            )
            return self._response(
                request,
                "The investigation context exceeds the configured budget.",
                [],
                [UnverifiedFinding(
                    claim=request.question,
                    explanation="The question and required instructions could not fit the context budget.",
                )],
                selected,
                steps,
                "Stopped because the required context exceeded its budget.",
                recommendations,
                warnings,
                token_estimate,
            )
        if self.provider is None:
            warnings.append("No investigation provider is configured; generated claims are unavailable.")
            return self._response(
                request, "Investigation completed without a text generation provider.",
                [], [], selected, steps, plan.stop_reason, recommendations, warnings,
                token_estimate,
            )
        if not any(item.source_content and item.source_content.strip() for item in selected):
            warnings.append("No source-backed evidence was available; the provider was not called.")
            return self._response(
                request, "No source-backed evidence was available to investigate.",
                [], [UnverifiedFinding(
                    claim=request.question,
                    explanation="No retrieved source snippet can support an answer.",
                )], selected, steps, "Stopped because no source-backed evidence was available.",
                recommendations, warnings, token_estimate,
            )

        try:
            generated = self.provider.generate(prepared.prompt)
        except Exception:  # noqa: BLE001 - provider implementations use varied errors.
            warnings.append("The investigation provider failed; no generated claims were returned.")
            return self._response(
                request, "The investigation could not generate a response.", [], [], selected,
                steps, "Stopped because the provider failed.", recommendations, warnings,
                token_estimate,
            )
        return self._classify_response(
            request, generated, selected, steps, plan.stop_reason,
            recommendations, warnings, token_estimate,
        )

    def _classify_response(
        self,
        request: InvestigationRequest,
        generated: object,
        evidence: list,
        steps: list,
        stop_reason: str,
        recommendations: list,
        warnings: list[str],
        token_estimate: int,
    ) -> InvestigationResponse:
        if not isinstance(generated, str) or not generated.strip():
            warnings.append("The investigation provider returned no usable response.")
            return self._response(
                request, "The investigation provider returned no usable response.", [], [],
                evidence, steps, "Stopped because the provider returned no content.",
                recommendations, warnings, token_estimate,
            )
        verified = []
        unverified = []
        claims = []
        indexed = {item.evidence_id.lower(): item for item in evidence}
        source_ids = {
            item.evidence_id.lower() for item in evidence
            if item.source_content and item.source_content.strip()
        }
        for sentence in _SENTENCE.split(generated.strip()):
            citations = {citation.lower() for citation in _CITATION.findall(sentence)}
            claim = _CITATION.sub("", sentence).strip(" \t\r\n[]() .,;:")
            if not claim:
                continue
            claim_id = "H-" + sha256(claim.encode("utf-8")).hexdigest()[:16]
            claims.append(
                InvestigationClaim(
                    claim_id=claim_id,
                    claim=claim,
                    evidence_ids=sorted(citations),
                )
            )
            cited_items = [indexed[item] for item in sorted(citations) if item in indexed]
            if citations - source_ids:
                unverified.append(UnverifiedFinding(
                    claim=claim,
                    status="unknown",
                    evidence_ids=sorted(citations),
                    explanation="At least one cited identifier has no source snippet or was not retrieved.",
                ))
                continue
            verification = self.verifier.verify(claim, cited_items)
            if hasattr(verification, "verification"):
                verified.append(verification)
            else:
                unverified.append(verification)
        if not claims:
            unverified.append(UnverifiedFinding(
                claim=request.question,
                explanation="The provider returned no claims that could be evaluated.",
            ))
        answer = (
            "Investigation findings are classified below. Claims that are not directly "
            "supported by source text remain unverified."
        )
        if unverified:
            warnings.append("One or more generated claims remain unverified under conservative source-text rules.")
        return self._response(
            request, answer, verified, unverified, evidence, steps, stop_reason,
            recommendations, warnings, token_estimate, claims,
        )

    @staticmethod
    def _response(
        request, answer, verified, unverified, evidence, steps, stop_reason,
        recommendations, warnings, token_estimate, claims=None,
    ):
        return InvestigationResponse(
            question=request.question,
            answer=answer,
            investigated_claims=claims or [],
            verified_findings=verified,
            unverified_findings=unverified,
            evidence=evidence,
            trace=InvestigationTrace(steps=steps, stop_reason=stop_reason),
            test_recommendations=recommendations,
            warnings=warnings,
            context_token_estimate=token_estimate,
            context_budget_tokens=request.context_token_budget,
        )
