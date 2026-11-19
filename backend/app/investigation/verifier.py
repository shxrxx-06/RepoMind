"""Conservative source-text claim verification rules."""

import re

from backend.app.investigation.models import (
    EvidenceReference,
    UnverifiedFinding,
    VerifiedFinding,
)
from backend.app.rag.models import EvidenceItem

_WHITESPACE = re.compile(r"\s+")
_CLAUSE = re.compile(r"\s+(?:and|but|while|whereas)\s+|[;,]")
_NEGATION = re.compile(r"\b(?:not|never|no)\s+$")


def _normalize(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip().casefold()


class ClaimVerifier:
    """Classify literal source-text matches without inferring semantic entailment."""

    def verify(
        self,
        claim: str,
        evidence: list[EvidenceItem],
    ) -> VerifiedFinding | UnverifiedFinding:
        """Match claim text literally; this does not establish execution or behavior.

        Exact substring checking is intentionally conservative. Comments and string
        literals count as text matches, but are not evidence that code executes. Semantic
        paraphrases remain unknown, even when paired with a valid evidence identifier.
        """
        claim_text = _normalize(claim)
        if not claim_text:
            return UnverifiedFinding(claim=claim, explanation="The claim is empty.")
        matches: list[EvidenceItem] = []
        partial: list[EvidenceItem] = []
        has_literal_conflict = False
        for item in evidence:
            if not item.source_content or not item.source_content.strip():
                continue
            source = _normalize(item.source_content)
            occurrences = list(re.finditer(re.escape(claim_text), source))
            if occurrences:
                positive_occurrences = [
                    occurrence
                    for occurrence in occurrences
                    if not _NEGATION.search(source[max(0, occurrence.start() - 16):occurrence.start()])
                ]
                if positive_occurrences:
                    matches.append(item)
                if len(positive_occurrences) != len(occurrences):
                    has_literal_conflict = True
            elif any(
                len(segment.strip()) >= 8 and segment.strip() in source
                for segment in _CLAUSE.split(claim_text)
            ):
                partial.append(item)
        if matches and has_literal_conflict:
            return UnverifiedFinding(
                claim=claim,
                status="unknown",
                evidence_ids=[item.evidence_id for item in evidence],
                explanation=(
                    "The supplied source text contains both positive and negated literal "
                    "matches; the claim remains ambiguous."
                ),
            )
        if matches:
            return VerifiedFinding(
                claim=claim,
                verification="verified",
                verification_method="literal_source_match",
                evidence=[self._reference(item) for item in matches],
                explanation=(
                    "The claim text appears literally in the cited source snippet. This does "
                    "not establish semantic entailment or that the code executes at runtime."
                ),
            )
        if partial:
            return VerifiedFinding(
                claim=claim,
                verification="partially_supported",
                verification_method="partial_literal_source_match",
                evidence=[self._reference(item) for item in partial],
                explanation=(
                    "A complete clause appears literally in the source text; other claim "
                    "content remains unsupported. This does not establish runtime behavior."
                ),
            )
        return UnverifiedFinding(
            claim=claim,
            status="unknown",
            evidence_ids=[item.evidence_id for item in evidence],
            explanation="No source snippet directly supports this claim under the verifier rules.",
        )

    @staticmethod
    def _reference(item: EvidenceItem) -> EvidenceReference:
        return EvidenceReference(
            evidence_id=item.evidence_id,
            source_file_path=item.source_file_path,
            qualified_name=item.qualified_name,
            start_line=item.start_line,
            end_line=item.end_line,
        )
