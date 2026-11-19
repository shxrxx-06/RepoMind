"""Exact rendered-prompt budgeting for repository investigations."""

import json
from dataclasses import dataclass

from backend.app.rag.models import EvidenceContext, EvidenceItem, GroundedPrompt
from backend.app.rag.prompt_builder import PromptBuilder

ADDITIONAL_INVESTIGATION_INSTRUCTIONS = (
    "For each factual statement, cite evidence and ensure the claim is directly "
    "verifiable from its source snippet. Do not infer beyond the text."
)


@dataclass(frozen=True)
class PreparedInvestigationContext:
    """The complete rendered prompt and its character-based budget measurement."""

    prompt: GroundedPrompt
    items: list[EvidenceItem]
    character_count: int
    token_estimate: int
    warnings: list[str]
    exceeds_budget: bool


class InvestigationContextBuilder:
    """Select whole evidence items using the final prompt's exact rendered length.

    Token estimates use the documented 4-characters-per-token approximation. They are
    estimates, not tokenizer output. The combined system and user message length is the
    budgeted character count; no component is charged separately or counted twice.
    """

    CHARS_PER_TOKEN = 4

    def build(
        self,
        question: str,
        trace_text: str,
        candidates: list[EvidenceItem],
        token_budget: int,
        *,
        max_items: int,
        prompt_builder: PromptBuilder | None = None,
    ) -> PreparedInvestigationContext:
        """Render, measure, and fit complete evidence records into the final prompt."""
        if token_budget < 1 or max_items < 1:
            raise ValueError("token_budget and max_items must be positive")
        builder = prompt_builder or PromptBuilder()
        budget_chars = token_budget * self.CHARS_PER_TOKEN
        warnings: list[str] = []
        selected: list[EvidenceItem] = []
        base_prompt = self._render(builder, question, trace_text, selected)
        base_characters = self._character_count(base_prompt)
        if base_characters > budget_chars:
            return PreparedInvestigationContext(
                prompt=base_prompt,
                items=[],
                character_count=base_characters,
                token_estimate=self._token_estimate(base_characters),
                warnings=[
                    "The mandatory instructions, question, prompt labels, and investigation "
                    "trace exceed the approximate context budget."
                ],
                exceeds_budget=True,
            )

        ordered = sorted(candidates, key=lambda item: (item.hybrid_rank, item.evidence_id))
        for item in ordered:
            if item in selected:
                continue
            if len(selected) >= max_items:
                warnings.append("Evidence item limit reached; remaining whole items were omitted.")
                break
            candidate_prompt = self._render(builder, question, trace_text, [*selected, item])
            if self._character_count(candidate_prompt) > budget_chars:
                warnings.append(
                    f"Evidence {item.evidence_id} was omitted as a whole item to stay within "
                    "the approximate prompt budget."
                )
                continue
            selected.append(item)

        final_prompt = self._render(builder, question, trace_text, selected)
        character_count = self._character_count(final_prompt)
        return PreparedInvestigationContext(
            prompt=final_prompt,
            items=selected,
            character_count=character_count,
            token_estimate=self._token_estimate(character_count),
            warnings=warnings,
            exceeds_budget=character_count > budget_chars,
        )

    @staticmethod
    def _render(
        prompt_builder: PromptBuilder,
        question: str,
        trace_text: str,
        items: list[EvidenceItem],
    ) -> GroundedPrompt:
        serialized = json.dumps(
            [item.model_dump(mode="json") for item in items],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        context = EvidenceContext(
            items=items,
            context_text=serialized,
            context_characters=len(serialized),
        )
        base = prompt_builder.build(question, context)
        return GroundedPrompt(
            system_instructions=(
                f"{base.system_instructions}\n{ADDITIONAL_INVESTIGATION_INSTRUCTIONS}"
            ),
            user_message=f"{base.user_message}\nInvestigation trace:\n{trace_text}",
        )

    @staticmethod
    def _character_count(prompt: GroundedPrompt) -> int:
        return len(prompt.system_instructions) + len(prompt.user_message)

    @classmethod
    def _token_estimate(cls, character_count: int) -> int:
        return (character_count + cls.CHARS_PER_TOKEN - 1) // cls.CHARS_PER_TOKEN
