"""Provider abstraction and deterministic mock implementation for RAG tests."""

from typing import Protocol

from backend.app.rag.models import GroundedPrompt


class LLMProvider(Protocol):
    """Interface implemented by any synchronous text generation provider."""

    def generate(self, prompt: GroundedPrompt) -> str:
        """Return a response for the supplied grounded prompt."""


class MockLLMProvider:
    """Test/development provider that returns configured text without network access."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[GroundedPrompt] = []

    def generate(self, prompt: GroundedPrompt) -> str:
        self.prompts.append(prompt)
        return self.response
