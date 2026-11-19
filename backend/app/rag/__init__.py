"""Evidence-grounded retrieval-augmented generation components."""

from backend.app.rag.evidence_builder import EvidenceBuilder
from backend.app.rag.models import (
    EvidenceContext,
    EvidenceItem,
    GroundedPrompt,
    RAGRequest,
    RAGResponse,
)
from backend.app.rag.prompt_builder import PromptBuilder
from backend.app.rag.provider import (
    ConfiguredLLMProvider,
    LLMConfigurationError,
    LLMProvider,
    LLMProviderError,
    MockLLMProvider,
    OpenAICompatibleProvider,
    configured_provider,
)
from backend.app.rag.service import RAGService

__all__ = [
    "ConfiguredLLMProvider",
    "EvidenceBuilder",
    "EvidenceContext",
    "EvidenceItem",
    "GroundedPrompt",
    "LLMConfigurationError",
    "LLMProvider",
    "LLMProviderError",
    "MockLLMProvider",
    "OpenAICompatibleProvider",
    "PromptBuilder",
    "RAGRequest",
    "RAGResponse",
    "RAGService",
    "configured_provider",
]
