"""Provider abstraction, deterministic mock, and OpenAI-compatible integration."""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, ClassVar, Protocol
from urllib.parse import urlsplit

import httpx

from backend.app.rag.models import GroundedPrompt

if TYPE_CHECKING:
    from backend.app.core.config import Settings


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


class LLMProviderError(RuntimeError):
    """Safe provider failure that never includes credentials or response bodies."""


class LLMConfigurationError(LLMProviderError):
    """Provider configuration is incomplete or invalid."""


class OpenAICompatibleProvider:
    """Synchronous chat-completions provider for OpenAI-compatible endpoints."""

    _RETRYABLE_STATUS_CODES: ClassVar[frozenset[int]] = frozenset({408, 409, 429})

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str,
        model: str | None,
        timeout: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        if not api_key or not api_key.strip():
            raise LLMConfigurationError("Production LLM API key is not configured.")
        if not model or not model.strip():
            raise LLMConfigurationError("Production LLM model is not configured.")
        if not base_url or not base_url.strip():
            raise LLMConfigurationError("Production LLM base URL is not configured.")
        if not math.isfinite(timeout) or not 0 < timeout <= 600 or not 0 <= max_retries <= 10:
            raise LLMConfigurationError("Production LLM request settings are invalid.")
        normalized_base_url = base_url.strip().rstrip("/")
        try:
            parsed_url = httpx.URL(normalized_base_url)
        except (TypeError, ValueError, httpx.InvalidURL):
            raise LLMConfigurationError("Production LLM base URL is invalid.") from None
        if (
            parsed_url.scheme not in {"http", "https"}
            or not parsed_url.host
            or parsed_url.query
            or parsed_url.fragment
        ):
            raise LLMConfigurationError("Production LLM base URL is invalid.")
        if (
            "@" in urlsplit(normalized_base_url).netloc
            or parsed_url.username
            or parsed_url.password
        ):
            raise LLMConfigurationError(
                "Production LLM base URL must not contain embedded credentials."
            )
        local_hosts = {"localhost", "127.0.0.1", "::1"}
        if parsed_url.scheme == "http" and parsed_url.host.casefold() not in local_hosts:
            raise LLMConfigurationError(
                "Production LLM endpoints must use HTTPS unless hosted locally."
            )
        self._api_key = api_key
        self._endpoint = f"{normalized_base_url}/chat/completions"
        self._model = model.strip()
        self._timeout = timeout
        self._max_retries = max_retries

    def generate(self, prompt: GroundedPrompt) -> str:
        """Request a completion with bounded retries for transient failures."""
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": prompt.system_instructions},
                {"role": "user", "content": prompt.user_message},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        for attempt in range(self._max_retries + 1):
            try:
                response = httpx.post(
                    self._endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self._timeout,
                )
            except (httpx.TimeoutException, httpx.NetworkError):
                if attempt < self._max_retries:
                    time.sleep(min(0.25 * (2**attempt), 8.0))
                    continue
                raise LLMProviderError("Production LLM request failed due to a network error.") from None
            except httpx.HTTPError:
                raise LLMProviderError("Production LLM request failed.") from None

            status_code = response.status_code
            retryable = status_code in self._RETRYABLE_STATUS_CODES or status_code >= 500
            if retryable:
                if attempt < self._max_retries:
                    time.sleep(min(0.25 * (2**attempt), 8.0))
                    continue
                raise LLMProviderError(
                    f"Production LLM request failed with HTTP status {status_code}."
                )
            if status_code < 200 or status_code >= 300:
                raise LLMProviderError(
                    f"Production LLM request failed with HTTP status {status_code}."
                )
            try:
                body = response.json()
                content = body["choices"][0]["message"]["content"]
            except (ValueError, KeyError, IndexError, TypeError):
                raise LLMProviderError("Production LLM returned a malformed response.") from None
            if not isinstance(content, str) or not content.strip():
                raise LLMProviderError("Production LLM returned no usable text.")
            return content.strip()
        raise LLMProviderError("Production LLM request failed.")


class ConfiguredLLMProvider:
    """Lazily select the explicitly configured provider on first generation."""

    def __init__(self, configuration: Settings) -> None:
        self._configuration = configuration
        self._provider: LLMProvider | None = None

    def _get_provider(self) -> LLMProvider:
        if self._provider is None:
            if self._configuration.LLM_PROVIDER == "mock":
                self._provider = MockLLMProvider(self._configuration.LLM_MOCK_RESPONSE)
            elif self._configuration.LLM_PROVIDER == "openai":
                key = self._configuration.OPENAI_API_KEY
                self._provider = OpenAICompatibleProvider(
                    api_key=key.get_secret_value() if key else None,
                    base_url=self._configuration.OPENAI_BASE_URL,
                    model=self._configuration.OPENAI_MODEL,
                    timeout=self._configuration.LLM_REQUEST_TIMEOUT,
                    max_retries=self._configuration.LLM_MAX_RETRIES,
                )
            else:
                raise LLMConfigurationError("Configured LLM provider is unsupported.")
        return self._provider

    def generate(self, prompt: GroundedPrompt) -> str:
        return self._get_provider().generate(prompt)


def configured_provider(configuration: Settings) -> ConfiguredLLMProvider:
    """Create a lazy provider adapter for the supplied application settings."""
    return ConfiguredLLMProvider(configuration)
