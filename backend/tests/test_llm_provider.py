import httpx
import pytest
from pydantic import SecretStr

from backend.app.core.config import Settings
from backend.app.rag.models import GroundedPrompt
from backend.app.rag.provider import (
    ConfiguredLLMProvider,
    LLMConfigurationError,
    LLMProviderError,
    MockLLMProvider,
    OpenAICompatibleProvider,
    configured_provider,
)


def _prompt():
    return GroundedPrompt(system_instructions="Be evidence grounded.", user_message="Question")


def _provider(*, retries=2, base_url="https://llm.example/v1/"):
    return OpenAICompatibleProvider(
        api_key="test-secret-key",
        base_url=base_url,
        model="test-model",
        timeout=3.0,
        max_retries=retries,
    )


class FakeResponse:
    def __init__(self, status_code=200, body=None):
        self.status_code = status_code
        self.body = body or {
            "choices": [{"message": {"content": "Grounded answer [E-abc]."}}]
        }

    def json(self):
        return self.body


@pytest.mark.parametrize(
    ("api_key", "model", "message"),
    [
        (None, "model", "API key"),
        ("  ", "model", "API key"),
        ("secret", None, "model"),
        ("secret", " ", "model"),
    ],
)
def test_production_configuration_requires_key_and_model(api_key, model, message):
    with pytest.raises(LLMConfigurationError, match=message) as error:
        OpenAICompatibleProvider(
            api_key=api_key,
            base_url="https://llm.example/v1",
            model=model,
        )
    assert "secret" not in str(error.value)


def test_openai_compatible_provider_sends_grounded_messages(monkeypatch):
    captured = {}

    def post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return FakeResponse()

    monkeypatch.setattr(httpx, "post", post)

    result = _provider().generate(_prompt())

    assert result == "Grounded answer [E-abc]."
    assert captured["url"] == "https://llm.example/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer test-secret-key"
    assert captured["timeout"] == 3.0
    assert captured["json"] == {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": "Be evidence grounded."},
            {"role": "user", "content": "Question"},
        ],
    }


def test_https_remote_endpoint_is_accepted():
    provider = _provider(base_url="https://llm.example/custom/v1")

    assert provider._endpoint == "https://llm.example/custom/v1/chat/completions"


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "[::1]"])
def test_http_is_accepted_for_explicit_local_hosts(monkeypatch, host):
    requested_urls = []
    monkeypatch.setattr(
        httpx,
        "post",
        lambda url, **kwargs: requested_urls.append(url) or FakeResponse(),
    )
    provider = _provider(base_url=f"http://{host}:8080/v1")

    assert provider.generate(_prompt()).startswith("Grounded answer")
    assert requested_urls == [f"http://{host}:8080/v1/chat/completions"]


@pytest.mark.parametrize(
    "base_url",
    [
        "http://llm.example/v1",
        "https://user:password@llm.example/v1",
        "https://@llm.example/v1",
        "https://llm.example:invalid/v1",
        "not-a-url",
    ],
)
def test_unsafe_or_malformed_urls_are_rejected_before_any_request(
    monkeypatch, base_url
):
    requests = []
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *args, **kwargs: requests.append((args, kwargs)),
    )
    settings = Settings(
        _env_file=None,
        LLM_PROVIDER="openai",
        OPENAI_API_KEY=SecretStr("test-secret-key"),
        OPENAI_BASE_URL=base_url,
        OPENAI_MODEL="test-model",
    )
    provider = configured_provider(settings)

    with pytest.raises(LLMConfigurationError):
        provider.generate(_prompt())

    assert requests == []


def test_authentication_failure_is_not_retried_or_exposed(monkeypatch):
    requests = []

    def post(*args, **kwargs):
        requests.append(1)
        return FakeResponse(401, {"error": {"message": "test-secret-key invalid"}})

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr("backend.app.rag.provider.time.sleep", lambda _: pytest.fail("slept"))

    with pytest.raises(LLMProviderError, match="401") as error:
        _provider().generate(_prompt())

    assert len(requests) == 1
    assert "test-secret-key" not in str(error.value)


def test_rate_limit_uses_bounded_exponential_backoff(monkeypatch):
    statuses = iter((429, 429, 200))
    requests = []
    delays = []

    def post(*args, **kwargs):
        requests.append(1)
        status = next(statuses)
        return FakeResponse(status)

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr("backend.app.rag.provider.time.sleep", delays.append)

    assert _provider(retries=2).generate(_prompt()).startswith("Grounded answer")
    assert len(requests) == 3
    assert delays == [0.25, 0.5]


@pytest.mark.parametrize("exception", [httpx.ReadTimeout, httpx.ConnectError])
def test_timeout_and_connection_failures_retry_then_succeed(monkeypatch, exception):
    requests = []

    def post(*args, **kwargs):
        requests.append(1)
        if len(requests) == 1:
            raise exception("secret-safe failure")
        return FakeResponse()

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr("backend.app.rag.provider.time.sleep", lambda _: None)

    assert _provider().generate(_prompt()).startswith("Grounded answer")
    assert len(requests) == 2


def test_retryable_server_failures_stop_after_configured_attempts(monkeypatch):
    requests = []

    def post(*args, **kwargs):
        requests.append(1)
        return FakeResponse(503)

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr("backend.app.rag.provider.time.sleep", lambda _: None)

    with pytest.raises(LLMProviderError, match="503"):
        _provider(retries=2).generate(_prompt())
    assert len(requests) == 3


@pytest.mark.parametrize(
    "response",
    [
        FakeResponse(body={"invalid": True}),
        FakeResponse(body={"choices": []}),
        FakeResponse(body={"choices": [{"message": {"content": []}}]}),
    ],
)
def test_malformed_provider_responses_are_rejected(monkeypatch, response):
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: response)
    with pytest.raises(LLMProviderError, match="malformed|usable"):
        _provider().generate(_prompt())


def test_provider_selection_supports_mock_and_production_without_fallback():
    mock_settings = Settings(
        _env_file=None,
        LLM_PROVIDER="mock",
        LLM_MOCK_RESPONSE="mock answer",
    )
    mock = configured_provider(mock_settings)
    assert isinstance(mock, ConfiguredLLMProvider)
    assert mock.generate(_prompt()) == "mock answer"
    assert isinstance(mock._provider, MockLLMProvider)

    production_settings = Settings(
        _env_file=None,
        LLM_PROVIDER="openai",
        OPENAI_API_KEY=SecretStr(""),
        OPENAI_MODEL="",
    )
    production = configured_provider(production_settings)
    with pytest.raises(LLMConfigurationError):
        production.generate(_prompt())
    assert production._provider is None


def test_mock_provider_remains_directly_usable():
    provider = MockLLMProvider("fixed response")
    assert provider.generate(_prompt()) == "fixed response"
    assert provider.prompts == [_prompt()]
