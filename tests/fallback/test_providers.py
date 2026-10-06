import base64
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from tagsort import AnthropicProvider, GeminiProvider, OpenAIProvider, ProviderError, VisionProvider
from tagsort.fallback._http import HttpProvider
from tagsort.fallback.base import VisionRequest
from tagsort.fallback.prompt import ANSWER_SCHEMA

KEY = "secret-key-123"
ANSWER = {
    "tags": [
        {
            "text": "MD04127",
            "legibility": "certain",
            "alternatives": [],
            "box": [100, 200, 300, 400],
            "angle": "0",
        }
    ]
}
REQUEST = VisionRequest(
    jpeg=b"\xff\xd8jpeg",
    width=1000,
    height=1000,
    instructions="Read the tags.",
    schema=ANSWER_SCHEMA,
)


def anthropic_ok(text: str = json.dumps(ANSWER), **changes: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": "claude-opus-5-5",
        "stop_reason": "end_turn",
        "content": [{"type": "thinking", "thinking": ""}, {"type": "text", "text": text}],
        "usage": {"input_tokens": 1500, "output_tokens": 80},
    }
    body.update(changes)
    return body


def openai_ok(text: str = json.dumps(ANSWER), **changes: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": "gpt-6.1-sol",
        "status": "completed",
        "output": [
            {"type": "reasoning", "summary": []},
            {"type": "message", "content": [{"type": "output_text", "text": text}]},
        ],
        "usage": {"input_tokens": 1200, "output_tokens": 60},
    }
    body.update(changes)
    return body


def gemini_ok(text: str = json.dumps(ANSWER), **changes: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "modelVersion": "gemini-3.1-pro-preview",
        "candidates": [
            {
                "content": {"parts": [{"text": "thinking...", "thought": True}, {"text": text}]},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 1100,
            "candidatesTokenCount": 50,
            "thoughtsTokenCount": 30,
        },
    }
    body.update(changes)
    return body


PROVIDERS: dict[str, tuple[type[HttpProvider], Callable[..., dict[str, Any]]]] = {
    "anthropic": (AnthropicProvider, anthropic_ok),
    "openai": (OpenAIProvider, openai_ok),
    "gemini": (GeminiProvider, gemini_ok),
}


class Recorder:
    def __init__(self, *responses: httpx.Response | Exception) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []
        self.sleeps: list[float] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def provider(self, cls: type[HttpProvider], **kwargs: Any) -> HttpProvider:
        client = httpx.Client(transport=httpx.MockTransport(self.handler))
        return cls(api_key=KEY, client=client, sleep=self.sleeps.append, **kwargs)

    def body(self, index: int = 0) -> dict[str, Any]:
        data: dict[str, Any] = json.loads(self.requests[index].content)
        return data


@pytest.mark.parametrize("name", PROVIDERS)
def test_successful_read(name: str) -> None:
    cls, ok = PROVIDERS[name]
    recorder = Recorder(httpx.Response(200, json=ok()))
    provider = recorder.provider(cls)
    assert isinstance(provider, VisionProvider)
    answer = provider.read(REQUEST)
    (tag,) = answer.tags
    assert tag.text == "MD04127"
    assert answer.usage.input_tokens > 0
    assert answer.usage.output_tokens > 0
    assert answer.model == provider.model
    sent = recorder.requests[0]
    assert KEY not in str(sent.url), "the API key must never be in the URL"
    assert base64.b64encode(REQUEST.jpeg).decode() in sent.content.decode()
    assert "Read the tags." in sent.content.decode()


def test_anthropic_request_shape() -> None:
    recorder = Recorder(httpx.Response(200, json=anthropic_ok()))
    recorder.provider(AnthropicProvider).read(REQUEST)
    request = recorder.requests[0]
    assert str(request.url) == "https://api.anthropic.com/v1/messages"
    assert request.headers["x-api-key"] == KEY
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert request.headers["anthropic-beta"] == "server-side-fallback-2026-07-01"
    body = recorder.body()
    assert body["model"] == "claude-opus-5-5"
    assert body["fallbacks"] == "default"
    assert body["output_config"]["format"] == {"type": "json_schema", "schema": ANSWER_SCHEMA}
    image, text = body["messages"][0]["content"]
    assert image["source"]["media_type"] == "image/jpeg"
    assert text == {"type": "text", "text": "Read the tags."}
    assert "thinking" not in body


def test_anthropic_fallbacks_only_on_supported_models() -> None:
    recorder = Recorder(httpx.Response(200, json=anthropic_ok()))
    recorder.provider(AnthropicProvider, model="claude-haiku-4-5").read(REQUEST)
    assert "fallbacks" not in recorder.body()
    assert "anthropic-beta" not in recorder.requests[0].headers


def test_openai_request_shape() -> None:
    recorder = Recorder(httpx.Response(200, json=openai_ok()))
    recorder.provider(OpenAIProvider).read(REQUEST)
    request = recorder.requests[0]
    assert str(request.url) == "https://api.openai.com/v1/responses"
    assert request.headers["authorization"] == f"Bearer {KEY}"
    body = recorder.body()
    text, image = body["input"][0]["content"]
    assert text == {"type": "input_text", "text": "Read the tags."}
    assert image["type"] == "input_image"
    assert image["image_url"].startswith("data:image/jpeg;base64,")
    assert body["text"]["format"]["type"] == "json_schema"
    assert body["text"]["format"]["strict"] is True


def test_openai_boxes_are_normalized() -> None:
    recorder = Recorder(httpx.Response(200, json=openai_ok()))
    (tag,) = (
        recorder.provider(OpenAIProvider)
        .read(
            VisionRequest(jpeg=b"x", width=2000, height=500, instructions="", schema=ANSWER_SCHEMA)
        )
        .tags
    )
    assert tag.box == (200.0, 100.0, 600.0, 200.0)


def test_gemini_request_shape() -> None:
    recorder = Recorder(httpx.Response(200, json=gemini_ok()))
    answer = recorder.provider(GeminiProvider, model="gemini-test").read(REQUEST)
    request = recorder.requests[0]
    assert str(request.url) == (
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-test:generateContent"
    )
    assert request.headers["x-goog-api-key"] == KEY
    body = recorder.body()
    image, text = body["contents"][0]["parts"]
    assert image["inline_data"]["mime_type"] == "image/jpeg"
    assert text == {"text": "Read the tags."}
    assert body["generationConfig"]["responseJsonSchema"] == ANSWER_SCHEMA
    # Gemini boxes are [y_min, x_min, y_max, x_max] on 0-1000.
    (tag,) = answer.tags
    assert tag.box == (200.0, 100.0, 400.0, 300.0)
    # Thinking tokens are billed as output.
    assert answer.usage.output_tokens == 80


REFUSALS = {
    "anthropic": [anthropic_ok(stop_reason="refusal", content=[])],
    "openai": [
        openai_ok(output=[{"type": "message", "content": [{"type": "refusal", "refusal": "no"}]}]),
        openai_ok(status="incomplete", incomplete_details={"reason": "content_filter"}),
    ],
    "gemini": [
        gemini_ok(promptFeedback={"blockReason": "SAFETY"}),
        gemini_ok(candidates=[{"finishReason": "PROHIBITED_CONTENT"}]),
    ],
}
TRUNCATIONS = {
    "anthropic": [anthropic_ok(stop_reason="max_tokens")],
    "openai": [openai_ok(status="incomplete", incomplete_details={"reason": "max_output_tokens"})],
    "gemini": [gemini_ok(candidates=[{"finishReason": "MAX_TOKENS"}])],
}
MALFORMED = {
    "anthropic": [anthropic_ok(content=[]), anthropic_ok(content=None), anthropic_ok(text="{}")],
    "openai": [
        openai_ok(output=[]),
        openai_ok(output=[{"type": "message", "content": [{"type": "x"}]}]),
    ],
    "gemini": [
        gemini_ok(candidates=[]),
        gemini_ok(candidates=[{"content": {"parts": []}}]),
        gemini_ok(candidates=["x"]),
    ],
}


@pytest.mark.parametrize(
    ("name", "body", "reason"),
    [(n, b, "refused") for n, bodies in REFUSALS.items() for b in bodies]
    + [(n, b, "truncated") for n, bodies in TRUNCATIONS.items() for b in bodies]
    + [(n, b, "invalid_response") for n, bodies in MALFORMED.items() for b in bodies],
)
def test_unusable_answers(name: str, body: dict[str, Any], reason: str) -> None:
    recorder = Recorder(httpx.Response(200, json=body))
    with pytest.raises(ProviderError) as info:
        recorder.provider(PROVIDERS[name][0]).read(REQUEST)
    assert info.value.reason == reason
    assert info.value.provider == name
    assert len(recorder.requests) == 1, "unusable answers are not retried"


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        (400, "bad_request"),
        (401, "authentication"),
        (403, "authentication"),
        (404, "bad_request"),
        (413, "bad_request"),
    ],
)
def test_client_errors_are_not_retried(status: int, reason: str) -> None:
    recorder = Recorder(httpx.Response(status, json={"error": {"message": "nope " * 200}}))
    with pytest.raises(ProviderError) as info:
        recorder.provider(AnthropicProvider).read(REQUEST)
    assert info.value.reason == reason
    assert info.value.status_code == status
    assert not info.value.retryable
    assert len(recorder.requests) == 1
    assert len(str(info.value)) < 400


def test_retries_then_succeeds() -> None:
    recorder = Recorder(
        httpx.Response(429, headers={"retry-after": "7"}, json={"error": {"message": "slow down"}}),
        httpx.Response(529, text="overloaded"),
        httpx.Response(200, json=anthropic_ok()),
    )
    answer = recorder.provider(AnthropicProvider).read(REQUEST)
    assert answer.tags
    assert len(recorder.requests) == 3
    assert recorder.sleeps[0] == 7
    assert 1 <= recorder.sleeps[1] <= 3


def test_retry_after_is_capped() -> None:
    recorder = Recorder(
        httpx.Response(503, headers={"retry-after": "3600"}), httpx.Response(200, json=openai_ok())
    )
    recorder.provider(OpenAIProvider).read(REQUEST)
    assert recorder.sleeps == [60]


def test_unparsable_retry_after_falls_back_to_backoff() -> None:
    recorder = Recorder(
        httpx.Response(503, headers={"retry-after": "soon"}), httpx.Response(200, json=openai_ok())
    )
    recorder.provider(OpenAIProvider).read(REQUEST)
    assert 1 <= recorder.sleeps[0] <= 2


def test_gives_up_after_max_retries() -> None:
    recorder = Recorder(*[httpx.Response(500, json={"error": "flat"})] * 3)
    with pytest.raises(ProviderError) as info:
        recorder.provider(GeminiProvider, max_retries=2).read(REQUEST)
    assert info.value.reason == "server"
    assert info.value.retryable
    assert len(recorder.requests) == 3


def test_network_errors_are_retried() -> None:
    recorder = Recorder(httpx.ConnectError("down"), httpx.Response(200, json=gemini_ok()))
    assert recorder.provider(GeminiProvider).read(REQUEST).tags
    recorder = Recorder(httpx.ReadTimeout("slow"))
    with pytest.raises(ProviderError) as info:
        recorder.provider(GeminiProvider, max_retries=0).read(REQUEST)
    assert info.value.reason == "network"


@pytest.mark.parametrize(
    "response", [httpx.Response(200, text="<html>"), httpx.Response(200, json=[1])]
)
def test_non_json_responses(response: httpx.Response) -> None:
    with pytest.raises(ProviderError) as info:
        Recorder(response).provider(AnthropicProvider).read(REQUEST)
    assert info.value.reason == "invalid_response"


def test_key_is_never_revealed() -> None:
    provider = AnthropicProvider(api_key=KEY)
    assert KEY not in repr(provider)
    assert KEY not in str(vars(provider).get("_model"))
    provider.close()


def test_configuration() -> None:
    with OpenAIProvider(api_key=KEY, model="gpt-x", max_side=512) as provider:
        assert provider.model == "gpt-x"
        assert provider.max_side == 512
        assert provider.name == "openai"
    with pytest.raises(ValueError, match="api_key"):
        AnthropicProvider(api_key="")
    with pytest.raises(ValueError, match="max_retries"):
        AnthropicProvider(api_key=KEY, max_retries=-1)


def test_caller_client_is_not_closed() -> None:
    client = httpx.Client()
    AnthropicProvider(api_key=KEY, client=client).close()
    assert not client.is_closed
    client.close()


def test_base_class_hooks_are_abstract() -> None:
    provider = HttpProvider(api_key=KEY, client=httpx.Client())
    with pytest.raises(NotImplementedError):
        provider._build(REQUEST, "")
    with pytest.raises(NotImplementedError):
        provider._extract({})


def test_openai_skips_unknown_parts() -> None:
    body = openai_ok()
    body["output"][1]["content"].insert(0, "noise")
    assert Recorder(httpx.Response(200, json=body)).provider(OpenAIProvider).read(REQUEST).tags


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(
            429, json={"error": {"message": "no credits", "code": "insufficient_quota"}}
        ),
        httpx.Response(400, json={"error": {"type": "billing_error", "message": "credit balance"}}),
        httpx.Response(402, json={"error": {"message": "payment required"}}),
    ],
)
def test_exhausted_quota_is_not_retried(response: httpx.Response) -> None:
    recorder = Recorder(response)
    with pytest.raises(ProviderError) as info:
        recorder.provider(OpenAIProvider).read(REQUEST)
    assert info.value.reason == "quota"
    assert not info.value.retryable
    assert len(recorder.requests) == 1
    assert recorder.sleeps == []


def test_error_code_must_be_a_string() -> None:
    recorder = Recorder(
        httpx.Response(429, json={"error": {"code": 42}}), httpx.Response(200, json=openai_ok())
    )
    assert recorder.provider(OpenAIProvider).read(REQUEST).tags
