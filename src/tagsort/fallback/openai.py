"""GPT models, through the OpenAI Responses API."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tagsort.fallback._http import HttpProvider
from tagsort.fallback.base import BoxFormat, Usage, VisionRequest

__all__ = ["OpenAIProvider"]


class OpenAIProvider(HttpProvider):
    """Reads tags with an OpenAI model through ``POST /v1/responses``.

    Example:
        >>> provider = OpenAIProvider(api_key="sk-...")  # doctest: +SKIP
    """

    name = "openai"
    # The API may resize images internally, so boxes are requested in normalized units.
    box_format: BoxFormat = "normalized_xyxy"
    default_model = "gpt-6.1-sol"
    default_max_side = 2048
    url = "https://api.openai.com/v1/responses"

    def _build(
        self, request: VisionRequest, image_base64: str
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        body: dict[str, Any] = {
            "model": self._model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": request.instructions},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{image_base64}",
                            "detail": "high",
                        },
                    ],
                }
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "tags",
                    "strict": True,
                    "schema": request.schema,
                }
            },
        }
        return self.url, headers, body

    def _extract(self, data: Mapping[str, Any]) -> tuple[str, str, Usage]:
        if data.get("status") == "incomplete":
            details = data.get("incomplete_details")
            reason = details.get("reason") if isinstance(details, dict) else None
            if reason == "content_filter":
                raise self._error("the answer was blocked by a content filter", "refused")
            raise self._error(f"the answer is incomplete ({reason})", "truncated")
        texts: list[str] = []
        output = data.get("output")
        for item in output if isinstance(output, list) else []:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            for part in item.get("content") or []:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "refusal":
                    raise self._error("the model declined to read this image", "refused")
                if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                    texts.append(part["text"])
        if not texts:
            raise self._error("the answer has no output text", "invalid_response")
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        assert isinstance(usage, dict)
        return (
            texts[0],
            str(data.get("model") or self._model),
            Usage(
                input_tokens=int(usage.get("input_tokens") or 0),
                output_tokens=int(usage.get("output_tokens") or 0),
            ),
        )
