"""DeepSeek models, through the OpenAI-compatible Chat Completions API."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from tagsort.fallback._http import HttpProvider
from tagsort.fallback.base import BoxFormat, Usage, VisionRequest

__all__ = ["DeepSeekProvider"]


class DeepSeekProvider(HttpProvider):
    """Reads tags with a DeepSeek model through ``POST /chat/completions``.

    DeepSeek's JSON mode does not enforce a schema, so the schema is given in the
    instructions and the answer is validated like every other provider's.

    Example:
        >>> provider = DeepSeekProvider(api_key="sk-...")  # doctest: +SKIP
    """

    name = "deepseek"
    # Images are resized server-side to at most about 1300 px, so boxes are requested in
    # normalized units.
    box_format: BoxFormat = "normalized_xyxy"
    default_model = "deepseek-flash"
    default_max_side = 1300
    url = "https://api.deepseek.com/chat/completions"

    def _build(
        self, request: VisionRequest, image_base64: str
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        instructions = (
            f"{request.instructions}\n\nAnswer with a single JSON object that follows this "
            f"JSON Schema, and nothing else:\n{json.dumps(request.schema)}"
        )
        body: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instructions},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                        },
                    ],
                }
            ],
            "response_format": {"type": "json_object"},
        }
        return self.url, headers, body

    def _extract(self, data: Mapping[str, Any]) -> tuple[str, str, Usage]:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise self._error("the answer has no choice", "invalid_response")
        choice = choices[0]
        finish = choice.get("finish_reason")
        if finish == "content_filter":
            raise self._error("the answer was blocked by a content filter", "refused")
        if finish == "length":
            raise self._error("the answer was cut off at the token limit", "truncated")
        message = choice.get("message")
        if not isinstance(message, dict):
            raise self._error("the answer has no message", "invalid_response")
        if message.get("refusal"):
            raise self._error("the model declined to read this image", "refused")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise self._error("the answer has no text", "invalid_response")
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        assert isinstance(usage, dict)
        return (
            content,
            str(data.get("model") or self._model),
            Usage(
                input_tokens=int(usage.get("prompt_tokens") or 0),
                output_tokens=int(usage.get("completion_tokens") or 0),
            ),
        )
