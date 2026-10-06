"""Gemini models, through the Gemini API ``generateContent`` endpoint."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tagsort.fallback._http import HttpProvider
from tagsort.fallback.base import BoxFormat, Usage, VisionRequest

__all__ = ["GeminiProvider"]

_REFUSED = frozenset(
    {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY"}
)


class GeminiProvider(HttpProvider):
    """Reads tags with a Gemini model through ``models/{model}:generateContent``.

    Example:
        >>> provider = GeminiProvider(api_key="...")  # doctest: +SKIP
    """

    name = "gemini"
    # Gemini is trained to report boxes as [y_min, x_min, y_max, x_max] on a 0-1000 scale.
    box_format: BoxFormat = "normalized_yxyx"
    # Measured on lot-02 (eval/reports): as accurate as larger models, fastest and among
    # the cheapest. See docs/providers.md.
    default_model = "gemini-3.5-flash-lite"
    default_max_side = 2048
    base_url = "https://generativelanguage.googleapis.com/v1beta"

    def _build(
        self, request: VisionRequest, image_base64: str
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        # The key goes in a header, never in the URL, so it cannot leak into logs.
        headers = {"x-goog-api-key": self._api_key}
        body: dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"inline_data": {"mime_type": "image/jpeg", "data": image_base64}},
                        {"text": request.instructions},
                    ],
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseJsonSchema": request.schema,
            },
        }
        return f"{self.base_url}/models/{self._model}:generateContent", headers, body

    def _extract(self, data: Mapping[str, Any]) -> tuple[str, str, Usage]:
        feedback = data.get("promptFeedback")
        if isinstance(feedback, dict) and feedback.get("blockReason"):
            raise self._error(f"the image was blocked ({feedback['blockReason']})", "refused")
        candidates = data.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise self._error("the answer has no candidate", "invalid_response")
        candidate = candidates[0] if isinstance(candidates[0], dict) else {}
        finish = candidate.get("finishReason")
        if finish in _REFUSED:
            raise self._error(f"the answer was blocked ({finish})", "refused")
        if finish == "MAX_TOKENS":
            raise self._error("the answer was cut off at the token limit", "truncated")
        content = candidate.get("content")
        parts = content.get("parts") if isinstance(content, dict) else None
        texts = (
            [
                part["text"]
                for part in parts
                if isinstance(part, dict)
                and isinstance(part.get("text"), str)
                and not part.get("thought")
            ]
            if isinstance(parts, list)
            else []
        )
        if not texts:
            raise self._error("the answer has no text", "invalid_response")
        usage = data.get("usageMetadata") if isinstance(data.get("usageMetadata"), dict) else {}
        assert isinstance(usage, dict)
        output_tokens = int(usage.get("candidatesTokenCount") or 0) + int(
            usage.get("thoughtsTokenCount") or 0
        )
        return (
            "".join(texts),
            str(data.get("modelVersion") or self._model),
            Usage(
                input_tokens=int(usage.get("promptTokenCount") or 0), output_tokens=output_tokens
            ),
        )
