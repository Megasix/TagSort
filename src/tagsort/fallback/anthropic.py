"""Claude, through the Anthropic Messages API."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tagsort.fallback._http import HttpProvider
from tagsort.fallback.base import BoxFormat, Usage, VisionRequest

__all__ = ["AnthropicProvider"]

# Model families that read images up to 2576 px with 1:1 pixel coordinates.
_HIGH_RES = (
    "claude-fable",
    "claude-mythos",
    "claude-opus-5",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-sonnet-5",
)

# Models that accept server-side refusal fallbacks with the "default" routing.
_FALLBACK_MODELS = frozenset({"claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"})


class AnthropicProvider(HttpProvider):
    """Reads tags with Claude through ``POST /v1/messages``.

    Example:
        >>> provider = AnthropicProvider(api_key="sk-ant-...")  # doctest: +SKIP
    """

    name = "anthropic"
    box_format: BoxFormat = "pixels_xyxy"
    default_model = "claude-opus-5-5"
    # Recent models read up to 2576 px on the long edge and report pixel coordinates 1:1
    # with the image sent; older ones (Haiku 4.5, Opus 4.6...) downscale beyond 1568 px.
    default_max_side = 2576
    legacy_max_side = 1568
    url = "https://api.anthropic.com/v1/messages"

    @property
    def max_side(self) -> int:
        """Longest image edge sent: 2576 px for high-resolution models, else 1568 px."""
        if self._max_side != self.default_max_side or self._model.startswith(_HIGH_RES):
            return self._max_side
        return self.legacy_max_side

    def _build(
        self, request: VisionRequest, image_base64: str
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        headers = {"x-api-key": self._api_key, "anthropic-version": "2023-06-01"}
        body: dict[str, Any] = {
            "model": self._model,
            "max_tokens": 16000,
            "output_config": {"format": {"type": "json_schema", "schema": request.schema}},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_base64,
                            },
                        },
                        {"type": "text", "text": request.instructions},
                    ],
                }
            ],
        }
        if self._model in _FALLBACK_MODELS:
            # On a safety decline, the API retries on Anthropic's recommended model.
            headers["anthropic-beta"] = "server-side-fallback-2026-07-01"
            body["fallbacks"] = "default"
        return self.url, headers, body

    def _extract(self, data: Mapping[str, Any]) -> tuple[str, str, Usage]:
        stop_reason = data.get("stop_reason")
        if stop_reason == "refusal":
            raise self._error("the model declined to read this image", "refused")
        if stop_reason == "max_tokens":
            raise self._error("the answer was cut off at max_tokens", "truncated")
        content = data.get("content")
        texts = (
            [
                block.get("text")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            if isinstance(content, list)
            else []
        )
        if not texts or not isinstance(texts[0], str):
            raise self._error("the answer has no text block", "invalid_response")
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
