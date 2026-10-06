"""HTTP plumbing shared by the vision API providers."""

from __future__ import annotations

import base64
import json
import logging
import random
import time
from collections.abc import Callable, Mapping
from typing import Any

import httpx

from tagsort.errors import ProviderError
from tagsort.fallback.base import BoxFormat, ProviderTag, Usage, VisionAnswer, VisionRequest
from tagsort.fallback.prompt import parse_answer

__all__ = ["HttpProvider"]

logger = logging.getLogger("tagsort.fallback")

_RETRY_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504, 529})
_MAX_BACKOFF = 60.0


class HttpProvider:
    """Base class for providers reached over HTTPS with an API key.

    Subclasses implement :meth:`_build` and :meth:`_extract`.
    """

    name = "http"
    box_format: BoxFormat = "pixels_xyxy"
    default_model = ""
    default_max_side = 2048

    def __init__(
        self,
        *,
        api_key: str,
        model: str | None = None,
        max_side: int | None = None,
        timeout: float = 120.0,
        max_retries: int = 2,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Create the provider.

        Args:
            api_key: The caller's API key for this provider. It is only sent to the
                provider's endpoint and never logged.
            model: Model to request; defaults to :attr:`default_model`.
            max_side: Longest image edge to send, in pixels.
            timeout: Seconds to wait for each HTTP request.
            max_retries: Retries after rate limits, server errors and network errors.
            client: HTTP client to use, for example with a proxy. Created if omitted.
            sleep: Function used to wait between retries.
        """
        if not api_key or not isinstance(api_key, str):
            raise ValueError(f"{self.name}: api_key must be a non-empty string")
        if max_retries < 0:
            raise ValueError("max_retries must be at least 0")
        self._api_key = api_key
        self._model = model or self.default_model
        self._max_side = max_side or self.default_max_side
        self._max_retries = max_retries
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=timeout)
        self._sleep = sleep

    def __repr__(self) -> str:
        """Describe the provider without revealing the API key."""
        return f"{type(self).__name__}(model={self._model!r})"

    @property
    def model(self) -> str:
        """Model requested from the provider."""
        return self._model

    @property
    def max_side(self) -> int:
        """Longest image edge, in pixels, sent to the provider."""
        return self._max_side

    def close(self) -> None:
        """Close the HTTP client if this provider created it."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> HttpProvider:
        """Return the provider; :meth:`close` is called on exit."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the provider."""
        self.close()

    def read(self, request: VisionRequest) -> VisionAnswer:
        """Send ``request`` and return the validated answer.

        Raises:
            ProviderError: If the provider fails or its answer is unusable.
        """
        url, headers, body = self._build(request, base64.b64encode(request.jpeg).decode("ascii"))
        data = self._post(url, headers, body)
        text, model, usage = self._extract(data)
        tags: tuple[ProviderTag, ...] = parse_answer(
            text,
            provider=self.name,
            box_format=self.box_format,
            width=request.width,
            height=request.height,
        )
        return VisionAnswer(tags=tags, model=model, usage=usage)

    # Subclass hooks

    def _build(
        self, request: VisionRequest, image_base64: str
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        """Return the URL, headers and JSON body of the request."""
        raise NotImplementedError

    def _extract(self, data: Mapping[str, Any]) -> tuple[str, str, Usage]:
        """Return the answer text, the model that answered and the usage.

        Raises:
            ProviderError: If the provider refused, was cut off or answered oddly.
        """
        raise NotImplementedError

    # Helpers

    def _error(self, message: str, reason: str, status: int | None = None) -> ProviderError:
        return ProviderError(message, provider=self.name, reason=reason, status_code=status)

    def _post(self, url: str, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
        attempt = 0
        while True:
            try:
                response = self._client.post(url, headers=headers, json=body)
            except httpx.TransportError as error:
                failure = self._error(f"network error: {type(error).__name__}", "network")
                delay = None
            else:
                if response.is_success:
                    return self._decode(response)
                failure = self._status_error(response)
                delay = _retry_after(response)
            if not failure.retryable or attempt >= self._max_retries:
                raise failure
            delay = delay if delay is not None else min(2.0**attempt + random.random(), 30.0)
            logger.warning("%s; retrying in %.1f s", failure, delay)
            self._sleep(min(delay, _MAX_BACKOFF))
            attempt += 1

    def _decode(self, response: httpx.Response) -> dict[str, Any]:
        try:
            data = response.json()
        except json.JSONDecodeError:
            raise self._error("response is not JSON", "invalid_response") from None
        if not isinstance(data, dict):
            raise self._error("response is not a JSON object", "invalid_response")
        return data

    def _status_error(self, response: httpx.Response) -> ProviderError:
        status = response.status_code
        detail = _error_detail(response)
        if status in (401, 403):
            reason = "authentication"
        elif status == 429:
            reason = "rate_limit"
        elif status in _RETRY_STATUSES or status >= 500:
            reason = "server"
        else:
            reason = "bad_request"
        return self._error(f"HTTP {status}: {detail}", reason, status)


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(float(value), 0.0)
    except ValueError:
        return None


def _error_detail(response: httpx.Response) -> str:
    """Extract the provider's error message, short and without echoing request data."""
    try:
        data = response.json()
    except json.JSONDecodeError:
        return response.reason_phrase or "error"
    error = data.get("error") if isinstance(data, dict) else None
    message = error.get("message") if isinstance(error, dict) else None
    if not isinstance(message, str):
        return response.reason_phrase or "error"
    return message[:300]
