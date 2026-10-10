"""TagSort: read specimen ID tags in photos.

Only names exported from this module are part of the public API. The vision API
providers (``AnthropicProvider``, ``DeepSeekProvider``, ``GeminiProvider``,
``OpenAIProvider``) need the ``api`` extra: ``pip install "tagsort[api]"``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from tagsort._version import __version__
from tagsort.errors import ImageError, PatternError, ProfileError, ProviderError, TagSortError
from tagsort.fallback.base import ProviderTag, Usage, VisionAnswer, VisionProvider, VisionRequest
from tagsort.models import ModelError, available_models, download_model
from tagsort.pipeline.local import LocalPipeline
from tagsort.pipeline.reader import Reader
from tagsort.profile import Profile, TagSpec
from tagsort.types import (
    ApiUsage,
    Candidate,
    ImageInfo,
    OtherText,
    Point,
    ReadResult,
    Tag,
    TagSource,
    TagStatus,
)

if TYPE_CHECKING:
    from tagsort.fallback.anthropic import AnthropicProvider
    from tagsort.fallback.deepseek import DeepSeekProvider
    from tagsort.fallback.gemini import GeminiProvider
    from tagsort.fallback.openai import OpenAIProvider

__all__ = [
    "AnthropicProvider",
    "ApiUsage",
    "Candidate",
    "DeepSeekProvider",
    "GeminiProvider",
    "ImageError",
    "ImageInfo",
    "LocalPipeline",
    "ModelError",
    "OpenAIProvider",
    "OtherText",
    "PatternError",
    "Point",
    "Profile",
    "ProfileError",
    "ProviderError",
    "ProviderTag",
    "ReadResult",
    "Reader",
    "Tag",
    "TagSortError",
    "TagSource",
    "TagSpec",
    "TagStatus",
    "Usage",
    "VisionAnswer",
    "VisionProvider",
    "VisionRequest",
    "__version__",
    "available_models",
    "download_model",
]

logging.getLogger("tagsort").addHandler(logging.NullHandler())

_PROVIDERS = {
    "AnthropicProvider": "tagsort.fallback.anthropic",
    "DeepSeekProvider": "tagsort.fallback.deepseek",
    "GeminiProvider": "tagsort.fallback.gemini",
    "OpenAIProvider": "tagsort.fallback.openai",
}


def __getattr__(name: str) -> Any:  # noqa: ANN401
    """Import provider classes on first use, so the core does not need httpx."""
    if name not in _PROVIDERS:
        raise AttributeError(f"module 'tagsort' has no attribute {name!r}")
    from importlib import import_module

    try:
        module = import_module(_PROVIDERS[name])
    except ModuleNotFoundError as error:
        if error.name != "httpx":
            raise
        raise ImportError(f"{name} needs the 'api' extra: pip install \"tagsort[api]\"") from None
    return getattr(module, name)
