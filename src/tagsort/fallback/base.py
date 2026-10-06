"""Interface between the reader and vision API providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

__all__ = [
    "BoxFormat",
    "Legibility",
    "ProviderTag",
    "Usage",
    "VisionAnswer",
    "VisionProvider",
    "VisionRequest",
]

Legibility = Literal["certain", "uncertain", "unreadable"]

BoxFormat = Literal["pixels_xyxy", "normalized_xyxy", "normalized_yxyx"]
"""How a provider reports boxes.

``pixels_xyxy``: ``[x_min, y_min, x_max, y_max]`` in pixels of the image sent.
``normalized_xyxy``: ``[x_min, y_min, x_max, y_max]`` scaled to 0-1000.
``normalized_yxyx``: ``[y_min, x_min, y_max, x_max]`` scaled to 0-1000.
"""


@dataclass(frozen=True)
class VisionRequest:
    """One image to read, with the instructions and the answer schema.

    Attributes:
        jpeg: The image, JPEG-encoded, without metadata.
        width: Width of the encoded image, in pixels.
        height: Height of the encoded image, in pixels.
        instructions: What to read and how to answer.
        schema: JSON Schema the answer must follow.
    """

    jpeg: bytes = field(repr=False)
    width: int
    height: int
    instructions: str
    schema: dict[str, object] = field(repr=False)


@dataclass(frozen=True)
class ProviderTag:
    """One tag as a provider reported it, before validation against the profile.

    Attributes:
        text: Text as written on the tag; empty when unreadable.
        legibility: How sure the provider is of ``text``.
        alternatives: Other plausible readings, most likely first.
        box: ``(x_min, y_min, x_max, y_max)`` in pixels of the image sent.
        angle: Orientation of the text, clockwise from upright: 0, 90, 180 or 270.
    """

    text: str
    legibility: Legibility
    alternatives: tuple[str, ...]
    box: tuple[float, float, float, float]
    angle: int


@dataclass(frozen=True)
class Usage:
    """Tokens billed for one request."""

    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class VisionAnswer:
    """What a provider read in one image.

    Attributes:
        tags: Tags found, possibly none.
        model: Exact model that answered.
        usage: Tokens billed.
    """

    tags: tuple[ProviderTag, ...]
    model: str
    usage: Usage = Usage()


@runtime_checkable
class VisionProvider(Protocol):
    """A vision API that finds and reads tags in an image.

    Implementations must be safe to call from several threads, must only send the
    request's image and instructions, and must raise
    :class:`~tagsort.errors.ProviderError` rather than return a partial answer.
    """

    @property
    def name(self) -> str:
        """Short provider name, for example ``anthropic``."""
        ...

    @property
    def model(self) -> str:
        """Model requested from the provider."""
        ...

    @property
    def max_side(self) -> int:
        """Longest image edge, in pixels, worth sending to this provider."""
        ...

    @property
    def box_format(self) -> BoxFormat:
        """How this provider is asked to report boxes."""
        ...

    def read(self, request: VisionRequest) -> VisionAnswer:
        """Send ``request`` and return the validated answer.

        Raises:
            ProviderError: If the provider fails or its answer is unusable.
        """
        ...
