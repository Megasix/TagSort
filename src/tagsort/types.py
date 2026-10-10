"""Public result types.

They mirror ``schemas/result.v1.json`` field for field. Every instance is validated when it
is built, so :meth:`ReadResult.to_json` always produces a document that conforms to the
schema.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import pairwise
from types import MappingProxyType
from typing import Any, ClassVar, Literal, get_args

__all__ = ["Candidate", "ImageInfo", "Point", "ReadResult", "Tag", "TagSource", "TagStatus"]

TagStatus = Literal["accepted", "review", "unreadable"]
"""``accepted``: confident reading. ``review``: a person should check it.
``unreadable``: no reading."""

TagSource = Literal["local", "fallback"]
"""``local``: read by the on-device models. ``fallback``: read by the vision API fallback."""

Point = tuple[float, float]
"""``(x, y)`` in pixels of the original image after EXIF rotation."""

_STATUSES = frozenset(get_args(TagStatus))
_SOURCES = frozenset(get_args(TagSource))
_ROTATIONS = frozenset({0, 90, 180, 270})


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _check_number(name: str, value: object, low: float, high: float | None = None) -> None:
    """Check ``low <= value <= high`` (or ``low <= value`` when ``high`` is None)."""
    if not _is_number(value):
        raise TypeError(f"{name} must be a number, not {type(value).__name__}")
    assert isinstance(value, int | float)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, not {value}")
    if value < low or (high is not None and value > high):
        bounds = f"between {low} and {high}" if high is not None else f"at least {low}"
        raise ValueError(f"{name} must be {bounds}, not {value}")


def _check_text(name: str, value: object, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str):
        expected = "a str or None" if optional else "a str"
        raise TypeError(f"{name} must be {expected}, not {type(value).__name__}")
    if not value:
        raise ValueError(f"{name} must not be empty")


@dataclass(frozen=True)
class ImageInfo:
    """Size and orientation of the image that was read.

    Attributes:
        width: Width in pixels, after EXIF rotation.
        height: Height in pixels, after EXIF rotation.
        exif_rotation: Clockwise rotation in degrees (0, 90, 180 or 270) applied to the
            stored pixels to honor the EXIF orientation.
    """

    width: int
    height: int
    exif_rotation: int = 0

    def __post_init__(self) -> None:
        """Validate the fields."""
        for name in ("width", "height"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an int, not {type(value).__name__}")
            if value < 1:
                raise ValueError(f"{name} must be at least 1, not {value}")
        if self.exif_rotation not in _ROTATIONS or isinstance(self.exif_rotation, bool):
            raise ValueError(f"exif_rotation must be 0, 90, 180 or 270, not {self.exif_rotation}")

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-ready form of this object."""
        return {"width": self.width, "height": self.height, "exif_rotation": self.exif_rotation}


@dataclass(frozen=True)
class Candidate:
    """An alternative reading of a tag.

    Attributes:
        text: The alternative text.
        confidence: Calibrated probability that this text is exactly right, in [0, 1].
    """

    text: str
    confidence: float

    def __post_init__(self) -> None:
        """Validate the fields."""
        _check_text("text", self.text)
        _check_number("confidence", self.confidence, 0, 1)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-ready form of this object."""
        return {"text": self.text, "confidence": self.confidence}


@dataclass(frozen=True)
class Tag:
    """One tag found in an image.

    Attributes:
        tag_id: Id of the profile tag whose pattern the text matches, or ``None`` when it
            matches none. Never ``None`` for an accepted tag.
        text: Text read on the tag, or ``None`` exactly when ``status`` is ``unreadable``.
        confidence: Calibrated probability that ``text`` is exactly right, in [0, 1].
        status: ``accepted``, ``review`` or ``unreadable``.
        polygon: The four corners of the tag, clockwise, starting at the top-left corner
            of the upright text.
        angle: Orientation of the text: clockwise rotation in degrees from upright, in
            [0, 360). 0 reads left to right, 90 top to bottom, 180 upside down.
        candidates: Other possible readings, most likely first.
        source: ``local`` or ``fallback``.
    """

    tag_id: str | None
    text: str | None
    confidence: float
    status: TagStatus
    polygon: tuple[Point, Point, Point, Point]
    angle: float
    candidates: tuple[Candidate, ...] = ()
    source: TagSource = "local"

    def __post_init__(self) -> None:
        """Validate the fields and normalize sequences to tuples."""
        _check_text("tag_id", self.tag_id, optional=True)
        _check_text("text", self.text, optional=True)
        _check_number("confidence", self.confidence, 0, 1)
        if self.status not in _STATUSES:
            raise ValueError(f"status must be one of {sorted(_STATUSES)}, not {self.status!r}")
        if self.source not in _SOURCES:
            raise ValueError(f"source must be one of {sorted(_SOURCES)}, not {self.source!r}")
        if (self.text is None) != (self.status == "unreadable"):
            raise ValueError("text must be None exactly when status is 'unreadable'")
        if self.status == "accepted" and self.tag_id is None:
            raise ValueError("an accepted tag must have a tag_id")

        polygon = tuple(tuple(point) for point in self.polygon)
        if len(polygon) != 4 or any(len(point) != 2 for point in polygon):
            raise ValueError("polygon must have exactly 4 points of 2 coordinates each")
        for point in polygon:
            for coordinate in point:
                _check_number("polygon coordinate", coordinate, -math.inf)
        object.__setattr__(self, "polygon", polygon)

        _check_number("angle", self.angle, 0)
        if self.angle >= 360:
            raise ValueError(f"angle must be below 360, not {self.angle}")

        candidates = tuple(self.candidates)
        for candidate in candidates:
            if not isinstance(candidate, Candidate):
                raise TypeError(f"candidates must be Candidate objects, not {candidate!r}")
        if any(a.confidence < b.confidence for a, b in pairwise(candidates)):
            raise ValueError("candidates must be sorted by decreasing confidence")
        object.__setattr__(self, "candidates", candidates)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-ready form of this object."""
        return {
            "tag_id": self.tag_id,
            "text": self.text,
            "confidence": self.confidence,
            "status": self.status,
            "polygon": [list(point) for point in self.polygon],
            "angle": self.angle,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "source": self.source,
        }


TextKind = Literal["id", "header", "noise"]
_TEXT_KINDS = frozenset(get_args(TextKind))


@dataclass(frozen=True)
class OtherText:
    """A line of text that fits no kind of tag of the profile (open reading).

    Attributes:
        text: What the recognizer read, without the help of any pattern.
        confidence: How sure the recognizer is of the characters, in [0, 1].
        tag_likelihood: How likely the text is a specimen tag, in [0, 1], from generic
            cues (see ``docs/open-reading.md``); not a calibrated probability yet.
        kind_guess: ``id`` (looks like a specimen number), ``header`` (printed words) or
            ``noise`` (rulers, stray marks).
        polygon: The four corners of the line, clockwise from the top-left corner of the
            upright text.
        angle: Orientation of the text, as for :class:`Tag`.
    """

    text: str
    confidence: float
    tag_likelihood: float
    kind_guess: TextKind
    polygon: tuple[Point, Point, Point, Point]
    angle: float

    def __post_init__(self) -> None:
        """Validate the fields."""
        _check_text("text", self.text)
        _check_number("confidence", self.confidence, 0, 1)
        _check_number("tag_likelihood", self.tag_likelihood, 0, 1)
        if self.kind_guess not in _TEXT_KINDS:
            raise ValueError(f"kind_guess must be one of {sorted(_TEXT_KINDS)}")
        polygon = tuple(tuple(point) for point in self.polygon)
        if len(polygon) != 4 or any(len(point) != 2 for point in polygon):
            raise ValueError("polygon must have exactly 4 points of 2 coordinates each")
        object.__setattr__(self, "polygon", polygon)
        _check_number("angle", self.angle, 0)
        if self.angle >= 360:
            raise ValueError(f"angle must be below 360, not {self.angle}")

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-ready form of this object."""
        return {
            "text": self.text,
            "confidence": self.confidence,
            "tag_likelihood": self.tag_likelihood,
            "kind_guess": self.kind_guess,
            "polygon": [list(point) for point in self.polygon],
            "angle": self.angle,
        }


@dataclass(frozen=True)
class ApiUsage:
    """What reading one image cost at a vision API, for billing.

    Attributes:
        calls: Requests the provider answered.
        input_tokens: Input tokens billed, over all those requests.
        output_tokens: Output tokens billed (thinking tokens included).
    """

    calls: int
    input_tokens: int = 0
    output_tokens: int = 0

    def __post_init__(self) -> None:
        """Check the counts are non-negative integers."""
        for name in ("calls", "input_tokens", "output_tokens"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer, not {value!r}")

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-ready form, as in ``result.v1.json``."""
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


@dataclass(frozen=True)
class ReadResult:
    """Everything TagSort found in one image.

    Attributes:
        engine_version: Version of the TagSort engine that produced the result.
        model_version: Versions of the models used, for example ``det-1.0+rec-1.0``.
        image: Size and orientation of the image.
        tags: Tags found in the image, possibly none.
        timings_ms: Time spent in each pipeline stage, in milliseconds.
        other_texts: With open reading, the lines that fit no kind of tag, most likely
            tags first; empty otherwise (and then absent from the JSON form).
        api_usage: Requests and tokens billed by a vision API for this image, when one
            was asked (the API backend, or the fallback); ``None`` otherwise (and then
            absent from the JSON form).
    """

    SCHEMA_VERSION: ClassVar[str] = "1.0"
    """Version of ``result.v1.json`` this class produces."""

    engine_version: str
    model_version: str
    image: ImageInfo
    tags: tuple[Tag, ...] = ()
    timings_ms: Mapping[str, float] = field(default_factory=dict)
    other_texts: tuple[OtherText, ...] = ()
    api_usage: ApiUsage | None = None

    def __post_init__(self) -> None:
        """Validate the fields and freeze the collections."""
        _check_text("engine_version", self.engine_version)
        _check_text("model_version", self.model_version)
        if not isinstance(self.image, ImageInfo):
            raise TypeError(f"image must be an ImageInfo, not {type(self.image).__name__}")
        tags = tuple(self.tags)
        for tag in tags:
            if not isinstance(tag, Tag):
                raise TypeError(f"tags must be Tag objects, not {tag!r}")
        object.__setattr__(self, "tags", tags)
        timings = dict(self.timings_ms)
        for stage, value in timings.items():
            _check_text("timings_ms key", stage)
            _check_number(f"timings_ms[{stage!r}]", value, 0)
        object.__setattr__(self, "timings_ms", MappingProxyType(timings))
        others = tuple(self.other_texts)
        for other in others:
            if not isinstance(other, OtherText):
                raise TypeError(f"other_texts must be OtherText objects, not {other!r}")
        object.__setattr__(self, "other_texts", others)
        if self.api_usage is not None and not isinstance(self.api_usage, ApiUsage):
            raise TypeError(f"api_usage must be an ApiUsage, not {self.api_usage!r}")

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-ready form of this result, conforming to ``result.v1.json``."""
        return {
            "schema_version": self.SCHEMA_VERSION,
            "engine_version": self.engine_version,
            "model_version": self.model_version,
            "image": self.image.to_dict(),
            "tags": [tag.to_dict() for tag in self.tags],
            "timings_ms": dict(self.timings_ms),
            **(
                {"other_texts": [other.to_dict() for other in self.other_texts]}
                if self.other_texts
                else {}
            ),
            **({"api_usage": self.api_usage.to_dict()} if self.api_usage is not None else {}),
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Return this result as a JSON document conforming to ``result.v1.json``.

        Args:
            indent: Indentation for pretty-printing, or ``None`` for a compact document.
        """
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, allow_nan=False)
