"""Instructions, answer schema and answer parsing shared by every provider."""

from __future__ import annotations

import json
import math

from tagsort.errors import ProviderError
from tagsort.fallback.base import BoxFormat, Legibility, ProviderTag
from tagsort.profile import Profile

__all__ = ["ANSWER_SCHEMA", "build_instructions", "parse_answer"]

_LEGIBILITY: tuple[Legibility, ...] = ("certain", "uncertain", "unreadable")
_ANGLES = ("0", "90", "180", "270")
MAX_ALTERNATIVES = 3

# Kept to the JSON Schema subset every provider's structured output accepts: no numeric or
# length constraints, closed objects, every property required. Counts and ranges are
# checked by parse_answer instead.
ANSWER_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["tags"],
    "properties": {
        "tags": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "legibility", "alternatives", "box", "angle"],
                "properties": {
                    "text": {"type": "string"},
                    "legibility": {"type": "string", "enum": list(_LEGIBILITY)},
                    "alternatives": {"type": "array", "items": {"type": "string"}},
                    "box": {"type": "array", "items": {"type": "number"}},
                    "angle": {"type": "string", "enum": list(_ANGLES)},
                },
            },
        }
    },
}

_BOX_HELP: dict[BoxFormat, str] = {
    "pixels_xyxy": (
        "box: [x_min, y_min, x_max, y_max] of the tag, in pixels of this image "
        "({width} x {height}), origin at the top-left corner."
    ),
    "normalized_xyxy": (
        "box: [x_min, y_min, x_max, y_max] of the tag, normalized to 0-1000, origin at "
        "the top-left corner."
    ),
    "normalized_yxyx": ("box: [y_min, x_min, y_max, x_max] of the tag, normalized to 0-1000."),
}


def build_instructions(profile: Profile, *, box_format: BoxFormat, width: int, height: int) -> str:
    """Return the instructions sent with every image for ``profile``."""
    patterns = "\n".join(f"- {tag.id}: {tag.pattern}" for tag in profile.tags)
    box_help = _BOX_HELP[box_format].format(width=width, height=height)
    return f"""\
This photo shows a natural history specimen with identification tags: small labels, \
handwritten or printed, in any orientation. Find every tag and read it.

For each tag, report:
- text: the text exactly as written, keeping case, spaces and punctuation. Never correct, \
complete or guess characters you cannot see. Empty string if the tag cannot be read.
- legibility: "certain" if every character is clearly readable, "uncertain" if any \
character could be read another way, "unreadable" if the text cannot be read.
- alternatives: other plausible readings, most likely first, at most {MAX_ALTERNATIVES}. \
Empty list if there are none.
- {box_help}
- angle: orientation of the text, clockwise from upright. "0" reads left to right, \
"90" reads top to bottom, "180" is upside down, "270" reads bottom to top.

In this collection, tag texts follow these regular expressions:
{patterns}
Use them to choose between similar-looking characters, such as 0 and O or 1 and I. If a \
tag does not match any of them, report it as written and mark it "uncertain".

Report only identification tags, not other text in the photo such as rulers or color \
charts. If there is no tag, return an empty list."""


def parse_answer(
    text: str, *, provider: str, box_format: BoxFormat, width: int, height: int
) -> tuple[ProviderTag, ...]:
    """Validate a provider's JSON answer and convert boxes to pixels of the image sent.

    Raises:
        ProviderError: With reason ``invalid_response`` if the answer does not follow
            :data:`ANSWER_SCHEMA` or has impossible values.
    """
    try:
        data = json.loads(text)
        raw_tags = data["tags"]
        if not isinstance(raw_tags, list):
            raise TypeError("tags is not a list")
        return tuple(
            _parse_tag(raw, box_format=box_format, width=width, height=height) for raw in raw_tags
        )
    except (ValueError, KeyError, TypeError) as error:
        raise ProviderError(
            f"answer does not follow the expected schema: {error}",
            provider=provider,
            reason="invalid_response",
        ) from None


def _parse_tag(raw: object, *, box_format: BoxFormat, width: int, height: int) -> ProviderTag:
    if not isinstance(raw, dict) or set(raw) != {
        "text",
        "legibility",
        "alternatives",
        "box",
        "angle",
    }:
        raise ValueError(f"unexpected tag object {raw!r}")
    text = raw["text"]
    if not isinstance(text, str):
        raise TypeError("text is not a string")
    legibility = raw["legibility"]
    if legibility not in _LEGIBILITY:
        raise ValueError(f"unknown legibility {legibility!r}")
    if raw["angle"] not in _ANGLES:
        raise ValueError(f"unknown angle {raw['angle']!r}")
    alternatives = raw["alternatives"]
    if not isinstance(alternatives, list) or not all(isinstance(a, str) for a in alternatives):
        raise TypeError("alternatives is not a list of strings")
    text = text.strip()
    if not text:
        legibility = "unreadable"
    seen = {text}
    kept: list[str] = []
    for alternative in (a.strip() for a in alternatives):
        if alternative and alternative not in seen:
            seen.add(alternative)
            kept.append(alternative)
    return ProviderTag(
        text=text,
        legibility=legibility,
        alternatives=tuple(kept[:MAX_ALTERNATIVES]),
        box=_parse_box(raw["box"], box_format=box_format, width=width, height=height),
        angle=int(raw["angle"]),
    )


def _parse_box(
    raw: object, *, box_format: BoxFormat, width: int, height: int
) -> tuple[float, float, float, float]:
    if not isinstance(raw, list) or len(raw) != 4:
        raise ValueError(f"box must have 4 numbers, not {raw!r}")
    numbers: list[float] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError(f"box value {value!r} is not a number")
        if not math.isfinite(value):
            raise ValueError(f"box value {value!r} is not finite")
        numbers.append(float(value))
    if box_format == "normalized_yxyx":
        y0, x0, y1, x1 = numbers
    else:
        x0, y0, x1, y1 = numbers
    if box_format != "pixels_xyxy":
        x0, x1 = x0 * width / 1000, x1 * width / 1000
        y0, y1 = y0 * height / 1000, y1 * height / 1000
    x0, x1 = sorted((min(max(x0, 0.0), width), min(max(x1, 0.0), width)))
    y0, y1 = sorted((min(max(y0, 0.0), height), min(max(y1, 0.0), height)))
    if x1 - x0 < 1 or y1 - y0 < 1:
        raise ValueError(f"box {raw!r} is empty or outside the image")
    return (x0, y0, x1, y1)
