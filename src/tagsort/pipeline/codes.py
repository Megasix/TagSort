"""Barcodes and 2D codes printed on tags: decoded exactly, before any text is read.

A code is not read, it is decoded: its own checksums make the text certain. A code whose
text (or a part of it, such as the last segment of a URL) fits the profile becomes an
accepted tag with source ``barcode``; others are returned as other texts when open
reading is on. Decoding uses zxing-cpp (Apache 2.0): 1D barcodes, QR, Data Matrix, Aztec
and PDF417.
"""

from __future__ import annotations

import importlib.util
import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from PIL import Image

from tagsort.profile import Profile

__all__ = ["CodeDecoder", "DecodedCode", "ZxingDecoder", "fit_code", "zxing_installed"]


def zxing_installed() -> bool:
    """Whether the ``codes`` extra (zxing-cpp) is installed."""
    return importlib.util.find_spec("zxingcpp") is not None


Quad = tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]


@dataclass(frozen=True)
class DecodedCode:
    """One code found in an image.

    Attributes:
        text: The decoded content, as text.
        symbology: The kind of code, such as ``QR Code`` or ``Code 128``.
        quad: Corners in pixels of the decoded image, clockwise from the top-left corner
            of the upright code.
        angle: Clockwise rotation of the code from upright, in degrees.
    """

    text: str
    symbology: str
    quad: Quad
    angle: float


@runtime_checkable
class CodeDecoder(Protocol):
    """Finds and decodes the codes in an image."""

    def decode(self, image: Image.Image) -> list[DecodedCode]:
        """Return every code found in ``image``, in no particular order."""
        ...


class ZxingDecoder:
    """Decodes codes with zxing-cpp."""

    def decode(self, image: Image.Image) -> list[DecodedCode]:
        """Return every valid code found in ``image``."""
        import zxingcpp

        codes = []
        for found in zxingcpp.read_barcodes(image.convert("L")):
            text = found.text.strip()
            if not found.valid or not text:
                continue
            p = found.position
            corners = (p.top_left, p.top_right, p.bottom_right, p.bottom_left)
            quad: Quad = (
                (float(corners[0].x), float(corners[0].y)),
                (float(corners[1].x), float(corners[1].y)),
                (float(corners[2].x), float(corners[2].y)),
                (float(corners[3].x), float(corners[3].y)),
            )
            codes.append(
                DecodedCode(
                    text=text,
                    symbology=str(found.format),
                    quad=quad,
                    # Clockwise from upright, as TagSort angles.
                    angle=float(found.orientation) % 360,
                )
            )
        return codes


_SEPARATORS = re.compile(r"[^A-Za-z0-9._-]+")


def fit_code(text: str, profile: Profile) -> list[tuple[str, str]]:
    """Return the readings of a code that fit the profile, each with its tag id.

    The whole content is tried first, then without spaces, then each of its parts (a QR
    code often holds a URL ending with the catalog number). Parts in lower case are also
    tried in upper case. More than one reading means a person has to choose.
    """
    for whole in (text, "".join(text.split())):
        tag_id = profile.match(whole)
        if tag_id is not None:
            return [(whole, tag_id)]
    fits: dict[str, str] = {}
    for part in _SEPARATORS.split(text):
        for variant in dict.fromkeys((part, part.upper())):
            tag_id = profile.match(variant) if variant else None
            if tag_id is not None:
                fits.setdefault(variant, tag_id)
                break
    return list(fits.items())
