"""Open reading: how likely a line of text is a specimen tag, from generic cues.

With open reading, TagSort also returns the lines that fit no kind of tag of the profile,
so applications can show them and learn which ones matter. Each line gets a likelihood of
being a specimen tag and a guess of what it is. The cues are deliberately generic, true of
specimen tags in most collections, and are meant to be replaced by a trained selector:

- specimen numbers hold digits, are short (3 to 16 characters) and have few words;
- a few capital letters before the digits (``GJ``, ``NMC``) and a dash or slash inside
  (``15950-1``, ``12/345``) are common;
- printed headers are words without digits (``NATIONAL MUSEUM OF CANADA``);
- rulers and scales are rows of short numbers (``1 2 3 4 5``), and single characters are
  stray marks.
"""

from __future__ import annotations

import re

from tagsort.types import TextKind

__all__ = ["tag_likelihood"]

_PREFIXED = re.compile(r"^[A-Z]{1,6}[-. /]?\d")
_INNER_SEPARATOR = re.compile(r"\d[-/.]\d")


def tag_likelihood(text: str) -> tuple[float, TextKind]:
    """Return how likely ``text`` is a specimen tag, in [0, 1], and what it looks like."""
    value = " ".join(text.upper().split())
    compact = value.replace(" ", "")
    digits = sum(char.isdigit() for char in compact)
    letters = sum(char.isalpha() for char in compact)
    words = value.split(" ") if value else []
    if len(compact) < 2:
        return 0.02, "noise"
    if digits == 0:
        # Words without digits: a printed header when there is enough of it.
        return (0.1, "header") if letters >= 4 else (0.05, "noise")
    if len(words) >= 3 and all(word.isdigit() and len(word) <= 2 for word in words):
        return 0.05, "noise"  # a ruler or a scale
    score = 0.55
    if 3 <= len(compact) <= 16:
        score += 0.15
    else:
        score -= 0.3
    if len(words) > 3:
        score -= 0.3
    if _PREFIXED.match(compact):
        score += 0.15
    if _INNER_SEPARATOR.search(compact):
        score += 0.1
    if digits < 2:
        score -= 0.25
    if letters > digits * 2:
        score -= 0.2
    score = round(min(1.0, max(0.0, score)), 2)
    return score, "id" if score >= 0.5 else "noise"
