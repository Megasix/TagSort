"""Printed headers: choosing the kind of tag from text printed on it.

Many collections use pre-printed tags: a printed header (``NATIONAL MUSEUM OF CANADA``)
and a number written by hand. The number alone may fit several kinds of tag; the header
says which. A profile gives a kind its ``header``; when a reading fits several kinds, the
kind whose header is seen in the photo is reported. See ``docs/headers.md``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from difflib import SequenceMatcher

from tagsort.profile import Profile

__all__ = ["HEADER_SIMILARITY", "choose_kind", "headers_seen", "normalize"]

HEADER_SIMILARITY = 0.8
"""How close a stretch of the photo's text must be to a header (0 to 1) to count as seen.
Recognition errors are common on printed capitals (``0`` for ``O``, a missed letter)."""


def normalize(text: str) -> str:
    """Upper-case letters and digits only: spaces, punctuation and line breaks do not count."""
    return "".join(char for char in text.upper() if char.isalnum())


def _contains(text: str, header: str, threshold: float) -> bool:
    if not header:
        return False
    if header in text:
        return True
    size = len(header)
    if len(text) < size * threshold:
        return False
    # Compare the header with every stretch of the text of about its length.
    best = 0.0
    for start in range(max(1, len(text) - size + 1)):
        window = text[start : start + size]
        best = max(best, SequenceMatcher(None, header, window, autojunk=False).ratio())
        if best >= threshold:
            return True
    return False


def headers_seen(
    profile: Profile, lines: Iterable[str], threshold: float = HEADER_SIMILARITY
) -> frozenset[str]:
    """Return the ids of the kinds whose header appears in the text read on the photo.

    ``lines`` is the unconstrained reading of every text line, in reading order; a header
    may be split over several lines.
    """
    text = normalize("".join(lines))
    return frozenset(
        tag.id
        for tag in profile.tags
        if tag.header is not None and _contains(text, normalize(tag.header), threshold)
    )


def choose_kind(
    profile: Profile, text: str, current: str, seen: Sequence[str] | frozenset[str]
) -> str:
    """Return the kind to report for ``text``, read as ``current``.

    Among the kinds whose pattern accepts ``text``: a kind whose header was seen first,
    then kinds without a header, then kinds whose header was not seen; in profile order
    within each group. ``current`` is kept when it is among the best.
    """
    fitting = [tag for tag in profile.tags if tag.matches(text)]
    if not fitting:
        return current

    def rank(tag_id: str, header: str | None) -> int:
        if header is None:
            return 1
        return 0 if tag_id in seen else 2

    best = min(rank(tag.id, tag.header) for tag in fitting)
    winners = [tag.id for tag in fitting if rank(tag.id, tag.header) == best]
    return current if current in winners else winners[0]
