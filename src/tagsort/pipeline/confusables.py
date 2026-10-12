r"""Characters a recognizer confuses, and readings they make ambiguous between kinds of tag.

``FW54019`` fits a kind ``[A-Z]{2}\d{5}``; with ``5`` read for ``S`` it would be
``FWS4019``, which fits another kind ``FWS\d{4}``. The recognizer cannot tell such pairs
apart on a worn handwritten tag, so a reading one swap away from another kind of tag is
left for a person to choose, with both readings (lot-03, #123).
"""

from __future__ import annotations

from tagsort.profile import Profile

__all__ = ["CONFUSABLE", "other_kind_readings"]

#: Letters and digits a recognizer mixes up, both ways.
CONFUSABLE: dict[str, str] = {
    "S": "5",
    "5": "S",
    "O": "0",
    "0": "O",
    "I": "1",
    "1": "I",
    "B": "8",
    "8": "B",
    "Z": "2",
    "2": "Z",
    "G": "6",
    "6": "G",
}


def other_kind_readings(profile: Profile, text: str, tag_id: str | None) -> list[str]:
    """Readings one confusable swap away from ``text`` that fit another kind of tag.

    Kinds that differ only in the swapped character's class are told apart by nothing
    but that character; readings of the same kind are ordinary alternatives and left out.
    """
    found: list[str] = []
    upper = text.upper()
    for index, char in enumerate(upper):
        swap = CONFUSABLE.get(char)
        if swap is None:
            continue
        variant = upper[:index] + swap + upper[index + 1 :]
        kind = profile.match(variant)
        if kind is not None and kind != tag_id and variant not in found:
            found.append(variant)
    return found
