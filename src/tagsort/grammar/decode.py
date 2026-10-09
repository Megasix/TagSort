"""CTC decoding of recognizer outputs, optionally constrained by a pattern automaton.

A CTC recognizer outputs, for each of T time steps, a probability for every class: the
blank (class 0) and each character. A text is read by choosing one class per step, then
removing repeats and blanks; the probability of a text is the sum over all choices that
spell it.

:func:`constrained` runs a CTC prefix beam search in which a prefix may only grow by a
character the automaton allows next, and only texts the automaton accepts are returned.
It is what makes a recognizer read ``GJ07966`` where an unconstrained reading would say
``GJO7966``.

A space where the pattern allows none does not count against a reading: tags are often
written or printed with a gap (``GJ 07966``), so such a space is read like a blank, as a
separator, and the text comes out without it. Where the pattern has a space, it is a
character like any other.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tagsort.grammar.automaton import Automaton

__all__ = ["Reading", "constrained", "greedy"]

Probabilities = NDArray[np.floating]
"""Array of shape (T, C): per time step, the probability of each class."""


@dataclass(frozen=True)
class Reading:
    """A text and the probability the recognizer gives it."""

    text: str
    probability: float


def greedy(probs: Probabilities, classes: Sequence[str]) -> Reading:
    """Read the most likely class at each step, then collapse repeats and blanks.

    The probability is that of the single best path, a lower bound of the text's.
    """
    best = probs.argmax(axis=1)
    chars = []
    previous = -1
    for k in best:
        if k != 0 and k != previous:
            chars.append(classes[k])
        previous = int(k)
    probability = float(np.prod(probs[np.arange(len(best)), best]))
    return Reading("".join(chars), probability)


def constrained(
    probs: Probabilities,
    classes: Sequence[str],
    automaton: Automaton,
    *,
    beam: int = 16,
    min_char_probability: float = 1e-4,
) -> list[Reading]:
    """Return the most likely texts the automaton accepts, most likely first.

    Args:
        probs: Recognizer output of shape (T, C), class 0 being the blank.
        classes: Label of each class; ``classes[0]`` is the blank.
        automaton: The pattern every returned text must match.
        beam: Prefixes kept after each step. Larger is slower but closer to exact.
        min_char_probability: Characters less likely than this at a step are not
            considered there, which keeps the search fast.

    Returns:
        Readings sorted by decreasing probability, each text once; empty when no text
        accepted by the automaton is possible. Spaces the pattern does not allow are left
        out of the texts (see the module docstring).
    """
    index = {char: k for k, char in enumerate(classes) if k != 0}
    space = index.get(" ")
    # prefix -> [probability ending in blank, probability ending in a character, state]
    beams: dict[str, list[float]] = {"": [1.0, 0.0, float(automaton.start)]}
    for row in probs:
        blank = float(row[0])
        grown: dict[str, list[float]] = {}
        for prefix, (p_blank, p_char, state_value) in beams.items():
            state = int(state_value)
            total = p_blank + p_char
            # Emit a blank, or a space the pattern does not allow here: the text does not
            # change, and a character repeated after it counts twice.
            separator = blank
            if space is not None and " " not in automaton.allowed(state):
                separator += float(row[space])
            _add(grown, prefix, state, total * separator, 0.0)
            if prefix:
                # Repeat the last character without a blank: it collapses into one.
                _add(grown, prefix, state, 0.0, p_char * float(row[index[prefix[-1]]]))
            for char in automaton.allowed(state):
                k = index.get(char)
                if k is None:
                    continue
                p = float(row[k])
                if p <= 0.0 or p < min_char_probability:
                    continue
                next_state = automaton.step(state, char)
                assert next_state is not None
                # A repeated character needs a blank in between to count twice.
                source = p_blank if prefix and prefix[-1] == char else total
                _add(grown, prefix + char, next_state, 0.0, source * p)
        ranked = sorted(grown.items(), key=lambda item: item[1][0] + item[1][1], reverse=True)
        beams = dict(ranked[:beam])

    readings = [
        Reading(text, p_blank + p_char)
        for text, (p_blank, p_char, state) in beams.items()
        if automaton.is_accepting(int(state)) and p_blank + p_char > 0.0
    ]
    return sorted(readings, key=lambda reading: reading.probability, reverse=True)


def _add(
    beams: dict[str, list[float]], prefix: str, state: int, ending_blank: float, ending_char: float
) -> None:
    entry = beams.get(prefix)
    if entry is None:
        beams[prefix] = [ending_blank, ending_char, float(state)]
    else:
        entry[0] += ending_blank
        entry[1] += ending_char
