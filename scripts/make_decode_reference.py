"""Write CTC decoding test vectors to reference/decode.v1.json.

Each case gives a recognizer output (per step, the probability of each class), the
classes and, for some, a pattern; a port's greedy and constrained decoders must return
the same texts, in the same order, with the same probabilities (within a relative
tolerance). Probabilities are rounded before decoding, so the vectors are exact for the
rounded inputs. Run with ``uv run python scripts/make_decode_reference.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tagsort.grammar import compile_pattern
from tagsort.grammar.decode import constrained, greedy

OUT = Path(__file__).parents[1] / "reference" / "decode.v1.json"

# The blank first, as in every CTC recognizer, then a small alphabet with the space.
CLASSES = ["", *"0123456789", "G", "J", "O", "A", "B", "-", " "]


def peaked(rng: np.random.Generator, text: str, steps: int, sharpness: float) -> np.ndarray:
    """Probabilities spelling ``text`` with blanks between characters, plus noise."""
    logits = rng.normal(0.0, 1.0, size=(steps, len(CLASSES)))
    path = []
    for char in text:
        path += [CLASSES.index(char), 0]
    for t in range(steps):
        target = path[t] if t < len(path) else 0
        logits[t, target] += sharpness
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def confuse(probs: np.ndarray, step: int, char: str, share: float) -> np.ndarray:
    """Move ``share`` of a step's probability to ``char``, as a lookalike would."""
    out = probs.copy()
    out[step] *= 1 - share
    out[step, CLASSES.index(char)] += share
    return out


def case(name: str, probs: np.ndarray, pattern: str | None) -> dict[str, object]:
    rounded = np.round(probs, 6)
    reading = greedy(rounded, CLASSES)
    entry: dict[str, object] = {
        "name": name,
        "probabilities": rounded.tolist(),
        "greedy": {"text": reading.text, "probability": reading.probability},
    }
    if pattern is not None:
        entry["pattern"] = pattern
        entry["constrained"] = [
            {"text": r.text, "probability": r.probability}
            for r in constrained(rounded, CLASSES, compile_pattern(pattern))
        ]
    return entry


def main() -> None:
    rng = np.random.default_rng(20261012)
    cases = [
        # A clear tag: the constrained reading agrees with the greedy one.
        case("clear", peaked(rng, "GJ07966", 16, 6.0), r"GJ\d{5}"),
        # An O where the pattern needs a digit, its step giving the 0 a fair share as a
        # real O-like 0 would: the grammar reads 0.
        case("letter_o", confuse(peaked(rng, "GJO7966", 16, 6.0), 4, "0", 0.3), r"GJ\d{5}"),
        # A doubled digit needs a blank between its two characters.
        case("double", peaked(rng, "GJ07766", 16, 5.0), r"GJ\d{5}"),
        # A gap the pattern does not allow reads as a separator.
        case("gap", peaked(rng, "GJ 07966", 18, 6.0), r"GJ\d{5}"),
        # The pattern has a space: there it is a character.
        case("space_in_pattern", peaked(rng, "AB 123", 14, 6.0), r"[A-Z]{2} \d{3}"),
        # Optional length: three or four digits after the dash.
        case("range", peaked(rng, "AB-1234", 16, 5.0), r"[A-Z]{2}-\d{3,4}"),
        # Blurry: several readings close together, beam truncation matters.
        case("blurry", peaked(rng, "1234", 10, 1.5), r"\d{4}"),
        # Nothing the pattern accepts can be read.
        case("impossible", peaked(rng, "GJ", 6, 8.0), r"\d{4}"),
        # Only blanks: an empty greedy reading.
        case("blank", peaked(rng, "", 5, 8.0), None),
    ]
    document = {
        "description": (
            "Test vectors for CTC decoding (tagsort.grammar.decode). For each case, a port's "
            "greedy decoder must return the same text with the same probability, and its "
            "constrained decoder (beam 16, characters below probability 1e-4 skipped) the same "
            "texts in the same order with the same probabilities, within relative_tolerance. "
            "classes[0] is the CTC blank."
        ),
        "version": "1.0",
        "classes": CLASSES,
        "beam": 16,
        "min_char_probability": 1e-4,
        "relative_tolerance": 1e-6,
        "cases": cases,
    }
    OUT.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(OUT.parents[1])} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
