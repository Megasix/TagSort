"""Constrained CTC decoding, checked against an exhaustive computation on small cases."""

import itertools
import math

import numpy as np
import pytest

from tagsort.grammar import compile_pattern
from tagsort.grammar.decode import Reading, constrained, greedy

CLASSES = ("", "G", "J", "O", "0", "6", "7", "9", " ")  # blank first, like PP-OCR


def frames(*rows: dict[str, float]) -> np.ndarray:
    """Probability matrix from per-frame dicts; the rest of each frame goes to the blank."""
    probs = np.zeros((len(rows), len(CLASSES)))
    for t, row in enumerate(rows):
        for char, p in row.items():
            probs[t, CLASSES.index(char)] = p
        probs[t, 0] = 1 - sum(row.values())
    return probs


def spell(text: str, confidence: float = 0.98) -> np.ndarray:
    """Frames spelling text, a blank frame between characters."""
    rows: list[dict[str, float]] = []
    for char in text:
        rows += [{char: confidence}, {}]
    return frames(*rows)


def test_greedy_collapses_repeats_and_blanks() -> None:
    probs = frames({"G": 0.9}, {"G": 0.9}, {}, {"J": 0.9}, {"6": 0.8}, {}, {"6": 0.8})
    reading = greedy(probs, CLASSES)
    assert reading.text == "GJ66"
    assert reading.probability == pytest.approx(0.9 * 0.9 * 1.0 * 0.9 * 0.8 * 1.0 * 0.8)


def test_constrained_reads_a_clean_tag() -> None:
    (best, *_) = constrained(spell("GJ07966"), CLASSES, compile_pattern(r"GJ\d{5}"))
    assert best.text == "GJ07966"
    assert best.probability > 0.8


def test_grammar_turns_a_letter_o_into_a_zero() -> None:
    probs = spell("GJ07966")
    probs[4] = 0
    probs[4, CLASSES.index("O")] = 0.6
    probs[4, CLASSES.index("0")] = 0.4
    assert greedy(probs, CLASSES).text == "GJO7966"
    (best, *_) = constrained(probs, CLASSES, compile_pattern(r"GJ\d{5}"))
    assert best.text == "GJ07966"


def test_no_valid_reading() -> None:
    assert constrained(spell("GJ"), CLASSES, compile_pattern(r"\d{4}")) == []


def test_characters_the_model_cannot_output_are_impossible() -> None:
    assert constrained(spell("GJ"), CLASSES, compile_pattern("GX")) == []


def test_readings_are_sorted_and_unique() -> None:
    probs = spell("6767")
    probs[0, CLASSES.index("6")] = 0.5
    probs[0, CLASSES.index("9")] = 0.45
    readings = constrained(probs, CLASSES, compile_pattern(r"[679]{4}"), beam=8)
    assert [r.text for r in readings][:2] == ["6767", "9767"]
    assert [r.probability for r in readings] == sorted(
        (r.probability for r in readings), reverse=True
    )
    assert len({r.text for r in readings}) == len(readings)


def test_a_space_the_pattern_does_not_allow_does_not_count() -> None:
    probs = spell("GJ 07966")
    assert greedy(probs, CLASSES).text == "GJ 07966"
    (best, *_) = constrained(probs, CLASSES, compile_pattern(r"GJ\d{5}"))
    assert best.text == "GJ07966"
    # As likely as the same tag written without the gap.
    (plain, *_) = constrained(spell("GJ07966"), CLASSES, compile_pattern(r"GJ\d{5}"))
    assert best.probability == pytest.approx(plain.probability, rel=0.05)


def test_a_space_separates_repeated_characters() -> None:
    (best, *_) = constrained(spell("66 6"), CLASSES, compile_pattern(r"\d{3}"))
    assert best.text == "666"


def test_a_space_the_pattern_has_is_a_character() -> None:
    (best, *_) = constrained(spell("GJ 07966"), CLASSES, compile_pattern(r"GJ \d{5}"))
    assert best.text == "GJ 07966"
    assert constrained(spell("GJ07966"), CLASSES, compile_pattern(r"GJ \d{5}")) == []


def exact_label_probabilities(probs: np.ndarray, classes: tuple[str, ...]) -> dict[str, float]:
    """Sum the probability of every alignment, grouped by the label it collapses to."""
    totals: dict[str, float] = {}
    for path in itertools.product(range(len(classes)), repeat=probs.shape[0]):
        p = math.prod(probs[t, k] for t, k in enumerate(path))
        label = []
        previous = 0
        for k in path:
            if k != 0 and k != previous:
                label.append(classes[k])
            previous = k
        text = "".join(label)
        totals[text] = totals.get(text, 0.0) + p
    return totals


@pytest.mark.parametrize("seed", range(25))
def test_matches_exhaustive_computation(seed: int) -> None:
    classes = ("", "6", "7", "9")
    rng = np.random.default_rng(seed)
    probs = rng.dirichlet(np.ones(len(classes)) * 0.6, size=6)
    automaton = compile_pattern("[679]{2,3}")
    exact = {
        t: p for t, p in exact_label_probabilities(probs, classes).items() if automaton.matches(t)
    }
    readings = constrained(probs, classes, automaton, beam=64, min_char_probability=0.0)
    assert {r.text for r in readings} == set(exact)
    for reading in readings:
        assert reading.probability == pytest.approx(exact[reading.text], rel=1e-9)
    assert readings[0].text == max(exact, key=exact.__getitem__)


@pytest.mark.parametrize("seed", range(25))
def test_ignored_spaces_match_exhaustive_computation(seed: int) -> None:
    classes = ("", "6", "7", " ")
    rng = np.random.default_rng(seed)
    probs = rng.dirichlet(np.ones(len(classes)) * 0.6, size=6)
    automaton = compile_pattern("[67]{2,3}")
    # Collapse each alignment as usual, then drop the spaces the pattern does not allow.
    exact: dict[str, float] = {}
    for text, p in exact_label_probabilities(probs, classes).items():
        text = text.replace(" ", "")
        if automaton.matches(text):
            exact[text] = exact.get(text, 0.0) + p
    readings = constrained(probs, classes, automaton, beam=64, min_char_probability=0.0)
    assert {r.text for r in readings} == set(exact)
    for reading in readings:
        assert reading.probability == pytest.approx(exact[reading.text], rel=1e-9)


def test_reading_is_a_value() -> None:
    assert Reading("GJ1", 0.5) == Reading("GJ1", 0.5)
