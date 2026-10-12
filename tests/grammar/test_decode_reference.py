"""Check decoding against the portable test vectors in reference/decode.v1.json."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tagsort.grammar import compile_pattern
from tagsort.grammar.decode import constrained, greedy

VECTORS: dict[str, Any] = json.loads(
    (Path(__file__).parents[2] / "reference" / "decode.v1.json").read_text(encoding="utf-8")
)
CLASSES: list[str] = VECTORS["classes"]


@pytest.mark.parametrize("case", VECTORS["cases"], ids=lambda case: case["name"])
def test_decoding(case: dict[str, Any]) -> None:
    probs = np.array(case["probabilities"])
    reading = greedy(probs, CLASSES)
    assert reading.text == case["greedy"]["text"]
    assert reading.probability == pytest.approx(case["greedy"]["probability"], rel=1e-6)
    if "pattern" in case:
        readings = constrained(
            probs,
            CLASSES,
            compile_pattern(case["pattern"]),
            beam=VECTORS["beam"],
            min_char_probability=VECTORS["min_char_probability"],
        )
        assert [r.text for r in readings] == [r["text"] for r in case["constrained"]]
        for r, expected in zip(readings, case["constrained"], strict=True):
            assert r.probability == pytest.approx(expected["probability"], rel=1e-6)


def test_cases_cover_what_they_say() -> None:
    by_name = {case["name"]: case for case in VECTORS["cases"]}
    assert by_name["letter_o"]["greedy"]["text"] == "GJO7966"
    assert by_name["letter_o"]["constrained"][0]["text"] == "GJ07966"
    assert by_name["gap"]["constrained"][0]["text"] == "GJ07966"
    assert by_name["space_in_pattern"]["constrained"][0]["text"] == "AB 123"
    assert by_name["double"]["constrained"][0]["text"] == "GJ07766"
    assert by_name["impossible"]["constrained"] == []
    assert by_name["blank"]["greedy"]["text"] == ""
