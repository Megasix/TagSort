"""Check the grammar against the portable test vectors in reference/grammar.v1.json."""

import json
from pathlib import Path
from typing import Any

import pytest

from tagsort.errors import PatternError
from tagsort.grammar import compile_pattern

VECTORS: dict[str, Any] = json.loads(
    (Path(__file__).parents[2] / "reference" / "grammar.v1.json").read_text(encoding="utf-8")
)


@pytest.mark.parametrize("case", VECTORS["valid"], ids=lambda case: case["pattern"][:40])
def test_valid_pattern(case: dict[str, Any]) -> None:
    automaton = compile_pattern(case["pattern"])
    assert automaton.min_length == case["min_length"]
    assert automaton.max_length == case["max_length"]
    assert automaton.num_states == case["num_states"]
    for text in case["match"]:
        assert automaton.matches(text), text
    for text in case["no_match"]:
        assert not automaton.matches(text), text


@pytest.mark.parametrize("case", VECTORS["invalid"], ids=lambda case: ascii(case["pattern"]))
def test_invalid_pattern(case: dict[str, Any]) -> None:
    with pytest.raises(PatternError) as info:
        compile_pattern(case["pattern"])
    assert info.value.code == case["code"]
    assert info.value.position == case["position"]
    assert info.value.pattern == case["pattern"]


def test_every_error_code_has_a_vector() -> None:
    documented = (Path(__file__).parents[2] / "docs" / "patterns.md").read_text(encoding="utf-8")
    table = documented.split("## Error codes", 1)[1].split("## Examples", 1)[0]
    codes = {line.split("`")[1] for line in table.splitlines() if line.startswith("| `")}
    covered = {case["code"] for case in VECTORS["invalid"]}
    # too_complex needs more states than any pattern within the other limits produces;
    # it is covered by a unit test that lowers the limit.
    assert codes - covered == {"too_complex"}
    assert covered <= codes


def test_documented_examples_are_vectors() -> None:
    documented = (Path(__file__).parents[2] / "docs" / "patterns.md").read_text(encoding="utf-8")
    examples = documented.split("## Examples", 1)[1]
    patterns = {case["pattern"] for case in VECTORS["valid"]}
    for line in examples.splitlines()[4:]:
        pattern = line.split("`")[1].replace("\\|", "|")
        assert pattern in patterns
