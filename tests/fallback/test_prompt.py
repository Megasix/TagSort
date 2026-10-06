import json
from typing import Any

import pytest

from tagsort import Profile, ProviderError
from tagsort.fallback.prompt import ANSWER_SCHEMA, build_instructions, parse_answer
from tests.helpers import DATA


def answer(**changes: Any) -> str:
    tag: dict[str, Any] = {
        "text": "MD04127",
        "legibility": "certain",
        "alternatives": [],
        "box": [10, 20, 110, 70],
        "angle": "0",
    }
    tag.update(changes)
    return json.dumps({"tags": [tag]})


def parse(text: str, box_format: Any = "pixels_xyxy") -> Any:
    return parse_answer(text, provider="test", box_format=box_format, width=200, height=100)


def test_instructions_list_the_profile_patterns() -> None:
    profile = Profile.from_file(DATA / "museum_a.json")
    text = build_instructions(profile, box_format="pixels_xyxy", width=640, height=480)
    assert "- primary: MD\\d{5}" in text
    assert "- secondary: [A-Z]{2}-\\d{3,4}" in text
    assert "(640 x 480)" in text
    assert "Never correct" in text


@pytest.mark.parametrize(
    ("box_format", "expected"),
    [("normalized_xyxy", "[x_min, y_min"), ("normalized_yxyx", "[y_min, x_min")],
)
def test_instructions_describe_the_box_format(box_format: Any, expected: str) -> None:
    profile = Profile.from_file(DATA / "museum_a.json")
    assert expected in build_instructions(profile, box_format=box_format, width=1, height=1)


def test_schema_uses_only_portable_keywords() -> None:
    """No numeric or length constraints: some providers reject them in strict mode."""
    text = json.dumps(ANSWER_SCHEMA)
    for keyword in (
        "minimum",
        "maximum",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "pattern",
    ):
        assert f'"{keyword}"' not in text


def test_parse_valid_answer() -> None:
    (tag,) = parse(answer(alternatives=["MD04121", "MD04127", " ", "MD04121", "a", "b"]))
    assert tag.text == "MD04127"
    assert tag.legibility == "certain"
    assert tag.alternatives == ("MD04121", "a", "b")
    assert tag.box == (10, 20, 110, 70)
    assert tag.angle == 0


def test_parse_empty_answer() -> None:
    assert parse('{"tags": []}') == ()


def test_empty_text_is_unreadable() -> None:
    (tag,) = parse(answer(text="  ", legibility="certain"))
    assert tag.text == ""
    assert tag.legibility == "unreadable"


@pytest.mark.parametrize(
    ("box_format", "box", "expected"),
    [
        ("pixels_xyxy", [10, 20, 110, 70], (10, 20, 110, 70)),
        ("pixels_xyxy", [110, 70, 10, 20], (10, 20, 110, 70)),
        ("pixels_xyxy", [-5, -5, 500, 500], (0, 0, 200, 100)),
        ("normalized_xyxy", [100, 200, 500, 700], (20, 20, 100, 70)),
        ("normalized_yxyx", [200, 100, 700, 500], (20, 20, 100, 70)),
        ("normalized_yxyx", [0, 0, 1000, 1000], (0, 0, 200, 100)),
    ],
)
def test_box_conversion(box_format: Any, box: list[float], expected: tuple[float, ...]) -> None:
    (tag,) = parse(answer(box=box), box_format)
    assert tag.box == pytest.approx(expected)


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        "[]",
        '{"tags": {}}',
        '{"items": []}',
        answer(text=3),
        answer(legibility="maybe"),
        answer(angle="45"),
        answer(angle=90),
        answer(alternatives="MD1"),
        answer(alternatives=[1]),
        answer(box=[1, 2, 3]),
        answer(box=[1, 2, 3, "4"]),
        answer(box=[1, 2, 3, True]),
        answer(box=[1, 2, 3, float("inf")]),
        answer(box=[10, 10, 10.5, 50]),
        answer(box=[300, 10, 400, 50]),
        answer(extra=1),
        json.dumps({"tags": ["MD04127"]}),
    ],
)
def test_invalid_answers_raise(text: str) -> None:
    with pytest.raises(ProviderError) as info:
        parse(text)
    assert info.value.reason == "invalid_response"
    assert not info.value.retryable
