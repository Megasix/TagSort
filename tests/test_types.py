import json
import math
from dataclasses import replace
from typing import Any

import pytest
from jsonschema import Draft202012Validator

import tagsort
from tagsort import ApiUsage, Candidate, ImageInfo, ReadResult, Tag

SQUARE = ((812, 400), (1210, 410), (1205, 560), (808, 552))


def make_tag(**changes: Any) -> Tag:
    fields: dict[str, Any] = {
        "tag_id": "primary",
        "text": "MD04127",
        "confidence": 0.97,
        "status": "accepted",
        "polygon": SQUARE,
        "angle": 180,
        "candidates": (Candidate("MD04121", 0.02),),
        "source": "local",
    }
    fields.update(changes)
    return Tag(**fields)


def make_result(**changes: Any) -> ReadResult:
    fields: dict[str, Any] = {
        "engine_version": tagsort.__version__,
        "model_version": "det-1.0+rec-1.0",
        "image": ImageInfo(4032, 3024, exif_rotation=90),
        "tags": (make_tag(),),
        "timings_ms": {"detect": 41, "recognize": 18},
    }
    fields.update(changes)
    return ReadResult(**fields)


VALID_RESULTS = {
    "example": make_result(),
    "no tags": make_result(tags=(), timings_ms={}),
    "review without tag id": make_result(
        tags=(make_tag(tag_id=None, status="review", confidence=0.4, candidates=()),)
    ),
    "unreadable": make_result(
        tags=(make_tag(tag_id=None, text=None, status="unreadable", confidence=0.0),)
    ),
    "fallback, float coordinates": make_result(
        tags=(make_tag(source="fallback", polygon=((0.5, 1.5),) * 4, angle=359.9),)
    ),
    "unicode text": make_result(tags=(make_tag(text="Muséum 12", tag_id=None, status="review"),)),
}


@pytest.mark.parametrize("name", VALID_RESULTS)
def test_output_conforms_to_schema(result_validator: Draft202012Validator, name: str) -> None:
    result = VALID_RESULTS[name]
    result_validator.validate(result.to_dict())
    result_validator.validate(json.loads(result.to_json()))


def test_example_matches_claude_md() -> None:
    document = json.loads(make_result().to_json(indent=2))
    assert document == {
        "schema_version": "1.0",
        "engine_version": tagsort.__version__,
        "model_version": "det-1.0+rec-1.0",
        "image": {"width": 4032, "height": 3024, "exif_rotation": 90},
        "tags": [
            {
                "tag_id": "primary",
                "text": "MD04127",
                "confidence": 0.97,
                "status": "accepted",
                "polygon": [[812, 400], [1210, 410], [1205, 560], [808, 552]],
                "angle": 180,
                "candidates": [{"text": "MD04121", "confidence": 0.02}],
                "source": "local",
            }
        ],
        "timings_ms": {"detect": 41, "recognize": 18},
    }


def test_to_json_keeps_unicode_readable() -> None:
    result = make_result(tags=(make_tag(text="Ré-1", tag_id=None, status="review"),))
    assert "Ré-1" in result.to_json()


INVALID_TAGS: list[tuple[dict[str, Any], type[Exception], str]] = [
    ({"confidence": 1.01}, ValueError, "confidence must be between 0 and 1"),
    ({"confidence": -0.1}, ValueError, "confidence"),
    ({"confidence": math.nan}, ValueError, "finite"),
    ({"confidence": True}, TypeError, "confidence must be a number"),
    ({"confidence": "0.9"}, TypeError, "confidence must be a number"),
    ({"status": "maybe"}, ValueError, "status must be one of"),
    ({"source": "cloud"}, ValueError, "source must be one of"),
    ({"text": None}, ValueError, "text must be None exactly when"),
    ({"text": ""}, ValueError, "text must not be empty"),
    ({"text": 42}, TypeError, "text must be a str or None"),
    ({"status": "unreadable"}, ValueError, "text must be None exactly when"),
    ({"tag_id": None}, ValueError, "accepted tag must have a tag_id"),
    ({"tag_id": ""}, ValueError, "tag_id must not be empty"),
    ({"polygon": SQUARE[:3]}, ValueError, "exactly 4 points"),
    ({"polygon": ((1, 2, 3),) * 4}, ValueError, "exactly 4 points"),
    ({"polygon": ((1, math.inf),) * 4}, ValueError, "finite"),
    ({"polygon": (("1", 2),) * 4}, TypeError, "polygon coordinate must be a number"),
    ({"angle": 360}, ValueError, "angle must be below 360"),
    ({"angle": -90}, ValueError, "angle must be at least 0"),
    ({"candidates": ("MD1",)}, TypeError, "candidates must be Candidate objects"),
    (
        {"candidates": (Candidate("A", 0.1), Candidate("B", 0.2))},
        ValueError,
        "sorted by decreasing confidence",
    ),
]


@pytest.mark.parametrize(("changes", "error", "message"), INVALID_TAGS)
def test_invalid_tag_is_rejected(
    changes: dict[str, Any], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        make_tag(**changes)


@pytest.mark.parametrize(
    ("args", "error", "message"),
    [
        ((0, 10), ValueError, "width must be at least 1"),
        ((10, 0), ValueError, "height must be at least 1"),
        ((10.0, 10), TypeError, "width must be an int"),
        ((True, 10), TypeError, "width must be an int"),
        ((10, 10, 45), ValueError, "exif_rotation must be 0, 90, 180 or 270"),
        ((10, 10, False), ValueError, "exif_rotation"),
    ],
)
def test_invalid_image_is_rejected(
    args: tuple[Any, ...], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        ImageInfo(*args)


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"engine_version": ""}, ValueError, "engine_version must not be empty"),
        ({"model_version": None}, TypeError, "model_version must be a str"),
        ({"image": {"width": 1, "height": 1}}, TypeError, "image must be an ImageInfo"),
        ({"tags": ({"text": "MD"},)}, TypeError, "tags must be Tag objects"),
        ({"timings_ms": {"detect": -1}}, ValueError, "at least 0"),
        ({"timings_ms": {"": 1}}, ValueError, "timings_ms key must not be empty"),
    ],
)
def test_invalid_result_is_rejected(
    changes: dict[str, Any], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        make_result(**changes)


def test_candidate_validation() -> None:
    with pytest.raises(ValueError, match="text must not be empty"):
        Candidate("", 0.5)
    with pytest.raises(ValueError, match="confidence"):
        Candidate("A", 2)


def test_results_are_immutable() -> None:
    result = make_result()
    with pytest.raises(AttributeError):
        result.model_version = "other"  # type: ignore[misc]
    with pytest.raises(TypeError):
        result.timings_ms["detect"] = 0  # type: ignore[index]


def test_sequences_are_normalized_and_inputs_copied() -> None:
    timings = {"detect": 1.0}
    tags = [make_tag()]
    result = make_result(tags=tags, timings_ms=timings)
    timings["detect"] = 99
    tags.clear()
    assert result.tags == (make_tag(),)
    assert result.timings_ms == {"detect": 1.0}


def test_tag_sequences_are_normalized_to_tuples() -> None:
    tag = make_tag(polygon=[[1, 2], [3, 4], [5, 6], [7, 8]], candidates=[])
    assert isinstance(tag.candidates, tuple)
    assert not tag.candidates
    assert tag.polygon == ((1, 2), (3, 4), (5, 6), (7, 8))


def test_equality() -> None:
    assert make_result() == make_result()
    assert make_tag() != replace(make_tag(), confidence=0.5)


def test_schema_version_is_not_a_field() -> None:
    with pytest.raises(TypeError):
        make_result(schema_version="2.0")
    assert ReadResult.SCHEMA_VERSION == "1.0"


def test_api_usage_round_trips_and_validates(result_validator: Draft202012Validator) -> None:
    result = make_result(api_usage=ApiUsage(calls=1, input_tokens=812, output_tokens=40))
    document = result.to_dict()
    assert document["api_usage"] == {"calls": 1, "input_tokens": 812, "output_tokens": 40}
    result_validator.validate(document)
    assert "api_usage" not in make_result().to_dict()
    for bad in (-1, 1.5, True):
        with pytest.raises(ValueError, match="calls"):
            ApiUsage(calls=bad)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="api_usage"):
        make_result(api_usage={"calls": 1})
