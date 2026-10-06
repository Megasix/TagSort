"""The local pipeline reproduces reference/pipeline.v1.json, the vectors given to ports."""

import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from tagsort import ModelError, Profile, Reader
from tagsort.models import load_manifest, model_files
from tagsort.pipeline.detect import OnnxDetector
from tagsort.pipeline.local import LocalPipeline
from tests.helpers import ROOT

REFERENCE: dict[str, Any] = json.loads(
    (ROOT / "reference" / "pipeline.v1.json").read_text(encoding="utf-8")
)
PIXELS = REFERENCE["pixel_tolerance"]
PROBABILITY = REFERENCE["probability_tolerance"]
ANGLE = REFERENCE["angle_tolerance"]


@pytest.fixture(scope="module")
def pipeline() -> LocalPipeline:
    try:
        model_files(REFERENCE["model"])
    except ModelError:
        pytest.skip("run `tagsort models download` to check the reference vectors")
    assert load_manifest(REFERENCE["model"]).version == REFERENCE["model_version"]
    return LocalPipeline(REFERENCE["model"], detection_side=REFERENCE["detection_side"])


def close_quads(found: Any, expected: Any) -> bool:
    return all(
        abs(a - b) <= PIXELS
        for p, q in zip(found, expected, strict=True)
        for a, b in zip(p, q, strict=True)
    )


def angle_gap(a: float, b: float) -> float:
    return abs((a - b + 180) % 360 - 180)


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: Path(case["image"]).stem)
def test_reference_case(pipeline: LocalPipeline, case: dict[str, Any]) -> None:
    path = ROOT / "reference" / case["image"]
    image = Image.open(path).convert("RGB")
    profile = Profile.from_dict(REFERENCE["profile"])

    files = model_files(REFERENCE["model"])
    detector = OnnxDetector(
        files["det"],
        load_manifest(REFERENCE["model"]).detection,
        limit_side=REFERENCE["detection_side"],
    )
    lines = detector.detect(image)
    assert len(lines) == len(case["detection"])
    for line, expected in zip(lines, case["detection"], strict=True):
        assert close_quads(line.quad, expected["quad"])
        assert line.score == pytest.approx(expected["score"], abs=PROBABILITY)

    readings = pipeline.read(image, profile)
    assert len(readings) == len(case["lines"])
    for reading, expected in zip(readings, case["lines"], strict=True):
        assert close_quads(reading.quad, expected["quad"])
        assert angle_gap(reading.angle, expected["angle"]) <= ANGLE
        assert reading.agrees == expected["agrees"]
        assert [r[0] for r in reading.readings] == [r["text"] for r in expected["readings"]]
        for (_, probability, tag_id), want in zip(
            reading.readings, expected["readings"], strict=True
        ):
            assert probability == pytest.approx(want["probability"], abs=PROBABILITY)
            assert tag_id == want["tag_id"]

    result = Reader(profile, backend=pipeline).read(path).to_dict()
    assert len(result["tags"]) == len(case["result"]["tags"])
    for tag, want in zip(result["tags"], case["result"]["tags"], strict=True):
        assert (tag["tag_id"], tag["text"], tag["status"]) == (
            want["tag_id"],
            want["text"],
            want["status"],
        )
        assert angle_gap(tag["angle"], want["angle"]) <= ANGLE
        assert close_quads(tag["polygon"], want["polygon"])


def test_reference_images_are_synthetic() -> None:
    """Reference images are rendered, never photos: no camera metadata."""
    for case in REFERENCE["cases"]:
        with Image.open(ROOT / "reference" / case["image"]) as image:
            assert image.format == "PNG"
            assert not image.getexif()
