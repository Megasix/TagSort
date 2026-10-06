from typing import Any

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from tagsort import ModelError, Profile
from tagsort.grammar.decode import constrained, greedy
from tagsort.models import DEFAULT_MODEL, load_manifest, model_files
from tagsort.pipeline.recognize import OnnxRecognizer, Recognizer


def model_path() -> Any:
    try:
        return model_files(DEFAULT_MODEL)["rec"]
    except ModelError:
        pytest.skip("run `tagsort models download` to test with the real model")


def rendered(text: str, angle: int = 0) -> Image.Image:
    font = ImageFont.load_default(size=40)
    image = Image.new("RGB", (40 + 26 * len(text), 64), "white")
    ImageDraw.Draw(image).text((20, 8), text, fill="black", font=font)
    return image.rotate(angle, expand=True, fillcolor="white")


def test_reads_a_rendered_tag_with_the_real_model() -> None:
    recognizer = OnnxRecognizer(model_path(), load_manifest(DEFAULT_MODEL).recognition)
    assert isinstance(recognizer, Recognizer)
    probs = recognizer.probabilities(rendered("GJ07966"))
    assert probs.shape[1] == len(recognizer.classes)
    assert np.allclose(probs.sum(axis=1), 1, atol=1e-3)
    automaton = (
        Profile.from_dict(
            {
                "schema_version": "1.0",
                "name": "t",
                "tags_per_individual": 1,
                "tags": [{"id": "primary", "pattern": "GJ\\d{5}"}],
            }
        )
        .tags[0]
        ._automaton
    )
    (best, *_) = constrained(probs, recognizer.classes, automaton)
    assert best.text == "GJ07966"
    assert best.probability > 0.5
    assert greedy(probs, recognizer.classes).text.replace(" ", "") == "GJ07966"


def test_upside_down_text_is_not_read_as_valid() -> None:
    recognizer = OnnxRecognizer(model_path(), load_manifest(DEFAULT_MODEL).recognition)
    probs = recognizer.probabilities(rendered("GJ07966", angle=180))
    automaton = (
        Profile.from_dict(
            {
                "schema_version": "1.0",
                "name": "t",
                "tags_per_individual": 1,
                "tags": [{"id": "primary", "pattern": "GJ\\d{5}"}],
            }
        )
        .tags[0]
        ._automaton
    )
    readings = constrained(probs, recognizer.classes, automaton)
    assert not readings or readings[0].probability < 0.1


def test_tall_and_wide_lines_are_resized(monkeypatch: pytest.MonkeyPatch) -> None:
    recognizer = OnnxRecognizer(model_path(), load_manifest(DEFAULT_MODEL).recognition)
    assert recognizer.probabilities(Image.new("L", (5, 200), 255)).shape[0] >= 1
    assert recognizer.probabilities(Image.new("RGB", (8000, 20), "white")).shape[0] >= 1


def test_class_count_must_match_the_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    recognizer = OnnxRecognizer(model_path(), load_manifest(DEFAULT_MODEL).recognition)
    monkeypatch.setattr(recognizer, "_classes", recognizer.classes[:-1])
    with pytest.raises(ValueError, match="classes but the manifest lists"):
        recognizer.probabilities(rendered("1"))
