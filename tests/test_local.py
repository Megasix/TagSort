"""The local pipeline inside Reader: statuses, orientation, cascade to a fallback."""

from dataclasses import dataclass, field
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from PIL import Image, ImageDraw, ImageFont

from tagsort import (
    ApiUsage,
    ModelError,
    Profile,
    ProviderError,
    ProviderTag,
    Reader,
    Usage,
    VisionAnswer,
    VisionRequest,
)
from tagsort.fallback.base import BoxFormat
from tagsort.models import DEFAULT_MODEL, model_files
from tagsort.pipeline.local import MIN_ACCEPT_PROBABILITY, LineReading, LocalPipeline, score

PROFILE = Profile.from_dict(
    {
        "schema_version": "1.0",
        "name": "Lot",
        "tags_per_individual": 1,
        "tags": [
            {"id": "primary", "pattern": "GJ\\d{5}"},
            {"id": "secondary", "pattern": "\\d{4}"},
        ],
    }
)
QUAD = ((10.0, 20.0), (110.0, 20.0), (110.0, 50.0), (10.0, 50.0))


def line(*readings: tuple[str, float, str], agrees: bool = True, angle: float = 0.0) -> LineReading:
    return LineReading(
        quad=QUAD,
        angle=angle,
        readings=readings,
        agrees=agrees,
        crop=Image.new("RGB", (100, 30), "white"),
    )


class StubPipeline(LocalPipeline):
    """A LocalPipeline that returns fixed lines instead of running a model."""

    def __init__(self, *lines: LineReading) -> None:
        self.model = "stub"
        self.version = "1.0"
        self.lines = list(lines)

    def read(self, image: Image.Image, profile: Profile) -> list[LineReading]:
        return self.lines


@dataclass
class FakeFallback:
    tags: tuple[ProviderTag, ...] = ()
    fail: bool = False
    name: str = "fake"
    model: str = "fake-1"
    max_side: int = 512
    box_format: BoxFormat = "pixels_xyxy"
    requests: list[VisionRequest] = field(default_factory=list)

    def read(self, request: VisionRequest) -> VisionAnswer:
        self.requests.append(request)
        if self.fail:
            raise ProviderError("down", provider="fake", reason="server")
        return VisionAnswer(tags=self.tags, model="fake-1", usage=Usage(10, 1))


def raw(text: str, legibility: Any = "certain") -> ProviderTag:
    return ProviderTag(
        text=text, legibility=legibility, alternatives=(), box=(1, 1, 50, 20), angle=0
    )


def read(pipeline: LocalPipeline, fallback: Any = None) -> Any:
    return Reader(PROFILE, backend=pipeline, fallback=fallback).read(
        Image.new("RGB", (400, 300), "white")
    )


def test_score() -> None:
    assert score(0.98, agrees=True) == pytest.approx(0.998)
    assert score(MIN_ACCEPT_PROBABILITY, agrees=True) >= 0.9
    assert score(0.2, agrees=True) == 0.2
    assert score(0.99, agrees=False) == 0.89


def test_agreeing_confident_reading_is_accepted(result_validator: Draft202012Validator) -> None:
    result = read(StubPipeline(line(("GJ07966", 0.97, "primary"), ("GJ07986", 0.01, "primary"))))
    (tag,) = result.tags
    assert (tag.text, tag.tag_id, tag.status, tag.source) == (
        "GJ07966",
        "primary",
        "accepted",
        "local",
    )
    assert tag.confidence >= 0.9
    assert [c.text for c in tag.candidates] == ["GJ07986"]
    assert tag.polygon[0] == (10.0, 20.0)
    assert result.model_version == "local:stub-1.0"
    assert set(result.timings_ms) == {"preprocess", "local"}
    result_validator.validate(result.to_dict())


def test_disagreeing_reading_needs_review() -> None:
    (tag,) = read(StubPipeline(line(("GJ07966", 0.97, "primary"), agrees=False))).tags
    assert tag.status == "review"
    assert tag.confidence == 0.89


def test_weak_reading_is_unreadable_with_its_guess_kept() -> None:
    (tag,) = read(StubPipeline(line(("GJ07966", 0.05, "primary"), agrees=False))).tags
    assert tag.status == "unreadable"
    assert tag.text is None
    assert tag.tag_id == "primary"
    assert [c.text for c in tag.candidates] == ["GJ07966"]


def test_upside_down_ambiguity_needs_review() -> None:
    (tag,) = read(StubPipeline(line(("9800", 0.95, "secondary")))).tags
    assert tag.status == "review"
    assert [c.text for c in tag.candidates] == ["0086"]


def test_polygon_is_scaled_to_the_original_image() -> None:
    reader = Reader(PROFILE, backend=StubPipeline(line(("GJ07966", 0.97, "primary"), angle=359.97)))
    result = reader.read(Image.new("RGB", (8192, 4096), "white"))
    (tag,) = result.tags
    assert tag.polygon[0] == (20.0, 40.0)
    assert tag.angle == 0.0


def test_fallback_reads_doubtful_tags_from_their_crop(
    result_validator: Draft202012Validator,
) -> None:
    fallback = FakeFallback(tags=(raw("GJ07966"),))
    pipeline = StubPipeline(
        line(("GJ07966", 0.97, "primary")),  # accepted locally: not sent
        line(("GJ07968", 0.4, "primary"), agrees=False),  # doubtful: sent
    )
    result = read(pipeline, fallback)
    accepted, rescued = result.tags
    assert accepted.source == "local"
    assert (rescued.text, rescued.status, rescued.source) == ("GJ07966", "accepted", "fallback")
    assert "GJ07968" in [c.text for c in rescued.candidates]
    assert rescued.polygon == accepted.polygon
    (request,) = fallback.requests
    assert (request.width, request.height) == (100, 30), "only the crop is sent"
    assert result.model_version == "local:stub-1.0+fake:fake-1"
    assert "fallback" in result.timings_ms
    result_validator.validate(result.to_dict())


def test_fallback_answers_with_a_gap_fit_without_it() -> None:
    fallback = FakeFallback(tags=(raw("GJ 07966"),))
    doubtful = line(("GJ07968", 0.04, "primary"), agrees=False)
    (tag,) = read(StubPipeline(doubtful), fallback).tags
    assert (tag.text, tag.tag_id, tag.status, tag.source) == (
        "GJ07966",
        "primary",
        "accepted",
        "fallback",
    )


def test_fallback_answers_that_do_not_help_keep_the_local_tag() -> None:
    doubtful = line(("GJ07968", 0.4, "primary"), agrees=False)
    for fallback in (
        FakeFallback(fail=True),
        FakeFallback(tags=()),
        FakeFallback(tags=(raw("Field"),)),
    ):
        (tag,) = read(StubPipeline(doubtful), fallback).tags
        assert (tag.text, tag.status, tag.source) == ("GJ07968", "review", "local")


def test_uncertain_fallback_stays_in_review() -> None:
    (tag,) = read(
        StubPipeline(line(("GJ07968", 0.4, "primary"), agrees=False)),
        FakeFallback(tags=(raw("GJ07966", "uncertain"),)),
    ).tags
    assert (tag.text, tag.status, tag.source) == ("GJ07966", "review", "fallback")


def test_no_tag_found() -> None:
    result = read(StubPipeline())
    assert result.tags == ()


def test_default_backend_needs_the_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    monkeypatch.setenv("TAGSORT_MODELS", str(tmp_path))
    with pytest.raises(ModelError, match="tagsort models download"):
        Reader(PROFILE)


def test_reads_a_rendered_tag_end_to_end_with_the_real_model(
    result_validator: Draft202012Validator,
) -> None:
    try:
        model_files(DEFAULT_MODEL)
    except ModelError:
        pytest.skip("run `tagsort models download` to test with the real model")
    image = Image.new("RGB", (1600, 1200), (230, 225, 210))
    font = ImageFont.load_default(size=56)
    tag = Image.new("RGB", (420, 110), "white")
    ImageDraw.Draw(tag).text((25, 20), "GJ08104", fill="black", font=font)
    image.paste(tag.rotate(180, expand=True), (300, 400))  # upside down
    ImageDraw.Draw(image).text((900, 900), "Field notes", fill="black", font=font)
    result = Reader(PROFILE).read(image)
    (found,) = result.tags
    assert (found.text, found.tag_id, found.status, found.source) == (
        "GJ08104",
        "primary",
        "accepted",
        "local",
    )
    assert 170 <= found.angle <= 190
    result_validator.validate(result.to_dict())


def test_a_profile_without_tags_returns_every_line_as_other_text() -> None:
    try:
        model_files(DEFAULT_MODEL)
    except ModelError:
        pytest.skip("run `tagsort models download` to test with the real model")
    image = Image.new("RGB", (1600, 1200), (230, 225, 210))
    font = ImageFont.load_default(size=56)
    ImageDraw.Draw(image).text((300, 400), "GJ08104", fill="black", font=font)
    empty = Profile(name="New collection", tags_per_individual=1, tags=())
    result = Reader(empty, open_reading=True).read(image)
    assert result.tags == ()
    assert "GJ08104" in [other.text for other in result.other_texts]


def test_the_fallback_bill_is_reported(result_validator: Draft202012Validator) -> None:
    fallback = FakeFallback(tags=(raw("GJ07966"),))
    pipeline = StubPipeline(
        line(("GJ07968", 0.4, "primary"), agrees=False),
        line(("GJ07969", 0.4, "primary"), agrees=False),
    )
    result = read(pipeline, fallback)
    assert result.api_usage == ApiUsage(calls=2, input_tokens=20, output_tokens=2)
    assert result.to_dict()["api_usage"] == {"calls": 2, "input_tokens": 20, "output_tokens": 2}
    result_validator.validate(result.to_dict())


def test_no_bill_without_an_answer() -> None:
    assert read(StubPipeline(line(("GJ07966", 0.97, "primary")))).api_usage is None
    doubtful = line(("GJ07968", 0.4, "primary"), agrees=False)
    result = read(StubPipeline(doubtful), FakeFallback(fail=True))
    assert result.api_usage is None
    assert "api_usage" not in result.to_dict()
