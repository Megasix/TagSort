"""Printed headers choose the kind of tag when several accept the same reading."""

import pytest
from PIL import Image

from tagsort import Profile
from tagsort.errors import ProfileError
from tagsort.pipeline.detect import DetectedLine
from tagsort.pipeline.headers import choose_kind, headers_seen, normalize
from tagsort.pipeline.local import LineReading, LocalPipeline, PlainLine

PROFILE = Profile.from_dict(
    {
        "schema_version": "1.0",
        "name": "Museum A",
        "tags_per_individual": 1,
        "tags": [
            {"id": "plain", "pattern": "\\d{5}"},
            {"id": "printed_a", "pattern": "\\d{5}", "header": "NATIONAL MUSEUM OF CANADA"},
            {"id": "printed_b", "pattern": "\\d{4,5}", "header": "Herbarium B"},
            {"id": "letters", "pattern": "MD\\d{5}"},
        ],
    }
)


def test_normalize_keeps_letters_and_digits_in_capitals() -> None:
    assert normalize("National Museum\nof Canada, 1921") == "NATIONALMUSEUMOFCANADA1921"


def test_a_header_is_seen_despite_reading_errors_and_line_breaks() -> None:
    assert headers_seen(PROFILE, ["NATIONAL MUSEUM", "OF CANADA", "16368"]) == {"printed_a"}
    assert headers_seen(PROFILE, ["NATI0NAL MUSEUM 0F CANADA"]) == {"printed_a"}
    assert headers_seen(PROFILE, ["HERBARIUM B", "1234"]) == {"printed_b"}
    assert headers_seen(PROFILE, ["16368", "a ruler", "MD00001"]) == frozenset()


def test_unrelated_text_is_not_a_header() -> None:
    assert headers_seen(PROFILE, ["NATURAL HISTORY OF CANNES"]) == frozenset()


def test_the_kind_whose_header_was_seen_wins() -> None:
    assert choose_kind(PROFILE, "16368", "plain", frozenset({"printed_a"})) == "printed_a"
    assert choose_kind(PROFILE, "16368", "plain", frozenset({"printed_b"})) == "printed_b"


def test_without_a_header_seen_kinds_without_header_come_first() -> None:
    assert choose_kind(PROFILE, "16368", "printed_a", frozenset()) == "plain"
    # A kind whose header was not seen is kept when nothing else fits.
    only_printed = Profile.from_dict(
        {
            "schema_version": "1.0",
            "name": "B",
            "tags_per_individual": 1,
            "tags": [{"id": "printed", "pattern": "\\d{5}", "header": "MUSEUM"}],
        }
    )
    assert choose_kind(only_printed, "16368", "printed", frozenset()) == "printed"


def test_readings_of_other_shapes_keep_their_kind() -> None:
    assert choose_kind(PROFILE, "MD00001", "letters", frozenset({"printed_a"})) == "letters"


def test_header_round_trips_and_is_checked() -> None:
    data = PROFILE.to_dict()
    assert data["tags"][1]["header"] == "NATIONAL MUSEUM OF CANADA"
    assert "header" not in data["tags"][0]
    assert Profile.from_dict(data) == PROFILE
    for bad in ("", "   ", "x" * 201, 12):
        with pytest.raises(ProfileError):
            Profile.from_dict(
                {
                    "schema_version": "1.0",
                    "name": "C",
                    "tags_per_individual": 1,
                    "tags": [{"id": "a", "pattern": "\\d{2}", "header": bad}],
                }
            )


class _Detector:
    def __init__(self, count: int) -> None:
        self.count = count

    def detect(self, image: Image.Image) -> list[DetectedLine]:
        quad = ((0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (0.0, 5.0))
        return [DetectedLine(quad=quad, score=0.9) for _ in range(self.count)]


class _Lines(LocalPipeline):
    """The local pipeline over lines already read: (tag reading or None, plain text)."""

    def __init__(self, lines: list[tuple[LineReading | None, str]]) -> None:
        self.model = "stub"
        self.version = "1.0"
        self._detector = _Detector(len(lines))
        self._lines = iter(lines)

    def _read_line(
        self, image: Image.Image, line: DetectedLine, profile: Profile
    ) -> tuple[LineReading | None, PlainLine]:
        reading, text = next(self._lines)
        quad = ((0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (0.0, 5.0))
        return reading, PlainLine(text=text, confidence=0.9, quad=quad, angle=0.0)


def _tag(text: str, tag_id: str) -> LineReading:
    return LineReading(
        quad=((0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (0.0, 5.0)),
        angle=0.0,
        readings=((text, 0.95, tag_id),),
        agrees=True,
        crop=Image.new("RGB", (10, 5)),
    )


def test_the_pipeline_reports_the_kind_its_printed_header_points_to() -> None:
    image = Image.new("RGB", (100, 100))
    pipeline = _Lines(
        [(None, "NATIONAL MUSEUM"), (None, "OF CANADA"), (_tag("16368", "plain"), "16368")]
    )
    [line] = pipeline.read(image, PROFILE)
    assert line.readings == (("16368", 0.95, "printed_a"),)
    # No header in the photo: the kind without a header.
    [line] = _Lines([(_tag("16368", "printed_a"), "16368")]).read(image, PROFILE)
    assert line.readings[0][2] == "plain"
