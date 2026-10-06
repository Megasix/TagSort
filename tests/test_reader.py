import threading
from dataclasses import dataclass, field
from typing import Any

import pytest
from jsonschema import Draft202012Validator

import tagsort
from tagsort import Profile, ProviderTag, Reader, ReadResult, VisionAnswer, VisionRequest
from tagsort.fallback.base import BoxFormat, Legibility
from tests.helpers import DATA
from tests.images import jpeg_bytes, make_image

PROFILE = Profile.from_file(DATA / "museum_a.json")


def raw(
    text: str = "MD04127",
    legibility: Legibility = "certain",
    alternatives: tuple[str, ...] = (),
    box: tuple[float, float, float, float] = (10, 20, 50, 30),
    angle: int = 0,
) -> ProviderTag:
    return ProviderTag(
        text=text, legibility=legibility, alternatives=alternatives, box=box, angle=angle
    )


@dataclass
class FakeProvider:
    tags: tuple[ProviderTag, ...] = ()
    name: str = "fake"
    model: str = "fake-1"
    max_side: int = 1000
    box_format: BoxFormat = "pixels_xyxy"
    requests: list[VisionRequest] = field(default_factory=list)

    def read(self, request: VisionRequest) -> VisionAnswer:
        self.requests.append(request)
        return VisionAnswer(tags=self.tags, model="fake-1.2")


def read(*tags: ProviderTag, image: Any = None, **reader_args: Any) -> ReadResult:
    provider = FakeProvider(tags=tags)
    reader = Reader(PROFILE, backend=provider, **reader_args)
    return reader.read(image if image is not None else make_image(100, 50))


def test_result_conforms_to_schema(result_validator: Draft202012Validator) -> None:
    result = read(
        raw(),
        raw("AB-12", "uncertain", ("AB-123",)),
        raw("", "unreadable"),
        raw("??", "unreadable", ("MD04127",)),
        raw("XYZ", "certain", angle=90),
    )
    result_validator.validate(result.to_dict())
    assert result.engine_version == tagsort.__version__
    assert result.model_version == "fake:fake-1.2"
    assert set(result.timings_ms) == {"preprocess", "api"}


def test_certain_reading_that_fits_the_profile_is_accepted() -> None:
    (tag,) = read(raw()).tags
    assert (tag.tag_id, tag.text, tag.status, tag.source) == (
        "primary",
        "MD04127",
        "accepted",
        "fallback",
    )
    assert tag.confidence >= 0.9
    assert tag.candidates == ()


def test_uncertain_reading_needs_review() -> None:
    (tag,) = read(raw(legibility="uncertain", alternatives=("MD04121",))).tags
    assert tag.status == "review"
    assert tag.text == "MD04127"
    assert [c.text for c in tag.candidates] == ["MD04121"]
    assert tag.candidates[0].confidence < tag.confidence


def test_alternative_that_fits_the_profile_wins_over_one_that_does_not() -> None:
    (tag,) = read(raw("MDO4127", alternatives=("MD04127",))).tags
    assert tag.text == "MD04127"
    assert tag.tag_id == "primary"
    assert tag.status == "review"
    assert [c.text for c in tag.candidates] == ["MDO4127"]


def test_reading_outside_the_profile_is_never_accepted() -> None:
    (tag,) = read(raw("HELLO"), accept_threshold=0.1, review_threshold=0.0).tags
    assert tag.status == "review"
    assert tag.tag_id is None


def test_unreadable_without_text() -> None:
    (tag,) = read(raw("", "unreadable")).tags
    assert tag.status == "unreadable"
    assert tag.text is None
    assert tag.confidence == 0
    assert tag.candidates == ()


def test_unreadable_keeps_guesses_as_candidates() -> None:
    (tag,) = read(raw("MD0412?", "unreadable", ("MD04127",)), review_threshold=0.5).tags
    assert tag.status == "unreadable"
    assert tag.text is None
    assert [c.text for c in tag.candidates] == ["MD04127", "MD0412?"]


def test_provider_unreadable_stays_unreadable_even_if_the_guess_fits() -> None:
    (tag,) = read(raw("MD04127", "unreadable")).tags
    assert tag.status == "unreadable"
    assert [c.text for c in tag.candidates] == ["MD04127"]


def test_candidates_are_sorted() -> None:
    (tag,) = read(raw("X1", "uncertain", ("MD04127", "AB-123", "Y2"))).tags
    confidences = [tag.confidence] + [c.confidence for c in tag.candidates]
    assert confidences == sorted(confidences, reverse=True)


def test_no_tags() -> None:
    result = read()
    assert result.tags == ()


@pytest.mark.parametrize(
    ("angle", "first_corner"),
    [(0, (10.0, 20.0)), (90, (50.0, 20.0)), (180, (50.0, 30.0)), (270, (10.0, 30.0))],
)
def test_polygon_starts_at_top_left_of_upright_text(
    angle: int, first_corner: tuple[float, float]
) -> None:
    (tag,) = read(raw(angle=angle)).tags
    assert tag.angle == angle
    assert tag.polygon[0] == first_corner
    assert set(tag.polygon) == {(10.0, 20.0), (50.0, 20.0), (50.0, 30.0), (10.0, 30.0)}


def test_polygon_is_in_original_pixels() -> None:
    provider = FakeProvider(tags=(raw(box=(10, 20, 50, 30)),), max_side=100)
    result = Reader(PROFILE, backend=provider).read(make_image(400, 200))
    request = provider.requests[0]
    assert (request.width, request.height) == (100, 50)
    assert result.image.width == 400
    assert result.tags[0].polygon[0] == (40.0, 80.0)


def test_exif_rotation_is_reported_and_image_sent_upright() -> None:
    provider = FakeProvider()
    result = Reader(PROFILE, backend=provider).read(jpeg_bytes(make_image(60, 40), orientation=6))
    assert (result.image.width, result.image.height, result.image.exif_rotation) == (40, 60, 90)
    request = provider.requests[0]
    assert (request.width, request.height) == (40, 60)
    assert request.jpeg.startswith(b"\xff\xd8")
    assert b"Exif" not in request.jpeg
    assert "- primary: MD\\d{5}" in request.instructions


def test_read_batch_is_lazy() -> None:
    provider = FakeProvider(tags=(raw(),))
    reader = Reader(PROFILE, backend=provider)
    results = reader.read_batch(make_image() for _ in range(3))
    assert provider.requests == []
    assert len(list(results)) == 3
    assert len(provider.requests) == 3


def test_reader_is_thread_safe() -> None:
    reader = Reader(PROFILE, backend=FakeProvider(tags=(raw(),)))
    results: list[ReadResult] = []
    threads = [
        threading.Thread(target=lambda: results.append(reader.read(make_image()))) for _ in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(results) == 8
    assert all(r.tags[0].text == "MD04127" for r in results)


def test_reader_validation() -> None:
    with pytest.raises(TypeError, match="profile must be a Profile"):
        Reader({"name": "x"}, backend=FakeProvider())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="backend must be 'local', a LocalPipeline"):
        Reader(PROFILE, backend="api")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="only used with the local backend"):
        Reader(PROFILE, backend=FakeProvider(), fallback=FakeProvider())
    with pytest.raises(TypeError, match="fallback must be a VisionProvider"):
        Reader(PROFILE, backend=FakeProvider(), fallback="gemini")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="thresholds"):
        Reader(PROFILE, backend=FakeProvider(), accept_threshold=0.2, review_threshold=0.5)
    assert Reader(PROFILE, backend=FakeProvider()).profile is PROFILE


def test_digits_that_read_differently_upside_down_need_review() -> None:
    profile = Profile.from_dict(
        {
            "schema_version": "1.0",
            "name": "Digits",
            "tags_per_individual": 1,
            "tags": [{"id": "number", "pattern": "\\d{4}"}],
        }
    )
    (tag,) = Reader(profile, backend=FakeProvider(tags=(raw("9800"),))).read(make_image()).tags
    assert tag.status == "review"
    assert tag.text == "9800"
    assert [c.text for c in tag.candidates] == ["0086"]


@pytest.mark.parametrize("text", ["GJ07966", "8008", "1001"])
def test_readings_without_a_different_upside_down_reading_are_kept(text: str) -> None:
    profile = Profile.from_dict(
        {
            "schema_version": "1.0",
            "name": "Mixed",
            "tags_per_individual": 1,
            "tags": [{"id": "number", "pattern": "\\d{4}"}, {"id": "code", "pattern": "GJ\\d{5}"}],
        }
    )
    (tag,) = Reader(profile, backend=FakeProvider(tags=(raw(text),))).read(make_image()).tags
    assert tag.status == "accepted"


def test_upside_down_reading_outside_the_profile_is_ignored() -> None:
    # "MOS" upside down is "SOW", which does not fit the museum_a profile.
    (tag,) = read(raw("MD04127", alternatives=())).tags
    assert tag.status == "accepted"


def test_upside_down_reading_already_listed_is_not_duplicated() -> None:
    profile = Profile.from_dict(
        {
            "schema_version": "1.0",
            "name": "Digits",
            "tags_per_individual": 1,
            "tags": [{"id": "number", "pattern": "\\d{4}"}],
        }
    )
    provider = FakeProvider(tags=(raw("9800", "uncertain", ("0086",)),))
    (tag,) = Reader(profile, backend=provider).read(make_image()).tags
    assert [c.text for c in tag.candidates] == ["0086"]


@dataclass
class SlowProvider(FakeProvider):
    """Answers after a delay and records how many requests overlap."""

    delay: float = 0.05
    active: int = 0
    peak: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def read(self, request: VisionRequest) -> VisionAnswer:
        import time

        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(self.delay)
        with self.lock:
            self.active -= 1
        width = request.width
        return VisionAnswer(tags=(raw(text=f"GJ{width:05d}"),), model="slow")


def test_parallel_batch_keeps_order_and_overlaps_requests() -> None:
    profile = Profile.from_dict(
        {
            "schema_version": "1.0",
            "name": "W",
            "tags_per_individual": 1,
            "tags": [{"id": "primary", "pattern": "GJ\\d{5}"}],
        }
    )
    provider = SlowProvider()
    reader = Reader(profile, backend=provider)
    widths = [20 + i for i in range(16)]
    results = list(reader.read_batch((make_image(w, 10) for w in widths), workers=4))
    assert [r.tags[0].text for r in results] == [f"GJ{w:05d}" for w in widths]
    assert provider.peak == 4


def test_parallel_batch_holds_a_bounded_window() -> None:
    reader = Reader(PROFILE, backend=SlowProvider(delay=0.0))
    consumed = 0

    def images() -> Any:
        nonlocal consumed
        for _ in range(100):
            consumed += 1
            yield make_image()

    batch = reader.read_batch(images(), workers=3)
    next(batch)
    assert consumed <= 2 * 3 + 1
    batch.close()


def test_parallel_batch_stops_at_the_first_error() -> None:
    from tagsort import ImageError

    reader = Reader(PROFILE, backend=SlowProvider(delay=0.0))
    images: list[Any] = [make_image(), b"not an image", make_image()]
    batch = reader.read_batch(images, workers=2)
    next(batch)
    with pytest.raises(ImageError):
        next(batch)


def test_workers_must_be_positive() -> None:
    with pytest.raises(ValueError, match="workers"):
        list(Reader(PROFILE, backend=FakeProvider()).read_batch([], workers=0))
