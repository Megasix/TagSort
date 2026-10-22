"""Barcodes and 2D codes: decoded before text, accepted when they fit the profile."""

from dataclasses import dataclass, field

import numpy as np
import pytest
import zxingcpp
from jsonschema import Draft202012Validator
from PIL import Image

from tagsort import (
    Profile,
    ProviderTag,
    Reader,
    TagSortError,
    Usage,
    VisionAnswer,
    VisionRequest,
)
from tagsort.fallback.base import BoxFormat
from tagsort.pipeline.codes import DecodedCode, ZxingDecoder, fit_code
from tagsort.pipeline.local import LineReading, LocalPipeline, PlainLine

PROFILE = Profile.from_dict(
    {
        "schema_version": "1.0",
        "name": "Lot",
        "tags_per_individual": 1,
        "tags": [
            {"id": "primary", "pattern": "GJ\\d{5}"},
            {"id": "loan", "pattern": "XYZ\\d{4}"},
        ],
    }
)


def photo(
    content: str,
    symbology: zxingcpp.BarcodeFormat = zxingcpp.BarcodeFormat.QRCode,
    *,
    turn: float = 0,
    size: tuple[int, int] = (900, 700),
) -> Image.Image:
    """A white photo with one generated code, turned counterclockwise by ``turn``."""
    code = np.asarray(zxingcpp.create_barcode(content, symbology).to_image(scale=6))
    image = Image.fromarray(code).convert("RGB")
    if turn:
        image = image.rotate(turn, expand=True, fillcolor="white")
    canvas = Image.new("RGB", size, "white")
    canvas.paste(image, (150, 150))
    return canvas


class StubPipeline(LocalPipeline):
    """A local pipeline that 'reads' fixed lines."""

    def __init__(self, *texts: str) -> None:
        self.model = "stub"
        self.version = "1.0"
        self.texts = texts

    def read(self, image: Image.Image, profile: Profile) -> list[LineReading]:
        return [
            LineReading(
                quad=((10.0, 10.0), (90.0, 10.0), (90.0, 30.0), (10.0, 30.0)),
                angle=0.0,
                readings=((text, 0.97, profile.match(text) or "primary"),),
                agrees=True,
                crop=Image.new("RGB", (80, 20), "white"),
            )
            for text in self.texts
        ]

    def read_all(
        self, image: Image.Image, profile: Profile
    ) -> tuple[list[LineReading], list[PlainLine]]:
        return self.read(image, profile), []


@dataclass
class FakeApi:
    tags: tuple[ProviderTag, ...] = ()
    name: str = "fake"
    model: str = "fake-1"
    max_side: int = 1024
    box_format: BoxFormat = "pixels_xyxy"
    requests: list[VisionRequest] = field(default_factory=list)

    def read(self, request: VisionRequest) -> VisionAnswer:
        self.requests.append(request)
        return VisionAnswer(tags=self.tags, model="fake-1", usage=Usage(10, 1))


def test_fit_code_tries_the_whole_text_then_its_parts() -> None:
    assert fit_code("GJ07470", PROFILE) == [("GJ07470", "primary")]
    assert fit_code("GJ 07470", PROFILE) == [("GJ07470", "primary")]
    assert fit_code("https://museum.test/specimen/gj07470?lang=en", PROFILE) == [
        ("GJ07470", "primary")
    ]
    assert fit_code("4006381333931", PROFILE) == []
    # Two numbers in one code: a person decides.
    assert fit_code("GJ07470;XYZ1234", PROFILE) == [("GJ07470", "primary"), ("XYZ1234", "loan")]


@pytest.mark.parametrize(
    ("symbology", "turn", "angle"),
    [
        (zxingcpp.BarcodeFormat.QRCode, 0, 0),
        (zxingcpp.BarcodeFormat.QRCode, 30, 330),
        (zxingcpp.BarcodeFormat.DataMatrix, 90, 270),
        (zxingcpp.BarcodeFormat.Code128, 0, 0),
    ],
)
def test_zxing_decodes_codes_in_any_orientation(
    symbology: zxingcpp.BarcodeFormat, turn: float, angle: float
) -> None:
    (code,) = ZxingDecoder().decode(photo("GJ07470", symbology, turn=turn))
    assert code.text == "GJ07470"
    # Clockwise from upright, as every TagSort angle.
    assert min(abs(code.angle - angle), 360 - abs(code.angle - angle)) < 3
    assert len(code.quad) == 4


def test_a_qr_tag_is_read_with_no_ocr(result_validator: Draft202012Validator) -> None:
    pipeline = StubPipeline()
    result = Reader(PROFILE, backend=pipeline).read(photo("GJ07470"))
    (tag,) = result.tags
    assert (tag.text, tag.tag_id, tag.status, tag.source, tag.confidence) == (
        "GJ07470",
        "primary",
        "accepted",
        "barcode",
        1.0,
    )
    # Corners in the original image, around where the code was pasted.
    xs = [x for x, _ in tag.polygon]
    assert min(xs) >= 150
    assert max(xs) < 900
    assert "codes" in result.timings_ms
    result_validator.validate(result.to_dict())


def test_the_text_printed_under_a_barcode_is_not_returned_twice() -> None:
    pipeline = StubPipeline("XYZ1234", "GJ07470")
    result = Reader(PROFILE, backend=pipeline).read(
        photo("XYZ1234", zxingcpp.BarcodeFormat.Code128)
    )
    assert [(tag.text, tag.source) for tag in result.tags] == [
        ("XYZ1234", "barcode"),
        ("GJ07470", "local"),
    ]


def test_a_code_with_two_numbers_needs_review() -> None:
    (tag,) = Reader(PROFILE, backend=StubPipeline()).read(photo("GJ07470;XYZ1234")).tags
    assert tag.status == "review"
    assert [c.text for c in tag.candidates] == ["XYZ1234"]


def test_codes_outside_the_profile_are_other_texts_with_open_reading() -> None:
    image = photo("4006381333931", zxingcpp.BarcodeFormat.EAN13)
    assert Reader(PROFILE, backend=StubPipeline()).read(image).tags == ()
    result = Reader(PROFILE, backend=StubPipeline(), open_reading=True).read(image)
    (other,) = result.other_texts
    assert (other.text, other.confidence) == ("4006381333931", 1.0)


def test_codes_can_be_turned_off_or_replaced() -> None:
    image = photo("GJ07470")
    assert Reader(PROFILE, backend=StubPipeline(), codes=False).read(image).tags == ()

    class Fixed:
        def decode(self, image: Image.Image) -> list[DecodedCode]:
            quad = ((1.0, 1.0), (5.0, 1.0), (5.0, 5.0), (1.0, 5.0))
            return [DecodedCode("GJ00001", "QR Code", quad, 0.0)]

    blank = Image.new("RGB", (100, 100), "white")
    (tag,) = Reader(PROFILE, backend=StubPipeline(), codes=Fixed()).read(blank).tags
    assert tag.text == "GJ00001"
    with pytest.raises(TypeError):
        Reader(PROFILE, backend=StubPipeline(), codes="zxing")  # type: ignore[arg-type]


def test_codes_need_the_extra_only_when_asked_for(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tagsort.pipeline.reader.zxing_installed", lambda: False)
    # Without the extra, codes are simply not decoded...
    assert Reader(PROFILE, backend=StubPipeline()).read(photo("GJ07470")).tags == ()
    # ...unless they were required.
    with pytest.raises(TagSortError, match="codes extra"):
        Reader(PROFILE, backend=StubPipeline(), codes=True)


def test_codes_also_come_first_with_an_api_backend() -> None:
    api = FakeApi(
        tags=(
            ProviderTag(
                text="GJ07470", legibility="certain", alternatives=(), box=(1, 1, 50, 20), angle=0
            ),
        )
    )
    result = Reader(PROFILE, backend=api).read(photo("GJ07470"))
    assert [(tag.text, tag.source) for tag in result.tags] == [("GJ07470", "barcode")]
