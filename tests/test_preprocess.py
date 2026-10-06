import io
from pathlib import Path

import pytest
from PIL import Image

from tagsort import ImageError
from tagsort.pipeline.preprocess import encode_jpeg, prepare
from tests.images import EXIF_ORIENTATION, GPS_IFD, jpeg_bytes, make_image


def dark_corner(image: Image.Image) -> str:
    """Name the corner holding the dark marker."""
    w, h = image.size
    corners = {
        "top-left": (2, 2),
        "top-right": (w - 3, 2),
        "bottom-right": (w - 3, h - 3),
        "bottom-left": (2, h - 3),
    }
    dark = [name for name, xy in corners.items() if sum(image.getpixel(xy)) < 100]  # type: ignore[arg-type]
    assert len(dark) == 1, dark
    return dark[0]


@pytest.mark.parametrize(
    ("orientation", "rotation", "size", "corner"),
    [
        (1, 0, (60, 40), "top-left"),
        (2, 0, (60, 40), "top-right"),
        (3, 180, (60, 40), "bottom-right"),
        (4, 180, (60, 40), "bottom-left"),
        (5, 90, (40, 60), "top-left"),
        (6, 90, (40, 60), "top-right"),
        (7, 270, (40, 60), "bottom-right"),
        (8, 270, (40, 60), "bottom-left"),
    ],
)
def test_exif_orientation_is_applied(
    orientation: int, rotation: int, size: tuple[int, int], corner: str
) -> None:
    prepared = prepare(jpeg_bytes(orientation=orientation), max_side=1000)
    assert prepared.exif_rotation == rotation
    assert (prepared.width, prepared.height) == size
    assert prepared.image.size == size
    assert dark_corner(prepared.image) == corner


def test_without_exif() -> None:
    buffer = io.BytesIO()
    make_image().save(buffer, format="PNG")
    prepared = prepare(buffer.getvalue(), max_side=1000)
    assert prepared.exif_rotation == 0
    assert prepared.scale == 1


def test_downscales_long_edge_and_keeps_original_size() -> None:
    prepared = prepare(make_image(400, 300), max_side=100)
    assert prepared.image.size == (100, 75)
    assert (prepared.width, prepared.height) == (400, 300)
    assert prepared.scale == 4


def test_downscaling_does_not_modify_the_callers_image() -> None:
    image = make_image(400, 300)
    prepare(image, max_side=100)
    assert image.size == (400, 300)


def test_accepts_path_and_str(tmp_path: Path) -> None:
    path = tmp_path / "tag.jpg"
    path.write_bytes(jpeg_bytes(orientation=6))
    assert prepare(path, max_side=1000).exif_rotation == 90
    assert prepare(str(path), max_side=1000).exif_rotation == 90


def test_converts_to_rgb() -> None:
    for mode in ("L", "RGBA", "P", "I;16"):
        prepared = prepare(Image.new(mode, (10, 10)), max_side=100)
        assert prepared.image.mode == "RGB"


class FakeArray(bytearray):
    """Minimal buffer exposing the array interface, like a numpy array."""

    def __init__(self, width: int, height: int) -> None:
        super().__init__(width * height * 3)
        self.__array_interface__: dict[str, object] = {
            "shape": (height, width, 3),
            "typestr": "|u1",
            "version": 3,
        }


def test_accepts_arrays() -> None:
    prepared = prepare(FakeArray(30, 20), max_side=100)
    assert prepared.image.size == (30, 20)


def test_rejects_bad_arrays() -> None:
    array = FakeArray(3, 2)
    array.__array_interface__["typestr"] = "<c16"
    with pytest.raises(ImageError, match="unsupported array"):
        prepare(array, max_side=100)


@pytest.mark.parametrize("data", [b"", b"not an image", b"\x00\x00\x00\x18ftypheic" + bytes(64)])
def test_undecodable_bytes(data: bytes) -> None:
    with pytest.raises(ImageError, match="unsupported or corrupt"):
        prepare(data, max_side=100)


def test_truncated_jpeg() -> None:
    data = jpeg_bytes(make_image(200, 200))
    with pytest.raises(ImageError, match="cannot decode"):
        prepare(data[: len(data) // 2], max_side=100)


def test_decompression_bomb(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)
    with pytest.raises(ImageError, match="cannot decode"):
        prepare(jpeg_bytes(make_image(60, 40)), max_side=100)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        prepare(tmp_path / "missing.jpg", max_side=100)


def test_wrong_type() -> None:
    with pytest.raises(TypeError, match="image must be a path"):
        prepare(42, max_side=100)  # type: ignore[arg-type]


def test_max_side_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_side"):
        prepare(make_image(), max_side=0)


def test_encoded_jpeg_has_no_metadata() -> None:
    source = jpeg_bytes(orientation=6, gps=True)
    with Image.open(io.BytesIO(source)) as original:
        assert GPS_IFD in original.getexif()
    encoded = encode_jpeg(prepare(source, max_side=1000).image)
    with Image.open(io.BytesIO(encoded)) as image:
        assert image.format == "JPEG"
        assert not image.getexif()
        assert EXIF_ORIENTATION not in image.getexif()
        assert "icc_profile" not in image.info
        assert "exif" not in image.info


def test_decoding_failure_after_open(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(image: Image.Image) -> Image.Image:
        raise OSError("broken data stream")

    monkeypatch.setattr("tagsort.pipeline.preprocess.ImageOps.exif_transpose", broken)
    with pytest.raises(ImageError, match="broken data stream"):
        prepare(make_image(), max_side=100)


@pytest.mark.parametrize("orientation", [0, 9])
def test_invalid_orientation_is_ignored(orientation: int) -> None:
    assert prepare(jpeg_bytes(orientation=orientation), max_side=100).exif_rotation == 0
