"""Image preprocessing: decoding, EXIF orientation, resizing and metadata-free encoding."""

from __future__ import annotations

import io
import os
import warnings
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from PIL import Image, ImageOps, UnidentifiedImageError

from tagsort.errors import ImageError

__all__ = ["ImageSource", "PreparedImage", "encode_jpeg", "prepare"]

_EXIF_ORIENTATION = 0x0112

# EXIF orientation -> clockwise rotation, in degrees, that ImageOps.exif_transpose applies.
# Orientations 2, 4, 5 and 7 also mirror the image; only the rotation is reported.
_ROTATION = {1: 0, 2: 0, 3: 180, 4: 180, 5: 90, 6: 90, 7: 270, 8: 270}


@runtime_checkable
class _ArrayLike(Protocol):
    @property
    def __array_interface__(self) -> dict[str, object]: ...


ImageSource = str | os.PathLike[str] | bytes | Image.Image | _ArrayLike
"""A file path, encoded image bytes, a PIL image or an array (height, width[, channels])."""


@dataclass(frozen=True)
class PreparedImage:
    """An image ready to be read.

    Attributes:
        image: The image, upright and in RGB, at most ``max_side`` pixels on its long edge.
        width: Width of the original image after EXIF rotation.
        height: Height of the original image after EXIF rotation.
        exif_rotation: Clockwise rotation, in degrees, applied to honor the EXIF orientation.
    """

    image: Image.Image
    width: int
    height: int
    exif_rotation: int

    @property
    def scale(self) -> float:
        """Ratio of the original size to the working size, at least 1."""
        return self.width / self.image.width


def prepare(source: ImageSource, *, max_side: int) -> PreparedImage:
    """Decode ``source``, apply its EXIF orientation and downscale it to ``max_side``.

    Raises:
        ImageError: If the image cannot be decoded or its format is not supported.
        FileNotFoundError: If ``source`` is a path that does not exist.
    """
    if max_side < 1:
        raise ValueError(f"max_side must be at least 1, not {max_side}")
    image = _open(source)
    try:
        orientation = image.getexif().get(_EXIF_ORIENTATION, 1)
        rotation = _ROTATION.get(orientation, 0) if isinstance(orientation, int) else 0
        upright = ImageOps.exif_transpose(image)
        upright = upright.convert("RGB")
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        raise ImageError(f"cannot decode image: {error}") from None
    width, height = upright.size
    if max(width, height) > max_side:
        upright = upright.copy()
        upright.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return PreparedImage(image=upright, width=width, height=height, exif_rotation=rotation)


def encode_jpeg(image: Image.Image, *, quality: int = 90) -> bytes:
    """Encode ``image`` as a baseline JPEG with no metadata (no EXIF, GPS or ICC profile)."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()


def _open(source: ImageSource) -> Image.Image:
    if isinstance(source, Image.Image):
        return source
    if isinstance(source, _ArrayLike) and not isinstance(source, bytes | str):
        try:
            return Image.fromarray(source)  # type: ignore[arg-type]
        except (TypeError, ValueError) as error:
            raise ImageError(f"unsupported array: {error}") from None
    if isinstance(source, bytes):
        stream: io.BytesIO | str | os.PathLike[str] = io.BytesIO(source)
    elif isinstance(source, str | os.PathLike):
        stream = source
    else:
        raise TypeError(
            f"image must be a path, bytes, a PIL image or an array, not {type(source).__name__}"
        )
    try:
        with warnings.catch_warnings():
            # Treat suspiciously large images as errors rather than warnings.
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            image = Image.open(stream)
            image.load()
    except FileNotFoundError:
        raise
    except UnidentifiedImageError:
        raise ImageError(
            "unsupported or corrupt image; use JPEG, PNG, WebP or TIFF "
            "(convert HEIC photos to JPEG)"
        ) from None
    except (OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise ImageError(f"cannot decode image: {error}") from None
    return image
