"""Synthetic test images: no photo ever enters the repository."""

import io

from PIL import Image

EXIF_ORIENTATION = 0x0112
GPS_IFD = 0x8825


def make_image(
    width: int = 60, height: int = 40, color: tuple[int, int, int] = (200, 50, 50)
) -> Image.Image:
    image = Image.new("RGB", (width, height), color)
    # A dark top-left corner makes rotations and mirrors detectable.
    image.paste((0, 0, 0), (0, 0, width // 4, height // 4))
    return image


def jpeg_bytes(
    image: Image.Image | None = None, *, orientation: int | None = None, gps: bool = False
) -> bytes:
    image = image or make_image()
    exif = Image.Exif()
    if orientation is not None:
        exif[EXIF_ORIENTATION] = orientation
    if gps:
        exif[GPS_IFD] = {2: (48.0, 51.0, 24.0)}  # GPSLatitude
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif.tobytes(), quality=95)
    return buffer.getvalue()
