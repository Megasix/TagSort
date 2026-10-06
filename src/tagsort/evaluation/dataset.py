"""Labeled datasets: a folder of photos and the true text of every tag.

The format is specified in ``docs/evaluation.md``. Labels come from ``labels.jsonl`` (one
JSON object per line) or, for hand-filled files, ``labels.csv``.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from tagsort.errors import TagSortError

__all__ = ["DatasetError", "LabeledImage", "LabeledTag", "load_dataset"]

UNREADABLE = "?"
NO_TAG = "-"
PLACEHOLDER = "A_REMPLIR"
PHOTO_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"})


class DatasetError(TagSortError, ValueError):
    """A dataset folder or its labels are malformed."""


@dataclass(frozen=True)
class LabeledTag:
    """The truth for one tag.

    Attributes:
        text: Exact text, or ``None`` when no person can read the tag.
        tag_id: Profile tag id, if the labeler gave one.
        polygon: Corners in original pixels, if the labeler drew them.
    """

    text: str | None
    tag_id: str | None = None
    polygon: tuple[tuple[float, float], ...] | None = None


@dataclass(frozen=True)
class LabeledImage:
    """A photo and every tag on it.

    Attributes:
        path: The photo file.
        session: Shooting session or site; photos of a session stay in the same split.
        split: Optional split name such as ``train`` or ``test``.
        tags: The tags on the photo; empty when the photo shows none.
    """

    path: Path
    session: str
    split: str | None = None
    tags: tuple[LabeledTag, ...] = field(default=())


def load_dataset(folder: str | Path, *, split: str | None = None) -> list[LabeledImage]:
    """Load the labeled photos of ``folder``, optionally keeping one split.

    Photos without labels, and labels still holding the ``A_REMPLIR`` placeholder, are
    left out. Images are listed in file name order.

    Raises:
        DatasetError: If the labels are malformed, name a missing photo, or put one
            session in several splits.
    """
    root = Path(folder)
    photos = root / "photos"
    if not photos.is_dir():
        raise DatasetError(f"{root}: no photos/ folder")
    if (root / "labels.jsonl").is_file():
        records = list(_read_jsonl(root / "labels.jsonl"))
    elif (root / "labels.csv").is_file():
        records = list(_read_csv(root / "labels.csv"))
    else:
        raise DatasetError(f"{root}: no labels.jsonl or labels.csv")

    images: dict[str, dict[str, object]] = {}
    session_splits: dict[str, str | None] = {}
    for where, record in records:
        name = record.get("image")
        if not isinstance(name, str) or not name:
            raise DatasetError(f"{where}: image is missing")
        if not (photos / name).is_file():
            raise DatasetError(f"{where}: photo {name!r} not found in {photos}")
        session = str(record.get("session") or "")
        record_split = record.get("split") or None
        if not isinstance(record_split, str | None):
            raise DatasetError(f"{where}: split must be a string")
        known = session_splits.setdefault(session, record_split)
        if known != record_split:
            raise DatasetError(
                f"{where}: session {session!r} is in splits {known!r} and {record_split!r}; "
                "split by session, never by photo"
            )
        entry = images.setdefault(name, {"session": session, "split": record_split, "tags": []})
        tag = _tag(record, where)
        if tag is not None:
            tags = entry["tags"]
            assert isinstance(tags, list)
            tags.append(tag)

    return [
        LabeledImage(
            path=photos / name,
            session=str(entry["session"]),
            split=entry["split"] if isinstance(entry["split"], str) else None,
            tags=tuple(entry["tags"]),  # type: ignore[arg-type]
        )
        for name, entry in sorted(images.items())
        if split is None or entry["split"] == split
    ]


def _tag(record: dict[str, object], where: str) -> LabeledTag | None:
    """Return the tag a record describes, or ``None`` for a photo without a tag."""
    text = record.get("text")
    if text is None or text in ("", NO_TAG):
        return None
    if not isinstance(text, str):
        raise DatasetError(f"{where}: text must be a string")
    tag_id = record.get("tag_id") or None
    if not isinstance(tag_id, str | None):
        raise DatasetError(f"{where}: tag_id must be a string")
    polygon = record.get("polygon")
    points: tuple[tuple[float, float], ...] | None = None
    if polygon is not None:
        try:
            points = tuple((float(x), float(y)) for x, y in polygon)  # type: ignore[attr-defined]
        except (TypeError, ValueError):
            raise DatasetError(f"{where}: polygon must be a list of [x, y] points") from None
    return LabeledTag(text=None if text == UNREADABLE else text, tag_id=tag_id, polygon=points)


def _read_jsonl(path: Path) -> Iterator[tuple[str, dict[str, object]]]:
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        where = f"{path.name}:{number}"
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise DatasetError(f"{where}: invalid JSON ({error.msg})") from None
        if not isinstance(record, dict):
            raise DatasetError(f"{where}: each line must be a JSON object")
        if record.get("text") == PLACEHOLDER:
            continue
        yield where, record


def _read_csv(path: Path) -> Iterator[tuple[str, dict[str, object]]]:
    text = path.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    if not lines:
        return
    header = lines[0]
    delimiter = ";" if header.count(";") > header.count(",") else ","
    for number, row in enumerate(csv.DictReader(lines, delimiter=delimiter), 2):
        record: dict[str, object] = {
            key: (value or "").strip() for key, value in row.items() if key
        }
        name = record.get("image")
        if not name or str(name).startswith("EXEMPLE_") or record.get("text") == PLACEHOLDER:
            continue
        yield f"{path.name}:{number}", record
