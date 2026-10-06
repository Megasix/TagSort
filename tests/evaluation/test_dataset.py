import json
from pathlib import Path

import pytest

from tagsort.evaluation.dataset import DatasetError, LabeledTag, load_dataset


def make(tmp_path: Path, *names: str) -> Path:
    (tmp_path / "photos").mkdir()
    for name in names:
        (tmp_path / "photos" / name).write_bytes(b"jpeg")
    return tmp_path


def test_jsonl_labels(tmp_path: Path) -> None:
    root = make(tmp_path, "a.jpg", "b.jpg", "c.jpg")
    lines = [
        {
            "image": "b.jpg",
            "session": "s1",
            "split": "test",
            "tag_id": "primary",
            "text": "GJ07966",
            "polygon": [[0, 0], [10, 0], [10, 5], [0, 5]],
        },
        {"image": "b.jpg", "session": "s1", "split": "test", "text": "?"},
        {"image": "a.jpg", "session": "s1", "split": "test", "text": None},
        {"image": "c.jpg", "session": "s2", "split": "train", "text": "A_REMPLIR"},
    ]
    (root / "labels.jsonl").write_text("\n".join(map(json.dumps, lines)) + "\n\n", encoding="utf-8")
    images = load_dataset(root)
    assert [i.path.name for i in images] == ["a.jpg", "b.jpg"]
    assert images[0].tags == ()
    assert images[1].tags == (
        LabeledTag("GJ07966", "primary", ((0, 0), (10, 0), (10, 5), (0, 5))),
        LabeledTag(None),
    )
    assert images[1].split == "test"
    assert load_dataset(root, split="train") == []


def test_csv_labels_with_semicolons(tmp_path: Path) -> None:
    root = make(tmp_path, "a.jpg", "b.jpg", "c.jpg")
    (root / "labels.csv").write_text(
        "﻿image;session;tag_id;text;notes\n"
        "a.jpg;2025-05-08;;GJ07966;\n"
        "a.jpg;2025-05-08;;0086;second tag\n"
        "b.jpg;2025-05-08;;-;\n"
        "c.jpg;2025-05-10;;A_REMPLIR;\n"
        "EXEMPLE_1.jpg;x;;MD1;\n",
        encoding="utf-8",
    )
    images = load_dataset(root)
    assert [(i.path.name, [t.text for t in i.tags]) for i in images] == [
        ("a.jpg", ["GJ07966", "0086"]),
        ("b.jpg", []),
    ]
    assert images[0].session == "2025-05-08"
    assert images[0].split is None


def test_empty_csv(tmp_path: Path) -> None:
    root = make(tmp_path)
    (root / "labels.csv").write_text("", encoding="utf-8")
    assert load_dataset(root) == []


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("not json", "invalid JSON"),
        ("[1]", "JSON object"),
        ('{"text": "x"}', "image is missing"),
        ('{"image": "missing.jpg"}', "not found"),
        ('{"image": "a.jpg", "text": 3}', "text must be a string"),
        ('{"image": "a.jpg", "text": "x", "tag_id": 3}', "tag_id must be a string"),
        ('{"image": "a.jpg", "text": "x", "polygon": [1, 2]}', "polygon"),
        ('{"image": "a.jpg", "split": 1}', "split must be a string"),
    ],
)
def test_malformed_labels(tmp_path: Path, line: str, message: str) -> None:
    root = make(tmp_path, "a.jpg")
    (root / "labels.jsonl").write_text(line + "\n", encoding="utf-8")
    with pytest.raises(DatasetError, match=message):
        load_dataset(root)


def test_session_must_stay_in_one_split(tmp_path: Path) -> None:
    root = make(tmp_path, "a.jpg", "b.jpg")
    (root / "labels.jsonl").write_text(
        '{"image": "a.jpg", "session": "s", "split": "train"}\n'
        '{"image": "b.jpg", "session": "s", "split": "test"}\n',
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="split by session"):
        load_dataset(root)


def test_missing_folders(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="no photos"):
        load_dataset(tmp_path)
    make(tmp_path)
    with pytest.raises(DatasetError, match="no labels"):
        load_dataset(tmp_path)
