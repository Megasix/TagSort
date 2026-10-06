import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from tests.helpers import ROOT


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("try_batch", ROOT / "eval" / "try_batch.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["try_batch"] = module
    spec.loader.exec_module(module)
    return module


try_batch = _load()


def test_load_labels_skips_examples_and_keeps_empty_photos(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    path.write_text(
        "﻿image,session,tag_id,text,notes\n"
        "EXEMPLE_1.jpg,s,primary,MD1,\n"
        "IMG_1.jpg,s,primary,MD04127,\n"
        "IMG_1.jpg,s,secondary,AB-123,\n"
        "IMG_2.jpg,s,,,no tag\n"
        "IMG_3.jpg,s,primary,?,unreadable\n"
        "IMG_4.jpg,s,,-,\n"
        "IMG_5.jpg,s,,A_REMPLIR,\n",
        encoding="utf-8",
    )
    assert try_batch.load_labels(path) == {
        "IMG_1.jpg": [("primary", "MD04127"), ("secondary", "AB-123")],
        "IMG_2.jpg": [],
        "IMG_3.jpg": [("primary", "?")],
        "IMG_4.jpg": [],
    }


def test_load_labels_accepts_semicolons(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    path.write_text(
        "image;session;tag_id;text;notes\nIMG_1.jpg;s;;MD04127;a, b\n", encoding="utf-8"
    )
    assert try_batch.load_labels(path) == {"IMG_1.jpg": [("", "MD04127")]}


def test_prefilled_file_has_no_labels_yet(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    path.write_text("image,session,tag_id,text,notes\nIMG_1.jpg,s,,A_REMPLIR,\n", encoding="utf-8")
    assert try_batch.load_labels(path) == {}


def test_prices_cover_default_models() -> None:
    # DeepSeek's pricing page could not be read on 2026-10-06; its cost is reported as unknown.
    for name, (cls, _) in try_batch.PROVIDERS.items():
        assert cls.default_model in try_batch.PRICES or name == "deepseek"
