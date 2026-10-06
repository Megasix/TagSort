import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from tagsort import Profile
from tagsort.cli import main
from tagsort.pipeline.local import LineReading, LocalPipeline

PROFILE = {
    "schema_version": "1.0",
    "name": "Lot",
    "tags_per_individual": 1,
    "tags": [{"id": "primary", "pattern": "GJ\\d{5}"}],
}


class StubPipeline(LocalPipeline):
    def __init__(self, model: str = "stub") -> None:
        self.model = model
        self.version = "1.0"

    def read(self, image: Image.Image, profile: Profile) -> list[LineReading]:
        return [
            LineReading(
                ((1.0, 1.0), (9.0, 1.0), (9.0, 4.0), (1.0, 4.0)),
                0.0,
                (("GJ07966", 0.97, "primary"),),
                True,
                Image.new("RGB", (8, 3)),
            )
        ]


@pytest.fixture
def photos(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr("tagsort.pipeline.local.LocalPipeline", StubPipeline)
    folder = tmp_path / "photos"
    folder.mkdir()
    for name in ("b.jpg", "a.png"):
        Image.new("RGB", (40, 30), "white").save(folder / name)
    (folder / "notes.txt").write_text("not a photo")
    (folder / "broken.jpg").write_bytes(b"not an image")
    (tmp_path / "profile.json").write_text(json.dumps(PROFILE))
    return folder


def test_read_prints_one_line_per_photo_and_keeps_going(
    photos: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "results"
    code = main(
        ["read", str(photos), "--profile", str(tmp_path / "profile.json"), "--out", str(out)]
    )
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert code == 1, "one photo failed"
    assert [Path(line["file"]).name for line in lines] == ["a.png", "b.jpg", "broken.jpg"]
    assert lines[0]["result"]["tags"][0]["text"] == "GJ07966"
    assert "error" in lines[2]
    assert sorted(p.name for p in out.iterdir()) == ["a.json", "b.json"]


def test_read_single_files_succeeds(
    photos: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["read", str(photos / "a.png"), "--profile", str(tmp_path / "profile.json")]) == 0
    assert len(capsys.readouterr().out.splitlines()) == 1


def test_read_reports_setup_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["read", str(tmp_path), "--profile", str(tmp_path / "missing.json")]) == 1
    assert "tagsort:" in capsys.readouterr().err


def test_read_with_a_fallback_needs_its_key(
    photos: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    code = main(
        ["read", str(photos), "--profile", str(tmp_path / "profile.json"), "--fallback", "gemini"]
    )
    assert code == 1
    assert "GEMINI_API_KEY is not set" in capsys.readouterr().err


def test_serve_needs_a_token(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("TAGSORT_API_TOKEN", raising=False)
    monkeypatch.delenv("TAGSORT_ALLOW_NO_TOKEN", raising=False)
    assert main(["serve"]) == 1
    assert "TAGSORT_API_TOKEN" in capsys.readouterr().err


def test_serve_starts_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    import uvicorn

    calls: list[dict[str, Any]] = []
    monkeypatch.setenv("TAGSORT_API_TOKEN", "t")
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: calls.append(kwargs))
    assert main(["serve", "--host", "0.0.0.0", "--port", "9000"]) == 0
    assert calls == [{"host": "0.0.0.0", "port": 9000, "log_level": "warning"}]


def test_serve_without_the_extra(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "uvicorn", None)
    assert main(["serve"]) == 1
    assert "tagsort[server]" in capsys.readouterr().err
