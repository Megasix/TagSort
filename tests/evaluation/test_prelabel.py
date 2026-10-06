import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from tagsort import Profile, ProviderTag, Tag, Usage, VisionAnswer, VisionRequest
from tagsort.evaluation import cli
from tagsort.evaluation.dataset import load_dataset
from tagsort.evaluation.metrics import summarize
from tagsort.evaluation.prelabel import build_rows, prelabel, session_of
from tagsort.evaluation.report import Report
from tagsort.fallback.base import BoxFormat
from tests.images import jpeg_bytes, make_image

PROFILE = {
    "schema_version": "1.0",
    "name": "Test",
    "tags_per_individual": 1,
    "tags": [{"id": "primary", "pattern": "GJ\\d{5}"}],
}
BOX = ((0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (0.0, 5.0))


def tag(text: str | None, status: Any = "accepted") -> Tag:
    if text is None:
        status = "unreadable"
    return Tag(
        tag_id="primary" if text else None,
        text=text,
        confidence=0.9 if status == "accepted" else 0.3,
        status=status,
        polygon=BOX,
        angle=0,
        source="fallback",
    )


def test_session_from_file_name_or_exif(tmp_path: Path) -> None:
    assert session_of(Path("20250508_140652.jpg")) == "2025-05-08"
    assert session_of(Path("PXL_20240102_101010.jpg")) == "2024-01-02"
    photo = tmp_path / "photo.jpg"
    image = make_image()
    exif = image.getexif()
    exif[0x0132] = "2023:07:14 10:00:00"
    image.save(photo, exif=exif.tobytes())
    assert session_of(photo) == "2023-07-14"
    plain = tmp_path / "plain.jpg"
    plain.write_bytes(jpeg_bytes())
    assert session_of(plain) == ""
    assert session_of(tmp_path / "missing.jpg") == ""


def test_rows_when_both_agree_disagree_or_find_nothing() -> None:
    agree = build_rows("a.jpg", "s", [tag("GJ07966")], [tag("GJ07966")])
    assert [(r.text, r.ai_1, r.ai_2, r.priority) for r, _ in agree] == [
        ("GJ07966", "GJ07966", "GJ07966", 3)
    ]
    differ = build_rows("a.jpg", "s", [tag("GJ07966")], [tag("GJ07968")])
    assert [(r.ai_1, r.ai_2, r.priority) for r, _ in differ] == [("GJ07966", "GJ07968", 0)]
    only_second = build_rows("a.jpg", "s", [], [tag("GJ07966")])
    assert [(r.text, r.ai_1, r.ai_2, r.priority) for r, _ in only_second] == [
        ("GJ07966", "", "GJ07966", 0)
    ]
    unsure = build_rows("a.jpg", "s", [tag("GJ07966", "review")], [tag("GJ07966", "review")])
    assert unsure[0][0].priority == 2
    nothing = build_rows("a.jpg", "s", [], [])
    assert [(r.text, r.priority) for r, t in nothing] == [("-", 1)]
    assert nothing[0][1] is None
    single = build_rows("a.jpg", "s", [tag(None)], None)
    assert [(r.text, r.ai_2, r.priority) for r, _ in single] == [("?", "", 0)]
    single_ok = build_rows("a.jpg", "s", [tag("GJ07966")], None)
    assert single_ok[0][0].priority == 3


@dataclass
class FakeProvider:
    texts: dict[int, str] = field(default_factory=dict)
    name: str = "fake"
    model: str = "fake-1"
    max_side: int = 200
    box_format: BoxFormat = "pixels_xyxy"
    calls: int = 0
    closed: bool = False

    def read(self, request: VisionRequest) -> VisionAnswer:
        self.calls += 1
        text = self.texts.get(request.width)
        tags = (
            (
                ProviderTag(
                    text=text, legibility="certain", alternatives=(), box=(2, 2, 30, 12), angle=90
                ),
            )
            if text
            else ()
        )
        return VisionAnswer(tags=tags, model=self.model, usage=Usage(10, 1))

    def close(self) -> None:
        self.closed = True


def dataset(tmp_path: Path) -> Path:
    (tmp_path / "photos").mkdir()
    for name, width in (("20250508_a.jpg", 60), ("20250508_b.jpg", 70), ("20250510_c.jpg", 80)):
        (tmp_path / "photos" / name).write_bytes(jpeg_bytes(make_image(width, 40)))
    (tmp_path / "photos" / "notes.txt").write_text("ignored")
    (tmp_path / "profile.json").write_text(json.dumps(PROFILE), encoding="utf-8")
    return tmp_path


def test_prelabel_writes_labels_page_and_provenance(tmp_path: Path) -> None:
    root = dataset(tmp_path)
    first = FakeProvider(texts={60: "GJ07966", 70: "GJ07967"}, name="one")
    second = FakeProvider(texts={60: "GJ07966", 70: "GJ07961"}, name="two", model="m2")
    rows = prelabel(root, profile=Profile.from_dict(PROFILE), providers=[first, second])
    assert len(rows) == 3
    with (root / "labels.csv").open(encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(handle))
    assert [
        (r["image"], r["session"], r["text"], r["ai_1"], r["ai_2"], r["verified"]) for r in records
    ] == [
        ("20250508_a.jpg", "2025-05-08", "GJ07966", "GJ07966", "GJ07966", ""),
        ("20250508_b.jpg", "2025-05-08", "GJ07967", "GJ07967", "GJ07961", ""),
        ("20250510_c.jpg", "2025-05-10", "-", "", "", ""),
    ]
    page = (root / "review.html").read_text(encoding="utf-8")
    assert (
        page.index("20250508_b.jpg") < page.index("20250510_c.jpg") < page.index("20250508_a.jpg")
    ), "disagreements first, then photos where nothing was found, then agreements"
    assert page.count("data:image/jpeg;base64,") == 3 + 2  # photo thumbnails + tag crops
    assert "Development and training tool" in page
    assert json.loads((root / "prelabel.json").read_text())["models"] == ["one:fake-1", "two:m2"]
    assert (root / "predictions" / "one-fake-1" / "20250508_a.json").is_file()

    # Nothing is verified yet, so no photo counts in an evaluation.
    assert load_dataset(root) == []
    with pytest.raises(FileExistsError):
        prelabel(root, profile=Profile.from_dict(PROFILE), providers=[first])
    prelabel(root, profile=Profile.from_dict(PROFILE), providers=[first], overwrite=True)
    assert first.calls == 3, "answers are cached"


def test_provider_count_is_checked(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="one or two"):
        prelabel(tmp_path, profile=Profile.from_dict(PROFILE), providers=[])


def test_only_verified_photos_count(tmp_path: Path) -> None:
    root = dataset(tmp_path)
    (root / "labels.csv").write_text(
        "image,session,tag_id,text,verified,notes,ai_1,ai_2\n"
        "20250508_a.jpg,s,,GJ07966,yes,,GJ07966,GJ07966\n"
        "20250508_b.jpg,s,,GJ07967,oui,,,\n"
        "20250508_b.jpg,s,,GJ07999,,,,\n"
        "20250510_c.jpg,t,,-,X,,,\n",
        encoding="utf-8",
    )
    assert [i.path.name for i in load_dataset(root)] == ["20250508_a.jpg", "20250510_c.jpg"]
    (root / "labels.csv").unlink()
    (root / "labels.jsonl").write_text(
        '{"image": "20250508_a.jpg", "text": "GJ07966", "verified": true}\n'
        '{"image": "20250508_b.jpg", "text": "GJ07967", "verified": false}\n',
        encoding="utf-8",
    )
    assert [i.path.name for i in load_dataset(root)] == ["20250508_a.jpg"]


def test_report_flags_optimistic_scores() -> None:
    base: dict[str, Any] = {
        "dataset": "d",
        "split": None,
        "metrics": summarize([]),
        "input_tokens": 0,
        "output_tokens": 0,
        "price_per_million": None,
    }
    own = Report(provider="gemini", model="m", prelabeled_with=("gemini:m",), **base)
    assert "**This backend pre-filled them, so its scores are optimistic.**" in own.to_markdown()
    assert own.to_dict()["prelabeled_with"] == ["gemini:m"]
    other = Report(provider="anthropic", model="x", prelabeled_with=("gemini:m",), **base)
    assert "Scores of those models are optimistic." in other.to_markdown()
    plain = Report(provider="anthropic", model="x", **base)
    assert "pre-filled" not in plain.to_markdown()


def test_cli_prelabel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import tagsort

    root = dataset(tmp_path)
    made: list[FakeProvider] = []

    def factory(name: str) -> Any:
        def build(api_key: str, model: str | None) -> FakeProvider:
            provider = FakeProvider(texts={60: "GJ07966"}, name=name, model=model or "default")
            made.append(provider)
            return provider

        return build

    monkeypatch.setattr(tagsort, "GeminiProvider", factory("gemini"), raising=False)
    monkeypatch.setattr(tagsort, "OpenAIProvider", factory("openai"), raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    assert cli.main(["prelabel", str(root)]) == 0
    assert [(p.name, p.model) for p in made] == [
        ("gemini", "gemini-3.5-flash-lite"),
        ("openai", "gpt-6-luna"),
    ]
    assert all(p.closed for p in made)
    out = capsys.readouterr().out
    assert "3 rows written" in out
    assert "review.html" in out

    assert (
        cli.main(
            ["prelabel", str(root), "--with", "gemini", "--with", "openai", "--with", "gemini"]
        )
        == 2
    )
    assert cli.main(["prelabel", str(root), "--with", "gemini"]) == 2  # labels.csv exists
    assert "exists" in capsys.readouterr().err
    assert cli.main(["prelabel", str(root), "--with", "nope"]) == 2
    monkeypatch.delenv("OPENAI_API_KEY")
    assert cli.main(["prelabel", str(root), "--with", "openai", "--overwrite"]) == 2
    assert "OPENAI_API_KEY is not set" in capsys.readouterr().err


def test_rows_of_a_photo_stay_together_on_the_page() -> None:
    from tagsort.evaluation.prelabel import Row, _page

    rows = [
        Row("a.jpg", "s", "GJ07966", priority=3),
        Row("a.jpg", "s", "0086", priority=0),
        Row("b.jpg", "s", "-", priority=1),
        Row("c.jpg", "s", "GJ07967", priority=3),
    ]
    page = _page("lot", rows)
    order = [page.index(f'data-index="{i}"') for i in range(4)]
    # a.jpg (both rows, since one needs attention), then b.jpg, then c.jpg.
    assert order[0] < order[1] < order[2] < order[3]
