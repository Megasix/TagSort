import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from tagsort import Profile, ProviderError, ProviderTag, Usage, VisionAnswer, VisionRequest
from tagsort.evaluation import cli
from tagsort.evaluation.dataset import load_dataset
from tagsort.evaluation.metrics import summarize
from tagsort.evaluation.report import Report, compare
from tagsort.evaluation.run import run_dataset
from tagsort.fallback.base import BoxFormat
from tests.images import jpeg_bytes, make_image

PROFILE = {
    "schema_version": "1.0",
    "name": "Test",
    "tags_per_individual": 1,
    "tags": [{"id": "primary", "pattern": "GJ\\d{5}"}],
}


@dataclass
class FakeProvider:
    texts: dict[int, str] = field(default_factory=dict)
    name: str = "fake"
    model: str = "fake-1"
    max_side: int = 200
    box_format: BoxFormat = "pixels_xyxy"
    calls: int = 0
    fail_on: int | None = None

    def read(self, request: VisionRequest) -> VisionAnswer:
        self.calls += 1
        if self.fail_on == self.calls:
            raise ProviderError("quota", provider="fake", reason="quota")
        text = self.texts.get(request.width)
        tags = (
            (
                ProviderTag(
                    text=text, legibility="certain", alternatives=(), box=(1, 1, 20, 10), angle=0
                ),
            )
            if text
            else ()
        )
        return VisionAnswer(tags=tags, model="fake-1", usage=Usage(1000, 50))

    def __enter__(self) -> "FakeProvider":
        return self

    def __exit__(self, *args: object) -> None:
        pass


def dataset(tmp_path: Path) -> Path:
    """Three photos told apart by width: 60 px has GJ07966, 70 px GJ07967, 80 px no tag."""
    (tmp_path / "photos").mkdir()
    for name, width in (("a.jpg", 60), ("b.jpg", 70), ("c.jpg", 80)):
        (tmp_path / "photos" / name).write_bytes(jpeg_bytes(make_image(width, 40)))
    (tmp_path / "labels.csv").write_text(
        "image,session,tag_id,text,notes\na.jpg,s1,,GJ07966,\nb.jpg,s1,,GJ07967,\nc.jpg,s2,,-,\n",
        encoding="utf-8",
    )
    (tmp_path / "profile.json").write_text(json.dumps(PROFILE), encoding="utf-8")
    return tmp_path


def test_run_reads_then_reuses_the_cache(tmp_path: Path) -> None:
    root = dataset(tmp_path)
    images = load_dataset(root)
    provider = FakeProvider(texts={60: "GJ07966", 70: "GJ07961"})
    cache = root / "predictions" / "fake"
    seen: list[str] = []
    first = run_dataset(
        images,
        profile=Profile.from_dict(PROFILE),
        provider=provider,
        cache=cache,
        progress=lambda i, n, o: seen.append(f"{i}/{n} {o.image}"),
    )
    assert provider.calls == 3
    assert seen == ["1/3 a.jpg", "2/3 b.jpg", "3/3 c.jpg"]
    assert (first.input_tokens, first.output_tokens) == (3000, 150)
    m = summarize(first.outcomes)
    assert (m.exact_matches, m.silent_errors, m.invented_tags) == (1, 1, 0)

    again = run_dataset(images, profile=Profile.from_dict(PROFILE), provider=provider, cache=cache)
    assert provider.calls == 3, "cached photos are not read again"
    assert summarize(again.outcomes) == m
    assert again.input_tokens == 3000

    run_dataset(
        images[:1], profile=Profile.from_dict(PROFILE), provider=provider, cache=cache, force=True
    )
    assert provider.calls == 4


def test_failed_photos_are_reported_and_retried_next_time(tmp_path: Path) -> None:
    root = dataset(tmp_path)
    images = load_dataset(root)
    provider = FakeProvider(fail_on=2)
    cache = root / "predictions" / "fake"
    result = run_dataset(images, profile=Profile.from_dict(PROFILE), provider=provider, cache=cache)
    assert [o.error for o in result.outcomes] == [None, "fake: quota", None]
    run_dataset(images, profile=Profile.from_dict(PROFILE), provider=provider, cache=cache)
    assert provider.calls == 4, "only the failed photo is read again"


def report(**changes: Any) -> Report:
    fields: dict[str, Any] = {
        "dataset": "lot-01",
        "split": None,
        "provider": "fake",
        "model": "fake-1",
        "metrics": summarize([]),
        "input_tokens": 3_000_000,
        "output_tokens": 100_000,
        "price_per_million": (0.3, 2.5),
        "created": "2026-10-06",
    }
    fields.update(changes)
    return Report(**fields)


def test_report_cost_and_rendering(tmp_path: Path) -> None:
    root = dataset(tmp_path)
    provider = FakeProvider(texts={60: "GJ07966"})
    result = run_dataset(
        load_dataset(root), profile=Profile.from_dict(PROFILE), provider=provider, cache=root / "p"
    )
    r = report(metrics=summarize(result.outcomes), input_tokens=3000, output_tokens=150)
    assert r.cost_per_1000_photos == round((3000 * 0.3 + 150 * 2.5) / 1e6 / 3 * 1000, 2)
    data = r.to_dict()
    assert data["report_version"] == 1
    assert data["backend"] == {"provider": "fake", "model": "fake-1"}
    assert data["metrics"]["exact_matches"] == 1
    json.dumps(data)
    text = r.to_markdown()
    assert "# Evaluation: lot-01" in text
    assert "| Exact match | 1 / 2 (50.0%) |" in text
    assert "## By session" in text
    assert "GJ07966" not in text, "reports hold aggregate numbers only"


def test_report_without_price() -> None:
    r = report(price_per_million=None, split="test", created="")
    assert r.cost_per_1000_photos is None
    assert "unknown" in r.to_markdown()
    assert "(test)" in r.to_markdown()
    assert r.to_dict()["price_per_million_tokens"] is None


def test_compare() -> None:
    base = {"metrics": {"exact_match_rate": 0.9, "silent_error_rate": 0.0}}
    same = {"metrics": {"exact_match_rate": 0.9, "silent_error_rate": 0.0}}
    worse = {"metrics": {"exact_match_rate": 0.85, "silent_error_rate": 0.02}}
    assert compare(base, same) == []
    assert len(compare(base, worse)) == 2
    assert compare(base, worse, tolerance=0.05) == []


def test_cli_run_and_compare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = dataset(tmp_path)
    import tagsort

    monkeypatch.setattr(
        tagsort,
        "GeminiProvider",
        lambda api_key, model: FakeProvider(texts={60: "GJ07966", 70: "GJ07967"}),
        raising=False,
    )
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    reports = tmp_path / "reports"
    code = cli.main(
        [
            "run",
            str(root),
            "--provider",
            "gemini",
            "--reports",
            str(reports),
            "--price",
            "0.3",
            "2.5",
            "--name",
            "lot",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "[1/3] a.jpg: GJ07966 (accepted)" in out
    assert "[3/3] c.jpg: no tag" in out
    (json_report,) = reports.glob("*.json")
    assert json_report.name.endswith("_lot_all_gemini-fake-1.json")
    assert (reports / json_report.name.replace(".json", ".md")).is_file()
    assert json.loads(json_report.read_text())["metrics"]["exact_match_rate"] == 1.0

    assert cli.main(["compare", str(json_report), str(json_report)]) == 0
    worse = tmp_path / "worse.json"
    data = json.loads(json_report.read_text())
    data["metrics"]["exact_match_rate"] = 0.5
    worse.write_text(json.dumps(data))
    assert cli.main(["compare", str(json_report), str(worse)]) == 1
    assert "REGRESSION" in capsys.readouterr().out


def test_cli_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert cli.main(["run", str(tmp_path), "--provider", "openai"]) == 2
    assert "OPENAI_API_KEY is not set" in capsys.readouterr().err
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    assert cli.main(["run", str(tmp_path), "--provider", "openai"]) == 2
    assert "profile.json" in capsys.readouterr().err
    (tmp_path / "d").mkdir()
    root = dataset(tmp_path / "d")
    assert cli.main(["run", str(root), "--provider", "openai", "--split", "nope"]) == 2
    assert "no labeled photo" in capsys.readouterr().err
