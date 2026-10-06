"""The ``tagsort-eval`` command.

    tagsort-eval run DATASET --provider gemini [--model M] [--split test] [--limit N]
    tagsort-eval prelabel DATASET [--with gemini:gemini-3.5-flash-lite] [--with openai]
    tagsort-eval compare BASE.json NEW.json [--tolerance 0.01]

``run`` reads the API key from the environment (ANTHROPIC_API_KEY, OPENAI_API_KEY,
GEMINI_API_KEY or DEEPSEEK_API_KEY), caches answers in DATASET/predictions/ so no photo is
paid for twice, and writes a Markdown and a JSON report.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from collections.abc import Sequence
from pathlib import Path

from tagsort.errors import TagSortError
from tagsort.evaluation.dataset import load_dataset
from tagsort.evaluation.metrics import ImageOutcome, summarize
from tagsort.evaluation.prelabel import PRELABEL_FILE, prelabel
from tagsort.evaluation.report import PRICES, Report, compare
from tagsort.evaluation.run import run_dataset
from tagsort.fallback._http import HttpProvider
from tagsort.profile import Profile

__all__ = ["main"]

KEY_VARIABLES = {
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
}

DEFAULT_REPORTS = Path("eval/reports")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line; return the exit code."""
    parser = argparse.ArgumentParser(prog="tagsort-eval", description="Evaluate TagSort.")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="read a labeled dataset and write a report")
    run.add_argument("dataset", type=Path, help="folder with photos/, labels and profile.json")
    run.add_argument("--provider", required=True, choices=sorted(KEY_VARIABLES))
    run.add_argument("--model", help="model to use instead of the provider's default")
    run.add_argument("--split", help="evaluate only this split, for example test")
    run.add_argument("--limit", type=int, help="read only the first N photos")
    run.add_argument("--force", action="store_true", help="read again photos already cached")
    run.add_argument("--reports", type=Path, default=DEFAULT_REPORTS, help="report folder")
    run.add_argument("--name", help="dataset name in the report (default: folder name)")
    run.add_argument(
        "--price",
        nargs=2,
        type=float,
        metavar=("IN", "OUT"),
        help="USD per million input and output tokens",
    )

    pre = commands.add_parser(
        "prelabel", help="pre-fill labels with AI readings and write a review page"
    )
    pre.add_argument("dataset", type=Path, help="folder with photos/ and profile.json")
    pre.add_argument(
        "--with",
        dest="backends",
        action="append",
        metavar="PROVIDER[:MODEL]",
        help="backend to pre-label with, once or twice (default: gemini:gemini-3.5-flash-lite "
        "and openai:gpt-6-luna)",
    )
    pre.add_argument("--overwrite", action="store_true", help="replace an existing labels.csv")

    comparison = commands.add_parser("compare", help="fail if NEW regresses against BASE")
    comparison.add_argument("base", type=Path)
    comparison.add_argument("new", type=Path)
    comparison.add_argument("--tolerance", type=float, default=0.0)

    args = parser.parse_args(argv)
    try:
        if args.command == "compare":
            return _compare(args)
        if args.command == "prelabel":
            return _prelabel(args)
        return _run(args)
    except (TagSortError, OSError) as error:
        print(f"tagsort-eval: {error}", file=sys.stderr)
        return 2


def _compare(args: argparse.Namespace) -> int:
    base = json.loads(args.base.read_text(encoding="utf-8"))
    new = json.loads(args.new.read_text(encoding="utf-8"))
    problems = compare(base, new, tolerance=args.tolerance)
    for problem in problems:
        print(f"REGRESSION: {problem}")
    if not problems:
        print("No regression.")
    return 1 if problems else 0


def _provider(name: str, model: str | None) -> HttpProvider:
    """Build a provider with its key from the environment."""
    import tagsort

    if name not in KEY_VARIABLES:
        raise TagSortError(f"unknown provider {name!r}; choose from {', '.join(KEY_VARIABLES)}")
    key = os.environ.get(KEY_VARIABLES[name])
    if not key:
        raise TagSortError(f"{KEY_VARIABLES[name]} is not set")
    cls = getattr(tagsort, f"{_CLASS_NAMES[name]}Provider")
    provider: HttpProvider = cls(api_key=key, model=model)
    return provider


_CLASS_NAMES = {
    "anthropic": "Anthropic",
    "deepseek": "DeepSeek",
    "gemini": "Gemini",
    "openai": "OpenAI",
}
DEFAULT_PRELABEL = ("gemini:gemini-3.5-flash-lite", "openai:gpt-6-luna")


def _prelabel(args: argparse.Namespace) -> int:
    specs = args.backends or list(DEFAULT_PRELABEL)
    if len(specs) > 2:
        print("tagsort-eval: give --with once or twice", file=sys.stderr)
        return 2
    profile = Profile.from_file(args.dataset / "profile.json")
    providers = []
    for spec in specs:
        name, _, model = spec.partition(":")
        providers.append(_provider(name, model or None))
    try:
        rows = prelabel(
            args.dataset, profile=profile, providers=providers, overwrite=args.overwrite
        )
    finally:
        for provider in providers:
            provider.close()
    first = sum(row.priority < 3 for row in rows)
    print(f"{len(rows)} rows written to {args.dataset / 'labels.csv'}; {first} need a close look.")
    print(f"Open {args.dataset / 'review.html'} in a browser to check them.")
    return 0


def _run(args: argparse.Namespace) -> int:
    if not os.environ.get(KEY_VARIABLES[args.provider]):
        raise TagSortError(f"{KEY_VARIABLES[args.provider]} is not set")

    dataset: Path = args.dataset
    profile = Profile.from_file(dataset / "profile.json")
    images = load_dataset(dataset, split=args.split)[: args.limit]
    if not images:
        print("tagsort-eval: no labeled photo to evaluate", file=sys.stderr)
        return 2

    with _provider(args.provider, args.model) as provider:
        model = provider.model
        cache = dataset / "predictions" / f"{args.provider}-{model}"

        def progress(index: int, total: int, outcome: ImageOutcome) -> None:
            if outcome.error:
                status = f"ERROR {outcome.error}"
            else:
                read = [t for _, t in outcome.pairs if t is not None] + list(outcome.invented)
                status = ", ".join(f"{t.text or '-'} ({t.status})" for t in read) or "no tag"
            print(f"[{index}/{total}] {outcome.image}: {status}"[:200])

        result = run_dataset(
            images,
            profile=profile,
            provider=provider,
            cache=cache,
            force=args.force,
            progress=progress,
        )

    prelabel_file = dataset / PRELABEL_FILE
    prelabeled_with: list[str] = (
        json.loads(prelabel_file.read_text(encoding="utf-8"))["models"]
        if prelabel_file.is_file()
        else []
    )
    price: tuple[float, float] | None = (
        (args.price[0], args.price[1]) if args.price else PRICES.get(model)
    )
    report = Report(
        dataset=args.name or dataset.name,
        split=args.split,
        provider=args.provider,
        model=model,
        metrics=summarize(result.outcomes),
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        price_per_million=price,
        created=dt.date.today().isoformat(),
        prelabeled_with=tuple(prelabeled_with),
    )
    args.reports.mkdir(parents=True, exist_ok=True)
    stem = _slug(f"{report.created}_{report.dataset}_{args.split or 'all'}_{args.provider}-{model}")
    (args.reports / f"{stem}.json").write_text(
        json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    (args.reports / f"{stem}.md").write_text(report.to_markdown(), encoding="utf-8")
    print()
    print(report.to_markdown())
    print(f"Reports written to {args.reports / stem}.{{md,json}}")
    return 0


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")
