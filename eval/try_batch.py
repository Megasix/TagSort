"""Read a folder of photos with one or more vision API providers and compare to labels.

This is a development tool for milestone M2, not part of the engine. The full evaluation
harness (``tagsort-eval``) comes in M3.

Dataset layout (kept outside the repository):

    <dataset>/photos/*.jpg      the photos
    <dataset>/labels.csv        image,session,tag_id,text,notes (one row per tag)
    <dataset>/profile.json      the profile, following schemas/profile.v1.json

API keys are read from the environment: ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY,
DEEPSEEK_API_KEY.

Usage:

    uv run python eval/try_batch.py <dataset> --check
    uv run python eval/try_batch.py <dataset> --provider anthropic --limit 5
    uv run python eval/try_batch.py <dataset> --provider anthropic openai gemini
    uv run python eval/try_batch.py <dataset> --provider anthropic --model claude-haiku-4-5

Results are written to <dataset>/results/<provider>/<image>.json.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import httpx

from tagsort import (
    AnthropicProvider,
    DeepSeekProvider,
    GeminiProvider,
    OpenAIProvider,
    Profile,
    ProviderError,
    Reader,
    ReadResult,
    TagSortError,
    VisionAnswer,
    VisionProvider,
    VisionRequest,
)
from tagsort.fallback._http import HttpProvider
from tagsort.fallback.base import BoxFormat

PROVIDERS: dict[str, tuple[type[HttpProvider], str]] = {
    "anthropic": (AnthropicProvider, "ANTHROPIC_API_KEY"),
    "openai": (OpenAIProvider, "OPENAI_API_KEY"),
    "gemini": (GeminiProvider, "GEMINI_API_KEY"),
    "deepseek": (DeepSeekProvider, "DEEPSEEK_API_KEY"),
}

# USD per million input and output tokens, from the providers' pricing pages on
# 2026-10-06. Check them before relying on the cost estimate.
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "gpt-6.1-sol": (2.0, 10.0),
    "gpt-6-luna": (0.10, 0.50),
    "gemini-3.1-pro-preview": (2.0, 12.0),
    "gemini-3.5-flash-lite": (0.30, 2.50),
}

PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


@dataclass
class Counting:
    """Wraps a provider to add up the tokens it bills."""

    inner: VisionProvider
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def model(self) -> str:
        return self.inner.model

    @property
    def max_side(self) -> int:
        return self.inner.max_side

    @property
    def box_format(self) -> BoxFormat:
        return self.inner.box_format

    def read(self, request: VisionRequest) -> VisionAnswer:
        answer = self.inner.read(request)
        self.input_tokens += answer.usage.input_tokens
        self.output_tokens += answer.usage.output_tokens
        return answer


PLACEHOLDER = "A_REMPLIR"
NO_TAG = "-"
UNREADABLE = "?"


def load_labels(path: Path) -> dict[str, list[tuple[str, str]]]:
    """Return image name -> list of (tag_id, text) for every labeled photo.

    ``?`` marks a tag nobody can read; ``-`` or an empty text marks a photo without a
    tag. Rows still holding the ``A_REMPLIR`` placeholder are not labeled yet and are
    skipped. Comma- and semicolon-separated files are both accepted.
    """
    text = path.read_text(encoding="utf-8-sig")
    delimiter = ";" if text.splitlines()[0].count(";") > text.splitlines()[0].count(",") else ","
    labels: dict[str, list[tuple[str, str]]] = {}
    for row in csv.DictReader(text.splitlines(), delimiter=delimiter):
        image = (row.get("image") or "").strip()
        tag_text = (row.get("text") or "").strip()
        if not image or image.startswith("EXEMPLE_") or tag_text == PLACEHOLDER:
            continue
        labels.setdefault(image, [])
        if tag_text and tag_text != NO_TAG:
            labels[image].append(((row.get("tag_id") or "").strip(), tag_text))
    return labels


def check_models() -> int:
    """List each provider's models and say whether the default model is available."""
    urls = {
        "anthropic": "https://api.anthropic.com/v1/models?limit=1000",
        "openai": "https://api.openai.com/v1/models",
        "gemini": "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000",
        "deepseek": "https://api.deepseek.com/models",
    }
    status = 0
    for name, (cls, variable) in PROVIDERS.items():
        key = os.environ.get(variable)
        if not key:
            print(f"{name}: {variable} is not set, skipped")
            continue
        headers = {
            "anthropic": {"x-api-key": key, "anthropic-version": "2023-06-01"},
            "openai": {"Authorization": f"Bearer {key}"},
            "gemini": {"x-goog-api-key": key},
            "deepseek": {"Authorization": f"Bearer {key}"},
        }[name]
        try:
            response = httpx.get(urls[name], headers=headers, timeout=30)
            response.raise_for_status()
        except httpx.HTTPError as error:
            print(f"{name}: cannot list models ({type(error).__name__})")
            status = 1
            continue
        items = response.json().get("data") or response.json().get("models") or []
        ids = {
            str(item.get("id") or item.get("name", "")).removeprefix("models/") for item in items
        }
        family = {"anthropic": "claude", "openai": "gpt", "gemini": "gemini"}.get(name, name)
        wanted = [cls.default_model, *(m for m in PRICES if m.startswith(family))]
        for model in dict.fromkeys(wanted):
            print(f"{name}: {model} {'available' if model in ids else 'NOT FOUND'}")
        if cls.default_model not in ids:
            status = 1
            print("  available:", ", ".join(sorted(ids)[:40]))
    return status


def run(dataset: Path, provider_name: str, limit: int | None, model: str | None) -> None:
    cls, variable = PROVIDERS[provider_name]
    key = os.environ.get(variable)
    if not key:
        sys.exit(f"{variable} is not set")
    profile = Profile.from_file(dataset / "profile.json")
    labels = load_labels(dataset / "labels.csv")
    photos = sorted(
        p for p in (dataset / "photos").iterdir() if p.suffix.lower() in PHOTO_SUFFIXES
    )[:limit]
    output = dataset / "results" / f"{provider_name}-{model or cls.default_model}"
    output.mkdir(parents=True, exist_ok=True)

    provider = Counting(cls(api_key=key, model=model))
    reader = Reader(profile, backend=provider)
    expected = found = silent = unlabeled = failed = 0
    statuses: dict[str, int] = defaultdict(int)
    for index, photo in enumerate(photos, 1):
        try:
            result: ReadResult = reader.read(photo)
        except (ProviderError, TagSortError) as error:
            failed += 1
            print(f"[{index}/{len(photos)}] {photo.name}: ERROR {error}")
            continue
        (output / f"{photo.stem}.json").write_text(result.to_json(indent=2), encoding="utf-8")
        truth = labels.get(photo.name)
        read_texts = [tag.text for tag in result.tags if tag.text]
        for tag in result.tags:
            statuses[tag.status] += 1
        if truth is None:
            unlabeled += 1
            verdict = "no label"
        else:
            readable = [text for _, text in truth if text != UNREADABLE]
            hits = sum(1 for text in readable if text in read_texts)
            wrong = [
                tag.text
                for tag in result.tags
                if tag.status == "accepted" and tag.text not in readable
            ]
            expected += len(readable)
            found += hits
            silent += len(wrong)
            verdict = f"{hits}/{len(readable)} read" + (f", SILENT ERROR {wrong}" if wrong else "")
        summary = ", ".join(f"{t.text or '-'} ({t.status})" for t in result.tags) or "no tag"
        print(f"[{index}/{len(photos)}] {photo.name}: {summary} -> {verdict}")

    price_in, price_out = PRICES.get(provider.model, (0.0, 0.0))
    cost = provider.input_tokens / 1e6 * price_in + provider.output_tokens / 1e6 * price_out
    read_count = len(photos) - failed
    print(f"\n{provider_name} ({provider.model}) on {len(photos)} photos")
    print(
        f"  tags read exactly: {found}/{expected}"
        + (f" ({found / expected:.0%})" if expected else "")
    )
    print(f"  silent errors (accepted but wrong): {silent}")
    print(f"  statuses: {dict(statuses)}; failed photos: {failed}; unlabeled: {unlabeled}")
    print(f"  tokens: {provider.input_tokens} in, {provider.output_tokens} out")
    if read_count and price_in:
        per_thousand = cost / read_count * 1000
        print(f"  cost: ${cost:.3f}, about ${per_thousand:.2f} per 1,000 photos")
    elif read_count:
        print(f"  cost: price of {provider.model} unknown, see the provider's pricing page")
    print(f"  results: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--provider", nargs="+", choices=sorted(PROVIDERS), default=[])
    parser.add_argument("--limit", type=int, default=None, help="read only the first N photos")
    parser.add_argument("--model", help="model to use instead of the provider's default")
    parser.add_argument("--check", action="store_true", help="check the models exist")
    args = parser.parse_args()
    if args.check:
        sys.exit(check_models())
    if not args.provider:
        parser.error("give --provider or --check")
    if args.model and len(args.provider) > 1:
        parser.error("--model needs exactly one --provider")
    for name in args.provider:
        run(args.dataset, name, args.limit, args.model)


if __name__ == "__main__":
    main()
