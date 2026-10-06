"""Evaluation reports: a JSON file for machines and a Markdown file for people.

Reports hold aggregate numbers only; per-photo texts stay in the dataset folder.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

from tagsort._version import __version__
from tagsort.evaluation.metrics import Metrics

__all__ = ["PRICES", "PRICES_DATE", "Report", "compare"]

REPORT_VERSION = 1

PRICES_DATE = "2026-10-06"
"""When :data:`PRICES` was copied from the providers' pricing pages."""

PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "gpt-6.1-sol": (2.0, 10.0),
    "gpt-6-luna": (0.10, 0.50),
    "gemini-3.1-pro-preview": (2.0, 12.0),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    # Off-peak rate; DeepSeek doubles it at peak hours. Consistent with the invoice of
    # the lot-01 runs (under 1 cent CAD for about 36,000 tokens).
    "deepseek-flash": (0.15, 0.60),
}
"""USD per million input and output tokens, standard tier, cache misses."""


@dataclass(frozen=True)
class Report:
    """One evaluation run: what was measured, on what, and the numbers."""

    dataset: str
    split: str | None
    provider: str
    model: str
    metrics: Metrics
    input_tokens: int
    output_tokens: int
    price_per_million: tuple[float, float] | None
    created: str = ""
    prelabeled_with: tuple[str, ...] = ()

    @property
    def cost_per_1000_photos(self) -> float | None:
        """Estimated USD per 1,000 photos read, or ``None`` if the price is unknown."""
        read = self.metrics.photos - self.metrics.failed_photos
        if self.price_per_million is None or not read:
            return None
        price_in, price_out = self.price_per_million
        cost = (self.input_tokens * price_in + self.output_tokens * price_out) / 1e6
        return round(cost / read * 1000, 2)

    def to_dict(self) -> dict[str, Any]:
        """Return the report as a JSON-ready dictionary."""
        return {
            "report_version": REPORT_VERSION,
            "created": self.created or dt.date.today().isoformat(),
            "engine_version": __version__,
            "dataset": self.dataset,
            "split": self.split,
            "backend": {"provider": self.provider, "model": self.model},
            "metrics": self.metrics.to_dict(),
            "tokens": {"input": self.input_tokens, "output": self.output_tokens},
            "price_per_million_tokens": (
                {"input": self.price_per_million[0], "output": self.price_per_million[1]}
                if self.price_per_million
                else None
            ),
            "cost_per_1000_photos_usd": self.cost_per_1000_photos,
            "prelabeled_with": list(self.prelabeled_with),
        }

    def to_markdown(self) -> str:
        """Return the report as a Markdown page."""
        m = self.metrics
        cost = self.cost_per_1000_photos
        rows = [
            ("Photos read", f"{m.photos - m.failed_photos} / {m.photos}"),
            ("Readable tags (labels)", str(m.readable_tags)),
            ("Tags no person can read", str(m.unreadable_tags)),
            ("Exact match", f"{m.exact_matches} / {m.readable_tags} ({m.exact_match_rate:.1%})"),
            ("Character error rate", f"{m.character_error_rate:.2%}"),
            (
                "Accepted and correct (automation)",
                f"{m.accepted_correct} ({m.automation_rate:.1%})",
            ),
            (
                "**Silent errors** (accepted but wrong)",
                f"**{m.silent_errors}** ({m.silent_error_rate:.1%} of accepted)",
            ),
            ("Sent to review", str(m.review_tags)),
            ("Marked unreadable", str(m.unreadable_predictions)),
            ("Missed tags", str(m.missed_tags)),
            ("Invented tags", str(m.invented_tags)),
            ("Share read by the fallback", f"{m.fallback_share:.0%}"),
            ("Time per photo", f"{m.seconds_per_photo:.1f} s"),
            ("Cost per 1,000 photos", f"${cost:.2f}" if cost is not None else "unknown"),
        ]
        lines = [
            f"# Evaluation: {self.dataset}" + (f" ({self.split})" if self.split else ""),
            "",
            f"- Backend: `{self.provider}` / `{self.model}`",
            f"- Engine: {__version__}",
            f"- Date: {self.created or dt.date.today().isoformat()}",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            *(f"| {name} | {value} |" for name, value in rows),
        ]
        backend = f"{self.provider}:{self.model}"
        if self.prelabeled_with:
            lines += [
                "",
                f"Labels were pre-filled by {', '.join(self.prelabeled_with)} and checked by a "
                "person. "
                + (
                    "**This backend pre-filled them, so its scores are optimistic.**"
                    if backend in self.prelabeled_with
                    else "Scores of those models are optimistic."
                ),
            ]
        if len(m.sessions) > 1:
            lines += [
                "",
                "## By session",
                "",
                "| Session | Photos | Readable tags | Exact match | Silent errors |",
                "| --- | --- | --- | --- | --- |",
                *(
                    f"| {name} | {s['photos']:.0f} | {s['readable_tags']:.0f} | "
                    f"{s['exact_match_rate']:.1%} | {s['silent_errors']:.0f} |"
                    for name, s in m.sessions.items()
                ),
            ]
        lines += ["", "Metric definitions: `docs/evaluation.md`.", ""]
        return "\n".join(lines)


def compare(base: dict[str, Any], new: dict[str, Any], *, tolerance: float = 0.0) -> list[str]:
    """Return the regressions of ``new`` against ``base``; an empty list means none.

    A regression is a lower exact match rate or a higher silent error rate, beyond
    ``tolerance`` (an absolute difference of rates).
    """
    problems = []
    b, n = base["metrics"], new["metrics"]
    if n["exact_match_rate"] < b["exact_match_rate"] - tolerance:
        problems.append(
            f"exact match rate dropped: {b['exact_match_rate']:.1%} -> {n['exact_match_rate']:.1%}"
        )
    if n["silent_error_rate"] > b["silent_error_rate"] + tolerance:
        problems.append(
            f"silent error rate rose: {b['silent_error_rate']:.1%} -> {n['silent_error_rate']:.1%}"
        )
    return problems
