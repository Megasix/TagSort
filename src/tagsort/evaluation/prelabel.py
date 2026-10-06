"""Pre-label a dataset with one or two AI backends, for a person to check.

Development and training tool only (CLAUDE.md §4). It writes, in the dataset folder:

- ``labels.csv``: one row per tag found, the text pre-filled, and an empty ``verified``
  column; only rows marked verified count in evaluations;
- ``review.html``: a plain local page showing each tag, upright, to check and correct
  the texts, then export the corrected ``labels.csv``;
- ``prelabel.json``: which models pre-filled the labels, so reports can say their scores
  are optimistic.
"""

from __future__ import annotations

import base64
import csv
import datetime as dt
import html
import io
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from tagsort.evaluation.dataset import PHOTO_SUFFIXES, LabeledImage, LabeledTag
from tagsort.evaluation.metrics import match_tags
from tagsort.evaluation.run import _tag, run_dataset
from tagsort.fallback.base import VisionProvider
from tagsort.pipeline.preprocess import prepare
from tagsort.profile import Profile
from tagsort.types import Tag

__all__ = ["PRELABEL_FILE", "Row", "build_rows", "prelabel", "session_of"]

PRELABEL_FILE = "prelabel.json"
CSV_COLUMNS = ("image", "session", "tag_id", "text", "verified", "notes", "ai_1", "ai_2")
CROP_SIDE = 360
PHOTO_SIDE = 320


@dataclass
class Row:
    """One line to check: a tag at least one AI found, or a photo where none found any."""

    image: str
    session: str
    text: str
    tag_id: str = ""
    ai_1: str = ""
    ai_2: str = ""
    crop: str = ""  # JPEG as a data URL
    photo: str = ""  # JPEG as a data URL
    priority: int = 3  # 0 = check first

    def csv_record(self) -> dict[str, str]:
        """Return the row as written to ``labels.csv``."""
        return {
            "image": self.image,
            "session": self.session,
            "tag_id": self.tag_id,
            "text": self.text,
            "verified": "",
            "notes": "",
            "ai_1": self.ai_1,
            "ai_2": self.ai_2,
        }


def session_of(path: Path) -> str:
    """Guess the shooting session: the date in a phone file name, else the EXIF date."""
    match = re.match(r"(?:IMG_|PXL_)?(\d{4})(\d{2})(\d{2})", path.name)
    if match:
        return "-".join(match.groups())
    try:
        with Image.open(path) as image:
            stamp = image.getexif().get_ifd(0x8769).get(0x9003) or image.getexif().get(0x0132)
    except OSError:
        return ""
    if isinstance(stamp, str) and re.match(r"\d{4}:\d{2}:\d{2}", stamp):
        return stamp[:10].replace(":", "-")
    return ""


def build_rows(
    image: str, session: str, first: Sequence[Tag], second: Sequence[Tag] | None
) -> list[tuple[Row, Tag | None]]:
    """Merge what one or two AIs read on a photo into rows, each with the tag to crop."""
    if not first and not second:
        return [(Row(image, session, text="-", priority=1), None)]
    pairs, extra = match_tags(list(second or ()), [LabeledTag(tag.text) for tag in first])
    rows: list[tuple[Row, Tag | None]] = []
    for tag, (_, other) in zip(first, pairs, strict=True):
        ai_1 = tag.text or "?"
        ai_2 = (other.text or "?") if other is not None else ""
        if second is None:
            priority = 0 if tag.status != "accepted" else 3
        else:
            priority = 0 if ai_1 != ai_2 else (2 if tag.status != "accepted" else 3)
        rows.append(
            (Row(image, session, ai_1, tag.tag_id or "", ai_1, ai_2, priority=priority), tag)
        )
    for tag in extra:
        text = tag.text or "?"
        rows.append((Row(image, session, text, tag.tag_id or "", "", text, priority=0), tag))
    return rows


def prelabel(
    dataset: Path,
    *,
    profile: Profile,
    providers: Sequence[VisionProvider],
    overwrite: bool = False,
) -> list[Row]:
    """Read every photo of ``dataset`` with one or two providers and write the review files.

    Answers are cached like evaluation runs, so evaluating the same models later costs
    nothing more.

    Raises:
        FileExistsError: If ``labels.csv`` exists and ``overwrite`` is not set.
        ValueError: If ``providers`` does not hold one or two providers.
    """
    if not 1 <= len(providers) <= 2:
        raise ValueError("prelabel needs one or two providers")
    labels = dataset / "labels.csv"
    if labels.exists() and not overwrite:
        raise FileExistsError(f"{labels} exists; pass --overwrite to replace it")
    photos = sorted(p for p in (dataset / "photos").iterdir() if p.suffix.lower() in PHOTO_SUFFIXES)
    images = [LabeledImage(path=p, session=session_of(p)) for p in photos]

    readings: list[dict[str, list[Tag]]] = []
    for provider in providers:
        cache = dataset / "predictions" / f"{provider.name}-{provider.model}"
        run_dataset(images, profile=profile, provider=provider, cache=cache)
        readings.append({p.name: _cached_tags(cache / f"{p.stem}.json") for p in photos})

    rows: list[Row] = []
    for image in images:
        name = image.path.name
        second = readings[1][name] if len(readings) == 2 else None
        merged = build_rows(name, image.session, readings[0][name], second)
        prepared = prepare(image.path, max_side=4096)
        for row, tag in merged:
            row.photo = _data_url(prepared.image, PHOTO_SIDE * (2 if tag is None else 1))
            if tag is not None:
                row.crop = _data_url(_crop(prepared.image, tag, prepared.scale), CROP_SIDE)
            rows.append(row)

    with labels.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(row.csv_record() for row in rows)
    (dataset / "review.html").write_text(_page(dataset.name, rows), encoding="utf-8")
    (dataset / PRELABEL_FILE).write_text(
        json.dumps(
            {
                "created": dt.date.today().isoformat(),
                "models": [f"{p.name}:{p.model}" for p in providers],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return rows


def _cached_tags(path: Path) -> list[Tag]:
    """Tags of a cached result, or none if the photo could not be read."""
    if not path.is_file():
        return []
    result: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return [_tag(data) for data in result["tags"]]


def _crop(image: Image.Image, tag: Tag, scale: float) -> Image.Image:
    xs = [x / scale for x, _ in tag.polygon]
    ys = [y / scale for _, y in tag.polygon]
    pad_x = (max(xs) - min(xs)) * 0.15 + 4
    pad_y = (max(ys) - min(ys)) * 0.15 + 4
    box = (
        max(int(min(xs) - pad_x), 0),
        max(int(min(ys) - pad_y), 0),
        min(int(max(xs) + pad_x), image.width),
        min(int(max(ys) + pad_y), image.height),
    )
    # The text is rotated clockwise by tag.angle; PIL rotates counterclockwise.
    return image.crop(box).rotate(tag.angle, expand=True)


def _data_url(image: Image.Image, side: int) -> str:
    thumbnail = image.copy()
    thumbnail.thumbnail((side, side))
    buffer = io.BytesIO()
    thumbnail.save(buffer, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _page(name: str, rows: list[Row]) -> str:
    # Photos needing attention come first; rows of one photo always stay together.
    first_row: dict[str, int] = {}
    priority: dict[str, int] = {}
    for index, row in enumerate(rows):
        first_row.setdefault(row.image, index)
        priority[row.image] = min(priority.get(row.image, row.priority), row.priority)
    ordered = sorted(
        enumerate(rows),
        key=lambda item: (priority[item[1].image], first_row[item[1].image], item[0]),
    )
    body = "\n".join(_row_html(index, row) for index, row in ordered)
    data = json.dumps([row.csv_record() for row in rows])
    return (
        _TEMPLATE.replace("{{NAME}}", html.escape(name))
        .replace("{{ROWS}}", body)
        .replace("{{DATA}}", data.replace("</", "<\\/"))
        .replace(
            "{{STORAGE}}", json.dumps(f"tagsort-review-{name}-{dt.datetime.now():%Y%m%d%H%M%S}")
        )
    )


def _row_html(index: int, row: Row) -> str:
    reason = {
        0: "AIs disagree or are unsure",
        1: "nothing found: check for missed tags",
        2: "unsure",
    }
    note = reason.get(row.priority, "")
    image = row.crop or row.photo
    readings = " / ".join(text for text in (row.ai_1, row.ai_2) if text)
    return f"""<tr data-index="{index}">
<td><img src="{row.photo}" alt=""></td>
<td>{f'<img src="{image}" alt="">' if row.crop else ""}</td>
<td><b>{html.escape(row.image)}</b><br><small>{html.escape(note)}</small><br>
<small>AI: {html.escape(readings) or "nothing"}</small></td>
<td><input class="text" value="{html.escape(row.text, quote=True)}"><br>
<button type="button" class="none">no tag</button>
<button type="button" class="unreadable">unreadable (?)</button>
<button type="button" class="add">+ tag</button></td>
<td><label><input type="checkbox" class="verified"> verified</label></td>
</tr>"""


_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Review {{NAME}}</title>
<style>
body{font-family:sans-serif;margin:16px}
td{border-bottom:1px solid #ccc;padding:6px;vertical-align:top}
img{max-width:360px}
input.text{font-size:18px;width:14em}
tr.done{background:#eef8ee}
</style></head><body>
<h1>Review {{NAME}}</h1>
<p>Development and training tool. Correct each text, tick <b>verified</b>, then export and
put <code>labels.csv</code> in the dataset folder. Rows needing attention come first.
Progress is kept in this browser.</p>
<p><button type="button" id="export">Export labels.csv</button> <span id="count"></span></p>
<table>{{ROWS}}</table>
<script>
const rows = {{DATA}};
const key = {{STORAGE}};
function save() {
  try { localStorage.setItem(key, JSON.stringify(rows)); } catch (e) {}
  document.getElementById("count").textContent =
    rows.filter(r => r.verified === "yes").length + " / " + rows.length + " verified";
}
function bind(tr) {
  const row = rows[tr.dataset.index];
  const text = tr.querySelector("input.text");
  const box = tr.querySelector("input.verified");
  text.value = row.text; box.checked = row.verified === "yes";
  tr.classList.toggle("done", box.checked);
  text.oninput = () => { row.text = text.value.trim(); save(); };
  box.onchange = () => {
    row.verified = box.checked ? "yes" : "";
    tr.classList.toggle("done", box.checked);
    save();
  };
  tr.querySelector(".none").onclick = () => { text.value = "-"; text.oninput(); };
  tr.querySelector(".unreadable").onclick = () => { text.value = "?"; text.oninput(); };
  tr.querySelector(".add").onclick = () => {
    rows.push({...row, text: "", verified: "", ai_1: "", ai_2: ""});
    const copy = tr.cloneNode(true);
    copy.dataset.index = rows.length - 1;
    copy.cells[1].innerHTML = "";
    tr.after(copy); bind(copy); save();
  };
}
try {
  const saved = JSON.parse(localStorage.getItem(key) || "null");
  if (Array.isArray(saved) && saved.length >= rows.length) { rows.length = 0; rows.push(...saved); }
} catch (e) {}
const table = document.querySelector("table");
const initial = table.querySelectorAll("tr[data-index]").length;
for (let i = initial; i < rows.length; i++) {
  // Rows added with "+ tag" in an earlier session: show them under their photo again.
  const source = [...table.querySelectorAll("tr[data-index]")]
    .find(tr => rows[tr.dataset.index].image === rows[i].image);
  if (!source) continue;
  const copy = source.cloneNode(true);
  copy.dataset.index = i; copy.cells[1].innerHTML = "";
  source.after(copy);
}
document.querySelectorAll("tr[data-index]").forEach(bind);
save();
document.getElementById("export").onclick = () => {
  const columns = ["image", "session", "tag_id", "text", "verified", "notes", "ai_1", "ai_2"];
  const quote = v => /[",\\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v;
  const line = r => columns.map(c => quote(r[c] || "")).join(",");
  const lines = [columns.join(",")].concat(rows.map(line));
  const blob = new Blob(["\\ufeff" + lines.join("\\n") + "\\n"], {type: "text/csv"});
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob); link.download = "labels.csv"; link.click();
};
</script></body></html>
"""
