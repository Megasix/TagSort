"""Build reference vectors for porting the local pipeline to other languages.

Development tool, run by a maintainer when the pipeline or a model changes:

    uv run python scripts/make_reference.py

It renders synthetic tag images (no photo ever enters the repository) into
reference/images/, runs the local pipeline with the default model, and writes the
expected intermediate and final outputs to reference/pipeline.v1.json. A port must
reproduce them within the tolerances stated in that file.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from tagsort import Profile, Reader
from tagsort.models import DEFAULT_MODEL, load_manifest, model_files
from tagsort.pipeline.detect import OnnxDetector
from tagsort.pipeline.local import DETECTION_SIDE, LocalPipeline

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reference"
PROFILE = {
    "schema_version": "1.0",
    "name": "Reference",
    "tags_per_individual": 2,
    "tags": [
        {"id": "primary", "pattern": "GJ\\d{5}"},
        {"id": "secondary", "pattern": "\\d{4}"},
    ],
}
BACKGROUND = (226, 220, 205)


def tag_card(text: str, size: int = 64) -> Image.Image:
    font = ImageFont.load_default(size=size)
    left, top, right, bottom = (int(v) for v in font.getbbox(text))
    card = Image.new("RGB", (right - left + 60, bottom - top + 50), "white")
    ImageDraw.Draw(card).text((30 - left, 25 - top), text, fill=(20, 20, 20), font=font)
    return card


def scene(*placed: tuple[str, int, tuple[int, int]], extra: str | None = None) -> Image.Image:
    image = Image.new("RGB", (1600, 1200), BACKGROUND)
    for text, angle, (x, y) in placed:
        card = tag_card(text).rotate(angle, expand=True, fillcolor=BACKGROUND)
        image.paste(card, (x, y))
    if extra:
        font = ImageFont.load_default(size=48)
        ImageDraw.Draw(image).text((120, 1000), extra, fill=(60, 60, 60), font=font)
    return image.filter(ImageFilter.GaussianBlur(0.6))


IMAGES = {
    "upright": scene(("GJ07966", 0, (400, 400))),
    "upside_down": scene(("GJ08104", 180, (500, 300))),
    "quarter_turn": scene(("GJ08305", 90, (700, 250))),
    "tilted": scene(("GJ07990", 15, (350, 350))),
    "distractor": scene(("GJ07986", 0, (300, 200)), extra="Centre for Biodiversity Research"),
    "two_tags": scene(("GJ07966", 0, (200, 250)), ("2025", 0, (900, 700))),
    "no_tag": scene(extra="NEW BRUNSWICK MUSEUM"),
}


def rounded(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def main() -> None:
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(DEFAULT_MODEL)
    files = model_files(DEFAULT_MODEL)
    detector = OnnxDetector(files["det"], manifest.detection, limit_side=DETECTION_SIDE)
    profile = Profile.from_dict(PROFILE)
    pipeline = LocalPipeline(DEFAULT_MODEL)
    reader = Reader(profile, backend=pipeline)
    cases = []
    for name, image in IMAGES.items():
        path = OUT / "images" / f"{name}.png"
        image.save(path, optimize=True)
        image = Image.open(path).convert("RGB")
        lines = detector.detect(image)
        readings = pipeline.read(image, profile)
        result = reader.read(path).to_dict()
        cases.append(
            {
                "image": f"images/{name}.png",
                "detection": [
                    {
                        "quad": [[rounded(x, 1), rounded(y, 1)] for x, y in line.quad],
                        "score": rounded(line.score),
                    }
                    for line in lines
                ],
                "lines": [
                    {
                        "quad": [[rounded(x, 1), rounded(y, 1)] for x, y in r.quad],
                        "angle": rounded(r.angle, 1),
                        "agrees": r.agrees,
                        "readings": [
                            {"text": t, "probability": rounded(p), "tag_id": i}
                            for t, p, i in r.readings
                        ],
                    }
                    for r in readings
                ],
                "result": {
                    "tags": [
                        {k: tag[k] for k in ("tag_id", "text", "status", "angle", "polygon")}
                        for tag in result["tags"]
                    ]
                },
            }
        )
        print(f"{name}: {len(lines)} lines, tags {[t['text'] for t in result['tags']]}")
    document = {
        "description": (
            "Reference vectors for the local pipeline (docs/local.md). A port runs the same "
            "model on each image with the same profile and must match: texts, tag ids and "
            "statuses exactly; probabilities and scores within probability_tolerance; "
            "coordinates within pixel_tolerance; angles within angle_tolerance degrees."
        ),
        "version": "1.0",
        "model": DEFAULT_MODEL,
        "model_version": manifest.version,
        "detection_side": DETECTION_SIDE,
        "profile": PROFILE,
        "pixel_tolerance": 3.0,
        "probability_tolerance": 0.03,
        "angle_tolerance": 2.0,
        "cases": cases,
    }
    (OUT / "pipeline.v1.json").write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    print(f"wrote reference/pipeline.v1.json ({len(cases)} cases)")


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    main()
