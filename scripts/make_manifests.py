"""Build the model manifests in src/tagsort/models/manifests/ from the upstream releases.

Development tool, run by a maintainer when adding or updating a model:

    uv run python scripts/make_manifests.py

It downloads each pinned upstream file, records its size and SHA-256, and copies the
preprocessing parameters and the recognizer's character set from the upstream
``inference.yml``. Uses the standard library and httpx (api extra) only.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src" / "tagsort" / "models" / "manifests"

# Upstream repositories pinned to a commit so a manifest never changes under our feet.
RELEASES = {
    "ppocrv6-tiny": {
        "det": ("PaddlePaddle/PP-OCRv6_tiny_det_onnx", "2ba1506c0380b8f0b03dd142459aac66d4421f6c"),
        "rec": ("PaddlePaddle/PP-OCRv6_tiny_rec_onnx", "2612ab37152ae0a677521bae4e1e3d4fb4cf7c30"),
    },
    "ppocrv6-small": {
        "det": ("PaddlePaddle/PP-OCRv6_small_det_onnx", "28fe5895c24fd108c19eb3e8479f4ab385fbfc62"),
        "rec": ("PaddlePaddle/PP-OCRv6_small_rec_onnx", "b8f84f0b80c529de40b4fbb3544b84fa7233a513"),
    },
    "ppocrv6-medium": {
        "det": (
            "PaddlePaddle/PP-OCRv6_medium_det_onnx",
            "61323801669c338b7891481ec7bac61ce31b576a",
        ),
        "rec": (
            "PaddlePaddle/PP-OCRv6_medium_rec_onnx",
            "50c7eacafc52fa7bcf4194e8cd08e46f8558504b",
        ),
    },
}


def url(repo: str, revision: str, name: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{revision}/{name}"


def yaml_scalar(text: str) -> str:
    """Decode one YAML scalar as PaddleOCR writes them (plain, '...' or "...")."""
    text = text.strip()
    if text.startswith("'") and text.endswith("'") and len(text) >= 2:
        return text[1:-1].replace("''", "'")
    if text.startswith('"') and text.endswith('"') and len(text) >= 2:
        decoded: str = json.loads(text)
        return decoded
    return text


def character_dict(yml: str) -> list[str]:
    lines = yml.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == "character_dict:")
    chars = []
    for line in lines[start + 1 :]:
        match = re.match(r"^  - (.*)$", line)
        if not match:
            break
        chars.append(yaml_scalar(match.group(1)))
    return chars


def number(yml: str, key: str) -> float:
    match = re.search(rf"^\s*{key}: ([0-9.]+)$", yml, re.M)
    assert match, key
    return float(match.group(1))


def rec_shape(yml: str) -> list[int]:
    match = re.search(r"image_shape:\n\s*- (\d+)\n\s*- (\d+)\n\s*- (\d+)", yml)
    assert match
    return [int(v) for v in match.groups()]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with httpx.Client(follow_redirects=True, timeout=300) as client:
        for name, parts in RELEASES.items():
            files = {}
            ymls = {}
            for kind, (repo, revision) in parts.items():
                ymls[kind] = (
                    client.get(url(repo, revision, "inference.yml")).raise_for_status().text
                )
                data = client.get(url(repo, revision, "inference.onnx")).raise_for_status().content
                files[kind] = {
                    "url": url(repo, revision, "inference.onnx"),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "size": len(data),
                }
                print(f"{name} {kind}: {len(data) / 1e6:.1f} MB")
            det, rec = ymls["det"], ymls["rec"]
            manifest = {
                "manifest_version": "1.0",
                "name": name,
                "version": "1.0",
                "license": "Apache-2.0",
                "source": f"https://huggingface.co/{parts['det'][0].rsplit('_det', 1)[0]}",
                "files": files,
                "detection": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                    "channel_order": "BGR",
                    "limit_side": 960,
                    "multiple_of": 32,
                    "threshold": number(det, "thresh"),
                    "box_threshold": number(det, "box_thresh"),
                    "unclip_ratio": number(det, "unclip_ratio"),
                },
                "recognition": {
                    "input_shape": rec_shape(rec),
                    "mean": [0.5, 0.5, 0.5],
                    "std": [0.5, 0.5, 0.5],
                    "channel_order": "BGR",
                    "decoder": "ctc",
                    # Class 0 is the CTC blank, then the charset, then a space (checked
                    # against the model output size: blank + charset + space).
                    "blank_index": 0,
                    "space_appended": True,
                    "charset": character_dict(rec),
                },
            }
            path = OUT / f"{name}.json"
            path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
            )
            count = len(character_dict(rec))
            print(f"wrote {path.relative_to(ROOT)} ({count} characters)")


if __name__ == "__main__":
    main()
