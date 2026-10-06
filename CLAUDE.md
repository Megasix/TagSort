# CLAUDE.md: TagSort

This file is read at the start of every session. It describes the project, the rules that apply, and how we work. When in doubt, it takes precedence over any other habit.

## 0. Repository description

> Open-source engine that reads handwritten or printed specimen ID tags in photos, in any orientation. Returns tag text, position, angle and a calibrated confidence score. Lightweight ONNX core, versioned JSON contracts and pluggable backends, built to embed in mobile, web and research apps.

Use this text for the GitHub repository description, the PyPI summary and the first paragraph of the README.

## 1. Language rule

**Everything in this repository is written in English.** No French anywhere:

- code identifiers, comments, docstrings, type hints, error messages and log messages;
- README, CONTRIBUTING, docs, model cards, schemas and their descriptions;
- commit messages, branch names, pull request titles and descriptions, issue titles;
- test names, fixtures and sample data (use neutral names such as `museum_a.json`).

The maintainer may write to Claude in French. Claude answers in French in the conversation, but everything written to the repository stays in English.

## 2. Project overview

TagSort is an open-source engine that reads identification tags on specimen photos: short text, often handwritten, in any orientation. It takes an image and returns every tag found, its position, its orientation, the text read and a confidence score.

TagSort is a **building block**, not an application. It will be embedded in:

- **FieldSort** (Flutter, Android and iOS): on-device model, offline reading;
- **BenchSort** (Next.js web app): in-browser detection, batch recognition on a hosted service;
- any third-party application (museums, universities, other developers).

Every decision must pass one test: can another developer, in another language, use this without reading our code?

## 3. Design principles (non-negotiable)

1. **Small, stable public API.** Few entry points, clearly named, documented, versioned with SemVer. Anything not exported from `tagsort/__init__.py` is private.
2. **No side effects.** The engine reads images and returns data. It never renames, moves or writes files, and never touches the network unless the API fallback is explicitly enabled.
3. **Contracts in JSON, not in Python.** Configuration inputs (profiles) and outputs (results) follow versioned JSON Schemas published in `schemas/`. Dart and TypeScript must be able to consume them without translation.
4. **Lightweight core, optional extras.** Inference runs on ONNX Runtime. PyTorch and training tools are never required to read a tag.
5. **Swappable backends.** Detection, orientation, recognition and fallback are interfaces (`Protocol`). Each stage can be replaced without touching the others.
6. **Portable models.** Every published model ships with a manifest (input size, normalization, charset, version) and reference vectors, so a Dart or JavaScript implementation can verify it produces the same results.
7. **No telemetry.** Ever.
8. **A silent error is worse than a doubt.** The engine flags a tag for review rather than return a wrong reading with high confidence.

## 4. Out of scope

These belong to the applications, not to TagSort:

- grouping photos into specimens ("tag first" and "tag in every photo" modes);
- naming templates and file renaming (`naming-spec` repository);
- accounts, credits, billing;
- review user interface.

If a task touches any of these, stop and ask.

## 5. Required stack

| Item | Choice |
| --- | --- |
| Language | Python ≥ 3.10, fully typed |
| Project management | `uv`, `pyproject.toml` (`hatchling` build backend) |
| Quality | `ruff` (lint and format), `mypy --strict`, `pytest`, `pytest-cov` |
| Inference | `onnxruntime`, `numpy`, `pillow` |
| Extras | `tagsort[api]` (vision API fallback), `tagsort[server]` (FastAPI), `tagsort[train]` (PyTorch and training tools) |
| CI | GitHub Actions: lint, type check, tests, license check, engine evaluation |
| License | Apache 2.0 |

Ask before adding any dependency, and state its license. AGPL and GPL are rejected (known case: Ultralytics YOLO is AGPL). Accepted licenses: MIT, MIT-CMU and HPND (for example Pillow), BSD, Apache 2.0, ISC, PSF, Zlib, CC0 1.0 (parts of numpy), and MPL 2.0 for unmodified dependencies only (for example `certifi`). Claude may add another permissive license to this list without asking, and reports it; copyleft licenses and licenses with use restrictions always need the maintainer's approval. CI fails if a runtime or extra dependency falls outside this list; dev-only tools are not distributed and are not checked.

## 6. Environment setup

The workstation starts with VS Code only.

1. Install `uv` following Astral's official documentation.
2. `uv python install 3.12`
3. Clone the repository, then `uv sync --all-extras`
4. VS Code extensions: Python, Ruff, Mypy Type Checker, Even Better TOML.
5. Check: `uv run pytest` and `uv run ruff check .`

Claude gives each command and explains what it does before running it.

## 7. Repository layout

```
tagsort/
├── CLAUDE.md
├── README.md
├── LICENSE                  # Apache 2.0
├── CONTRIBUTING.md
├── pyproject.toml
├── schemas/
│   ├── profile.v1.json      # naming profile
│   └── result.v1.json       # read result
├── src/tagsort/
│   ├── __init__.py          # public API only
│   ├── types.py             # public dataclasses
│   ├── profile.py           # profile loading and validation
│   ├── grammar/             # regex subset → automaton, constrained decoding
│   ├── pipeline/
│   │   ├── reader.py        # cascade orchestration
│   │   ├── preprocess.py    # EXIF, resizing
│   │   ├── detect.py        # interface + ONNX implementation
│   │   ├── orient.py
│   │   ├── recognize.py
│   │   └── calibrate.py
│   ├── fallback/            # interface + vision API providers
│   ├── models/              # model download and manifests
│   ├── cli.py
│   └── server.py            # [server] extra
├── training/                # [train] extra, never imported by the core
├── eval/                    # evaluation harness, versioned reports
├── reference/               # images and expected outputs for validating ports
├── tests/
└── docker/
```

## 8. Public API

```python
from tagsort import Reader, Profile

profile = Profile.from_file("museum_a.json")   # or Profile.from_dict(...)
reader = Reader(profile=profile, backend="local", fallback=None)

result = reader.read("IMG_0412.jpg")           # path, bytes, numpy array or PIL image
for tag in result.tags:
    print(tag.tag_id, tag.text, tag.confidence, tag.status)

for result in reader.read_batch(paths):        # generator, constant memory
    ...

result.to_json()                               # conforms to result.v1.json
```

Rules:

- `Reader` is reusable and thread-safe once built; models load once.
- `accept_threshold` and `review_threshold` are configurable; the calling app decides how to present results.
- Errors are raised as typed exceptions (`TagSortError` and subclasses), never printed.
- Logging uses the standard `logging` module only, `WARNING` level by default.

## 9. Data contracts

### Profile (`profile.v1.json`)

```json
{
  "schema_version": "1.0",
  "name": "Museum A",
  "tags_per_individual": 2,
  "tags": [
    { "id": "primary",   "pattern": "MD\\d{5}" },
    { "id": "secondary", "pattern": "[A-Z]{2}-\\d{3,4}" }
  ]
}
```

Patterns use a **regex subset** that compiles to a finite automaton: literals, character classes `[...]`, `\d`, quantifiers `{n}` and `{n,m}`, alternation `(a|b)`. No anchors, no `*` or `+`, no backreferences. A pattern outside this subset raises a clear error at load time.

### Result (`result.v1.json`)

```json
{
  "schema_version": "1.0",
  "engine_version": "0.1.0",
  "model_version": "det-1.0+rec-1.0",
  "image": { "width": 4032, "height": 3024, "exif_rotation": 90 },
  "tags": [
    {
      "tag_id": "primary",
      "text": "MD04127",
      "confidence": 0.97,
      "status": "accepted",
      "polygon": [[812, 400], [1210, 410], [1205, 560], [808, 552]],
      "angle": 180,
      "candidates": [{ "text": "MD04121", "confidence": 0.02 }],
      "source": "local"
    }
  ],
  "timings_ms": { "detect": 41, "recognize": 18 }
}
```

`status` is `accepted`, `review` or `unreadable`. `source` is `local` or `fallback`. Coordinates are in pixels of the original image, after EXIF rotation.

Any breaking schema change creates a `v2`; `v1` remains supported.

## 10. Pipeline

1. **Preprocessing**: apply EXIF orientation, resize to working size.
2. **Detection**: oriented polygon for each tag, or no tag.
3. **Orientation**: 0/90/180/270 classifier, then fine correction from the polygon; when uncertain, read all four rotations.
4. **Recognition**: handwritten text model on the crop.
5. **Constrained decoding**: the profile grammar restricts allowed characters at each position.
6. **Validation and calibration**: calibrated score, status from thresholds.
7. **Fallback** (optional): below threshold, only the crop is sent to the vision API provider, with the grammar in the prompt; the answer goes back through validation.

Fallback is off by default. The API key comes from the calling application, never from a file in the repository.

## 11. Evaluation

- Dataset format: a folder of images and a `labels.jsonl` file (image, tag_id, text, polygon).
- Split by session or site, never at random.
- Metrics: exact match, character error rate, **silent error rate** (wrong text with `accepted` status), share sent to fallback, cost per 1,000 photos, time per photo.
- `uv run tagsort-eval` writes a Markdown and JSON report, versioned under `eval/reports/`.
- In CI, a drop in accuracy or a rise in silent errors on the test set blocks the merge.

Training data comes from the beta: tags validated by curators, with consent. No photo ever enters the repository. Only anonymized crops, stripped of EXIF and GPS, may one day be published, and only with written approval.

## 12. Milestones

One milestone at a time. At the end of each, Claude stops, summarizes what was done and waits for approval before the next.

| Milestone | Scope | Exit criteria |
| --- | --- | --- |
| M0 | Repository skeleton, `pyproject.toml`, ruff, mypy, pytest, CI, license check, LICENSE, CONTRIBUTING | Green CI on an empty package |
| M1 | Public types, JSON Schemas, profile loading, regex subset → automaton | Exhaustive grammar tests, schemas validated |
| M2 | End-to-end `Reader` with the `api` backend only (vision fallback) | Correct reads on a real photo batch; output conforms to schema |
| M3 | Evaluation harness and first report (API backend as baseline) | Versioned report with numbers |
| M4 | Local pipeline: detection, orientation, recognition, constrained decoding, calibration, cascade | Target set in M3 reached |
| M5 | Training scripts, fine-tuning, ONNX and LiteRT (formerly TFLite) export, manifests, reference vectors | Same results in Python and on the export, within tolerance |
| M6 | CLI, HTTP server, Docker image | `docker run` reads a folder with no Python install |
| M7 | v0.1 release: PyPI, weights on Hugging Face, model card, documentation | A newcomer reads a tag in under five minutes using the README |

M2 comes before the local pipeline: the BenchSort beta starts on the API fallback and produces the data that will train the local model.

## 13. Working method

- **Tests first** for the grammar, constrained decoding, calibration and schema conformance.
- Small pull requests, one topic each, linked to a GitHub issue of the current milestone.
- Docstrings on the whole public API; runnable examples in the docs.
- Before adding a dependency, a pretrained model or an API provider: ask, and state the license.
- Before any change to a schema or the public API: ask.
- Never write keys, tokens or personal paths into the repository.
- If a rule in this file seems to block a better solution, say so and propose an edit to this file instead of working around it.

## 14. Open decisions

- Starting models (PaddleOCR, docTR, TrOCR, PARSeq): after M3 numbers, licenses verified.
- Vision fallback provider for the beta: compare cost and accuracy in M2.
- Export route to LiteRT and Flutter inference package: validate in M5.
