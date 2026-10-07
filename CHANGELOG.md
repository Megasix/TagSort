# Changelog

All notable changes to TagSort. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). Contracts (`schemas/`) are versioned
separately: a breaking change creates a new major version of the schema.

## Unreleased

### Added

- `LocalPipeline(device=..., threads=...)`: read on an NVIDIA GPU with `device="cuda"`, or
  `device="auto"` to use one when available; `threads` sets the CPU threads per photo.
- Server settings `TAGSORT_DEVICE` and `TAGSORT_THREADS`.
- `docker/Dockerfile.gpu`: the server on a GPU (`onnxruntime-gpu`, CUDA 13), published as
  `ghcr.io/megasix/tagsort:<version>-gpu`.

### Changed

- ONNX Runtime threads and the server's default concurrency now follow the CPUs the
  process may really use (affinity and container CPU quota) instead of the host's cores.

## [0.1.0] - 2026-10-06

First public release.

### Added

- **Local reading** with PP-OCRv6 detection and recognition models (tiny 6 MB, small 31 MB,
  medium 139 MB; Apache 2.0, official ONNX releases), downloaded only on request and checked
  against SHA-256 manifests: `tagsort models download`.
- **Profiles** (`profile.v1.json`) with a regex subset compiled to a minimal automaton;
  invalid patterns are rejected with stable error codes and positions.
- **Grammar-constrained decoding**: recognition only returns texts the profile accepts.
- **Results** (`result.v1.json`) with text, confidence, status (`accepted`, `review`,
  `unreadable`), polygon, angle, candidates and source. A reading is accepted only when it
  fits the profile; tags that read differently upside down always go to review.
- **Vision API providers**: Gemini (recommended default), Anthropic, OpenAI and DeepSeek,
  as a whole-photo backend or as a fallback that receives only the crop of doubtful tags.
- `Reader.read_batch(workers=...)`: parallel reading in input order with constant memory.
- **Command line**: `tagsort read` (one JSON line per photo), `tagsort serve`,
  `tagsort models`.
- **HTTP server** (`server` extra, Starlette) implementing `api.v1.openapi.json`, and a
  **Docker image** with the tiny and small models.
- **Evaluation**: `tagsort-eval run`, `compare` and `prelabel` (AI pre-labeling with a
  plain review page), with reports in `eval/reports/`.
- **Reference vectors** for porting the pattern grammar and the local pipeline to other
  languages.

### Measured

On the reference set (160 photos, 60 printed tags): zero silent errors in every
configuration; `ppocrv6-medium` read 60 of 60 locally, `ppocrv6-tiny` 58 of 60 at 283
photos per minute on a laptop, Gemini Flash-Lite 60 of 60 for about 0.51 USD per 1,000
photos. Confidence scores are provisional until calibration on more labeled data.

[0.1.0]: https://github.com/Megasix/TagSort/releases/tag/v0.1.0
