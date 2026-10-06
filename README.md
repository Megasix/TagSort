# TagSort

Open-source engine that reads handwritten or printed specimen ID tags in photos, in any orientation. Returns tag text, position, angle and a calibrated confidence score. Lightweight ONNX core, versioned JSON contracts and pluggable backends, built to embed in mobile, web and research apps.

[![CI](https://github.com/Megasix/TagSort/actions/workflows/ci.yml/badge.svg)](https://github.com/Megasix/TagSort/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

## Status

Early development. The repository currently contains the project skeleton only; there is no
usable reading API yet. The public API, JSON Schemas and models will land in the next milestones.

## Development setup

TagSort uses [uv](https://docs.astral.sh/uv/) and supports Python 3.10 and later.

```sh
uv python install 3.12
uv sync --all-extras
uv run pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the quality checks, commit style and dependency
license policy.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
