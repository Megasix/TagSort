# TagSort

Open-source engine that reads handwritten or printed specimen ID tags in photos, in any orientation. Returns tag text, position, angle and a calibrated confidence score. Lightweight ONNX core, versioned JSON contracts and pluggable backends, built to embed in mobile, web and research apps.

[![CI](https://github.com/Megasix/TagSort/actions/workflows/ci.yml/badge.svg)](https://github.com/Megasix/TagSort/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

## Status

Early development. Profiles, result types and their JSON Schemas are in place; reading images
comes in the next milestones.

## Profiles

A profile describes the tags of one collection: what each kind of tag may say, as a pattern.

```json
{
  "schema_version": "1.0",
  "name": "Museum A",
  "tags_per_individual": 2,
  "tags": [
    { "id": "primary", "pattern": "MD\\d{5}" },
    { "id": "secondary", "pattern": "[A-Z]{2}-\\d{3,4}" }
  ]
}
```

```python
from tagsort import Profile

profile = Profile.from_file("museum_a.json")
profile.match("MD04127")  # "primary"
profile.match("MD4127")  # None
```

An invalid profile raises `ProfileError` when it is loaded, with the location of the problem.
Patterns use a small regex subset that describes a finite set of texts; see
[docs/patterns.md](docs/patterns.md).

## Contracts

Inputs and outputs are versioned JSON Schemas, so applications in any language can use them
without reading the Python code:

- [`schemas/profile.v1.json`](schemas/profile.v1.json): profiles;
- [`schemas/result.v1.json`](schemas/result.v1.json): read results, produced by
  `ReadResult.to_json()`;
- [`reference/grammar.v1.json`](reference/grammar.v1.json): test vectors for implementing the
  pattern grammar in another language.

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
