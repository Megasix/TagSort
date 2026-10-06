# Contributing to TagSort

Thank you for your interest in TagSort. This document explains how to set up a development
environment, which checks must pass, and the rules every contribution follows.

## Ground rules

- **English only.** Code, comments, docstrings, error messages, docs, commit messages, branch
  names, pull requests and test data are written in English.
- **No photos, datasets or model weights in the repository.** The `.gitignore` blocks images,
  `data/` folders and weight files; do not force-add them.
- **No secrets.** Never commit API keys, tokens or personal paths. Keys are passed in by the
  calling application at runtime.
- **No telemetry**, and no network access from the engine unless the API fallback is explicitly
  enabled by the caller.
- **Small public API.** Anything not exported from `tagsort/__init__.py` is private. Changes to
  the public API or to a JSON Schema in `schemas/` need maintainer approval first.

## Development setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Install the Python versions we test against:

   ```sh
   uv python install 3.10 3.11 3.12
   ```

3. Clone the repository and install all extras and dev tools into `.venv/`:

   ```sh
   git clone https://github.com/Megasix/TagSort.git
   cd TagSort
   uv sync --all-extras
   ```

4. Recommended VS Code extensions: Python, Ruff, Mypy Type Checker, Even Better TOML.

## Quality checks

Every pull request must pass the same checks CI runs:

```sh
uv lock --check              # uv.lock matches pyproject.toml (commit uv.lock)
uv run ruff check .          # lint
uv run ruff format --check . # formatting (run `uv run ruff format .` to fix)
uv run mypy                  # strict type checking
uv run pytest                # tests with coverage (minimum 90%)
```

CI also checks the licenses of runtime dependencies and extras (see
[Dependency license policy](#dependency-license-policy)).

To run the tests on another supported Python version without touching `.venv/`:

```sh
uv run --isolated --python 3.10 --all-extras pytest
```

Tests come first for the grammar, constrained decoding, calibration and schema conformance.
Every public function, class and method has a docstring (Google style).

## Commits and pull requests

- Use [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `docs:`,
  `test:`, `refactor:`, `perf:`, `chore:`, `ci:`. Use the imperative mood, e.g.
  `feat: add regex subset compiler`.
- Keep pull requests small, one topic each, and link them to a GitHub issue of the current
  milestone.
- Breaking changes to a JSON Schema create a new version (`v2`); `v1` stays supported.

## Dependency license policy

TagSort is Apache 2.0 and is meant to be embedded in other applications, so every dependency
that ships to users must be compatible.

| Status | Licenses |
| --- | --- |
| Accepted | MIT, BSD, Apache 2.0, ISC, PSF |
| Accepted, unmodified only | MPL 2.0 (for example `certifi`) |
| Rejected | GPL, AGPL, LGPL, SSPL and any other license not listed above |

- Open an issue before adding a dependency, a pretrained model or an API provider, and state its
  license.
- CI checks the licenses of runtime dependencies and of every extra (`api`, `server`, `train`)
  with `.github/scripts/check_licenses.py`. Dev-only tools are not distributed and are not
  checked. To run the check locally:

  ```sh
  uv sync --no-dev --all-extras
  uv run --no-sync python .github/scripts/check_licenses.py
  uv sync --all-extras
  ```

- The check fails closed: a package passes only if its SPDX expression, every license
  classifier, or its exact license name is on the allowlist. A package that fails but is
  acceptable after manual review is recorded, with the reason, in `REVIEWED` in that script.
- Pretrained model weights follow the same policy as code; check the license of the weights,
  not only of the code that trains them.

## License

By contributing, you agree that your contributions are licensed under the Apache License 2.0.
