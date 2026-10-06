"""Fail if an installed distribution has a license outside the project allowlist.

Run it with the interpreter of the environment to check, for example:

    uv sync --no-dev --all-extras
    uv run --no-sync python .github/scripts/check_licenses.py

The check fails closed: a distribution passes only when its license can be shown to be on
the allowlist (CLAUDE.md §5). Sources, in order of precedence:

1. ``License-Expression`` (PEP 639): parsed as SPDX; ``OR`` needs one allowed branch,
   ``AND`` needs every operand allowed, ``WITH`` is rejected.
2. ``License ::`` classifiers: every specific classifier must be allowed.
3. ``License`` field: must match an allowed name exactly (case-insensitive).

Anything else, including free-text license fields, fails and must be reviewed by hand and
recorded in ``REVIEWED``. Uses the standard library only.
"""

from __future__ import annotations

import re
import sys
from importlib.metadata import distributions

ALLOWED_SPDX = frozenset(
    {
        "MIT",
        "MIT-0",
        "MIT-CMU",
        "HPND",
        "Zlib",
        "CC0-1.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "0BSD",
        "Apache-2.0",
        "ISC",
        "PSF-2.0",
        "Python-2.0",
        "MPL-2.0",
    }
)

ALLOWED_CLASSIFIERS = frozenset(
    {
        "License :: OSI Approved :: MIT License",
        "License :: OSI Approved :: MIT No Attribution License (MIT-0)",
        "License :: OSI Approved :: Historical Permission Notice and Disclaimer (HPND)",
        "License :: OSI Approved :: zlib/libpng License",
        "License :: CC0 1.0 Universal (CC0 1.0) Public Domain Dedication",
        "License :: OSI Approved :: BSD License",
        "License :: OSI Approved :: Apache Software License",
        "License :: OSI Approved :: ISC License (ISCL)",
        "License :: OSI Approved :: Python Software Foundation License",
        "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
    }
)

_ALLOWED_SPDX_LOWER = frozenset(spdx.lower() for spdx in ALLOWED_SPDX)

# Classifiers that carry no license information on their own.
NEUTRAL_CLASSIFIERS = frozenset({"License :: OSI Approved"})

ALLOWED_LICENSE_NAMES = _ALLOWED_SPDX_LOWER | frozenset(
    {
        "mit license",
        "zlib license",
        "bsd",
        "bsd license",
        "new bsd license",
        "3-clause bsd license",
        "simplified bsd license",
        "apache 2.0",
        "apache license 2.0",
        "apache license, version 2.0",
        "apache software license",
        "isc license",
        "psf",
        "python software foundation license",
        "mpl 2.0",
        "mozilla public license 2.0",
    }
)

# Distributions reviewed by hand: normalized name -> reason. Keep this list short.
REVIEWED: dict[str, str] = {}

# Distributions that are part of this project, not dependencies.
SELF = frozenset({"tagsort"})

_TOKEN = re.compile(r"\(|\)|[^\s()]+")


class _Parser:
    """Recursive-descent evaluator for SPDX license expressions."""

    def __init__(self, expression: str) -> None:
        self.tokens = _TOKEN.findall(expression)
        self.pos = 0

    def _peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _take(self) -> str:
        token = self._peek()
        if token is None:
            raise ValueError("unexpected end of expression")
        self.pos += 1
        return token

    def parse(self) -> bool:
        result = self._or()
        if self._peek() is not None:
            raise ValueError(f"unexpected token {self._peek()!r}")
        return result

    def _or(self) -> bool:
        result = self._and()
        while (token := self._peek()) is not None and token.upper() == "OR":
            self._take()
            right = self._and()
            result = result or right
        return result

    def _and(self) -> bool:
        result = self._atom()
        while (token := self._peek()) is not None and token.upper() == "AND":
            self._take()
            right = self._atom()
            result = result and right
        return result

    def _atom(self) -> bool:
        token = self._take()
        if token == "(":
            result = self._or()
            if self._take() != ")":
                raise ValueError("missing closing parenthesis")
        elif token == ")" or token.upper() in {"AND", "OR", "WITH"}:
            raise ValueError(f"unexpected token {token!r}")
        else:
            # SPDX identifiers match case-insensitively.
            result = token.lower() in _ALLOWED_SPDX_LOWER
        if (nxt := self._peek()) is not None and nxt.upper() == "WITH":
            # License exceptions change the terms; review them by hand.
            self._take()
            self._take()
            result = False
        return result


def expression_allowed(expression: str) -> bool:
    """Return whether an SPDX license expression is satisfied by the allowlist."""
    return _Parser(expression).parse()


def license_problem(
    expression: str | None, classifiers: list[str], license_field: str | None
) -> str | None:
    """Return why a distribution's license is not allowed, or None if it is allowed."""
    if expression and expression.strip():
        try:
            allowed = expression_allowed(expression)
        except ValueError as exc:
            return f"unparseable License-Expression {expression!r}: {exc}"
        return None if allowed else f"License-Expression {expression!r} is not allowed"

    specific = [
        c for c in classifiers if c.startswith("License ::") and c not in NEUTRAL_CLASSIFIERS
    ]
    if specific:
        rejected = [c for c in specific if c not in ALLOWED_CLASSIFIERS]
        return f"classifiers not allowed: {rejected}" if rejected else None

    if license_field and license_field.strip():
        if license_field.strip().lower() in ALLOWED_LICENSE_NAMES:
            return None
        summary = license_field.strip().splitlines()[0][:80]
        return f"License field {summary!r} is not a known allowed license"

    return "no license metadata"


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def main() -> int:
    """Check every distribution in the running environment; return the exit code."""
    failures: list[str] = []
    seen: set[str] = set()
    for dist in distributions():
        name = _normalize(dist.metadata["Name"])
        if name in seen or name in SELF:
            continue
        seen.add(name)
        meta = dist.metadata
        expressions = meta.get_all("License-Expression") or []
        licenses = meta.get_all("License") or []
        problem = license_problem(
            expressions[0] if expressions else None,
            meta.get_all("Classifier") or [],
            licenses[0] if licenses else None,
        )
        if problem is None:
            status = "ok"
        elif name in REVIEWED:
            status = f"reviewed: {REVIEWED[name]}"
        else:
            status = f"FAIL: {problem}"
            failures.append(name)
        print(f"{name} {dist.version}: {status}")

    if failures:
        print(f"\n{len(failures)} distribution(s) outside the license allowlist: {failures}")
        return 1
    print(f"\n{len(seen)} distribution(s) checked, all allowed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
