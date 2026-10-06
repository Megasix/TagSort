import importlib.util
import sys
from email.message import Message
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / ".github" / "scripts" / "check_licenses.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_licenses", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_licenses"] = module
    spec.loader.exec_module(module)
    return module


check = _load()

MIT = "License :: OSI Approved :: MIT License"
GPL = "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"
PROPRIETARY = "License :: Other/Proprietary License"


@pytest.mark.parametrize(
    "expression",
    [
        "MIT",
        "Apache-2.0",
        "MPL-2.0",
        "Apache-2.0 OR BSD-2-Clause",
        "MIT AND BSD-3-Clause",
        "GPL-3.0-only OR MIT",
        "(MIT OR GPL-3.0-only) AND ISC",
        "mit or isc",
        "mit",
        "MIT-CMU",  # Pillow
        "BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0",  # numpy
    ],
)
def test_allowed_expressions(expression: str) -> None:
    assert check.expression_allowed(expression)


@pytest.mark.parametrize(
    "expression",
    [
        "GPL-3.0-only",
        "AGPL-3.0-or-later",
        "LGPL-2.1-only",
        "MIT AND GPL-3.0-only",
        "LicenseRef-Proprietary",
        "MIT AND LicenseRef-Proprietary",
        "Apache-2.0 WITH LLVM-exception",
        "(MIT OR GPL-3.0-only) AND SSPL-1.0",
        "CC-BY-NC-4.0",
        "CC-BY-SA-4.0",
        "OpenRAIL",
    ],
)
def test_rejected_expressions(expression: str) -> None:
    assert not check.expression_allowed(expression)


@pytest.mark.parametrize("expression", ["MIT AND", "(MIT", "MIT)", "AND MIT", ""])
def test_malformed_expressions_raise(expression: str) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        check.expression_allowed(expression)


def test_malformed_expression_is_a_problem() -> None:
    assert check.license_problem("MIT AND", [], None) is not None


def test_expression_takes_precedence_over_classifiers() -> None:
    assert check.license_problem("MIT", [GPL], None) is None
    assert check.license_problem("GPL-3.0-only", [MIT], None) is not None


def test_classifiers_must_all_be_allowed() -> None:
    assert check.license_problem(None, [MIT], None) is None
    assert check.license_problem(None, ["License :: OSI Approved", MIT], None) is None
    assert check.license_problem(None, [MIT, PROPRIETARY], None) is not None
    assert check.license_problem(None, [GPL], None) is not None


def test_classifiers_take_precedence_over_license_field() -> None:
    full_text = "Permission is hereby granted, free of charge, ..."
    assert check.license_problem(None, [MIT], full_text) is None


@pytest.mark.parametrize("field", ["MIT", "BSD License", "Apache License, Version 2.0", " isc "])
def test_license_field_exact_names_are_allowed(field: str) -> None:
    assert check.license_problem(None, [], field) is None


@pytest.mark.parametrize(
    "field",
    [
        "Proprietary. All rights reserved. Redistribution is not permitted.",
        "CC-BY-NC-4.0 commercial use prohibited, see disclaimer",
        "GPLv3",
        "MIT-like",
    ],
)
def test_license_field_free_text_is_rejected(field: str) -> None:
    assert check.license_problem(None, [], field) is not None


@pytest.mark.parametrize("field", [None, "", "   "])
def test_missing_license_is_rejected(field: str | None) -> None:
    assert check.license_problem(None, [], field) == "no license metadata"


class _FakeDist:
    def __init__(self, name: str, **fields: str) -> None:
        self.version = "1.0"
        self.metadata = Message()
        self.metadata["Name"] = name
        for key, value in fields.items():
            self.metadata[key.replace("_", "-")] = value


def test_main_passes_and_skips_self(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    dists = [_FakeDist("tagsort"), _FakeDist("Good_Pkg", License_Expression="MIT")]
    monkeypatch.setattr(check, "distributions", lambda: dists)
    assert check.main() == 0
    out = capsys.readouterr().out
    assert "good-pkg 1.0: ok" in out
    assert "tagsort" not in out


def test_main_fails_on_disallowed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    dists = [_FakeDist("good", License="MIT"), _FakeDist("bad", License_Expression="GPL-3.0-only")]
    monkeypatch.setattr(check, "distributions", lambda: dists)
    assert check.main() == 1
    assert "bad 1.0: FAIL" in capsys.readouterr().out


def test_main_accepts_reviewed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(check, "distributions", lambda: [_FakeDist("odd", License="Custom")])
    monkeypatch.setitem(check.REVIEWED, "odd", "checked by hand")
    assert check.main() == 0
    assert "reviewed: checked by hand" in capsys.readouterr().out
