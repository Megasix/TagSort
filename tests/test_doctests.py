import doctest
import importlib
import pkgutil

import pytest

import tagsort

MODULES = sorted(info.name for info in pkgutil.walk_packages(tagsort.__path__, prefix="tagsort."))


@pytest.mark.parametrize("name", MODULES)
def test_docstring_examples(name: str) -> None:
    result = doctest.testmod(importlib.import_module(name), optionflags=doctest.ELLIPSIS)
    assert result.failed == 0
