"""Run the docstring examples in `pysimlr` as part of the normal suite.

Several examples were wrong or flaky and nothing ran them: `rvcoef`'s used an
unseeded `torch.randn`, so its expected `1.0` matched only when float32 rounding
happened to cooperate; `map_asym_var` and `map_lr_average_var` printed float32
digits for code that casts to float64; and the smoothing-operator examples
referenced names that do not exist.
"""
import doctest
import importlib
import pkgutil

import pytest

import pysimlr


def _module_names():
    names = []
    for _, name, is_pkg in pkgutil.walk_packages(pysimlr.__path__, "pysimlr."):
        if is_pkg:
            continue
        names.append(name)
    return sorted(names)


@pytest.mark.parametrize("module_name", _module_names())
def test_module_doctests(module_name):
    module = importlib.import_module(module_name)
    results = doctest.testmod(
        module,
        optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE,
        verbose=False,
    )
    assert results.failed == 0, (
        f"{module_name}: {results.failed} of {results.attempted} doctests failed"
    )
