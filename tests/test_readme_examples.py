"""The README's Quick Start runs as written.

Two of its three snippets were broken before 0.3.0: `simlr(optimizer=...,
random_state=...)` silently ran the default optimizer unseeded, and
`SiMLR(optimizer=..., sparseness=...)` raised `TypeError`. Nothing executed
them. This test runs every fenced ``python`` block under "## Quick Start", in
order, in one namespace.
"""
import pathlib
import re
import warnings

README = pathlib.Path(__file__).resolve().parents[1] / "README.md"


def _quick_start_blocks():
    text = README.read_text()
    start = text.index("## Quick Start")
    end = text.index("\n## ", start + 1)
    return re.findall(r"```python\n(.*?)```", text[start:end], flags=re.S)


def test_the_quick_start_has_code():
    assert len(_quick_start_blocks()) >= 3


def test_the_quick_start_runs():
    namespace = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, block in enumerate(_quick_start_blocks()):
            try:
                exec(compile(block, f"README quick start block {i + 1}", "exec"), namespace)
            except Exception as err:  # noqa: BLE001 -- re-raised with the block number
                raise AssertionError(f"README quick start block {i + 1} failed: {err!r}") from err
