"""The package must not pull plotting or optional heavy dependencies at import.

`__init__` used to call `matplotlib.use("Agg")` at module level and eagerly
import `visualization` and `benchmarks`, the latter dragging in seaborn,
scipy.stats and ipywidgets. Resolving the optional NSA-Flow backend at import
time re-added matplotlib through that package's own utils module. Import cost
was 2.27s; it is now about 1.2s.
"""
import subprocess
import sys


def _probe(code):
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=180
    )


def test_heavy_optional_modules_are_not_imported_eagerly():
    r = _probe(
        "import sys, pysimlr; "
        "print(','.join(m for m in ('matplotlib','seaborn','sklearn','nsa_flow') "
        "if m in sys.modules))"
    )
    assert r.returncode == 0, r.stderr
    eager = [m for m in r.stdout.strip().split(",") if m]
    assert eager == [], f"imported eagerly: {eager}"


def test_plotting_entry_points_still_resolve():
    r = _probe(
        "import pysimlr; "
        "assert callable(pysimlr.plot_path_model); "
        "assert callable(pysimlr.generate_all_architecture_graphs); "
        "assert pysimlr.benchmarks.__name__.endswith('benchmarks'); "
        "assert pysimlr.viz.__name__.endswith('viz'); "
        "print('ok')"
    )
    assert r.returncode == 0, r.stderr
    assert "ok" in r.stdout


def test_every_public_name_resolves():
    r = _probe(
        "import pysimlr; "
        "missing=[n for n in pysimlr.__all__ if not hasattr(pysimlr,n)]; "
        "assert not missing, missing; print('ok')"
    )
    assert r.returncode == 0, r.stderr


def test_unknown_attribute_raises_attribute_error():
    r = _probe(
        "import pysimlr\n"
        "try:\n"
        "    pysimlr.definitely_not_a_thing\n"
        "except AttributeError:\n"
        "    print('ok')\n"
    )
    assert "ok" in r.stdout, r.stderr


def test_nsa_backend_is_reported_for_provenance():
    """Which retraction backend resolved must be inspectable: results differ
    between environments that do and do not have it installed."""
    from pysimlr.nsa_backend import backend_report

    rep = backend_report()
    assert set(rep) == {"available", "module", "entry_point"}
    assert isinstance(rep["available"], bool)
    if rep["available"]:
        assert rep["module"] in ("nsa_flow", "nsa")
        assert rep["entry_point"] is not None
