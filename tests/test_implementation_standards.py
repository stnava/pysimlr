"""Engineering rules for `src/pysimlr`, checked by reading the source.

A prose rule nobody checks is not a rule. Each check below parses every module
and fails on a site that is not on its allowlist; every allowlist entry says
why it is there. A new site is either fixed or added here with a reason --
the test is never worked around.

Rules:
  * no broad exception handlers (``except Exception``/``BaseException``/bare):
    each is a candidate silent fallback;
  * no global RNG seeding outside the seed utilities and the benchmark
    harness -- fits take ``random_state`` and run on a private stream;
  * no process-wide warning filter outside a ``warnings.catch_warnings()``
    block;
  * no mutable (list/dict/set literal) default arguments beyond the listed
    ones;
  * no ``**kwargs`` catch-all on a registered native fit, so a parameter can
    never vanish into one (`api` derives what each method accepts from it).
"""
import ast
import inspect
import pathlib
from collections import Counter


SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "pysimlr"

#: file -> number of broad handlers allowed, with the reason. Empty since the
#: D4 audit (2026-11-01) removed or narrowed all 37 remaining sites; a new
#: entry needs a reason a reviewer would accept for not naming the exception.
BROAD_EXCEPT_ALLOWED = {}

#: (file, enclosing function) -> reason global seeding is allowed there.
GLOBAL_SEED_ALLOWED = {
    ("utils.py", "set_all_seeds"): "the explicit seed utility",
    ("utils.py", "set_seed_based_on_time"): "the explicit seed utility",
    ("deep.py", "_seeded_rng"): "seeds inside torch.random.fork_rng, which restores the caller's "
        "CPU/CUDA generator state on exit",
    ("benchmarks/runner.py", "run_single_experiment"): "benchmark harness: pins the downstream "
        "metric estimators; the fit itself takes random_state",
    ("benchmarks/protocol.py", "run_repeated_benchmark"): "benchmark harness, per replicate",
    ("benchmarks/synthetic_cases.py", "build_linear_footprint_case"): "synthetic data generator",
    ("benchmarks/synthetic_cases.py", "build_nonlinear_shared_case"): "synthetic data generator",
    ("benchmarks/synthetic_cases.py", "build_shared_plus_private_case"): "synthetic data generator",
    ("benchmarks/synthetic_cases.py", "build_nonnegative_parts_case"): "synthetic data generator",
}

#: (file, function) -> reason a list/dict/set literal default is allowed.
MUTABLE_DEFAULT_ALLOWED = {
    ("benchmarks/runner.py", "sweep_benchmark"): "not mutated; benchmark entry, TODO(human) tuple-ize",
    ("benchmarks/synthetic_cases.py", "build_linear_footprint_case"): "not mutated; p_list",
    ("benchmarks/synthetic_cases.py", "build_nonlinear_shared_case"): "not mutated; p_list",
    ("benchmarks/synthetic_cases.py", "build_shared_plus_private_case"): "not mutated; p_list",
    ("benchmarks/synthetic_cases.py", "build_nonnegative_parts_case"): "not mutated; p_list",
    ("nnh.py", "extend_simlr_embedding_with_new_modalities"): "not mutated; split_prefixes",
}

SEED_CALLS = {"torch.manual_seed", "np.random.seed", "numpy.random.seed", "random.seed",
              "torch.cuda.manual_seed_all", "torch.cuda.manual_seed"}


def _modules():
    for f in sorted(SRC.rglob("*.py")):
        tree = ast.parse(f.read_text())
        parents = {c: n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}
        yield str(f.relative_to(SRC)), tree, parents


def _enclosing_function(node, parents):
    while node in parents:
        node = parents[node]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node.name
    return None


def test_broad_exception_handlers_do_not_grow():
    counts = Counter()
    for rel, tree, _ in _modules():
        for n in ast.walk(tree):
            if not isinstance(n, ast.ExceptHandler):
                continue
            t = n.type
            names = {"<bare>"} if t is None else (
                {t.id} if isinstance(t, ast.Name) else
                {e.id for e in getattr(t, "elts", []) if isinstance(e, ast.Name)})
            if names & {"Exception", "BaseException", "<bare>"}:
                counts[rel] += 1
    over = {rel: (n, BROAD_EXCEPT_ALLOWED.get(rel, (0, ""))[0]) for rel, n in counts.items()
            if n > BROAD_EXCEPT_ALLOWED.get(rel, (0, ""))[0]}
    assert not over, (f"new broad exception handler(s) (found, allowed): {over}. Catch the "
                      f"specific exception, or log and record provenance, or raise.")
    shrunk = {rel: (counts.get(rel, 0), n) for rel, (n, _) in BROAD_EXCEPT_ALLOWED.items()
              if counts.get(rel, 0) < n}
    assert not shrunk, f"handlers were removed (found, allowed): {shrunk}; lower the allowlist."


def test_no_global_seeding_outside_the_allowlist():
    bad = []
    for rel, tree, parents in _modules():
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and ast.unparse(n.func) in SEED_CALLS:
                key = (rel, _enclosing_function(n, parents))
                if key not in GLOBAL_SEED_ALLOWED:
                    bad.append((*key, n.lineno))
    assert not bad, f"global RNG seeding in library code: {bad}; take random_state instead."


def test_warning_filters_are_scoped():
    bad = []
    for rel, tree, parents in _modules():
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and ast.unparse(n.func) in ("warnings.filterwarnings",
                                                                  "warnings.simplefilter"):
                p, scoped = n, False
                while p in parents:
                    p = parents[p]
                    if isinstance(p, ast.With) and any(
                            "catch_warnings" in ast.unparse(i.context_expr) for i in p.items):
                        scoped = True
                        break
                if not scoped:
                    bad.append((rel, n.lineno))
    assert not bad, f"process-wide warning filter outside catch_warnings(): {bad}"


def test_no_new_mutable_default_arguments():
    bad = []
    for rel, tree, _ in _modules():
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defaults = n.args.defaults + [d for d in n.args.kw_defaults if d is not None]
                if any(isinstance(d, (ast.List, ast.Dict, ast.Set)) for d in defaults):
                    if (rel, n.name) not in MUTABLE_DEFAULT_ALLOWED:
                        bad.append((rel, n.name, n.lineno))
    assert not bad, f"mutable default argument(s): {bad}; use None or a tuple."


def test_registered_native_fits_have_no_kwargs_catch_all():
    from pysimlr import api
    bad = []
    for name, spec in api.METHODS.items():
        params = inspect.signature(spec.fit).parameters.values()
        var_kw = [p.name for p in params if p.kind is p.VAR_KEYWORD]
        # `simlr` takes optimizer hyperparameters through **opt_params and
        # validates them against SIMLR_OPTIMIZER_KEYS (unknown keys raise).
        if var_kw and spec.fit.__name__ != "simlr":
            bad.append((name, var_kw))
    assert not bad, f"registered fits with a **kwargs catch-all: {bad}"


def test_every_parameter_of_a_registered_fit_is_read():
    """A parameter that is accepted, documented and recorded as "used" but
    never read is the failure `api`'s parameter record cannot see on its own:
    `ned_simr_shared_private(private_recon_weight=...)` was such a parameter.
    Every named parameter must be loaded somewhere in the function body."""
    import textwrap
    from pysimlr import api
    dead = []
    for name, spec in api.METHODS.items():
        fn = spec.fit
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        fdef = tree.body[0]
        params = [a.arg for a in fdef.args.args + fdef.args.kwonlyargs]
        loaded = {n.id for n in ast.walk(fdef) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        # `simlr` re-binds its own arguments through locals() for the
        # learning-rate probes; that counts as a read of every one of them.
        if "locals" in loaded:
            continue
        unused = [p for p in params if p not in loaded]
        if unused:
            dead.append((name, unused))
    assert not dead, f"registered fits with parameters that are never read: {dead}"


#: Line substrings that may legitimately contain a removed parameter name:
#: local loops with their own `epochs`, CLI arguments, and the per-unit budget map.
_LEGACY_NAME_OK = ("warmup_epochs", "ramp_epochs", "budget_by_unit",
                   "fit_encoders", "args.epochs", "a.epochs", "def ", "run_unified_benchmark(",
                   "run_real_benchmark(", "run_single_experiment(", "run_seeded_benchmark",
                   "sweep_benchmark", "run_deep_benchmark", "run_operating_point_suite",
                   "workers=workers")  # continuation lines of run_deep_benchmark(...)


def test_scripts_and_docs_use_the_current_parameter_names():
    """Plan gate 1 (docs/plans/2026-11-01_central_interface.md), as a test.

    The AST rename in 0.3.0 reached only direct calls; indirect calls,
    wrapper **kwargs and dict-built arguments kept `epochs`,
    `sparseness_quantile` and `init_seed`, which now raise -- or, through a
    local signature filter, were silently dropped."""
    import re
    root = SRC.parents[1]
    pattern = re.compile(r"(^|[^_a-zA-Z.])epochs=|[\"']epochs[\"']\s*:|init_seed|sparseness_quantile"
                         r"|[\"']sparsity[\"']\s*:|filter_kwargs")
    files = [*root.glob("scripts/*.py"), *root.glob("paper/*.py"), *root.glob("paper/*.qmd"),
             *root.glob("docs/*.qmd"), root / "README.md"]
    bad = []
    for f in files:
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if line.lstrip().startswith("#") or not pattern.search(line):
                continue
            if any(ok in line for ok in _LEGACY_NAME_OK):
                continue
            bad.append(f"{f.relative_to(root)}:{i}: {line.strip()[:90]}")
    # Prose in a docstring describing a historical budget is not a call.
    bad = [b for b in bad if "fixed budget used earlier" not in b]
    assert not bad, "removed parameter names in use:\n" + "\n".join(bad)
