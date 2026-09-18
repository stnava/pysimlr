"""Single source of truth for every statistic reported in the paper.

Before this module existed, the inference was implemented three times with
three different sets of choices: inline in `03_experiments.qmd`, inline in
`appendix_flow_simr_v.qmd`, and again in `calc_stats.py`. They disagreed on

* **which architectures are "Deep"** -- `calc_stats.py` put LEND in the
  *Linear* group while the manuscript text and the `.qmd` blocks put it in
  Deep. LEND is a Linear Encoder with a *Nonlinear* Decoder, so Deep is
  correct; grouping it as Linear diluted the contrast and understated the
  effect being argued for.
* **whether Flow-SiMLR-V is included at all** -- it was absent from
  `calc_stats.py`'s map, so `pandas.Series.map` returned NaN and all of its
  rows were silently dropped; the `.qmd` blocks included it.
* **what the consensus groups are called** -- "Algebraic/Iterative" in the
  `.qmd` blocks versus "SVD/PCA" and "Newton/ICA" in `calc_stats.py`.
* **what the unit of replication is** -- all three pooled every row into a
  two-sample Welch t-test.

That last point is the substantive one. The benchmark crosses loss functions
and consensus algorithms over a handful of seeds, so a cell holding 200 rows
holds only 5 *independent* replicates; the rest are repeated measures on the
same data splits. A two-sample t-test treats them as independent, which
understates the standard error and yields p-values that shrink with compute
rather than with evidence. Everything here aggregates within seed first.

With `n` independent replicates the smallest attainable two-sided sign-flip
permutation p-value is `2 / 2**n` -- 0.0625 for five seeds. No p-value below
that floor is supportable from such a design by any distribution-free method,
so `PERMUTATION_FLOOR_NOTE` is quoted wherever results are tabulated.
"""
from __future__ import annotations

import itertools
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

#: Results-cache version the manuscript reports. Every chapter resolves its
#: data path through :func:`cache_path`, so moving to a new benchmark run is a
#: one-line change here rather than an edit in each code block.
RESULTS_VERSION = "v21"


def cache_path(kind: str, version: str | None = None) -> str:
    """
    Path to a vendored results cache.

    Parameters
    ----------
    kind : str
        "synthetic" or "real".
    version : str, optional
        Version tag; defaults to `RESULTS_VERSION`.

    Returns
    -------
    str
        Path relative to the paper directory.
    """
    if kind not in ("synthetic", "real"):
        raise ValueError(f"kind must be 'synthetic' or 'real', got {kind!r}")
    return f"results_cache/unified_{kind}_{version or RESULTS_VERSION}.csv"


#: Architecture grouping. LEND's encoder is linear; the architecture is not.
ARCH_GROUP: Dict[str, str] = {
    "SiMLR": "Linear",
    "linear": "Linear",
    "LEND": "Deep",
    "lend": "Deep",
    "NED": "Deep",
    "ned": "Deep",
    "NEDPP": "Deep",
    "shared_private": "Deep",
    "Flow-SiMLR-V": "Deep",
    "flow_simlr_v": "Deep",
}

#: Canonical display names. The runner emits snake_case identifiers; the
#: manuscript refers to the architectures by their published names. This
#: mapping was previously repeated inline in three separate code blocks.
DISPLAY_NAMES: Dict[str, str] = {
    "linear": "SiMLR",
    "lend": "LEND",
    "ned": "NED",
    "shared_private": "NEDPP",
    "flow_simlr_v": "Flow-SiMLR-V",
}

#: Consensus grouping, named for the algorithms themselves rather than for a
#: characterisation of them ("Algebraic"/"Iterative" was used elsewhere).
CONSENSUS_GROUP: Dict[str, str] = {
    "svd": "SVD/PCA",
    "pca": "SVD/PCA",
    "newton": "Newton/ICA",
    "ica": "Newton/ICA",
}

#: Beyond this many replicates, enumerating all sign patterns is infeasible.
MAX_EXACT_REPLICATES = 22

PERMUTATION_FLOOR_NOTE = (
    "Inference is at the level of the independent replicate (the random seed). "
    "With $n$ seeds the smallest attainable two-sided sign-flip permutation "
    "$p$-value is $2/2^{n}$; for $n = 5$ this floor is $0.0625$. Values at the "
    "floor indicate perfect seed-consistency, which is the strongest statement "
    "this design supports. We do not report $p$-values below the floor."
)


def load_results(path: str | Path) -> pd.DataFrame:
    """
    Read a results cache and check the columns the analyses depend on.

    Parameters
    ----------
    path : str or Path
        CSV produced by the benchmark runner.

    Returns
    -------
    pd.DataFrame
        The results, with "Arch" and "Cons" grouping columns added and "Model"
        canonicalised to the published architecture names.

    Raises
    ------
    FileNotFoundError
        If `path` does not exist. The caches are vendored under
        `paper/results_cache/` so the manuscript builds from a clean clone.
    ValueError
        If a model in the data is absent from `ARCH_GROUP`. Mapping an unknown
        model would yield NaN and silently drop its rows.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"results cache not found: {path}. Expected it vendored under "
            f"paper/results_cache/; see appendix_operational.qmd for provenance."
        )
    df = pd.read_csv(path)
    for column in ("Model", "Seed", "Consensus"):
        if column not in df.columns:
            raise ValueError(f"{path.name} is missing the '{column}' column")

    unknown = sorted(set(df["Model"].dropna().unique()) - set(ARCH_GROUP))
    if unknown:
        raise ValueError(
            f"{path.name} contains models absent from ARCH_GROUP: {unknown}. "
            f"Add them explicitly rather than letting their rows be dropped."
        )
    df = df.copy()
    df["Arch"] = df["Model"].map(ARCH_GROUP)
    df["Cons"] = df["Consensus"].str.lower().map(CONSENSUS_GROUP)
    df["Model"] = df["Model"].map(DISPLAY_NAMES).fillna(df["Model"])
    return df


def permutation_floor(n_replicates: int) -> float:
    """Smallest two-sided sign-flip p-value attainable with `n_replicates`."""
    if n_replicates <= 0:
        return float("nan")
    if n_replicates > MAX_EXACT_REPLICATES:
        return 0.0
    return 2.0 / 2 ** n_replicates


def exact_sign_permutation_p(diffs: Iterable[float], n_sample: int = 200_000) -> float:
    """
    Two-sided p-value from sign-flip permutation of paired differences.

    Makes no distributional assumption, unlike the paired t-test reported
    alongside it. Enumerates all `2**n` sign assignments where feasible and
    samples otherwise.

    Parameters
    ----------
    diffs : iterable of float
        One within-replicate difference per independent replicate.
    n_sample : int, default=200000
        Monte-Carlo draws used when exact enumeration is infeasible.

    Returns
    -------
    float
        Proportion of sign assignments whose mean absolute difference is at
        least as large as the observed one. NaN for an empty input.
    """
    diffs = np.asarray(list(diffs), dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    n = diffs.size
    if n == 0:
        return float("nan")
    observed = abs(float(diffs.mean()))
    if n <= MAX_EXACT_REPLICATES:
        hits = sum(
            1 for signs in itertools.product((-1.0, 1.0), repeat=n)
            if abs(float(np.mean(np.asarray(signs) * diffs))) >= observed - 1e-15
        )
        return hits / 2 ** n
    rng = np.random.default_rng(0)
    signs = rng.choice((-1.0, 1.0), size=(n_sample, n))
    means = np.abs((signs * diffs).mean(axis=1))
    return float((1 + int(np.sum(means >= observed - 1e-15))) / (1 + n_sample))


def _paired_summary(diffs: np.ndarray, confidence: float = 0.95) -> Dict[str, object]:
    """Effect size, CI, replicate-consistency and both p-values for one cell."""
    n = diffs.size
    mean = float(diffs.mean())
    if n < 2:
        return {
            "mean_diff": mean, "ci_low": float("nan"), "ci_high": float("nan"),
            "n_replicates": n, "n_favouring": int(np.sum(diffs > 0)),
            "p_paired": float("nan"), "p_permutation": float("nan"),
            "floor": permutation_floor(n),
        }
    se = float(diffs.std(ddof=1)) / np.sqrt(n)
    crit = float(stats.t.ppf((1 + confidence) / 2.0, n - 1))
    _, p_paired = stats.ttest_1samp(diffs, 0.0)
    return {
        "mean_diff": mean,
        "ci_low": mean - crit * se,
        "ci_high": mean + crit * se,
        "n_replicates": n,
        "n_favouring": int(np.sum(diffs > 0)),
        "p_paired": float(p_paired),
        "p_permutation": exact_sign_permutation_p(diffs),
        "floor": permutation_floor(n),
    }


def _fmt_p(p: float, floor: float) -> str:
    """Render a p-value, marking those at the design's resolution floor."""
    if not np.isfinite(p):
        return "--"
    if abs(p - floor) < 1e-12:
        return f"{p:.4f} (floor)"
    return f"{p:.4f}"


def deep_vs_linear(df: pd.DataFrame, unit_col: str, metric: str) -> pd.DataFrame:
    """
    Deep versus Linear within each (unit, consensus group), paired by seed.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`load_results`.
    unit_col : str
        "Regime" for the synthetic benchmark, "Dataset" for the clinical one.
    metric : str
        Column to compare.

    Returns
    -------
    pd.DataFrame
        One row per (unit, consensus group), ready to tabulate.
    """
    rows: List[Dict[str, object]] = []
    for unit in df[unit_col].dropna().unique():
        for cons in sorted(df["Cons"].dropna().unique()):
            sub = df[(df[unit_col] == unit) & (df["Cons"] == cons)].dropna(subset=[metric])
            wide = sub.groupby(["Seed", "Arch"])[metric].mean().unstack()
            if not {"Deep", "Linear"}.issubset(wide.columns):
                continue
            wide = wide.dropna()
            if len(wide) < 2:
                continue
            summary = _paired_summary((wide["Deep"] - wide["Linear"]).to_numpy())
            rows.append({
                unit_col: unit,
                "Consensus": cons,
                "Linear mean": round(float(wide["Linear"].mean()), 3),
                "Deep mean": round(float(wide["Deep"].mean()), 3),
                "Difference [95% CI]": (
                    f"{summary['mean_diff']:+.3f} "
                    f"[{summary['ci_low']:+.3f}, {summary['ci_high']:+.3f}]"
                ),
                "Seeds favouring Deep": f"{summary['n_favouring']}/{summary['n_replicates']}",
                "$p$ (paired)": _fmt_p(summary["p_paired"], -1.0),
                "$p$ (permutation)": _fmt_p(summary["p_permutation"], summary["floor"]),
            })
    return pd.DataFrame(rows)


def per_model_vs_baseline(df: pd.DataFrame, unit_col: str, metric: str,
                          baseline: str = "SiMLR",
                          consensus: Optional[str] = None) -> pd.DataFrame:
    """
    Each architecture against `baseline`, paired by seed.

    Reporting every model separately avoids the ambiguity of a pooled
    Deep-versus-Linear contrast, and shows which architecture is actually
    responsible for a group-level effect.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`load_results`.
    unit_col : str
        "Regime" or "Dataset".
    metric : str
        Column to compare.
    baseline : str, default="SiMLR"
        Reference architecture.
    consensus : str, optional
        Restrict to one consensus group.

    Returns
    -------
    pd.DataFrame
        One row per (unit, model).
    """
    rows: List[Dict[str, object]] = []
    frame = df if consensus is None else df[df["Cons"] == consensus]
    for unit in frame[unit_col].dropna().unique():
        sub = frame[frame[unit_col] == unit].dropna(subset=[metric])
        base = sub[sub["Model"] == baseline].groupby("Seed")[metric].mean()
        if base.empty:
            continue
        for model in sub["Model"].unique():
            if model == baseline:
                continue
            got = sub[sub["Model"] == model].groupby("Seed")[metric].mean()
            joined = pd.concat([got, base], axis=1, keys=["model", "base"]).dropna()
            if len(joined) < 2:
                continue
            summary = _paired_summary((joined["model"] - joined["base"]).to_numpy())
            rows.append({
                unit_col: unit,
                "Model": model,
                "Mean": round(float(joined["model"].mean()), 3),
                f"vs {baseline} [95% CI]": (
                    f"{summary['mean_diff']:+.3f} "
                    f"[{summary['ci_low']:+.3f}, {summary['ci_high']:+.3f}]"
                ),
                "Seeds favouring": f"{summary['n_favouring']}/{summary['n_replicates']}",
                "$p$ (permutation)": _fmt_p(summary["p_permutation"], summary["floor"]),
            })
    return pd.DataFrame(rows)


def design_summary(df: pd.DataFrame, unit_col: str) -> Dict[str, object]:
    """Describe the replication structure, so the tables can state it."""
    return {
        "n_seeds": int(df["Seed"].nunique()),
        "seeds": sorted(df["Seed"].unique().tolist()),
        "n_losses": int(df["Loss"].nunique()) if "Loss" in df.columns else None,
        "n_consensus": int(df["Consensus"].nunique()),
        "units": sorted(df[unit_col].dropna().unique().tolist()),
        "models": sorted(df["Model"].dropna().unique().tolist()),
        "n_rows": int(len(df)),
        "permutation_floor": permutation_floor(int(df["Seed"].nunique())),
    }


def environment_report() -> Dict[str, str]:
    """
    Capture the software state the reported numbers depend on.

    Includes the optional NSA-Flow backend, because whether it is installed
    changes which retraction the Stiefel and orthogonality constraints use and
    therefore changes numerical results.

    Returns
    -------
    Dict[str, str]
        Human-readable environment description.
    """
    import platform
    import sys

    report: Dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.machine()}",
    }
    for module_name in ("numpy", "scipy", "pandas", "torch", "sklearn", "matplotlib"):
        try:
            module = __import__(module_name)
            report[module_name] = getattr(module, "__version__", "unknown")
        except ImportError:
            report[module_name] = "not installed"
    try:
        import pysimlr
        report["pysimlr"] = pysimlr.__version__
        from pysimlr.nsa_backend import backend_report
        nsa = backend_report()
        report["nsa_flow backend"] = (
            f"{nsa['module']} ({nsa['entry_point']})" if nsa["available"] else "not installed"
        )
    except Exception as exc:  # pragma: no cover - diagnostic path
        report["pysimlr"] = f"import failed: {exc}"
    try:
        report["git commit"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() or "unknown"
    except Exception:  # pragma: no cover
        report["git commit"] = "unknown"
    return report
