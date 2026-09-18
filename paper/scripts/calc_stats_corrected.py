#!/usr/bin/env python
"""Seed-level statistics for the benchmark tables, replacing the pooled t-test.

`calc_stats.py` computes ``scipy.stats.ttest_ind`` over every row of the
results cache. In the v21 real-data cache each (Dataset, Consensus-group,
Architecture-group) cell contains 80 rows but only **5 distinct seeds** -- the
other 16x comes from crossing 4 loss functions, 2 architectures and 2 consensus
variants, all evaluated on the *same* data splits. Those rows are repeated
measures, not independent samples, so a two-sample t-test understates the
standard error and the p-value shrinks with compute rather than with evidence.

This script reports, per comparison:

  * the pooled p-value, so the two can be compared directly;
  * a seed-level paired t-test on within-seed means;
  * an **exact** two-sided sign-flip permutation p-value over the seeds, which
    makes no distributional assumption.

The last column is the important one. With n seeds the smallest attainable
two-sided permutation p-value is ``2 / 2**n``, which is 0.0625 for n = 5. No
p-value below that floor is supportable from a 5-seed design however it is
computed, so the manuscript's 8e-11 and 3e-4 reflect parametric assumptions
about the seed-level differences rather than the resolution of the experiment.

It also reports each model against the SiMLR baseline individually, because
`calc_stats.py` assigns LEND to the *Linear* group. That both dilutes the
"Deep" contrast with a deep model on the wrong side and means the
manuscript's claim that "LEND achieved ... significance over the linear
baseline" cannot have come from that test.

Usage
-----
    python paper/scripts/calc_stats_corrected.py [results_cache_dir]
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_CACHE = Path(
    "/Users/stnava/Library/Mobile Documents/com~apple~CloudDocs/code/pysimlr/"
    "paper/results_cache"
)

#: LEND is a Linear Encoder with a Nonlinear Decoder, so it belongs in the Deep
#: group; `calc_stats.py` had it as Linear. Flow-SiMLR-V was absent from that
#: map entirely, which made pandas .map() drop its rows silently.
ARCH_GROUP = {
    "SiMLR": "Linear",
    "LEND": "Deep",
    "NED": "Deep",
    "NEDPP": "Deep",
    "Flow-SiMLR-V": "Deep",
}
CONSENSUS_GROUP = {
    "newton": "Newton/ICA", "ica": "Newton/ICA",
    "svd": "SVD/PCA", "pca": "SVD/PCA",
}
MAX_EXACT_SEEDS = 22  # 2**22 sign patterns; beyond this, sample instead


def exact_sign_permutation_p(diffs: np.ndarray, n_sample: int = 200_000) -> float:
    """
    Two-sided p-value from sign-flip permutation of paired differences.

    Parameters
    ----------
    diffs : np.ndarray
        Within-seed differences (one per independent replicate).
    n_sample : int, default=200000
        Monte-Carlo draws used when enumerating all sign patterns is infeasible.

    Returns
    -------
    float
        The proportion of sign assignments whose mean absolute difference is at
        least as large as observed. Bounded below by ``2 / 2**n`` for exact
        enumeration, which is the resolution floor of the design.
    """
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    n = diffs.size
    if n == 0:
        return float("nan")
    observed = abs(diffs.mean())
    if n <= MAX_EXACT_SEEDS:
        hits = sum(
            1 for signs in itertools.product((-1.0, 1.0), repeat=n)
            if abs(np.mean(np.asarray(signs) * diffs)) >= observed - 1e-15
        )
        return hits / 2 ** n
    rng = np.random.default_rng(0)
    signs = rng.choice((-1.0, 1.0), size=(n_sample, n))
    means = np.abs((signs * diffs).mean(axis=1))
    return float((1 + np.sum(means >= observed - 1e-15)) / (1 + n_sample))


def permutation_floor(n_seeds: int) -> float:
    """Smallest two-sided sign-flip p-value attainable with `n_seeds` replicates."""
    return 2.0 / 2 ** n_seeds if n_seeds <= MAX_EXACT_SEEDS else 0.0


def _paired_frame(sub: pd.DataFrame, metric: str, by: str) -> pd.DataFrame:
    """Collapse repeated measures to one value per (seed, level of `by`)."""
    return sub.groupby(["Seed", by])[metric].mean().unstack()


def group_comparison(df: pd.DataFrame, unit_col: str, metric: str) -> pd.DataFrame:
    """Deep vs Linear within each (unit, consensus group), at the seed level."""
    unmapped = sorted(set(df["Model"].dropna().unique()) - set(ARCH_GROUP))
    if unmapped:
        raise ValueError(
            f"Models absent from ARCH_GROUP: {unmapped}. Their rows would be "
            f"dropped from the comparison without notice."
        )
    df = df[df["Model"].isin(ARCH_GROUP)].copy()
    df["Arch"] = df["Model"].map(ARCH_GROUP)
    df["Cons"] = df["Consensus"].str.lower().map(CONSENSUS_GROUP)

    rows = []
    for unit in df[unit_col].unique():
        for cons in sorted(df["Cons"].dropna().unique()):
            sub = df[(df[unit_col] == unit) & (df["Cons"] == cons)].dropna(subset=[metric])
            if sub.empty:
                continue
            deep = sub[sub["Arch"] == "Deep"][metric].values
            lin = sub[sub["Arch"] == "Linear"][metric].values
            if len(deep) < 2 or len(lin) < 2:
                continue
            _, p_pooled = stats.ttest_ind(deep, lin, equal_var=False)

            wide = _paired_frame(sub, metric, "Arch")
            if not {"Deep", "Linear"}.issubset(wide.columns):
                continue
            wide = wide.dropna()
            diffs = (wide["Deep"] - wide["Linear"]).values
            n_seeds = len(diffs)
            if n_seeds < 2:
                continue
            _, p_paired = stats.ttest_rel(wide["Deep"].values, wide["Linear"].values)
            se = diffs.std(ddof=1) / np.sqrt(n_seeds)
            crit = stats.t.ppf(0.975, n_seeds - 1)
            rows.append({
                unit_col: unit,
                "Consensus": cons,
                "n rows (pooled)": len(deep) + len(lin),
                "n seeds": n_seeds,
                "mean diff (Deep-Lin)": round(float(diffs.mean()), 4),
                "95% CI": f"[{diffs.mean() - crit * se:.4f}, {diffs.mean() + crit * se:.4f}]",
                "seeds favouring Deep": f"{int((diffs > 0).sum())}/{n_seeds}",
                "p pooled (as published)": f"{p_pooled:.2e}",
                "p seed-level paired": f"{p_paired:.4f}",
                "p exact permutation": f"{exact_sign_permutation_p(diffs):.4f}",
                "permutation floor": f"{permutation_floor(n_seeds):.4f}",
            })
    return pd.DataFrame(rows)


def per_model_vs_baseline(df: pd.DataFrame, unit_col: str, metric: str,
                          baseline: str = "SiMLR") -> pd.DataFrame:
    """Each model against `baseline`, paired within seed."""
    rows = []
    for unit in df[unit_col].unique():
        sub = df[df[unit_col] == unit]
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
            diffs = (joined["model"] - joined["base"]).values
            n_seeds = len(diffs)
            _, p_paired = stats.ttest_rel(joined["model"].values, joined["base"].values)
            se = diffs.std(ddof=1) / np.sqrt(n_seeds)
            crit = stats.t.ppf(0.975, n_seeds - 1)
            rows.append({
                unit_col: unit,
                "Model": model,
                "mean": round(float(joined["model"].mean()), 4),
                f"vs {baseline}": round(float(diffs.mean()), 4),
                "95% CI": f"[{diffs.mean() - crit * se:.4f}, {diffs.mean() + crit * se:.4f}]",
                "seeds favouring model": f"{int((diffs > 0).sum())}/{n_seeds}",
                "p seed-level paired": f"{p_paired:.4f}",
                "p exact permutation": f"{exact_sign_permutation_p(diffs):.4f}",
            })
    return pd.DataFrame(rows)


def main(cache_dir: Path) -> int:
    real = cache_dir / "unified_real_v21.csv"
    synth = cache_dir / "unified_synthetic_v21.csv"
    if not real.exists():
        print(f"results cache not found: {real}", file=sys.stderr)
        return 1

    real_df = pd.read_csv(real)
    print("## Real data: Deep vs Linear, corrected inference\n")
    print(group_comparison(real_df, "Dataset", "Predictive Accuracy (Y)").to_markdown(index=False))
    print("\n## Real data: each model against the SiMLR baseline\n")
    print(per_model_vs_baseline(real_df, "Dataset", "Predictive Accuracy (Y)").to_markdown(index=False))

    if synth.exists():
        synth_df = pd.read_csv(synth)
        print("\n## Synthetic: Deep vs Linear, corrected inference\n")
        print(group_comparison(synth_df, "Regime", "Predictive Accuracy (Y)").to_markdown(index=False))
        print("\n## Synthetic: CMC\n")
        print(group_comparison(synth_df, "Regime", "CMC").to_markdown(index=False))

    n_seeds = real_df["Seed"].nunique()
    print(f"\n> The design has {n_seeds} seeds. The smallest attainable two-sided")
    print(f"> sign-flip permutation p-value is 2/2**{n_seeds} = "
          f"{permutation_floor(n_seeds):.4f}. Any p-value below that reflects a")
    print("> parametric assumption about the seed-level differences, not the")
    print("> resolution of the experiment. Report effect sizes with confidence")
    print("> intervals and the seed-level consistency count, and raise the seed")
    print("> count if smaller p-values are wanted.")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CACHE))
