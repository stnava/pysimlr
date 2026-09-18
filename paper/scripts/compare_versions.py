#!/usr/bin/env python
"""Compare the conclusions two benchmark versions support.

Re-running a benchmark against a corrected library, or with a different seed
count, can change which claims are supportable. This prints the two versions
side by side so that every change is visible rather than silently absorbed into
a re-render, and so that a claim which *stops* holding cannot slip through.

Reports, per (unit, consensus group) cell and metric: the effect size, the
fraction of seeds favouring the deep architectures, the exact permutation
p-value, and whether the cell separates at alpha = 0.05. The verdict column is
the one to read -- a cell that flips from "yes" to "no" (or back) is a change in
what the manuscript can claim.

Note that the attainable p-value floor is 2/2**n_seeds, so a version with more
seeds can separate a cell that an earlier version could not have separated at
any effect size. Such flips are resolution, not new signal, and are marked.

Usage
-----
    python paper/scripts/compare_versions.py [old_version] [new_version]
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from simlr_paper_stats import (  # noqa: E402
    cache_path, deep_vs_linear, load_results, permutation_floor,
)

ALPHA = 0.05
METRICS = {
    "synthetic": ["Strictly Linear Accuracy", "Predictive Accuracy (Y)", "CMC"],
    "real": ["Strictly Linear Accuracy", "Predictive Accuracy (Y)"],
}
UNIT = {"synthetic": "Regime", "real": "Dataset"}


def _p(value: str) -> float:
    """Parse a p-value cell, which may carry a '(floor)' annotation."""
    return float(str(value).split()[0])


def summarise(version: str, kind: str, metric: str) -> pd.DataFrame:
    df = load_results(cache_path(kind, version))
    table = deep_vs_linear(df, UNIT[kind], metric)
    if table.empty:
        return table
    table = table.rename(columns={UNIT[kind]: "Unit"})
    table["p"] = table["$p$ (permutation)"].map(_p)
    table["separates"] = table["p"] <= ALPHA
    table["n_seeds"] = df["Seed"].nunique()
    return table[["Unit", "Consensus", "Difference [95% CI]",
                  "Seeds favouring Deep", "p", "separates", "n_seeds"]]


def main(old: str, new: str) -> int:
    changes = []
    for kind, metrics in METRICS.items():
        for metric in metrics:
            try:
                a = summarise(old, kind, metric)
                b = summarise(new, kind, metric)
            except FileNotFoundError as exc:
                print(f"skipping {kind}/{metric}: {exc}")
                continue
            merged = a.merge(b, on=["Unit", "Consensus"], suffixes=(f"_{old}", f"_{new}"))
            print(f"\n{'=' * 100}\n{kind.upper()} / {metric}\n{'=' * 100}")
            print(f"{'Unit':>11} {'Consensus':>11} "
                  f"{old + ' diff':>24} {old + ' p':>9} "
                  f"{new + ' diff':>24} {new + ' p':>9}  verdict")
            for _, r in merged.iterrows():
                sa, sb = r[f"separates_{old}"], r[f"separates_{new}"]
                floor_old = permutation_floor(int(r[f"n_seeds_{old}"]))
                if sa == sb:
                    verdict = "unchanged"
                elif sb and floor_old > ALPHA:
                    verdict = "NOW SEPARATES (resolution: old floor > alpha)"
                elif sb:
                    verdict = "NOW SEPARATES"
                else:
                    verdict = "NO LONGER SEPARATES"
                if sa != sb:
                    changes.append((kind, metric, r["Unit"], r["Consensus"], verdict))
                print(f"{r['Unit']:>11} {r['Consensus']:>11} "
                      f"{r[f'Difference [95% CI]_{old}']:>24} {r[f'p_{old}']:>9.4f} "
                      f"{r[f'Difference [95% CI]_{new}']:>24} {r[f'p_{new}']:>9.4f}  {verdict}")

    print(f"\n{'=' * 100}\nCHANGED CELLS ({len(changes)})\n{'=' * 100}")
    if not changes:
        print("  none -- every cell reaches the same verdict in both versions")
    for kind, metric, unit, cons, verdict in changes:
        print(f"  {kind}/{metric}: {unit} / {cons} -> {verdict}")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    sys.exit(main(args[0] if args else "v21", args[1] if len(args) > 1 else "v22"))
