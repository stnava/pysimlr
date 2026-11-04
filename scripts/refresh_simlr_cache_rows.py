"""Regenerate the linear-SiMLR rows of the paper's benchmark caches at the current commit.

D26 changed the scale of linear-SiMLR `latents`. That moves only the metrics that fit a penalised
estimator to them; every other model is bit-identical. This script reruns each SiMLR row with the
exact task that produced it (dataset, case type, method, seed, sparseness target, positivity mode;
package-default budgets). It splices the new row in place, keeping row order and columns, and writes
a provenance record with the largest change per metric.

  python scripts/refresh_simlr_cache_rows.py [--workers 8] [--dry-run]
"""
import argparse, datetime, json, os, subprocess, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.join(HERE, "..")
sys.path[:0] = [os.path.join(REPO, "src"), HERE]
CACHE = os.path.join(REPO, "paper", "results_cache")
FILES = ["deep_ranking_benchmark.csv", "deep_ranking_benchmark_q05.csv", "deep_ranking_benchmark_q_none.csv",
         "deep_ranking_benchmark_q08.csv", "operating_points_05_08_benchmark.csv"]
REAL_BIO = {"ADNI-MCI", "MultiOmics", "TCGA-KIRC"}
KEY = ["Dataset", "Method", "Seed", "Sparseness_Target", "Positivity"]


def task_of(r):
    sp = None if pd.isna(r.Sparseness_Target) else float(r.Sparseness_Target)
    mode = "positive" if (r.Positivity == "positive" and r.Dataset in REAL_BIO) else "default"
    return (r.Dataset, r.Case_Type, r.Method, r.Model, int(r.Seed), None, None, sp, mode)


def key_of(t):
    return (t[0], t[2], t[4], -1.0 if t[7] is None else t[7], t[8])


def main():
    import run_deep_benchmark as rdb
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reuse", action="store_true", help="splice the saved refreshed rows instead of refitting")
    a = ap.parse_args()
    frames = {f: pd.read_csv(os.path.join(CACHE, f)) for f in FILES}
    tasks = {}
    for f, d in frames.items():
        for _, r in d[d.Method == "simlr"].iterrows():
            t = task_of(r); tasks[key_of(t)] = t
    print(f"{len(tasks)} unique SiMLR tasks across {len(FILES)} files")
    if a.dry_run:
        return
    rows_csv = os.path.join(CACHE, "simlr_refresh_rows_D26.csv")
    if os.path.exists(rows_csv) and a.reuse:
        new = pd.read_csv(rows_csv)
    else:
        new = rdb._execute_tasks(list(tasks.values()), workers=a.workers, label="SiMLR cache refresh")
        new.to_csv(rows_csv, index=False)
        new = pd.read_csv(rows_csv)          # the cached CSV's own text representation (lists, dicts)
    new["_key"] = [key_of(task_of(r)) for _, r in new.iterrows()]
    if new._key.duplicated().any() or len(new) != len(tasks):
        raise RuntimeError("refreshed rows do not map one-to-one onto the tasks")
    by_key = new.set_index("_key")
    changes = {}
    for f, d in frames.items():
        d = d.copy(); n_rep = 0
        for i, r in d[d.Method == "simlr"].iterrows():
            nr = by_key.loc[[key_of(task_of(r))]].iloc[0]
            for c in d.columns:
                if c in nr.index:
                    old, val = d.at[i, c], nr[c]
                    if isinstance(old, (int, float, np.floating)) and isinstance(val, (int, float, np.floating)) \
                            and np.isfinite(old) and np.isfinite(val):
                        changes.setdefault(f, {}).setdefault(c, 0.0)
                        if c not in ("Fit_Seconds",):
                            changes[f][c] = max(changes[f][c], abs(float(val) - float(old)))
                    if pd.isna(val) and not pd.isna(old):
                        raise RuntimeError(f"{f} row {i} column {c}: refreshed value is missing")
                    d.at[i, c] = val
            n_rep += 1
        d.to_csv(os.path.join(CACHE, f), index=False)
        print(f"{f}: replaced {n_rep} SiMLR rows")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    import pysimlr
    prov = {"cli": "scripts/refresh_simlr_cache_rows.py", "git_commit": commit, "pysimlr": pysimlr.__version__,
            "written_utc": datetime.datetime.utcnow().isoformat() + "Z", "reason": "D26: linear-SiMLR latents at the scale of u",
            "n_tasks": len(tasks), "files": FILES,
            "max_abs_change": {f: {c: v for c, v in cs.items() if v > 0} for f, cs in changes.items()}}
    with open(os.path.join(CACHE, "provenance_simlr_refresh_D26.json"), "w") as fh:
        json.dump(prov, fh, indent=1)
    print(json.dumps(prov["max_abs_change"], indent=1))


if __name__ == "__main__":
    main()
