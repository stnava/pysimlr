"""Regenerate the ADNI-MCI rows of the paper's benchmark caches at the current commit (D37).

D37 fixed the Protocol B sex covariate (`PTGENDER` is 0/1 in the export; the old string test made
it a constant 0, so ADNI-MCI was adjusted for age and education only). Only ADNI-MCI rows depend
on it. This script reruns every ADNI-MCI row, every model, with the exact task that produced it
(dataset, case type, method, seed, sparseness target, positivity mode; package-default budgets),
splices the new row in place keeping row order and columns, and writes a provenance record with
the largest change per metric. Rows of the other eleven datasets are untouched.

  python scripts/refresh_adni_cache_rows.py [--workers 8] [--dry-run] [--reuse]
"""
import argparse
import datetime
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..")
sys.path[:0] = [os.path.join(REPO, "src"), HERE]
from refresh_simlr_cache_rows import FILES, key_of, task_of

CACHE = os.path.join(REPO, "paper", "results_cache")
DATASET = "ADNI-MCI"
ROWS_CSV = os.path.join(CACHE, "adni_refresh_rows_D37.csv")
PROV = os.path.join(CACHE, "provenance_adni_refresh_D37.json")


def main():
    import run_deep_benchmark as rdb
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reuse", action="store_true", help="splice the saved refreshed rows instead of refitting")
    a = ap.parse_args()
    frames = {f: pd.read_csv(os.path.join(CACHE, f)) for f in FILES}
    tasks = {}
    for d in frames.values():
        for _, r in d[d.Dataset == DATASET].iterrows():
            t = task_of(r)
            tasks[key_of(t)] = t
    print(f"{len(tasks)} unique {DATASET} tasks across {len(FILES)} files")
    if a.dry_run:
        return
    if a.reuse:
        if not os.path.exists(ROWS_CSV):
            raise FileNotFoundError(f"--reuse given but {ROWS_CSV} does not exist")
    else:
        new = rdb._execute_tasks(list(tasks.values()), workers=a.workers, label=f"{DATASET} cache refresh")
        new.to_csv(ROWS_CSV, index=False)
    new = pd.read_csv(ROWS_CSV)              # the cached CSV's own text representation (lists, dicts)
    new["_key"] = [key_of(task_of(r)) for _, r in new.iterrows()]
    if new._key.duplicated().any() or len(new) != len(tasks):
        raise RuntimeError("refreshed rows do not map one-to-one onto the tasks")
    by_key = new.set_index("_key")
    changes = {}
    for f, d in frames.items():
        d = d.copy()
        n_rep = 0
        for i, r in d[d.Dataset == DATASET].iterrows():
            nr = by_key.loc[[key_of(task_of(r))]].iloc[0]
            for c in d.columns:
                if c not in nr.index:
                    continue
                old, val = d.at[i, c], nr[c]
                if isinstance(old, (int, float, np.floating)) and isinstance(val, (int, float, np.floating)) \
                        and np.isfinite(old) and np.isfinite(val) and c != "Fit_Seconds":
                    cs = changes.setdefault(f, {})
                    cs[c] = max(cs.get(c, 0.0), abs(float(val) - float(old)))
                if pd.isna(val) and not pd.isna(old):
                    raise RuntimeError(f"{f} row {i} column {c}: refreshed value is missing")
                d.at[i, c] = val
            n_rep += 1
        d.to_csv(os.path.join(CACHE, f), index=False)
        print(f"{f}: replaced {n_rep} {DATASET} rows")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                            capture_output=True, text=True, check=True).stdout.strip()
    import pysimlr
    prov = {"cli": "scripts/refresh_adni_cache_rows.py", "git_commit": commit, "pysimlr": pysimlr.__version__,
            "written_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "reason": "D37: Protocol B sex covariate built from 0/1 PTGENDER (was constant 0)",
            "n_tasks": len(tasks), "files": FILES,
            "max_abs_change": {f: {c: v for c, v in cs.items() if v > 0} for f, cs in changes.items()}}
    with open(PROV, "w") as fh:
        json.dump(prov, fh, indent=1)
    print(json.dumps(prov["max_abs_change"], indent=1))


if __name__ == "__main__":
    main()
