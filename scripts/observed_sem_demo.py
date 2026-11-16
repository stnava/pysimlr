"""Demo of `pysimlr.observed_sem`: compare path models on a planted chain A -> B -> C with a covariate.

Usage:  python scripts/observed_sem_demo.py --smoke
        python scripts/observed_sem_demo.py --n 500 --n-boot 500 --n-jobs 4 --out demo.json
--smoke sets every count to 2 and still runs every stage (fit, comparison, bootstrap with strata,
equivalence, JSON); the JSON carries "smoke": true.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import pysimlr


def planted(n, seed):
    r = np.random.default_rng(seed)
    age = r.standard_normal(n)
    a = 0.3 * age + r.standard_normal(n)
    b = 0.6 * a + 0.2 * age + r.standard_normal(n) * 0.8
    c = 0.5 * b + r.standard_normal(n) * 0.85
    return {"age": age, "A": a, "B": b, "C": c}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--n-draws", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    n, n_boot, n_draws, n_jobs = (20, 2, 2, 2) if args.smoke else (args.n, args.n_boot, args.n_draws, args.n_jobs)
    t0 = time.perf_counter()
    data = planted(n, args.seed)
    strata = np.arange(n) % 2
    cov = [("age", "A"), ("age", "B"), ("age", "C")]
    models = {
        "chain": {"edges": [("A", "B"), ("B", "C")] + cov},
        "reverse": {"edges": [("C", "B"), ("B", "A")] + cov},
        "no_BC": {"edges": [("A", "B")] + cov},
        "saturated": {"edges": [("A", "B"), ("B", "C"), ("A", "C")] + cov},
    }
    cmp = pysimlr.compare_observed_path_models(data, models, exogenous=["age"], n_boot=n_boot, strata=strata,
                                               random_state=args.seed, n_jobs=n_jobs)
    eq = pysimlr.covariance_equivalence(models, exogenous=["age"], n_draws=n_draws, random_state=args.seed)
    for r in cmp["table"]:
        print(f"{r['model']:10s} df={r['df']} chi2={r['chi2']:.2f} rmsea={r['rmsea']:.3f} cfi={r['cfi']:.3f} "
              f"dAIC={r['daic']:.1f} dBIC={r['dbic']:.1f} rejected={r['rejected']}")
    print("best by BIC:", cmp["best_by_bic"], "| equivalence classes:", eq["classes"])
    rows = [{k: v for k, v in r.items()} for r in cmp["table"]]
    out = {"smoke": bool(args.smoke), "n": n, "n_boot": n_boot, "table": rows, "classes": eq["classes"],
           "bootstrap_chain": cmp.get("bootstrap", {}).get("chain"), "seconds": round(time.perf_counter() - t0, 2)}
    if args.out:
        args.out.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({"smoke": out["smoke"], "seconds": out["seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
