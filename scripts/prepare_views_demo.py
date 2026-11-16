"""Demo of `pysimlr.prepare_views` (order effect) and `pysimlr.multistart_fit` (several starts) on synthetic data.

Usage:  python scripts/prepare_views_demo.py --smoke
        python scripts/prepare_views_demo.py --n 150 --n-starts 10 --iters 200
--smoke sets every count to 2 and still runs every stage (preparation with and without the second
standardization, apply_prepared on new rows, a fit on the prepared views, the multi-start fit, JSON); the JSON
carries "smoke": true.
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

import pysimlr


def make_views(n, seed):
    """View 'a' loses most of its variance to the nuisance; view 'b' does not; one shared signal in both."""
    rng = np.random.default_rng(seed)
    c = rng.standard_normal((n, 2))
    z = rng.standard_normal((n, 1))
    a = 3.0 * (c @ rng.standard_normal((2, 12))) + z @ rng.standard_normal((1, 12)) + 0.5 * rng.standard_normal((n, 12))
    b = z @ rng.standard_normal((1, 8)) + rng.standard_normal((n, 8))
    return z[:, 0], {"a": a, "b": b}, {"a": c}


def two_solution_views(n, seed, s2=0.9):
    rng = np.random.default_rng(seed)
    za, zb = rng.standard_normal((n, 1)), rng.standard_normal((n, 1))
    out = []
    for _ in range(3):
        a = np.zeros((1, 10))
        a[0, :5] = 1
        b = np.zeros((1, 10))
        b[0, 5:] = s2
        out.append(za @ a + zb @ b + 0.5 * rng.standard_normal((n, 10)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--n-starts", type=int, default=10)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="JSON path (default: prepare_views_demo[_smoke].json in the cwd)")
    ap.add_argument("--smoke", action="store_true", help="counts of 2, every stage still runs")
    args = ap.parse_args(argv)
    n_starts, iters, n_jobs = (2, 2, 2) if args.smoke else (args.n_starts, args.iters, args.n_jobs)
    n = 40 if args.smoke else args.n
    warnings.filterwarnings("ignore", message=".*did not converge.*")
    t0 = time.time()

    _, views, nuis = make_views(n, args.seed)
    fit_rows = np.arange(n) < int(0.7 * n)
    on = pysimlr.prepare_views(views, fit_rows=fit_rows, nuisance=nuis)
    off = pysimlr.prepare_views(views, fit_rows=fit_rows, nuisance=nuis, restandardize=False)
    new = pysimlr.apply_prepared(on["params"], {k: v[~fit_rows] for k, v in views.items()})
    res = pysimlr.fit("simlr", on["views"], 1, scale_list=["none"], iterations=iters)
    order = {
        "variance_removed": on["variance_removed"],
        "median_column_sd_view_a": {"standardized_again": float(np.median(np.nanstd(on["views"]["a"][fit_rows], axis=0))),
                                    "not_standardized_again": float(np.median(np.nanstd(off["views"]["a"][fit_rows], axis=0)))},
        "apply_prepared_new_rows_shape": list(new["a"].shape),
        "fit_best_energy": float(res["best_energy"]),
        "record": on["record"],
    }

    ms = pysimlr.multistart_fit("simlr", two_solution_views(n, args.seed), 1, n_starts=n_starts, iterations=iters,
                                random_state=args.seed, n_jobs=n_jobs)
    multi = {"chosen": ms["chosen"], "n_distinct_solutions": ms["n_distinct_solutions"], "energy_gap": ms["energy_gap"],
             "starts": ms["starts"]}
    out = {"smoke": bool(args.smoke), "n": n, "n_starts": n_starts, "iters": iters, "n_jobs": n_jobs, "seed": args.seed,
           "seconds": round(time.time() - t0, 1), "order_effect": order, "multistart": multi}
    path = Path(args.out or ("prepare_views_demo_smoke.json" if args.smoke else "prepare_views_demo.json"))
    path.write_text(json.dumps(out, indent=2, default=float))
    sd = order["median_column_sd_view_a"]
    print(f"nuisance removed {on['variance_removed']['a']:.2f} of view a; its median column SD: "
          f"{sd['standardized_again']:.2f} with the second standardization, {sd['not_standardized_again']:.2f} without")
    print(f"multistart: chosen {ms['chosen']}, {ms['n_distinct_solutions']} distinct solution(s), energy_gap {ms['energy_gap']}")
    for s in ms["starts"]:
        print(f"  {s['label']:>9} energy {s['energy']:.4f} |r| with chosen {s['score_correlation_with_chosen']:.2f}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
