"""Demo of `pysimlr.bagged_fit` on a planted sparse component (both modes by default, or one via --aggregate).

Usage:  python scripts/bagged_fit_demo.py --smoke
        python scripts/bagged_fit_demo.py --n-subsamples 50 --n 120 --method simlr
--smoke sets every count to 2 and still runs every stage (fits, selection, prediction, out-of-bag, between-run
agreement, JSON); the JSON carries "smoke": true.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import pysimlr

N_PLANTED, P = 5, 30


def corr(a, b):
    return float(np.corrcoef(np.asarray(a).ravel(), np.asarray(b).ravel())[0, 1])


def planted(n, seed, n_views=3, p=P):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 1))
    views = [np.hstack([z @ rng.uniform(0.8, 1.2, (1, N_PLANTED)) + 0.8 * rng.standard_normal((n, N_PLANTED)),
                        rng.standard_normal((n, p - N_PLANTED))]) for _ in range(n_views)]
    return z[:, 0], views


def run_mode(method, aggregate, z, views, z_new, views_new, n_sub, seed, extra):
    kw = dict(n_subsamples=n_sub, random_state=seed, **extra)
    a = pysimlr.bagged_fit(method, views, 1, aggregate=aggregate, **kw)
    b = pysimlr.bagged_fit(method, views, 1, aggregate=aggregate, draw_random_state=seed + 1000, **kw)
    pred_a = pysimlr.predict(a, views_new)["u"].numpy()[:, 0]
    pred_b = pysimlr.predict(b, views_new)["u"].numpy()[:, 0]
    oob = a["oob_scores"][:, 0]
    ok = np.isfinite(oob)
    rec = {
        "aggregate": aggregate, "method": method,
        "corr_with_latent_train": abs(corr(a["scores"], z)),
        "corr_with_latent_new_rows": abs(corr(pred_a, z_new)),
        "kept_features": [int(m.sum()) for m in a["feature_mask"]],
        "kept_planted": [int(m[:N_PLANTED].sum()) for m in a["feature_mask"]],
        "planted_total": N_PLANTED,
        "effective_features": a["effective_features"],
        "between_run_agreement_train": abs(corr(a["scores"], b["scores"])),
        "between_run_agreement_new_rows": abs(corr(pred_a, pred_b)),
        "oob_rows_available": int(ok.sum()),
        "oob_corr_with_latent": abs(corr(oob[ok], z[ok])) if ok.sum() > 2 else None,
        "in_bag_corr_with_latent": abs(corr(np.asarray(a["scores"])[ok, 0], z[ok])) if ok.sum() > 2 else None,
        "component_stability": a["component_stability"],
        "converged_fraction": a["converged_fraction"],
    }
    if ok.sum() > 2:
        rec["oob_vs_inbag_score_corr"] = abs(corr(oob[ok], np.asarray(a["scores"])[ok, 0]))
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-subsamples", type=int, default=50)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--method", default="simlr", choices=pysimlr.method_names())
    ap.add_argument("--aggregate", default=None, choices=["select", "average"],
                    help="one mode (bagged_fit's own default is select); omitted: run both")
    ap.add_argument("--out", default=None, help="JSON path (default: bagged_fit_demo[_smoke].json in the cwd)")
    ap.add_argument("--smoke", action="store_true", help="counts of 2, every stage still runs")
    args = ap.parse_args(argv)
    n_sub, n = (2, 2 * (N_PLANTED + 10)) if args.smoke else (args.n_subsamples, args.n)
    if args.smoke:
        n = 30                      # smallest n that the 3-view fit accepts comfortably; subsamples stay at 2
    extra = {"iterations": 2} if args.smoke and args.method not in ("pca", "nsa_pipeline") else {}
    modes = [args.aggregate] if args.aggregate else ["select", "average"]
    z, views = planted(n, args.seed)
    z_new, views_new = planted(max(n // 2, 20), args.seed + 1)
    t0 = time.time()
    rows = [run_mode(args.method, m, z, views, z_new, views_new, n_sub, args.seed, extra) for m in modes]
    out = {"smoke": bool(args.smoke), "n": n, "n_subsamples": n_sub, "seed": args.seed, "method": args.method,
           "seconds": round(time.time() - t0, 1), "results": rows}
    path = Path(args.out or ("bagged_fit_demo_smoke.json" if args.smoke else "bagged_fit_demo.json"))
    path.write_text(json.dumps(out, indent=2, default=float))
    for r in rows:
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                          if k not in ("component_stability",)}, default=float))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
