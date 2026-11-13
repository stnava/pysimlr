"""Order dependence of simlr (5 weakly coupled views + root, k=1) under greedy and spectral sign alignment."""
import sys, argparse, json, warnings, numpy as np, pysimlr
sys.path.insert(0, 'tests')
from test_view_order_sign_alignment import _noise_views, _fit, _udiff
ap = argparse.ArgumentParser()
ap.add_argument("--n-seeds", type=int, default=8)
ap.add_argument("--iters", type=int, default=None, help="default: pysimlr default number of iterations")
ap.add_argument("--signal", type=float, default=0.0, help="shared-signal strength added to the five brain views")
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--out", default=None)
args = ap.parse_args()
if args.smoke: args.n_seeds = 2; args.iters = 2
from pysimlr.defaults import DEFAULT_SIMLR_ITERATIONS
full = args.iters or DEFAULT_SIMLR_ITERATIONS
o1 = [f"V{i}" for i in range(5)] + ["Root"]; o2 = ["V0", "V1", "V2", "V4", "V3", "Root"]; o3 = ["Root", "V4", "V3", "V2", "V1", "V0"]
rows = []
for seed in range(3, 3 + args.n_seeds):
    views = _noise_views(seed)
    if args.signal:
        rng = np.random.default_rng(seed + 999); z = rng.standard_normal(views["Root"].shape[0])
        for nm in o1[:5]: views[nm] = views[nm] + args.signal * np.outer(z, rng.standard_normal(views[nm].shape[1]))
    for sa in ("greedy", "spectral"):
        for its, label in ((1, "first sweep"), (full, f"full ({full})")):
            d12 = _udiff(_fit(views, o1, its, sa), _fit(views, o2, its, sa))
            d13 = _udiff(_fit(views, o1, its, sa), _fit(views, o3, its, sa))
            rows.append(dict(seed=seed, sign_alignment=sa, stage=label, d_swap_V3V4=float(d12), d_reversed=float(d13)))
            print(sa, label, "seed", seed, "swap %.3e reversed %.3e" % (d12, d13), flush=True)
import collections
agg = collections.defaultdict(list)
for r in rows: agg[(r["sign_alignment"], r["stage"])].append(r)
summary = {f"{k[0]} / {k[1]}": dict(max_swap=max(x["d_swap_V3V4"] for x in v), median_swap=float(np.median([x["d_swap_V3V4"] for x in v])),
           max_reversed=max(x["d_reversed"] for x in v), n_over_0_01=sum(x["d_swap_V3V4"] > 0.01 or x["d_reversed"] > 0.01 for x in v), n=len(v)) for k, v in agg.items()}
print(json.dumps(dict(smoke=args.smoke, signal=args.signal, summary=summary), indent=1))
if args.out: json.dump(dict(smoke=args.smoke, rows=rows, summary=summary), open(args.out, "w"), indent=1)
