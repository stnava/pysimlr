"""ADNI-MCI relevance test through `pysimlr.feature_relevance` (docs/audit/evidence/relevance/*.csv).

Where does the predictive component's explained variance concentrate? Protocol B (seed 42),
the Figure 3.9 SiMLR fit; the predictive component is the one most correlated with training
CogPC1. Tested views: volume / surface area and DiReCT thickness; cognition is missing at test
time, so the targets are built from MRI alone in training, held-out data and refits alike.
Groups: bilateral pairs and DKT lobes. See `pysimlr.relevance` for the tests and the tiers.

  python scripts/adni_relevance_test.py                  # ~25 s: 10 refits, no refit null
  python scripts/adni_relevance_test.py --full-null 300  # adds p_null_refit (600 more fits)
  python scripts/adni_relevance_test.py --figure         # also docs/figures/adni_volarea_relevance.png
"""
import os, re, sys, time, numpy as np, pandas as pd, torch
W = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path[:0] = [W + "/src", W + "/scripts"]
OUT = os.path.join(W, "docs", "audit", "evidence", "relevance")
VIEW_NAMES = ["demographics", "vol_area", "direct_thickness", "cognition"]


def base_name(n):
    n = re.sub(r"^(left|right)_", "", n)
    return re.sub(r"_(l|r)dktlobes$", "_dktlobes", n)


def bilateral(n):
    """Pair label for a left/right feature (None otherwise)."""
    return base_name(n) if (n.startswith(("left_", "right_")) or re.search(r"_(l|r)dktlobes$", n)) else None


def lobe(n):
    """DKT lobe of a cortical feature (None otherwise)."""
    from antsxviz import DKT_LOBES
    b = re.sub(r"dkt(cortex|regions)$", "", base_name(n)).strip("_")
    return next((lb for lb, regs in DKT_LOBES.items() if b in regs), None)


def run(refits=10, full_null=0, b_train=100_000, b_test=50_000, workers=8):
    """(rel, data): the feature_relevance result for views 1 and 2 and the protocol-B data."""
    import pysimlr, paper_adni_ablations as paa
    torch.set_num_threads(1)
    d = paa.protocol_b(42)
    res, _ = paa.fit_adni(d, "simlr", k=3, positivity="positive", sparseness=0.85, random_state=42)
    names = [list(d["names"][0]), list(d["names"][1]), list(d["names"][2]),
             [f"cog{i}" for i in range(d["y_tr"].shape[1])]]
    rel = pysimlr.feature_relevance(
        res, paa.fit_views(d), [*d["x_te"][:3], None], outcome=d["pc1_tr"], outcome_test=d["pc1_te"],
        views=(1, 2), feature_names=names, view_names=VIEW_NAMES, groups={"bilateral": bilateral, "lobe": lobe},
        n_permutations=b_train, n_permutations_test=b_test, n_refits=refits, n_null_refits=full_null,
        n_jobs=workers)
    return rel, d


def main():
    import argparse
    ap = argparse.ArgumentParser(description="ADNI relevance test (see module docstring)")
    ap.add_argument("--refits", type=int, default=10, help="half-sample refits for stability (0: skip)")
    ap.add_argument("--full-null", type=int, default=0,
                    help="whole-model refits per view with that view's rows shuffled (0: skip)")
    ap.add_argument("--b-train", type=int, default=100_000)
    ap.add_argument("--b-test", type=int, default=50_000)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--figure", action="store_true", help="also draw docs/figures/adni_volarea_relevance.png")
    a = ap.parse_args()
    t0 = time.time()
    rel, d = run(a.refits, a.full_null, a.b_train, a.b_test, a.workers)
    print(f"feature_relevance (incl. the fit): {time.time() - t0:.1f} s", flush=True)
    os.makedirs(a.out, exist_ok=True)
    rel["features"].to_csv(a.out + "/features.csv", index=False)
    rel["groups"].to_csv(a.out + "/groups.csv", index=False)
    if a.figure:
        from adni_relevance_figure import draw
        draw(rel, d)
        print(f"figure done: {time.time() - t0:.1f} s total", flush=True)
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
    print(f"\ncomponent {rel['component'] + 1}; train n={len(d['pc1_tr'])}, test n={len(d['pc1_te'])}")
    print(rel["summary"].to_string(index=False))
    for m in (1, 2):
        f, g = rel["features"], rel["groups"]
        f, g = f[f.view == m], g[g.view == m]
        print(f"\n===== {VIEW_NAMES[m]}: replicated features =====")
        print(f[f.tier == 2].sort_values("share_train", ascending=False).round(4).to_string(index=False))
        print("lobes:")
        print(g[g.grouping == "lobe"].sort_values("p_train").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
