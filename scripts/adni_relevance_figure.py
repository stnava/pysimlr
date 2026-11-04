"""ADNI-MCI volume / surface area relevance figure (docs/figures/adni_volarea_relevance.png).

A thin caller: `pysimlr.feature_relevance` (via adni_relevance_test.run) and
`pysimlr.plotting.plot_feature_relevance`. The ADNI-specific parts are the region labels, the
whole-brain totals panel, and the DiReCT counts stated in the subtitle.
"""
import os, sys, time
W = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path[:0] = [W + "/src", W + "/scripts", W + "/paper"]
import matplotlib; matplotlib.use("Agg")
OUT = os.path.join(W, "docs", "figures", "adni_volarea_relevance.png")


def pretty(n):
    from paper_numbers import region_label
    return region_label(n).replace(" (DKT cortex label)", "").replace(" (DKT parcellation)", " (parc.)")


def draw(rel, d, out=OUT):
    from pysimlr.plotting import plot_feature_relevance
    s = rel["summary"].set_index("view")
    a = rel["alpha"]
    dt, dg = s.loc[2], rel["groups"]
    dlobes = int(((dg.view == 2) & (dg.grouping == "lobe") & (dg.p_train < a)).sum())
    vs = s.loc[1]
    sub = (f"Share of the variance this view explains in component {rel['component'] + 1} (relative weights). "
           f"Fair share = 1/{int(vs.n_features)}. Rule (set before plotting): dark = permutation p < {a:g} in "
           f"training AND in {len(d['pc1_te'])} held-out subjects; pale = training only; grey = neither.\n"
           f"Same test on DiReCT cortical thickness: {int(dt.n_train)} of {int(dt.n_features)} regions at "
           f"p < {a:g} in training, {int(dt.n_test)} held-out (chance {dt.expected_by_chance:.1f}), "
           f"{dlobes} lobe{'s' if dlobes != 1 else ''} significant in training. "
           "L/R verified: per-subject ADNI DKT labels are correct (antsxviz label_hemisphere_report).\n"
           "Share and r differ: a large share with a weak r (CSF) is a non-redundant slice of the component, not a "
           "strong marker; relative weights split variance among correlated features.")
    fig = plot_feature_relevance(
        rel, 1, label=pretty, separate=lambda n: n.endswith("tissues"),
        separate_title="Whole-brain tissue totals (not regions)", grouping="lobe", grouping_title="DKT lobes", brain=True,
        r_label="r with CogPC1\n(held-out)", figsize=(15.5, 11.6),
        title=f"ADNI-MCI volume / surface area, predictive component {rel['component'] + 1}: "
              "where its variance concentrates", subtitle=sub)
    fig.savefig(out, dpi=160)
    print("wrote", out)
    return fig


def main():
    from adni_relevance_test import run
    t0 = time.time()
    rel, d = run()
    draw(rel, d)
    print(f"{time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
