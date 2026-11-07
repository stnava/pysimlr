"""Reference demo of the score-level path model on a planted chain, with an HTML report.

Usage: python scripts/path_model_demo.py [--out docs/audit/evidence/path_model] [--method simlr]

Generates views 0 - 1 - 2 from a latent chain (L_a and L_c independent given L_b), an outcome from
view 2's first latent, and two nuisance covariates that leak into every view. Fits the method on
70% of rows, runs `pysimlr.sem.path_model` on the held-out 30%, compares four graphs, draws the four
figures of plan section 9 (reference versions; the executor moves them to `plotting.py`) and
writes a single self-contained HTML report with the tables and the figures embedded.
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import pysimlr  # noqa: E402
from pysimlr import plotting as P  # noqa: E402
from pysimlr import sem  # noqa: E402
from pysimlr.benchmarks.synthetic_cases import build_path_graph_case  # noqa: E402


# ----------------------------------------------------------------------------- data
def chain_case(n=500, k=3, p=(20, 25, 18), noise=0.5, cov_effect=0.8, seed=0):
    """Planted chain view0 - view1 - view2 (`benchmarks.synthetic_cases.build_path_graph_case`)."""
    case = build_path_graph_case([(0, 1), (1, 2)], n, k, p, noise, covariate_effect=cov_effect, random_state=seed)
    return case["views"], case["C"], case["outcome"]


# ----------------------------------------------------------------------------- figures
from pysimlr.plotting import (  # noqa: E402
    plot_path_model,
    plot_edge_matrix,
    plot_path_comparison,
    plot_edge_null,
    plot_edge_features,
)


# ----------------------------------------------------------------------------- report
def _png(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=160); import matplotlib.pyplot as plt; plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _table(df: pd.DataFrame, floatfmt="{:.4g}"):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
    return df.to_html(index=False, border=0, classes="t")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/audit/evidence/path_model")
    ap.add_argument("--method", default="simlr")
    ap.add_argument("--n-permutations", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    warnings.simplefilter("ignore")
    views, C, y = chain_case(seed=a.seed)
    n = views[0].shape[0]; n_tr = int(0.7 * n)
    tr, te = [v[:n_tr] for v in views], [v[n_tr:] for v in views]
    res = pysimlr.fit(a.method, tr, 3, random_state=a.seed, confounds=C[:n_tr])  # the sink removes the leak
    names = ["view0", "view1", "view2", "outcome"]
    graphs = {"chain + outcome": [(0, 1), (1, 2), (2, 3)], "star on view0 + outcome": [(0, 1), (0, 2), (2, 3)],
              "complete": None, "empty": [], "chain, no outcome edge": [(0, 1), (1, 2)]}
    pm = sem.path_model(res, te, graph=graphs["chain + outcome"], covariates=C[n_tr:], outcome=y[n_tr:],
                        conditioning="parents", n_permutations=a.n_permutations, random_state=a.seed)
    pm_pair = sem.path_model(res, te, graph=graphs["chain + outcome"], covariates=C[n_tr:], outcome=y[n_tr:],
                             conditioning="pairwise", n_permutations=a.n_permutations, random_state=a.seed)
    cmp = sem.compare_path_models(res, tr, te, graphs, covariates_train=C[:n_tr], covariates_test=C[n_tr:],
                                  outcome_train=y[:n_tr], outcome_test=y[n_tr:], n_bootstrap=2000,
                                  n_permutations=1000, random_state=a.seed)
    pp = sem.predict_path(pm, [te[0], None, te[2]], covariates=C[n_tr:])
    true1 = sem.view_scores(res, te)["scores"][1]
    r2_missing = 1 - ((pp["scores"]["view1"] - true1) ** 2).sum() / ((true1 - true1.mean(0)) ** 2).sum()
    yhat = pp["outcome_pred"][:, 0]; yz = (y[n_tr:] - y[n_tr:].mean()) / y[n_tr:].std()
    r2_outcome = 1 - ((yhat - yz) ** 2).sum() / ((yz - yz.mean()) ** 2).sum()

    figs = {"path_model": plot_path_model(pm),
            "path_model_component0": plot_path_model(pm, component=0),
            "edge_matrix": plot_edge_matrix(pm),
            "comparison": plot_path_comparison(cmp),
            "edge_null": plot_edge_null(pm),
            "edge_features": plot_edge_features(pm, "view0 - view1")}
    for nm, f in figs.items():
        f.savefig(os.path.join(a.out, f"reference_{nm}.png"), dpi=200)
    edges = pm["edges"]; edges_pair = pm_pair["edges"]
    comps = pm["edge_components"]
    comparison = pd.DataFrame([{k: v for k, v in r.items() if k != "edges"} for r in cmp["comparison"]])
    edges.to_csv(os.path.join(a.out, "reference_edges_parents.csv"), index=False)
    edges_pair.to_csv(os.path.join(a.out, "reference_edges_pairwise.csv"), index=False)
    comparison.to_csv(os.path.join(a.out, "reference_comparison.csv"), index=False)

    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Path model reference report</title>
<style>body{{font-family:Helvetica Neue,Helvetica,Arial,sans-serif;color:#1a1a1a;max-width:1100px;margin:32px auto;padding:0 16px;line-height:1.45}}
h1{{font-size:22px}}h2{{font-size:16px;margin-top:34px;border-bottom:1px solid #e1e0d9;padding-bottom:4px}}
.t{{border-collapse:collapse;font-size:12.5px;margin:10px 0}}.t th{{text-align:left;color:#52514e;font-weight:600;border-bottom:1px solid #c3c2b7;padding:4px 10px}}
.t td{{padding:3px 10px;border-bottom:1px solid #eeede8;font-variant-numeric:tabular-nums}}img{{max-width:100%;}}p.note{{color:#52514e;font-size:13px}}
code{{background:#f0efec;padding:1px 4px;border-radius:3px}}</style></head><body>
<h1>Score-level path model: reference report</h1>
<p class="note">Planted chain view0 - view1 - view2, outcome from view2's first latent, two covariates leaking into every
view (strength 0.8; removed by the confound sink at fit time and partialled out again as nuisance). Method <code>{a.method}</code>, k = 3, n = {n} (train {n_tr}, held-out {n - n_tr}),
B = {a.n_permutations} Freedman-Lane sweeps on the held-out rows, seed {a.seed}. Scores source: <code>{pm['scores_source']}</code>.</p>

<h2>1. Fitted path model (hypothesis: chain + outcome edge; conditioning on parents)</h2>
<img src="data:image/png;base64,{_png(figs['path_model'])}">
<img src="data:image/png;base64,{_png(figs['path_model_component0'])}">
<p class="note">Omnibus: max Pillai {pm['omnibus']['max_pillai']:.3f} (p = {pm['omnibus']['p_max']:.3g}), mean Pillai
{pm['omnibus']['mean_pillai']:.3f} (p = {pm['omnibus']['p_mean']:.3g}). Graph fit: present edges passing (Holm) =
{pm['fit']['present_pass']}, absent edges passing = {pm['fit']['absent_pass']}, max absent Pillai {pm['fit']['max_absent_pillai']:.3f}
(p = {pm['fit']['max_absent_p']:.3g}).</p>
{_table(edges)}
<p class="note">The same edges under pairwise conditioning (nuisance only). The absent view0 - view2 edge is strong
pairwise and weak given view1: the chain's conditional independence, attenuated by measurement error in view1's scores.</p>
{_table(edges_pair)}

<h2>2. Per-component partial correlations</h2>
<img src="data:image/png;base64,{_png(figs['edge_matrix'])}">
{_table(comps)}

<h2>3. Graph comparison (measurement model fixed)</h2>
<img src="data:image/png;base64,{_png(figs['comparison'])}">
{_table(comparison)}
<p class="note">Best (fewest edges whose paired delta CI against the top includes 0, plan F1): <b>{cmp['best']}</b>;
top by held-out log-likelihood: <b>{cmp['best_by_loglik']}</b>. BIC counts each undirected edge once (plan F4).</p>

<h2>4. Null distributions</h2>
<img src="data:image/png;base64,{_png(figs['edge_null'])}">

<h2>5. Prediction through the graph</h2>
<p>With view1 missing, its scores predicted from view0 and view2 reach R² = {r2_missing:.3f} against the scores computed
from the real view1. The outcome predicted from its graph parents (all views present) reaches held-out R² = {r2_outcome:.3f}.</p>

<h2>6. Feature contributions to edge view0 - view1</h2>
<img src="data:image/png;base64,{_png(figs['edge_features'])}">
</body></html>"""
    path = os.path.join(a.out, "reference_report.html")
    with open(path, "w") as fh:
        fh.write(html)
    print(f"wrote {path}")
    print(edges.to_string(index=False)); print(comparison.to_string(index=False))
    print(f"best {cmp['best']}; best_by_loglik {cmp['best_by_loglik']}; fit {pm['fit']}")
    print(f"missing-view R2 {r2_missing:.3f}; outcome R2 {r2_outcome:.3f}")


if __name__ == "__main__":
    main()
