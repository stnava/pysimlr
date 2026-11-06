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
def plot_path_model(pm, alpha=0.05, ax=None, component=None):
    import matplotlib.pyplot as plt
    with P.theme():
        fig, ax = P._figure(ax, figsize=(6.4, 5.6), title="Fitted path model",
                            subtitle=f"{pm['rows']} rows, n = {pm['n']}, conditioning = {pm['conditioning']}, "
                                     f"B = {pm['parameters']['n_permutations']}; omnibus p = {pm['omnibus']['p_max']:.4g}")
        nodes = pm["nodes"]; m = len(nodes)
        ang = {nm: np.pi / 2 + 2 * np.pi * i / m for i, nm in enumerate(nodes)}
        pos = {nm: (np.cos(a), np.sin(a)) for nm, a in ang.items()}
        col = {nm: ("#1a1a1a" if nm == "outcome" else P._view_color(i)) for i, nm in enumerate(nodes)}
        for e in pm["edges"].to_dict("records"):
            a, b = e["a"], e["b"]
            (x1, y1), (x2, y2) = pos[a], pos[b]
            sig = e["p_maxt"] is not None and e["p_maxt"] < alpha
            if not e["present"] and not sig:
                continue
            width = 0.6 + 5.4 * e["pillai"]
            color = P._WARN if (sig and not e["present"]) else P._INK2
            ax.plot([x1, x2], [y1, y2], color=color, lw=width, ls="-" if sig else (0, (3, 2.5)),
                    alpha=0.9 if sig else 0.6, solid_capstyle="round", zorder=1)
            xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
            # offset the label perpendicular to the edge so labels do not sit on the line
            dx, dy = x2 - x1, y2 - y1; nrm = np.hypot(dx, dy)
            ox, oy = -dy / nrm * 0.22, dx / nrm * 0.22
            ax.text(xm + ox, ym + oy, f"{e['pillai']:.2f}\np = {e['p_maxt']:.3g}", ha="center", va="center",
                    fontsize=8, color=P._INK, bbox=dict(boxstyle="round,pad=0.25", fc=P._SURFACE, ec="none", alpha=0.9), zorder=3)
        for nm in nodes:
            x, y = pos[nm]
            marker = "s" if nm == "outcome" else "o"
            ax.scatter([x], [y], s=2300 if nm == "outcome" else 1700, marker=marker, color=col[nm], zorder=4,
                       edgecolor=P._SURFACE, linewidth=2)
            ax.text(x, y, nm, ha="center", va="center", fontsize=8.5, color="white", weight="semibold", zorder=5)
        from matplotlib.lines import Line2D
        handles = [Line2D([], [], color=P._INK2, lw=2.5, label=f"edge, max-T p < {alpha}"),
                   Line2D([], [], color=P._INK2, lw=2.0, ls=(0, (3, 2.5)), alpha=0.6, label="hypothesised, not significant"),
                   Line2D([], [], color=P._WARN, lw=2.5, label="not hypothesised, significant")]
        ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=1, fontsize=8)
        ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.45, 1.35); ax.set_aspect("equal"); ax.axis("off")
        return fig


def plot_edge_matrix(pm, alpha=0.05):
    import matplotlib.pyplot as plt
    comps = pm["edge_components"]
    labels = sorted({f"{a} - {b}" for a, b in zip(comps.a, comps.b)})
    k = pm["k"]
    M = np.zeros((len(labels), k)); Q = np.ones((len(labels), k))
    for _, r in comps.iterrows():
        i = labels.index(f"{r.a} - {r.b}"); M[i, int(r.component)] = r.r; Q[i, int(r.component)] = r.get("q_bh", 1.0)
    with P.theme():
        fig = P._new((1.1 * k + 4.2, 0.55 * len(labels) + 2.0), title="Per-component partial correlations",
                     subtitle=f"hatched: BH q >= {alpha}; right column: edge Pillai")
        gs = fig.add_gridspec(1, 2, width_ratios=[k, 1.1], wspace=0.05)
        ax = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1])
        lim = max(0.3, float(np.abs(M).max()))
        im = P._matrix(ax, M, P._cmap("div"), -lim, lim, rows=labels, cols=[f"c{j + 1}" for j in range(k)])
        for i in range(len(labels)):
            for j in range(k):
                if Q[i, j] >= alpha:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="///", ec=P._MUTED, lw=0))
        pil = np.array([[next(e["pillai"] for e in pm["edges"].to_dict("records") if f"{e['a']} - {e['b']}" == lab)] for lab in labels])
        P._matrix(ax2, pil, P._cmap("seq"), 0, 1, rows=None, cols=["Pillai"])
        ax2.set_yticks([]); ax2.set_ylim(ax.get_ylim())
        P._colorbar(fig, im, ax, "partial r")
        return fig


def plot_path_comparison(cmp):
    with P.theme():
        rows = sorted(cmp["comparison"], key=lambda r: r["heldout_loglik"])
        fig = P._new((7.2, 0.5 * len(rows) + 2.4), title="Which graph predicts held-out scores best",
                     subtitle="held-out log-likelihood minus the top graph's; paired subject bootstrap 95% CI")
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 1.4], wspace=0.08)
        ax = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1], sharey=ax)
        y = np.arange(len(rows))
        for i, r in enumerate(rows):
            best = r["graph"] == cmp["best"]
            c = P._BLUE[6] if best else P._INK2
            ax.hlines(i, r["ci_low"], r["ci_high"], color=c, lw=2.2, alpha=0.6)
            P._dots(ax, [r["delta_vs_top"]], [i], c, s=52)
            ax.text(r["ci_high"], i, f"  best in {100 * r['best_fraction']:.0f}% of replicates", va="center", fontsize=8, color=P._INK2)
            ax2.scatter([r["present_pass"] if r["present_pass"] is not None else np.nan], [i], color=P._BLUE[6], s=40, label="present edges pass" if i == 0 else None)
            ax2.scatter([r["absent_pass"] if r["absent_pass"] is not None else np.nan], [i], color=P._RED[4], s=40, marker="D", label="absent edges pass" if i == 0 else None)
        ax.axvline(0, color=P._AXIS, lw=0.9)
        ax.set_yticks(y); ax.set_yticklabels([f"{r['graph']}  ({r['n_edges']} edges)" for r in rows])
        ax.set_xlabel("delta held-out log-likelihood"); P._clean(ax, grid="x")
        ax2.set_xlim(-0.08, 1.08); ax2.set_xlabel("fraction passing"); ax2.tick_params(labelleft=False); P._clean(ax2, grid="x")
        ax2.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=7.5, ncol=1, frameon=False)
        return fig


def plot_edge_null(pm, alpha=0.05):
    with P.theme():
        edges = [e for e in pm["edges"].to_dict("records") if e["present"] or (e["p_maxt"] is not None and e["p_maxt"] < alpha)]
        n = len(edges); cols = min(3, n); rws = int(np.ceil(n / cols))
        fig = P._new((3.2 * cols + 0.4, 2.4 * rws + 1.2), title="Observed edge statistic against its permutation null",
                     subtitle=f"Freedman-Lane, {pm['parameters']['n_permutations']} sweeps; the p-value is add-one")
        for i, e in enumerate(edges):
            ax = fig.add_subplot(rws, cols, i + 1)
            null = np.asarray(pm["null"][(e["a"], e["b"])])
            ax.hist(null, bins=40, color=P._BLUE[2], edgecolor="none")
            ax.axvline(e["pillai"], color=P._RED[4], lw=2)
            ax.set_title(f"{e['a']} - {e['b']}", fontsize=9.5)
            ax.text(0.97, 0.92, f"Pillai {e['pillai']:.2f}\np = {e['p_perm']:.3g}", transform=ax.transAxes, ha="right", va="top", fontsize=8, color=P._INK)
            ax.set_xlim(0, max(1.0, e["pillai"] * 1.1) if e["pillai"] > 0.8 else max(null.max(), e["pillai"]) * 1.15)
            ax.set_yticks([]); P._clean(ax, grid=None, baseline=True)
            if i // cols == rws - 1:
                ax.set_xlabel("Pillai under the null")
        return fig


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

    figs = {"path_model": plot_path_model(pm), "edge_matrix": plot_edge_matrix(pm),
            "comparison": plot_path_comparison(cmp), "edge_null": plot_edge_null(pm)}
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
