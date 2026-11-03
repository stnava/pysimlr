#!/usr/bin/env python3
"""Render every pysimlr.plotting plot on one synthetic example into docs/plotting_gallery.html.

A visual check of the plotting conventions; regenerate after changing a plot.
Usage: python scripts/plotting_gallery.py [--out docs/plotting_gallery.html]
"""
import argparse, base64, io, os, sys, warnings
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, torch
import pysimlr


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(REPO, "docs/plotting_gallery.html"))
    a = ap.parse_args(); warnings.simplefilter("ignore")
    rng = np.random.default_rng(0); n, k = 200, 3
    z = np.abs(rng.standard_normal((n, k)))
    views, names = [], []
    for m, p in enumerate((30, 24)):
        v = np.zeros((p, k)); blk = p // k
        for j in range(k):
            v[j * blk:(j + 1) * blk, j] = 0.5 + rng.random(blk)
        views.append(torch.tensor(z @ v.T + 0.3 * rng.standard_normal((n, p))).float())
        names.append([f"{'AB'[m]}{i}" for i in range(p)])
    tr, te = [x[:170] for x in views], [x[170:] for x in views]
    fit = pysimlr.fit("simlr", tr, k, positivity="positive", learning_rate=0.01, random_state=0)
    inf = pysimlr.loading_inference("simlr", tr, k, n_subsamples=20, positivity="positive",
                                    learning_rate=0.01, random_state=0, n_jobs=-1)
    rank = pysimlr.select_rank("simlr", tr, [2, 3, 4], n_subsamples=10, positivity="positive",
                               learning_rate=0.01, random_state=0, n_jobs=-1)
    X = rng.standard_normal((300, 8)); yv = X @ np.r_[1.0, -0.8, 0.5, np.zeros(5)] + rng.logistic(size=300)
    cv = pysimlr.fused_ridge_cv(X, yv, thresholds=np.quantile(yv, [.25, .5, .75]), lambda1s=(0.01, 0.1),
                                lambda2s=(0.0, 0.5, 5.0), top_ks=(None, 3), n_folds=4,
                                feature_names=[f"f{i}" for i in range(8)])
    bench = pd.DataFrame({"model": np.repeat(["simlr", "ned", "lend"], 12),
                          "sparseness": np.tile(np.repeat([0.3, 0.5, 0.7], 4), 3),
                          "recovery": rng.uniform(0.6, 1.0, 36)})
    figs = [
        ("plot_loadings", pysimlr.plot_loadings(fit, view=0, feature_names=names)),
        ("plot_structure_intervals", pysimlr.plot_structure_intervals(inf, view=0, component=0, feature_names=names)),
        ("plot_component_stability", pysimlr.plot_component_stability(rank)),
        ("plot_scores (with held-out)", pysimlr.plot_scores(fit, new_views=te)),
        ("plot_view_agreement", pysimlr.plot_view_agreement(fit, component=0)),
        ("plot_convergence", pysimlr.plot_convergence(fit)),
        ("plot_view_correlations", pysimlr.plot_view_correlations(tr, names=["view A", "view B"])),
        ("plot_latent_recovery", pysimlr.plot_latent_recovery(fit, z[:170])),
        ("plot_fused_coefficients", pysimlr.plot_fused_coefficients(cv["result"])),
        ("plot_coefficient_paths", pysimlr.plot_coefficient_paths(cv["result"])),
        ("plot_fused_ridge_cv", pysimlr.plot_fused_ridge_cv(cv)),
        ("plot_benchmark", pysimlr.plot_benchmark(bench, x="sparseness", y="recovery")),
        ("plot_predictions", pysimlr.plot_predictions(yv >= np.median(yv), pysimlr.predict_fused_ridge(
            cv["result"], X, task=cv["result"]["tasks"][1], type="response"))),
    ]
    cards = []
    for title, fig in figs:
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=90, bbox_inches="tight"); plt.close(fig)
        cards.append(f'<figure><img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'
                     f"<figcaption><code>{title}</code></figcaption></figure>")
    html = ("<!doctype html><meta charset='utf-8'><title>pysimlr plotting gallery</title>"
            "<style>body{font-family:system-ui,sans-serif;margin:16px;background:#fff;color:#222}"
            "main{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:16px}"
            "figure{margin:0;border:1px solid #ddd;padding:8px}img{max-width:100%}</style>"
            "<h1>pysimlr.plotting</h1><p>Every public plot on one synthetic example "
            "(scripts/plotting_gallery.py).</p><main>" + "".join(cards) + "</main>")
    open(a.out, "w").write(html)
    print(f"wrote {a.out} ({len(figs)} plots)")


if __name__ == "__main__":
    main()
