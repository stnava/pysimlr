#!/usr/bin/env python3
"""Render every pysimlr.plotting function on realistic synthetic data into docs/plotting_gallery.html.

Three named views (cortical regions grouped by lobe, white-matter tracts,
functional networks) share three planted non-negative latents; an outcome
depends on the first. Fits: simlr (positive and signed), flow_v and pca, with
held-out subjects, loading_inference (with a permutation null), select_rank, a
repeated-split benchmark, and both fused-ridge models. Regenerate after
changing any plot.

Usage: python scripts/plotting_gallery.py [--out docs/plotting_gallery.html]
"""
import argparse, base64, html, io, os, sys, time, warnings
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, torch
import pysimlr

LOBES = ["frontal", "parietal", "temporal", "occipital", "cingulate"]


def data(seed=0, n=240):
    rng = np.random.default_rng(seed); k = 3
    z = np.abs(rng.standard_normal((n, k)))
    specs = [("cortex", [f"{LOBES[i % 5]}_{i:02d}" for i in range(30)]),
             ("tracts", [f"tract_{i:02d}" for i in range(24)]),
             ("networks", [f"net_{i:02d}" for i in range(18)])]
    views, names = [], []
    for name, feats in specs:
        p = len(feats); v = np.zeros((p, k)); blk = p // k
        for j in range(k):
            v[j * blk:(j + 1) * blk, j] = 0.4 + rng.random(blk)
        views.append(torch.tensor(z @ v.T + 0.6 * rng.standard_normal((n, p))).float()); names.append(feats)
    y = 1.0 * z[:, 0] + 0.6 * rng.standard_normal(n)
    return z, views, names, y, [s[0] for s in specs]


def heldout_r(method, views, y, k, seed, lr):
    from sklearn.linear_model import LinearRegression
    rng = np.random.default_rng(seed); n = len(y); te = rng.choice(n, n // 5, replace=False)
    tr = np.setdiff1d(np.arange(n), te)
    if method == "ridge":
        from sklearn.linear_model import RidgeCV
        X = np.hstack([v.numpy() for v in views]); m = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(X[tr], y[tr])
        return float(np.corrcoef(m.predict(X[te]), y[te])[0, 1])
    kw = dict(learning_rate=lr) if method == "simlr" else {}
    res = pysimlr.fit(method, [v[tr] for v in views], k, random_state=seed, positivity="positive", **kw)
    utr = pysimlr.predict(res, [v[tr] for v in views])["u"].numpy(); ute = pysimlr.predict(res, [v[te] for v in views])["u"].numpy()
    m = LinearRegression().fit(utr, y[tr]); return float(np.corrcoef(m.predict(ute), y[te])[0, 1])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(REPO, "docs/plotting_gallery.html"))
    a = ap.parse_args(); warnings.simplefilter("ignore"); t0 = time.time()
    z, views, names, y, vnames = data()
    tr, te = [x[:200] for x in views], [x[200:] for x in views]
    lr = 0.01
    pos = pysimlr.fit("simlr", tr, 3, positivity="positive", learning_rate=lr, random_state=0)
    sig = pysimlr.fit("simlr", tr, 3, learning_rate=lr, random_state=0)
    short = pysimlr.fit("simlr", tr, 3, iterations=3, learning_rate=lr, random_state=0)
    flow = pysimlr.fit("flow_v", tr, 3, positivity="positive", dynamic_weights=True, random_state=0)
    rngn = np.random.default_rng(7); zn = rngn.standard_normal((300, 2))     # nonlinear views for the warp
    nl_views = [torch.tensor(np.c_[np.exp(0.8 * zn[:, :1]), np.tanh(2 * zn[:, 1:])] @ rngn.random((2, p))
                             + 0.1 * rngn.standard_normal((300, p))).float() for p in (10, 8)]
    flow_nl = pysimlr.fit("flow_v", nl_views, 2, random_state=0)
    pca = pysimlr.fit("pca", tr, 3, random_state=0)
    pred = pysimlr.predict(pos, te)
    inf = pysimlr.loading_inference("simlr", tr, 3, n_subsamples=30, n_permutations=10, positivity="positive",
                                    learning_rate=lr, random_state=0, n_jobs=-1)
    rank = pysimlr.select_rank("simlr", tr, [2, 3, 4, 5], n_subsamples=12, positivity="positive",
                               learning_rate=lr, random_state=0, n_jobs=-1)
    lobe = [n.split("_")[0] for n in names[0]]
    rows = [dict(method=m, split=s, score=heldout_r(m, views, y, 3, s, lr))
            for s in range(8) for m in ("ridge", "simlr", "pca")]
    splits = pd.DataFrame(rows)
    bench = pd.DataFrame([dict(model=m, sparseness=sp, seed=s,
                               recovery=float(np.clip(0.95 - 0.3 * abs(sp - (0.5 if m == "simlr" else 0.3))
                                                      + 0.03 * np.random.default_rng(s).standard_normal(), 0, 1)))
                          for m in ("simlr", "ned") for sp in (0.3, 0.5, 0.7) for s in range(5)])
    X = np.hstack([views[0].numpy(), views[2].numpy()])[:, :8]
    Xf = np.random.default_rng(1).standard_normal((300, 8))
    yf = Xf @ np.r_[1.0, -0.8, 0.5, np.zeros(5)] + np.random.default_rng(2).logistic(size=300)
    cv = pysimlr.fused_ridge_cv(Xf, yf, thresholds=np.quantile(yf, [.2, .4, .6, .8]), lambda1s=(0.01, 0.1),
                                lambda2s=(0.0, 0.5, 5.0), top_ks=(None, 3), n_folds=4,
                                feature_names=[f"f{i}" for i in range(8)])
    expo = np.random.default_rng(3).uniform(0, 1, 300)
    strata = [expo > q for q in (0.0, 0.25, 0.5, 0.75)]
    ys = Xf @ np.r_[0.5, 0.5, 0, 0, 0, 0, 0, 0] + 1.5 * expo * Xf[:, 0] + 0.5 * np.random.default_rng(4).standard_normal(300)
    fr_strata = pysimlr.fit_fused_ridge(Xf, ys, groups=strata, lambda1=0.01, lambda2=0.5,
                                        feature_names=[f"f{i}" for i in range(8)])
    fr_strata["tasks"] = ["expo>0", "expo>.25", "expo>.5", "expo>.75"]

    sections = [
        ("Optimisation health, basis and rank", "Check these before reading any result.", [
            ("plot_convergence(res)", "simlr, converged: the returned best iterate is starred.", pysimlr.plot_convergence(pos)),
            ("plot_convergence(res)", "An unconverged fit (iterations=3) is titled in orange with its stop reason.", pysimlr.plot_convergence(short)),
            ("plot_convergence(res)", "flow_v: total loss with similarity and reconstruction terms (dashed, right axis).", pysimlr.plot_convergence(flow)),
            ("plot_convergence(res)", "pca: closed form, so a summary card.", pysimlr.plot_convergence(pca)),
            ("plot_basis_diagnostics(res)", "Effective rank against k (orange if columns merge), overlap, conditioning, sparsity.", pysimlr.plot_basis_diagnostics(pos, view_names=vnames)),
            ("plot_rank_selection(sel)", "Planted k = 3: all components stable at k = 3, a weak one appears at k >= 4.", pysimlr.plot_rank_selection(rank)),
        ]),
        ("Cross-view coupling, weights and flows", "Is the shared signal real out of sample, and which views drive it?", [
            ("plot_latent_coupling(res, pred=pred)", "View-vs-consensus r per component, training and held-out (change in brackets), and view-vs-view.", pysimlr.plot_latent_coupling(pos, pred=pred, view_names=vnames)),
            ("plot_latent_coupling(pca_res)", "Joint-basis baselines have no per-view scores; the figure says so.", pysimlr.plot_latent_coupling(pca)),
            ("plot_modality_weights(flow_res)", "flow_v with dynamic_weights=True: MAI (dotted) rises as views align; weights stay at 1/M because all three views align about equally.", pysimlr.plot_modality_weights(flow, view_names=vnames)),
            ("plot_flow_warping(flow_res, views)", "flow_v on nonlinear views (exp and tanh of the latents): the flow bends the linear projection; Spearman shows it stays monotone.", pysimlr.plot_flow_warping(flow_nl, nl_views, components=(0, 1), view_names=["view A", "view B"])),
            ("plot_flow_warping(flow_res, views)", "flow_v on the linear study: the flow is close to affine, as it should be.", pysimlr.plot_flow_warping(flow, tr, components=(0,), view_names=vnames)),
            ("plot_view_correlations(views)", "Adjusted RV between views.", pysimlr.plot_view_correlations(tr, names=vnames)),
        ]),
        ("Interpretable features", "Never the encoder v: structure, loadings, intervals and stability.", [
            ("plot_top_features(res, inf=inf, component=0)", "Component 1 per view: 95% intervals, permutation null (dashed), selected in bold, stability in the title.", pysimlr.plot_top_features(pos, inf=inf, component=0, view_names=vnames, feature_names=names, top_n=12)),
            ("plot_top_features(res, inf=inf, component=1, rank_by='selection')", "Component 2, ranked by stability-selection frequency.", pysimlr.plot_top_features(pos, inf=inf, component=1, rank_by="selection", view_names=vnames, feature_names=names, top_n=10)),
            ("plot_top_features(res, component=2, kind='loadings')", "Hard-assigned loadings of component 3 (no inference needed).", pysimlr.plot_top_features(pos, component=2, kind="loadings", view_names=vnames, feature_names=names, top_n=10)),
            ("plot_feature_groups(res, groups)", "Cortex summarised by lobe: mean |structure| per component.", pysimlr.plot_feature_groups(pos, lobe, view=0, feature_names=names)),
            ("plot_feature_groups(res, groups, inf, statistic='n_significant')", "Regions per lobe whose 95% interval excludes 0, per component.", pysimlr.plot_feature_groups(pos, lobe, inf=inf, view=0, feature_names=names, statistic="n_significant")),
            ("plot_feature_groups(res, groups, inf, statistic='n_selected')", "Stability-selected regions per lobe (strict: expected false selections <= 1).", pysimlr.plot_feature_groups(pos, lobe, inf=inf, view=0, feature_names=names, statistic="n_selected")),
            ("plot_loadings(res, view)", "Positive fit: hard-assigned loadings (median margin in the title).", pysimlr.plot_loadings(pos, view=0, feature_names=names)),
            ("plot_variance_explained(res, views, pred)", "Reconstruction R^2, training against held-out; structure captured per component.", pysimlr.plot_variance_explained(pos, tr, pred=pred, view_names=vnames)),
        ]),
        ("Subject scores", "Scores live in the rotated frame u @ R.", [
            ("plot_scores(res, new_views=..., color_by=y)", "Training scores coloured by the outcome; held-out subjects as crosses.", pysimlr.plot_scores(pos, color_by=y[:200], new_views=te)),
            ("plot_latent_recovery(res, z_true)", "Synthetic studies: |corr| of components with the planted latents.", pysimlr.plot_latent_recovery(pos, z[:200])),
        ]),
        ("One-call diagnostic sheet", "Everything above that matters, for one fit.", [
            ("plot_fit_summary(res, views, pred=pred, inf=inf)", "Convergence, basis rank, R^2, held-out coupling, top features of the most coupled component.", pysimlr.plot_fit_summary(pos, tr, pred=pred, inf=inf, view_names=vnames, feature_names=names)),
        ]),
        ("Benchmarks", "Compare methods with their uncertainty, not point estimates.", [
            ("plot_benchmark_uncertainty(df, baseline='ridge')", "Held-out r of y from 8 random splits (real fits): bootstrap CI per method and paired difference against ridge.", pysimlr.plot_benchmark_uncertainty(splits, baseline="ridge", metric_label="held-out r")),
            ("plot_benchmark(df, x, y)", "A metric against a setting, mean +- SE per method.", pysimlr.plot_benchmark(bench, x="sparseness", y="recovery")),
            ("plot_benchmark(df, x, y, kind='bar')", "Grouped bars.", pysimlr.plot_benchmark(bench, x="sparseness", y="recovery", kind="bar")),
        ]),
        ("Fused ridge", "Coefficients fused across outcome thresholds (ANTsR model) or subject strata.", [
            ("plot_coefficient_paths(res)", "Thresholds model (binomial): slopes across P(y >= c_t), tied by lambda2.", pysimlr.plot_coefficient_paths(cv["result"])),
            ("plot_fused_coefficients(res)", "Same fit as a heatmap (top_k selected features only).", pysimlr.plot_fused_coefficients(cv["result"])),
            ("plot_fused_ridge_cv(cv)", "Cross-validated loss over (lambda1, lambda2) at the chosen top_k.", pysimlr.plot_fused_ridge_cv(cv)),
            ("plot_coefficient_paths(strata_res)", "Strata model: f0's effect grows with exposure; fusion smooths it across nested strata.", pysimlr.plot_coefficient_paths(fr_strata, features=["f0", "f1", "f2"])),
            ("plot_predictions(y, yhat)", "Predicted P(y >= median) against the observed indicator.", pysimlr.plot_predictions(yf >= np.median(yf), pysimlr.predict_fused_ridge(cv["result"], Xf, task=cv["result"]["tasks"][1], type="response"))),
        ]),
    ]
    out, n = [], 0
    for title, intro, items in sections:
        cards = []
        for code, cap, fig in items:
            buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=85, bbox_inches="tight"); plt.close(fig); n += 1
            cards.append(f'<figure><img alt="{html.escape(code)}" src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'
                         f"<figcaption><code>{html.escape(code)}</code><br>{html.escape(cap)}</figcaption></figure>")
        out.append(f"<section><h2>{html.escape(title)}</h2><p class='intro'>{html.escape(intro)}</p><div class='grid'>{''.join(cards)}</div></section>")
    page = ("<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
            "<title>pysimlr plotting gallery</title><style>"
            ":root{--bg:#fff;--fg:#1d1d1f;--muted:#5f6368;--card:#fafafa;--line:#e3e3e3}"
            "@media (prefers-color-scheme:dark){:root{--bg:#141414;--fg:#ececec;--muted:#a0a0a0;--card:#1e1e1e;--line:#333}}"
            "body{font-family:system-ui,-apple-system,sans-serif;margin:0 auto;max-width:1400px;padding:16px;background:var(--bg);color:var(--fg)}"
            "h1{margin:.2em 0}h2{border-bottom:1px solid var(--line);padding-bottom:.2em;margin-top:1.6em}.intro,.lede{color:var(--muted)}"
            ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,440px),1fr));gap:14px}"
            "figure{margin:0;border:1px solid var(--line);border-radius:6px;padding:8px;background:var(--card)}"
            "img{width:100%;height:auto;background:#fff;border-radius:3px}figcaption{font-size:.85em;margin-top:6px}"
            "code{font-size:.92em}</style>"
            f"<h1>pysimlr.plotting</h1><p class='lede'>Every public plot on one synthetic study (three named views, planted latents, an outcome), "
            f"pysimlr {pysimlr.__version__}. Generated by <code>scripts/plotting_gallery.py</code> in {time.time() - t0:.0f} s.</p>"
            + "".join(out) + "</html>")
    open(a.out, "w").write(page)
    print(f"wrote {a.out} ({n} figures, {len(sections)} sections, {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
