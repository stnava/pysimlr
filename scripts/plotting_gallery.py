#!/usr/bin/env python3
"""Render every pysimlr.plotting function on a planted multi-view study into docs/plotting_gallery.html.

The study (n = 300; 240 training, 60 held-out subjects) has four views with
different coupling, so the figures have something to show:

* C1 is shared by all four views and drives the outcome y;
* C2 couples cortex and tracts only (structural);
* C3 couples cortex and networks only (functional);
* cognition is a narrow view (8 tests) carrying C1 alone.

Fits: simlr (positive) with held-out prediction, loading_inference (with a
permutation null), select_rank, flow_v with dynamic weights, a flow on
nonlinear views, an unconverged fit, a repeated-split benchmark, and both
fused-ridge models. Regenerate after changing any plot.

Usage: python scripts/plotting_gallery.py [--out docs/plotting_gallery.html] [--png-dir DIR]
"""
import argparse, base64, html, io, os, sys, time, warnings
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, torch
import pysimlr

CORTEX = (["sup frontal", "mid frontal", "inf frontal", "orbitofrontal", "precentral", "paracentral"],
          ["sup parietal", "inf parietal", "supramarginal", "angular", "precuneus", "postcentral"],
          ["sup temporal", "mid temporal", "inf temporal", "fusiform", "parahippocampal", "entorhinal"],
          ["lat occipital", "cuneus", "lingual", "pericalcarine", "calcarine", "occipital pole"],
          ["rostral ACC", "caudal ACC", "posterior cing", "isthmus cing", "cingulate gyrus", "subcallosal"])
LOBES = ["frontal", "parietal", "temporal", "occipital", "cingulate"]
TRACTS = ["SLF L", "SLF R", "ILF L", "ILF R", "IFOF L", "IFOF R", "uncinate L", "uncinate R",
          "CST L", "CST R", "ant thal rad L", "ant thal rad R", "sup thal rad L", "sup thal rad R",
          "post thal rad L", "post thal rad R", "forceps major", "forceps minor", "CC body", "CC genu",
          "CC splenium", "tapetum", "fornix", "cingulum hipp"]
NETWORKS = ["DMN PCC", "DMN mPFC", "DMN angular", "DMN hipp", "DMN temp pole", "DMN dlPFC",
            "VIS V1", "VIS V2", "VIS dorsal", "VIS ventral", "VIS MT", "VIS LO",
            "SAL ACC", "SAL AI L", "SAL AI R", "SAL SMA", "SAL TPJ", "SAL thal"]
COGNITION = ["memory delayed", "memory immediate", "executive", "processing speed", "language",
             "visuospatial", "attention", "global"]
VIEWS = ["cortex", "tracts", "networks", "cognition"]


def study(seed=0, n=400):
    """Planted latents z (n x 3, non-negative), four views, and an outcome driven by z1."""
    rng = np.random.default_rng(seed)
    z = np.abs(rng.standard_normal((n, 3)))
    # which latent each feature block loads on (None = noise only)
    blocks = {"cortex": [0] * 12 + [1] * 6 + [2] * 12,               # frontal+parietal / temporal / occipital+cingulate
              "tracts": [0] * 8 + [1] * 8 + [1] * 4 + [None] * 4,    # association / projection / commissural
              "networks": [0] * 6 + [2] * 12,                        # default mode / visual + salience
              "cognition": [0] * 8}
    noise = {"cortex": 0.55, "tracts": 0.6, "networks": 0.6, "cognition": 0.7}
    views = []
    for v in VIEWS:
        cols = []
        for b in blocks[v]:
            x = noise[v] * rng.standard_normal(n)
            if b is not None:
                x = x + (0.7 + 0.5 * rng.random()) * z[:, b]
            cols.append(x)
        views.append(torch.tensor(np.stack(cols, 1)).float())
    y = 1.0 * z[:, 0] + 0.7 * rng.standard_normal(n)
    names = [sum(CORTEX, []), TRACTS, NETWORKS, COGNITION]
    lobe = [LOBES[i // 6] for i in range(30)]
    return z, views, names, y, lobe


def heldout_r(method, views, y, seed):
    from sklearn.linear_model import LinearRegression, RidgeCV
    rng = np.random.default_rng(seed); n = len(y); te = rng.choice(n, n // 5, replace=False)
    tr = np.setdiff1d(np.arange(n), te)
    if method == "ridge":
        X = np.hstack([v.numpy() for v in views]); m = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(X[tr], y[tr])
        return float(np.corrcoef(m.predict(X[te]), y[te])[0, 1])
    res = pysimlr.fit(method, [v[tr] for v in views], 3, random_state=seed, positivity="positive")
    utr = pysimlr.predict(res, [v[tr] for v in views])["u"].numpy(); ute = pysimlr.predict(res, [v[te] for v in views])["u"].numpy()
    m = LinearRegression().fit(utr, y[tr]); return float(np.corrcoef(m.predict(ute), y[te])[0, 1])


CSS = """
:root{color-scheme:light;--page:#f9f9f7;--card:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--hair:rgba(11,11,11,.10);--accent:#2a78d6}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
header{padding:56px max(16px,6vw) 28px;border-bottom:1px solid var(--hair);background:var(--card)}
header h1{margin:0 0 8px;font-size:34px;font-weight:650;letter-spacing:-.02em}
header p{margin:0;max-width:72ch;color:var(--ink2)}
header .meta{margin-top:14px;font-size:13px;color:var(--muted)}
nav{position:sticky;top:0;z-index:5;background:rgba(249,249,247,.92);backdrop-filter:blur(8px);
border-bottom:1px solid var(--hair);padding:10px max(16px,6vw);display:flex;gap:8px;flex-wrap:wrap}
nav a{font-size:13px;color:var(--ink2);text-decoration:none;padding:5px 11px;border-radius:999px;border:1px solid var(--hair);background:var(--card)}
nav a:hover{color:var(--ink);border-color:rgba(11,11,11,.25)}
main{padding:8px max(16px,6vw) 80px}
section{padding-top:44px}
section h2{margin:0 0 6px;font-size:22px;font-weight:650;letter-spacing:-.01em}
section>p{margin:0 0 20px;color:var(--ink2);max-width:80ch}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,560px),1fr));gap:20px;align-items:start}
figure{margin:0;background:var(--card);border:1px solid var(--hair);border-radius:14px;overflow:hidden}
figure.wide{grid-column:1/-1}
figure img{display:block;width:100%;height:auto}
figcaption{padding:12px 16px 15px;border-top:1px solid var(--hair);font-size:13.5px;color:var(--ink2)}
figcaption code{display:inline-block;margin-bottom:4px;font:12.5px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--ink);
background:rgba(42,120,214,.07);padding:2px 7px;border-radius:6px}
footer{padding:24px max(16px,6vw) 48px;color:var(--muted);font-size:13px;border-top:1px solid var(--hair)}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "docs/plotting_gallery.html"))
    ap.add_argument("--png-dir", default=None, help="also write each figure as a PNG here")
    a = ap.parse_args(); warnings.simplefilter("ignore"); t0 = time.time()
    z, views, names, y, lobe = study()
    tr, te = [x[:320] for x in views], [x[320:] for x in views]
    pos = pysimlr.fit("simlr", tr, 3, positivity="positive", random_state=0)
    short = pysimlr.fit("simlr", tr, 3, iterations=3, random_state=0)
    flow = pysimlr.fit("flow_v", tr, 3, positivity="positive", dynamic_weights=True, random_state=0)
    rngn = np.random.default_rng(7); zn = rngn.standard_normal((300, 2))     # nonlinear views for the warp
    nl_views = [torch.tensor(np.c_[np.exp(0.8 * zn[:, :1]), np.tanh(2 * zn[:, 1:])] @ rngn.random((2, p))
                             + 0.1 * rngn.standard_normal((300, p))).float() for p in (10, 8)]
    flow_nl = pysimlr.fit("flow_v", nl_views, 2, random_state=0)
    pred = pysimlr.predict(pos, te)
    inf = pysimlr.loading_inference("simlr", tr, 3, n_subsamples=30, n_permutations=10, positivity="positive",
                                    random_state=0, n_jobs=-1)
    rank = pysimlr.select_rank("simlr", tr, [2, 3, 4, 5], n_subsamples=12, positivity="positive",
                               random_state=0, n_jobs=-1)
    splits = pd.DataFrame([dict(method=m, split=s, score=heldout_r(m, views, y, s))
                           for s in range(8) for m in ("ridge", "simlr", "pca")])
    bench = pd.DataFrame([dict(model=m, sparseness=sp, seed=s,
                               recovery=float(np.clip(0.95 - 0.3 * abs(sp - (0.5 if m == "simlr" else 0.3))
                                                      + 0.03 * np.random.default_rng(s).standard_normal(), 0, 1)))
                          for m in ("simlr", "ned") for sp in (0.3, 0.5, 0.7) for s in range(5)])
    Xf = np.random.default_rng(1).standard_normal((300, 8))
    yf = Xf @ np.r_[1.0, -0.8, 0.5, np.zeros(5)] + np.random.default_rng(2).logistic(size=300)
    fnames = ["age", "APOE4", "education", "BMI", "systolic BP", "HbA1c", "sleep", "steps"]
    cv = pysimlr.fused_ridge_cv(Xf, yf, thresholds=np.quantile(yf, [.2, .4, .6, .8]), lambda1s=(0.01, 0.1),
                                lambda2s=(0.0, 0.5, 5.0), top_ks=(None, 3), n_folds=4, feature_names=fnames)
    expo = np.random.default_rng(3).uniform(0, 1, 300)
    strata = [expo > q for q in (0.0, 0.25, 0.5, 0.75)]
    ys = Xf @ np.r_[0.5, 0.5, 0, 0, 0, 0, 0, 0] + 1.5 * expo * Xf[:, 0] + 0.5 * np.random.default_rng(4).standard_normal(300)
    fr_strata = pysimlr.fit_fused_ridge(Xf, ys, groups=strata, lambda1=0.01, lambda2=0.5, feature_names=fnames)
    fr_strata["tasks"] = ["exposure > 0", "> .25", "> .5", "> .75"]
    V = VIEWS
    W, N = "wide", ""
    from pysimlr.plotting import _pairwise
    P = _pairwise(pos, pred)

    def who(j, thr=0.5):
        pr = [(P[j, a, b], V[a], V[b]) for a in range(4) for b in range(a + 1, 4) if P[j, a, b] >= thr]
        if len(pr) == 6:
            return f"C{j + 1} joins all four views (every pair r \u2265 {min(p[0] for p in pr):.2f})"
        if not pr:
            return f"C{j + 1} links no pair above r = {thr}"
        return f"C{j + 1} links " + ", ".join(f"{a}\u2013{b} ({r:.2f})" for r, a, b in sorted(pr, reverse=True))
    chord_cap = "Held-out subjects. " + "; ".join(who(j) for j in range(P.shape[0])) + "."

    sections = [
        ("shared", "Shared signal across views", "Which views carry which component, and does it hold on held-out subjects?", [
            (W, "plot_view_chord(res, pred=pred)", chord_cap, pysimlr.plot_view_chord(pos, pred=pred, view_names=V)),
            (N, "plot_component_radar(res, pred=pred)", "Each view's held-out coupling with the consensus across components; the grey outline is the mean over views. A sub-coupled view is a spike, not a full polygon.", pysimlr.plot_component_radar(pos, pred=pred, view_names=V)),
            (N, "plot_view_radar(res, views, pred=pred)", "A fingerprint per view: coupling and reconstruction, training and held-out, rank and sparsity.", pysimlr.plot_view_radar(pos, tr, pred=pred, view_names=V)),
            (W, "plot_latent_coupling(res, pred=pred)", "View against consensus per component (ring training, dot held-out) and the view-by-view correlations of the strongest component.", pysimlr.plot_latent_coupling(pos, pred=pred, view_names=V)),
            (N, "plot_view_correlations(views)", "Shared multivariate signal before any fit (adjusted RV), as a chord.", pysimlr.plot_view_correlations(tr, names=V)),
        ]),
        ("features", "What builds each component", "Never the encoder v: structure, loadings, intervals and stability.", [
            (W, "plot_feature_chord(res, inf=inf)", "The strongest features of every view, joined to the components they build. Only features whose interval excludes 0 or that are stability-selected are drawn.", pysimlr.plot_feature_chord(pos, inf=inf, view_names=V, feature_names=names)),
            (W, "plot_top_features(res, inf=inf, component=0)", "Component 1 in every view: 95% intervals, permutation null band, selected names in bold.", pysimlr.plot_top_features(pos, inf=inf, component=0, view_names=V, feature_names=names, top_n=10)),
            (W, "plot_top_features(res, inf=inf, component=1, views=[0, 1])", "A second component, cortex and tracts side by side.", pysimlr.plot_top_features(pos, inf=inf, component=1, views=[0, 1], view_names=V, feature_names=names, top_n=10)),
            (N, "plot_feature_groups(res, lobes, inf, statistic='n_significant')", "Cortical regions per lobe whose interval excludes 0, per component.", pysimlr.plot_feature_groups(pos, lobe, inf=inf, view=0, feature_names=names, statistic="n_significant", view_names=V)),
            (N, "plot_feature_groups(res, lobes)", "Mean |structure| per lobe and component.", pysimlr.plot_feature_groups(pos, lobe, view=0, feature_names=names, view_names=V)),
            (N, "plot_loadings(res, view=0)", "Hard-assigned loadings of the cortex, ordered by component.", pysimlr.plot_loadings(pos, view=0, feature_names=names)),
            (W, "plot_variance_explained(res, views, pred)", "Reconstruction R² per view, and the structure each component carries.", pysimlr.plot_variance_explained(pos, tr, pred=pred, view_names=V)),
        ]),
        ("scores", "Subject scores", "Scores live in the rotated frame u R.", [
            (N, "plot_scores(res, color_by=y, new_views=held_out)", "Training subjects shaded by the outcome; held-out subjects ringed; marginal densities on the edges.", pysimlr.plot_scores(pos, color_by=y[:320], new_views=te, color_label="outcome y")),
            (N, "plot_latent_recovery(res, z_true)", "Synthetic studies: |corr| of each component with each planted latent.", pysimlr.plot_latent_recovery(pos, z[:320])),
        ]),
        ("health", "Optimisation, basis and rank", "Check these before reading any result.", [
            (N, "plot_convergence(res)", "simlr: the objective per sweep; the returned iterate is ringed.", pysimlr.plot_convergence(pos)),
            (N, "plot_convergence(unconverged)", "An unconverged fit (iterations=3) says so, with its stop reason.", pysimlr.plot_convergence(short)),
            (N, "plot_convergence(flow_res)", "flow_v: total loss with its similarity and reconstruction terms on their own scales.", pysimlr.plot_convergence(flow)),
            (N, "plot_rank_selection(sel)", "Component stability per candidate k (planted k = 3), with the 0.9 and 0.8 reference bars.", pysimlr.plot_rank_selection(rank)),
            (W, "plot_basis_diagnostics(res)", "Effective rank against k, column overlap, conditioning and sparsity, per view.", pysimlr.plot_basis_diagnostics(pos, view_names=V)),
        ]),
        ("weights", "View weights and flows", "Dynamic weights and what a nonlinear model does after the linear first layer.", [
            (W, "plot_modality_weights(flow_res)", "flow_v with dynamic_weights=True: final weights against 1/M and their trajectories.", pysimlr.plot_modality_weights(flow, view_names=V)),
            (N, "plot_flow_warping(flow_res, nonlinear_views)", "On nonlinear views the flow bends the projection but stays monotone (Spearman in each title).", pysimlr.plot_flow_warping(flow_nl, nl_views, components=(0, 1), view_names=["view A", "view B"])),
        ]),
        ("summary", "One-call diagnostic sheet", "Everything above that matters, for one fit.", [
            (W, "plot_fit_summary(res, views, pred=pred, inf=inf)", "Convergence, basis rank, coupling, the strongest component's view chord, and its top features.", pysimlr.plot_fit_summary(pos, tr, pred=pred, inf=inf, view_names=V, feature_names=names)),
        ]),
        ("bench", "Benchmarks", "Compare methods with their uncertainty, not point estimates.", [
            (W, "plot_benchmark_uncertainty(df, baseline='ridge')", "Held-out r of y over 8 random splits (real fits): bootstrap interval per method and the paired difference against ridge.", pysimlr.plot_benchmark_uncertainty(splits, baseline="ridge", metric_label="held-out r")),
            (N, "plot_benchmark(df, x, y)", "A metric against a setting, mean ± SE per method, labelled at the line ends.", pysimlr.plot_benchmark(bench, x="sparseness", y="recovery")),
            (N, "plot_benchmark(df, x, y, kind='bar')", "The same as thin grouped bars.", pysimlr.plot_benchmark(bench, x="sparseness", y="recovery", kind="bar")),
        ]),
        ("fused", "Fused ridge", "Coefficients fused across outcome thresholds (the ANTsR model) or subject strata.", [
            (N, "plot_coefficient_paths(res)", "Thresholds model (binomial): slopes across P(y ≥ c), tied by λ₂.", pysimlr.plot_coefficient_paths(cv["result"])),
            (N, "plot_fused_coefficients(res)", "The same fit as a matrix (top_k selected features only).", pysimlr.plot_fused_coefficients(cv["result"])),
            (N, "plot_fused_ridge_cv(cv)", "Cross-validated loss over (λ₁, λ₂) at the chosen top_k; the chosen cell is outlined.", pysimlr.plot_fused_ridge_cv(cv)),
            (N, "plot_coefficient_paths(strata_res)", "Strata model: age's effect grows with exposure; fusion smooths it across nested strata.", pysimlr.plot_coefficient_paths(fr_strata, features=["age", "APOE4", "education"])),
            (N, "plot_predictions(y, yhat)", "Predicted P(y ≥ median) for the two observed classes.", pysimlr.plot_predictions(yf >= np.median(yf), pysimlr.predict_fused_ridge(cv["result"], Xf, task=cv["result"]["tasks"][1], type="response"))),
        ]),
    ]
    if a.png_dir:
        os.makedirs(a.png_dir, exist_ok=True)
    out, n = [], 0
    for sid, title, intro, items in sections:
        cards = []
        for cls, code, cap, fig in items:
            buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=170); plt.close(fig); n += 1
            if a.png_dir:
                open(os.path.join(a.png_dir, f"{n:02d}_{code.split('(')[0]}.png"), "wb").write(buf.getvalue())
            cards.append(f'<figure class="{cls}"><img loading="lazy" alt="{html.escape(code)}" '
                         f'src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'
                         f"<figcaption><code>{html.escape(code)}</code><br>{html.escape(cap)}</figcaption></figure>")
        out.append(f"<section id='{sid}'><h2>{html.escape(title)}</h2><p>{html.escape(intro)}</p>"
                   f"<div class='grid'>{''.join(cards)}</div></section>")
    nav = "".join(f"<a href='#{sid}'>{html.escape(t)}</a>" for sid, t, _, _ in sections)
    page = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>pysimlr figure gallery</title><style>{CSS}</style></head><body>"
            f"<header><h1>pysimlr figure gallery</h1><p>Every figure in <code>pysimlr.plotting</code>, on one planted "
            f"four-view study: a component shared by all views, one shared by cortex and tracts, one by cortex and "
            f"networks, and a narrow cognition view that carries only the first.</p>"
            f"<p class='meta'>pysimlr {pysimlr.__version__} · {n} figures · rendered in {time.time() - t0:.0f} s by "
            f"<code>scripts/plotting_gallery.py</code></p></header><nav>{nav}</nav><main>{''.join(out)}</main>"
            f"<footer>Colours: views take fixed categorical slots validated for colour-vision deficiency; magnitude is one "
            f"blue ramp; signed values use a blue–red scale with a neutral midpoint.</footer></body></html>")
    open(a.out, "w").write(page)
    print(f"{n} figures -> {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB, {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
