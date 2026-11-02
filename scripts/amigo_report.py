"""Build docs/amigo_report.html from the amigo evidence CSVs (self-contained)."""
import html
import os

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EV = os.path.join(REPO, "docs", "audit", "evidence")


def table(df, fmt=4):
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        cells = []
        for v in r.values:
            if isinstance(v, float):
                cells.append(f"<td class='n'>{v:.{fmt}f}</td>" if pd.notna(v) else "<td class='n'>-</td>")
            else:
                cells.append(f"<td>{html.escape(str(v))}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def bars(items, vmax):
    """Horizontal bars: items = [(label, value, cls)]."""
    h = 22 * len(items) + 10
    out = [f"<svg viewBox='0 0 560 {h}' role='img' aria-label='span instability'>"]
    for i, (lab, v, cls) in enumerate(items):
        y = 8 + 22 * i
        w = max(1.0, 360 * v / vmax)
        out.append(f"<text x='0' y='{y + 13}' class='lab'>{html.escape(lab)}</text>"
                   f"<rect x='170' y='{y}' width='{w:.1f}' height='16' class='{cls}'/>"
                   f"<text x='{176 + w:.1f}' y='{y + 13}' class='val'>{v:.4f}</text>")
    out.append("</svg>")
    return "".join(out)


def main():
    S = pd.read_csv(os.path.join(EV, "amigo_loadings_synthetic.csv"))
    A = pd.read_csv(os.path.join(EV, "amigo_loadings_adni.csv"))
    C = pd.read_csv(os.path.join(EV, "constraint_selection.csv"))
    T = pd.read_csv(os.path.join(EV, "linear_tol_selection.csv"))
    long = S.melt(id_vars=["gen", "pos", "seed", "latent_r2", "converged", "n_steps"])
    long[["object", "metric"]] = long.variable.str.split("|", expand=True)
    syn = long.groupby(["pos", "object", "metric"]).value.mean().unstack("metric").reset_index()
    syn = syn[["pos", "object", "instability", "matched_r", "support_f1", "coherence", "overlap"]]
    syn.columns = ["sign mode", "object", "span instability", "matched |r|", "support F1", "column coherence", "support overlap"]
    bar_items = []
    for pos in ("either", "positive"):
        for obj, cls in (("v", "b1"), ("loadings", "b2")):
            v = float(syn[(syn["sign mode"] == pos) & (syn["object"] == obj)]["span instability"].iloc[0])
            bar_items.append((f"{pos} - {'V (encoder)' if obj == 'v' else 'loadings'}", v, cls))
    adn = A[["pos", "k", "v|instability", "loadings|instability", "loadings|matched_r",
             "component_1_stability", "component_2_stability", "component_3_stability",
             "selected_features", "expected_false_selections"]].copy()
    adn.columns = ["sign mode", "k", "V instability", "loadings instability", "loadings matched |r|",
                   "comp 1 stab.", "comp 2 stab.", "comp 3 stab.", "selected", "E[false] bound"]
    s = C[C.fixture == "synthetic"]
    des = s.groupby("arm").agg(F1=("f1", "mean"), frame_defect=("frame_defect", "mean"),
                               overlap=("support_overlap", "mean"),
                               converged=("stop", lambda x: f"{(x == 'converged').sum()}/{len(x)}"),
                               steps=("n_steps", "mean")).reset_index()
    des = des[des.arm.isin(["w0.5", "w0.5_nodefect"])].replace(
        {"w0.5": "projection + defect penalty (before)", "w0.5_nodefect": "projection only (amigo)"})
    des.columns = ["objective", "support F1", "frame defect", "support overlap", "converged (300)", "mean steps"]
    tol = T.groupby("tol").agg(converged=("stop", lambda x: f"{(x == 'converged').sum()}/{len(x)}"),
                               steps=("n_steps", "mean"), r2=("r2", "mean"), f1=("f1", "mean")).reset_index()
    tol["tol"] = tol["tol"].map(lambda v: f"{v:g}")
    tol.columns = ["tol", "converged (100 sweeps)", "mean steps", "latent R2", "support F1"]
    css = """
:root{--bg:#fbfbfa;--fg:#1d1d1f;--mut:#5f6368;--line:#dcdcd8;--card:#ffffff;--b1:#c4553f;--b2:#2f6db5;--acc:#2f6db5}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#17181a;--fg:#e8e8e6;--mut:#a0a3a8;--line:#34363a;--card:#1f2124;--b1:#e07a62;--b2:#6ea3e6;--acc:#6ea3e6}}
:root[data-theme="dark"]{--bg:#17181a;--fg:#e8e8e6;--mut:#a0a3a8;--line:#34363a;--card:#1f2124;--b1:#e07a62;--b2:#6ea3e6;--acc:#6ea3e6}
body{background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:0;padding:24px 16px}
main{max-width:980px;margin:0 auto} h1{font-size:24px;margin:0 0 4px} h2{font-size:18px;margin:32px 0 8px}
p,li{color:var(--fg)} .mut{color:var(--mut)} .card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px 16px;margin:12px 0;overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px} th,td{border-bottom:1px solid var(--line);padding:5px 8px;text-align:left;white-space:nowrap}
td.n{text-align:right;font-variant-numeric:tabular-nums} th{color:var(--mut);font-weight:600}
svg{width:100%;max-width:560px} .b1{fill:var(--b1)} .b2{fill:var(--b2)} .lab,.val{fill:var(--fg);font-size:12px}
code{font-size:13px}"""
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>amigo loadings report</title><style>{css}</style></head><body><main>
<h1>amigo: identified loadings for every pysimlr fit</h1>
<p class="mut">Branch <code>amigo</code>. Evidence: <code>docs/audit/evidence/amigo_loadings_*.csv</code>, <code>constraint_selection.csv</code>, <code>linear_tol_selection.csv</code>. Script: <code>scripts/amigo_loadings_evaluation.py</code>. Tracking: <code>docs/tracking_amigo.md</code>.</p>
<div class="card"><b>Summary.</b>
<ul>
<li>The encoder <code>V</code> is weakly identified: two refits on 98% of the rows move it substantially, while the latent <code>U</code> barely moves.</li>
<li>Every fit now also returns <code>structure</code> (feature-latent correlations, in a canonical varimax basis) and <code>loadings</code> (their OLEE summary). On synthetic data these are 50-150&times; more stable than <code>V</code> and meet the contract exactly: column coherence 0, and support overlap 0 under positivity.</li>
<li>On ADNI-MCI the loadings are only as stable as <code>U</code>. At k = 3 one component is not identified; at k = 2 under positivity everything is stable.</li>
</ul></div>
<h2>1. Synthetic: encoder vs loadings</h2>
<p>Default <code>simlr</code>, {{buried, linear}} &times; 3 seeds per sign mode. Every fit converged ({int(S.n_steps.min())}-{int(S.n_steps.max())} sweeps); latent R&sup2; {S.latent_r2.min():.3f}-{S.latent_r2.max():.3f}.</p>
<div class="card">{bars(bar_items, max(v for _, v, _ in bar_items))}<p class="mut">Span instability: mean sin&sup2; of the principal angles between the two refits' bases (0 = the same subspace).</p></div>
<div class="card">{table(syn)}</div>
<p>Signed loadings are dense (a polar factor), so their support F1 is below <code>V</code>'s. Sparse signed claims come from stability selection.</p>
<h2>2. ADNI-MCI</h2>
<p>4 views (p = 160/160/62/11), the benchmark's graph topology. Component stability is the median correlation between each refit's aligned column and the reference, from <code>loading_inference</code> with 30 half-sample refits. Stability selection uses threshold 0.8 and tolerates 1 expected false selection per view.</p>
<div class="card">{table(adn, 3)}</div>
<p>Claims per component need a stability of about 0.9 or more.</p>
<ul>
<li>At k = 3 one component falls below 0.4 (signed) or 0.74 (positive).</li>
<li>At k = 2, positive mode is stable throughout: loadings instability 0.007, matched |r| 0.995.</li>
</ul>
<p>Whether ADNI should use k = 2 is open item A4.</p>
<h2>3. The objective: projection only</h2>
<div class="card">{table(des, 4)}</div>
<h2>4. Stopping tolerance</h2>
<div class="card">{table(tol, 4)}</div>
<p class="mut">The 100-sweep fits match the 300-sweep fits to &lt; 0.001 latent R&sup2;. Default now <code>tol = 1e-4</code>, <code>iterations = 300</code>.</p>
<h2>5. Known costs and open items</h2>
<ul>
<li><b>Three measured regressions</b> from removing the defect penalty are recorded as strict xfails (thresholds unchanged): the near-Stiefel tolerance at <code>nsa_w=1</code>, the face-residual calibration, and one support-recovery fixture (0.978 &rarr; 0.813). See A3.</li>
<li><b>Positive-mode <code>V</code></b> is less stable without the penalty (0.030 &rarr; 0.161 on the planted case). The loadings are unaffected.</li>
<li><b>Plug-in <code>structure</code> is inflated</b> because <code>U</code> is fitted on the same rows. Thresholds come from <code>loading_inference</code>, never from point estimates.</li>
<li><b>The paper</b> still reads <code>V</code> and the removed <code>Lobe_Crosstalk</code> column (A2).</li>
</ul>
</main></body></html>"""
    out = os.path.join(REPO, "docs", "amigo_report.html")
    with open(out, "w") as fh:
        fh.write(page)
    print(out)


if __name__ == "__main__":
    main()
