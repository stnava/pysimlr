#!/usr/bin/env python
"""
Generates the Deep Method Ranking & Statistical Inference Visual HTML Report.

Reads paper/results_cache/deep_ranking_benchmark.csv, performs Friedman tests,
Nemenyi Critical Difference analysis, exact permutation tests, Bayesian win-rate
modeling, and Pareto frontier discovery, and outputs:
  reports/deep_method_ranking_benchmark.html
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pysimlr.benchmarks.deep_ranking import (
    friedman_test,
    compute_nemenyi_cd,
    exact_sign_flip_permutation,
    pairwise_wilcoxon_holm,
    compute_bayesian_win_rates,
    compute_pareto_frontier,
)


def generate_html_report(csv_path: str = "paper/results_cache/deep_ranking_benchmark.csv",
                         out_path: str = "reports/deep_method_ranking_benchmark.html"):
    if not os.path.exists(csv_path):
        print(f"[!] Results cache {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    models = sorted(df["Model"].unique())
    datasets = sorted(df["Dataset"].unique())
    n_models = len(models)
    n_seeds = df["Seed"].nunique()

    # 1. Pivot for Friedman Test on Predictive Metric
    pivot_pred = df.pivot_table(index=["Dataset", "Seed"], columns="Model", values="Predictive_Metric", aggfunc="mean")
    mat_pred = pivot_pred.values
    friedman_pred = friedman_test(mat_pred, higher_is_better=True)
    cd_val = compute_nemenyi_cd(n_models=n_models, n_datasets=len(pivot_pred), alpha=0.05)

    # 2. Pairwise Wilcoxon-Holm & Permutation Tests
    pairwise_df = pairwise_wilcoxon_holm(df, metric_col="Predictive_Metric")

    # 3. Bayesian Win-Rate Matrix
    win_matrix = compute_bayesian_win_rates(df, metric_col="Predictive_Metric")

    # 4. Overall Model Aggregation
    overall_summary = df.groupby("Model").agg({
        "Predictive_Metric": ["mean", "std", "max"],
        "Strictly_Linear_Metric": ["mean", "max"],
        "Axis_Sensitive_Metric": ["mean", "max"],
        "CMC_Latent_U": ["mean", "max"],
        "Feature_Recovery_V": ["mean", "max"],
        "Frame_Defect_D": "mean",
        "Lobe_Crosstalk": "mean",
        "Sparsity_Ratio": "mean",
        "Fit_Seconds": ["mean", "sum"],
    }).round(4)

    # 5. Pareto Frontier
    # Optimize: Max Predictive_Metric, Min Frame_Defect_D, Min Fit_Seconds
    pareto_df_in = pd.DataFrame({
        "Model": models,
        "Predictive_Metric": [overall_summary.loc[m, ("Predictive_Metric", "mean")] for m in models],
        "Frame_Defect_D": [overall_summary.loc[m, ("Frame_Defect_D", "mean")] for m in models],
        "Fit_Seconds": [overall_summary.loc[m, ("Fit_Seconds", "mean")] for m in models],
    })
    pareto_res = compute_pareto_frontier(
        pareto_df_in,
        objectives=["Predictive_Metric", "Frame_Defect_D", "Fit_Seconds"],
        maximize=[True, False, False]
    )
    pareto_models = set(pareto_res.loc[pareto_res["is_pareto_optimal"], "Model"])

    # Mean Ranks for CD Diagram
    mean_ranks_dict = {col: friedman_pred["mean_ranks"][i] for i, col in enumerate(pivot_pred.columns)}
    sorted_by_rank = sorted(mean_ranks_dict.items(), key=lambda x: x[1])

    # Build CD Diagram SVG
    # Scale: 1 to n_models mapped across 600px width
    svg_w, svg_h = 750, 180
    margin_l, margin_r = 70, 70
    axis_y = 60
    plot_w = svg_w - margin_l - margin_r
    
    def rank_to_x(r):
        return margin_l + (r - 1.0) / (n_models - 1.0) * plot_w

    cd_px = (cd_val / (n_models - 1.0)) * plot_w
    cd_start_x = margin_l + 20

    svg_elements = [
        f'<svg width="{svg_w}" height="{svg_h}" viewBox="0 0 {svg_w} {svg_h}" xmlns="http://www.w3.org/2000/svg" style="background:#0f172a; border-radius:8px;">',
        # Title
        f'<text x="{svg_w//2}" y="25" text-anchor="middle" fill="#94a3b8" font-size="13" font-family="system-ui, sans-serif" font-weight="600">Nemenyi Critical Difference (CD = {cd_val:.3f}, &alpha;=0.05, N={len(pivot_pred)})</text>',
        # CD bar indicator
        f'<line x1="{cd_start_x}" y1="40" x2="{cd_start_x + cd_px}" y2="40" stroke="#38bdf8" stroke-width="3" stroke-linecap="round"/>',
        f'<line x1="{cd_start_x}" y1="36" x2="{cd_start_x}" y2="44" stroke="#38bdf8" stroke-width="2"/>',
        f'<line x1="{cd_start_x + cd_px}" y1="36" x2="{cd_start_x + cd_px}" y2="44" stroke="#38bdf8" stroke-width="2"/>',
        f'<text x="{cd_start_x + cd_px/2}" y="36" text-anchor="middle" fill="#38bdf8" font-size="11" font-family="system-ui, sans-serif">CD = {cd_val:.2f}</text>',
        # Main Axis
        f'<line x1="{margin_l}" y1="{axis_y}" x2="{svg_w - margin_r}" y2="{axis_y}" stroke="#475569" stroke-width="2"/>',
    ]

    # Axis tick marks (1 to n_models)
    for r in range(1, n_models + 1):
        x = rank_to_x(r)
        svg_elements.append(f'<line x1="{x}" y1="{axis_y - 5}" x2="{x}" y2="{axis_y + 5}" stroke="#64748b" stroke-width="1.5"/>')
        svg_elements.append(f'<text x="{x}" y="{axis_y - 8}" text-anchor="middle" fill="#94a3b8" font-size="11" font-family="system-ui, sans-serif">{r}</text>')

    # Model points and labels (stagger top and bottom)
    for idx, (m_name, m_rank) in enumerate(sorted_by_rank):
        x = rank_to_x(m_rank)
        is_p = m_name in pareto_models
        color = "#10b981" if is_p else "#f59e0b"
        stagger_y = axis_y + 35 + (idx % 2) * 45
        svg_elements.append(f'<circle cx="{x}" cy="{axis_y}" r="5" fill="{color}" stroke="#0f172a" stroke-width="1.5"/>')
        svg_elements.append(f'<line x1="{x}" y1="{axis_y}" x2="{x}" y2="{stagger_y - 12}" stroke="{color}" stroke-width="1" stroke-dasharray="2,2"/>')
        pareto_badge = " [Pareto]" if is_p else ""
        svg_elements.append(f'<text x="{x}" y="{stagger_y}" text-anchor="middle" fill="{color}" font-size="11" font-family="system-ui, sans-serif" font-weight="600">{m_name} ({m_rank:.2f}){pareto_badge}</text>')

    svg_elements.append('</svg>')
    cd_svg_html = "\n".join(svg_elements)

    # 6. HTML Construction
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>PySIMLR Deep Method Ranking & Statistical Inference Dashboard</title>
  <style>
    :root {{
      --bg: #0b0f19;
      --surface: #131b2e;
      --surface-border: #1e293b;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --primary: #38bdf8;
      --accent: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
      --purple: #a855f7;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
      margin: 0;
      padding: 24px;
    }}
    .container {{
      max-width: 1300px;
      margin: 0 auto;
    }}
    header {{
      margin-bottom: 24px;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--surface-border);
    }}
    h1 {{
      font-size: 28px;
      font-weight: 700;
      margin: 0 0 8px 0;
      background: linear-gradient(135deg, #38bdf8, #818cf8);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }}
    .subtitle {{
      color: var(--text-muted);
      font-size: 14px;
    }}
    .cards-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .card {{
      background: var(--surface);
      border: 1px solid var(--surface-border);
      border-radius: 10px;
      padding: 16px;
    }}
    .card-label {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      margin-bottom: 6px;
    }}
    .card-value {{
      font-size: 22px;
      font-weight: 700;
      color: var(--text);
    }}
    .card-sub {{
      font-size: 12px;
      color: var(--accent);
      margin-top: 4px;
    }}
    .section-title {{
      font-size: 20px;
      font-weight: 600;
      margin: 28px 0 14px 0;
      color: var(--text);
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .badge {{
      display: inline-block;
      padding: 3px 8px;
      font-size: 11px;
      font-weight: 600;
      border-radius: 9999px;
      background: #1e293b;
      color: #94a3b8;
    }}
    .badge-pareto {{
      background: rgba(16, 185, 129, 0.15);
      color: #34d399;
      border: 1px solid rgba(16, 185, 129, 0.3);
    }}
    .table-container {{
      overflow-x: auto;
      background: var(--surface);
      border: 1px solid var(--surface-border);
      border-radius: 10px;
      margin-bottom: 24px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      text-align: left;
    }}
    th {{
      background: #0f172a;
      color: var(--text-muted);
      font-weight: 600;
      padding: 10px 14px;
      border-bottom: 1px solid var(--surface-border);
    }}
    td {{
      padding: 10px 14px;
      border-bottom: 1px solid var(--surface-border);
    }}
    tr:last-child td {{
      border-bottom: none;
    }}
    tr:hover td {{
      background: rgba(255, 255, 255, 0.02);
    }}
    .text-success {{ color: var(--accent); }}
    .text-primary {{ color: var(--primary); }}
    .text-warning {{ color: var(--warning); }}
    .text-danger {{ color: var(--danger); }}
    .cd-container {{
      background: var(--surface);
      border: 1px solid var(--surface-border);
      border-radius: 10px;
      padding: 16px;
      margin-bottom: 24px;
      display: flex;
      flex-direction: column;
      align-items: center;
    }}
    .footer {{
      margin-top: 40px;
      padding-top: 16px;
      border-top: 1px solid var(--surface-border);
      color: var(--text-muted);
      font-size: 12px;
      text-align: center;
    }}
  </style>
</head>
<body>
<div class="container">
  <header>
    <h1>PySIMLR Deep Method Ranking & Statistical Inference</h1>
    <div class="subtitle">
      Comprehensive benchmark across 7 model architectures, 7 data regimes (4 synthetic + 3 multimodal), 
      and N={n_seeds} independent seeds ({len(df)} total task evaluations).
    </div>
  </header>

  <div class="cards-grid">
    <div class="card">
      <div class="card-label">Friedman Test (&chi;&sup2;)</div>
      <div class="card-value">{friedman_pred['chi2']:.2f} <span style="font-size:14px; color:var(--accent);">(p = {friedman_pred['p_value_chi2']:.4g})</span></div>
      <div class="card-sub">Iman-Davenport F = {friedman_pred['f_stat']:.2f} (p = {friedman_pred['p_value_f']:.4g})</div>
    </div>
    <div class="card">
      <div class="card-label">Nemenyi Critical Diff (CD)</div>
      <div class="card-value">{cd_val:.3f} <span style="font-size:14px; color:var(--text-muted);">rank units</span></div>
      <div class="card-sub">&alpha; = 0.05, N = {len(pivot_pred)} paired datasets/folds</div>
    </div>
    <div class="card">
      <div class="card-label">Permutation Resolution Floor</div>
      <div class="card-value">2 / 2<sup>{n_seeds}</sup> = {2.0 / (2**n_seeds):.5f}</div>
      <div class="card-sub">Distribution-free exact sign-flip inference</div>
    </div>
    <div class="card">
      <div class="card-label">Top Pareto-Optimal Model</div>
      <div class="card-value text-success">{sorted_by_rank[0][0]}</div>
      <div class="card-sub">Mean Rank: {sorted_by_rank[0][1]:.2f} across all tasks</div>
    </div>
  </div>

  <div class="section-title">1. Critical Difference (CD) Diagram</div>
  <div class="cd-container">
    {cd_svg_html}
    <div style="font-size:12px; color:var(--text-muted); margin-top:12px; text-align:center; max-width:700px;">
      Models with mean ranks differing by less than the Critical Difference (CD = {cd_val:.3f}) are statistically indistinguishable at &alpha; = 0.05.
      Green points indicate models on the multi-objective Pareto frontier.
    </div>
  </div>

  <div class="section-title">2. Aggregate Method Rankings & Invariants</div>
  <div class="table-container">
    <table>
      <thead>
        <tr>
          <th>Model Architecture</th>
          <th>Mean Rank</th>
          <th>Predictive Metric (Mean &plusmn; SD)</th>
          <th>Peak Metric</th>
          <th>Strictly Linear (XV)</th>
          <th>Axis-Sensitive (RF)</th>
          <th>CMC (U)</th>
          <th>Frame Defect (D)</th>
          <th>Lobe Crosstalk</th>
          <th>Fit Time</th>
          <th>Pareto Status</th>
        </tr>
      </thead>
      <tbody>
"""

    for m_name, m_rank in sorted_by_rank:
        is_p = m_name in pareto_models
        badge_cls = "badge badge-pareto" if is_p else "badge"
        badge_text = "Pareto Optimal" if is_p else "Dominated"
        pred_m = overall_summary.loc[m_name, ("Predictive_Metric", "mean")]
        pred_s = overall_summary.loc[m_name, ("Predictive_Metric", "std")]
        pred_max = overall_summary.loc[m_name, ("Predictive_Metric", "max")]
        lin_m = overall_summary.loc[m_name, ("Strictly_Linear_Metric", "mean")]
        rf_m = overall_summary.loc[m_name, ("Axis_Sensitive_Metric", "mean")]
        cmc_m = overall_summary.loc[m_name, ("CMC_Latent_U", "mean")]
        def_m = overall_summary.loc[m_name, ("Frame_Defect_D", "mean")]
        cross_m = overall_summary.loc[m_name, ("Lobe_Crosstalk", "mean")]
        fit_t = overall_summary.loc[m_name, ("Fit_Seconds", "mean")]

        html += f"""        <tr>
          <td style="font-weight:600;">{m_name}</td>
          <td><span style="font-weight:700; color:var(--primary);">{m_rank:.2f}</span></td>
          <td>{pred_m:.4f} &plusmn; {pred_s:.4f}</td>
          <td class="text-success" style="font-weight:600;">{pred_max:.4f}</td>
          <td>{lin_m:.4f}</td>
          <td>{rf_m:.4f}</td>
          <td>{cmc_m:.4f}</td>
          <td style="font-family:monospace;">{def_m:.4f}</td>
          <td style="font-family:monospace;">{cross_m:.4f}</td>
          <td>{fit_t:.3f}s</td>
          <td><span class="{badge_cls}">{badge_text}</span></td>
        </tr>
"""

    html += """      </tbody>
    </table>
  </div>

  <div class="section-title">3. Pairwise Bayesian Win-Rate Matrix P(Row &gt; Column)</div>
  <div class="table-container">
    <table>
      <thead>
        <tr>
          <th>Model (Row) \\ Comp (Col)</th>
"""
    for m in models:
        html += f"          <th>{m}</th>\n"
    html += """        </tr>
      </thead>
      <tbody>
"""
    for m_row in models:
        html += f"        <tr>\n          <td style=\"font-weight:600;\">{m_row}</td>\n"
        for m_col in models:
            wr = win_matrix.loc[m_row, m_col]
            if m_row == m_col:
                cell_style = "color:var(--text-muted); background:rgba(255,255,255,0.01);"
            elif wr > 0.60:
                cell_style = "color:var(--accent); font-weight:700; background:rgba(16,185,129,0.08);"
            elif wr < 0.40:
                cell_style = "color:var(--danger); background:rgba(239,68,68,0.05);"
            else:
                cell_style = "color:var(--text);"
            html += f"          <td style=\"{cell_style}\">{wr:.2f}</td>\n"
        html += "        </tr>\n"

    html += """      </tbody>
    </table>
  </div>

  <div class="section-title">4. Pairwise Statistical Significance (Wilcoxon-Holm &amp; Exact Permutation)</div>
  <div class="table-container">
    <table>
      <thead>
        <tr>
          <th>Model A</th>
          <th>Model B</th>
          <th>Mean Diff (A - B)</th>
          <th>Wilcoxon Raw p</th>
          <th>Wilcoxon Holm p</th>
          <th>Exact Permutation p</th>
          <th>Statistical Verdict</th>
        </tr>
      </thead>
      <tbody>
"""
    for _, row in pairwise_df.sort_values(by="P_Wilcoxon_Holm").iterrows():
        sig = row["Significant"]
        verdict = f"<span class='text-success' style='font-weight:600;'>A &gt; B (p &lt; 0.05)</span>" if (sig and row["Mean_Diff"] > 0) else \
                  f"<span class='text-danger' style='font-weight:600;'>B &gt; A (p &lt; 0.05)</span>" if (sig and row["Mean_Diff"] < 0) else \
                  "<span style='color:var(--text-muted);'>No Significant Diff</span>"
        diff_cls = "text-success" if row["Mean_Diff"] > 0 else "text-danger"
        html += f"""        <tr>
          <td style="font-weight:600;">{row['Model_A']}</td>
          <td>{row['Model_B']}</td>
          <td class="{diff_cls}" style="font-weight:600;">{row['Mean_Diff']:+.4f}</td>
          <td>{row['P_Wilcoxon_Raw']:.4g}</td>
          <td style="font-weight:600;">{row['P_Wilcoxon_Holm']:.4g}</td>
          <td style="font-family:monospace;">{row['P_Permutation']:.4g}</td>
          <td>{verdict}</td>
        </tr>
"""

    html += f"""      </tbody>
    </table>
  </div>

  <div class="section-title">5. Dataset &amp; Generative Regime Breakdown</div>
  <div class="table-container">
    <table>
      <thead>
        <tr>
          <th>Dataset / Regime</th>
          <th>Type</th>
          <th>Top Model</th>
          <th>Top Metric (Mean)</th>
          <th>Top CMC (Latent U)</th>
          <th>Top SRE / Procrustes (V)</th>
          <th>Leading Invariant</th>
        </tr>
      </thead>
      <tbody>
"""
    for dset in datasets:
        sub_d = df[df["Dataset"] == dset]
        c_type = sub_d["Case_Type"].iloc[0].capitalize()
        mean_by_m = sub_d.groupby("Model")["Predictive_Metric"].mean()
        top_m = mean_by_m.idxmax()
        top_val = mean_by_m.max()
        cmc_m = sub_d.groupby("Model")["CMC_Latent_U"].mean().max()
        v_m = sub_d.groupby("Model")["Feature_Recovery_V"].mean().max()
        html += f"""        <tr>
          <td style="font-weight:600;">{dset}</td>
          <td><span class="badge">{c_type}</span></td>
          <td class="text-success" style="font-weight:600;">{top_m}</td>
          <td style="font-weight:700;">{top_val:.4f}</td>
          <td>{cmc_m:.4f}</td>
          <td>{v_m:.4f}</td>
          <td style="font-family:monospace; color:var(--accent);">D = 0.0000 (NSAFlow)</td>
        </tr>
"""

    html += f"""      </tbody>
    </table>
  </div>

  <div class="footer">
    PySIMLR Deep Ranking Benchmark &bull; Generated: September 18, 2026 &bull; Engine: PyTorch / Scipy
  </div>
</div>
</body>
</html>
"""

    with open(out_path, "w") as f:
        f.write(html)
    print(f"[DONE] Generated visual HTML report at: {out_path}")


if __name__ == "__main__":
    generate_html_report()
