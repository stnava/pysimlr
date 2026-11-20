"""Every number the manuscript cites, computed from ``paper/results_cache`` and ``docs/audit/evidence``.

The chapters reference these values inline (`` `{python} N.adni_best_r2` ``)
instead of typing them, so the prose cannot drift from the data: before
2026-11-01 about 25 typed numbers disagreed with the caches they cited.

``N`` holds preformatted strings; ``RAW`` holds the underlying floats, so a
chapter can phrase a comparison conditionally.

Grouping conventions (one place, used everywhere):
  * "main" = ``deep_ranking_benchmark.csv``: q = 0.5, default positivity,
    12 datasets x 7 models x 3 seeds; every per-model figure averages the 12
    datasets with equal weight.
  * "operating points" = the three default-positivity files
    ``deep_ranking_benchmark_q{_none,05,08}.csv`` (756 fits). The 189 paired
    positivity="positive" reruns in ``operating_points_05_08_benchmark.csv``
    are excluded from pooled means, which they would double-weight.
  * mean +/- SD across seeds is the SD of the 3 per-seed values (or 20 seeds
    for ``v043_synthetic_theorems.json``).
"""
from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "results_cache")
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from pysimlr.benchmarks.deep_ranking import (  # noqa: E402
    build_ranking_blocks, compute_nemenyi_cd, compute_pareto_frontier, friedman_test,
    wilcoxon_vs_reference,
)

RENAME = {"Flow-SiMLR-V": "Flow-SiMR-V", "Flow-SiMLR": "Flow-SiMR"}
SYNTH = ["Linear", "Polynomial", "Sine", "Shared+Private", "NonnegParts", "NonnegPartsWeak"]
REAL = ["ADNI-MCI", "MultiOmics", "TCGA-KIRC", "Heart", "Diabetes", "mfeat"]
DEEP = ["LEND", "NED", "NEDPP", "Flow-SiMR-V"]


def _csv(name):
    d = pd.read_csv(os.path.join(CACHE, name))
    if "Model" in d.columns:
        d["Model"] = d["Model"].replace(RENAME)
    return d


def f3(x):
    return f"{x:.3f}"


def f4(x):
    return f"{x:.4f}"


def pct(x, nd=1):
    return f"{100 * x:.{nd}f}%"


def sci(x):
    m, e = f"{x:.2e}".split("e")
    return f"${m} \\times 10^{{{int(e)}}}$"


def _seed_stats(d, dataset, model, col):
    v = d[(d.Dataset == dataset) & (d.Model == model)].groupby("Seed")[col].mean()
    return float(v.mean()), float(v.std(ddof=1))


def op_default():
    """The 756 default-positivity operating-point fits (q = None, 0.5, 0.8),
    with ``Sparseness_Target`` = "None" for the unthresholded fits. Use this,
    not ``operating_points_05_08_benchmark.csv``, for pooled means: that file
    also holds 189 paired positivity="positive" reruns of three cohorts."""
    frames = []
    for f in ("deep_ranking_benchmark_q_none.csv", "deep_ranking_benchmark_q05.csv",
              "deep_ranking_benchmark_q08.csv"):
        frames.append(_csv(f))
    d = pd.concat(frames, ignore_index=True)
    d["Sparseness_Target"] = d["Sparseness_Target"].astype(object).where(d["Sparseness_Target"].notna(), "None")
    return d


def region_label(name):
    """Readable ADNI region name: 'left_entorhinaldktcortex' -> 'L entorhinal (DKT cortex label)'."""
    import re
    tissues = {"csf": "CSF", "gm": "grey matter", "wm": "white matter", "deepgray": "deep grey matter",
               "brainstem": "brainstem", "cerebellum": "cerebellum"}
    if name.endswith("tissues") and name[:-len("tissues")] in tissues:
        return f"{tissues[name[:-len('tissues')]]} (whole-brain total)"
    side = "L " if name.startswith("left_") else ("R " if name.startswith("right_") else "")
    n = re.sub(r"^(left|right)_", "", name)
    m = re.match(r"(.*?)dkt(regions|cortex|lobes)$", n)
    if m:
        base, kind = m.groups()
        if kind == "lobes":
            base, s = base.rsplit("_", 1)
            return f"{s.upper()} {base} lobe"
        return f"{side}{base.replace('_', ' ')} ({'DKT parcellation' if kind == 'regions' else 'DKT cortex label'})"
    m = re.match(r"(.*?)(mtl|cit168)$", n)
    if m:
        return f"{side}{m.group(1).replace('_', ' ')} ({m.group(2).upper()} atlas)"
    return side + n.replace("_", " ")


def build():
    n, raw = {}, {}

    def put(key, value, fmt=f3):
        raw[key] = value
        n[key] = fmt(value) if callable(fmt) else value

    main = _csv("deep_ranking_benchmark.csv")
    qfiles = {"None": "deep_ranking_benchmark_q_none.csv", 0.5: "deep_ranking_benchmark_q05.csv",
              0.8: "deep_ranking_benchmark_q08.csv"}
    op = pd.concat([_csv(f).assign(q=str(q)) for q, f in qfiles.items()], ignore_index=True)
    op_all = _csv("operating_points_05_08_benchmark.csv")
    sweep = _csv("adni_mci_1214_grid_sweep.csv")

    # ---- design counts
    put("n_default_fits", len(op), str)
    put("n_positive_reruns", len(op_all) - len(op), str)
    put("n_total_fits", f"{len(op_all):,}", str)
    put("n_main_fits", len(main), str)
    put("n_seeds", main.Seed.nunique(), str)
    _words = {6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
    n_models = main.Model.nunique()
    put("n_models", n_models, str)
    put("n_models_word", _words.get(n_models, str(n_models)), str)
    put("sweep_n", f"{len(sweep):,}", str)
    put("sweep_ok", f"{int((sweep.Status == 'ok').sum()):,}", str)

    # ---- Friedman / Nemenyi
    for tag, col in (("lin", "Strictly_Linear_Metric"), ("u", "Predictive_Metric"), ("rf", "Axis_Sensitive_Metric")):
        blocks = build_ranking_blocks(main, col, floor_threshold=None)[0]
        fr = friedman_test(blocks.values)
        put(f"fr_{tag}_chi2", fr["chi2"], lambda x: f"{x:.2f}")
        put(f"fr_{tag}_p", fr["p_value_chi2"], sci)
        put(f"fr_{tag}_F", fr["f_stat"], lambda x: f"{x:.2f}")
        put(f"fr_{tag}_pF", fr["p_value_f"], sci)
        ranks = blocks.rank(axis=1, ascending=False).mean()
        for m, r in ranks.items():
            put(f"rank_{tag}_{m}", float(r), lambda x: f"{x:.2f}")
        raw[f"ranks_{tag}"] = ranks.sort_values()
        if tag == "lin":
            put("n_blocks", blocks.shape[0], str)
            put("cd", compute_nemenyi_cd(blocks.shape[1], blocks.shape[0]), f3)
    order = raw["ranks_lin"]
    put("rank_lin_first", f"`{order.index[0]}`", str)
    put("rank_lin_top4", ", ".join(f"`{m}` ({order[m]:.2f})" for m in order.index[:4]), str)

    # ---- per-model means over 12 datasets, equal weight
    per_ds = main.groupby(["Model", "Dataset"])[["Strictly_Linear_Metric", "Axis_Sensitive_Metric",
                                                  "Predictive_Metric"]].mean()
    by_model = per_ds.groupby("Model").mean()
    drop = (by_model.Axis_Sensitive_Metric - by_model.Strictly_Linear_Metric)
    raw["axis_drop"] = drop
    for m, v in drop.items():
        put(f"drop_{m}", float(v), lambda x: f"{x:+.3f}")
    nsa_models = [m for m in drop.index if m not in ("PCA",)]
    put("drop_pca", float(drop["PCA"]), lambda x: f"{x:+.3f}")
    smallest = drop.drop("PCA").sort_values(ascending=False)
    put("drop_best_models", ", ".join(f"`{m}` ({v:+.3f})" for m, v in smallest.iloc[:2].items()), str)

    # ---- dataset-level cells cited in prose
    def cell(dataset, model, col="Strictly_Linear_Metric"):
        return float(main[(main.Dataset == dataset) & (main.Model == model)][col].mean())

    for ds, key in (("Shared+Private", "sp"), ("mfeat", "mfeat"), ("ADNI-MCI", "adni"), ("MultiOmics", "brca"),
                    ("TCGA-KIRC", "kirc"), ("Heart", "heart"), ("Diabetes", "diab"),
                    ("NonnegPartsWeak", "npw"), ("NonnegParts", "np")):
        sub = main[main.Dataset == ds].groupby("Model")["Strictly_Linear_Metric"].mean().sort_values(ascending=False)
        put(f"{key}_pca", cell(ds, "PCA"))
        put(f"{key}_best_model", f"`{sub.index[0]}`", str)
        put(f"{key}_best", float(sub.iloc[0]))
        for m in sub.index:
            put(f"{key}_{m}", float(sub[m]))
            put(f"{key}_{m}_u", cell(ds, m, "Predictive_Metric"))
            put(f"{key}_{m}_rf", cell(ds, m, "Axis_Sensitive_Metric"))
        put(f"{key}_pca_rf", cell(ds, "PCA", "Axis_Sensitive_Metric"))
        put(f"{key}_pca_u", cell(ds, "PCA", "Predictive_Metric"))

    # ---- ADNI-MCI headline
    adni = main[main.Dataset == "ADNI-MCI"].groupby(["Model", "Seed"])["Strictly_Linear_Metric"].mean().unstack()
    adni_mean, adni_sd = adni.mean(1), adni.std(1, ddof=1)
    best = adni_mean.sort_values(ascending=False)
    b = best.index[0]
    put("adni_best_model", f"`{b}`", str)
    put("adni_best_r2", float(best.iloc[0]))
    put("adni_best_sd", float(adni_sd[b]))
    put("adni_best_rf", cell("ADNI-MCI", b, "Axis_Sensitive_Metric"))
    put("adni_pca_sd", float(adni_sd["PCA"]))
    put("adni_gain_pct", 100 * (best.iloc[0] / adni_mean["PCA"] - 1), lambda x: f"{x:+.1f}%")
    rf = main[main.Dataset == "ADNI-MCI"].groupby("Model")["Axis_Sensitive_Metric"].mean()
    rf_best = rf.drop("PCA").sort_values(ascending=False)
    put("adni_rf_best_model", f"`{rf_best.index[0]}`", str)
    put("adni_rf_best", float(rf_best.iloc[0]))
    put("adni_rf_ratio", float(rf_best.iloc[0] / rf["PCA"]), lambda x: f"{x:.1f}")
    put("adni_pca_rf_drop_pct", 100 * (1 - rf["PCA"] / adni_mean["PCA"]), lambda x: f"{x:.0f}%")
    put("adni_ranked", ", ".join(f"`{m}` ({best[m]:.3f} $\\pm$ {adni_sd[m]:.3f})" for m in best.index[:4]), str)
    put("adni_simlr", float(adni_mean["SiMLR"]))

    # ---- real cohort cells with SD
    for ds, key in (("MultiOmics", "brca"), ("TCGA-KIRC", "kirc")):
        for m in ("Flow-SiMR-V", "LEND", "NEDPP", "PCA"):
            mu, sd = _seed_stats(main, ds, m, "Strictly_Linear_Metric")
            put(f"{key}_{m}_pm", f"{100 * mu:.1f}% $\\pm$ {100 * sd:.1f}%", str)
            mu, sd = _seed_stats(main, ds, m, "Predictive_Metric")
            put(f"{key}_{m}_u_pm", f"{100 * mu:.1f}% $\\pm$ {100 * sd:.1f}%", str)

    # ---- operating points (default positivity, equal dataset weight)
    opq = op.groupby(["q", "Model", "Dataset"])[["Strictly_Linear_Metric", "Axis_Sensitive_Metric",
                                                 "Sparsity_Ratio"]].mean().groupby("q").mean()
    for q in ("None", "0.5", "0.8"):
        k = q.replace(".", "")
        put(f"op_lin_{k}", float(opq.loc[q, "Strictly_Linear_Metric"]))
        put(f"op_rf_{k}", float(opq.loc[q, "Axis_Sensitive_Metric"]))
        put(f"op_sparsity_{k}", float(opq.loc[q, "Sparsity_Ratio"]), pct)
    put("op_retained_08", float(opq.loc["0.8", "Strictly_Linear_Metric"] / opq["Strictly_Linear_Metric"].max()),
        lambda x: f"{100 * x:.1f}%")
    put("op_best_q", str(opq["Strictly_Linear_Metric"].idxmax()), str)
    f1 = op[op.Case_Type == "synthetic"].groupby(["q", "Dataset", "Model"])["Support_Recovery_V"].mean()
    lin05 = f1.loc["0.5", "Linear"]
    put("f1_linear_q05_min", float(lin05.min())); put("f1_linear_q05_max", float(lin05.max()))
    put("f1_linear_q05_min_model", f"`{lin05.idxmin()}`", str); put("f1_linear_q05_max_model", f"`{lin05.idxmax()}`", str)
    npq = f1.loc["0.5", "NonnegParts"]
    put("f1_np_q05_best", float(npq.max())); put("f1_np_q05_best_model", f"`{npq.idxmax()}`", str)
    for ds, key in (("NonnegPartsWeak", "npw"), ("Shared+Private", "sp")):
        put(f"f1_{key}_q08_best", float(f1.loc["0.8", ds].max()))
        put(f"f1_{key}_q08_best_model", f"`{f1.loc['0.8', ds].idxmax()}`", str)
        put(f"f1_{key}_q05_best", float(f1.loc["0.5", ds].max()))
    simlr_f1 = f1.xs("SiMLR", level="Model")
    put("f1_simlr_np", float(simlr_f1.loc[("0.5", "NonnegParts")]))
    put("f1_simlr_npw", float(simlr_f1.loc[("0.5", "NonnegPartsWeak")]))
    for m in ("NED", "NEDPP"):
        put(f"f1_{m}_np", float(f1.loc[("0.5", "NonnegParts", m)]))
        put(f"f1_{m}_npw", float(f1.loc[("0.5", "NonnegPartsWeak", m)]))

    # ---- Pareto (speed vs consensus)
    agg = main.groupby("Model", as_index=False).agg(Predictive_Metric=("Predictive_Metric", "mean"),
                                                    Fit_Seconds=("Fit_Seconds", "mean"))
    pf = compute_pareto_frontier(agg, objectives=["Predictive_Metric", "Fit_Seconds"], maximize=[True, False])
    front = pf[pf.is_pareto_optimal].sort_values("Fit_Seconds").Model.tolist()
    put("pareto_2d", ", ".join(f"`{m}`" for m in front), str)
    raw["pareto_2d_models"] = front
    put("pca_fit_seconds", float(main[main.Model == "PCA"].Fit_Seconds.median()), lambda x: f"{x:.3f}")

    # ---- Wilcoxon vs PCA (12 blocks, Holm)
    wv = wilcoxon_vs_reference(main, "Strictly_Linear_Metric", "PCA",
                               models=[m for m in order.index if m != "PCA"])
    sig = wv[wv.P_Holm < 0.05].Model.tolist()
    raw["wilcoxon_sig"] = sig
    put("wilcoxon_sig", ", ".join(f"`{m}`" for m in sig) if sig else "none", str)
    put("wilcoxon_n_sig", len(sig), str)

    # ---- ADNI sweep
    ok = sweep[sweep.Status == "ok"]
    bm = ok.sort_values("CogPC1_R2", ascending=False).groupby("Model").first()
    lin_fams = [m for m in bm.index if m.startswith("SiMLR")]
    put("sweep_best_model", f"`{bm.CogPC1_R2.idxmax()}`", str)
    put("sweep_best_r2", float(bm.CogPC1_R2.max()), f4)
    put("sweep_linear_best_model", f"`{bm.loc[lin_fams].CogPC1_R2.idxmax()}`", str)
    put("sweep_linear_best_r2", float(bm.loc[lin_fams].CogPC1_R2.max()), f4)
    put("sweep_pca_r2", float(bm.loc["PCA", "CogPC1_R2"]), f4)
    m11 = ok.groupby("Model")["Mean11_RidgeCV"].max()
    put("sweep_m11_best_model", f"`{m11.drop('PCA').idxmax()}`", str)
    put("sweep_m11_best", float(m11.drop("PCA").max()), f3)
    put("sweep_m11_pca", float(m11["PCA"]), f3)
    put("sweep_m11_gain", 100 * (m11.drop("PCA").max() / m11["PCA"] - 1), lambda x: f"{x:+.1f}%")
    put("sweep_flow_best", float(bm.loc["Flow-SiMR-V", "CogPC1_R2"]), f4)

    # ---- ADNI ablations (scripts/paper_adni_ablations.py)
    w = _csv("ablation_w_sweep.csv").set_index("w")
    put("w_r2_050", float(w.loc[0.5, "ADNI_CogPC1_R2"])); put("w_r2_095", float(w.loc[0.95, "ADNI_CogPC1_R2"]))
    put("w_r2_001", float(w.loc[0.01, "ADNI_CogPC1_R2"]))
    put("w_effrank_001", float(w.loc[0.01, "ADNI_EffRank"]), lambda x: f"{x:.2f}")
    put("w_effrank_010", float(w.loc[0.10, "ADNI_EffRank"]), lambda x: f"{x:.2f}")
    put("w_D_001", float(w.loc[0.01, "ADNI_Defect_D"]))
    put("w_time_ratio_adni", float(w.loc[0.95, "ADNI_Fit_Sec"] / w.loc[0.5, "ADNI_Fit_Sec"]), lambda x: f"{x:.1f}")
    put("w_time_ratio_synth", float(w.loc[0.95, "Synth_Fit_Sec"] / w.loc[0.5, "Synth_Fit_Sec"]), lambda x: f"{x:.1f}")
    plateau = w[(w.index >= 0.2) & (w.index <= 0.85)].ADNI_CogPC1_R2
    put("w_plateau_min", float(plateau.min())); put("w_plateau_max", float(plateau.max()))

    rm = _csv("ablation_rowmean_ica_positivity.csv")
    fitted = rm[rm.Stage.str.startswith("Bipartite")]

    def rcell(prep, init, pos, col):
        s = fitted[fitted.Preprocessing.str.startswith(prep) & (fitted.Init == init) &
                   fitted.Positivity.str.startswith(pos)]
        return float(s[col].iloc[0])

    put("rm_col_pca_pos_r2", rcell("Col", "PCA", "Non", "CogPC1_R2_XV"))
    put("rm_row_pca_pos_r2", rcell("Row", "PCA", "Non", "CogPC1_R2_XV"))
    put("rm_row_ica_pos_r2", rcell("Row", "ICA", "Non", "CogPC1_R2_XV"))
    put("rm_col_pca_pos_sp", rcell("Col", "PCA", "Non", "sparsity_all"), lambda x: pct(x, 0))
    put("rm_row_pca_pos_sp", rcell("Row", "PCA", "Non", "sparsity_all"), lambda x: pct(x, 0))
    put("rm_row_ica_pos_sp", rcell("Row", "ICA", "Non", "sparsity_all"), lambda x: pct(x, 0))

    # ---- ADNI rank selection latent reliability (docs/audit/evidence/adni_rank_selection.csv)
    ars_path = os.path.join(HERE, "..", "docs", "audit", "evidence", "adni_rank_selection.csv")
    if not os.path.exists(ars_path):
        raise FileNotFoundError(f"Required rank-selection evidence missing: {ars_path}")
    ars = pd.read_csv(ars_path)
    put("adni_rank_margin", float(ars["margin"].iloc[0]), lambda x: f"{x:g}")
    for pos_name, ptag in (("positive", "pos"), ("either", "either")):
        sub_p = ars[ars.pos == pos_name]
        sel_vals = sub_p["selected_k"].dropna()
        put(f"adni_{ptag}_selected_k", int(sel_vals.iloc[0]) if len(sel_vals) else "None", str)
        for k_val in (2, 3):
            row = sub_p[sub_p.k == k_val].iloc[0]
            rels = [float(x) for x in str(row["latent_reliability"]).split()]
            gains = [float(x) for x in str(row["latent_reliability_gain"]).split()]
            for idx, (rv, gv) in enumerate(zip(rels, gains), 1):
                put(f"adni_k{k_val}_{ptag}_rel_c{idx}", rv, f3)
                put(f"adni_k{k_val}_{ptag}_gain_c{idx}", gv, f3)
            put(f"adni_k{k_val}_{ptag}_min_rel", float(row["min_latent_reliability"]), f3)
            put(f"adni_k{k_val}_{ptag}_min_gain", float(row["min_latent_reliability_gain"]), f3)
            put(f"adni_k{k_val}_{ptag}_mv_rel", float(row["latent_reliability_multivariate"]), f3)
            put(f"adni_k{k_val}_{ptag}_mv_gain", float(row["latent_reliability_multivariate_gain"]), f3)
            put(f"adni_k{k_val}_{ptag}_passed", bool(row["latent_reliability_passed"]), str)
            put(f"adni_k{k_val}_{ptag}_all_coupled", bool(row["all_coupled"]), str)

    ks = _csv("ablation_k_sparsity_sweep.csv")
    raw["k_sweep"] = ks

    q = _csv("ablation_quantile_sparseness.csv")
    raw["quantile"] = q

    with open(os.path.join(CACHE, "adni_mci_feature_loadings.json")) as fh:
        fl = json.load(fh)
    s = fl["summaries"]
    for key in ("pca", "col_pca_simr", "varimax_simr", "row_ica_simr", "col_pca_q85_simr"):
        put(f"fl_{key}_sp", s[key]["dkt_sparsity"], lambda x: pct(x, 0))
        put(f"fl_{key}_active", "[" + ", ".join(str(a) for a in s[key]["dkt_active_regions"]) + "]", str)
        put(f"fl_{key}_r2", s[key]["cog_pc1_r2"])
    q85 = fl["Col-Centered + PCA + Non-Neg + Quantile (q=0.85)"]
    put("fl_q85_D", q85["defect_D"], sci)
    put("fl_q85_sparsity_all", q85["sparsity_all"], lambda x: pct(x, 0))
    raw["fl_q85_components"] = q85["components"]
    names = fl["view2_names"]; q85m = np.asarray(fl["view2_col_pca_q85_simr"])
    for reg in ("left_precentral", "right_precentral", "left_postcentral", "right_postcentral"):
        put(f"fl_q85_{reg}", float(q85m[names.index(reg)].max()), lambda x: f"{x:.3f}")
    sm = [float(q85m[names.index(r)].max()) for r in ("left_precentral", "right_precentral",
                                                       "left_postcentral", "right_postcentral")]
    raw["sensorimotor_max"] = max(sm)
    put("fl_sensorimotor_max", max(sm), lambda x: f"{x:.3f}")
    for j, comp in enumerate(q85["components"], 1):
        top = comp["regions"][:5]
        put(f"fl_q85_c{j}_top", ", ".join(f"`{r}` ({v:+.3f})" for r, v in top) or "none", str)
        put(f"fl_q85_c{j}_n", comp["active_regions"], str)

    # Feature attribution (scripts/paper_adni_attribution.py): which component carries the held-out
    # CogPC1 prediction, why hard-assigned DiReCT loadings collapse onto it, and which regions
    # replicate across 20 half-sample refits with and without the collapse penalty.
    with open(os.path.join(CACHE, "adni_attribution.json")) as fh:
        at = json.load(fh)
    raw["attribution"] = at
    put("attr_full_r2", at["full_r2"])
    for c in at["components"]:
        tag = {"volume": "vol", "vol_area": "va", "direct_thickness": "dkt"}[c["view"]]
        put(f"attr_{tag}_c{c['component']}_r2", c["alone"])
        put(f"attr_{tag}_c{c['component']}_unique", c["drop"], lambda x: f"{x:+.3f}")
        put(f"attr_{tag}_c{c['component']}_n", c["nnz"], str)
    put("attr_dkt_pc1", at["dkt_eigen_share"][0], lambda x: pct(x, 0))
    put("attr_dkt_mean_corr", at["dkt_mean_corr"], lambda x: f"{x:.2f}")
    put("attr_c2_global_r", at["consensus_vs_global"][1], lambda x: f"{x:.2f}")
    put("attr_hard_counts", " / ".join(str(v) for v in at["dkt_loadings_counts"]), str)
    # Component numbers are not identifiable across fits (labels switch between runs and refits), so the
    # prose cites the predictive column of each view and the global-factor component by value, not by index.
    best = {}
    for c in at["components"]:
        tag = {"volume": "vol", "vol_area": "va", "direct_thickness": "dkt"}[c["view"]]
        if tag not in best or c["alone"] > best[tag]["alone"]:
            best[tag] = c
    for tag, c in best.items():
        put(f"attr_{tag}_best_c", c["component"], str)
        put(f"attr_{tag}_best_r2", c["alone"])
        put(f"attr_{tag}_best_unique", c["drop"], lambda x: f"{x:+.3f}")
        put(f"attr_{tag}_best_n", c["nnz"], str)
    others = [c["alone"] for c in at["components"] if all(c is not b for b in best.values())]
    put("attr_other_max_r2", max(others))
    g = int(np.argmax(np.abs(at["consensus_vs_global"])))
    put("attr_global_c", g + 1, str)
    put("attr_global_r", at["consensus_vs_global"][g], lambda x: f"{x:.2f}")
    for tag, key in (("cur", "halves"), ("nc", "halves_no_collapse")):
        hv = at[key]
        nnz = np.array([sum(h["2"]["support"]) for h in hv])
        put(f"attr_{tag}_dkt_med_r2", float(np.median([h["2"]["r2"] for h in hv])))
        put(f"attr_{tag}_dkt_degenerate", int((nnz <= 2).sum()), str)
        put(f"attr_{tag}_dkt_med_n", float(np.median(nnz)), lambda x: f"{x:.0f}")
        put(f"attr_{tag}_va_med_r2", float(np.median([h["1"]["r2"] for h in hv])))
        for v, vt in (("1", "va"), ("2", "dkt")):
            freq = np.mean([h[v]["support"] for h in hv], 0)
            cut = 0.5 if vt == "va" else 0.4
            top = [(at["names"][v][i], freq[i]) for i in np.argsort(-freq, kind="stable") if freq[i] >= cut][:6]
            put(f"attr_{tag}_{vt}_top", "; ".join(f"{region_label(n)}: {f:.2f}" for n, f in top), str)
    put("attr_n_halves", len(at["halves"]), str)
    put("attr_nc_dkt_top_max", float(np.mean([h["2"]["support"] for h in at["halves_no_collapse"]], 0).max()),
        lambda x: f"{x:.2f}")

    ctl = _csv("adni_mci_axis_alignment_controls.csv")
    cm = ctl.groupby(["Model", "Positivity"]).mean(numeric_only=True)
    raw["controls"] = cm
    lin_inv = float((ctl.linear_score - ctl.linear_pca_rotated).abs().max())
    put("ctl_linear_invariance", lin_inv, sci)
    eith = cm.xs("either", level="Positivity")
    pca_rf = float(eith.loc["PCA", "rf_axis_aligned"])
    put("ctl_pca_rf", pca_rf)
    shuffled_drop = (eith.rf_support_shuffled - eith.rf_axis_aligned).drop("PCA")
    rot_drop = (eith.rf_pca_rotated - eith.rf_axis_aligned).drop("PCA")
    raw["ctl_shuffle_drop"] = shuffled_drop; raw["ctl_rot_drop"] = rot_drop
    collapse = shuffled_drop[shuffled_drop < 0].index.tolist()
    put("ctl_shuffle_lowers", ", ".join(f"`{m}` ({eith.loc[m, 'rf_axis_aligned']:.3f} $\\to$ "
                                         f"{eith.loc[m, 'rf_support_shuffled']:.3f})" for m in collapse), str)
    put("ctl_shuffle_n_lower", len(collapse), str)
    put("ctl_n_models", len(shuffled_drop), str)
    put("ctl_rot_n_lower", int((rot_drop < 0).sum()), str)
    ratios = (eith.rf_axis_aligned.drop("PCA") / pca_rf)
    put("ctl_rf_ratio_min", float(ratios.min()), lambda x: f"{x:.1f}")
    put("ctl_rf_ratio_max", float(ratios.max()), lambda x: f"{x:.1f}")

    # ---- v0.43.1 Synthetic Theorems (Propositions 1-3 + 20-seed Design A & B)
    def _pm(mu, sd, fn=f3):
        return f"{fn(mu)} $\\pm$ {fn(sd)}"

    def _delta3(x):
        return f"{x:+.3f}"

    with open(os.path.join(CACHE, "v043_synthetic_theorems.json")) as fh:
        syn = json.load(fh)
    raw["v043_synthetic_theorems"] = syn
    sprov = syn["provenance"]
    put("syn_n_seeds", int(sprov["n_seeds"]), str)
    put("syn_n_train", int(sprov["n_train"]), str)
    put("syn_n_test", int(sprov["n_test"]), str)
    put("syn_view_dims", "(" + ", ".join(str(x) for x in sprov["view_dims"]) + ")", str)

    p1 = syn["propositions"]["prop1"]
    put("syn_prop1_max_linf", float(p1["max_linf_diff_quantile_soft_vs_alpha0"]), sci)
    p1_last = p1["soft_threshold_floor_erosion"][-1]
    put("syn_prop1_soft_cos_a099", float(p1_last["cos_with_alpha0_soft_thresh"]), f3)
    put("syn_prop1_soft_floor_a099", float(p1_last["uniform_floor"]), f4)
    put("syn_prop1_soft_tau_a099", float(p1_last["soft_tau"]), f4)

    p2 = syn["propositions"]["prop2"]
    put("syn_prop2_cs_err", float(p2["cauchy_schwarz_abs_error"]), sci)
    put("syn_prop2_euclid_exact", float(p2["euclid_sq_err_exact_topk"]), lambda x: f"{x:.2f}")
    put("syn_prop2_euclid_soft", float(p2["euclid_sq_err_soft_thresh"]), lambda x: f"{x:.2f}")
    put("syn_prop2_cos_exact", float(p2["cos_sim_exact_topk"]), f3)
    put("syn_prop2_cos_soft", float(p2["cos_sim_soft_thresh"]), f3)
    put("syn_prop2_legacy_theta", float(p2["legacy_convexify_theta_subtracted"]), f4)
    put("syn_prop2_legacy_zeroed_pct", float(p2["legacy_convexify_fraction_zeroed_at_sparseness_none"]),
        lambda x: pct(x, 0))
    put("syn_prop2_legacy_cos", float(p2["legacy_convexify_cos_with_input"]), f3)
    put("syn_prop2_radial_cos", float(p2["radial_convexify_cos_with_input"]), f3)

    p3 = syn["propositions"]["prop3"]
    put("syn_prop3_max_fro", float(p3["max_col_support_fro_error"]), sci)
    p3_last = p3["alpha_sweep"][-1]
    put("syn_prop3_global_fro_a1", float(p3_last["global_smoother_fro_err"]), f3)
    put("syn_prop3_global_cos_a1", float(p3_last["global_smoother_cos"]), f3)

    def _paired_wilcoxon(runs, arm_a, arm_b, metric):
        va = np.asarray([r[arm_a][metric] for r in runs], dtype=float)
        vb = np.asarray([r[arm_b][metric] for r in runs], dtype=float)
        diff = va - vb
        _, pval = stats.wilcoxon(va, vb, zero_method="wilcox", alternative="two-sided")
        return float(np.mean(diff)), float(pval)

    da = syn["design_a_rank1_sparse"]
    da_sum, da_runs = da["summary"], da["per_seed"]
    da_arm_tags = (
        ("legacy_hard_0p5", "legacy_hard"),
        ("legacy_soft_0p3", "legacy_soft"),
        ("legacy_euclidean_convex_0p3", "legacy_ecvx"),
        ("global_shrunk_conical", "global_conical"),
        ("global_shrunk_radial_convex", "global_rcvx"),
        ("twostage_conical_q0p0_ric", "q00_conical_ric"),
        ("twostage_conical_q0p3_ric", "q03_conical_ric"),
        ("twostage_conical_q0p3_plugin", "q03_conical_plug"),
        ("twostage_conical_q0p5_ric", "q05_conical_ric"),
        ("twostage_conical_q0p5_plugin", "q05_conical_plug"),
        ("twostage_radial_convex_q0p3_ric", "q03_rcvx_ric"),
        ("twostage_radial_convex_q0p5_ric", "q05_rcvx_ric"),
        ("twostage_radial_convex_q0p5_plugin", "q05_rcvx_plug"),
    )
    for arm, tag in da_arm_tags:
        s_a = da_sum[arm]
        put(f"ts_a_{tag}_r2", float(s_a["test_r2"]["mean"]), f3)
        put(f"ts_a_{tag}_r2_pm", _pm(s_a["test_r2"]["mean"], s_a["test_r2"]["sd"]), str)
        put(f"ts_a_{tag}_f1", float(s_a["support_f1"]["mean"]), f3)
        put(f"ts_a_{tag}_f1_pm", _pm(s_a["support_f1"]["mean"], s_a["support_f1"]["sd"]), str)
        put(f"ts_a_{tag}_cos", float(s_a["weight_cos"]["mean"]), f3)
        put(f"ts_a_{tag}_act", float(s_a["active_frac"]["mean"]), f3)

    for cname, arm_a, arm_b, metric in (
        ("con_ric_vs_global_r2", "twostage_conical_q0p5_ric", "global_shrunk_conical", "test_r2"),
        ("con_ric_vs_global_f1", "twostage_conical_q0p5_ric", "global_shrunk_conical", "support_f1"),
        ("con_plug_vs_global_r2", "twostage_conical_q0p5_plugin", "global_shrunk_conical", "test_r2"),
        ("con_plug_vs_global_f1", "twostage_conical_q0p5_plugin", "global_shrunk_conical", "support_f1"),
        ("con_plug_vs_soft_r2", "twostage_conical_q0p5_plugin", "legacy_soft_0p3", "test_r2"),
        ("rcvx_ric_vs_global_r2", "twostage_radial_convex_q0p5_ric", "global_shrunk_radial_convex", "test_r2"),
        ("rcvx_ric_vs_global_f1", "twostage_radial_convex_q0p5_ric", "global_shrunk_radial_convex", "support_f1"),
        ("rcvx_plug_vs_ecvx_r2", "twostage_radial_convex_q0p5_plugin", "legacy_euclidean_convex_0p3", "test_r2"),
        ("rcvx_plug_vs_ecvx_f1", "twostage_radial_convex_q0p5_plugin", "legacy_euclidean_convex_0p3", "support_f1"),
    ):
        d_mu, p_val = _paired_wilcoxon(da_runs, arm_a, arm_b, metric)
        put(f"ts_a_delta_{cname}", d_mu, _delta3)
        put(f"ts_a_p_{cname}", p_val, sci)

    db = syn["design_b_rank2_disjoint"]
    db_sum, db_runs = db["summary"], db["per_seed"]
    db_arm_tags = (
        ("legacy_conical_hard_0p5", "legacy_hard"),
        ("legacy_conical_soft_0p3", "legacy_soft"),
        ("legacy_global_shrunk_k2", "legacy_global"),
        ("legacy_euclidean_convex_0p5", "legacy_ecvx"),
        ("twostage_conical_q0p0_ric", "q00_conical_ric"),
        ("twostage_conical_q0p3_ric", "q03_conical_ric"),
        ("twostage_conical_q0p3_plugin", "q03_conical_plug"),
        ("twostage_conical_q0p5_ric", "q05_conical_ric"),
        ("twostage_conical_q0p5_plugin", "q05_conical_plug"),
        ("twostage_radial_convex_q0p0_ric", "q00_rcvx_ric"),
        ("twostage_radial_convex_q0p3_ric", "q03_rcvx_ric"),
        ("twostage_radial_convex_q0p5_ric", "q05_rcvx_ric"),
    )
    for arm, tag in db_arm_tags:
        s_b = db_sum[arm]
        put(f"ts_b_{tag}_r2", float(s_b["test_r2_mean"]["mean"]), f3)
        put(f"ts_b_{tag}_r2_pm", _pm(s_b["test_r2_mean"]["mean"], s_b["test_r2_mean"]["sd"]), str)
        put(f"ts_b_{tag}_f1", float(s_b["support_f1"]["mean"]), f3)
        put(f"ts_b_{tag}_f1_pm", _pm(s_b["support_f1"]["mean"], s_b["support_f1"]["sd"]), str)
        put(f"ts_b_{tag}_fro", float(s_b["max_orth_fro_err"]["mean"]), f4)
        put(f"ts_b_{tag}_cos", float(s_b["max_col_cos"]["mean"]), f4)
        put(f"ts_b_{tag}_ucorr", float(s_b["abs_corr_u1_u2"]["mean"]), f3)

    for cname, arm_a, arm_b, metric in (
        ("con_ric_vs_global_r2", "twostage_conical_q0p5_ric", "legacy_global_shrunk_k2", "test_r2_mean"),
        ("con_ric_vs_global_f1", "twostage_conical_q0p5_ric", "legacy_global_shrunk_k2", "support_f1"),
        ("con_ric_vs_global_fro", "twostage_conical_q0p5_ric", "legacy_global_shrunk_k2", "max_orth_fro_err"),
        ("con_ric_vs_hard_r2", "twostage_conical_q0p5_ric", "legacy_conical_hard_0p5", "test_r2_mean"),
        ("con_ric_vs_hard_f1", "twostage_conical_q0p5_ric", "legacy_conical_hard_0p5", "support_f1"),
        ("rcvx_ric_vs_ecvx_r2", "twostage_radial_convex_q0p5_ric", "legacy_euclidean_convex_0p5", "test_r2_mean"),
        ("rcvx_ric_vs_ecvx_f1", "twostage_radial_convex_q0p5_ric", "legacy_euclidean_convex_0p5", "support_f1"),
    ):
        d_mu, p_val = _paired_wilcoxon(db_runs, arm_a, arm_b, metric)
        put(f"ts_b_delta_{cname}", d_mu, _delta3)
        put(f"ts_b_p_{cname}", p_val, sci)

    # ---- v0.43.1 Real-data revisit Part A & Part B
    pa_raw = _csv("part_a_simlr_v043_raw.csv")
    pa_cmp = _csv("part_a_comparison_summary.csv")
    pa_cmp["Sparseness_Key"] = pa_cmp["Sparseness_Key"].map(
        lambda v: "None" if pd.isna(v) else f"{float(v):.1f}"
    )
    raw["part_a_raw"] = pa_raw
    raw["part_a_summary"] = pa_cmp

    put("v043_n_fits", len(pa_raw), str)
    n_conv_a = int(pa_raw["Solver_Converged"].sum())
    put("v043_n_conv", n_conv_a, str)
    put("v043_conv_pct", float(n_conv_a / len(pa_raw)), pct)
    matched_a = pa_cmp[pa_cmp["Cached_Metric"].notna()]
    put("v043_n_matched_cells", len(matched_a), str)
    put("v043_n_matched", len(matched_a) * int(pa_raw["Seed"].nunique()), str)

    eith_m = matched_a[matched_a["Positivity"] == "either"]
    pos_m = matched_a[matched_a["Positivity"] == "positive"]
    put("v043_mean_either_delta", float(eith_m["Delta_Metric"].mean()), _delta3)
    put("v043_either_q05_mean_delta", float(eith_m[eith_m["Sparseness_Key"] == "0.5"]["Delta_Metric"].mean()), _delta3)
    put("v043_either_q08_mean_delta", float(eith_m[eith_m["Sparseness_Key"] == "0.8"]["Delta_Metric"].mean()), _delta3)
    put("v043_max_pos_delta", float(pos_m["Delta_Metric"].max()), _delta3)
    put("v043_pos_max_abs_delta", float(pos_m["Delta_Metric"].abs().max()), f3)

    ds_tags = (("ADNI-MCI", "adni"), ("MultiOmics", "brca"), ("TCGA-KIRC", "kirc"),
               ("Heart", "heart"), ("Diabetes", "diab"), ("mfeat", "mfeat"))
    pos_tags = (("either", "either"), ("positive", "pos"), ("convex", "cvx"))
    for ds_full, ds_key in ds_tags:
        for pos_full, ptag in pos_tags:
            sub_raw = pa_raw[(pa_raw["Dataset"] == ds_full) & (pa_raw["Positivity"] == pos_full)]
            if sub_raw.empty:
                continue
            put(f"v043_{ds_key}_{ptag}_mean_u", float(sub_raw["Predictive_Metric"].mean()), f3)
            put(f"v043_{ds_key}_{ptag}_mean_lin", float(sub_raw["Strictly_Linear_Metric"].mean()), f3)
            sub_cmp = pa_cmp[(pa_cmp["Dataset"] == ds_full) & (pa_cmp["Positivity"] == pos_full)]
            put(f"v043_{ds_key}_{ptag}_best_u", float(sub_cmp["V043_Metric"].max()), f3)
            for sp_str, sp_key, sp_raw in (("0.5", "q05", 0.5), ("0.8", "q08", 0.8), ("None", "qnone", None)):
                row_c = sub_cmp[sub_cmp["Sparseness_Key"] == sp_str]
                if not row_c.empty:
                    rc = row_c.iloc[0]
                    put(f"v043_{ds_key}_{ptag}_{sp_key}_u", float(rc["V043_Metric"]), f3)
                    put(f"v043_{ds_key}_{ptag}_{sp_key}_sp", float(rc["V043_Sparsity"]), pct)
                    put(f"v043_{ds_key}_{ptag}_{sp_key}_D", float(rc["V043_Defect"]), f4)
                    if pd.notna(rc["Delta_Metric"]):
                        put(f"v043_{ds_key}_{ptag}_{sp_key}_cached", float(rc["Cached_Metric"]), f3)
                        put(f"v043_{ds_key}_{ptag}_{sp_key}_delta", float(rc["Delta_Metric"]), _delta3)
                sub_sp = sub_raw[sub_raw["Sparseness_Target"].isna()] if sp_raw is None else sub_raw[np.isclose(sub_raw["Sparseness_Target"].fillna(-1.0), sp_raw)]
                if not sub_sp.empty:
                    put(f"v043_{ds_key}_{ptag}_{sp_key}_lin", float(sub_sp["Strictly_Linear_Metric"].mean()), f3)

    pb_raw = _csv("part_b_uniform_shrunk_raw.csv")
    pb_sum = _csv("part_b_uniform_shrunk_summary.csv")
    raw["part_b_raw"] = pb_raw
    raw["part_b_summary"] = pb_sum

    put("us_n_fits", len(pb_raw), str)
    pb_con = pb_raw[pb_raw["Positivity"] == "conical"]
    pb_cvx = pb_raw[pb_raw["Positivity"] == "convex"]
    put("us_n_conical_fits", len(pb_con), str)
    put("us_n_conical_conv", int(pb_con["Solver_Converged"].sum()), str)
    put("us_n_convex_fits", len(pb_cvx), str)
    put("us_n_convex_conv", int(pb_cvx["Solver_Converged"].sum()), str)
    put("us_conical_max_defect", float(pb_con["Frame_Defect_D"].max()), sci)
    put("us_conical_max_cos", float(pb_con["Max_Col_Cosine"].max()), f3)
    put("us_convex_max_cos", float(pb_cvx["Max_Col_Cosine"].max()), f3)
    put("us_convex_max_defect", float(pb_cvx["Frame_Defect_D"].max()), f3)

    for pos_full, ptag in (("conical", "con"), ("convex", "cvx")):
        for k_val in (1, 2, 3):
            sub_k = pb_sum[(pb_sum["Positivity"] == pos_full) & (pb_sum["k"] == k_val)]
            if not sub_k.empty:
                put(f"us_{ptag}_k{k_val}_u", float(sub_k["US_Metric"].mean()), f3)
                put(f"us_{ptag}_k{k_val}_lin", float(sub_k["US_Lin"].mean()), f3)
                put(f"us_{ptag}_k{k_val}_defect", float(sub_k["US_Defect"].mean()), sci if pos_full == "conical" else f3)
                put(f"us_{ptag}_k{k_val}_alpha", float(sub_k["US_Alpha"].mean()), f3)

    target_k = {"Heart": 2, "Diabetes": 2, "ADNI-MCI": 3, "MultiOmics": 3, "TCGA-KIRC": 3, "mfeat": 3}
    for ds_full, ds_key in ds_tags:
        for pos_full, ptag in (("conical", "con"), ("convex", "cvx")):
            sub_dp = pb_sum[(pb_sum["Dataset"] == ds_full) & (pb_sum["Positivity"] == pos_full)]
            for _, r in sub_dp.iterrows():
                kv = int(r["k"])
                put(f"us_{ds_key}_{ptag}_k{kv}_u", float(r["US_Metric"]), f3)
                put(f"us_{ds_key}_{ptag}_k{kv}_lin", float(r["US_Lin"]), f3)
                put(f"us_{ds_key}_{ptag}_k{kv}_defect", float(r["US_Defect"]), sci if pos_full == "conical" else f3)
                put(f"us_{ds_key}_{ptag}_k{kv}_cos", float(r["US_MaxCos"]), f3)
                put(f"us_{ds_key}_{ptag}_k{kv}_alpha", float(r["US_Alpha"]), f3)
                if kv == target_k[ds_full]:
                    put(f"us_{ds_key}_{ptag}_u", float(r["US_Metric"]), f3)
                    put(f"us_{ds_key}_{ptag}_lin", float(r["US_Lin"]), f3)
                    put(f"us_{ds_key}_{ptag}_defect", float(r["US_Defect"]), sci if pos_full == "conical" else f3)
                    put(f"us_{ds_key}_{ptag}_cos", float(r["US_MaxCos"]), f3)
                    put(f"us_{ds_key}_{ptag}_alpha", float(r["US_Alpha"]), f3)

    # ---- Audit evidence tables (docs/audit/evidence/*.csv)
    ev_dir = os.path.join(HERE, "..", "docs", "audit", "evidence")
    s1 = pd.read_csv(os.path.join(ev_dir, "default_selection_stage1.csv"))
    s2 = pd.read_csv(os.path.join(ev_dir, "default_selection_stage2.csv"))
    s2_win = s2[(s2.energy == "nc") & (s2.optimizer == "hybrid_adam") & (s2.mixing == "newton")]
    put("default_stage1_n", f"{len(s1):,}", str)
    put("default_stage2_n", f"{len(s2):,}", str)
    put("default_stage2_ncells", len(s2_win), str)
    put("default_stage2_conv_n", int(s2_win.converged.sum()), str)
    put("default_stage2_conv_rate", float(s2_win.converged.mean()), f3)
    put("default_stage2_latent_r2", float(s2_win.latent_r2.mean()), f4)
    put("default_stage2_worst_r2", float(s2_win.latent_r2.min()), f4)
    put("default_stage2_support", float(s2_win.loadings_support.mean()), f4)

    vw = pd.read_csv(os.path.join(ev_dir, "view_weighting_acc.csv"))
    vw_dc = vw[vw.group == "default_cells"]
    put("view_weighting_acc_max_latent_loss", float(-vw_dc[vw_dc.metric == "latent_r2"]["difference"].min()), f4)
    put("view_weighting_acc_max_support_loss", float(-vw_dc[vw_dc.metric == "loadings_support_f1"]["difference"].min()), f4)

    ct = pd.read_csv(os.path.join(ev_dir, "collapse_threshold_selection.csv"))
    put("collapse_t07_narrow_corr", float(ct[(ct.group == "narrow_fixture") & (ct.threshold == 0.7)]["score_corr"].abs().max()),
        lambda x: f"{x:.2f}")
    r15_07 = ct[(ct.group == "r15_evaluation") & (ct.case_id == "r15_non_inf_narrow_0.7")].iloc[0]
    r15_diff = float(r15_07["difference"])
    r15_val = float(r15_07["outcome_r2"])
    put("collapse_t07_narrow_r2_drop", r15_diff, lambda x: f"{x:.2f}")
    put("collapse_t07_narrow_r2_val", r15_val, f4)
    put("collapse_t07_narrow_r2_base", r15_val - r15_diff, f4)
    put("collapse_t03_avg_rho05", float(ct[(ct.group == "corr_case_avg") & (ct.threshold == 0.3) & (ct.rho == 0.5)]["score_corr"].max()), f3)
    put("collapse_t03_avg_rho07", float(ct[(ct.group == "corr_case_avg") & (ct.threshold == 0.3) & (ct.rho == 0.7)]["score_corr"].max()), f3)
    put("collapse_t03_newton_max", float(ct[(ct.group == "corr_case_newton") & (ct.threshold == 0.3)]["score_corr"].max()), f3)

    dc = pd.read_csv(os.path.join(ev_dir, "deep_collapse_diagnosis.csv"))
    dc_s3 = dc[(dc.group == "step3_energy") & (dc.penalty == "none")]
    dc_c = dc_s3[dc_s3.collapsed == True]  # noqa: E712
    dc_ok = dc_s3[dc_s3.collapsed == False]  # noqa: E712
    put("deep_collapse_n_collapsed", len(dc_c), str)
    put("deep_collapse_loss_collapsed", float(dc_c["sim_loss"].mean()), lambda x: f"{x:.2f}")
    put("deep_collapse_n_ok", len(dc_ok), str)
    put("deep_collapse_loss_ok", float(dc_ok["sim_loss"].mean()), lambda x: f"{x:.2f}")

    dcons = pd.read_csv(os.path.join(ev_dir, "deep_default_consistency.csv"))
    put("deep_cons_n_per_family", int(dcons.groupby("Method")["converged"].count().iloc[0]), str)
    for m_id, m_tag in (("flow_v", "flow_v"), ("lend", "lend"), ("ned", "ned"), ("ned_shared_private", "nedpp")):
        sub_m = dcons[dcons.Method == m_id]
        if not sub_m.empty:
            put(f"deep_cons_conv_{m_tag}", float(sub_m["converged"].mean()), f3)

    return SimpleNamespace(**n), raw


def _md(v):
    from IPython.display import Markdown
    return Markdown(str(v))


_n, RAW = build()
#: Preformatted Markdown values for inline use (`{python} N.key`); Quarto
#: renders a Markdown object as Markdown, so math and code spans survive.
N = SimpleNamespace(**{k.replace("-", "_").replace("+", "p"): _md(v) for k, v in vars(_n).items()})
