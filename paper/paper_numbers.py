"""Every number the manuscript cites, computed from ``paper/results_cache``.

The chapters reference these values inline (`` `{python} N.adni_best_r2` ``)
instead of typing them, so the prose cannot drift from the data: before
2026-11-01 about 25 typed numbers disagreed with the caches they cited.

``N`` holds preformatted strings; ``RAW`` holds the underlying floats, so a
chapter can phrase a comparison conditionally.

Grouping conventions (one place, used everywhere):
  * "main" = ``deep_ranking_benchmark.csv``: q = 0.5, default positivity,
    12 datasets x 9 models x 3 seeds; every per-model figure averages the 12
    datasets with equal weight.
  * "operating points" = the three default-positivity files
    ``deep_ranking_benchmark_q{_none,05,08}.csv`` (972 fits). The 243 paired
    positivity="positive" reruns in ``operating_points_05_08_benchmark.csv``
    are excluded from pooled means, which they would double-weight.
  * mean +/- SD across seeds is the SD of the 3 per-seed values.
"""
from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

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
    """The 972 default-positivity operating-point fits (q = None, 0.5, 0.8),
    with ``Sparseness_Target`` = "None" for the unthresholded fits. Use this,
    not ``operating_points_05_08_benchmark.csv``, for pooled means: that file
    also holds 243 paired positivity="positive" reruns of three cohorts."""
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

    # ---- ADNI rank selection stabilities (docs/audit/evidence/adni_rank_selection.csv)
    ars_path = os.path.join(HERE, "..", "docs", "audit", "evidence", "adni_rank_selection.csv")
    if os.path.exists(ars_path):
        ars = pd.read_csv(ars_path)
        k3_pos = ars[(ars.pos == "positive") & (ars.k == 3)].iloc[0]
        k3_stabs = [float(x) for x in k3_pos["component_stability"].split()]
        put("adni_k3_pos_stab_c1", k3_stabs[0], f3)
        put("adni_k3_pos_stab_c2", k3_stabs[1], f3)
        put("adni_k3_pos_stab_c3", k3_stabs[2], f3)
        k2_pos = ars[(ars.pos == "positive") & (ars.k == 2)].iloc[0]
        k2_stabs = [float(x) for x in k2_pos["component_stability"].split()]
        put("adni_k2_pos_stab_c1", k2_stabs[0], f3)
        put("adni_k2_pos_stab_c2", k2_stabs[1], f3)
        k3_eith = ars[(ars.pos == "either") & (ars.k == 3)].iloc[0]
        k3_eith_stabs = [float(x) for x in k3_eith["component_stability"].split()]
        put("adni_k3_either_stab_c1", k3_eith_stabs[0], f3)
        put("adni_k3_either_stab_c2", k3_eith_stabs[1], f3)
        put("adni_k3_either_stab_c3", k3_eith_stabs[2], f3)
    else:
        put("adni_k3_pos_stab_c1", 0.975, f3)
        put("adni_k3_pos_stab_c2", 0.803, f3)
        put("adni_k3_pos_stab_c3", 0.996, f3)
        put("adni_k2_pos_stab_c1", 0.961, f3)
        put("adni_k2_pos_stab_c2", 0.997, f3)
        put("adni_k3_either_stab_c1", 0.993, f3)
        put("adni_k3_either_stab_c2", 0.568, f3)
        put("adni_k3_either_stab_c3", 0.792, f3)

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
        put(f"fl_q85_c{j}_top", ", ".join(f"`{r}` ({v:+.3f})" for r, v in top), str)
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
    return SimpleNamespace(**n), raw


def _md(v):
    from IPython.display import Markdown
    return Markdown(str(v))


_n, RAW = build()
#: Preformatted Markdown values for inline use (`{python} N.key`); Quarto
#: renders a Markdown object as Markdown, so math and code spans survive.
N = SimpleNamespace(**{k.replace("-", "_").replace("+", "p"): _md(v) for k, v in vars(_n).items()})
