#!/usr/bin/env python3
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

sys.path.insert(0, "/Users/stnava/data/repos/pysimlr/src")
sys.path.insert(0, "/Users/stnava/data/repos/nsa_flow")

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA as SkPCA
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from pysimlr.benchmarks.runner import run_single_experiment

warnings.filterwarnings("ignore")
torch.set_num_threads(1)

OUT_CSV = os.path.join(os.path.dirname(__file__), "..", "paper", "results_cache", "adni_mci_1214_grid_sweep.csv")


def build_adni_case_11():
    csv_path = "/Users/stnava/data/repos/pysimlr/data/adni/adni_mci_cognitive_merged.csv"
    df = pd.read_csv(csv_path)
    vol_cols = [c for c in df.columns if c.startswith("T1Hier_vol__")]
    thk_cols = [c for c in df.columns if c.startswith("T1Hier_thk__")]
    t1w_cols = [c for c in df.columns if c.startswith("T1w__")]

    drop_cols = {"Cog__MOCA", "Cog__EcogPtTotal", "Cog__EcogSPTotal"}
    cog11_cols = [c for c in df.columns if c.startswith("Cog__") and c not in drop_cols] + ["CDRSB"]

    n_samples = len(df)
    sex_bin = (df["PTGENDER"].astype(str).str.upper().str.startswith("F")).astype(float).values
    covars = np.column_stack([np.ones(n_samples), df["AGE"].values, sex_bin, df["PTEDUCAT"].values]).astype(np.float32)
    y_cog11 = df[cog11_cols].values.astype(np.float32)

    mats = [
        df[vol_cols].values.astype(np.float32),
        df[thk_cols].values.astype(np.float32),
        df[t1w_cols].values.astype(np.float32),
        y_cog11,
    ]
    k = 3
    case = {
        "kind": "adni_mci_real",
        "data": [torch.tensor(m).float() for m in mats],
        "outcome": torch.tensor(y_cog11).float(),
        "covariates": torch.tensor(covars).float(),
        "eval_view_indices": [0, 1, 2],
        "topology": "graph",
        "path_graph": {0: [3], 1: [3], 2: [3], 3: [0, 1, 2]},
        "true_u": torch.zeros(n_samples, k),
        "true_v": [np.zeros((m.shape[1], k)) for m in mats],
        "shared_k": k,
        "is_classification": False,
        "needs_scaling": True,
    }

    rng = np.random.default_rng(42)
    perm = rng.permutation(n_samples)
    split = int(0.7 * n_samples)
    train_idx, test_idx = perm[:split], perm[split:]

    c_train, c_test = covars[train_idx], covars[test_idx]
    pinv_c = np.linalg.pinv(c_train)
    x_train_list, x_test_list = [], []
    for m in mats[:3]:
        b = pinv_c @ m[train_idx]
        tr = m[train_idx] - c_train @ b
        te = m[test_idx] - c_test @ b
        sc = StandardScaler()
        x_train_list.append(sc.fit_transform(tr))
        x_test_list.append(sc.transform(te))

    by = pinv_c @ y_cog11[train_idx]
    y_tr = y_cog11[train_idx] - c_train @ by
    y_te = y_cog11[test_idx] - c_test @ by
    scy = StandardScaler()
    y_tr = scy.fit_transform(y_tr)
    y_te = scy.transform(y_te)

    pca_y = SkPCA(n_components=1).fit(y_tr)
    pc1_tr = pca_y.transform(y_tr).ravel()
    pc1_te = pca_y.transform(y_te).ravel()

    return case, x_train_list, x_test_list, y_tr, y_te, pc1_tr, pc1_te


_GLOBAL_DATA = None


def _init_worker():
    global _GLOBAL_DATA
    torch.set_num_threads(1)
    _GLOBAL_DATA = build_adni_case_11()


def eval_task(task):
    global _GLOBAL_DATA
    if _GLOBAL_DATA is None:
        _GLOBAL_DATA = build_adni_case_11()
    case, x_train_list, x_test_list, y_tr, y_te, pc1_tr, pc1_te = _GLOBAL_DATA

    family, model_type, model_label, opt_type, energy_type, mixing_alg, mai_label = task

    params = {
        "energy_type": energy_type,
        "mixing_algorithm": mixing_alg,
        "positivity": "either",
        "nsa_w": 0.5,
        "nsa_iterations": 2,
        "consolidate": False,
    }
    if opt_type is not None:
        params["optimizer_type"] = opt_type

    if mai_label == "off":
        params["dynamic_weights"] = False
    elif mai_label == "rank_procrustes_r2":
        params["dynamic_weights"] = True
        params["mai_metric"] = "procrustes_r2"
        params["use_rank_mai"] = True
    else:
        params["dynamic_weights"] = True
        params["mai_metric"] = mai_label
        params["use_rank_mai"] = False

    if model_type in ("linear", "simlr_lbfgs", "simlr_lbfgsb"):
        params["iterations"] = 25
    elif model_type in ("lend", "ned", "shared_private", "flow_v"):
        params["epochs"] = 80

    t0 = time.perf_counter()
    try:
        exp_res = run_single_experiment(model_type, case, seed=42, **params)
        elapsed = time.perf_counter() - t0
        metrics = exp_res["metrics"]
        vs = [np.asarray(v.detach().cpu() if hasattr(v, "detach") else v) for v in exp_res["result"]["v"]]

        z_tr = np.hstack([x_train_list[i] @ vs[i] for i in range(3)])
        z_te = np.hstack([x_test_list[i] @ vs[i] for i in range(3)])

        rcv = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(z_tr, y_tr)
        r2_rcv = float(r2_score(y_te, rcv.predict(z_te), multioutput="uniform_average"))

        ols_pc1 = LinearRegression().fit(z_tr, pc1_tr)
        r2_pc1 = float(r2_score(pc1_te, ols_pc1.predict(z_te)))

        return {
            "Family": family,
            "Model": model_label,
            "Optimizer": opt_type or "default",
            "Similarity": energy_type,
            "Mixing": mixing_alg,
            "MAI": mai_label,
            "Mean11_OLS": float(metrics["first_layer_test_r2"]),
            "Mean11_RF": float(metrics["first_layer_axis_test_r2"]),
            "Mean11_RidgeCV": r2_rcv,
            "CogPC1_R2": r2_pc1,
            "Runtime_s": elapsed,
            "Status": "ok",
        }
    except Exception as exc:
        return {
            "Family": family,
            "Model": model_label,
            "Optimizer": opt_type or "default",
            "Similarity": energy_type,
            "Mixing": mixing_alg,
            "MAI": mai_label,
            "Mean11_OLS": float("nan"),
            "Mean11_RF": float("nan"),
            "Mean11_RidgeCV": float("nan"),
            "CogPC1_R2": float("nan"),
            "Runtime_s": time.perf_counter() - t0,
            "Status": f"err:{type(exc).__name__}:{exc}",
        }


def build_task_list():
    tasks = []
    seen = set()

    def add(family, model_type, model_label, opt_type, energy, mixing, mai):
        key = (model_label, opt_type, energy, mixing, mai)
        if key not in seen:
            seen.add(key)
            tasks.append((family, model_type, model_label, opt_type, energy, mixing, mai))

    # Baselines first
    add("Baseline", "pca", "PCA", "svd", "none", "none", "off")
    add("Baseline", "nsa_pipeline", "NSAFlow-PCA", "nsa", "none", "none", "off")

    mai_opts = ["off", "procrustes_r2", "procrustes_r2_sharp", "cca", "rvcoef", "rank_procrustes_r2"]
    mixing_opts = ["svd", "newton", "pca", "avg", "ica"]
    deep_energies = ["acc", "align", "procrustes", "nc", "logcosh", "exp"]
    lin_energies = ["acc", "align", "procrustes", "nc", "recon_r2", "logcosh", "exp"]

    deep_models = [
        ("Deep", "ned", "NED", "adam"),
        ("Deep", "shared_private", "NEDPP", "adam"),
        ("Deep", "lend", "LEND", "adam"),
        ("Flow", "flow_v", "Flow-SiMLR-V", "adam"),
    ]
    lin_models = [
        ("Linear", "linear", "SiMLR", "armijo_gradient"),
        ("Linear", "linear", "SiMLR-BiArmijo", "bidirectional_armijo_gradient"),
        ("Linear", "simlr_lbfgs", "SiMLR-LBFGS", "torch_lbfgs"),
        ("Linear", "simlr_lbfgsb", "SiMLR-LBFGSB", "nsa_lbfgsb"),
        ("Linear", "linear", "SiMLR-HybridAdam", "hybrid_adam"),
        ("Linear", "linear", "SiMLR-LARS", "lars"),
        ("Linear", "linear", "SiMLR-NSAFlow", "nsa_flow"),
        ("Linear", "linear", "SiMLR-Lookahead", "lookahead"),
    ]

    # WAVE 0 (First 45s): Fast balanced probe across Deep, Flow, and Linear for immediate 1-minute report
    for fam, mt, ml, opt in deep_models:
        for e in ["acc", "align", "procrustes"]:
            add(fam, mt, ml, opt, e, "svd", "off")
            add(fam, mt, ml, opt, e, "svd", "procrustes_r2")
            add(fam, mt, ml, opt, e, "newton", "off")
            add(fam, mt, ml, opt, e, "avg", "off")
            add(fam, mt, ml, opt, e, "svd", "cca")
            add(fam, mt, ml, opt, e, "svd", "rvcoef")

    for fam, mt, ml, opt in lin_models:
        for e in ["acc", "align", "procrustes", "recon_r2", "logcosh"]:
            add(fam, mt, ml, opt, e, "svd", "off")
            add(fam, mt, ml, opt, e, "newton", "off")
            add(fam, mt, ml, opt, e, "svd", "procrustes_r2")
            add(fam, mt, ml, opt, e, "svd", "cca")

    # WAVE 1A: Core Deep/Flow + Linear sweep across MAI, Mixing, Similarity, Optimizer
    for fam, mt, ml, opt in deep_models:
        for e in ["acc", "align", "procrustes"]:
            for mix in ["svd", "newton", "avg", "pca"]:
                for mai in ["off", "procrustes_r2", "cca", "rvcoef"]:
                    add(fam, mt, ml, opt, e, mix, mai)

    for fam, mt, ml, opt in lin_models:
        for e in lin_energies:
            for mix in ["svd", "newton", "pca", "avg", "ica"]:
                add(fam, mt, ml, opt, e, mix, "off")
            for mai in ["procrustes_r2", "procrustes_r2_sharp", "cca", "rvcoef"]:
                add(fam, mt, ml, opt, e, "svd", mai)

    # WAVE 1B: Remaining Deep & Flow MAI x Mixing
    for fam, mt, ml, opt in deep_models:
        for e in ["acc", "align", "procrustes"]:
            for mix in mixing_opts:
                for mai in mai_opts:
                    add(fam, mt, ml, opt, e, mix, mai)

    # WAVE 2: Deep Models with LARS / LARSLOW optimizers + Remaining Deep Energies
    for fam, mt, ml, _ in deep_models[:3]:
        for opt in ["lars", "larslow"]:
            for e in ["acc", "align", "procrustes"]:
                for mix in ["svd", "newton"]:
                    for mai in ["off", "procrustes_r2", "cca"]:
                        add(fam, mt, f"{ml}-{opt.upper()}", opt, e, mix, mai)

    for fam, mt, ml, opt in deep_models:
        for e in deep_energies:
            for mix in mixing_opts:
                for mai in ["off", "procrustes_r2", "cca", "rvcoef"]:
                    add(fam, mt, ml, opt, e, mix, mai)

    return tasks


def main():
    tasks = build_task_list()
    print(f"Total combinations queued: {len(tasks)}", flush=True)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    if os.path.exists(OUT_CSV):
        os.remove(OUT_CSV)

    completed = 0
    t_start = time.perf_counter()
    rows = []

    with ProcessPoolExecutor(max_workers=8, initializer=_init_worker) as ex:
        # Submit in priority waves of 32 so Wave 1 finishes first
        batch_size = 32
        for b_start in range(0, len(tasks), batch_size):
            batch = tasks[b_start:b_start + batch_size]
            futs = {ex.submit(eval_task, t): t for t in batch}
            for fut in as_completed(futs):
                res = fut.result()
                rows.append(res)
                completed += 1
                df_curr = pd.DataFrame(rows)
                df_curr.to_csv(OUT_CSV, index=False)
                if completed % 10 == 0 or completed == len(tasks):
                    ok_df = df_curr[df_curr["Status"] == "ok"].sort_values("Mean11_OLS", ascending=False)
                    best = ok_df.iloc[0] if len(ok_df) > 0 else None
                    best_str = (
                        f"{best['Model']} ({best['Optimizer']}/{best['Similarity']}/{best['Mixing']}/MAI={best['MAI']}): "
                        f"OLS={best['Mean11_OLS']:.4f}, CogPC1={best['CogPC1_R2']:.4f}"
                        if best is not None else "none"
                    )
                    print(
                        f"[{completed:3d}/{len(tasks)}] ({time.perf_counter()-t_start:5.1f}s) | Best so far -> {best_str}",
                        flush=True,
                    )

    print("SWEEP COMPLETE.", flush=True)


if __name__ == "__main__":
    main()
