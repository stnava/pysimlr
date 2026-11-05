"""Empirical validation of the PySiMLR Confound Sink (`confounds`).

Evaluates two complementary regimes:
1. Controlled synthetic 3-view benchmark with a 3.5x signed scanner/motion dipole
   under `positivity="positive"` (both orthogonal and partially covarying biology),
   comparing unadjusted `simlr`, naive pre-residualization (with and without test-time
   `C_test`), and `simlr(..., confounds=...)` at `confound_strength` in {0.7, 1.0}.
2. Real 4-cohort blast neuroimaging dataset (`N = 246` complete 3-modality subjects
   across `breacher`, `ExpArt`, `FPA`, `SOCOM`), separating regional T1w cortical
   thickness, DTI white-matter tract FA, and rsfMRI regional ALFF/fALFF from
   multi-cohort scanner indicators, head motion, eddy shift, ICV, and blind-QC metrics.

Outputs:
- `docs/audit/evidence/confound_sink_benchmark.json`
- `docs/audit/evidence/confound_sink_benchmark.csv`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

import pysimlr

DEFAULT_BLAST_CSV = Path(
    "/Users/stnava/data/blast_cohorts/antsxbrainmappingprivate_results/"
    "blast_cohorts_clean_idps_with_demog.csv"
)
REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = REPO_ROOT / "docs" / "audit" / "evidence" / "confound_sink_benchmark.json"
OUT_CSV = REPO_ROOT / "docs" / "audit" / "evidence" / "confound_sink_benchmark.csv"


def _safe_abs_corr_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise |Pearson r| between columns of `a` (n, p) and `b` (n, q)."""
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    if a64.ndim == 1:
        a64 = a64[:, None]
    if b64.ndim == 1:
        b64 = b64[:, None]
    ac = a64 - a64.mean(axis=0, keepdims=True)
    bc = b64 - b64.mean(axis=0, keepdims=True)
    an = np.linalg.norm(ac, axis=0, keepdims=True)
    bn = np.linalg.norm(bc, axis=0, keepdims=True)
    denom = np.maximum(an.T @ bn, 1e-12)
    return np.abs((ac.T @ bc) / denom)


def _impute_and_augment_design(c_train: np.ndarray, c_eval: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray | None]:
    """Column-mean impute confounds using training statistics and prepend intercept."""
    ctr = np.asarray(c_train, dtype=np.float64).copy()
    if ctr.ndim == 1:
        ctr = ctr[:, None]
    col_means = np.nanmean(ctr, axis=0)
    col_means = np.where(np.isfinite(col_means), col_means, 0.0)
    nan_tr = ~np.isfinite(ctr)
    if np.any(nan_tr):
        ctr[nan_tr] = np.take(col_means, np.where(nan_tr)[1])
    d_tr = np.column_stack([np.ones(ctr.shape[0], dtype=np.float64), ctr])
    if c_eval is None:
        return d_tr, None
    cev = np.asarray(c_eval, dtype=np.float64).copy()
    if cev.ndim == 1:
        cev = cev[:, None]
    nan_ev = ~np.isfinite(cev)
    if np.any(nan_ev):
        cev[nan_ev] = np.take(col_means, np.where(nan_ev)[1])
    d_ev = np.column_stack([np.ones(cev.shape[0], dtype=np.float64), cev])
    return d_tr, d_ev


def _ols_residualize_views(
    train_views: Sequence[np.ndarray],
    test_views: Sequence[np.ndarray],
    c_train: np.ndarray,
    c_test_oracle: np.ndarray | None = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Regress `c_train` out of `train_views` (leaving feature mean intact so scale works)."""
    d_tr, _ = _impute_and_augment_design(c_train, None)
    # Zero-mean covariates only so baseline feature offsets remain intact
    c_zero_tr = d_tr[:, 1:] - d_tr[:, 1:].mean(axis=0, keepdims=True)
    pinv_tr = np.linalg.pinv(c_zero_tr, rcond=1e-10)
    tr_out: List[np.ndarray] = []
    te_out: List[np.ndarray] = []
    for x_tr, x_te in zip(train_views, test_views):
        beta = pinv_tr @ x_tr
        tr_out.append(x_tr - c_zero_tr @ beta)
        if c_test_oracle is not None and x_te.shape[0] > c_zero_tr.shape[1] + 1:
            _, d_te = _impute_and_augment_design(c_train, c_test_oracle)
            c_zero_te = d_te[:, 1:] - d_te[:, 1:].mean(axis=0, keepdims=True)
            pinv_te = np.linalg.pinv(c_zero_te, rcond=1e-10)
            te_out.append(x_te - c_zero_te @ (pinv_te @ x_te))
        else:
            te_out.append(x_te.copy())
    return tr_out, te_out


def _pct_negative_v(res: Dict[str, Any]) -> float:
    total = 0
    neg = 0
    for vm in res["v"]:
        arr = vm.detach().cpu().numpy() if hasattr(vm, "detach") else np.asarray(vm)
        total += arr.size
        neg += int(np.sum(arr < -1e-7))
    return 100.0 * float(neg) / float(max(total, 1))


def _single_subject_discrepancy(res: Dict[str, Any], test_views: Sequence[np.ndarray], n_check: int = 12) -> float:
    pred_batch = pysimlr.predict(res, list(test_views))["u"].detach().cpu().numpy()
    diffs: List[float] = []
    limit = min(n_check, test_views[0].shape[0])
    for idx in range(limit):
        single = [v[idx : idx + 1] for v in test_views]
        u_one = pysimlr.predict(res, single)["u"].detach().cpu().numpy()[0]
        diffs.append(float(np.max(np.abs(u_one - pred_batch[idx]))))
    return float(max(diffs)) if diffs else 0.0


def _subspace_multiple_r(predictors: np.ndarray, targets: np.ndarray) -> float:
    """Mean multiple |R| of `predictors` explaining each column of `targets` (basis-rotation invariant)."""
    x = np.asarray(predictors, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    xc = x - x.mean(axis=0, keepdims=True)
    yc = y - y.mean(axis=0, keepdims=True)
    yhat = xc @ (np.linalg.pinv(xc, rcond=1e-10) @ yc)
    rs = [_safe_abs_corr_matrix(yhat[:, j], yc[:, j])[0, 0] for j in range(yc.shape[1])]
    return float(np.mean(rs))


def _generate_synthetic_dipole_split(
    seed: int,
    n_train: int = 160,
    n_test: int = 80,
    bio_confound_corr: float = 0.0,
    dipole_scale: float = 3.5,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    n_tot = n_train + n_test
    c_raw = rng.standard_normal((n_tot, 2))
    c_raw = (c_raw - c_raw.mean(axis=0, keepdims=True)) / c_raw.std(axis=0, keepdims=True)

    # Biological factors: either orthogonal to C on the full draw or partially covarying
    z_indep = rng.standard_normal((n_tot, 2))
    if abs(bio_confound_corr) < 1e-12:
        q_c, _ = np.linalg.qr(c_raw)
        z_indep = z_indep - q_c @ (q_c.T @ z_indep)
        q_bio, _ = np.linalg.qr(z_indep)
        bio = q_bio * np.sqrt(float(n_tot))
    else:
        rho = float(bio_confound_corr)
        z0 = rho * c_raw[:, 0] + np.sqrt(max(1.0 - rho * rho, 0.0)) * z_indep[:, 0]
        z1 = z_indep[:, 1] - (np.dot(z0, z_indep[:, 1]) / max(np.dot(z0, z0), 1e-12)) * z0
        bio = np.column_stack([
            (z0 - z0.mean()) / z0.std(),
            (z1 - z1.mean()) / z1.std(),
        ])

    widths = (40, 40, 40)
    views_tr: List[np.ndarray] = []
    views_te: List[np.ndarray] = []
    for p in widths:
        w_bio = np.zeros((p, 2), dtype=np.float64)
        w_bio[0:5, 0] = rng.uniform(0.9, 1.2, size=5)
        w_bio[5:10, 1] = rng.uniform(0.9, 1.2, size=5)

        # Two dominant signed scanner/motion dipoles across broader sensor channels
        w_conf = np.zeros((p, 2), dtype=np.float64)
        w_conf[10:17, 0] = +dipole_scale * rng.uniform(0.9, 1.15, size=7)
        w_conf[17:24, 0] = -dipole_scale * rng.uniform(0.9, 1.15, size=7)
        w_conf[24:31, 1] = +dipole_scale * rng.uniform(0.9, 1.15, size=7)
        w_conf[31:38, 1] = -dipole_scale * rng.uniform(0.9, 1.15, size=7)

        x_full = bio @ w_bio.T + c_raw @ w_conf.T + 0.15 * rng.standard_normal((n_tot, p))
        views_tr.append(x_full[:n_train])
        views_te.append(x_full[n_train:])

    return {
        "views_tr": views_tr,
        "views_te": views_te,
        "c_tr": c_raw[:n_train],
        "c_te": c_raw[n_train:],
        "bio_tr": bio[:n_train],
        "bio_te": bio[n_train:],
    }


def run_synthetic_benchmarks(n_seeds: int = 5) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    scenarios = [
        ("synthetic_orthogonal_dipole", 0.0),
        ("synthetic_covarying_bio_r045", 0.45),
    ]
    fit_kw = dict(
        positivity="positive",
        energy_type="nc_abs",
        sparseness=0.5,
        iterations=120,
        learning_rate=0.05,
        random_state=0,
    )

    for scenario_name, rho in scenarios:
        for seed in range(n_seeds):
            split = _generate_synthetic_dipole_split(seed=100 + seed, bio_confound_corr=rho)
            v_tr, v_te = split["views_tr"], split["views_te"]
            c_tr, c_te = split["c_tr"], split["c_te"]
            bio_te = split["bio_te"]

            # 1. Unadjusted
            res_unadj = pysimlr.fit("simlr", v_tr, k=2, **fit_kw)
            # 2. Naive pre-residualized (oracle batch test C vs realistic no-C_test)
            v_tr_res, v_te_oracle = _ols_residualize_views(v_tr, v_te, c_tr, c_test_oracle=c_te)
            res_pre = pysimlr.fit("simlr", v_tr_res, k=2, **fit_kw)
            # 3. Confound sink hard (gamma=1.0) and soft (gamma=0.7)
            res_sink_10 = pysimlr.fit(
                "simlr", v_tr, k=2, confounds=c_tr, confound_strength=1.0, **fit_kw
            )
            res_sink_07 = pysimlr.fit(
                "simlr", v_tr, k=2, confounds=c_tr, confound_strength=0.7, **fit_kw
            )

            method_specs = [
                ("unadjusted", res_unadj, v_te, True),
                ("naive_preresid_oracle_batch", res_pre, v_te_oracle, False),
                ("naive_preresid_no_ctest", res_pre, v_te, True),
                ("confound_sink_soft_0.7", res_sink_07, v_te, True),
                ("confound_sink_hard_1.0", res_sink_10, v_te, True),
            ]

            for m_name, res, eval_views, batch_invariant in method_specs:
                u_tr = res["u"].detach().cpu().numpy()
                pred = pysimlr.predict(res, eval_views)
                u_te = pred["u"].detach().cpu().numpy()

                tr_corr_c = float(_safe_abs_corr_matrix(u_tr, c_tr).max())
                te_corr_c = float(_safe_abs_corr_matrix(u_te, c_te).max())
                bio_mat = _safe_abs_corr_matrix(u_te, bio_te)
                bio1_r = float(bio_mat[:, 0].max())
                bio2_r = float(bio_mat[:, 1].max())

                if "u_sink" in pred:
                    u_sink_te = pred["u_sink"].detach().cpu().numpy()
                    sink_c_r = _subspace_multiple_r(u_sink_te, c_te)
                else:
                    sink_c_r = float("nan")

                # Missing modality test (views 0 and 1 only, view 2 absent)
                pred_sub = pysimlr.predict(res, {0: eval_views[0], 1: eval_views[1]})
                u_sub = pred_sub["u"].detach().cpu().numpy()
                sub_bio_mat = _safe_abs_corr_matrix(u_sub, bio_te)
                sub_bio_mean = 0.5 * float(sub_bio_mat[:, 0].max() + sub_bio_mat[:, 1].max())

                single_diff = (
                    _single_subject_discrepancy(res, eval_views)
                    if batch_invariant
                    else float("nan")  # N=1 batch residualization has 0 degrees of freedom
                )
                hc = pysimlr.heldout_coupling(res, eval_views)
                heldout_cv = float(np.nanmean(hc["cross_view_heldout"]))

                rows.append(
                    {
                        "dataset": scenario_name,
                        "seed": seed,
                        "method": m_name,
                        "train_max_abs_corr_C": tr_corr_c,
                        "test_max_abs_corr_C": te_corr_c,
                        "test_bio1_abs_r": bio1_r,
                        "test_bio2_abs_r": bio2_r,
                        "test_bio_mean_abs_r": 0.5 * (bio1_r + bio2_r),
                        "test_missing_view_bio_mean_r": sub_bio_mean,
                        "test_sink_abs_corr_C": sink_c_r,
                        "heldout_cross_view_coupling": heldout_cv,
                        "single_subject_max_diff": single_diff,
                        "pct_negative_v": _pct_negative_v(res),
                    }
                )
    return rows


def _prepare_blast_dataset(csv_path: Path) -> Dict[str, Any]:
    df = pd.read_csv(csv_path, low_memory=False)

    # Regional physical macro-views (all p_m < n_train, zero QC columns inside biological views)
    t1_cols = sorted(
        c
        for c in df.columns
        if (c.startswith("T1Hier+thk_left_") or c.startswith("T1Hier+thk_right_"))
        and pd.api.types.is_numeric_dtype(df[c])
    )
    dti_cols = sorted(
        c
        for c in df.columns
        if (c.startswith("DTI+FA_JHU_") or c.startswith("DTI+bundle_FA_"))
        and pd.api.types.is_numeric_dtype(df[c])
    )
    rsf_cols = sorted(
        c
        for c in df.columns
        if (c.startswith("rsfMRI+rsf_alff_") or c.startswith("rsfMRI+rsf_falff_"))
        and pd.api.types.is_numeric_dtype(df[c])
    )

    # Keep subjects with >=80% observed features in all 3 core modalities
    valid_mask = (
        (df[t1_cols].notna().mean(axis=1) >= 0.80)
        & (df[dti_cols].notna().mean(axis=1) >= 0.80)
        & (df[rsf_cols].notna().mean(axis=1) >= 0.80)
    )
    sub = df.loc[valid_mask].reset_index(drop=True)

    # Drop any individual feature column with >10% missingness on the complete-modality cohort
    t1_cols = [c for c in t1_cols if sub[c].notna().mean() >= 0.90 and sub[c].std() > 1e-8]
    dti_cols = [c for c in dti_cols if sub[c].notna().mean() >= 0.90 and sub[c].std() > 1e-8]
    rsf_cols = [c for c in rsf_cols if sub[c].notna().mean() >= 0.90 and sub[c].std() > 1e-8]

    # Explicit multi-site/scanner + head-size + motion + blind-QC confound set
    cohorts = sub["cohort"].astype(str).to_numpy()
    cohort_dummies = pd.get_dummies(sub["cohort"].astype(str), drop_first=True, dtype=np.float64)

    qc_t1_cols = ["T1Hier+icv"]
    qc_dti_cols = [
        "DTI+dti_FD_raw_mean",
        "DTI+dti_eddy_pe_shift_mm_abs_mean",
    ]
    qc_rsf_cols = [
        "rsfMRI+rsf_FD_mean",
        "rsfMRI+rsf_blindqc_dvars",
        "rsfMRI+rsf_blindqc_tsnr",
        "rsfMRI+rsf_blindqc_ghosting_ratio",
    ]
    qc_all_cols = qc_t1_cols + qc_dti_cols + qc_rsf_cols
    c_shared_df = pd.concat([cohort_dummies.reset_index(drop=True), sub[qc_all_cols]], axis=1)
    c_t1_df = pd.concat([cohort_dummies.reset_index(drop=True), sub[qc_t1_cols]], axis=1)
    c_dti_df = pd.concat([cohort_dummies.reset_index(drop=True), sub[qc_dti_cols]], axis=1)
    c_rsf_df = pd.concat([cohort_dummies.reset_index(drop=True), sub[qc_rsf_cols]], axis=1)

    age = pd.to_numeric(sub["unified_age"], errors="coerce").to_numpy(dtype=np.float64)
    gbev = pd.to_numeric(
        sub["unified_log10_gbev_blast_only_plus1"], errors="coerce"
    ).to_numpy(dtype=np.float64)

    return {
        "n_subjects": int(len(sub)),
        "cohort_counts": {k: int(v) for k, v in pd.Series(cohorts).value_counts().items()},
        "cohorts": cohorts,
        "view_names": ["T1Hier_thk", "DTI_tract_FA", "rsfMRI_alff_falff"],
        "view_widths": [len(t1_cols), len(dti_cols), len(rsf_cols)],
        "raw_views": [
            sub[t1_cols].to_numpy(dtype=np.float64),
            sub[dti_cols].to_numpy(dtype=np.float64),
            sub[rsf_cols].to_numpy(dtype=np.float64),
        ],
        "confound_names": list(c_shared_df.columns),
        "c_shared": c_shared_df.to_numpy(dtype=np.float64),
        "c_per_view": [
            c_t1_df.to_numpy(dtype=np.float64),
            c_dti_df.to_numpy(dtype=np.float64),
            c_rsf_df.to_numpy(dtype=np.float64),
        ],
        "age": age,
        "gbev": gbev,
    }


def _impute_views_from_train(
    raw_views: Sequence[np.ndarray], train_idx: np.ndarray, test_idx: np.ndarray
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    tr_list: List[np.ndarray] = []
    te_list: List[np.ndarray] = []
    for v in raw_views:
        v_tr = v[train_idx].copy()
        v_te = v[test_idx].copy()
        means = np.nanmean(v_tr, axis=0)
        means = np.where(np.isfinite(means), means, 0.0)
        nan_tr = ~np.isfinite(v_tr)
        if np.any(nan_tr):
            v_tr[nan_tr] = np.take(means, np.where(nan_tr)[1])
        nan_te = ~np.isfinite(v_te)
        if np.any(nan_te):
            v_te[nan_te] = np.take(means, np.where(nan_te)[1])
        tr_list.append(v_tr)
        te_list.append(v_te)
    return tr_list, te_list


def _stratified_split(cohorts: np.ndarray, test_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    tr_parts: List[np.ndarray] = []
    te_parts: List[np.ndarray] = []
    for label in sorted(set(cohorts.tolist())):
        idx = np.where(cohorts == label)[0]
        perm = rng.permutation(idx)
        n_te = max(1, int(round(len(idx) * test_frac)))
        te_parts.append(perm[:n_te])
        tr_parts.append(perm[n_te:])
    return np.sort(np.concatenate(tr_parts)), np.sort(np.concatenate(te_parts))


def _partial_bio_multiple_r(
    u_tr: np.ndarray,
    u_te: np.ndarray,
    y_tr: np.ndarray,
    y_te: np.ndarray,
    c_tr: np.ndarray,
    c_te: np.ndarray,
) -> float:
    """Out-of-sample |corr(y_resid_te, yhat_te)| predicting confound-adjusted biology from U."""
    d_tr, d_te = _impute_and_augment_design(c_tr, c_te)
    valid_tr = np.isfinite(y_tr)
    valid_te = np.isfinite(y_te)
    if valid_tr.sum() < 15 or valid_te.sum() < 10:
        return float("nan")

    # Partial out confounds from outcome using training-fitted nuisance slope
    beta_c = np.linalg.pinv(d_tr[valid_tr], rcond=1e-10) @ y_tr[valid_tr]
    y_res_tr = y_tr[valid_tr] - d_tr[valid_tr] @ beta_c
    y_res_te = y_te[valid_te] - d_te[valid_te] @ beta_c

    x_tr = np.column_stack([np.ones(valid_tr.sum()), u_tr[valid_tr]])
    x_te = np.column_stack([np.ones(valid_te.sum()), u_te[valid_te]])
    # Mild ridge on latent columns for stable 4-component linear readout
    reg = np.diag([0.0] + [1.0] * u_tr.shape[1])
    w = np.linalg.solve(x_tr.T @ x_tr + reg, x_tr.T @ y_res_tr)
    yhat_te = x_te @ w
    return float(_safe_abs_corr_matrix(yhat_te, y_res_te)[0, 0])


def run_blast_benchmarks(csv_path: Path, n_splits: int = 5) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    data = _prepare_blast_dataset(csv_path)
    rows: List[Dict[str, Any]] = []
    fit_kw = dict(
        positivity="positive",
        energy_type="nc_abs",
        sparseness=0.5,
        iterations=200,
        learning_rate=0.02,
        random_state=0,
    )
    k = 4

    for split_idx in range(n_splits):
        tr_idx, te_idx = _stratified_split(data["cohorts"], test_frac=0.30, seed=500 + split_idx)
        v_tr, v_te = _impute_views_from_train(data["raw_views"], tr_idx, te_idx)
        c_tr_raw = data["c_shared"][tr_idx]
        c_te_raw = data["c_shared"][te_idx]
        d_tr, d_te = _impute_and_augment_design(c_tr_raw, c_te_raw)
        c_tr_imp = d_tr[:, 1:]
        c_te_imp = d_te[:, 1:]

        c_pv_tr = [pv[tr_idx] for pv in data["c_per_view"]]
        age_tr, age_te = data["age"][tr_idx], data["age"][te_idx]
        gbev_tr, gbev_te = data["gbev"][tr_idx], data["gbev"][te_idx]

        res_unadj = pysimlr.fit("simlr", v_tr, k=k, **fit_kw)

        v_tr_res, _ = _ols_residualize_views(v_tr, v_te, c_tr_raw, c_test_oracle=None)
        res_pre = pysimlr.fit("simlr", v_tr_res, k=k, **fit_kw)

        res_sink_07 = pysimlr.fit(
            "simlr", v_tr, k=k, confounds=c_tr_raw, confound_strength=0.7, **fit_kw
        )
        res_sink_10 = pysimlr.fit(
            "simlr", v_tr, k=k, confounds=c_tr_raw, confound_strength=1.0, **fit_kw
        )
        res_sink_pv = pysimlr.fit(
            "simlr",
            v_tr,
            k=k,
            confounds={0: c_pv_tr[0], 1: c_pv_tr[1], 2: c_pv_tr[2]},
            confound_strength=1.0,
            **fit_kw,
        )

        specs = [
            ("unadjusted", res_unadj),
            ("naive_preresid_no_ctest", res_pre),
            ("confound_sink_soft_0.7", res_sink_07),
            ("confound_sink_hard_1.0", res_sink_10),
            ("confound_sink_per_view_1.0", res_sink_pv),
        ]

        for m_name, res in specs:
            u_tr = res["u"].detach().cpu().numpy()
            pred = pysimlr.predict(res, v_te)
            u_te = pred["u"].detach().cpu().numpy()

            tr_corr_mat = _safe_abs_corr_matrix(u_tr, c_tr_imp)
            te_corr_mat = _safe_abs_corr_matrix(u_te, c_te_imp)

            if "u_sink" in pred:
                u_sink_te = pred["u_sink"].detach().cpu().numpy()
                sink_corr_te = float(_safe_abs_corr_matrix(u_sink_te, c_te_imp).max())
            else:
                sink_corr_te = float("nan")

            # Missing-modality deployment: rsfMRI absent at test time ({0: T1, 1: DTI})
            pred_no_rsf = pysimlr.predict(res, {0: v_te[0], 1: v_te[1]})
            u_no_rsf = pred_no_rsf["u"].detach().cpu().numpy()
            diag_agree = float(
                np.mean([_safe_abs_corr_matrix(u_te[:, j], u_no_rsf[:, j])[0, 0] for j in range(k)])
            )
            no_rsf_corr_c = float(_safe_abs_corr_matrix(u_no_rsf, c_te_imp).max())

            hc = pysimlr.heldout_coupling(res, v_te)
            heldout_cv = float(np.nanmean(hc["cross_view_heldout"]))

            age_partial_r = _partial_bio_multiple_r(u_tr, u_te, age_tr, age_te, c_tr_raw, c_te_raw)
            gbev_partial_r = _partial_bio_multiple_r(u_tr, u_te, gbev_tr, gbev_te, c_tr_raw, c_te_raw)

            rows.append(
                {
                    "dataset": "blast_4cohort_3modality",
                    "seed": split_idx,
                    "method": m_name,
                    "train_max_abs_corr_C": float(tr_corr_mat.max()),
                    "test_max_abs_corr_C": float(te_corr_mat.max()),
                    "test_mean_abs_corr_C": float(te_corr_mat.max(axis=0).mean()),
                    "test_missing_rsfmri_max_corr_C": no_rsf_corr_c,
                    "test_missing_rsfmri_u_agreement": diag_agree,
                    "test_sink_abs_corr_C": sink_corr_te,
                    "heldout_cross_view_coupling": heldout_cv,
                    "test_age_partial_r": age_partial_r,
                    "test_blast_gbev_partial_r": gbev_partial_r,
                    "single_subject_max_diff": _single_subject_discrepancy(res, v_te),
                    "pct_negative_v": _pct_negative_v(res),
                }
            )

    meta = {
        "n_subjects": data["n_subjects"],
        "cohort_counts": data["cohort_counts"],
        "view_names": data["view_names"],
        "view_widths": data["view_widths"],
        "confound_names": data["confound_names"],
        "k": k,
        "n_splits": n_splits,
    }
    return meta, rows


def _summarize(df: pd.DataFrame, group_cols: Sequence[str]) -> List[Dict[str, Any]]:
    num_cols = [
        c for c in df.columns if c not in set(group_cols) and c != "seed" and pd.api.types.is_numeric_dtype(df[c])
    ]
    agg = df.groupby(list(group_cols), sort=False)[num_cols].mean().reset_index()
    return agg.to_dict(orient="records")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blast-csv", type=Path, default=DEFAULT_BLAST_CSV)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    syn_rows = run_synthetic_benchmarks(n_seeds=args.seeds)
    syn_df = pd.DataFrame(syn_rows)

    blast_meta: Dict[str, Any] = {}
    blast_rows: List[Dict[str, Any]] = []
    if args.blast_csv.exists():
        blast_meta, blast_rows = run_blast_benchmarks(args.blast_csv, n_splits=args.seeds)

    all_df = pd.concat([syn_df, pd.DataFrame(blast_rows)], ignore_index=True, sort=False)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(OUT_CSV, index=False)

    summary_payload = {
        "synthetic_summary": _summarize(syn_df, ("dataset", "method")),
        "blast_metadata": blast_meta,
        "blast_summary": _summarize(pd.DataFrame(blast_rows), ("dataset", "method")) if blast_rows else [],
    }
    OUT_JSON.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")

    print("=== Synthetic Confound-Dipole Summary (mean over seeds) ===")
    syn_sum = pd.DataFrame(summary_payload["synthetic_summary"])
    print(
        syn_sum[
            [
                "dataset",
                "method",
                "train_max_abs_corr_C",
                "test_max_abs_corr_C",
                "test_bio_mean_abs_r",
                "test_missing_view_bio_mean_r",
                "test_sink_abs_corr_C",
                "single_subject_max_diff",
                "pct_negative_v",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    if blast_rows:
        print("\n=== Real 4-Cohort Blast Neuroimaging Summary (mean over stratified splits) ===")
        print(
            f"Subjects: N={blast_meta['n_subjects']} {blast_meta['cohort_counts']}, "
            f"views={list(zip(blast_meta['view_names'], blast_meta['view_widths']))}, "
            f"confounds ({len(blast_meta['confound_names'])}): {blast_meta['confound_names']}"
        )
        blast_sum = pd.DataFrame(summary_payload["blast_summary"])
        print(
            blast_sum[
                [
                    "method",
                    "train_max_abs_corr_C",
                    "test_max_abs_corr_C",
                    "test_mean_abs_corr_C",
                    "test_missing_rsfmri_max_corr_C",
                    "test_sink_abs_corr_C",
                    "heldout_cross_view_coupling",
                    "test_age_partial_r",
                    "test_blast_gbev_partial_r",
                    "single_subject_max_diff",
                    "pct_negative_v",
                ]
            ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
        )


if __name__ == "__main__":
    main()
