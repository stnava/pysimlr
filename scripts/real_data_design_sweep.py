#!/usr/bin/env python
"""
Real-data design sweep: which method, energy, mixing, w and MAI actually win.

Three arms, because a full cross product would be mostly wasted cells:

  A  method x energy x mixing        (w fixed at 0.5)      -- the main question
  B  w sweep for the SiMLR family    (energy x w, newton)  -- the structural knob
  C  MAI choice for the flow model   (dynamic_weights=True) -- only meaningful there

Every fit goes through `run_single_experiment`, so all arms share the seeded,
stratified split and the train-only scaling (`needs_scaling`). Results land in
paper/results_cache/real_design_sweep.csv.
"""
from __future__ import annotations

import os
import multiprocessing

for _v in ("OMP", "MKL", "OPENBLAS", "VECLIB_MAXIMUM", "NUMEXPR"):
    os.environ[f"{_v}_NUM_THREADS"] = "1"

import argparse
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_diabetes
from scipy.linalg import qr as scipy_qr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pysimlr.benchmarks.runner import run_single_experiment  # noqa: E402


# --------------------------------------------------------------------------
# cases: unscaled on purpose; the scaler is fitted on the training split only
# --------------------------------------------------------------------------
def _drop_collinear(m: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Keep a maximal set of numerically independent columns (QR with pivoting).

    ``mfeat-fac`` ships **three exact duplicate columns** out of 216. A repeated
    column makes the Gram matrix exactly singular, and the whole constrained
    family died on it: NSAFlow-Turnkey raised
    ``torch.linalg.solve: matrix is singular`` and SiMLR / SiMLR-LBFGS /
    SiMLR-LBFGSB each raised "unusable projection for a (216, 3) basis",
    72 of 72 fits apiece. Dropping the duplicates is sufficient -- the views
    remain merely ill-conditioned afterwards and every method runs.

    Note this is *exact* duplication, not general rank deficiency: the other
    views stay rank-deficient after cleaning and cause no trouble. It is still
    worth recording that an exactly repeated feature makes these estimators
    crash rather than degrade, which is a robustness gap of their own.
    """
    if m.shape[1] == 0:
        return m
    mc = m - m.mean(axis=0, keepdims=True)
    sd = mc.std(axis=0)
    mc = mc[:, sd > 0]
    if mc.shape[1] == 0:
        return m[:, :1]
    mc = mc / mc.std(axis=0)
    _q, r, piv = scipy_qr(mc, mode="economic", pivoting=True)
    diag = np.abs(np.diag(r))
    rank = int((diag > tol * diag[0]).sum()) if diag.size else 0
    keep = np.sort(piv[:max(rank, 1)])
    return m[:, np.flatnonzero(sd > 0)[keep]]


def get_diabetes_case(seed: int = 42) -> dict:
    d = load_diabetes()
    X, y, k = d.data, d.target, 2
    return {"kind": "diabetes", "data": [torch.tensor(X[:, :5]).float(),
                                         torch.tensor(X[:, 5:]).float()],
            "outcome": torch.tensor(y).float(),
            "true_u": torch.zeros(X.shape[0], k),
            "true_v": [np.zeros((5, k)), np.zeros((X.shape[1] - 5, k))],
            "shared_k": k, "needs_scaling": True, "is_classification": False}


def get_heart_case(seed: int = 42) -> dict:
    url = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
           'heart-disease/processed.cleveland.data')
    cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
    try:
        df = pd.read_csv(url, names=cols).replace('?', np.nan).dropna().apply(pd.to_numeric)
    except Exception:
        rng = np.random.default_rng(0)
        df = pd.DataFrame(rng.standard_normal((300, 13)), columns=cols[:13])
        df['num'] = rng.integers(0, 2, 300)
    X = df.drop('num', axis=1).values.astype(float)
    y = (df['num'].values > 0).astype(int)     # binary, as in the deep benchmark
    k = 2
    return {"kind": "heart", "data": [torch.tensor(X[:, :7]).float(),
                                      torch.tensor(X[:, 7:]).float()],
            "outcome": torch.tensor(y).float(),
            "true_u": torch.zeros(X.shape[0], k),
            "true_v": [np.zeros((7, k)), np.zeros((X.shape[1] - 7, k))],
            "shared_k": k, "needs_scaling": True, "is_classification": True}


# --------------------------------------------------------------------------
# High-dimensional real cohorts.
#
# Diabetes and Heart have views of 5, 5, 7 and 6 features. At p <= 6 the
# achievable sparsity grid has step 1/(2p) >= 0.083 and the projection already
# sits near disjointness at w = 0.1, so the measured range of sparsity over
# w in [0.1, 0.9] is *exactly* 0.000 -- the w question cannot be asked there.
# It has range 0.14 at p = 7 and 0.15 at p = 20. Energy and mixing likewise sit
# at chance on those two datasets. These cohorts put p in the hundreds to
# thousands, which is the regime the questions need.
# --------------------------------------------------------------------------
_NSA_EXP = os.path.expanduser("~/data/repos/nsa_flow")


def _nsa_experiments():
    if _NSA_EXP not in sys.path:
        sys.path.insert(0, _NSA_EXP)
    from experiments import data as _d, ppmi as _p       # noqa: E402
    return _d, _p


def get_ppmi_case(seed: int = 42) -> dict:
    """PPMI T1 imaging, split into its two natural derived modalities.

    ``vol`` (141 regional volumes) and ``thk`` (141 cortical thicknesses) are
    distinct measurements of the same anatomy, which is exactly the multi-view
    structure SiMLR is for -- unlike splitting one feature block in half.
    """
    _d, _p = _nsa_experiments()
    X, y, _cov, names = _p.load_ppmi(modality="T1Hier", task="PD vs CN")
    X = np.asarray(X, dtype=float)
    vol = [i for i, n in enumerate(names) if "_vol" in n]
    thk = [i for i, n in enumerate(names) if "_thk" in n]
    mats = [X[:, vol], X[:, thk]]
    k = 3
    return {"kind": "ppmi", "data": [torch.tensor(m).float() for m in mats],
            "outcome": torch.tensor(np.asarray(y)).float(),
            "true_u": torch.zeros(X.shape[0], k),
            "true_v": [np.zeros((m.shape[1], k)) for m in mats],
            "shared_k": k, "needs_scaling": True, "is_classification": True}


def get_adni_case(seed: int = 42, views: str = "t1_dti",
                  min_complete_rows: int = 150) -> dict:
    """ADNI at **baseline**, predicting **CDRSB** -- a regression, not a class.

    The earlier version of this case predicted a diagnosis label, which was
    wrong twice over: ADNI's target here is the Clinical Dementia Rating Sum of
    Boxes, a continuous severity score, and it is predicted from the baseline
    visit alone rather than from anything longitudinal. Scored as a two-class
    problem it also sat under its own majority-class baseline (0.717 accuracy
    against a 0.724 prior, balanced accuracy 0.579), so it was reporting
    almost nothing.

    Genuinely multi-modal, which is the point: separate imaging assays on the
    same subjects rather than one feature block split up.

    ==========  =====  ================
    view        cols   complete at bl
    ==========  =====  ================
    T1 volume     336  369 / 372
    T1 thickness  336  369 / 372
    T1 area       277  369 / 372
    DTI mean      494  371 / 372
    rsfMRI       6459  196 / 372
    ==========  =====  ================

    ``views="t1_dti"`` takes the four dense modalities; ``"all"`` adds rsfMRI
    and drops to roughly 196 subjects, which is the trade this cohort offers.
    ``NM2DMT`` and ``T2Flair`` have no complete rows at all and are excluded.
    """
    if _NSA_EXP not in sys.path:
        sys.path.insert(0, _NSA_EXP)
    from experiments.rmd_support import THK                     # noqa: E402

    df = pd.read_csv(THK, low_memory=False)
    df = df[(df.studyName == "ADNI") & (df.yearsbl == 0)]

    fams = [("T1Hier_vol_", "vol"), ("T1Hier_thk_", "thk"),
            ("T1Hier_area_", "area"), ("DTI_mean_", "dti")]
    if views == "all":
        fams.append(("rsfMRI_", "rsfmri"))

    blocks = []
    for pre, _name in fams:
        cols = [c for c in df.columns if pre in c
                and not any(x in c for x in ("Asym", "reference", "adjusted"))]
        blocks.append(df[cols].apply(pd.to_numeric, errors="coerce"))

    y = pd.to_numeric(df["CDRSB"], errors="coerce")
    ok = y.notna()
    for b in blocks:
        ok &= b.notna().all(axis=1)
    if int(ok.sum()) < min_complete_rows:
        raise ValueError(
            f"ADNI views={views!r} leaves only {int(ok.sum())} complete rows; "
            f"raise min_complete_rows to accept that or drop rsfMRI.")

    mats = [b[ok].to_numpy(float) for b in blocks]
    mats = [m[:, m.std(axis=0) > 0] for m in mats]     # constant cols carry nothing
    y = y[ok].to_numpy(float)
    k = 3
    return {"kind": "adni_cdrsb", "data": [torch.tensor(m).float() for m in mats],
            "outcome": torch.tensor(y).float(),
            "true_u": torch.zeros(mats[0].shape[0], k),
            "true_v": [np.zeros((m.shape[1], k)) for m in mats],
            "shared_k": k, "needs_scaling": True, "is_classification": False}


def get_metabric_case(seed: int = 42, n_genes: int = 2000) -> dict:
    """METABRIC expression, top-variance genes split into two views.

    The split is arbitrary (this is one assay, not two), so read this cohort as
    the high-dimensional stress case -- p in the thousands, p >> per-view k --
    rather than as evidence about multi-modal integration.
    """
    cache = os.path.join(_NSA_EXP, "experiments", "cache")
    z = np.load(os.path.join(cache, "metabric_expr.npz"), allow_pickle=True)
    X = np.asarray(z["X"], dtype=float).T                  # genes x samples -> n x p
    # 16 NaNs across 11 genes. Dropping those genes costs nothing at p=19676
    # and keeps every method on identical data: PCA raises on NaN while
    # `simlr` silently `nan_to_num`s it to zero, so leaving them in would have
    # compared a zero-imputed fit against a refusal.
    X = X[:, ~np.isnan(X).any(axis=0)]
    clin = pd.read_csv(os.path.join(cache, "metabric_clinical.csv"))
    lab = clin["THREEGENE"].astype(str).values
    keep = lab != "Unknown"
    X, lab = X[keep], lab[keep]
    var = X.var(axis=0)
    top = np.argsort(var)[::-1][:n_genes]
    X = X[:, np.sort(top)]
    half = X.shape[1] // 2
    mats = [X[:, :half], X[:, half:]]
    y = pd.factorize(lab)[0].astype(float)
    k = 3
    return {"kind": "metabric", "data": [torch.tensor(m).float() for m in mats],
            "outcome": torch.tensor(y).float(),
            "true_u": torch.zeros(X.shape[0], k),
            "true_v": [np.zeros((m.shape[1], k)) for m in mats],
            "shared_k": k, "needs_scaling": True, "is_classification": True}


def get_golub_case(seed: int = 42) -> dict:
    """Golub leukemia expression, n=72, p=7129. The p >> n extreme."""
    cache = os.path.join(_NSA_EXP, "experiments", "cache")
    X = pd.read_csv(os.path.join(cache, "golub_X.csv")).values.astype(float)
    y = pd.read_csv(os.path.join(cache, "golub_y3.csv")).iloc[:, 0].values
    half = X.shape[1] // 2
    mats = [X[:, :half], X[:, half:]]
    k = 3
    return {"kind": "golub", "data": [torch.tensor(m).float() for m in mats],
            "outcome": torch.tensor(pd.factorize(y)[0].astype(float)).float(),
            "true_u": torch.zeros(X.shape[0], k),
            "true_v": [np.zeros((m.shape[1], k)) for m in mats],
            "shared_k": k, "needs_scaling": True, "is_classification": True}


#: The appendix cohort. Lives in iCloud rather than the repo, so `data/BRCA`
#: is absent from a fresh clone and the BRCA appendix cannot be re-rendered
#: without it.
BRCA_CSV = os.path.expanduser(
    "~/Library/Mobile Documents/com~apple~CloudDocs/code/pysimlr/"
    "data/BRCA/brca_data_w_subtypes.csv")


def get_brca_case(seed: int = 42, views: str = "rs_cn_pp") -> dict:
    """TCGA-BRCA, the paper's appendix cohort: a genuine *three-view* problem.

    This is the only real cohort here where the modalities are separate assays
    rather than a feature block split in half -- RNA-Seq, copy number and
    RPPA proteomics measured on the same tumours. That is the structure SiMLR
    exists for, so it is the cohort its claims should rest on.

    Filtering ``ER.Status`` to Positive/Negative reproduces the appendix's
    ``N = 549`` exactly (414 + 135); the remaining 156 samples are
    "Not Performed", "Indeterminate" or unavailable.

    Parameters
    ----------
    views : str, default "rs_cn_pp"
        Which assays to use. ``"rs_cn_pp"`` matches the appendix's three
        modalities; ``"all"`` adds the 249 mutation features as a fourth view.
    """
    if not os.path.exists(BRCA_CSV):
        raise FileNotFoundError(
            f"TCGA-BRCA not found at {BRCA_CSV}. The appendix reports results "
            f"on it but the repo ships only the rendered figure; the source is "
            f"the Kaggle 'BRCA Multi-Omics (TCGA)' release.")
    df = pd.read_csv(BRCA_CSV, on_bad_lines="skip")
    er = df["ER.Status"].astype(str)
    keep = er.isin(["Positive", "Negative"])
    df = df[keep]
    y = (er[keep] == "Positive").astype(float).values

    prefixes = ["rs_", "cn_", "pp_"] + (["mu_"] if views == "all" else [])
    mats = [df[[c for c in df.columns if c.startswith(pre)]].values.astype(float)
            for pre in prefixes]
    # Mutation calls are 0/1 indicators; a constant column carries no signal
    # and gives the scaler a zero variance to divide by.
    mats = [m[:, m.std(axis=0) > 0] for m in mats]
    k = 3
    return {"kind": "brca", "data": [torch.tensor(m).float() for m in mats],
            "outcome": torch.tensor(y).float(),
            "true_u": torch.zeros(mats[0].shape[0], k),
            "true_v": [np.zeros((m.shape[1], k)) for m in mats],
            "shared_k": k, "needs_scaling": True, "is_classification": True}


#: PPMI is excluded. Its PD-vs-CN split is 202/819, a majority-class rate of
#: 0.802, and every one of the nine models scored exactly 0.800 raw accuracy
#: -- i.e. all of them predicted the majority class, for a balanced accuracy of
#: exactly 0.500 across the board. The cohort as posed carries no signal any
#: method can separate on, so including it would contribute one row of
#: nine-way ties to every ranking. `get_ppmi_case` is kept for the record; a
#: different task or a matched design would be needed to make it usable.
EXCLUDED_CASES = {"PPMI": "all models at chance (balanced accuracy 0.500)"}


#: UCI Multiple Features: the canonical multi-view benchmark. Six genuinely
#: different feature extractions of the *same* 2000 handwritten digits, 200
#: per class, so classes are exactly balanced and there is no majority-class
#: trap. Unlike every biomedical cohort here the views are known a priori to
#: be complementary, which makes it the control: if a multi-view method cannot
#: beat concatenated PCA on mfeat, "multi-view helps" has no support anywhere.
MFEAT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "mfeat")
MFEAT_VIEWS = ["fac", "fou", "kar", "pix", "zer", "mor"]


def get_mfeat_case(seed: int = 42, views: int = 6) -> dict:
    """UCI mfeat handwritten numerals: 2000 x 10 classes over up to 6 views.

    ``views`` truncates to the first ``n`` feature sets, so the same cohort
    gives a 2-, 3-, 4- or 6-view problem with everything else held fixed. That
    is the clean way to ask whether adding views helps, which the BRCA 3 -> 4
    comparison could only answer confounded with adding information.
    """
    names = MFEAT_VIEWS[:views]
    mats = []
    for v in names:
        f = os.path.join(MFEAT_DIR, f"mfeat-{v}")
        if not os.path.exists(f):
            raise FileNotFoundError(
                f"mfeat view not found: {f}. Fetch with "
                f"curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/mfeat/mfeat-{v}")
        mats.append(np.loadtxt(f))
    # The file order is 200 of digit 0, then 200 of digit 1, and so on.
    y = np.repeat(np.arange(10), 200).astype(float)
    mats = [_drop_collinear(m) for m in mats]
    k = 3
    return {"kind": f"mfeat{views}", "data": [torch.tensor(m).float() for m in mats],
            "outcome": torch.tensor(y).float(),
            "true_u": torch.zeros(mats[0].shape[0], k),
            "true_v": [np.zeros((m.shape[1], k)) for m in mats],
            "shared_k": k, "needs_scaling": True, "is_classification": True}


CASES = {
    "Diabetes": get_diabetes_case,
    "Heart": get_heart_case,
    "ADNI": get_adni_case,
    "METABRIC": get_metabric_case,
    "Golub": get_golub_case,
    # 5-view: adds rsfMRI, at the cost of about half the subjects.
    "ADNI5": lambda seed=42: get_adni_case(seed=seed, views="all"),
    "BRCA": get_brca_case,
    # Same tumours, four assays: the mutation block promoted to its own view.
    "BRCA4": lambda seed=42: get_brca_case(seed=seed, views="all"),
    # Same digits, more views: an unconfounded test of "do extra views help".
    "mfeat2": lambda seed=42: get_mfeat_case(seed=seed, views=2),
    "mfeat4": lambda seed=42: get_mfeat_case(seed=seed, views=4),
    "mfeat6": lambda seed=42: get_mfeat_case(seed=seed, views=6),
}

SIMLR_FAMILY = [("linear", "SiMLR"), ("simlr_lbfgs", "SiMLR-LBFGS"),
                ("simlr_lbfgsb", "SiMLR-LBFGSB")]
ALL_MODELS = SIMLR_FAMILY + [
    ("pca", "PCA"), ("lend", "LEND"), ("ned", "NED"), ("shared_private", "NEDPP"),
    ("flow_v", "Flow-SiMLR-V"), ("nsa_pipeline", "NSAFlow-Turnkey"),
]
ENERGIES = ["regression", "acc", "logcosh", "nc"]
MIXINGS = ["newton", "svd", "pca", "ica"]
WS = [0.1, 0.3, 0.5, 0.7, 0.9]
MAIS = ["procrustes_r2", "procrustes_r2_sharp", "cca", "rvcoef", "trace"]


def run_one(task: tuple) -> dict:
    torch.set_num_threads(1)
    (arm, ds, model_type, label, energy, mixing, w, mai, rank_mai,
     seed, iterations, epochs) = task
    case = CASES[ds](seed=seed)
    is_cls = bool(case.get("is_classification"))

    params = {
        "iterations": iterations, "epochs": epochs,
        "energy_type": energy, "mixing_algorithm": mixing,
        "use_nsa": True, "positivity": "positive",
        "nsa_w": w, "constraint": f"orthox{w}x1",
        "sparseness_quantile": 0.0, "consolidate": True,
    }
    if mai is not None:
        params.update({"dynamic_weights": True, "mai_metric": mai,
                       "use_rank_mai": bool(rank_mai)})

    out = run_single_experiment(model_type, case, seed=seed, **params)
    m = out["metrics"]
    pick = (lambda a, b: m.get(a, np.nan) if is_cls else m.get(b, np.nan))
    return {
        "Arm": arm, "Dataset": ds, "Model": label, "Energy": energy,
        "Mixing": mixing, "W": w, "MAI": mai or "-", "RankMAI": bool(rank_mai),
        "Seed": seed,
        "Predictive": float(pick("test_accuracy", "test_r2")),
        "Train": float(pick("train_accuracy", "train_r2")),
        "StrictlyLinear": float(pick("first_layer_test_accuracy", "first_layer_test_r2")),
        "AxisSensitive": float(pick("first_layer_axis_test_accuracy", "first_layer_axis_test_r2")),
        "Sparsity": float(m.get("sparsity_ratio", np.nan)),
        "FrameDefect": float(m.get("frame_defect", np.nan)),
        "GradMap": (float(m["grad_map"]) if m.get("grad_map") is not None else np.nan),
        "EnergyReduction": (float(m["energy_reduction"])
                            if m.get("energy_reduction") is not None else np.nan),
        "Converged": m.get("solver_converged"),
        "StopReason": m.get("stop_reason"),
        "FitSeconds": float(m.get("fit_seconds", np.nan)),
    }


def build_tasks(n_seeds: int, iterations: int, epochs: int,
                datasets=None, mixings=None) -> list:
    seeds = list(range(42, 42 + n_seeds))
    mixings = list(mixings) if mixings else MIXINGS
    chosen = list(datasets) if datasets else list(CASES)
    tasks = []
    for ds in chosen:
        # Arm A: method x energy x mixing, w fixed
        for mt, lbl in ALL_MODELS:
            for e in ENERGIES:
                for mx in mixings:
                    for s in seeds:
                        tasks.append(("A", ds, mt, lbl, e, mx, 0.5, None, False,
                                      s, iterations, epochs))
        # Arm B: w sweep, SiMLR family, newton mixing
        for mt, lbl in SIMLR_FAMILY:
            for e in ENERGIES:
                for w in WS:
                    for s in seeds:
                        tasks.append(("B", ds, mt, lbl, e, "newton", w, None, False,
                                      s, iterations, epochs))
        # Arm C: MAI choice, flow model only
        for mai in MAIS:
            for rank in (False, True):
                for s in seeds:
                    tasks.append(("C", ds, "flow_v", "Flow-SiMLR-V", "regression",
                                  "newton", 0.5, mai, rank, s, iterations, epochs))
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser(description="Real-data design sweep")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--iterations", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--datasets", nargs="*", default=None,
                    help=f"subset of {list(CASES)}")
    ap.add_argument("--mixings", nargs="*", default=None,
                    help=f"subset of {MIXINGS}")
    ap.add_argument("--out", type=str, default="real_design_sweep.csv")
    a = ap.parse_args()

    tasks = build_tasks(a.seeds, a.iterations, a.epochs,
                        datasets=a.datasets, mixings=a.mixings)
    print(f"{len(tasks)} tasks over {a.workers} workers", flush=True)

    rows, failures, t0, done = [], [], time.perf_counter(), 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_one, t): t for t in tasks}
        for f in as_completed(futs):
            done += 1
            try:
                rows.append(f.result())
            except Exception as exc:
                failures.append((futs[f], f"{type(exc).__name__}: {exc}"))
            if done % 250 == 0 or done == len(tasks):
                el = time.perf_counter() - t0
                print(f"[{done}/{len(tasks)}] {el:.0f}s "
                      f"eta {(len(tasks)-done)/max(done/el,1e-9):.0f}s "
                      f"fail={len(failures)}", flush=True)

    df = pd.DataFrame(rows)
    out_dir = os.path.join(os.path.dirname(__file__), "..", "paper", "results_cache")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, a.out)
    df.to_csv(out, index=False)
    print(f"\nsaved {len(df)} rows to {out}")
    if failures:
        print(f"\n{len(failures)} failures; first 5 distinct:")
        seen = set()
        for t, msg in failures:
            key = (t[2], t[4], t[5], msg.split(':')[0])
            if key in seen:
                continue
            seen.add(key)
            print(f"  model={t[3]} energy={t[4]} mixing={t[5]} w={t[6]} -> {msg[:150]}")
            if len(seen) >= 5:
                break


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
