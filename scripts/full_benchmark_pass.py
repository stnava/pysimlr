"""Full benchmark re-establishment across every dataset this session found
genuinely useful: Heart Disease, Diabetes, mfeat, ADNI-DX, ADNI-CDRSB
(full cohort), TCGA-KIRC. TCGA-BRCA is excluded (nothing beat baseline
there); the ADNI tau/amyloid attempts are excluded (N too small to count).

For each dataset, in order (one fully finishes before the next starts):
  1. Standalone: pca-raw, gcca-raw.
  2. Method comparison: SiMLR/LEND/NED/Flow-SiMR/Flow-SiMR-V, each
     initialized with PCA (their own default).
  3. Initialization x method sweep: 6 strategies (best_of_n excluded) x
     {SiMLR, LEND, NED, Flow-SiMR-V}.

Every deep/linear method call now uses a generous convergence cap
(`real_data_method_comparison.DEEP_EPOCHS=200`, `SIMLR_ITERATIONS=150`) and
prints a `[!]` line if it hits that cap without its own ConvergenceMonitor
reporting `converged=True` -- see that module's docstring for why the
epochs=60 fixed budget used earlier this session was not verified to be
enough. `positivity="either"` and `safe_svd`'s LAPACK-driver fallback (fixed
this session) apply throughout; `gcca`/`nndsvd` are expected to succeed on
every dataset now, not fail as they did on ADNI before that fix.

Usage: python scripts/full_benchmark_pass.py <pass_label> [seed]
`pass_label` is just a string for the printed header (e.g. "pass1", "pass2",
"pass3-seed123"); `seed` (default 42) is threaded through every method's
weight init, initialize_simlr's random-type generators, and the RF split.
"""
import sys

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "scripts")
import real_data_method_comparison as rmc
import real_data_init_x_method as rim
from real_data_init_comparison import (load_heart, load_diabetes, load_mfeat,
                                       load_adni, load_kirc, BENCHMARK_INITIALIZATION_TYPES)
from adni_loader import get_adni_cdrsb_case

LABELS = {"PCA": "pca-raw", "GCCA": "gcca-raw", "SiMLR": "SiMLR[pca]",
         "LEND": "LEND[pca]", "NED": "NED[pca]", "Flow-SiMR": "Flow-SiMR",
         "Flow-SiMR-V": "Flow-SiMR-V[pca]"}


def load_adni_cdrsb_full():
    case = get_adni_cdrsb_case()
    views = [torch.tensor(StandardScaler().fit_transform(v.numpy())).float() for v in case["views"]]
    return views, case["y"], "regression_with_covariates", case["covariates"]


def set_seed(seed):
    rmc.SEED = seed
    rim.SEED = seed


def run_dataset(name, views, y, task, seed, cov=None):
    print(f"\n{'='*70}\n{name}  (N={views[0].shape[0]}, views={[v.shape[1] for v in views]}, task={task})\n{'='*70}", flush=True)

    if task == "regression_with_covariates":
        idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.3, random_state=seed)

        def score(u):
            z = u.numpy() if torch.is_tensor(u) else np.asarray(u)
            feat = np.concatenate([z, cov], axis=1)
            rf = RandomForestRegressor(n_estimators=200, random_state=seed).fit(feat[idx_tr], y[idx_tr])
            return rf.score(feat[idx_te], y[idx_te])

        cov_rf = RandomForestRegressor(n_estimators=200, random_state=seed).fit(cov[idx_tr], y[idx_tr])
        print(f"  covariates-only            R2={cov_rf.score(cov[idx_te], y[idx_te]):.4f}", flush=True)
        metric = "R2"
    else:
        idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.3, random_state=seed,
                                          stratify=y if task == "classification" else None)
        if task == "classification":
            vals, counts = np.unique(y[idx_te], return_counts=True)
            print(f"  majority baseline          acc={counts.max()/counts.sum():.4f}", flush=True)
        else:
            # Plain regression (no covariates), e.g. Diabetes: a "majority
            # class" of the continuous target is meaningless -- R^2's own
            # zero point (predicting the mean) is the right anchor, and it's
            # implicit in every R2 printed below, not worth a separate line.
            print("  (regression target -- R2=0 is the mean-prediction baseline)", flush=True)

        def score(u):
            z = u.numpy() if torch.is_tensor(u) else np.asarray(u)
            model_cls = RandomForestClassifier if task == "classification" else RandomForestRegressor
            rf = model_cls(n_estimators=300, random_state=seed).fit(z[idx_tr], y[idx_tr])
            return rf.score(z[idx_te], y[idx_te])

        metric = "acc" if task == "classification" else "R2"

    print("  --- standalone / method comparison ---", flush=True)
    for mname, fn in rmc.METHODS:
        try:
            u = fn(views)
            print(f"  {LABELS.get(mname, mname):<16} {metric}={score(u):.4f}", flush=True)
        except Exception as e:
            print(f"  {LABELS.get(mname, mname):<16} FAILED: {e!r}", flush=True)

    print("  --- initialization x method ---", flush=True)
    for mname, fn in rim.METHODS:
        for strat in BENCHMARK_INITIALIZATION_TYPES:
            kwargs = {"initialization_type": strat}
            if strat == "random":
                kwargs["init_seed"] = seed
            if strat in ("perturbed_pca",):
                kwargs["init_seed"] = seed
            if strat == "domain":
                rng = np.random.default_rng(seed)
                kwargs["domain_matrices"] = [
                    torch.tensor(rng.standard_normal((3, v.shape[1]))).float() for v in views
                ]
            try:
                u = fn(views, **kwargs)
                print(f"    {mname:<12} {strat:<15} {metric}={score(u):.4f}", flush=True)
            except Exception as e:
                print(f"    {mname:<12} {strat:<15} FAILED: {e!r}", flush=True)


def main():
    pass_label = sys.argv[1] if len(sys.argv) > 1 else "pass1"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    print(f"### {pass_label}  (seed={seed}) ###", flush=True)
    set_seed(seed)

    views, y, task = load_heart()
    run_dataset("Heart Disease", views, y, task, seed)

    views, y, task = load_diabetes()
    run_dataset("Diabetes", views, y, task, seed)

    views, y, task = load_mfeat()
    run_dataset("mfeat", views, y, task, seed)

    views, y, task = load_adni()
    run_dataset("ADNI-DX", views, y, task, seed)

    views, y, task, cov = load_adni_cdrsb_full()
    run_dataset("ADNI-CDRSB", views, y, task, seed, cov=cov)

    views, y, task = load_kirc()
    run_dataset("TCGA-KIRC", views, y, task, seed)

    print(f"\n### {pass_label} complete ###", flush=True)


if __name__ == "__main__":
    main()
