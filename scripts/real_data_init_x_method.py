"""Effect of initialization strategy on every method that has one: SiMLR,
LEND, NED, and Flow-SiMR-V (Flow-SiMR excluded -- no linear encoder, no basis
to initialize; PCA/GCCA excluded -- they *are* initialization strategies, not
methods with one to vary).

Datasets: Heart Disease, Diabetes, mfeat -- the three "informative" real
datasets from the previous sweep. TCGA-BRCA is dropped here: every method
sat at/below its majority-class baseline there, so it cannot currently
distinguish initialization strategies from noise; see `real_data_method_comparison.py`'s
findings. `positivity="either"` is passed explicitly everywhere (the new
default, but explicit per this session's decision).

Downstream scoring: RandomForestClassifier / RandomForestRegressor, same
convention as `real_data_method_comparison.py`. Single run per cell, one
seed -- a spot check, not a seeded benchmark. Prints a line immediately after
each run finishes (flush=True).
"""
import sys

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

sys.path.insert(0, "scripts")
from real_data_init_comparison import load_heart, load_diabetes, load_mfeat, BENCHMARK_INITIALIZATION_TYPES

from pysimlr.simlr import simlr
from pysimlr.deep import lend_simr, ned_simr
from pysimlr.flows import flow_simr_v

SEED = 42
K = 3
#: See `real_data_method_comparison.py`'s identical constants for why these
#: are generous caps relying on each method's own ConvergenceMonitor, not
#: fixed budgets -- epochs=60 was measured right at the edge of convergence
#: even on the easiest dataset here (Heart Disease).
DEEP_EPOCHS = 200
#: Flow-SiMR-V needs more -- see `real_data_method_comparison.py`'s identical
#: constant for the measurement (pass 1, ADNI-DX: hit epochs=200 still
#: improving, converges properly by 400).
FLOW_EPOCHS = 400
SIMLR_ITERATIONS = 150
DEEP_WARMUP = None

DATASETS = [
    ("Heart Disease", load_heart),
    ("Diabetes", load_diabetes),
    ("mfeat", load_mfeat),
]


def _check_converged(label, out):
    conv = out.get("convergence")
    if conv is not None:
        if not conv.get("converged", False):
            print(f"    [!] {label} did not converge within its cap "
                 f"(best_epoch={conv.get('best_epoch')}, max_steps={conv.get('max_steps')})", flush=True)
        return
    if out.get("stop_reason") == "max_iter":
        print(f"    [!] {label} hit max_iter without converging", flush=True)


def run_simlr(views, **kw):
    out = simlr(views, k=K, iterations=SIMLR_ITERATIONS, positivity="either", verbose=False, **kw)
    _check_converged("SiMLR", out)
    return out["u"]


def run_lend(views, **kw):
    torch.manual_seed(SEED)
    out = lend_simr(views, k=K, epochs=DEEP_EPOCHS, warmup_epochs=DEEP_WARMUP,
                    positivity="either", verbose=False, **kw)
    _check_converged("LEND", out)
    return out["u"]


def run_ned(views, **kw):
    torch.manual_seed(SEED)
    out = ned_simr(views, k=K, epochs=DEEP_EPOCHS, warmup_epochs=DEEP_WARMUP,
                   positivity="either", verbose=False, **kw)
    _check_converged("NED", out)
    return out["u"]


def run_flow_simr_v(views, **kw):
    torch.manual_seed(SEED)
    out = flow_simr_v(views, k=K, epochs=FLOW_EPOCHS, warmup_epochs=DEEP_WARMUP,
                      positivity="either", device="cpu", verbose=False, **kw)
    _check_converged("Flow-SiMR-V", out)
    return out["u"]


METHODS = [
    ("SiMLR", run_simlr),
    ("LEND", run_lend),
    ("NED", run_ned),
    ("Flow-SiMR-V", run_flow_simr_v),
]


def evaluate(u, y, task):
    z = u.numpy() if torch.is_tensor(u) else np.asarray(u)
    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.3,
                                      random_state=SEED, stratify=y if task == "classification" else None)
    if task == "classification":
        model = RandomForestClassifier(n_estimators=300, random_state=SEED)
        model.fit(z[idx_tr], y[idx_tr])
        return float(model.score(z[idx_te], y[idx_te]))
    model = RandomForestRegressor(n_estimators=300, random_state=SEED)
    model.fit(z[idx_tr], y[idx_tr])
    return float(model.score(z[idx_te], y[idx_te]))


def init_kwargs(strat, views):
    kwargs = {"initialization_type": strat}
    if strat == "random":
        kwargs["init_seed"] = SEED
    if strat in ("perturbed_pca", "best_of_n"):
        kwargs["init_seed"] = SEED
    if strat == "domain":
        rng = np.random.default_rng(SEED)
        kwargs["domain_matrices"] = [
            torch.tensor(rng.standard_normal((K, v.shape[1]))).float() for v in views
        ]
    return kwargs


def main():
    for dname, loader in DATASETS:
        print(f"\n=== {dname} ===", flush=True)
        try:
            views, y, task = loader()
        except Exception as e:
            print(f"  LOAD FAILED: {e!r}", flush=True)
            continue
        print(f"  N={views[0].shape[0]}, views={[v.shape[1] for v in views]}, task={task}", flush=True)
        for mname, fn in METHODS:
            print(f"  --- {mname} ---", flush=True)
            for strat in BENCHMARK_INITIALIZATION_TYPES:
                try:
                    u = fn(views, **init_kwargs(strat, views))
                    score = evaluate(u, y, task)
                    metric = "acc" if task == "classification" else "R2"
                    print(f"    {strat:<15} {metric}={score:.4f}", flush=True)
                except Exception as e:
                    print(f"    {strat:<15} FAILED: {e!r}", flush=True)


if __name__ == "__main__":
    main()
