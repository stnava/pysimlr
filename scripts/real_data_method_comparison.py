"""One run per (dataset, method) on the same four real datasets as
`real_data_init_comparison.py`, comparing method families rather than
initialization strategies: PCA, GCCA, SiMLR, LEND, NED, Flow-SiMR, Flow-SiMR-V.

Downstream scoring uses a random forest (`RandomForestClassifier` /
`RandomForestRegressor`) instead of the linear/logistic model used in the
initialization comparison -- a nonlinear reader on top of each method's
latent code, so a method that packs a lot of curved structure into few
dimensions is not penalized for it the way a linear probe would.

Single run per cell, one seed -- a spot check, not a seeded benchmark. Prints
a line immediately after each run finishes (flush=True).
"""
import sys

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "scripts")
from real_data_init_comparison import load_heart, load_diabetes, load_mfeat, load_brca

from pysimlr.simlr import simlr, initialize_simlr
from pysimlr.consensus import compute_shared_consensus
from pysimlr.deep import lend_simr, ned_simr
from pysimlr.flows import flow_simr, flow_simr_v

SEED = 42
K = 3
#: Generous caps, not fixed budgets -- every deep method here has a real
#: ConvergenceMonitor (relative-improvement early stopping, patience=10) and
#: returns its best-loss iterate, not its last. Measured on Heart Disease:
#: NED and LEND both report `stop_reason='converged'` around epoch 60-61 at
#: the default epochs=150 cap, so a *fixed* epochs=60 budget used earlier in
#: this session was right at the edge for a small, easy dataset and had no
#: margin at all for harder/larger ones (KIRC, mfeat, ADNI) -- there was no
#: check that 60 was actually enough, only that it was fast. `epochs=200`
#: (above the 150 default) and `iterations=150` (above SiMLR's 100 default)
#: give real margin; `_check_converged` below verifies the run actually
#: stopped on its own criterion rather than hitting the cap.
DEEP_EPOCHS = 200
#: Flow-SiMR/Flow-SiMR-V specifically get more: measured on ADNI-DX in pass 1
#: of this session's benchmark re-establishment, Flow-SiMR-V[gcca] hit the
#: 200-epoch cap with `best_epoch=199` (still improving, not a plateau) --
#: verified it actually converges (`stop_reason='converged'`, best_epoch=285)
#: once given epochs=400. LEND/NED converge well within 200 on every dataset
#: checked, so they keep the lower, faster cap rather than paying the cost
#: of 400 everywhere.
FLOW_EPOCHS = 400
SIMLR_ITERATIONS = 150
#: `None` resolves to a budget-scaled warmup (`min(20, epochs // 4)`) rather
#: than a flat number -- see `resolve_warmup_epochs`'s docstring for why a
#: flat warmup can silently disable the alignment term on a short run.
DEEP_WARMUP = None

DATASETS = [
    ("Heart Disease", load_heart),
    ("Diabetes", load_diabetes),
    ("mfeat", load_mfeat),
    ("TCGA-BRCA", load_brca),
]


def run_pca(views):
    z = np.concatenate([v.numpy() for v in views], axis=1)
    return torch.tensor(PCA(n_components=K, random_state=SEED).fit_transform(z)).float()


def run_gcca(views):
    v_mats = initialize_simlr(views, K, initialization_type="gcca")
    zs = [x @ v for x, v in zip(views, v_mats)]
    return compute_shared_consensus(zs, mixing_algorithm="svd", k=K)


def _check_converged(label, out):
    """Print a note (not a failure) when a run hit its epoch/iteration cap
    rather than stopping on its own convergence criterion -- the thing a
    fixed small budget could never tell us."""
    conv = out.get("convergence")
    if conv is not None:
        if not conv.get("converged", False):
            print(f"    [!] {label} did not converge within its cap "
                 f"(best_epoch={conv.get('best_epoch')}, max_steps={conv.get('max_steps')})", flush=True)
        return
    stop_reason = out.get("stop_reason")
    if stop_reason == "max_iter":
        print(f"    [!] {label} hit max_iter without converging", flush=True)


def run_simlr(views):
    out = simlr(views, k=K, iterations=SIMLR_ITERATIONS, initialization_type="pca", verbose=False)
    _check_converged("SiMLR", out)
    return out["u"]


def run_lend(views):
    torch.manual_seed(SEED)
    out = lend_simr(views, k=K, epochs=DEEP_EPOCHS, warmup_epochs=DEEP_WARMUP, verbose=False)
    _check_converged("LEND", out)
    return out["u"]


def run_ned(views):
    torch.manual_seed(SEED)
    out = ned_simr(views, k=K, epochs=DEEP_EPOCHS, warmup_epochs=DEEP_WARMUP, verbose=False)
    _check_converged("NED", out)
    return out["u"]


def run_flow_simr(views):
    torch.manual_seed(SEED)
    out = flow_simr(views, k=K, epochs=FLOW_EPOCHS, warmup_epochs=DEEP_WARMUP,
                    device="cpu", verbose=False)
    _check_converged("Flow-SiMR", out)
    return out["u"]


def run_flow_simr_v(views):
    torch.manual_seed(SEED)
    out = flow_simr_v(views, k=K, epochs=FLOW_EPOCHS, warmup_epochs=DEEP_WARMUP,
                      device="cpu", verbose=False)
    _check_converged("Flow-SiMR-V", out)
    return out["u"]


METHODS = [
    ("PCA", run_pca),
    ("GCCA", run_gcca),
    ("SiMLR", run_simlr),
    ("LEND", run_lend),
    ("NED", run_ned),
    ("Flow-SiMR", run_flow_simr),
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


def main():
    for name, loader in DATASETS:
        print(f"\n=== {name} ===", flush=True)
        try:
            views, y, task = loader()
        except Exception as e:
            print(f"  LOAD FAILED: {e!r}", flush=True)
            continue
        print(f"  N={views[0].shape[0]}, views={[v.shape[1] for v in views]}, task={task}", flush=True)
        for mname, fn in METHODS:
            try:
                u = fn(views)
                score = evaluate(u, y, task)
                metric = "acc" if task == "classification" else "R2"
                print(f"  {mname:<12} {metric}={score:.4f}", flush=True)
            except Exception as e:
                print(f"  {mname:<12} FAILED: {e!r}", flush=True)


if __name__ == "__main__":
    main()
