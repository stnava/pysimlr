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

from pysimlr.simlr import simlr, initialize_simlr, predict_shared_latent
from pysimlr.consensus import compute_shared_consensus
from pysimlr.deep import lend_simr, ned_simr, predict_deep
from pysimlr.flows import flow_simr, flow_simr_v

SEED = 42
K = 3
DEEP_EPOCHS = 200
FLOW_EPOCHS = 400
SIMLR_ITERATIONS = 150
DEEP_WARMUP = None

DATASETS = [
    ("Heart Disease", load_heart),
    ("Diabetes", load_diabetes),
    ("mfeat", load_mfeat),
    ("TCGA-BRCA", load_brca),
]


def run_pca(views_tr, views_te):
    z_tr = np.concatenate([v.numpy() for v in views_tr], axis=1)
    z_te = np.concatenate([v.numpy() for v in views_te], axis=1)
    pca = PCA(n_components=K, random_state=SEED).fit(z_tr)
    return torch.tensor(pca.transform(z_tr)).float(), torch.tensor(pca.transform(z_te)).float()


def run_gcca(views_tr, views_te):
    v_mats = initialize_simlr(views_tr, K, initialization_type="gcca")
    zs_tr = [x @ v for x, v in zip(views_tr, v_mats)]
    zs_te = [x @ v for x, v in zip(views_te, v_mats)]
    u_tr, anchor = compute_shared_consensus(zs_tr, mixing_algorithm="svd", k=K, training=True)
    u_te = compute_shared_consensus(zs_te, mixing_algorithm="svd", k=K, training=False, anchor=anchor)
    return u_tr, u_te


def _check_converged(label, out):
    """Print a note (not a failure) when a run hit its epoch/iteration cap
    rather than stopping on its own convergence criterion."""
    conv = out.get("convergence")
    if conv is not None:
        if not conv.get("converged", False):
            print(f"    [!] {label} did not converge within its cap "
                  f"(best_epoch={conv.get('best_epoch')}, max_steps={conv.get('max_steps')})", flush=True)
        return
    stop_reason = out.get("stop_reason")
    if stop_reason == "max_iter":
        print(f"    [!] {label} hit max_iter without converging", flush=True)


def run_simlr(views_tr, views_te):
    out = simlr(views_tr, k=K, iterations=SIMLR_ITERATIONS, initialization_type="pca", verbose=False)
    _check_converged("SiMLR", out)
    return out["u"], predict_shared_latent(views_te, out)


def run_lend(views_tr, views_te):
    torch.manual_seed(SEED)
    out = lend_simr(views_tr, k=K, epochs=DEEP_EPOCHS, warmup_epochs=DEEP_WARMUP, verbose=False)
    _check_converged("LEND", out)
    return out["u"], predict_deep(views_te, out, device="cpu")["u"]


def run_ned(views_tr, views_te):
    torch.manual_seed(SEED)
    out = ned_simr(views_tr, k=K, epochs=DEEP_EPOCHS, warmup_epochs=DEEP_WARMUP, verbose=False)
    _check_converged("NED", out)
    return out["u"], predict_deep(views_te, out, device="cpu")["u"]


def run_flow_simr(views_tr, views_te):
    torch.manual_seed(SEED)
    out = flow_simr(views_tr, k=K, epochs=FLOW_EPOCHS, warmup_epochs=DEEP_WARMUP,
                    device="cpu", verbose=False)
    _check_converged("Flow-SiMR", out)
    return out["u"], predict_deep(views_te, out, device="cpu")["u"]


def run_flow_simr_v(views_tr, views_te):
    torch.manual_seed(SEED)
    out = flow_simr_v(views_tr, k=K, epochs=FLOW_EPOCHS, warmup_epochs=DEEP_WARMUP,
                      device="cpu", verbose=False)
    _check_converged("Flow-SiMR-V", out)
    return out["u"], predict_deep(views_te, out, device="cpu")["u"]


def run_lend_mai(views_tr, views_te):
    torch.manual_seed(SEED)
    out = lend_simr(views_tr, k=K, epochs=DEEP_EPOCHS, warmup_epochs=DEEP_WARMUP,
                    dynamic_weights=True, mai_metric="cca", verbose=False)
    _check_converged("LEND+MAI", out)
    return out["u"], predict_deep(views_te, out, device="cpu")["u"]


def run_flow_simr_v_mai(views_tr, views_te):
    torch.manual_seed(SEED)
    out = flow_simr_v(views_tr, k=K, epochs=FLOW_EPOCHS, warmup_epochs=DEEP_WARMUP,
                      dynamic_weights=True, mai_metric="cca", device="cpu", verbose=False)
    _check_converged("Flow-SiMR-V+MAI", out)
    return out["u"], predict_deep(views_te, out, device="cpu")["u"]


METHODS = [
    ("PCA", run_pca),
    ("GCCA", run_gcca),
    ("SiMLR", run_simlr),
    ("LEND", run_lend),
    ("LEND+MAI", run_lend_mai),
    ("NED", run_ned),
    ("Flow-SiMR", run_flow_simr),
    ("Flow-SiMR-V", run_flow_simr_v),
    ("Flow-SiMR-V+MAI", run_flow_simr_v_mai),
]


def evaluate(u_tr, u_te, y_tr, y_te, task):
    z_tr = u_tr.numpy() if torch.is_tensor(u_tr) else np.asarray(u_tr)
    z_te = u_te.numpy() if torch.is_tensor(u_te) else np.asarray(u_te)
    if task == "classification":
        model = RandomForestClassifier(n_estimators=300, random_state=SEED)
        model.fit(z_tr, y_tr)
        return float(model.score(z_te, y_te))
    model = RandomForestRegressor(n_estimators=300, random_state=SEED)
    model.fit(z_tr, y_tr)
    return float(model.score(z_te, y_te))


def main():
    for name, loader in DATASETS:
        print(f"\n=== {name} ===", flush=True)
        try:
            views, y, task = loader()
        except Exception as e:
            print(f"  LOAD FAILED: {e!r}", flush=True)
            continue
        print(f"  N={views[0].shape[0]}, views={[v.shape[1] for v in views]}, task={task}", flush=True)
        idx_tr, idx_te = train_test_split(
            np.arange(len(y)), test_size=0.3, random_state=SEED,
            stratify=y if task == "classification" else None,
        )
        views_tr = [v[idx_tr] for v in views]
        views_te = [v[idx_te] for v in views]
        y_tr, y_te = y[idx_tr], y[idx_te]
        for mname, fn in METHODS:
            try:
                u_tr, u_te = fn(views_tr, views_te)
                score = evaluate(u_tr, u_te, y_tr, y_te, task)
                metric = "acc" if task == "classification" else "R2"
                print(f"  {mname:<12} {metric}={score:.4f}", flush=True)
            except Exception as e:
                print(f"  {mname:<12} FAILED: {e!r}", flush=True)


if __name__ == "__main__":
    main()
