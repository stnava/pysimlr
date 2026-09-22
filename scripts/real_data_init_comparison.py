"""One run per (dataset, initialization strategy) on the four real,
complementary datasets picked in this session: Cleveland Heart Disease,
Diabetes, UCI Multiple Features (mfeat), and TCGA-BRCA.

Prints a result line immediately after each run finishes (flush=True) so it
can be watched live rather than read only at the end.

Single run per cell, one seed, few iterations -- this is a spot check across
a complementary set of real datasets, not a seeded statistical benchmark like
`CORRECTNESS_AUDIT.md`'s or `scripts/compare_initializations.py`'s. Read it as
"does each strategy work and roughly how does it do here", not as a
significance claim.
"""
import io
import sys
import urllib.request

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pysimlr.simlr import simlr, INITIALIZATION_TYPES

#: `INITIALIZATION_TYPES` minus `"best_of_n"`, for benchmark scripts. Kept
#: implemented and tested (`initialize_simlr(..., initialization_type="best_of_n")`
#: still works, see its docstring), but excluded here on its empirical record:
#: 0 wins out of 21 method-by-dataset blocks across every full sweep run this
#: session (Heart, Diabetes, mfeat, ADNI+PPMI, TCGA-KIRC x SiMLR/LEND/NED/
#: Flow-SiMR-V), even after fixing `_score_joint_basis`'s scale bias. Its
#: scoring proxy doesn't track downstream task performance, so it isn't worth
#: a benchmark slot until that's fixed.
BENCHMARK_INITIALIZATION_TYPES = tuple(t for t in INITIALIZATION_TYPES if t != "best_of_n")

SEED = 42
K = 3
ITERATIONS = 20


def load_heart():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
    with urllib.request.urlopen(url, timeout=30) as r:
        raw = r.read().decode()
    df = pd.read_csv(io.StringIO(raw), names=cols).replace('?', np.nan).dropna().apply(pd.to_numeric)
    y = (df['num'] > 0).astype(int).to_numpy()
    x = StandardScaler().fit_transform(df.drop('num', axis=1))
    views = [torch.tensor(x[:, :7]).float(), torch.tensor(x[:, 7:]).float()]
    return views, y, "classification"


def load_diabetes():
    from sklearn.datasets import load_diabetes as _load
    d = _load()
    x = StandardScaler().fit_transform(d.data)
    views = [torch.tensor(x[:, :5]).float(), torch.tensor(x[:, 5:]).float()]
    return views, d.target, "regression"


def load_mfeat():
    files = ["mfeat-fou", "mfeat-kar", "mfeat-mor", "mfeat-zer"]
    views = []
    for f in files:
        m = np.loadtxt(f"data/mfeat/{f}")
        views.append(torch.tensor(StandardScaler().fit_transform(m)).float())
    y = np.repeat(np.arange(10), 200)
    return views, y, "classification"


def load_brca():
    sys.path.insert(0, "scripts")
    from tcga_loader import get_tcga_case
    case = get_tcga_case("BRCA", seed=SEED, k=K, max_features=300)
    views = [StandardScaler().fit_transform(v.numpy()) for v in case["data"]]
    views = [torch.tensor(v).float() for v in views]
    y = case["outcome"].numpy()
    return views, y, "classification"


def load_adni():
    sys.path.insert(0, "scripts")
    from adni_loader import get_adni_case
    from sklearn.preprocessing import LabelEncoder
    case = get_adni_case(max_features=300)
    views = [StandardScaler().fit_transform(v.numpy()) for v in case["views"]]
    views = [torch.tensor(v).float() for v in views]
    y = LabelEncoder().fit_transform(case["y"])
    return views, y, "classification"


def load_kirc():
    sys.path.insert(0, "scripts")
    from tcga_loader import get_tcga_case
    case = get_tcga_case("KIRC", seed=SEED, k=K, max_features=300)
    views = [StandardScaler().fit_transform(v.numpy()) for v in case["data"]]
    views = [torch.tensor(v).float() for v in views]
    y = case["outcome"].numpy()
    return views, y, "classification"


DATASETS = [
    ("Heart Disease", load_heart),
    ("Diabetes", load_diabetes),
    ("mfeat", load_mfeat),
    ("TCGA-BRCA", load_brca),
    ("ADNI+PPMI", load_adni),
    ("TCGA-KIRC", load_kirc),
]


def evaluate(u, y, task):
    z = u.numpy() if torch.is_tensor(u) else u
    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.3,
                                      random_state=SEED, stratify=y if task == "classification" else None)
    if task == "classification":
        model = LogisticRegression(max_iter=2000)
        model.fit(z[idx_tr], y[idx_tr])
        return float(model.score(z[idx_te], y[idx_te]))
    model = LinearRegression()
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
        for strat in BENCHMARK_INITIALIZATION_TYPES:
            kwargs = {}
            if strat == "random":
                kwargs["init_seed"] = SEED
            if strat in ("perturbed_pca", "best_of_n"):
                kwargs["init_seed"] = SEED
            if strat == "domain":
                rng = np.random.default_rng(SEED)
                kwargs["domain_matrices"] = [
                    torch.tensor(rng.standard_normal((K, v.shape[1]))).float() for v in views
                ]
            try:
                out = simlr(views, k=K, iterations=ITERATIONS,
                           initialization_type=strat, verbose=False, **kwargs)
                score = evaluate(out["u"], y, task)
                metric = "acc" if task == "classification" else "R2"
                print(f"  {strat:<15} {metric}={score:.4f}", flush=True)
            except Exception as e:
                print(f"  {strat:<15} FAILED: {e!r}", flush=True)


if __name__ == "__main__":
    main()
