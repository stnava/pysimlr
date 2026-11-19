"""Synthetic benchmark of uniform & two-stage support-restricted shrinkage (`pysimlr.uniform_shrinkage`).

Evaluates four planted multi-view regimes:
- `n_lt_p`: k=1 dense positive loadings, n=40, widths=(100, 120, 100), sparseness=None.
- `n_lt_p_sparse0.3`: k=1 sparse positive loadings (30% active), n=40, widths=(100, 120, 100), sparseness=0.3.
- `n_gg_p`: k=1 large-n regime, n=600, widths=(20, 25, 15), sparseness=None.
- `k2_disjoint`: k=2 disjoint non-negative modes (z1 perp z2), n=60, widths=(80, 100, 60), sparseness=0.3.

Usage: ``python scripts/uniform_shrinkage_benchmark.py [--n-seeds 20 --n-boot 2000]``;
``--smoke`` sets every count to 2 and runs all stages (<10s, JSON + HTML).
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np

import pysimlr

REGIMES = {
    "n_lt_p": {"k": 1, "n_train": 40, "widths": (100, 120, 100), "sparseness": None, "mode": "dense"},
    "n_lt_p_sparse0.3": {"k": 1, "n_train": 40, "widths": (100, 120, 100), "sparseness": 0.3, "mode": "sparse"},
    "n_gg_p": {"k": 1, "n_train": 600, "widths": (20, 25, 15), "sparseness": None, "mode": "dense"},
    "k2_disjoint": {"k": 2, "n_train": 60, "widths": (80, 100, 60), "sparseness": 0.3, "mode": "k2_disjoint"},
}
BACKGROUND, FOCAL = 0.10, 0.20


def _make_k1(seed: int, n_train: int, widths, n_test: int, mode: str, noise: float = 1.0):
    rng = np.random.default_rng(seed)
    z_tr, z_te = rng.normal(size=n_train), rng.normal(size=n_test)
    v_tr, v_te = [], []
    for p in widths:
        if mode == "sparse":
            load = np.zeros(p, dtype=np.float64)
            load[: max(2, round(0.30 * p))] = 0.45
        else:
            load = np.full(p, BACKGROUND, dtype=np.float64)
            load[: p // 5] = FOCAL
        a = np.outer(z_tr, load) + noise * rng.normal(size=(n_train, p))
        b = np.outer(z_te, load) + noise * rng.normal(size=(n_test, p))
        mu, sd = a.mean(0), np.where(a.std(0) > 1e-12, a.std(0), 1.0)
        v_tr.append((a - mu) / sd)
        v_te.append((b - mu) / sd)
    return v_tr, v_te, z_te


def _make_k2(seed: int, n_train: int, widths, n_test: int, noise: float = 1.0):
    rng = np.random.default_rng(seed)
    z_tr = rng.normal(size=(n_train, 2))
    z_te = rng.normal(size=(n_test, 2))
    for Z in (z_tr, z_te):
        Z -= Z.mean(axis=0, keepdims=True)
        q_mat, _ = np.linalg.qr(Z)
        Z[:] = q_mat * np.sqrt(Z.shape[0])
    v_tr, v_te = [], []
    for p in widths:
        m = max(2, round(0.25 * p))
        W = np.zeros((p, 2), dtype=np.float64)
        W[:m, 0] = 0.55
        W[m : 2 * m, 1] = 0.45
        a = z_tr @ W.T + noise * rng.normal(size=(n_train, p))
        b = z_te @ W.T + noise * rng.normal(size=(n_test, p))
        mu, sd = a.mean(0), np.where(a.std(0) > 1e-12, a.std(0), 1.0)
        v_tr.append((a - mu) / sd)
        v_te.append((b - mu) / sd)
    return v_tr, v_te, z_te


def _corr(u, z) -> float:
    return abs(float(np.corrcoef(np.asarray(u).ravel(), np.asarray(z).ravel())[0, 1]))


def _k2_mean_corr(u_mat: np.ndarray, z_mat: np.ndarray) -> float:
    c00, c11 = _corr(u_mat[:, 0], z_mat[:, 0]), _corr(u_mat[:, 1], z_mat[:, 1])
    c01, c10 = _corr(u_mat[:, 0], z_mat[:, 1]), _corr(u_mat[:, 1], z_mat[:, 0])
    return 0.5 * max(c00 + c11, c01 + c10)


def one_seed(seed: int, regime: dict, iters: int, n_test: int = 400, lr: float | str | None = None) -> dict:
    extra = {"learning_rate": lr} if lr is not None else {}
    if regime["k"] == 1:
        v_tr, v_te, z_te = _make_k1(seed, regime["n_train"], regime["widths"], n_test, regime["mode"])
        out = {"plain_mean": _corr(np.mean([v.mean(1) for v in v_te], axis=0), z_te)}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kw_un = pysimlr.uniform_shrinkage_params(v_tr, alpha=[None] * len(v_tr), sparseness=regime["sparseness"])
            fit_un = pysimlr.fit("simlr", v_tr, 1, random_state=seed, iterations=iters, **kw_un, **extra)
            fit_sh = pysimlr.fit_uniform_shrunk(
                v_tr, k=1, alpha="auto", sparseness=regime["sparseness"],
                random_state=seed, iterations=iters, **extra,
            )
        out["unshrunk"] = _corr(pysimlr.project_uniform_shrunk(fit_un, v_te, post_shrink=0.0)["u"][:, 0], z_te)
        out["shrunk"] = _corr(pysimlr.project_uniform_shrunk(fit_sh, v_te, post_shrink=0.0)["u"][:, 0], z_te)
        return out

    v_tr, v_te, z_te = _make_k2(seed, regime["n_train"], regime["widths"], n_test)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_un = pysimlr.fit(
            "simlr", v_tr, 2, scale_list=["none"], positivity="conical",
            sparseness=regime["sparseness"], energy_type="acc", mixing_algorithm="pca",
            nsa_w=0.5, initialization_type="nndsvd", random_state=seed, iterations=iters, **extra,
        )
        fit_sh = pysimlr.fit_uniform_shrunk(
            v_tr, k=2, alpha="auto", sparseness=regime["sparseness"],
            random_state=seed, iterations=iters, **extra,
        )
    u_un = pysimlr.project_uniform_shrunk(fit_un, v_te, post_shrink=0.0)["u"]
    u_sh = pysimlr.project_uniform_shrunk(fit_sh, v_te, post_shrink=0.0)["u"]
    return {
        "plain_mean": 0.5 * _corr(np.mean([v.mean(1) for v in v_te], axis=0), z_te[:, 0]),
        "unshrunk": _k2_mean_corr(u_un, z_te),
        "shrunk": _k2_mean_corr(u_sh, z_te),
    }


def _boot_ci(d: np.ndarray, n_boot: int, rng) -> list[float]:
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    m = d[idx].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def run_benchmark(n_seeds: int, n_boot: int, iters: int, smoke: bool = False) -> dict:
    rng = np.random.default_rng(0)
    lr = 0.05 if smoke else None
    res = {"smoke": smoke, "n_seeds": n_seeds, "n_boot": n_boot, "iters": iters, "regimes": {}}
    for name, reg in REGIMES.items():
        rows = [one_seed(s, reg, iters, lr=lr) for s in range(n_seeds)]
        arr = {k: np.array([r[k] for r in rows]) for k in rows[0]}
        d = arr["shrunk"] - arr["unshrunk"]
        d_pm = arr["shrunk"] - arr["plain_mean"]
        res["regimes"][name] = {
            "k": reg["k"],
            "n_train": reg["n_train"],
            "sparseness": reg["sparseness"],
            "widths": list(reg["widths"]),
            "mean_abs_r": {k: float(v.mean()) for k, v in arr.items()},
            "shrunk_minus_unshrunk": {
                "mean": float(d.mean()),
                "ci95": _boot_ci(d, n_boot, rng),
                "frac_seeds_shrunk_better": float((d > 0).mean()),
            },
            "shrunk_minus_plain_mean": {"mean": float(d_pm.mean()), "ci95": _boot_ci(d_pm, n_boot, rng)},
            "per_seed": rows,
        }
    return res


def _html(res: dict) -> str:
    rows = "".join(
        f"<tr><td>{n}</td><td>{r['k']}</td><td>{r['n_train']}</td><td>{r['mean_abs_r']['plain_mean']:.3f}</td>"
        f"<td>{r['mean_abs_r']['unshrunk']:.3f}</td><td>{r['mean_abs_r']['shrunk']:.3f}</td>"
        f"<td>{r['shrunk_minus_unshrunk']['mean']:+.3f} [{r['shrunk_minus_unshrunk']['ci95'][0]:+.3f}, "
        f"{r['shrunk_minus_unshrunk']['ci95'][1]:+.3f}]</td></tr>"
        for n, r in res["regimes"].items()
    )
    return (
        "<!doctype html><meta charset='utf-8'><title>Uniform shrinkage benchmark</title>"
        f"<h1>Uniform &amp; two-stage shrinkage benchmark (synthetic){' [smoke]' if res['smoke'] else ''}</h1>"
        f"<p>seeds={res['n_seeds']}, bootstrap={res['n_boot']}, iterations={res['iters']}</p>"
        "<table border=1 cellpadding=4><tr><th>regime</th><th>k</th><th>n_train</th><th>plain mean</th>"
        f"<th>unshrunk</th><th>shrunk</th><th>shrunk - unshrunk (95% CI)</th></tr>{rows}</table>"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--out-dir", default="reports")
    ap.add_argument("--smoke", action="store_true", help="all counts = 2, all stages run")
    a = ap.parse_args()
    if a.smoke:
        a.n_seeds = a.n_boot = a.iters = 2
    t0 = time.time()
    res = run_benchmark(a.n_seeds, a.n_boot, a.iters, smoke=a.smoke)
    res["seconds"] = time.time() - t0
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = "uniform_shrinkage_benchmark" + ("_smoke" if a.smoke else "")
    (out / f"{stem}.json").write_text(json.dumps(res, indent=2))
    (out / f"{stem}.html").write_text(_html(res))
    for n, r in res["regimes"].items():
        print(n, r["mean_abs_r"], "shrunk-unshrunk", r["shrunk_minus_unshrunk"])
    print(f"wrote {out / stem}.json ({res['seconds']:.1f}s)")


if __name__ == "__main__":
    main()
