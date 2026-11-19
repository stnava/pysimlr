"""Synthetic benchmark of uniform shrinkage (`pysimlr.uniform_shrinkage`).

Planted model: one latent z; every feature of every view loads positively on z
(weak background loading, stronger loading on a focal fifth of the features),
unit Gaussian noise. Three scorers are compared by the absolute out-of-sample
correlation between the consensus score and the planted z:

* ``plain_mean``: mean of the training-standardized features per view, averaged over views.
* ``unshrunk``: ``simlr`` with ``alpha=None`` (no smoother), same sparseness and every other setting.
* ``shrunk``: ``simlr`` with ``alpha="auto"``.

Both fits use the regime's ``sparseness`` (same in both arms) and are scored with ``project_uniform_shrunk(post_shrink=0)``
(the fitted simplex weights only), so the only difference between the two arms is the
smoother. ``post_shrink`` and ``sparseness`` are separate knobs and are not part of this comparison. Differences are paired
by seed; the interval is a percentile bootstrap over seeds.

Usage: ``python scripts/uniform_shrinkage_benchmark.py [--n-seeds 20 --n-boot 2000]``;
``--smoke`` sets every count to 2 and runs all stages (fit, scoring, bootstrap, JSON, HTML).
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
    "n_lt_p": {"n_train": 40, "widths": (100, 120, 100), "sparseness": None},
    "n_lt_p_sparse0.3": {"n_train": 40, "widths": (100, 120, 100), "sparseness": 0.3},
    "n_gg_p": {"n_train": 600, "widths": (20, 25, 15), "sparseness": None},
}
BACKGROUND, FOCAL = 0.10, 0.20  # loadings: all features, then the first fifth of each view


def _make(seed: int, n_train: int, widths, n_test: int, noise: float = 1.0):
    rng = np.random.default_rng(seed)
    z_tr, z_te = rng.normal(size=n_train), rng.normal(size=n_test)
    v_tr, v_te = [], []
    for p in widths:
        load = np.full(p, BACKGROUND)
        load[: p // 5] = FOCAL
        a = np.outer(z_tr, load) + noise * rng.normal(size=(n_train, p))
        b = np.outer(z_te, load) + noise * rng.normal(size=(n_test, p))
        mu, sd = a.mean(0), a.std(0)
        v_tr.append((a - mu) / sd)
        v_te.append((b - mu) / sd)
    return v_tr, v_te, z_te


def _corr(u, z) -> float:
    return abs(float(np.corrcoef(np.asarray(u).ravel(), z)[0, 1]))


def one_seed(seed: int, regime: dict, iters: int, n_test: int = 400) -> dict:
    v_tr, v_te, z_te = _make(seed, regime["n_train"], regime["widths"], n_test)
    out = {"plain_mean": _corr(np.mean([v.mean(1) for v in v_te], axis=0), z_te)}
    for arm, alpha in (("unshrunk", [None] * len(v_tr)), ("shrunk", "auto")):
        kw = pysimlr.uniform_shrinkage_params(v_tr, alpha=alpha, sparseness=regime["sparseness"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = pysimlr.fit("simlr", v_tr, 1, random_state=seed, iterations=iters, **kw)
        out[arm] = _corr(pysimlr.project_uniform_shrunk(fit, v_te, post_shrink=0.0)["u"][:, 0], z_te)
    return out


def _boot_ci(d: np.ndarray, n_boot: int, rng) -> list[float]:
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    m = d[idx].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def run_benchmark(n_seeds: int, n_boot: int, iters: int, smoke: bool = False) -> dict:
    rng = np.random.default_rng(0)
    res = {"smoke": smoke, "n_seeds": n_seeds, "n_boot": n_boot, "iters": iters, "regimes": {}}
    for name, reg in REGIMES.items():
        rows = [one_seed(s, reg, iters) for s in range(n_seeds)]
        arr = {k: np.array([r[k] for r in rows]) for k in rows[0]}
        d = arr["shrunk"] - arr["unshrunk"]
        d_pm = arr["shrunk"] - arr["plain_mean"]
        res["regimes"][name] = {
            "n_train": reg["n_train"], "sparseness": reg["sparseness"], "widths": list(reg["widths"]),
            "mean_abs_r": {k: float(v.mean()) for k, v in arr.items()},
            "shrunk_minus_unshrunk": {"mean": float(d.mean()), "ci95": _boot_ci(d, n_boot, rng),
                                      "frac_seeds_shrunk_better": float((d > 0).mean())},
            "shrunk_minus_plain_mean": {"mean": float(d_pm.mean()), "ci95": _boot_ci(d_pm, n_boot, rng)},
            "per_seed": rows,
        }
    return res


def _html(res: dict) -> str:
    rows = "".join(
        f"<tr><td>{n}</td><td>{r['n_train']}</td><td>{r['mean_abs_r']['plain_mean']:.3f}</td>"
        f"<td>{r['mean_abs_r']['unshrunk']:.3f}</td><td>{r['mean_abs_r']['shrunk']:.3f}</td>"
        f"<td>{r['shrunk_minus_unshrunk']['mean']:+.3f} [{r['shrunk_minus_unshrunk']['ci95'][0]:+.3f}, "
        f"{r['shrunk_minus_unshrunk']['ci95'][1]:+.3f}]</td></tr>" for n, r in res["regimes"].items())
    return ("<!doctype html><meta charset='utf-8'><title>Uniform shrinkage benchmark</title>"
            f"<h1>Uniform shrinkage benchmark (synthetic){' [smoke]' if res['smoke'] else ''}</h1>"
            f"<p>seeds={res['n_seeds']}, bootstrap={res['n_boot']}, iterations={res['iters']}</p>"
            "<table border=1 cellpadding=4><tr><th>regime</th><th>n_train</th><th>plain mean</th>"
            f"<th>unshrunk</th><th>shrunk</th><th>shrunk - unshrunk (95% CI)</th></tr>{rows}</table>")


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
