#!/usr/bin/env python3
"""Time the deep first-layer Stiefel projection on MPS: SVD round-trip vs nsa_flow.polar_factor.

`deep._svd_project_columns` computes U V' from an SVD, which MPS lacks, so every basis access
copies the (p x k) basis to the CPU and back. `nsa_flow.polar_factor` (>= 3.6.0) gives the same
matrix, Y (Y'Y)^(-1/2), from an eigendecomposition of the k x k Gram matrix only.

Arms: ned and lend, n = 2000, 20 epochs, device="mps", 2 repeats each, interleaved.
Rule (fixed before running): adopt the polar path on MPS if it is faster by >= 15% in median
wall time for both methods, and the held-out latents agree with the SVD arm (min per-component
|corr| >= 0.99). Otherwise record the numbers and keep the SVD path.

Writes docs/audit/evidence/mps_polar_timing.csv.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import nsa_flow  # noqa: E402
import pysimlr  # noqa: E402
from pysimlr import deep  # noqa: E402

_svd = deep._svd_project_columns


def _polar(u):
    with torch.no_grad():
        return nsa_flow.polar_factor(u)


def data(seed=0, n=2000, k=4, ps=(300, 200, 150)):
    g = np.random.default_rng(seed)
    z = g.standard_normal((n, k))
    return [torch.tensor(z @ g.standard_normal((k, p)) + 0.5 * g.standard_normal((n, p))).float() for p in ps]


def run(method, arm, xs):
    deep._svd_project_columns = _svd if arm == "svd" else _polar
    try:
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        t0 = time.perf_counter()
        res = pysimlr.fit(method, xs, k=4, iterations=20, tol=0.0, random_state=0, device="mps")
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        dt = time.perf_counter() - t0
    finally:
        deep._svd_project_columns = _svd
    return dt, res


def main():
    xs = data()
    test = data(seed=1)
    rows = []
    for method in ("ned", "lend"):
        lat = {}
        for rep in range(2):
            for arm in ("svd", "polar"):
                dt, res = run(method, arm, xs)
                lat[arm] = np.asarray(pysimlr.predict(res, test)["u"])
                rows.append({"method": method, "arm": arm, "repeat": rep, "seconds": dt})
                print(method, arm, rep, f"{dt:.1f}s", flush=True)
        c = [abs(np.corrcoef(lat["svd"][:, j], lat["polar"][:, j])[0, 1]) for j in range(lat["svd"].shape[1])]
        rows.append({"method": method, "arm": "agreement", "repeat": -1, "seconds": float(np.min(c))})
    df = pd.DataFrame(rows)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "audit", "evidence", "mps_polar_timing.csv")
    df.to_csv(out, index=False)
    t = df[df.arm != "agreement"].groupby(["method", "arm"]).seconds.median().unstack()
    t["speedup"] = 1 - t["polar"] / t["svd"]
    print(t)
    print(df[df.arm == "agreement"])
    ok = (t["speedup"] >= 0.15).all() and (df[df.arm == "agreement"].seconds >= 0.99).all()
    print("RULE:", "ADOPT" if ok else "KEEP SVD")


if __name__ == "__main__":
    main()
