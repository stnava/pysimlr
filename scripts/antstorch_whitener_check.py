"""Verification of upstream ANTsTorch LAMNr-Flows whitener (Rule R10).

Evaluates whether the upstream ANTsTorch whitener defect (formerly off the
identity by 3.2 due to L^T triangular solve) is fixed in commit ed39d99.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from antstorch.lamnr_flows.apply_lamnr_flows_whitener import apply_lamnr_flows_whitener
from antstorch.lamnr_flows.lamnr_flows_whitener import lamnr_flows_whitener


def run_whitener_check(seed: int = 42, n: int = 2000, p: int = 8, cond: float = 50.0) -> dict[str, float | str]:
    antstorch_repo = Path("/Users/stnava/data/repos/ANTsTorch")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=antstorch_repo).decode().strip()

    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((p, p)))
    evals = np.geomspace(1.0, cond, p)
    cov = Q @ np.diag(evals) @ Q.T
    L = np.linalg.cholesky(cov)
    X = rng.standard_normal((n, p)) @ L.T
    df = pd.DataFrame(X, columns=[f"v_{i}" for i in range(p)])

    # Small configuration running well under 2 minutes
    res = lamnr_flows_whitener([df], max_iter=50, batch_size=256, verbose=False, seed=seed)
    mapped = apply_lamnr_flows_whitener(res, [df], output_space="whitened_full")

    if isinstance(mapped, list):
        Z = mapped[0].values
    else:
        Z = mapped.values

    cov_Z = np.cov(Z, rowvar=False)
    frob_diff = float(np.linalg.norm(cov_Z - np.eye(p), "fro"))
    max_offdiag = float(np.max(np.abs(cov_Z - np.diag(np.diag(cov_Z)))))

    rule_r10_pass = bool(frob_diff < 0.25 and max_offdiag < 0.1)

    return {
        "antstorch_commit": commit,
        "n": n,
        "p": p,
        "condition_number": cond,
        "frob_norm_diff": frob_diff,
        "max_off_diagonal": max_offdiag,
        "rule_r10_pass": rule_r10_pass,
    }


def main():
    parser = argparse.ArgumentParser(description="ANTsTorch whitener check (Rule R10)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = run_whitener_check(seed=args.seed)
    print("ANTsTorch Commit:", results["antstorch_commit"])
    print(f"||cov - I||_F: {results['frob_norm_diff']:.4f} (threshold: < 0.25)")
    print(f"Max |off-diagonal|: {results['max_off_diagonal']:.4f} (threshold: < 0.10)")
    print("Rule R10 Status:", "PASSED" if results["rule_r10_pass"] else "FAILED")


if __name__ == "__main__":
    main()
