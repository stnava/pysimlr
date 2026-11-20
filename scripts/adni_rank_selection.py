"""ADNI-MCI rank selection (`pysimlr.select_rank`) for k = 2..5 across both sign modes.

Passes `ks` together to `pysimlr.select_rank` per positivity mode (`"either"`,
`"positive"`), sharing one precomputed `bottleneck_rank`, and writes
``docs/audit/evidence/adni_rank_selection.csv`` plus
``docs/audit/evidence/adni_rank_selection_provenance.json``.
"""
import argparse
import contextlib
import datetime
import json
import os
import platform
import subprocess
import sys
import tempfile
import warnings

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))


def _fmt_vec(vals):
    if vals is None:
        return ""
    return " ".join(f"{float(v):.6f}" for v in vals)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=None, help="Output CSV path (defaults to docs/audit/evidence/adni_rank_selection.csv, or temp path under --smoke)")
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 3, 4, 5])
    ap.add_argument("--n-subsamples", type=int, default=20)
    ap.add_argument("--n-baseline", type=int, default=3)
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--smoke", action="store_true", help="Fast <10s smoke check (ks=[2, 3], counts=2, non-destructive default out)")
    a = ap.parse_args()

    ks = [2, 3] if a.smoke else list(a.ks)
    n_sub = 2 if a.smoke else a.n_subsamples
    n_base = 2 if a.smoke else a.n_baseline
    iters = 2 if a.smoke else a.iterations
    n_perm_bott = 2 if a.smoke else 100
    n_jobs = 1 if a.smoke else a.n_jobs
    out_csv = a.out or (
        os.path.join(tempfile.gettempdir(), "pysimlr_smoke_adni_rank_selection.csv")
        if a.smoke else os.path.join(REPO, "docs", "audit", "evidence", "adni_rank_selection.csv")
    )

    from concurrent.futures import ProcessPoolExecutor
    import pandas as pd
    import torch
    import pysimlr
    import run_deep_benchmark as R

    pd.set_option("display.width", 220)
    c = R.get_adni_mci_case(42)

    pool_ctx = ProcessPoolExecutor(max_workers=n_jobs) if n_jobs > 1 else contextlib.nullcontext()
    with warnings.catch_warnings(), pool_ctx as ex:
        warnings.simplefilter("ignore")
        bott = pysimlr.bottleneck_rank(c["data"], n_permutations=n_perm_bott, bound="max")
        rows = []
        selected = {}
        for pos in ("either", "positive"):
            kw = dict(
                n_subsamples=n_sub,
                n_baseline=n_base,
                random_state=42,
                positivity=pos,
                topology="graph",
                path_graph=c["path_graph"],
                n_jobs=n_jobs,
                _executor=ex,
                _bottleneck=bott,
            )
            if a.smoke:
                kw["tune_learning_rate"] = "reference"
                kw["learning_rate"] = 0.05
            if iters is not None:
                kw["iterations"] = iters
            out = pysimlr.select_rank("simlr", c["data"], ks, **kw)
            selected[pos] = out.get("k")
            margin_val = float(out.get("margin", 0.2))
            for r in out["table"]:
                rows.append({
                    "pos": pos,
                    "k": int(r["k"]),
                    "min_latent_reliability": float(r["min_latent_reliability"]),
                    "min_latent_reliability_gain": float(r["min_latent_reliability_gain"]),
                    "latent_reliability_multivariate": float(r["latent_reliability_multivariate"]),
                    "latent_reliability_multivariate_gain": float(r["latent_reliability_multivariate_gain"]),
                    "latent_reliability_passed": bool(r["latent_reliability_passed"]),
                    "latent_reliability": _fmt_vec(r["latent_reliability"]),
                    "latent_reliability_baseline": _fmt_vec(r["latent_reliability_baseline"]),
                    "latent_reliability_gain": _fmt_vec(r["latent_reliability_gain"]),
                    "cross_view_coupling": _fmt_vec(r["cross_view_coupling"]),
                    "min_cross_view_coupling": float(r["min_cross_view_coupling"]),
                    "all_coupled": bool(r["all_coupled"]),
                    "basis_degenerate": bool(r["basis_degenerate"]),
                    "margin": margin_val,
                    "selected_k": (int(out["k"]) if out.get("k") is not None else pd.NA),
                })

    d = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    d.to_csv(out_csv, index=False)

    prov_path = os.path.splitext(out_csv)[0] + "_provenance.json"
    try:
        commit = subprocess.run(
            ["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    cli_rel = " ".join([os.path.relpath(os.path.abspath(sys.argv[0]), REPO)] + sys.argv[1:])
    prov = {
        "cli": cli_rel,
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        "platform": f"{platform.system()} {platform.machine()}",
        "smoke": bool(a.smoke),
        "git_commit": commit,
        "pysimlr": pysimlr.__version__,
        "torch": torch.__version__,
        "ks": ks,
        "n_subsamples": n_sub,
        "n_baseline": n_base,
        "iterations": iters,
        "bottleneck_k_max": int(bott["k_max"]) if bott is not None else None,
        "selected_k": selected,
    }
    with open(prov_path, "w") as f:
        json.dump(prov, f, indent=2)

    print(d.to_string(index=False))
    for p in ("either", "positive"):
        print(p, "-> selected_k:", selected[p])


if __name__ == "__main__":
    main()
