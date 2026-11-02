"""Every user-facing default of every registered method, side by side.

Reads `pysimlr.describe_method` (signature defaults plus registry presets), so
the table is the code, not a description of it. Prints markdown and writes
`docs/audit/evidence/method_settings.csv`.
"""
import csv
from pathlib import Path

from pysimlr import api

KEYS = ["iterations", "tol", "patience", "learning_rate", "optimizer_type", "energy_type",
        "mixing_algorithm", "topology", "sparseness", "positivity", "nsa_w",
        "collapse_weight", "batch_size", "warmup_epochs", "scale_list", "initialization_type"]


def main():
    names = [n for n in api.method_names() if api.describe_method(n)["family"] != "baseline"]
    table = {}
    for n in names:
        d = api.describe_method(n)
        params = d["parameters"]
        row = {}
        for kk in KEYS:
            if kk in d["presets"]:
                row[kk] = repr(d["presets"][kk]) + " (preset)"
            elif kk in params:
                p = params[kk]
                row[kk] = repr(p.get("default") if isinstance(p, dict) else p)
            else:
                row[kk] = "-"
        table[n] = row
    out = Path(__file__).resolve().parents[1] / "docs/audit/evidence/method_settings.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["parameter", *names])
        for kk in KEYS:
            w.writerow([kk, *(table[n][kk] for n in names)])
    print("| parameter | " + " | ".join(names) + " |")
    print("|" + "---|" * (len(names) + 1))
    for kk in KEYS:
        print(f"| {kk} | " + " | ".join(table[n][kk] for n in names) + " |")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
