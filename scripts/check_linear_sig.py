import torch
import pandas as pd
import numpy as np
from pysimlr import simlr, simlr_perm
from sklearn.preprocessing import StandardScaler

def check_linear_sig():
    bl = pd.read_csv("data/expart_bl.csv").values
    mh = pd.read_csv("data/expart_mh.csv").values
    mr = pd.read_csv("data/expart_mr.csv").values
    pe = pd.read_csv("data/expart_pe.csv").values
    data = [bl, mh, mr, pe]
    
    print("Running Linear SiMLR Permutation Test (Star Topology)...")
    res = simlr_perm(data, k=3, n_perms=50, energy_type="acc", sparseness_quantile=0.5)
    
    print("\nLINEAR SIGNIFICANCE REPORT:")
    print("="*40)
    for edge, s in res["stats"].items():
        print(f"{edge:10} | Obs: {s['observed']:.4f} | p: {s['p_value']:.4f}")

if __name__ == "__main__":
    check_linear_sig()
