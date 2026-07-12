import torch
import pandas as pd
import numpy as np
from pysimlr.utils import adjusted_rvcoef
from sklearn.preprocessing import StandardScaler

def raw_rv():
    bl = pd.read_csv('data/expart_bl.csv').values
    mh = pd.read_csv('data/expart_mh.csv').values
    mr = pd.read_csv('data/expart_mr.csv').values
    pe = pd.read_csv('data/expart_pe.csv').values
    
    scaler = StandardScaler()
    mats = [scaler.fit_transform(m) for m in [bl, mh, mr, pe]]
    names = ['BL', 'MH', 'MR', 'PE']
    
    print("Raw Adjusted RV Coefficients (All-to-All):")
    for i in range(4):
        for j in range(i+1, 4):
            rv = adjusted_rvcoef(torch.from_numpy(mats[i]), torch.from_numpy(mats[j]))
            print(f"  {names[i]} <-> {names[j]}: {rv:.4f}")

if __name__ == "__main__":
    raw_rv()
