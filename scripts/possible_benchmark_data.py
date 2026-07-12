import pandas as pd
import numpy as np
import requests
import os

# Create directory if not exists
os.makedirs('./data/BRCA/', exist_ok=True)

# Direct URL to a pre-processed multi-view CSV (mRNA, miRNA, Protein, CNV)
url = "https://raw.githubusercontent.com/rbabaei82/MultiOmics_TCGA-BRCA/master/data.csv"
file_path = './data/BRCA/brca_multiomics.csv'

print("Downloading BRCA multi-omics data...")
r = requests.get(url)
with open(file_path, 'wb') as f:
    f.write(r.content)
print(f"Saved to {file_path}")

def load_brca_tabular(file_path):
    # Load the specific CSV file
    df = pd.read_csv(file_path)
    
    # This specific dataset uses prefixes: 
    # 'rs' (RNA-seq), 'cn' (Copy Number), 'mu' (Mutation), 'pp' (Protein)
    view_prefixes = ['rs_', 'cn_', 'mu_', 'pp_']
    views = []
    
    for pref in view_prefixes:
        cols = [c for c in df.columns if c.startswith(pref)]
        if cols:
            views.append(df[cols].values)
            print(f"View {pref} loaded: {len(cols)} features.")
            
    # The labels in this set are in the 'type' column
    y = df['type'].values if 'type' in df.columns else None
    return views, y

# Correct call:
XsBR, yBR = load_brca_tabular('./data/BRCA/brca_multiomics.csv')
