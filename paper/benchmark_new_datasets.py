import os
import torch
import numpy as np
import pandas as pd
import time
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

# pybioportal for TCGA data
PYBIOPORTAL_AVAILABLE = False
try:
    from pybioportal import clinical_data as cd
    from pybioportal import molecular_data as md
    from pybioportal import molecular_profiles as mp
    from pybioportal import genes
    PYBIOPORTAL_AVAILABLE = True
except ImportError:
    print("Error: pybioportal not installed. Please install 'pip install pybioportal'.")

# pysimlr imports
from pysimlr import simlr, predict_simlr
from pysimlr.deep import lend_simr, ned_simr, ned_simr_shared_private, predict_deep

def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.array(x)

def fetch_tcga_kirc_data(max_genes=100):
    """
    Fetches a subset of 5 views of TCGA-KIRC data.
    Limits genes to avoid huge API payloads that hang the script.
    """
    if not PYBIOPORTAL_AVAILABLE:
        print("Error: pybioportal not available.")
        return None, None, None

    study_id = "kirc_tcga"
    print(f"Connecting to cBioPortal for TCGA-KIRC (Gene Limit: {max_genes})...")

    # 1. Fetch Patient Clinical Data (contains OS_MONTHS)
    try:
        patient_df = cd.fetch_all_clinical_data_in_study(study_id=study_id, clinical_data_type='PATIENT')
        sample_df = cd.fetch_all_clinical_data_in_study(study_id=study_id, clinical_data_type='SAMPLE')
        
        # Merge sample and patient data to link OS_MONTHS to sampleId
        clinical_df = pd.merge(sample_df[['sampleId', 'patientId']], 
                               patient_df, on='patientId', how='inner')
        
        id_col = 'sampleId'
        target_col = 'OS_MONTHS'
        if target_col not in clinical_df.columns:
            potential_targets = [c for c in clinical_df.columns if 'MONTHS' in c.upper()]
            target_col = potential_targets[0] if potential_targets else None
        
        if not target_col:
            print(f"Error: Could not find survival target. Available columns: {clinical_df.columns.tolist()[:10]}")
            return None, None, None

        clinical_sub = clinical_df[[id_col, target_col]].dropna()
        clinical_sub[target_col] = pd.to_numeric(clinical_sub[target_col], errors='coerce')
        clinical_sub = clinical_sub.dropna()
        print(f"  Found {len(clinical_sub)} samples with clinical outcomes ({target_col}).")
    except Exception as e:
        print(f"Error fetching/processing clinical data: {e}")
        return None, None, None

    # 2. Get a small subset of Gene IDs to request
    try:
        all_genes = genes.get_all_genes()
        gene_ids = all_genes['entrezGeneId'].iloc[:max_genes].tolist()
    except Exception as e:
        print(f"Warning: Could not fetch gene IDs, using range fallback: {e}")
        gene_ids = list(range(1, max_genes + 1))

    profiles = {
        "mRNA": f"{study_id}_rna_seq_v2_mrna",
        "miRNA": f"{study_id}_mirna",
        "Methylation": f"{study_id}_methylation_hm27",
        "CNV": f"{study_id}_gistic",
        "Mutation": f"{study_id}_mutations"
    }

    view_data = {}
    common_samples = set(clinical_sub[id_col])

    for name, profile_id in profiles.items():
        print(f"  Fetching View: {name}...")
        try:
            df = md.fetch_molecular_data(molecular_profile_id=profile_id, 
                                        entrez_gene_ids=gene_ids,
                                        sample_list_id=f"{study_id}_all")
            
            if df is None or df.empty:
                print(f"    Warning: No data for {name}")
                continue
                
            pivoted = df.pivot_table(index='sampleId', columns='entrezGeneId', values='value')
            view_data[name] = pivoted
            common_samples = common_samples.intersection(set(pivoted.index))
            print(f"    {name} aligned. Samples so far: {len(common_samples)}")
        except Exception as e:
            print(f"    {name} fetch failed: {e}")

    if len(common_samples) < 5:
        print(f"Error: Only {len(common_samples)} common samples found. Insufficient data.")
        return None, None, None

    print(f"Successfully aligned {len(common_samples)} samples across {len(view_data)} views.")
    sample_list = sorted(list(common_samples))
    X_list = [view_data[name].loc[sample_list].values for name in view_data]
    X_list = [np.nan_to_num(X, nan=0.0) for X in X_list]
    y = clinical_sub.set_index(id_col).loc[sample_list][target_col].values
    
    return X_list, y, list(view_data.keys())

def evaluate_model(model_name, X_train, X_test, y_train, y_test, k, dynamic_weights):
    X_train_t = [torch.tensor(x, dtype=torch.float32) for x in X_train]
    X_test_t = [torch.tensor(x, dtype=torch.float32) for x in X_test]
    
    if model_name == "SiMLR":
        res = simlr(X_train, k=k, iterations=20, verbose=False)
        U_train = to_numpy(res['u'])
        reg = Ridge().fit(U_train, y_train)
        pred_res = predict_simlr(X_test, res)
        U_cons, U_views = to_numpy(pred_res['u']), [to_numpy(v) for v in pred_res['latents']]
        y_pred = reg.predict(U_cons)
    else:
        func = {"LEND": lend_simr, "NED": ned_simr, "NEDPP": ned_simr_shared_private}[model_name]
        model = func(X_train_t, k=k, epochs=20, dynamic_weights=dynamic_weights, verbose=False)
        pred_train, pred_test = predict_deep(X_train_t, model), predict_deep(X_test_t, model)
        U_train_np, U_test_np = to_numpy(pred_train['u']), to_numpy(pred_test['u'])
        reg = Ridge().fit(U_train_np, y_train)
        y_pred = reg.predict(U_test_np)
        U_cons, U_views = U_test_np, [to_numpy(uv) for uv in pred_test['latents']]

    r2 = r2_score(y_test, y_pred)
    corrs = []
    for uv in U_views:
        v_corrs = [np.abs(np.corrcoef(uv[:, i], U_cons[:, i])[0, 1]) for i in range(k) if np.std(uv[:, i]) > 1e-6 and np.std(U_cons[:, i]) > 1e-6]
        if v_corrs: corrs.append(np.mean(v_corrs))
    
    return {"r2": r2, "consistency": np.mean(corrs) if corrs else 0.0}

def run_benchmark():
    # 1. Fetch Real Data
    X_list, y, view_names = fetch_tcga_kirc_data(max_genes=100)
    
    if X_list is None:
        print("\nFATAL ERROR: Benchmark cannot proceed without TCGA data.")
        sys.exit(1)

    # 2. Preprocess
    X_list = [StandardScaler().fit_transform(X) for X in X_list]
    indices = np.arange(len(y))
    idx_train, idx_test = train_test_split(indices, test_size=0.2, random_state=42)
    X_tr, X_te, y_tr, y_te = [X[idx_train] for X in X_list], [X[idx_test] for X in X_list], y[idx_train], y[idx_test]

    print("\nStarting Grid Search over Architectures and MAI...")
    results = []
    for arch in ["SiMLR", "LEND", "NED", "NEDPP"]:
        for mai in [True, False]:
            if arch == "SiMLR" and mai: continue
            print(f"Testing {arch} (MAI={mai})...")
            try:
                res = evaluate_model(arch, X_tr, X_te, y_tr, y_te, k=3, dynamic_weights=mai)
                print(f"  -> R2: {res['r2']:.4f}, Consistency: {res['consistency']:.4f}")
                res['architecture'] = arch
                res['MAI'] = mai
                results.append(res)
            except Exception as e:
                print(f"  {arch} execution failed: {e}")

    df = pd.DataFrame(results)
    if not df.empty:
        os.makedirs("results_cache", exist_ok=True)
        df.to_csv("results_cache/benchmark_tcga_results.csv", index=False)
        print(f"\nFinal results saved to results_cache/benchmark_tcga_results.csv")

if __name__ == "__main__":
    run_benchmark()
