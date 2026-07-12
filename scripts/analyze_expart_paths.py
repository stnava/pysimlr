import torch
import pandas as pd
import numpy as np
from pysimlr import lend_simr, create_path_graph, predict_deep
from pysimlr.utils import adjusted_rvcoef

def analyze_expart():
    # 1. Load Data
    bl = pd.read_csv("data/expart_bl.csv").values
    mh = pd.read_csv("data/expart_mh.csv").values
    mr = pd.read_csv("data/expart_mr.csv").values
    pe = pd.read_csv("data/expart_pe.csv").values
    
    data = [bl, mh, mr, pe]
    # Modality map: 0:BL, 1:MH, 2:MR, 3:PE
    
    # 2. Define Structural Models
    # Model A: User Hypothesis (BL -> MR -> PE and MH)
    # Edges: (0,2), (2,3), (2,1)
    graph_user = create_path_graph([(0, 2), (2, 3), (2, 1)], n_modalities=4)
    
    # Model B: Sequential Chain (BL -> MR -> PE -> MH)
    graph_chain = create_path_graph([(0, 2), (2, 3), (3, 1)], n_modalities=4)
    
    # Model C: Mental Health Centric (MH is driven by everything else)
    # Edges: (0,1), (2,1), (3,1)
    graph_mh_centric = create_path_graph([(0, 1), (2, 1), (3, 1)], n_modalities=4)
    
    # Model D: Star (Standard SiMLR - Baseline)
    graph_star = create_path_graph([(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)], n_modalities=4)
    
    models = {
        "User Hypothesis (BL->MR->PE/MH)": graph_user,
        "Sequential Chain (BL->MR->PE->MH)": graph_chain,
        "MH Centric (BL/MR/PE->MH)": graph_mh_centric,
        "Star (Full Connectivity)": graph_star
    }
    
    # 3. Parameters from User
    # Architecture: lend
    # Optimizer: adam (default for lend_simr)
    # Energy: acc
    # Mixing: svd
    # Sparsity: 0.5
    
    k = 3 # Latent dimension
    epochs = 150
    sim_weight = 2.0 # Slightly higher sim weight to emphasize structure
    
    results = {}
    comparison_data = []
    
    for name, graph in models.items():
        print(f"\n>>> Fitting model: {name} ...")
        res = lend_simr(
            data, 
            k=k, 
            epochs=epochs, 
            topology="graph", 
            path_graph=graph,
            energy_type="acc",
            mixing_algorithm="svd",
            sparseness_quantile=0.5,
            sim_weight=sim_weight,
            warmup_epochs=20,
            verbose=False
        )
        results[name] = res
        
        # Metrics
        final_recon = np.mean(res["recon_history"][-10:])
        final_sim = np.mean(res["sim_history"][-10:])
        
        # Calculate Global Alignment (Average RV between all pairs)
        # This is a graph-independent metric to see how much signal is shared overall
        latents = res["latents"]
        rv_matrix = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                rv_matrix[i, j] = adjusted_rvcoef(latents[i], latents[j])
        avg_rv = np.mean(rv_matrix[np.triu_indices(4, k=1)])
        
        comparison_data.append({
            "Model": name,
            "Total Loss": np.mean(res["loss_history"][-10:]),
            "Recon Loss": final_recon,
            "Sim Loss (Alignment)": final_sim,
            "Global RV Alignment": avg_rv
        })

    # 4. Generate Report
    df_comp = pd.DataFrame(comparison_data)
    print("\n" + "="*60)
    print("EXPART MODALITY STRUCTURAL ANALYSIS REPORT")
    print("="*60)
    print(df_comp.to_string(index=False))
    
    best_model = df_comp.loc[df_comp["Global RV Alignment"].idxmax(), "Model"]
    print("\nSUMMARY:")
    print(f"The model with the highest global alignment is: {best_model}")
    print("This suggests that the connectivity pattern in this model best captures the shared variance.")
    
    # Detailed check for User Hypothesis
    user_res = results["User Hypothesis (BL->MR->PE/MH)"]
    print("\nUSER HYPOTHESIS DETAIL (BL -> MR -> PE/MH):")
    u_user = user_res["u"]
    # Check alignment of specific paths
    # BL-MR (0-2), MR-PE (2-3), MR-MH (2-1)
    l_bl, l_mh, l_mr, l_pe = user_res["latents"]
    print(f"  BL <-> MR Alignment (RV): {adjusted_rvcoef(l_bl, l_mr):.4f}")
    print(f"  MR <-> PE Alignment (RV): {adjusted_rvcoef(l_mr, l_pe):.4f}")
    print(f"  MR <-> MH Alignment (RV): {adjusted_rvcoef(l_mr, l_mh):.4f}")
    
    # Saving results for record
    df_comp.to_csv("expart_structural_comparison.csv", index=False)

if __name__ == "__main__":
    analyze_expart()
