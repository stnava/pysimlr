import torch
import numpy as np
import pandas as pd
from pysimlr.structural import fit_structural_models, create_path_graph

def test_structural_comparison():
    # 1. Generate synthetic data for A -> B -> C
    n = 200
    p = 20
    k = 2
    
    # Latent A
    u_a = torch.randn(n, k)
    # Latent B = Latent A + small noise
    u_b = u_a + 0.2 * torch.randn(n, k)
    # Latent C = Latent B + small noise (so C is related to A via B)
    u_c = u_b + 0.2 * torch.randn(n, k)
    
    # Views
    a = u_a @ torch.randn(k, p) + 0.1 * torch.randn(n, p)
    b = u_b @ torch.randn(k, p) + 0.1 * torch.randn(n, p)
    c = u_c @ torch.randn(k, p) + 0.1 * torch.randn(n, p)
    
    data = [a, b, c]
    
    # Define models
    # Model 1: A -> B -> C
    # Edges: (0,1), (1,2)
    m1_graph = create_path_graph([(0, 1), (1, 2)], n_modalities=3)
    
    # Model 2: A -> B, A -> C, B -/> C
    # Edges: (0,1), (0,2)
    m2_graph = create_path_graph([(0, 1), (0, 2)], n_modalities=3)
    
    # Model 3: Star (A, B, C all connected to each other - standard SiMLR)
    # In my graph implementation, star means everyone is a neighbor of everyone else
    m3_graph = create_path_graph([(0, 1), (0, 2), (1, 2)], n_modalities=3)
    
    models = {
        "Chain (A-B-C)": m1_graph,
        "Bifurcation (B-A-C)": m2_graph,
        "Star (Full)": m3_graph
    }
    
    print("Fitting models for comparison...")
    # Using small number of epochs for quick test
    results = fit_structural_models(data, k=k, models=models, epochs=30, warmup_epochs=5, verbose=False)
    
    df_comp = pd.DataFrame(results["comparison"])
    print("\nComparison Summary:")
    print(df_comp)
    
    # In this synthetic case, Chain and Star should probably have similar or better fit than Bifurcation
    # because C is actually derived from B.
    
    print("\nTest completed successfully.")

if __name__ == "__main__":
    test_structural_comparison()
