import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pysimlr import lend_simr, create_path_graph
from pysimlr.visualization import plot_path_model, plot_latent_2d, plot_energy

def generate_assets():
    out_dir = "expart_report_figures"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Plot the competing architectures
    modality_names = ["BL", "MH", "MR", "PE"]
    
    graph_chain = create_path_graph([(0, 2), (2, 3), (3, 1)], 4)
    fig_chain = plot_path_model(graph_chain, modality_names, title="Sequential Chain Model (BL -> MR -> PE -> MH)")
    if fig_chain:
        fig_chain.savefig(os.path.join(out_dir, "path_sequential_chain.png"), bbox_inches='tight', dpi=300)
    
    graph_user = create_path_graph([(0, 2), (2, 3), (2, 1)], 4)
    fig_user = plot_path_model(graph_user, modality_names, title="User Hypothesis (BL -> MR -> PE & MH)")
    if fig_user:
        fig_user.savefig(os.path.join(out_dir, "path_user_hypothesis.png"), bbox_inches='tight', dpi=300)
        
    graph_mh_centric = create_path_graph([(0, 1), (2, 1), (3, 1)], 4)
    fig_centric = plot_path_model(graph_mh_centric, modality_names, title="MH Centric Model (BL, MR, PE -> MH)")
    if fig_centric:
        fig_centric.savefig(os.path.join(out_dir, "path_mh_centric.png"), bbox_inches='tight', dpi=300)
    
    # 2. Fit the winning model to generate convergence and latent plots
    bl = pd.read_csv("data/expart_bl.csv").values
    mh = pd.read_csv("data/expart_mh.csv").values
    mr = pd.read_csv("data/expart_mr.csv").values
    pe = pd.read_csv("data/expart_pe.csv").values
    data = [bl, mh, mr, pe]
    
    print("Fitting Sequential Chain model (Final Winner Parameters) to generate asset plots...")
    res_chain = lend_simr(data, k=2, topology="graph", path_graph=graph_chain, epochs=150, 
                          energy_type="acc", mixing_algorithm="newton", sparseness_quantile=0.5, 
                          sim_weight=2.0, verbose=False, optimizer_type="lars", dynamic_weights=False)
    
    # Energy Plot
    fig_energy = plot_energy(res_chain["loss_history"], title="Sequential Chain: Total Loss Convergence")
    fig_energy.savefig(os.path.join(out_dir, "chain_convergence.png"), bbox_inches='tight', dpi=300)
    
    # Latent Space Plot
    u_shared = res_chain["u"]
    if isinstance(u_shared, list):
        u_shared = u_shared[0] 
    fig_latent = plot_latent_2d(u_shared, title="Sequential Chain: Shared Latent Space (2D)")
    fig_latent.savefig(os.path.join(out_dir, "chain_latent_space.png"), bbox_inches='tight', dpi=300)
    
    print(f"Assets generated in {out_dir}/")

if __name__ == "__main__":
    generate_assets()
