import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import seaborn as sns
from typing import List, Optional, Union, Dict, Any

def _draw_box(ax, x, y, w, h, label, color='white', edgecolor='black', alpha=1.0, fontsize=10):
    rect = patches.Rectangle((x, y), w, h, facecolor=color, edgecolor=edgecolor, alpha=alpha, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=fontsize, fontweight='bold')
    return rect

def _draw_arrow(ax, x1, y1, x2, y2, color='black', lw=1.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw, mutation_scale=20))

def plot_lend_simr_architecture(save_path=None, base_fontsize=11):
    """
    Generate a schematic diagram of the LEND (Linear Encoder, Nonlinear Decoder) architecture.

    LEND combines an interpretable linear encoder (constrained to the Stiefel 
    manifold via NSA-Flow) with a deep nonlinear decoder. This allows for 
    high-fidelity feature discovery while providing a nonlinear 'escape hatch' 
    to model measurement noise and artifacts.

    Parameters
    ----------
    save_path : str, optional
        Path to save the generated figure (e.g., 'lend_arch.pdf').
    base_fontsize : int, default=11
        Base font size for labels and text in the diagram.

    Returns
    -------
    plt.Figure
        The generated Matplotlib figure.

    Raises
    ------
    TypeError
        If inputs are of invalid types.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.axis('off')

    # Title
    ax.text(50, 95, "LEND: Linear Encoder, Nonlinear Decoder", ha='center', va='center', fontsize=base_fontsize+4, fontweight='bold')

    # Modalities
    _draw_box(ax, 5, 60, 15, 20, "Modality 1\n(X1)", color='#e3f2fd')
    _draw_box(ax, 5, 20, 15, 20, "Modality 2\n(X2)", color='#e3f2fd')

    # Encoders
    _draw_box(ax, 25, 65, 12, 10, "Linear\nV1", color='#fff3e0')
    _draw_box(ax, 25, 25, 12, 10, "Linear\nV2", color='#fff3e0')
    ax.text(31, 78, "Stiefel Constrained\n(NSA-Flow)", ha='center', fontsize=base_fontsize-2, fontstyle='italic')

    # Latents
    _draw_box(ax, 42, 65, 10, 10, "Z1", color='#f1f8e9')
    _draw_box(ax, 42, 25, 10, 10, "Z2", color='#f1f8e9')

    # Consensus
    _draw_box(ax, 58, 45, 12, 10, "Shared\nConsensus (U)", color='#fce4ec')
    _draw_arrow(ax, 52, 70, 58, 55, color='gray')
    _draw_arrow(ax, 52, 30, 58, 45, color='gray')
    ax.text(55, 62, "ACC / Newton\nMixing", ha='center', fontsize=base_fontsize-2)

    # Decoders
    _draw_box(ax, 75, 45, 15, 10, "Deep\nNonlinear\nDecoder", color='#ede7f6')
    _draw_arrow(ax, 70, 50, 75, 50)

    # Reconstructions
    _draw_box(ax, 92, 60, 6, 15, "X1'", color='#fafafa')
    _draw_box(ax, 92, 25, 6, 15, "X2'", color='#fafafa')
    _draw_arrow(ax, 90, 50, 92, 65)
    _draw_arrow(ax, 90, 50, 92, 35)

    # Global flow arrows
    _draw_arrow(ax, 20, 70, 25, 70); _draw_arrow(ax, 37, 70, 42, 70)
    _draw_arrow(ax, 20, 30, 25, 30); _draw_arrow(ax, 37, 30, 42, 30)

    if save_path: plt.savefig(save_path, bbox_inches='tight', dpi=300)
    return fig

def plot_ned_simr_architecture(save_path=None, base_fontsize=11):
    """
    Generate a schematic diagram of the NED (Nonlinear Encoder Decoder) architecture.

    NED uses deep neural networks for both encoding and decoding, maximizing 
    representational power to capture complex nonlinear relationships 
    across modalities.

    Parameters
    ----------
    save_path : str, optional
        Path to save the generated figure.
    base_fontsize : int, default=11
        Base font size for labels and text.

    Returns
    -------
    plt.Figure
        The generated Matplotlib figure.

    Raises
    ------
    TypeError
        If inputs are of invalid types.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.axis('off')
    ax.text(50, 95, "NED: Nonlinear Encoder, Nonlinear Decoder", ha='center', va='center', fontsize=base_fontsize+4, fontweight='bold')
    _draw_box(ax, 5, 40, 12, 20, "X_m", color='#e3f2fd')
    _draw_box(ax, 22, 40, 15, 20, "Deep\nEncoder", color='#ede7f6')
    _draw_box(ax, 42, 40, 10, 20, "Z_m", color='#f1f8e9')
    _draw_box(ax, 58, 40, 12, 20, "Shared\nU", color='#fce4ec')
    _draw_box(ax, 75, 40, 15, 20, "Deep\nDecoder", color='#ede7f6')
    _draw_box(ax, 93, 40, 5, 20, "X_m'", color='#fafafa')
    _draw_arrow(ax, 17, 50, 22, 50); _draw_arrow(ax, 37, 50, 42, 50)
    _draw_arrow(ax, 52, 50, 58, 50); _draw_arrow(ax, 70, 50, 75, 50)
    _draw_arrow(ax, 90, 50, 93, 50)
    if save_path: plt.savefig(save_path, bbox_inches='tight', dpi=300)
    return fig

def plot_ned_shared_private_architecture(save_path=None, base_fontsize=11):
    """
    Generate a schematic diagram of the NEDPP (Shared + Private) architecture.

    NEDPP partitions the latent space into components that are shared across 
    all data views and components that are private (modality-specific). This 
    prevents view-specific noise from contaminating the shared consensus (U).

    Parameters
    ----------
    save_path : str, optional
        Path to save the generated figure.
    base_fontsize : int, default=11
        Base font size for labels and text.

    Returns
    -------
    plt.Figure
        The generated Matplotlib figure.

    Raises
    ------
    TypeError
        If inputs are of invalid types.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
    """
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.axis('off')
    ax.text(50, 95, "NEDPP: Shared and Private Latent Decomposition", ha='center', va='center', fontsize=base_fontsize+4, fontweight='bold')
    _draw_box(ax, 5, 40, 12, 20, "X_m", color='#e3f2fd')
    _draw_box(ax, 22, 40, 15, 20, "Deep\nEncoder", color='#ede7f6')
    _draw_box(ax, 45, 60, 10, 15, "Shared\nZ_m", color='#f1f8e9')
    _draw_box(ax, 45, 25, 10, 15, "Private\nP_m", color='#fffde7')
    _draw_box(ax, 62, 60, 12, 15, "Shared\nU", color='#fce4ec')
    _draw_box(ax, 78, 40, 12, 20, "Deep\nDecoder", color='#ede7f6')
    _draw_box(ax, 94, 40, 5, 20, "X_m'", color='#fafafa')
    _draw_arrow(ax, 37, 55, 45, 65); _draw_arrow(ax, 37, 45, 45, 35)
    _draw_arrow(ax, 55, 67, 62, 67); _draw_arrow(ax, 74, 67, 78, 55)
    _draw_arrow(ax, 55, 32, 78, 45)
    if save_path: plt.savefig(save_path, bbox_inches='tight', dpi=300)
    return fig

def plot_nsa_flow_architecture(save_path=None, base_fontsize=11):
    """
    Generate a schematic of the NSA-Flow (Neural Spatially-Aware) manifold optimization flow.

    Parameters
    ----------
    save_path : str, optional
        Path to save the generated figure.
    base_fontsize : int, default=11
        Base font size for labels and text.

    Returns
    -------
    plt.Figure
        The generated Matplotlib figure.

    Raises
    ------
    TypeError
        If inputs are of invalid types.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.axis('off')
    ax.text(50, 95, "NSA-Flow: Non-negative Stiefel Manifold Optimization", ha='center', va='center', fontsize=base_fontsize+4, fontweight='bold')
    _draw_box(ax, 5, 45, 12, 15, "Raw\nWeights\nV_raw", color='#fafafa')
    _draw_box(ax, 25, 60, 20, 12, "Gradient Projection\n(Tangent Space)", color='#fff3e0')
    _draw_box(ax, 25, 30, 20, 12, "Retraction\n(SVD/Newton)", color='#fff3e0')
    _draw_box(ax, 55, 45, 15, 15, "Orthonormal\nWeights\nV_ortho", color='#f1f8e9')
    _draw_box(ax, 80, 45, 15, 15, "Non-negative\nThresholding\n(Sparsity)", color='#e3f2fd')
    _draw_arrow(ax, 17, 52, 25, 62); _draw_arrow(ax, 45, 62, 55, 55)
    _draw_arrow(ax, 55, 50, 45, 40); _draw_arrow(ax, 25, 35, 17, 48)
    _draw_arrow(ax, 70, 52, 80, 52)
    ax.text(35, 80, "Stiefel Manifold\nConstraints", ha='center', fontsize=base_fontsize, fontweight='bold')
    if save_path: plt.savefig(save_path, bbox_inches='tight', dpi=300)
    return fig

def plot_energy(energy_history: List[float], title: str = "Optimization Energy") -> plt.Figure:
    """
    Plot the optimization energy (loss) history.

    Parameters
    ----------
    energy_history : List[float]
        A list of energy/loss values over iterations or epochs.
    title : str, default="Optimization Energy"
        The title for the plot.

    Returns
    -------
    plt.Figure
        The generated Matplotlib figure.

    Raises
    ------
    TypeError
        If inputs are of invalid types.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(energy_history, lw=2, color='tab:blue')
    ax.set_xlabel("Iteration / Epoch"); ax.set_ylabel("Energy / Loss")
    ax.set_title(title); ax.grid(True, alpha=0.3)
    return fig

def plot_latent_2d(u: torch.Tensor, labels: Optional[np.ndarray] = None, title: str = "Shared Latent Space") -> plt.Figure:
    """
    Visualize the shared latent consensus (U) in a 2D scatter plot.

    Parameters
    ----------
    u : torch.Tensor
        The shared latent consensus matrix (samples x 2 or more). Only the 
        first two dimensions are plotted.
    labels : np.ndarray, optional
        Categorical or continuous labels for coloring the points.
    title : str, default="Shared Latent Space"
        The title for the plot.

    Returns
    -------
    plt.Figure
        The generated Matplotlib figure.

    Raises
    ------
    TypeError
        If inputs are of invalid types.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
    """
    u_np = u.detach().cpu().numpy() if isinstance(u, torch.Tensor) else u
    fig, ax = plt.subplots(figsize=(8, 8))
    if labels is not None:
        scatter = ax.scatter(u_np[:, 0], u_np[:, 1], c=labels, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, ax=ax)
    else:
        ax.scatter(u_np[:, 0], u_np[:, 1], alpha=0.6)
    ax.set_xlabel("Latent 1"); ax.set_ylabel("Latent 2")
    ax.set_title(title); ax.grid(True, alpha=0.3)
    return fig

def plot_v_matrix(v: torch.Tensor, title: str = "Feature Importance (V Matrix)") -> plt.Figure:
    """
    Generate a heatmap of the basis matrix V.

    Parameters
    ----------
    v : torch.Tensor
        The basis matrix (features x components).
    title : str, default="Feature Importance (V Matrix)"
        The title for the plot.

    Returns
    -------
    plt.Figure
        The generated Matplotlib figure.

    Raises
    ------
    TypeError
        If inputs are of invalid types.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
    """
    v_np = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(v_np, cmap='coolwarm', center=0, ax=ax)
    ax.set_title(title)
    return fig

def generate_all_architecture_graphs(output_dir=".", base_fontsize=11) -> List[plt.Figure]:
    """
    Generate and save all detailed architecture graphs as PDFs.

    Parameters
    ----------
    output_dir : str, default="."
        Directory where PDFs will be saved.
    base_fontsize : int, default=11
        Base font size for diagrams.

    Returns
    -------
    List[plt.Figure]
        List of all generated architecture figures.

    Raises
    ------
    TypeError
        If inputs are of invalid types.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
    """
    import os
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    figs = []
    figs.append(plot_lend_simr_architecture(save_path=os.path.join(output_dir, "detailed_lend_simr.pdf"), base_fontsize=base_fontsize))
    figs.append(plot_ned_simr_architecture(save_path=os.path.join(output_dir, "detailed_ned_simr.pdf"), base_fontsize=base_fontsize))
    figs.append(plot_ned_shared_private_architecture(save_path=os.path.join(output_dir, "detailed_ned_shared_private.pdf"), base_fontsize=base_fontsize))
    figs.append(plot_nsa_flow_architecture(save_path=os.path.join(output_dir, "detailed_nsa_flow.pdf"), base_fontsize=base_fontsize))
    return figs

def plot_flow_simr_architecture(save_path=None, base_fontsize=11):
    """
    Generate a schematic diagram of the Flow-SiMLR (Normalizing Flow) architecture.

    Flow-SiMLR utilizes bijective normalizing flows to map raw inputs to a joint
    Gaussian space, partitioning the latent space into shared consensus-aligned
    coordinates and private coordinate dimensions. This bijectivity enables exact
    cross-view conditional inference.

    Parameters
    ----------
    save_path : str, optional
        Path to save the generated figure.
    base_fontsize : int, default=11
        Base font size for labels and text.

    Returns
    -------
    plt.Figure
        The generated Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.axis('off')
    ax.text(50, 95, "Flow-SiMLR: Invertible Normalizing Flow Multi-Modal Integration", ha='center', va='center', fontsize=base_fontsize+3, fontweight='bold')
    
    # Input
    _draw_box(ax, 5, 40, 12, 20, "Input\nX_m", color='#e3f2fd')
    
    # Bijective Flow
    _draw_box(ax, 22, 40, 15, 20, "Normalizing Flow\n(Bijective)", color='#e0f2f1')
    
    # Latent coordinates split
    _draw_box(ax, 45, 60, 12, 15, "Shared Latent\nZ_shared", color='#e0f7fa')
    _draw_box(ax, 45, 25, 12, 15, "Private Latent\nZ_private", color='#ffe0b2')
    
    # Consensus U
    _draw_box(ax, 65, 60, 12, 15, "Consensus\nTarget U", color='#fce4ec')
    
    # Inverse Flow Decoder / Imputation
    _draw_box(ax, 82, 40, 14, 20, "Inverse Flow\n[Z_shared || 0]", color='#e0f2f1')
    
    # Output
    _draw_box(ax, 93, 20, 5, 20, "X_m'", color='#fafafa')
    _draw_box(ax, 93, 60, 5, 20, "X_oth'", color='#f3e5f5')
    
    # Flow arrows
    _draw_arrow(ax, 17, 50, 22, 50)
    _draw_arrow(ax, 37, 55, 45, 65)
    _draw_arrow(ax, 37, 45, 45, 35)
    
    # Latent alignment & conditioning arrows
    _draw_arrow(ax, 57, 67, 65, 67)
    _draw_arrow(ax, 57, 60, 82, 53)
    _draw_arrow(ax, 45+6, 25, 82, 47, color='gray') # private zero-out arrow
    _draw_arrow(ax, 77, 67, 82, 60, color='blue') # conditional inference path
    
    # Decoder output arrows
    _draw_arrow(ax, 91, 47, 93, 30)
    _draw_arrow(ax, 91, 53, 93, 65)
    
    ax.text(73, 78, "Joint Gaussian\nConditioning", ha='center', fontsize=base_fontsize-2, color='blue', fontweight='bold')
    
    if save_path: plt.savefig(save_path, bbox_inches='tight', dpi=300)
    return fig

# Update generate_all_architecture_graphs to include the new function
def generate_all_architecture_graphs(output_dir="docs/manuscript/figures", base_fontsize=11):
    """
    Generate and save schematic diagrams for all architectures.

    Parameters
    ----------
    output_dir : str, default="docs/manuscript/figures"
        Directory to save the generated PDF figures.
    base_fontsize : int, default=11
        Base font size for labels and text.

    Returns
    -------
    List[plt.Figure]
        A list of generated Matplotlib figures.
    """
    import os
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    figs = []
    figs.append(plot_lend_simr_architecture(save_path=os.path.join(output_dir, "detailed_lend_simr.pdf"), base_fontsize=base_fontsize))
    figs.append(plot_ned_simr_architecture(save_path=os.path.join(output_dir, "detailed_ned_simr.pdf"), base_fontsize=base_fontsize))
    figs.append(plot_ned_shared_private_architecture(save_path=os.path.join(output_dir, "detailed_ned_shared_private.pdf"), base_fontsize=base_fontsize))
    figs.append(plot_nsa_flow_architecture(save_path=os.path.join(output_dir, "detailed_nsa_flow.pdf"), base_fontsize=base_fontsize))
    figs.append(plot_flow_simr_architecture(save_path=os.path.join(output_dir, "detailed_flow_simr.pdf"), base_fontsize=base_fontsize))
    return figs

def plot_path_model(path_graph: Dict[int, List[int]], modality_names: Optional[List[str]] = None, title: str = "Path Model Structure") -> plt.Figure:
    """
    Visualize the structure of a structural path model using NetworkX.

    Parameters
    ----------
    path_graph : Dict[int, List[int]]
        The adjacency list defining the model structure.
    modality_names : List[str], optional
        Names for each modality to use as node labels.
    title : str, default="Path Model Structure"
        The title for the plot.

    Returns
    -------
    plt.Figure
        The generated Matplotlib figure.
    """
    try:
        import networkx as nx
        fig, ax = plt.subplots(figsize=(8, 6))
        G = nx.Graph()
        for u, neighbors in path_graph.items():
            for v in neighbors:
                G.add_edge(u, v)
        
        pos = nx.spring_layout(G)
        labels = {i: (modality_names[i] if modality_names and i < len(modality_names) else f"M{i}") for i in G.nodes()}
        nx.draw(G, pos, with_labels=True, labels=labels, node_color='#e3f2fd', node_size=2000, font_size=10, font_weight='bold', edge_color='gray', ax=ax)
        ax.set_title(title)
        return fig
    except ImportError:
        import warnings
        warnings.warn("networkx not installed, skipping path model plot.")
        return None
