"""
Architecture Visualization for pysimlr.

This module provides functions to generate high-quality, traditional graph-style 
architecture diagrams for the various SiMLR models using matplotlib. It details 
the layer infrastructure, normalization, mixing, and NSA-Flow constraints.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from typing import Optional, List, Union, Any

# Default NeurIPS ready typography
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Computer Modern Roman', 'DejaVu Serif'],
    'axes.linewidth': 1.2,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

class ArchitectureGraph:
    """Helper class to build traditional graph-style architecture diagrams."""
    def __init__(self, figsize=(14, 10), title="", base_fontsize=11):
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.base_fontsize = base_fontsize
        self.ax.axis('off')
        self.ax.set_title(title, fontsize=base_fontsize + 7, fontweight='bold', pad=30)
        self.nodes = {}
        self.min_x = float('inf')
        self.max_x = float('-inf')
        self.min_y = float('inf')
        self.max_y = float('-inf')

    def add_node(self, name, x, y, label, node_type="layer", width=2.8, height=1.2):
        """Add a node (box or circle) representing a layer or tensor."""
        bg_colors = {
            "input": "#e6f2ff",
            "linear": "#fff2e6",
            "nonlinear": "#ffe6e6",
            "norm": "#f2e6ff",
            "mixing": "#e6ffe6",
            "latent": "#e6ffe6",
            "output": "#e6f2ff",
            "nsa": "#ffcccc"
        }
        color = bg_colors.get(node_type, "#ffffff")
        
        if node_type == "mixing":
            # Draw an ellipse/circle for operations
            patch = patches.Ellipse((x, y), width, height, linewidth=1.5, edgecolor='black', facecolor=color, zorder=2)
        else:
            # Rounded rectangle for layers/tensors
            patch = patches.FancyBboxPatch(
                (x - width/2, y - height/2), width, height, 
                boxstyle="round,pad=0.2", linewidth=1.5, edgecolor='black', facecolor=color, zorder=2
            )
            
        self.ax.add_patch(patch)
        self.ax.text(x, y, label, ha='center', va='center', fontsize=self.base_fontsize, zorder=3)
        self.nodes[name] = {"x": x, "y": y, "w": width, "h": height, "type": node_type}
        
        # Track limits for dynamic sizing
        self.min_x = min(self.min_x, x - width/2)
        self.max_x = max(self.max_x, x + width/2)
        self.min_y = min(self.min_y, y - height/2)
        self.max_y = max(self.max_y, y + height/2)

    def add_edge(self, src, dst, label="", style="->"):
        """Draw a directed edge between two nodes."""
        if src not in self.nodes or dst not in self.nodes:
            return
            
        n1 = self.nodes[src]
        n2 = self.nodes[dst]
        
        dx = n2["x"] - n1["x"]
        dy = n2["y"] - n1["y"]
        
        if abs(dx) > abs(dy):
            x1 = n1["x"] + (n1["w"]/2 + 0.1) * (1 if dx > 0 else -1)
            y1 = n1["y"]
            x2 = n2["x"] - (n2["w"]/2 + 0.1) * (1 if dx > 0 else -1)
            y2 = n2["y"]
        else:
            x1 = n1["x"]
            y1 = n1["y"] + (n1["h"]/2 + 0.1) * (1 if dy > 0 else -1)
            x2 = n2["x"]
            y2 = n2["y"] - (n2["h"]/2 + 0.1) * (1 if dy > 0 else -1)

        self.ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle=style, lw=1.5, color='black', shrinkA=0, shrinkB=0),
            zorder=1
        )
        
        if label:
            mx, my = (x1 + x2)/2, (y1 + y2)/2
            self.ax.text(mx + 0.1, my + 0.2, label, ha='center', va='center', fontsize=max(8, self.base_fontsize - 1), zorder=3, 
                         bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=2))

    def save(self, filename):
        pad_x = (self.max_x - self.min_x) * 0.1 if self.max_x > self.min_x else 1
        pad_y = (self.max_y - self.min_y) * 0.1 if self.max_y > self.min_y else 1
        self.ax.set_xlim(self.min_x - pad_x, self.max_x + pad_x)
        self.ax.set_ylim(self.min_y - pad_y, self.max_y + pad_y)
        self.fig.tight_layout()
        self.fig.savefig(filename, bbox_inches='tight')
        plt.close(self.fig)

def plot_lend_simr_architecture(save_path=None, base_fontsize=11):
    """Plot the detailed architecture graph for LEND SiMR."""
    graph = ArchitectureGraph(figsize=(14, 12), title="LEND SiMR: Detailed Architecture Graph", base_fontsize=base_fontsize)
    graph.add_node("x1", 3, 10, "$X_1$\nInput", "input")
    graph.add_node("v1", 3, 8, "$V_1$ (Gate)\n$W_{raw} \\rightarrow$ NSA-Flow", "nsa")
    graph.add_node("u1", 3, 6, "$u_1 = X_1 V_1$\nLocal Latent", "latent")
    
    graph.add_node("x2", 11, 10, "$X_2$\nInput", "input")
    graph.add_node("v2", 11, 8, "$V_2$ (Gate)\n$W_{raw} \\rightarrow$ NSA-Flow", "nsa")
    graph.add_node("u2", 11, 6, "$u_2 = X_2 V_2$\nLocal Latent", "latent")
    
    graph.add_node("mix", 7, 6, "Mean Pool +\nNewton Step", "mixing", width=3.0, height=1.4)
    graph.add_node("norm", 7, 4, "LayerNorm\n(Bottleneck)", "norm")
    graph.add_node("s", 7, 2, "$S$\nShared Latent", "latent")
    
    graph.add_node("dec1", 3, 4, "Decoder 1\nLinear $\\rightarrow$ LN $\\rightarrow$ ReLU\n$\\rightarrow$ Dropout $\\rightarrow$ Linear", "nonlinear", width=3.6, height=1.4)
    graph.add_node("out1", 3, 2, "$\\hat{X}_1$\nOutput", "output")
    
    graph.add_node("dec2", 11, 4, "Decoder 2\nLinear $\\rightarrow$ LN $\\rightarrow$ ReLU\n$\\rightarrow$ Dropout $\\rightarrow$ Linear", "nonlinear", width=3.6, height=1.4)
    graph.add_node("out2", 11, 2, "$\\hat{X}_2$\nOutput", "output")
    
    graph.add_edge("x1", "v1"); graph.add_edge("v1", "u1", "MatMul"); graph.add_edge("u1", "mix")
    graph.add_edge("x2", "v2"); graph.add_edge("v2", "u2", "MatMul"); graph.add_edge("u2", "mix")
    graph.add_edge("mix", "norm"); graph.add_edge("norm", "s")
    graph.add_edge("s", "dec1"); graph.add_edge("dec1", "out1")
    graph.add_edge("s", "dec2"); graph.add_edge("dec2", "out2")
    if save_path: graph.save(save_path)
    return graph.fig, graph.ax

def plot_ned_simr_architecture(save_path=None, base_fontsize=11):
    """Plot the detailed architecture graph for NED SiMR."""
    graph = ArchitectureGraph(figsize=(14, 14), title="NED SiMR: Detailed Architecture Graph", base_fontsize=base_fontsize)
    graph.add_node("xi", 7, 12, "$X_i$\nInput", "input")
    graph.add_node("vi", 7, 10, "$V_i$ (Linear Gate)\nwith NSA-Flow", "nsa")
    graph.add_node("li", 7, 8, "$X_i V_i$\nLinear Features", "linear")
    graph.add_node("enc_i", 7, 6, "ModalityEncoder $i$\nLinear $\\rightarrow$ LN $\\rightarrow$ ReLU $\\rightarrow$ Linear", "nonlinear", width=4.0, height=1.4)
    graph.add_node("ui", 7, 4, "$u_i$\nNon-linear Latent", "latent")
    graph.add_node("mix", 11, 4, "Mixing\n(Mean + Newton)", "mixing", width=2.4, height=1.2)
    graph.add_node("norm", 11, 2, "LayerNorm\n(Bottleneck)", "norm")
    graph.add_node("s", 11, 0, "$S$\nShared Latent", "latent")
    graph.add_node("dec_i", 3, 2, "ModalityDecoder $i$\nLinear $\\rightarrow$ LN $\\rightarrow$ ReLU $\\rightarrow$ Linear", "nonlinear", width=4.0, height=1.4)
    graph.add_node("out_i", 3, 0, "$\\hat{X}_i$\nOutput", "output")
    graph.add_edge("xi", "vi"); graph.add_edge("vi", "li"); graph.add_edge("li", "enc_i"); graph.add_edge("enc_i", "ui")
    graph.add_edge("ui", "mix", "From all $i$"); graph.add_edge("mix", "norm"); graph.add_edge("norm", "s")
    graph.add_edge("s", "dec_i", "To all $i$"); graph.add_edge("dec_i", "out_i")
    if save_path: graph.save(save_path)
    return graph.fig, graph.ax

def plot_ned_shared_private_architecture(save_path=None, base_fontsize=11):
    """Plot the detailed architecture graph for NED Shared/Private SiMR."""
    graph = ArchitectureGraph(figsize=(16, 14), title="NED Shared/Private: Detailed Architecture Graph", base_fontsize=base_fontsize)
    graph.add_node("xi", 8, 12, "$X_i$\nInput", "input")
    graph.add_node("vi", 8, 10, "$V_i$ (Shared Linear Gate)\nNSA-Flow Retraction", "nsa", width=3.6, height=1.2)
    graph.add_node("li", 8, 8, "$X_i V_i$\nLinear Features", "linear")
    
    graph.add_node("enc_sh", 4, 6, "Shared Head $i$\nLin $\\rightarrow$ LN $\\rightarrow$ ReLU $\\rightarrow$ Lin", "nonlinear", width=3.6, height=1.4)
    graph.add_node("u_sh", 4, 4, "$u_{i,\\text{shared}}$", "latent")
    graph.add_node("mix", 4, 2, "Mixing\n(Mean + Newton)", "mixing", width=2.4, height=1.2)
    graph.add_node("s", 4, 0, "$S$\nShared Latent (Normalized)", "latent", width=3.0, height=1.2)
    
    graph.add_node("enc_pr", 12, 6, "Private Head $i$\nLin $\\rightarrow$ LN $\\rightarrow$ ReLU $\\rightarrow$ Lin", "nonlinear", width=3.6, height=1.4)
    graph.add_node("u_pr", 12, 4, "$P_i$\nPrivate Latent", "latent")
    
    graph.add_node("concat", 8, 2, "Concatenate\n$[S, P_i]$", "mixing", width=2.8, height=1.2)
    graph.add_node("dec", 8, 0, "Decoder $i$\nLin $\\rightarrow$ LN $\\rightarrow$ ReLU $\\rightarrow$ Lin", "nonlinear", width=3.6, height=1.4)
    graph.add_node("out", 8, -2, "$\\hat{X}_i$\nOutput", "output")
    
    graph.add_edge("xi", "vi"); graph.add_edge("vi", "li")
    graph.add_edge("li", "enc_sh"); graph.add_edge("li", "enc_pr")
    graph.add_edge("enc_sh", "u_sh"); graph.add_edge("enc_pr", "u_pr")
    graph.add_edge("u_sh", "mix"); graph.add_edge("mix", "s")
    graph.add_edge("s", "concat"); graph.add_edge("u_pr", "concat")
    graph.add_edge("concat", "dec"); graph.add_edge("dec", "out")
    if save_path: graph.save(save_path)
    return graph.fig, graph.ax

def plot_nsa_flow_architecture(save_path=None, base_fontsize=11):
    """Plot the detailed architecture graph for NSA-Flow Layers."""
    graph = ArchitectureGraph(figsize=(14, 8), title="NSA-Flow Layers: Manifold Constraint", base_fontsize=base_fontsize)
    
    graph.add_node("w_raw", 2, 4, "$W_{\\text{raw}}$\n(Raw Weights)", "input", width=2.4, height=1.2)
    
    # Branch 1
    graph.add_node("orth", 6, 6, "Orthogonal Projection\n$\\mathcal{V}_k(\\mathbb{R}^d)$", "norm", width=3.2, height=1.2)
    graph.add_node("w_orth", 10, 6, "$W_{\\text{orth}}$", "latent", width=2.0, height=1.2)
    
    # Branch 2
    graph.add_node("ident", 6, 2, "Identity\n(Unconstrained)", "linear", width=3.2, height=1.2)
    graph.add_node("w_raw_pass", 10, 2, "$W_{\\text{raw}}$", "latent", width=2.0, height=1.2)
    
    # Combine
    graph.add_node("mix", 13, 4, "Weighted Sum\n$\\times w, \\times (1-w)$", "mixing", width=3.0, height=1.8)
    
    # Output
    graph.add_node("w_nsa", 17, 4, "$W_{\\text{NSA}}$\n(Manifold Constrained)", "output", width=3.2, height=1.2)
    
    graph.add_edge("w_raw", "orth")
    graph.add_edge("w_raw", "ident")
    
    graph.add_edge("orth", "w_orth")
    graph.add_edge("ident", "w_raw_pass")
    
    graph.add_edge("w_orth", "mix", "$w$")
    graph.add_edge("w_raw_pass", "mix", "$1-w$")
    
    graph.add_edge("mix", "w_nsa")
    
    if save_path: graph.save(save_path)
    return graph.fig, graph.ax

def plot_energy(energy_history: List[float], title: str = "Optimization Energy") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(energy_history, lw=2, color='tab:blue')
    ax.set_xlabel("Iteration / Epoch"); ax.set_ylabel("Energy / Loss")
    ax.set_title(title); ax.grid(True, alpha=0.3)
    return fig

def plot_latent_2d(u: torch.Tensor, labels: Optional[np.ndarray] = None, title: str = "Shared Latent Space") -> plt.Figure:
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
    v_np = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(v_np, cmap='coolwarm', center=0, ax=ax)
    ax.set_title(title)
    return fig

def generate_all_architecture_graphs(output_dir=".", base_fontsize=11):
    """Generate and save all detailed architecture graphs as PDFs."""
    import os
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    plot_lend_simr_architecture(save_path=os.path.join(output_dir, "detailed_lend_simr.pdf"), base_fontsize=base_fontsize)
    plot_ned_simr_architecture(save_path=os.path.join(output_dir, "detailed_ned_simr.pdf"), base_fontsize=base_fontsize)
    plot_ned_shared_private_architecture(save_path=os.path.join(output_dir, "detailed_ned_shared_private.pdf"), base_fontsize=base_fontsize)
    plot_nsa_flow_architecture(save_path=os.path.join(output_dir, "detailed_nsa_flow.pdf"), base_fontsize=base_fontsize)
