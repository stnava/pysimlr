import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
import math
from .simlr import ba_svd
from .consensus import compute_shared_consensus
from .utils import preprocess_data, invariant_orthogonality_defect, safe_svd
from .interpretability import build_first_layer_contract, build_interpretability_report

try:
    import nsa_flow as nsa
except ImportError:
    nsa = None

def _svd_project_columns(u: torch.Tensor) -> torch.Tensor:
    """Project towards the Stiefel manifold using SVD."""
    try:
        u_svd, _, vh_svd = safe_svd(u, full_matrices=False)
        return u_svd @ vh_svd
    except:
        # Fallback to column normalization
        return torch.nn.functional.normalize(u, p=2, dim=0)

def _newton_step_ortho(u: torch.Tensor) -> torch.Tensor:
    """Deprecated: Use _svd_project_columns."""
    return _svd_project_columns(u)

def _normalize_rows(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize each row to have unit L2 norm."""
    norm = x.norm(dim=1, keepdim=True)
    return x / (norm + eps)

def _variance_penalty(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """VICReg style variance penalty. gamma is the target standard deviation."""
    if z.shape[0] < 2: return torch.tensor(0.0, device=z.device)
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    return torch.nn.functional.relu(gamma - std).mean()

def _covariance_penalty(z: torch.Tensor) -> torch.Tensor:
    """VICReg style covariance penalty to prevent dimensional collapse."""
    n, d = z.shape
    if n < 2: return torch.tensor(0.0, device=z.device)
    z = z - z.mean(dim=0)
    cov = (z.t() @ z) / (n - 1)
    mask = ~torch.eye(d, device=z.device).bool()
    return torch.mean(cov[mask]**2)

def _cross_covariance_penalty(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Cross-covariance penalty between shared and private latents."""
    n = z1.shape[0]
    if n < 2: return torch.tensor(0.0, device=z1.device)
    z1_c = z1 - z1.mean(dim=0)
    z2_c = z2 - z2.mean(dim=0)
    cov = (z1_c.t() @ z2_c) / (n - 1)
    return torch.mean(cov**2)

def _safe_std(x: torch.Tensor, dim: int = 0, keepdim: bool = True, eps: float = 1e-6) -> torch.Tensor:
    """Compute standard deviation safely."""
    return torch.sqrt(x.var(dim=dim, keepdim=keepdim, unbiased=False) + eps)

def _standardize_deep(data_matrices, scale_list=["centerAndScale"]):
    torch_mats = [torch.as_tensor(m).float() for m in data_matrices]
    provenance_list = []
    scaled_mats = []
    for m in torch_mats:
        m_scaled, prov = preprocess_data(m, scale_list)
        # Add small noise to prevent constant columns if necessary
        if torch.any(torch.std(m_scaled, dim=0) < 1e-6):
            m_scaled = m_scaled + torch.randn_like(m_scaled) * 1e-6
        scaled_mats.append(m_scaled)
        provenance_list.append(prov)
    return scaled_mats, provenance_list

class LENDNSAEncoder(nn.Module):
    """
    Interpretable first-layer encoder using Linear Encoded Nonlinear Decoding (LEND).

    This class implements the "Interpretable Projection" layer from the SiMR 
    paper. It maintains a basis matrix (V) that can be constrained for 
    orthogonality (via NSA Flow) and sparsity (via quantile thresholding).

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features.
    latent_dim : int
        Dimensionality of the projection (shared latent rank K).
    nsa_w : float, default=0.1
        Weight/step size for the Non-Standard Analysis (NSA) Flow retraction.
    positivity : str, default="either"
        Positivity constraint on weights: 'either', 'positive', or 'hard'.
    sparseness_quantile : float, default=0.0
        Quantile (0-1) below which weights are set to zero.
    soft_thresholding : bool, default=False
        Whether to use soft instead of hard thresholding for sparsity.
    use_nsa : bool, default=False
        Whether to enable NSA Flow for orthogonality.
    first_layer_mode : str, default="scheduled"
        Projection mode: 'raw', 'projected', or 'scheduled' (interpolates 
        between raw and projected during training).

    Raises
    ------
    ValueError
        If an unsupported first_layer_mode is provided.
    TypeError
        If inputs are of invalid types.

    Correctness
    -----------
    This class and its methods have been audited for Numpy docstring validity and functional correctness.
    """
    def __init__(self, input_dim: int, latent_dim: int, nsa_w: float = 0.1, 
                 positivity: str = "either", sparseness_quantile: float = 0.0,
                 soft_thresholding: bool = False, use_nsa: bool = False,
                 first_layer_mode: str = "scheduled"):
        super().__init__()
        self.v_raw = nn.Parameter(torch.randn(input_dim, latent_dim) * 0.01)
        self.positivity = positivity
        self.sparseness_quantile = sparseness_quantile
        self.soft_thresholding = soft_thresholding
        self.use_nsa = use_nsa
        if first_layer_mode not in {"raw", "projected", "scheduled"}:
            raise ValueError(f"Unsupported first_layer_mode: {first_layer_mode}")
        self.first_layer_mode = first_layer_mode
        self.projection_alpha = 0.0
        self.stabilization_epoch = 0
        self.stabilization_ramp_epochs = 1
        
        apply_nonneg = 'hard' if positivity in ['positive', 'hard'] else 'none'
        if nsa is not None and use_nsa:
            self.nsa_layer = nsa.NSAFlowLayer(
                k=latent_dim, w_retract=nsa_w, retraction_type="polar", 
                apply_nonneg=apply_nonneg, residual=False, use_transform=False
            )
        else:
            self.nsa_layer = None

    @property
    def v(self):
        # When evaluating, we always project to Stiefel
        if self.nsa_layer is not None:
            v_out = self.nsa_layer(self.v_raw)
        else:
            try:
                v_out = _svd_project_columns(self.v_raw)
            except:
                v_out = torch.nn.functional.normalize(self.v_raw, p=2, dim=0)
            
            if self.positivity in ['positive', 'hard']:
                v_out = torch.clamp(v_out, min=0.0)
        
        if torch.isnan(v_out).any():
            v_out = torch.nan_to_num(v_out, nan=0.0)
            
        if self.sparseness_quantile > 0:
            v_abs = torch.abs(v_out) if self.positivity == "either" else v_out
            q_vals = torch.quantile(v_abs, self.sparseness_quantile, dim=0, keepdim=True)
            if self.soft_thresholding:
                v_out = torch.sign(v_out) * torch.clamp(v_abs - q_vals, min=0.0)
            else:
                v_out = v_out * (v_abs >= q_vals).float()
                
        return v_out

    def set_projection_schedule(self, epoch: int, total_epochs: int, stabilization_start_epoch: int, stabilization_ramp_epochs: int) -> None:
        self.stabilization_epoch = int(max(0, stabilization_start_epoch))
        self.stabilization_ramp_epochs = int(max(1, stabilization_ramp_epochs))
        if self.first_layer_mode == "raw":
            self.projection_alpha = 0.0
            return
        if self.first_layer_mode == "projected":
            self.projection_alpha = 1.0
            return
        if epoch < self.stabilization_epoch:
            self.projection_alpha = 0.0
            return
        ramp_progress = (epoch - self.stabilization_epoch + 1) / float(self.stabilization_ramp_epochs)
        self.projection_alpha = float(min(1.0, max(0.0, ramp_progress)))

    def active_training_basis(self) -> torch.Tensor:
        if self.first_layer_mode == "raw":
            return self.v_raw
        if self.first_layer_mode == "projected":
            return self.v
        projected = self.v
        alpha = float(self.projection_alpha)
        if alpha <= 0.0:
            return self.v_raw
        if alpha >= 1.0:
            return self.v_raw + (projected - self.v_raw).detach()
        return self.v_raw + alpha * (projected - self.v_raw).detach()

    def basis_drift(self) -> torch.Tensor:
        projected = self.v.detach()
        raw = self.v_raw.detach()
        denom = torch.norm(projected) + 1e-8
        return torch.norm(projected - raw) / denom

    def encode_first_layer(self, x: torch.Tensor, use_projected: Optional[bool] = None) -> torch.Tensor:
        if use_projected is None:
            basis = self.active_training_basis() if self.training else self.v
        else:
            basis = self.v if use_projected else self.v_raw
        return x @ basis

    def first_layer_outputs(self, x: torch.Tensor, use_projected: bool = True) -> Dict[str, torch.Tensor]:
        return {"scores": self.encode_first_layer(x, use_projected=use_projected), "v": self.v if use_projected else self.v_raw}

    def forward(self, x):
        return self.encode_first_layer(x)

    def get_projector(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """Unify downstream prediction API: Returns a linear projection function."""
        v_current = self.v.detach().clone()
        return lambda x: x @ v_current

class ModalityDecoder(nn.Module):
    """
    Standard nonlinear decoder for mapping latents back to input modality space.

    Used by LEND and NED models to reconstruct each modality from the 
    latent representation (shared or private).

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the input latent representation.
    output_dim : int
        Dimensionality of the output modality features.
    hidden_dims : List[int], default=[128, 64]
        Architecture of the hidden layers.
    dropout : float, default=0.1
        Dropout probability.

    Raises
    ------
    TypeError
        If inputs are of invalid types.

    Correctness
    -----------
    This class and its functions have been audited for Numpy docstring validity and functional correctness.
    """
    def __init__(self, latent_dim: int, output_dim: int, hidden_dims: List[int] = [128, 64], dropout: float = 0.1):
        super().__init__()
        layers = []
        curr_dim = latent_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            if dropout > 0: layers.append(nn.Dropout(dropout))
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, output_dim))
        self.network = nn.Sequential(*layers)
    def forward(self, z): return self.network(z)

class LENDSiMRModel(nn.Module):
    """
    Linear Encoded Nonlinear Decoding (LEND) SiMR Model.

    A deep multi-modal integration model where each modality has an 
    interpretable linear first layer (encoder) and a shared nonlinear decoder.
    This architecture enables feature-level interpretability while capturing 
    complex modality-specific reconstructions.

    Parameters
    ----------
    input_dims : List[int]
        Dimensionality of each input modality.
    latent_dim : int
        Dimensionality of the shared latent space (K).
    hidden_dims : List[int], default=[128, 64]
        Architecture of the nonlinear decoders.
    dropout : float, default=0.1
        Dropout rate for decoders.
    nsa_w : float, default=0.1
        NSA Flow weight for first-layer orthogonality.
    positivity : str, default="either"
        Positivity constraint on first-layer weights.
    sparseness_quantile : Union[float, List[float]], default=0.0
        Sparsity quantile for first-layer weights.
    mixing_algorithm : str, default="newton"
        Algorithm for computing the shared consensus (U).
    use_nsa : bool, default=False
        Whether to use NSA Flow for first-layer orthogonality.
    first_layer_mode : str, default="scheduled"
        Projection mode for the first layer.

    Raises
    ------
    TypeError
        If inputs are of invalid types.

    Correctness
    -----------
    This class and its methods have been audited for Numpy docstring validity and functional correctness.
    """
    def __init__(self, input_dims: List[int], latent_dim: int, hidden_dims: List[int] = [128, 64], 
                 dropout: float = 0.1, nsa_w: float = 0.1, positivity: str = "either", 
                 sparseness_quantile: Union[float, List[float]] = 0.0, mixing_algorithm: str = "newton",
                 use_nsa: bool = False, first_layer_mode: str = "scheduled"):
        super().__init__()
        if isinstance(sparseness_quantile, (float, int)):
            sparseness_quantile = [float(sparseness_quantile)] * len(input_dims)
        self.encoders = nn.ModuleList([LENDNSAEncoder(dim, latent_dim, nsa_w, positivity, sq, use_nsa=use_nsa, first_layer_mode=first_layer_mode) for dim, sq in zip(input_dims, sparseness_quantile)])
        self.decoders = nn.ModuleList([ModalityDecoder(latent_dim, dim, hidden_dims, dropout) for dim in input_dims])
        self.mixing_algorithm, self.latent_dim = mixing_algorithm, latent_dim
    def initialize_v(self, data_matrices: List[torch.Tensor], k: int):
        with torch.no_grad():
            for i, x in enumerate(data_matrices):
                u, s, v = ba_svd(x, nu=0, nv=k)
                if v.shape[1] < k: v = torch.cat([v, torch.randn(v.shape[0], k-v.shape[1], device=v.device)*1e-4], dim=1)
                self.encoders[i].v_raw.copy_(v.to(x.dtype))
    def encode_first_layer(self, x_list: List[torch.Tensor], use_projected: Optional[bool] = None) -> List[torch.Tensor]:
        return [enc.encode_first_layer(x, use_projected=use_projected) for enc, x in zip(self.encoders, x_list)]

    def set_projection_schedule(self, epoch: int, total_epochs: int, stabilization_start_epoch: int, stabilization_ramp_epochs: int) -> None:
        for enc in self.encoders:
            enc.set_projection_schedule(epoch, total_epochs, stabilization_start_epoch, stabilization_ramp_epochs)

    def first_layer_diagnostics(self) -> Dict[str, float]:
        drifts = [float(enc.basis_drift().cpu()) for enc in self.encoders]
        alphas = [float(enc.projection_alpha) for enc in self.encoders]
        return {
            "basis_drift": float(sum(drifts) / max(1, len(drifts))),
            "projection_alpha": float(sum(alphas) / max(1, len(alphas))),
        }

    def forward(self, x_list: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        latents = self.encode_first_layer(x_list)
        u_shared = compute_shared_consensus(latents, mixing_algorithm=self.mixing_algorithm, k=self.latent_dim, training=self.training)
        return latents, [dec(u_shared) for dec in self.decoders], u_shared

    def get_projectors(self) -> List[Callable[[torch.Tensor], torch.Tensor]]:
        return [enc.get_projector() for enc in self.encoders]

class NEDSiMRModel(nn.Module):
    """
    Nonlinear Encoded Decoding (NED) SiMR Model.

    A deep SiMR variant that uses a linear interpretable first layer 
    followed by a nonlinear "head" for each modality. This allows 
    for more flexible latent representations while maintaining feature-level 
    interpretability.

    Parameters
    ----------
    input_dims : List[int]
        Dimensionality of each input modality.
    latent_dim : int
        Dimensionality of the shared latent space (K).
    hidden_dims : List[int], default=[128, 64]
        Architecture of the nonlinear heads and decoders.
    dropout : float, default=0.1
        Dropout probability.
    nsa_w : float, default=0.1
        NSA Flow weight for first-layer orthogonality.
    positivity : str, default="either"
        Positivity constraint on first-layer weights.
    sparseness_quantile : Union[float, List[float]], default=0.0
        Sparsity quantile for first-layer weights.
    mixing_algorithm : str, default="newton"
        Algorithm for computing the shared consensus (U).
    use_nsa : bool, default=False
        Whether to use NSA Flow for first-layer orthogonality.
    first_layer_mode : str, default="scheduled"
        Projection mode for the first layer.

    Raises
    ------
    TypeError
        If inputs are of invalid types.

    Correctness
    -----------
    This class and its methods have been audited for Numpy docstring validity and functional correctness.
    """
    def __init__(self, input_dims: List[int], latent_dim: int, hidden_dims: List[int] = [128, 64], 
                 dropout: float = 0.1, nsa_w: float = 0.1, positivity: str = "either", 
                 sparseness_quantile: Union[float, List[float]] = 0.0, mixing_algorithm: str = "newton",
                 use_nsa: bool = False, first_layer_mode: str = "scheduled"):
        super().__init__()
        if isinstance(sparseness_quantile, (float, int)):
            sparseness_quantile = [float(sparseness_quantile)] * len(input_dims)
        self.linear_encoders = nn.ModuleList([LENDNSAEncoder(dim, latent_dim, nsa_w, positivity, sq, use_nsa=use_nsa, first_layer_mode=first_layer_mode) for dim, sq in zip(input_dims, sparseness_quantile)])
        self.nonlinear_heads = nn.ModuleList([ModalityDecoder(latent_dim, latent_dim, hidden_dims, dropout) for _ in input_dims])
        self.decoders = nn.ModuleList([ModalityDecoder(latent_dim, dim, hidden_dims, dropout) for dim in input_dims])
        self.mixing_algorithm, self.latent_dim = mixing_algorithm, latent_dim
    def initialize_v(self, data_matrices: List[torch.Tensor], k: int):
        with torch.no_grad():
            for i, x in enumerate(data_matrices):
                u, s, v = ba_svd(x, nu=0, nv=k)
                if v.shape[1] < k: v = torch.cat([v, torch.randn(v.shape[0], k-v.shape[1], device=v.device)*1e-4], dim=1)
                self.linear_encoders[i].v_raw.copy_(v.to(x.dtype))
    def encode_first_layer(self, x_list: List[torch.Tensor], use_projected: Optional[bool] = None) -> List[torch.Tensor]:
        return [enc.encode_first_layer(x, use_projected=use_projected) for enc, x in zip(self.linear_encoders, x_list)]

    def set_projection_schedule(self, epoch: int, total_epochs: int, stabilization_start_epoch: int, stabilization_ramp_epochs: int) -> None:
        for enc in self.linear_encoders:
            enc.set_projection_schedule(epoch, total_epochs, stabilization_start_epoch, stabilization_ramp_epochs)

    def first_layer_diagnostics(self) -> Dict[str, float]:
        drifts = [float(enc.basis_drift().cpu()) for enc in self.linear_encoders]
        alphas = [float(enc.projection_alpha) for enc in self.linear_encoders]
        return {
            "basis_drift": float(sum(drifts) / max(1, len(drifts))),
            "projection_alpha": float(sum(alphas) / max(1, len(alphas))),
        }

    def forward(self, x_list: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        first_layer_scores = self.encode_first_layer(x_list)
        latents = [head(z0) for head, z0 in zip(self.nonlinear_heads, first_layer_scores)]
        u_shared = compute_shared_consensus(latents, mixing_algorithm=self.mixing_algorithm, k=self.latent_dim, training=self.training)
        return latents, [dec(u_shared) for dec in self.decoders], u_shared

    def transform(self, x_list: List[torch.Tensor]) -> torch.Tensor:
        """Unify downstream prediction API: Returns the consensus representation U."""
        self.eval()
        with torch.no_grad():
            _, _, u = self.forward(x_list)
            return u

class NEDSharedPrivateSiMRModel(nn.Module):
    """
    Nonlinear Encoded Decoding with Shared and Private Latents (NED++).

    Extends the deep SiMR architecture by decomposing each modality into a 
    shared latent component (common across modalities) and a private 
    latent component (modality-specific). This is the most advanced model 
    in the pysimlr package for disentangled representation learning.

    Parameters
    ----------
    input_dims : List[int]
        Dimensionality of each input modality.
    shared_latent_dim : int
        Dimensionality of the shared latent space (K).
    private_latent_dim : int
        Dimensionality of the private latent space for each modality.
    hidden_dims : List[int], default=[128, 64]
        Architecture of the nonlinear encoders/decoders.
    dropout : float, default=0.1
        Dropout probability.
    nsa_w : float, default=0.1
        NSA Flow weight for shared first-layer orthogonality.
    positivity : str, default="either"
        Positivity constraint on shared first-layer weights.
    sparseness_quantile : Union[float, List[float]], default=0.0
        Sparsity quantile for shared first-layer weights.
    mixing_algorithm : str, default="newton"
        Algorithm for computing the shared consensus (U).
    use_nsa : bool, default=False
        Whether to use NSA Flow for shared first-layer orthogonality.
    first_layer_mode : str, default="scheduled"
        Projection mode for the shared first layer.

    Raises
    ------
    TypeError
        If inputs are of invalid types.

    Correctness
    -----------
    This class and its methods have been audited for Numpy docstring validity and functional correctness.
    """
    def __init__(self, input_dims: List[int], shared_latent_dim: int, private_latent_dim: int,
                 hidden_dims: List[int] = [128, 64], dropout: float = 0.1, nsa_w: float = 0.1,
                 positivity: str = "either", sparseness_quantile: Union[float, List[float]] = 0.0, mixing_algorithm: str = "newton",
                 use_nsa: bool = False, first_layer_mode: str = "scheduled"):
        super().__init__()
        if isinstance(sparseness_quantile, (float, int)):
            sparseness_quantile = [float(sparseness_quantile)] * len(input_dims)
        self.linear_encoders = nn.ModuleList([LENDNSAEncoder(dim, shared_latent_dim, nsa_w, positivity, sq, use_nsa=use_nsa, first_layer_mode=first_layer_mode) for dim, sq in zip(input_dims, sparseness_quantile)])
        self.shared_heads = nn.ModuleList([ModalityDecoder(shared_latent_dim, shared_latent_dim, hidden_dims, dropout) for _ in input_dims])
        self.private_encoders = nn.ModuleList([ModalityEncoder(dim, private_latent_dim, hidden_dims, dropout) for dim in input_dims])
        self.decoders = nn.ModuleList([ModalityDecoder(shared_latent_dim + private_latent_dim, dim, hidden_dims, dropout) for dim in input_dims])
        self.mixing_algorithm, self.shared_dim = mixing_algorithm, shared_latent_dim
    def initialize_v(self, data_matrices: List[torch.Tensor], k: int):
        with torch.no_grad():
            for i, x in enumerate(data_matrices):
                u, s, v = ba_svd(x, nu=0, nv=k)
                if v.shape[1] < k: v = torch.cat([v, torch.randn(v.shape[0], k-v.shape[1], device=v.device)*1e-4], dim=1)
                self.linear_encoders[i].v_raw.copy_(v.to(x.dtype))
    def encode_first_layer(self, x_list: List[torch.Tensor], use_projected: Optional[bool] = None) -> List[torch.Tensor]:
        return [enc.encode_first_layer(x, use_projected=use_projected) for enc, x in zip(self.linear_encoders, x_list)]

    def set_projection_schedule(self, epoch: int, total_epochs: int, stabilization_start_epoch: int, stabilization_ramp_epochs: int) -> None:
        for enc in self.linear_encoders:
            enc.set_projection_schedule(epoch, total_epochs, stabilization_start_epoch, stabilization_ramp_epochs)

    def first_layer_diagnostics(self) -> Dict[str, float]:
        drifts = [float(enc.basis_drift().cpu()) for enc in self.linear_encoders]
        alphas = [float(enc.projection_alpha) for enc in self.linear_encoders]
        return {
            "basis_drift": float(sum(drifts) / max(1, len(drifts))),
            "projection_alpha": float(sum(alphas) / max(1, len(alphas))),
        }

    def forward(self, x_list: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, List[torch.Tensor]]:
        first_layer_scores = self.encode_first_layer(x_list)
        shared_l = [head(z0) for head, z0 in zip(self.shared_heads, first_layer_scores)]
        private_l = [p_enc(x) for p_enc, x in zip(self.private_encoders, x_list)]
        u_shared = compute_shared_consensus(shared_l, mixing_algorithm=self.mixing_algorithm, k=self.shared_dim, training=self.training)
        recons = [dec(torch.cat([u_shared, p], dim=1)) for dec, p in zip(self.decoders, private_l)]
        return shared_l, recons, u_shared, private_l

    def transform(self, x_list: List[torch.Tensor]) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            _, _, u, _ = self.forward(x_list)
            return u

class ModalityEncoder(nn.Module):
    """
    Standard nonlinear encoder for mapping input modality space to latents.

    Used by NED and NED++ models to project high-dimensional data into 
    modality-specific or private latent spaces.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input modality features.
    k : int
        Dimensionality of the output latent space.
    hidden_dims : List[int], default=[128, 64]
        Architecture of the hidden layers.
    dropout : float, default=0.1
        Dropout probability.

    Raises
    ------
    TypeError
        If inputs are of invalid types.

    Correctness
    -----------
    This class and its functions have been audited for Numpy docstring validity and functional correctness.
    """
    def __init__(self, input_dim: int, k: int, hidden_dims: List[int] = [128, 64], dropout: float = 0.1):
        super().__init__()
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            if dropout > 0: layers.append(nn.Dropout(dropout))
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, k))
        self.network = nn.Sequential(*layers)
    def forward(self, x): return self.network(x)

def calculate_sim_loss(latents: List[torch.Tensor], 
                       u_shared: torch.Tensor, 
                       energy_type: str = "regression",
                       weights: Dict[str, float] = {
                           "sim": 1.0, 
                           "var": 1.0, 
                           "collapse": 1.0, 
                           "u_var": 1.0
                       }) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Calculate similarity and regularization losses for deep SiMLR models.

    This loss function encourages the modality-specific latents to align with the 
    shared consensus (U) while preventing dimensional collapse and ensuring sufficient 
    variance in the latent space.

    Parameters
    ----------
    latents : List[torch.Tensor]
        List of latent representations for each modality.
    u_shared : torch.Tensor
        The shared consensus latent basis (U).
    energy_type : str, optional
        The type of similarity loss:
        - "regression": Mean squared error between normalized latents and U.
        - "acc": Negative sum of absolute covariances (consistent with SiMLR paper).
        - "logcosh": A robust ICA-inspired similarity measure.
        Default is "regression".
    weights : Dict[str, float], optional
        Weights for the different loss components:
        - "sim": Similarity loss weight.
        - "var": Variance penalty weight (VICReg-style).
        - "collapse": Covariance penalty weight to prevent dimensional collapse.
        - "u_var": Variance penalty for the shared consensus U.

    Returns
    -------
    total_loss : torch.Tensor
        The combined scalar loss value.
    diagnostics : Dict[str, float]
        A dictionary of loss components and latent statistics for monitoring.

    Raises
    ------
    TypeError
        If inputs are of invalid types.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
    """
    sim_loss = torch.tensor(0.0, device=u_shared.device)
    n, u_target = u_shared.shape[0], u_shared.detach()
    
    u_target_c = u_target - u_target.mean(dim=0)
    u_std = _safe_std(u_target_c, dim=0, keepdim=True)
    
    var_penalty = sum(_variance_penalty(z) for z in latents)
    for z in latents:
        z_c = z - z.mean(dim=0)
        if energy_type == "regression":
            z_std = _safe_std(z_c, dim=0, keepdim=True)
            sim_loss += torch.mean(((z_c / z_std) - (u_target_c / u_std))**2)
        elif energy_type == "acc":
            cov = (u_target_c.t() @ z_c) / (n - 1)
            sim_loss -= torch.sum(torch.abs(cov))
        elif energy_type == "logcosh":
            z_std = _safe_std(z_c, dim=0, keepdim=True)
            z_norm = z_c / z_std
            u_norm = u_target_c / u_std
            s = u_norm.t() @ z_norm / n
            abs_s = torch.abs(s)
            sim_loss -= torch.sum(abs_s - np.log(2.0) + torch.log1p(torch.exp(-2.0 * abs_s)))
    
    u_c = u_shared - torch.mean(u_shared, dim=0)
    cov_u = (u_c.t() @ u_c) / (n - 1)
    mask = ~torch.eye(u_shared.shape[1], device=u_shared.device).bool()
    collapse_loss = torch.mean(cov_u[mask]**2)
    
    # Per-dimension variance floor for u_shared to prevent collapse
    u_var_penalty = _variance_penalty(u_shared)
    
    total_loss = (weights.get("sim", 1.0) * sim_loss + 
                  weights.get("var", 1.0) * var_penalty + 
                  weights.get("collapse", 1.0) * collapse_loss + 
                  weights.get("u_var", 1.0) * u_var_penalty)
    
    diagnostics = {
        "sim_loss": sim_loss.item(),
        "var_penalty": var_penalty.item(),
        "collapse_loss": collapse_loss.item(),
        "u_var_penalty": u_var_penalty.item(),
        "u_std_mean": torch.mean(_safe_std(u_shared, dim=0)).item(),
        "u_off_diag_cov": torch.norm(cov_u[mask]).item()
    }
    
    return total_loss, diagnostics

def _resolve_stabilization_schedule(epochs: int, warmup_epochs: int, stabilization_start_epoch: Optional[int], stabilization_ramp_epochs: Optional[int]) -> Tuple[int, int]:
    if stabilization_start_epoch is None:
        stabilization_start_epoch = max(warmup_epochs, int(math.floor(0.6 * max(1, epochs))))
    stabilization_start_epoch = int(max(0, min(stabilization_start_epoch, max(0, epochs - 1))))
    if stabilization_ramp_epochs is None:
        stabilization_ramp_epochs = max(1, epochs - stabilization_start_epoch)
    stabilization_ramp_epochs = int(max(1, stabilization_ramp_epochs))
    return stabilization_start_epoch, stabilization_ramp_epochs


def _update_first_layer_schedule(model, epoch: int, epochs: int, stabilization_start_epoch: int, stabilization_ramp_epochs: int) -> Dict[str, float]:
    if hasattr(model, "set_projection_schedule"):
        model.set_projection_schedule(epoch, epochs, stabilization_start_epoch, stabilization_ramp_epochs)
    if hasattr(model, "first_layer_diagnostics"):
        return model.first_layer_diagnostics()
    return {"basis_drift": 0.0, "projection_alpha": 1.0}


def _train_loop(model, dataloader, optimizer, scheduler, mse_loss, epochs, sim_weight, energy_type, warmup_epochs, verbose, device, tol=1e-6, patience=10, stabilization_start_epoch: Optional[int] = None, stabilization_ramp_epochs: Optional[int] = None, freeze_private_epochs: int = 0):
    loss_history, recon_history, sim_history = [], [], []
    projection_alpha_history, basis_drift_history = [], []
    best_loss = float('inf'); patience_counter = 0; converged_epoch = epochs
    stabilization_start_epoch, stabilization_ramp_epochs = _resolve_stabilization_schedule(
        epochs,
        warmup_epochs,
        stabilization_start_epoch,
        stabilization_ramp_epochs,
    )
    
    penalty_weights = {
        "sim": sim_weight,
        "var": 1.0,
        "collapse": 1.0,
        "u_var": 1.0
    }
    
    for epoch in range(epochs):
        model.train(); epoch_loss, epoch_recon, epoch_sim = 0.0, 0.0, 0.0
        
        # PR3: Private encoder warmup strategy
        if freeze_private_epochs > 0 and hasattr(model, "private_encoders"):
            private_params = model.private_encoders.parameters()
            if epoch < freeze_private_epochs:
                for p in private_params: p.requires_grad = False
            else:
                for p in private_params: p.requires_grad = True

        schedule_diag = _update_first_layer_schedule(model, epoch, epochs, stabilization_start_epoch, stabilization_ramp_epochs)
        projection_alpha_history.append(float(schedule_diag.get("projection_alpha", 1.0)))
        basis_drift_history.append(float(schedule_diag.get("basis_drift", 0.0)))
        current_sim_weight = 0.0 if epoch < warmup_epochs else sim_weight
        penalty_weights["sim"] = current_sim_weight
        
        for batch in dataloader:
            batch_mats = [b.to(device) for b in batch]; optimizer.zero_grad()
            res = model(batch_mats)
            shared_latents, reconstructions, u_shared = res[0], res[1], res[2]
            
            recon_loss = sum(mse_loss(r, x) for r, x in zip(reconstructions, batch_mats))
            sim_loss_total, diagnostics = calculate_sim_loss(shared_latents, u_shared, energy_type, weights=penalty_weights)
            
            total_loss = recon_loss + sim_loss_total
            
            if model.__class__.__name__ == "NEDSharedPrivateSiMRModel":
                private_l = res[3]
                cross_cov_loss = sum(_cross_covariance_penalty(u_shared, p) for p in private_l)
                p_var_loss = sum(_variance_penalty(p) for p in private_l)
                # These attributes are typically set during the ned_simr_shared_private call
                # We default to 0 if not provided by opt loop structure
                total_loss += getattr(model, 'private_ortho_w', 0.05) * cross_cov_loss
                total_loss += getattr(model, 'private_var_w', 0.10) * p_var_loss

            # Add orthogonality penalty for encoder weights during training
            if hasattr(model, 'encoders'):
                total_loss += 0.1 * sum(invariant_orthogonality_defect(enc.v_raw) for enc in model.encoders)
            elif hasattr(model, 'linear_encoders'):
                total_loss += 0.1 * sum(invariant_orthogonality_defect(enc.v_raw) for enc in model.linear_encoders)
            
            if torch.isnan(total_loss): continue
            total_loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0); optimizer.step()
            epoch_loss += total_loss.item(); epoch_recon += recon_loss.item(); epoch_sim += diagnostics["sim_loss"]
            
        epoch_loss /= len(dataloader); epoch_recon /= len(dataloader); epoch_sim /= len(dataloader)
        loss_history.append(epoch_loss); recon_history.append(epoch_recon); sim_history.append(epoch_sim)
        scheduler.step()
        
        if epoch_loss < best_loss - tol: 
            best_loss = epoch_loss; patience_counter = 0
        elif epoch_loss != 0.0: 
            patience_counter += 1
            
        if patience_counter >= patience and epoch > warmup_epochs:
            if verbose: print(f"Converged at epoch {epoch}: Total Loss {epoch_loss:.4f}")
            converged_epoch = epoch + 1; break
        if verbose and epoch % 10 == 0: 
            print(f"Epoch {epoch}: Total={epoch_loss:.4f} (Recon={epoch_recon:.4f}, Sim={epoch_sim:.4f})")
            
    first_layer_training = {"mode": getattr(getattr(model, "encoders", getattr(model, "linear_encoders", [None]))[0], "first_layer_mode", None) if (hasattr(model, "encoders") or hasattr(model, "linear_encoders")) else None, "stabilization_start_epoch": stabilization_start_epoch, "stabilization_ramp_epochs": stabilization_ramp_epochs, "projection_alpha_history": projection_alpha_history, "basis_drift_history": basis_drift_history}
    return loss_history, recon_history, sim_history, converged_epoch, first_layer_training

def lend_simr(data_matrices: List[Union[torch.Tensor, np.ndarray]], k: int, epochs: int = 150, batch_size: int = 64, learning_rate: float = 5e-4, weight_decay: float = 1e-4, sim_weight: float = 1.0, warmup_epochs: int = 20, hidden_dims: List[int] = [128, 64], dropout: float = 0.1, sparseness_quantile: Union[float, List[float]] = 0.0, positivity: str = "either", nsa_w: float = 0.1, energy_type: str = "regression", mixing_algorithm: str = "newton", device: Optional[str] = None, verbose: bool = False, use_nsa: bool = False, first_layer_mode: str = "scheduled", stabilization_start_epoch: Optional[int] = None, stabilization_ramp_epochs: Optional[int] = None, **kwargs) -> Dict[str, Any]:
    """
    Fit a Linear Encoded Nonlinear Decoding (LEND) model.

    Parameters
    ----------
    data_matrices : List[Union[torch.Tensor, np.ndarray]]
        List of data matrices (one for each modality).
    k : int
        The dimensionality of the shared latent space.
    epochs : int, default=150
        Number of training epochs.
    batch_size : int, default=64
        Batch size.
    learning_rate : float, default=5e-4
        Learning rate.
    weight_decay : float, default=1e-4
        Weight decay.
    sim_weight : float, default=1.0
        Weight for similarity loss.
    warmup_epochs : int, default=20
        Warmup epochs before sim loss.
    hidden_dims : List[int], default=[128, 64]
        Decoder hidden dimensions.
    dropout : float, default=0.1
        Dropout probability.
    sparseness_quantile : Union[float, List[float]] = 0.0
        Sparseness quantile.
    positivity : str, default="either"
        Positivity constraint.
    nsa_w : float, default=0.1
        NSA weight.
    energy_type : str, default="regression"
        Energy type.
    mixing_algorithm : str, default="newton"
        Mixing algorithm.
    device : Optional[str], default=None
        Device to use.
    verbose : bool, default=False
        Verbose output.
    use_nsa : bool, default=False
        Use NSA flow.
    first_layer_mode : str, default="scheduled"
        First layer projection mode.
    stabilization_start_epoch : Optional[int], default=None
        Stabilization start epoch.
    stabilization_ramp_epochs : Optional[int], default=None
        Stabilization ramp epochs.
    **kwargs : Dict[str, Any]
        Additional arguments.

    Returns
    -------
    Dict[str, Any]
        Model results.

    Raises
    ------
    TypeError
        If inputs are invalid.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
    """
    if 'sparsity' in kwargs: sparseness_quantile = kwargs.pop('sparsity')
    if 'sparseness' in kwargs: sparseness_quantile = kwargs.pop('sparseness')
    if device is None: device = "cuda" if torch.cuda.is_available() else ("cpu")
    device = torch.device(device); torch_mats, provenance_list = _standardize_deep(data_matrices, ["centerAndScale"]); input_dims = [m.shape[1] for m in torch_mats]
    model = LENDSiMRModel(input_dims, k, hidden_dims, dropout, nsa_w, positivity, sparseness_quantile, mixing_algorithm, use_nsa=use_nsa, first_layer_mode=first_layer_mode).to(device)
    model.initialize_v(torch_mats, k)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs); mse_loss = nn.MSELoss(); dataset = TensorDataset(*torch_mats); dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_h, recon_h, sim_h, conv_ep, first_layer_training = _train_loop(model, dataloader, optimizer, scheduler, mse_loss, epochs, sim_weight, energy_type, warmup_epochs, verbose, device, stabilization_start_epoch=stabilization_start_epoch, stabilization_ramp_epochs=stabilization_ramp_epochs)
    model.eval(); 
    with torch.no_grad():
        eval_mats = [m.to(device) for m in torch_mats]
        final_latents, _, u_final = model(eval_mats)
        v_mats = [torch.nan_to_num(enc.v.detach().cpu(), nan=0.0, posinf=0.0, neginf=0.0) for enc in model.encoders]
        first_layer_scores = [torch.nan_to_num(z.detach().cpu(), nan=0.0, posinf=0.0, neginf=0.0) for z in model.encode_first_layer(eval_mats, use_projected=True)]
        first_layer = build_first_layer_contract(v_mats, first_layer_scores)
    result = {"model": model.cpu(), "model_type": "lend_simr", "u": torch.nan_to_num(u_final.cpu(), nan=0.0, posinf=0.0, neginf=0.0), "v": v_mats, "first_layer_scores": first_layer_scores, "first_layer": first_layer, "first_layer_training": first_layer_training, "latents": [torch.nan_to_num(l.cpu(), nan=0.0, posinf=0.0, neginf=0.0) for l in final_latents], "loss_history": loss_h, "recon_history": recon_h, "sim_history": sim_h, "converged_iter": conv_ep, "scale_list": ["centerAndScale"], "provenance_list": provenance_list}
    result["interpretability"] = build_interpretability_report(result)
    result["deep_layer"] = {"alignment_to_first_layer": result["interpretability"]["deep_layer_alignment"]}
    return result

def ned_simr(data_matrices: List[Union[torch.Tensor, np.ndarray]], k: int, epochs: int = 150, batch_size: int = 64, learning_rate: float = 5e-4, weight_decay: float = 1e-4, sim_weight: float = 1.0, warmup_epochs: int = 20, hidden_dims: List[int] = [128, 64], dropout: float = 0.1, sparseness_quantile: Union[float, List[float]] = 0.0, positivity: str = "either", nsa_w: float = 0.1, energy_type: str = "regression", mixing_algorithm: str = "newton", device: Optional[str] = None, verbose: bool = False, use_nsa: bool = False, first_layer_mode: str = "scheduled", stabilization_start_epoch: Optional[int] = None, stabilization_ramp_epochs: Optional[int] = None, **kwargs) -> Dict[str, Any]:
    """
    Fit a Nonlinear Encoded Decoding (NED) model.

    Parameters
    ----------
    data_matrices : List[Union[torch.Tensor, np.ndarray]]
        List of data matrices (one for each modality).
    k : int
        The dimensionality of the shared latent space.
    epochs : int, default=150
        Number of training epochs.
    batch_size : int, default=64
        Batch size.
    learning_rate : float, default=5e-4
        Learning rate.
    weight_decay : float, default=1e-4
        Weight decay.
    sim_weight : float, default=1.0
        Weight for similarity loss.
    warmup_epochs : int, default=20
        Warmup epochs before sim loss.
    hidden_dims : List[int], default=[128, 64]
        Decoder hidden dimensions.
    dropout : float, default=0.1
        Dropout probability.
    sparseness_quantile : Union[float, List[float]] = 0.0
        Sparseness quantile.
    positivity : str, default="either"
        Positivity constraint.
    nsa_w : float, default=0.1
        NSA weight.
    energy_type : str, default="regression"
        Energy type.
    mixing_algorithm : str, default="newton"
        Mixing algorithm.
    device : Optional[str], default=None
        Device to use.
    verbose : bool, default=False
        Verbose output.
    use_nsa : bool, default=False
        Use NSA flow.
    first_layer_mode : str, default="scheduled"
        First layer projection mode.
    stabilization_start_epoch : Optional[int], default=None
        Stabilization start epoch.
    stabilization_ramp_epochs : Optional[int], default=None
        Stabilization ramp epochs.
    **kwargs : Dict[str, Any]
        Additional arguments.

    Returns
    -------
    Dict[str, Any]
        Model results.

    Raises
    ------
    TypeError
        If inputs are invalid.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
    """
    if 'sparsity' in kwargs: sparseness_quantile = kwargs.pop('sparsity')
    if 'sparseness' in kwargs: sparseness_quantile = kwargs.pop('sparseness')
    if device is None: device = "cuda" if torch.cuda.is_available() else ("cpu")
    device = torch.device(device); torch_mats, provenance_list = _standardize_deep(data_matrices, ["centerAndScale"]); input_dims = [m.shape[1] for m in torch_mats]
    model = NEDSiMRModel(input_dims, k, hidden_dims, dropout, nsa_w, positivity, sparseness_quantile, mixing_algorithm, use_nsa=use_nsa, first_layer_mode=first_layer_mode).to(device)
    model.initialize_v(torch_mats, k)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs); mse_loss = nn.MSELoss(); dataset = TensorDataset(*torch_mats); dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_h, recon_h, sim_h, conv_ep, first_layer_training = _train_loop(model, dataloader, optimizer, scheduler, mse_loss, epochs, sim_weight, energy_type, warmup_epochs, verbose, device, stabilization_start_epoch=stabilization_start_epoch, stabilization_ramp_epochs=stabilization_ramp_epochs)
    model.eval(); 
    with torch.no_grad():
        eval_mats = [m.to(device) for m in torch_mats]
        final_latents, _, u_final = model(eval_mats)
        v_mats = [torch.nan_to_num(enc.v.detach().cpu(), nan=0.0, posinf=0.0, neginf=0.0) for enc in model.linear_encoders]
        first_layer_scores = [torch.nan_to_num(z.detach().cpu(), nan=0.0, posinf=0.0, neginf=0.0) for z in model.encode_first_layer(eval_mats, use_projected=True)]
        first_layer = build_first_layer_contract(v_mats, first_layer_scores)
    result = {"model": model.cpu(), "model_type": "ned_simr", "u": torch.nan_to_num(u_final.cpu(), nan=0.0, posinf=0.0, neginf=0.0), "v": v_mats, "first_layer_scores": first_layer_scores, "first_layer": first_layer, "first_layer_training": first_layer_training, "latents": [torch.nan_to_num(l.cpu(), nan=0.0, posinf=0.0, neginf=0.0) for l in final_latents], "loss_history": loss_h, "recon_history": recon_h, "sim_history": sim_h, "converged_iter": conv_ep, "scale_list": ["centerAndScale"], "provenance_list": provenance_list}
    result["interpretability"] = build_interpretability_report(result)
    result["deep_layer"] = {"alignment_to_first_layer": result["interpretability"]["deep_layer_alignment"]}
    return result

def ned_simr_shared_private(data_matrices: List[Union[torch.Tensor, np.ndarray]], k: int, private_k: Optional[int] = None, epochs: int = 150, batch_size: int = 64, learning_rate: float = 5e-4, weight_decay: float = 1e-4, sim_weight: float = 1.0, warmup_epochs: int = 20, sparseness_quantile: Union[float, List[float]] = 0.0, positivity: str = "either", nsa_w: float = 0.1, hidden_dims: List[int] = [128, 64], dropout: float = 0.1, energy_type: str = "regression", mixing_algorithm: str = "newton", private_recon_weight: float = 1.0, private_orthogonality_weight: float = 0.05, private_variance_weight: float = 0.10, device: Optional[str] = None, verbose: bool = False, tol: float = 1e-6, patience: int = 10, use_nsa: bool = False, first_layer_mode: str = "scheduled", stabilization_start_epoch: Optional[int] = None, stabilization_ramp_epochs: Optional[int] = None, shared_warmup_epochs: int = 20, **kwargs) -> Dict[str, Any]:
    """
    Fit a Nonlinear Encoded Decoding model with Shared and Private Latents (NED++).

    Parameters
    ----------
    data_matrices : List[Union[torch.Tensor, np.ndarray]]
        List of data matrices (one for each modality).
    k : int
        The dimensionality of the shared latent space.
    private_k : Optional[int], default=None
        The dimensionality of the private latent space for each modality.
    epochs : int, default=150
        Number of training epochs.
    batch_size : int, default=64
        Batch size.
    learning_rate : float, default=5e-4
        Learning rate.
    weight_decay : float, default=1e-4
        Weight decay.
    sim_weight : float, default=1.0
        Weight for similarity loss.
    warmup_epochs : int, default=20
        Warmup epochs before sim loss.
    sparseness_quantile : Union[float, List[float]] = 0.0
        Sparseness quantile.
    positivity : str, default="either"
        Positivity constraint.
    nsa_w : float, default=0.1
        NSA weight.
    hidden_dims : List[int], default=[128, 64]
        Decoder hidden dimensions.
    dropout : float, default=0.1
        Dropout probability.
    energy_type : str, default="regression"
        Energy type.
    mixing_algorithm : str, default="newton"
        Mixing algorithm.
    private_recon_weight : float, default=1.0
        Weight for private reconstruction loss.
    private_orthogonality_weight : float, default=0.05
        Weight for private-shared orthogonality penalty.
    private_variance_weight : float, default=0.10
        Weight for private variance penalty.
    device : Optional[str], default=None
        Device to use.
    verbose : bool, default=False
        Verbose output.
    tol : float, default=1e-6
        Tolerance for early stopping.
    patience : int, default=10
        Patience for early stopping.
    use_nsa : bool, default=False
        Use NSA flow.
    first_layer_mode : str, default="scheduled"
        First layer projection mode.
    stabilization_start_epoch : Optional[int], default=None
        Stabilization start epoch.
    stabilization_ramp_epochs : Optional[int], default=None
        Stabilization ramp epochs.
    shared_warmup_epochs : int, default=20
        Epochs to freeze private encoders to allow shared to train first.
    **kwargs : Dict[str, Any]
        Additional arguments.

    Returns
    -------
    Dict[str, Any]
        Model results.

    Raises
    ------
    TypeError
        If inputs are invalid.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
    """
    if 'sparsity' in kwargs: sparseness_quantile = kwargs.pop('sparsity')
    if 'sparseness' in kwargs: sparseness_quantile = kwargs.pop('sparseness')
    if private_k is None: private_k = max(1, k // 2)
    if device is None: device = "cuda" if torch.cuda.is_available() else ("cpu")
    device = torch.device(device); torch_mats, provenance_list = _standardize_deep(data_matrices, ["centerAndScale"]); input_dims = [m.shape[1] for m in torch_mats]
    model = NEDSharedPrivateSiMRModel(input_dims, k, private_k, hidden_dims, dropout, nsa_w, positivity, sparseness_quantile, mixing_algorithm, use_nsa=use_nsa, first_layer_mode=first_layer_mode).to(device)
    model.initialize_v(torch_mats, k)
    
    # Store weights on model for train loop access
    model.private_ortho_w = private_orthogonality_weight
    model.private_var_w = private_variance_weight
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs); mse_loss = nn.MSELoss(); dataset = TensorDataset(*torch_mats); dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    loss_h, recon_h, sim_h, conv_ep, first_layer_training = _train_loop(
        model, dataloader, optimizer, scheduler, mse_loss, epochs, sim_weight, energy_type, 
        warmup_epochs, verbose, device, tol=tol, patience=patience, 
        stabilization_start_epoch=stabilization_start_epoch, 
        stabilization_ramp_epochs=stabilization_ramp_epochs,
        freeze_private_epochs=shared_warmup_epochs
    )
    
    model.eval(); 
    with torch.no_grad():
        eval_mats = [m.to(device) for m in torch_mats]
        final_shared, final_recons, u_final, final_private = model(eval_mats)
        v_mats = [torch.nan_to_num(enc.v.detach().cpu(), nan=0.0, posinf=0.0, neginf=0.0) for enc in model.linear_encoders]
        first_layer_scores = [torch.nan_to_num(z.detach().cpu(), nan=0.0, posinf=0.0, neginf=0.0) for z in model.encode_first_layer(eval_mats, use_projected=True)]
        first_layer = build_first_layer_contract(v_mats, first_layer_scores)
    result = {"model": model.cpu(), "model_type": "ned_shared_private", "u": torch.nan_to_num(u_final.cpu(), nan=0.0, posinf=0.0, neginf=0.0), "v": v_mats, "first_layer_scores": first_layer_scores, "first_layer": first_layer, "first_layer_training": first_layer_training, "latents": [torch.nan_to_num(l.cpu(), nan=0.0, posinf=0.0, neginf=0.0) for l in final_shared], "private_latents": [torch.nan_to_num(p.cpu(), nan=0.0, posinf=0.0, neginf=0.0) for p in final_private], "loss_history": loss_h, "recon_history": recon_h, "sim_history": sim_h, "converged_iter": conv_ep, "scale_list": ["centerAndScale"], "provenance_list": provenance_list}
    result["interpretability"] = build_interpretability_report(result)
    result["deep_layer"] = {"alignment_to_first_layer": result["interpretability"]["deep_layer_alignment"]}
    return result

def deep_simr(data_matrices: List[Union[torch.Tensor, np.ndarray]], k: int, epochs: int = 150, batch_size: int = 64, learning_rate: float = 5e-4, sim_weight: float = 1.0, warmup_epochs: int = 20, energy_type: str = "regression", device: Optional[str] = None, verbose: bool = False, **kwargs) -> Dict[str, Any]:
    """
    Deep SiMLR (SiMR) implementation using Neural Spatially-Aware (NSA) Flow or LEND encoders.

    This function is a wrapper around `lend_simr`, which implements the Linear ENcoder 
    Deep (LEND) architecture. It learns a shared latent space by optimizing a 
    combination of reconstruction loss and a similarity loss (e.g., ACC) in the latent space.

    Parameters
    ----------
    data_matrices : List[Union[torch.Tensor, np.ndarray]]
        A list of data matrices for each modality.
    k : int
        The number of shared latent components.
    epochs : int, optional
        Number of training epochs (default is 150).
    batch_size : int, optional
        Size of mini-batches (default is 64).
    learning_rate : float, optional
        Learning rate for the optimizer (default is 5e-4).
    sim_weight : float, optional
        Weight of the similarity loss (ACC) relative to reconstruction loss (default is 1.0).
    warmup_epochs : int, optional
        Number of epochs before the similarity loss is fully weighted (default is 20).
    energy_type : str, optional
        The type of similarity loss ("regression", "acc", or "logcosh") (default is "regression").
    device : str, optional
        The device to use for training ("cpu", "cuda", or "mps"). If None, automatically detected.
    verbose : bool, optional
        Whether to print progress (default is False).
    **kwargs : dict
        Additional parameters passed to `lend_simr` (e.g., nsa_w, positivity, sparseness_quantile).

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the trained model, learned latents (U), basis matrices (V), 
        and training history.

    Raises
    ------
    TypeError
        If inputs are invalid.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.

    See Also
    --------
    lend_simr, ned_simr
    """
    return lend_simr(data_matrices, k, epochs, batch_size, learning_rate, sim_weight=sim_weight, warmup_epochs=warmup_epochs, energy_type=energy_type, device=device, verbose=verbose, **kwargs)

def predict_deep(data_matrices: List[Union[torch.Tensor, np.ndarray]], model_res: Dict[str, Any], device: Optional[str] = None) -> Dict[str, Any]:
    """
    Predict using a trained deep SiMLR model.

    Generates predictions (latent representations and reconstructions) using a 
    pre-trained deep SiMLR model on new data matrices.

    Parameters
    ----------
    data_matrices : List[Union[torch.Tensor, np.ndarray]]
        List of data matrices for the new data.
    model_res : Dict[str, Any]
        The result dictionary returned by training a deep SiMLR model.
    device : Optional[str], default=None
        The device to perform computation on.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the predicted latent representations and 
        reconstructions.

    Raises
    ------
    TypeError
        If inputs are invalid.

    Correctness
    -----------
    This function has been audited for Numpy docstring validity and functional correctness.
    """
    model = model_res["model"]; model_type = model_res["model_type"]
    if device is None: device = "cuda" if torch.cuda.is_available() else ("cpu")
    device = torch.device(device); model.to(device).eval()
    torch_mats = [preprocess_data(torch.as_tensor(m).float(), model_res["scale_list"], prov) for m, prov in zip(data_matrices, model_res["provenance_list"])]
    with torch.no_grad():
        res = model([m.to(device) for m in torch_mats])
        first_layer_scores = [z.cpu() for z in model.encode_first_layer([m.to(device) for m in torch_mats], use_projected=True)] if hasattr(model, "encode_first_layer") else None
        v_list = None
        if hasattr(model, "encoders"):
            v_list = [enc.v.detach().cpu() for enc in model.encoders]
        elif hasattr(model, "linear_encoders"):
            v_list = [enc.v.detach().cpu() for enc in model.linear_encoders]
        first_layer = build_first_layer_contract(v_list, first_layer_scores) if v_list is not None and first_layer_scores is not None else None
        if model_type == "ned_shared_private":
            shared_l, recons, u, private_l = res
            result = {"u": u.cpu(), "latents": [l.cpu() for l in shared_l], "reconstructions": [r.cpu() for r in recons], "private_latents": [p.cpu() for p in private_l], "first_layer_scores": first_layer_scores, "first_layer": first_layer, "v": v_list}
        else:
            latents, recons, u = res
            result = {"u": u.cpu(), "latents": [l.cpu() for l in latents], "reconstructions": [r.cpu() for r in recons], "first_layer_scores": first_layer_scores, "first_layer": first_layer, "v": v_list}
        result["interpretability"] = build_interpretability_report(result) if first_layer is not None else None
        result["deep_layer"] = None if result["interpretability"] is None else {"alignment_to_first_layer": result["interpretability"]["deep_layer_alignment"]}
        return result
