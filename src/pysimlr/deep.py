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
from .utils import preprocess_data, invariant_orthogonality_defect
from .interpretability import build_first_layer_contract, build_interpretability_report

try:
    import nsa_flow as nsa
except ImportError:
    nsa = None

def _svd_project_columns(u: torch.Tensor) -> torch.Tensor:
    """Project towards the Stiefel manifold using SVD."""
    try:
        u_svd, _, vh_svd = torch.linalg.svd(u, full_matrices=False)
        return u_svd @ vh_svd
    except:
        return torch.nn.functional.normalize(u, p=2, dim=0)

def _normalize_rows(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize each row to have unit L2 norm."""
    norm = x.norm(dim=1, keepdim=True)
    return x / (norm + eps)

def _variance_penalty(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """VICReg style variance penalty. gamma is the target standard deviation."""
    if z.shape[0] < 2: return torch.tensor(0.0, device=z.device)
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    return torch.mean(torch.nn.functional.relu(gamma - std))

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
        if torch.any(torch.std(m_scaled, dim=0) < 1e-6):
            m_scaled = m_scaled + torch.randn_like(m_scaled) * 1e-6
        scaled_mats.append(m_scaled)
        provenance_list.append(prov)
    return scaled_mats, provenance_list

class LENDNSAEncoder(nn.Module):
    """
    Stiefel-constrained Linear Encoder with Non-negative flow support.

    This layer implements the interpretable bottleneck of the LEND architecture. It
    maintains linear feature weights V that are constrained to the Stiefel manifold.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    latent_dim : int
        Target latent dimensionality.
    nsa_w : float, default=0.5
        Strength of the Non-negative Stiefel approximating flow.
    positivity : str, default="either"
        "positive" or "hard" to enforce non-negativity.
    sparseness_quantile : float, default=0.0
        Proportion of weights to zero-out (0.0 to 1.0).
    use_nsa : bool, default=False
        Whether to use the formal NSA-Flow layer for retraction.
    first_layer_mode : str, default="scheduled"
        Training mode for weights: "raw", "projected", or "scheduled".
    """
    def __init__(self, input_dim: int, latent_dim: int, nsa_w: float = 0.5, 
                 positivity: str = "either", sparseness_quantile: float = 0.0,
                 soft_thresholding: bool = False, use_nsa: bool = False,
                 first_layer_mode: str = "scheduled"):
        super().__init__()
        self.v_raw = nn.Parameter(torch.randn(input_dim, latent_dim) * 0.01)
        self.positivity = positivity
        self.sparseness_quantile = sparseness_quantile
        self.soft_thresholding = soft_thresholding
        self.use_nsa = use_nsa
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
        """Returns the constrained basis matrix V."""
        if self.nsa_layer is not None: v_out = self.nsa_layer(self.v_raw)
        else:
            try: v_out = _svd_project_columns(self.v_raw)
            except: v_out = torch.nn.functional.normalize(self.v_raw, p=2, dim=0)
            if self.positivity in ['positive', 'hard']: v_out = torch.clamp(v_out, min=0.0)
        if torch.isnan(v_out).any(): v_out = torch.nan_to_num(v_out, nan=0.0)
        if self.sparseness_quantile > 0:
            v_abs = torch.abs(v_out) if self.positivity == "either" else v_out
            q_vals = torch.quantile(v_abs, self.sparseness_quantile, dim=0, keepdim=True)
            if self.soft_thresholding: v_out = torch.sign(v_out) * torch.clamp(v_abs - q_vals, min=0.0)
            else: v_out = v_out * (v_abs >= q_vals).float()
        return v_out

    def forward(self, x):
        """Encode input x into latent scores."""
        return x @ (self.active_training_basis() if self.training else self.v)

    def get_projector(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """Unify downstream prediction API: Returns a linear projection function."""
        v_current = self.v.detach().clone()
        return lambda x: x @ v_current

class ModalityDecoder(nn.Module):
    """Deep Nonlinear Decoder for a single modality."""
    def __init__(self, latent_dim: int, output_dim: int, hidden_dims: List[int] = [64, 128], dropout: float = 0.1):
        super().__init__()
        layers = []
        curr_dim = latent_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim)); layers.append(nn.ReLU())
            if dropout > 0: layers.append(nn.Dropout(dropout))
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, output_dim))
        self.network = nn.Sequential(*layers)
    def forward(self, z): return self.network(z)

class LENDSiMRModel(nn.Module):
    """
    Linear Encoder, Nonlinear Decoder (LEND) Model.

    Parameters
    ----------
    input_dims : list of int
        Feature dimensions for each modality.
    latent_dim : int
        Shared latent dimensionality.
    """
    def __init__(self, input_dims: List[int], latent_dim: int, hidden_dims: List[int] = [64, 128], 
                 dropout: float = 0.1, nsa_w: float = 0.5, positivity: str = "either", 
                 sparseness_quantile: float = 0.0, mixing_algorithm: str = "newton",
                 use_nsa: bool = False, first_layer_mode: str = "scheduled"):
        super().__init__()
        self.encoders = nn.ModuleList([LENDNSAEncoder(dim, latent_dim, nsa_w, positivity, sparseness_quantile, use_nsa=use_nsa, first_layer_mode=first_layer_mode) for dim in input_dims])
        self.decoders = nn.ModuleList([ModalityDecoder(latent_dim, dim, hidden_dims, dropout) for dim in input_dims])
        self.mixing_algorithm, self.latent_dim = mixing_algorithm, latent_dim

    def forward(self, x_list: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        latents = [enc(x) for enc, x in zip(self.encoders, x_list)]
        u_shared = compute_shared_consensus(latents, mixing_algorithm=self.mixing_algorithm, k=self.latent_dim, training=self.training)
        return latents, [dec(u_shared) for dec in self.decoders], u_shared

class NEDSiMRModel(nn.Module):
    """Fully Nonlinear Encoder-Decoder (NED) Model."""
    def __init__(self, input_dims: List[int], latent_dim: int, hidden_dims: List[int] = [128, 64], 
                 dropout: float = 0.1, nsa_w: float = 0.5, positivity: str = "either", 
                 sparseness_quantile: float = 0.0, mixing_algorithm: str = "newton",
                 use_nsa: bool = False, first_layer_mode: str = "scheduled"):
        super().__init__()
        self.linear_encoders = nn.ModuleList([LENDNSAEncoder(dim, latent_dim, nsa_w, positivity, sparseness_quantile, use_nsa=use_nsa, first_layer_mode=first_layer_mode) for dim in input_dims])
        self.nonlinear_heads = nn.ModuleList([ModalityDecoder(latent_dim, latent_dim, hidden_dims, dropout) for _ in input_dims])
        self.decoders = nn.ModuleList([ModalityDecoder(latent_dim, dim, hidden_dims, dropout) for dim in input_dims])
        self.mixing_algorithm, self.latent_dim = mixing_algorithm, latent_dim

    def forward(self, x_list: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        z0 = [enc(x) for enc, x in zip(self.linear_encoders, x_list)]
        latents = [head(z) for head, z in zip(self.nonlinear_heads, z0)]
        u_shared = compute_shared_consensus(latents, mixing_algorithm=self.mixing_algorithm, k=self.latent_dim, training=self.training)
        return latents, [dec(u_shared) for dec in self.decoders], u_shared

    def transform(self, x_list: List[torch.Tensor]) -> torch.Tensor:
        """Unify deep encoding API: Returns the consensus representation U."""
        self.eval()
        with torch.no_grad():
            _, _, u = self.forward(x_list)
            return u

class NEDSharedPrivateSiMRModel(nn.Module):
    """NED Architecture with Shared and Private Latent Partitioning."""
    def __init__(self, input_dims: List[int], shared_latent_dim: int, private_latent_dim: int,
                 hidden_dims: List[int] = [128, 64], dropout: float = 0.1, nsa_w: float = 0.5,
                 positivity: str = "either", sparseness_quantile: float = 0.0, mixing_algorithm: str = "newton",
                 use_nsa: bool = False, first_layer_mode: str = "scheduled"):
        super().__init__()
        self.linear_encoders = nn.ModuleList([LENDNSAEncoder(dim, shared_latent_dim, nsa_w, positivity, sparseness_quantile, use_nsa=use_nsa, first_layer_mode=first_layer_mode) for dim in input_dims])
        self.shared_heads = nn.ModuleList([ModalityDecoder(shared_latent_dim, shared_latent_dim, hidden_dims, dropout) for _ in input_dims])
        self.private_encoders = nn.ModuleList([ModalityEncoder(dim, private_latent_dim, hidden_dims, dropout) for dim in input_dims])
        self.decoders = nn.ModuleList([ModalityDecoder(shared_latent_dim + private_latent_dim, dim, hidden_dims, dropout) for dim in input_dims])
        self.mixing_algorithm, self.shared_dim = mixing_algorithm, shared_latent_dim

    def forward(self, x_list: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, List[torch.Tensor]]:
        z0 = [enc(x) for enc, x in zip(self.linear_encoders, x_list)]
        shared_l = [head(z) for head, z in zip(self.shared_heads, z0)]
        private_l = [p_enc(x) for p_enc, x in zip(self.private_encoders, x_list)]
        u_shared = compute_shared_consensus(shared_l, mixing_algorithm=self.mixing_algorithm, k=self.shared_dim, training=self.training)
        recons = [dec(torch.cat([u_shared, p], dim=1)) for dec, p in zip(self.decoders, private_l)]
        return shared_l, recons, u_shared, private_l

class ModalityEncoder(nn.Module):
    """Deep Nonlinear Encoder for a single modality."""
    def __init__(self, input_dim: int, k: int, hidden_dims: List[int] = [128, 64], dropout: float = 0.1):
        super().__init__()
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim)); layers.append(nn.ReLU())
            if dropout > 0: layers.append(nn.Dropout(dropout))
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, k))
        self.network = nn.Sequential(*layers)
    def forward(self, x): return self.network(x)

def lend_simr(data_matrices: List[Union[torch.Tensor, np.ndarray]], k: int, epochs: int = 150, batch_size: int = 64, learning_rate: float = 5e-4, weight_decay: float = 1e-4, sim_weight: float = 1.0, warmup_epochs: int = 20, hidden_dims: List[int] = [64, 128], dropout: float = 0.1, sparseness_quantile: float = 0.0, positivity: str = "either", nsa_w: float = 0.5, energy_type: str = "regression", mixing_algorithm: str = "newton", device: Optional[str] = None, verbose: bool = False, use_nsa: bool = False, first_layer_mode: str = "scheduled", stabilization_start_epoch: Optional[int] = None, stabilization_ramp_epochs: Optional[int] = None) -> Dict[str, Any]:
    """
    High-level entry point for the LEND (Linear Encoder, Nonlinear Decoder) architecture.

    Parameters
    ----------
    data_matrices : list of (Tensor or ndarray)
        The input modalities.
    k : int
        Target dimension.
    epochs : int, default=150
        Training iterations.
    positivity : str, default="either"
        Weights constraint ("positive" or "either").
    nsa_w : float, default=0.5
        NSA-Flow weight for Stiefel manifold stabilization.

    Returns
    -------
    dict
        Model results including shared latent 'u', weights 'v', and 'interpretability' report.
    """
    if device is None: device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
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
    return result

def ned_simr(data_matrices: List[Union[torch.Tensor, np.ndarray]], k: int, epochs: int = 150, batch_size: int = 64, learning_rate: float = 5e-4, weight_decay: float = 1e-4, sim_weight: float = 1.0, warmup_epochs: int = 20, hidden_dims: List[int] = [128, 64], dropout: float = 0.1, sparseness_quantile: float = 0.0, positivity: str = "either", nsa_w: float = 0.5, energy_type: str = "regression", mixing_algorithm: str = "newton", device: Optional[str] = None, verbose: bool = False, use_nsa: bool = False, first_layer_mode: str = "scheduled", stabilization_start_epoch: Optional[int] = None, stabilization_ramp_epochs: Optional[int] = None) -> Dict[str, Any]:
    """Fully Nonlinear Encoder-Decoder (NED) multi-view learning."""
    if device is None: device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
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
    return result

def predict_deep(data_matrices: List[Union[torch.Tensor, np.ndarray]], model_res: Dict[str, Any], device: Optional[str] = None) -> Dict[str, Any]:
    """Prediction for trained deep SiMLR models."""
    model = model_res["model"]; model_type = model_res["model_type"]
    if device is None: device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(device); model.to(device).eval()
    torch_mats = [preprocess_data(torch.as_tensor(m).float(), model_res["scale_list"], prov) for m, prov in zip(data_matrices, model_res["provenance_list"])]
    with torch.no_grad():
        res = model([m.to(device) for m in torch_mats])
        first_layer_scores = [z.cpu() for z in model.encode_first_layer([m.to(device) for m in torch_mats], use_projected=True)] if hasattr(model, "encode_first_layer") else None
        v_list = [enc.v.detach().cpu() for enc in getattr(model, "encoders", getattr(model, "linear_encoders", []))]
        first_layer = build_first_layer_contract(v_list, first_layer_scores) if v_list else None
        if model_type == "ned_shared_private":
            shared_l, recons, u, private_l = res
            result = {"u": u.cpu(), "latents": [l.cpu() for l in shared_l], "reconstructions": [r.cpu() for r in recons], "private_latents": [p.cpu() for p in private_l], "first_layer_scores": first_layer_scores, "first_layer": first_layer, "v": v_list}
        else:
            latents, recons, u = res
            result = {"u": u.cpu(), "latents": [l.cpu() for l in latents], "reconstructions": [r.cpu() for r in recons], "first_layer_scores": first_layer_scores, "first_layer": first_layer, "v": v_list}
        result["interpretability"] = build_interpretability_report(result) if first_layer else None
        return result
