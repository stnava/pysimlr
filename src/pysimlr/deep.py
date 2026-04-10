import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from .simlr import calculate_u, ba_svd
from .utils import preprocess_data

try:
    import nsa_flow as nsa
except ImportError:
    nsa = None

def _newton_step_ortho(u: torch.Tensor) -> torch.Tensor:
    """Project towards the Stiefel manifold using SVD."""
    try:
        u_svd, _, vh_svd = torch.linalg.svd(u, full_matrices=False)
        return u_svd @ vh_svd
    except:
        return torch.nn.functional.normalize(u, p=2, dim=0)
    """Algebraic Newton-style projection towards the Stiefel manifold."""

def _normalize_rows(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize each row to have unit L2 norm."""
    norm = x.norm(dim=1, keepdim=True)
    return x / (norm + eps)

def _variance_penalty(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """VICReg style variance penalty."""
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    return torch.mean(torch.nn.functional.relu(gamma - std))

def _covariance_penalty(z: torch.Tensor) -> torch.Tensor:
    """VICReg style covariance penalty to prevent dimensional collapse."""
    n, d = z.shape
    z = z - z.mean(dim=0)
    cov = (z.t() @ z) / (n - 1)
    mask = ~torch.eye(d, device=z.device).bool()
    return torch.mean(cov[mask]**2)

def _standardize_deep(data_matrices, scale_list=["centerAndScale"]):
    torch_mats = [torch.as_tensor(m).float() for m in data_matrices]
    provenance_list = []
    scaled_mats = []
    for m in torch_mats:
        m_scaled, prov = preprocess_data(m, scale_list)
        scaled_mats.append(m_scaled)
        provenance_list.append(prov)
    return scaled_mats, provenance_list

class LENDNSAEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, nsa_w: float = 0.5, 
                 positivity: str = "either", sparseness_quantile: float = 0.0,
                 soft_thresholding: bool = False):
        super().__init__()
        self.v_raw = nn.Parameter(torch.randn(input_dim, latent_dim) * 0.01)
        self.positivity = positivity
        self.sparseness_quantile = sparseness_quantile
        self.soft_thresholding = soft_thresholding
        
        apply_nonneg = 'hard' if positivity in ['positive', 'hard'] else 'none'
        if nsa is not None:
            self.nsa_layer = nsa.NSAFlowLayer(
                k=latent_dim, w_retract=nsa_w, retraction_type="polar", 
                apply_nonneg=apply_nonneg, residual=False, use_transform=False
            )
        else:
            self.nsa_layer = None

    @property
    def v(self):
        if self.nsa_layer is not None:
            v_out = self.nsa_layer(self.v_raw)
        else:
            try:
                # Primary fallback: Newton step (stable, fast, gradient-friendly)
                v_out = _newton_step_ortho(self.v_raw)
            except:
                try:
                    u_svd, _, v_svd = torch.linalg.svd(self.v_raw, full_matrices=False)
                    v_out = u_svd @ v_svd
                except:
                    v_out = torch.nn.functional.normalize(self.v_raw, p=2, dim=0)
            
            if self.positivity in ['positive', 'hard']:
                v_out = torch.clamp(v_out, min=0.0)
        
        if torch.isnan(v_out).any():
            v_out = torch.nan_to_num(v_out, nan=0.0)
            
        if self.sparseness_quantile > 0:
            v_abs = torch.abs(v_out) if self.positivity == "either" else v_out
            # Vectorized quantile over columns
            q_vals = torch.quantile(v_abs, self.sparseness_quantile, dim=0, keepdim=True)
            
            if self.soft_thresholding:
                v_out = torch.sign(v_out) * torch.clamp(v_abs - q_vals, min=0.0)
            else:
                mask = (v_abs >= q_vals).float()
                v_out = v_out * mask
                
        return v_out

    def forward(self, x): return x @ self.v

class ModalityDecoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int, hidden_dims: List[int] = [64, 128], dropout: float = 0.1):
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
    def __init__(self, input_dims: List[int], latent_dim: int, hidden_dims: List[int] = [64, 128], 
                 dropout: float = 0.1, nsa_w: float = 0.5, positivity: str = "either", 
                 sparseness_quantile: float = 0.0, mixing_algorithm: str = "newton"):
        super().__init__()
        self.encoders = nn.ModuleList([LENDNSAEncoder(dim, latent_dim, nsa_w, positivity, sparseness_quantile) for dim in input_dims])
        self.decoders = nn.ModuleList([ModalityDecoder(latent_dim, dim, hidden_dims, dropout) for dim in input_dims])
        self.mixing_algorithm, self.latent_dim = mixing_algorithm, latent_dim
    def initialize_v(self, data_matrices: List[torch.Tensor], k: int):
        with torch.no_grad():
            for i, x in enumerate(data_matrices):
                u, s, v = ba_svd(x, nu=0, nv=k)
                if v.shape[1] < k: v = torch.cat([v, torch.randn(v.shape[0], k-v.shape[1], device=v.device)*1e-4], dim=1)
                self.encoders[i].v_raw.copy_(v.to(x.dtype))
    def forward(self, x_list: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        latents = [enc(x) for enc, x in zip(self.encoders, x_list)]
        if self.training:
            u_shared = torch.mean(torch.stack(latents), dim=0)
            if self.mixing_algorithm == "newton": u_shared = _newton_step_ortho(u_shared)
        else:
            with torch.no_grad(): u_shared = calculate_u(latents, mixing_algorithm=self.mixing_algorithm, k=self.latent_dim)
        return latents, [dec(u_shared) for dec in self.decoders], u_shared

class NEDSiMRModel(nn.Module):
    def __init__(self, input_dims: List[int], latent_dim: int, hidden_dims: List[int] = [128, 64], 
                 dropout: float = 0.1, nsa_w: float = 0.5, positivity: str = "either", 
                 sparseness_quantile: float = 0.0, mixing_algorithm: str = "newton"):
        super().__init__()
        self.linear_encoders = nn.ModuleList([LENDNSAEncoder(dim, latent_dim, nsa_w, positivity, sparseness_quantile) for dim in input_dims])
        self.nonlinear_heads = nn.ModuleList([ModalityDecoder(latent_dim, latent_dim, hidden_dims, dropout) for _ in input_dims])
        self.decoders = nn.ModuleList([ModalityDecoder(latent_dim, dim, hidden_dims, dropout) for dim in input_dims])
        self.mixing_algorithm, self.latent_dim = mixing_algorithm, latent_dim
    def initialize_v(self, data_matrices: List[torch.Tensor], k: int):
        with torch.no_grad():
            for i, x in enumerate(data_matrices):
                u, s, v = ba_svd(x, nu=0, nv=k)
                if v.shape[1] < k: v = torch.cat([v, torch.randn(v.shape[0], k-v.shape[1], device=v.device)*1e-4], dim=1)
                self.linear_encoders[i].v_raw.copy_(v.to(x.dtype))
    def forward(self, x_list: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        latents = [head(enc(x)) for enc, head, x in zip(self.linear_encoders, self.nonlinear_heads, x_list)]
        if self.training:
            u_shared = torch.mean(torch.stack(latents), dim=0)
            if self.mixing_algorithm == "newton": u_shared = _newton_step_ortho(u_shared)
        else:
            with torch.no_grad(): u_shared = calculate_u(latents, mixing_algorithm=self.mixing_algorithm, k=self.latent_dim)
        return latents, [dec(u_shared) for dec in self.decoders], u_shared

class NEDSharedPrivateSiMRModel(nn.Module):
    def __init__(self, input_dims: List[int], shared_latent_dim: int, private_latent_dim: int,
                 hidden_dims: List[int] = [128, 64], dropout: float = 0.1, nsa_w: float = 0.5,
                 positivity: str = "either", sparseness_quantile: float = 0.0, mixing_algorithm: str = "newton"):
        super().__init__()
        self.linear_encoders = nn.ModuleList([LENDNSAEncoder(dim, shared_latent_dim, nsa_w, positivity, sparseness_quantile) for dim in input_dims])
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
    def forward(self, x_list: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, List[torch.Tensor]]:
        shared_l = [head(enc(x)) for enc, head, x in zip(self.linear_encoders, self.shared_heads, x_list)]
        private_l = [p_enc(x) for p_enc, x in zip(self.private_encoders, x_list)]
        if self.training:
            u_shared = torch.mean(torch.stack(shared_l), dim=0)
            if self.mixing_algorithm == "newton": u_shared = _newton_step_ortho(u_shared)
        else:
            with torch.no_grad(): u_shared = calculate_u(shared_l, mixing_algorithm=self.mixing_algorithm, k=self.shared_dim)
        recons = [dec(torch.cat([u_shared, p], dim=1)) for dec, p in zip(self.decoders, private_l)]
        return shared_l, recons, u_shared, private_l

class ModalityEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: List[int] = [128, 64], dropout: float = 0.1):
        super().__init__()
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            if dropout > 0: layers.append(nn.Dropout(dropout))
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, latent_dim))
        self.network = nn.Sequential(*layers)
    def forward(self, x): return self.network(x)

def calculate_sim_loss(latents: List[torch.Tensor], u_shared: torch.Tensor, energy_type: str = "regression") -> torch.Tensor:
    sim_loss = torch.tensor(0.0, device=u_shared.device)
    n, u_target = u_shared.shape[0], u_shared.detach()
    var_penalty = sum(_variance_penalty(z) for z in latents)
    for z in latents:
        if energy_type == "regression":
            z_std, u_std = torch.std(z, dim=0, keepdim=True) + 1e-6, torch.std(u_target, dim=0, keepdim=True) + 1e-6
            sim_loss += torch.mean(((z / z_std) - (u_target / u_std))**2)
        elif energy_type == "acc":
            cov = ((u_target - u_target.mean(0)).t() @ (z - z.mean(0))) / (n - 1)
            sim_loss -= torch.sum(torch.abs(cov))
        elif energy_type == "logcosh":
            s = u_target.t() @ z; abs_s = torch.abs(s)
            sim_loss -= torch.sum(abs_s - np.log(2.0) + torch.log1p(torch.exp(-2.0 * abs_s))) / n
    u_c = u_shared - torch.mean(u_shared, dim=0)
    cov_u = (u_c.t() @ u_c) / (n - 1)
    collapse_loss = torch.sum((cov_u - torch.diag(torch.diag(cov_u)))**2) / u_shared.shape[1]
    return sim_loss + 0.1 * var_penalty + 0.1 * collapse_loss

def _train_loop(model, dataloader, optimizer, scheduler, mse_loss, epochs, sim_weight, energy_type, warmup_epochs, verbose, device, tol=1e-6, patience=10):
    loss_history, recon_history, sim_history = [], [], []
    best_loss = float('inf'); patience_counter = 0; converged_epoch = epochs
    for epoch in range(epochs):
        model.train(); epoch_loss, epoch_recon, epoch_sim = 0.0, 0.0, 0.0
        current_sim_weight = 0.0 if epoch < warmup_epochs else sim_weight
        for batch in dataloader:
            batch_mats = [b.to(device) for b in batch]; optimizer.zero_grad()
            res = model(batch_mats)
            shared_latents, reconstructions, u_shared = res[0], res[1], res[2]
            recon_loss = sum(mse_loss(r, x) for r, x in zip(reconstructions, batch_mats))
            sim_loss = calculate_sim_loss(shared_latents, u_shared, energy_type)
            total_loss = recon_loss + current_sim_weight * sim_loss
            if torch.isnan(total_loss): continue
            total_loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0); optimizer.step()
            epoch_loss += total_loss.item(); epoch_recon += recon_loss.item(); epoch_sim += sim_loss.item()
        epoch_loss /= len(dataloader); epoch_recon /= len(dataloader); epoch_sim /= len(dataloader)
        loss_history.append(epoch_loss); recon_history.append(epoch_recon); sim_history.append(epoch_sim)
        scheduler.step()
        if epoch_loss < best_loss - tol: best_loss = epoch_loss; patience_counter = 0
        else: patience_counter += 1
        if patience_counter >= patience and epoch > warmup_epochs:
            if verbose: print(f"Converged at epoch {epoch}: Total Loss {epoch_loss:.4f}")
            converged_epoch = epoch + 1; break
        if verbose and epoch % 10 == 0: print(f"Epoch {epoch}: Total={epoch_loss:.4f} (Recon={epoch_recon:.4f}, Sim={epoch_sim:.4f})")
    return loss_history, recon_history, sim_history, converged_epoch

def lend_simr(data_matrices: List[Union[torch.Tensor, np.ndarray]], k: int, epochs: int = 150, batch_size: int = 64, learning_rate: float = 5e-4, weight_decay: float = 1e-4, sim_weight: float = 1.0, warmup_epochs: int = 20, hidden_dims: List[int] = [64, 128], dropout: float = 0.1, sparseness_quantile: float = 0.0, positivity: str = "either", nsa_w: float = 0.5, energy_type: str = "regression", mixing_algorithm: str = "newton", device: Optional[str] = None, verbose: bool = False) -> Dict[str, Any]:
    if device is None: device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(device); torch_mats, provenance_list = _standardize_deep(data_matrices, ["centerAndScale"]); input_dims = [m.shape[1] for m in torch_mats]
    model = LENDSiMRModel(input_dims, k, hidden_dims, dropout, nsa_w, positivity, sparseness_quantile, mixing_algorithm).to(device)
    model.initialize_v(torch_mats, k)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs); mse_loss = nn.MSELoss(); dataset = TensorDataset(*torch_mats); dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_h, recon_h, sim_h, conv_ep = _train_loop(model, dataloader, optimizer, scheduler, mse_loss, epochs, sim_weight, energy_type, warmup_epochs, verbose, device)
    model.eval(); 
    with torch.no_grad():
        final_latents, _, u_final = model([m.to(device) for m in torch_mats])
        v_mats = [torch.nan_to_num(enc.v.detach().cpu(), nan=0.0, posinf=0.0, neginf=0.0) for enc in model.encoders]
    return {"model": model.cpu(), "model_type": "lend_simr", "u": torch.nan_to_num(u_final.cpu(), nan=0.0, posinf=0.0, neginf=0.0), "v": v_mats, "latents": [torch.nan_to_num(l.cpu(), nan=0.0, posinf=0.0, neginf=0.0) for l in final_latents], "loss_history": loss_h, "recon_history": recon_h, "sim_history": sim_h, "converged_iter": conv_ep, "scale_list": ["centerAndScale"], "provenance_list": provenance_list}

def ned_simr(data_matrices: List[Union[torch.Tensor, np.ndarray]], k: int, epochs: int = 150, batch_size: int = 64, learning_rate: float = 5e-4, weight_decay: float = 1e-4, sim_weight: float = 1.0, warmup_epochs: int = 20, hidden_dims: List[int] = [128, 64], dropout: float = 0.1, sparseness_quantile: float = 0.0, positivity: str = "either", nsa_w: float = 0.5, energy_type: str = "regression", mixing_algorithm: str = "newton", device: Optional[str] = None, verbose: bool = False) -> Dict[str, Any]:
    if device is None: device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(device); torch_mats, provenance_list = _standardize_deep(data_matrices, ["centerAndScale"]); input_dims = [m.shape[1] for m in torch_mats]
    model = NEDSiMRModel(input_dims, k, hidden_dims, dropout, nsa_w, positivity, sparseness_quantile, mixing_algorithm).to(device)
    model.initialize_v(torch_mats, k)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs); mse_loss = nn.MSELoss(); dataset = TensorDataset(*torch_mats); dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_h, recon_h, sim_h, conv_ep = _train_loop(model, dataloader, optimizer, scheduler, mse_loss, epochs, sim_weight, energy_type, warmup_epochs, verbose, device)
    model.eval(); 
    with torch.no_grad():
        final_latents, _, u_final = model([m.to(device) for m in torch_mats])
        v_mats = [torch.nan_to_num(enc.v.detach().cpu(), nan=0.0, posinf=0.0, neginf=0.0) for enc in model.linear_encoders]
    return {"model": model.cpu(), "model_type": "ned_simr", "u": torch.nan_to_num(u_final.cpu(), nan=0.0, posinf=0.0, neginf=0.0), "v": v_mats, "latents": [torch.nan_to_num(l.cpu(), nan=0.0, posinf=0.0, neginf=0.0) for l in final_latents], "loss_history": loss_h, "recon_history": recon_h, "sim_history": sim_h, "converged_iter": conv_ep, "scale_list": ["centerAndScale"], "provenance_list": provenance_list}

def ned_simr_shared_private(data_matrices: List[Union[torch.Tensor, np.ndarray]], k: int, private_k: Optional[int] = None, epochs: int = 150, batch_size: int = 64, learning_rate: float = 5e-4, weight_decay: float = 1e-4, sim_weight: float = 1.0, warmup_epochs: int = 20, sparseness_quantile: float = 0.0, positivity: str = "either", nsa_w: float = 0.5, hidden_dims: List[int] = [128, 64], dropout: float = 0.1, energy_type: str = "regression", mixing_algorithm: str = "newton", private_recon_weight: float = 1.0, private_orthogonality_weight: float = 0.05, private_variance_weight: float = 0.10, device: Optional[str] = None, verbose: bool = False, tol: float = 1e-6, patience: int = 10) -> Dict[str, Any]:
    if private_k is None: private_k = max(1, k // 2)
    if device is None: device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(device); torch_mats, provenance_list = _standardize_deep(data_matrices, ["centerAndScale"]); input_dims = [m.shape[1] for m in torch_mats]
    model = NEDSharedPrivateSiMRModel(input_dims, k, private_k, hidden_dims, dropout, nsa_w, positivity, sparseness_quantile, mixing_algorithm).to(device)
    model.initialize_v(torch_mats, k)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs); mse_loss = nn.MSELoss(); dataset = TensorDataset(*torch_mats); dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_history, recon_history, sim_history = [], [], []
    best_loss = float('inf'); patience_counter = 0; converged_epoch = epochs
    for epoch in range(epochs):
        model.train(); epoch_loss, epoch_recon, epoch_sim = 0.0, 0.0, 0.0
        current_sim_weight = 0.0 if epoch < warmup_epochs else sim_weight
        for batch in dataloader:
            batch_mats = [b.to(device) for b in batch]; optimizer.zero_grad()
            shared_l, recons, u_shared, private_l = model(batch_mats)
            recon_loss = sum(mse_loss(r, x) for r, x in zip(recons, batch_mats))
            sim_loss = calculate_sim_loss(shared_l, u_shared, energy_type)
            # Handle possible dimension mismatch in concatenation if using dummy zeros
            p_recon_loss = sum(mse_loss(model.decoders[i](torch.cat([torch.zeros_like(u_shared), p], dim=1)), batch_mats[i]) for i, p in enumerate(private_l))
            # Cross-orthogonality proxy: minimize squared cross-product sum
            p_ortho_loss = sum(torch.sum((u_shared.t() @ p)**2) for p in private_l)
            p_var_loss = sum(_variance_penalty(p) for p in private_l)
            total_loss = recon_loss + current_sim_weight * sim_loss + private_recon_weight * p_recon_loss + private_orthogonality_weight * p_ortho_loss + private_variance_weight * p_var_loss
            if torch.isnan(total_loss): continue
            total_loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0); optimizer.step()
            epoch_loss += total_loss.item(); epoch_recon += recon_loss.item(); epoch_sim += sim_loss.item()
        epoch_loss /= len(dataloader); epoch_recon /= len(dataloader); epoch_sim /= len(dataloader)
        loss_history.append(epoch_loss); recon_history.append(epoch_recon); sim_history.append(epoch_sim)
        scheduler.step()
        if epoch_loss < best_loss - tol: best_loss = epoch_loss; patience_counter = 0
        else: patience_counter += 1
        if patience_counter >= patience and epoch > warmup_epochs: 
            converged_epoch = epoch + 1; break
    model.eval(); 
    with torch.no_grad():
        final_shared, final_recons, u_final, final_private = model([m.to(device) for m in torch_mats])
        v_mats = [torch.nan_to_num(enc.v.detach().cpu(), nan=0.0, posinf=0.0, neginf=0.0) for enc in model.linear_encoders]
    return {"model": model.cpu(), "model_type": "ned_shared_private", "u": torch.nan_to_num(u_final.cpu(), nan=0.0, posinf=0.0, neginf=0.0), "v": v_mats, "latents": [torch.nan_to_num(l.cpu(), nan=0.0, posinf=0.0, neginf=0.0) for l in final_shared], "private_latents": [torch.nan_to_num(p.cpu(), nan=0.0, posinf=0.0, neginf=0.0) for p in final_private], "loss_history": loss_history, "recon_history": recon_history, "sim_history": sim_history, "converged_iter": converged_epoch, "scale_list": ["centerAndScale"], "provenance_list": provenance_list}

def deep_simr(data_matrices: List[Union[torch.Tensor, np.ndarray]], k: int, epochs: int = 150, batch_size: int = 64, learning_rate: float = 5e-4, sim_weight: float = 1.0, warmup_epochs: int = 20, energy_type: str = "regression", device: Optional[str] = None, verbose: bool = False) -> Dict[str, Any]:
    return lend_simr(data_matrices, k, epochs, batch_size, learning_rate, sim_weight=sim_weight, warmup_epochs=warmup_epochs, energy_type=energy_type, device=device, verbose=verbose)

def predict_deep(data_matrices: List[Union[torch.Tensor, np.ndarray]], model_res: Dict[str, Any], device: Optional[str] = None) -> Dict[str, Any]:
    model = model_res["model"]; model_type = model_res["model_type"]
    if device is None: device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(device); model.to(device).eval()
    torch_mats = [preprocess_data(torch.as_tensor(m).float(), model_res["scale_list"], prov) for m, prov in zip(data_matrices, model_res["provenance_list"])]
    with torch.no_grad():
        res = model([m.to(device) for m in torch_mats])
        if model_type == "ned_shared_private":
            shared_l, recons, u, private_l = res
            return {"u": u.cpu(), "latents": [l.cpu() for l in shared_l], "reconstructions": [r.cpu() for r in recons], "private_latents": [p.cpu() for p in private_l]}
        else:
            latents, recons, u = res
            return {"u": u.cpu(), "latents": [l.cpu() for l in latents], "reconstructions": [r.cpu() for r in recons]}
