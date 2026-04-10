import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from typing import List, Optional, Union, Dict, Any, Tuple
from .sparsification import orthogonalize_and_q_sparsify
from .svd import ba_svd
from .simlr import calculate_u
from .utils import preprocess_data

try:
    import nsa_flow as nsa
except ImportError:
    nsa = None

def _newton_step_ortho(u: torch.Tensor) -> torch.Tensor:
    """Algebraic Newton-style projection towards the Stiefel manifold."""
    norm = torch.norm(u, p=2) + 1e-8
    u_scaled = u / norm
    return 1.5 * u_scaled - 0.5 * u_scaled @ (u_scaled.t() @ u_scaled)

def _normalize_rows(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Differentiable row normalization with stability epsilon."""
    x_clamped = torch.clamp(x, min=-1e10, max=1e10)
    norm = x_clamped.norm(p=2, dim=1, keepdim=True)
    return x_clamped / (norm + eps)

def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
    res = x.clone()
    res.diagonal().zero_()
    return res

def _variance_penalty(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """VICReg style variance penalty."""
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    return torch.mean(torch.nn.functional.relu(gamma - std))

def _covariance_penalty(z: torch.Tensor) -> torch.Tensor:
    """VICReg style covariance penalty."""
    n = z.shape[0]
    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.T @ zc) / max(n - 1, 1)
    return _off_diagonal(cov).pow(2).sum() / z.shape[1]

def _cross_covariance_penalty(shared: torch.Tensor, private: torch.Tensor) -> torch.Tensor:
    """Penalize linear dependence between shared and private branches."""
    n = shared.shape[0]
    if n <= 1: return torch.tensor(0.0, device=shared.device)
    s_c = shared - shared.mean(dim=0, keepdim=True)
    p_c = private - private.mean(dim=0, keepdim=True)
    cross_cov = (s_c.T @ p_c) / (n - 1)
    return cross_cov.pow(2).mean()




def _standardize_deep(data_matrices, scale_list=["centerAndScale"]):
    torch_mats = [torch.as_tensor(m).float() for m in data_matrices]
    provenance_list = []
    scaled_mats = []
    for m in torch_mats:
        m_scaled, prov = preprocess_data(m, scale_list)
        scaled_mats.append(m_scaled)
        provenance_list.append(prov)
    return scaled_mats, provenance_list

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

class LENDNSAEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, nsa_w: float = 0.5, 
                 positivity: str = "either", sparseness_quantile: float = 0.0):
        super().__init__()
        self.v_raw = nn.Parameter(torch.randn(input_dim, latent_dim) * 0.01)
        self.positivity, self.sparseness_quantile = positivity, sparseness_quantile
        apply_nonneg = 'hard' if positivity in ['positive', 'hard'] else 'none'
        if nsa is not None:
            self.nsa_layer = nsa.NSAFlowLayer(k=latent_dim, w_retract=nsa_w, retraction_type="polar", apply_nonneg=apply_nonneg, residual=False, use_transform=False)
        else: self.nsa_layer = None
    @property
    def v(self):
        if self.nsa_layer is not None: v_out = self.nsa_layer(self.v_raw)
        else:
            try:
                u_svd, _, v_svd = torch.linalg.svd(self.v_raw, full_matrices=False)
                v_out = u_svd @ v_svd
            except: v_out = torch.nn.functional.normalize(self.v_raw, p=2, dim=0)
            if self.positivity in ['positive', 'hard']: v_out = torch.clamp(v_out, min=0.0)
        v_out = torch.nan_to_num(v_out, nan=0.0, posinf=0.0, neginf=0.0)
        if self.sparseness_quantile > 0:
            v_sparse = v_out.clone()
            with torch.no_grad():
                for col in range(v_sparse.shape[1]):
                    col_vals = v_sparse[:, col]
                    q_val = torch.quantile(torch.abs(col_vals) if self.positivity=="either" else col_vals, self.sparseness_quantile)
                    mask = (torch.abs(col_vals) < q_val) if self.positivity=="either" else (col_vals <= q_val)
                    v_sparse[mask, col] = 0.0
            return v_sparse
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
        self.nonlinear_heads = nn.ModuleList([ModalityEncoder(latent_dim, latent_dim, [h//2 for h in hidden_dims], dropout) for _ in input_dims])
        self.decoders = nn.ModuleList([ModalityDecoder(latent_dim, dim, hidden_dims[::-1], dropout) for dim in input_dims])
        
        self.mixing_algorithm, self.latent_dim = mixing_algorithm, latent_dim
    def initialize_v(self, data_matrices: List[torch.Tensor], k: int):
        with torch.no_grad():
            for i, x in enumerate(data_matrices):
                u, s, v = ba_svd(x, nu=0, nv=k)
                if v.shape[1] < k: v = torch.cat([v, torch.randn(v.shape[0], k-v.shape[1], device=v.device)*1e-4], dim=1)
                self.linear_encoders[i].v_raw.copy_(v.to(x.dtype))
    def forward(self, x_list: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        latents = [nl_head(l_enc(x)) for l_enc, nl_head, x in zip(self.linear_encoders, self.nonlinear_heads, x_list)]
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
        self.shared_heads = nn.ModuleList([ModalityEncoder(shared_latent_dim, shared_latent_dim, hidden_dims, dropout) for _ in input_dims])
        self.private_heads = nn.ModuleList([ModalityEncoder(shared_latent_dim, private_latent_dim, hidden_dims, dropout) for _ in input_dims])
        self.decoders = nn.ModuleList([ModalityDecoder(shared_latent_dim + private_latent_dim, dim, hidden_dims[::-1], dropout) for dim in input_dims])
        
        self.mixing_algorithm, self.shared_latent_dim = mixing_algorithm, shared_latent_dim
    def initialize_v(self, data_matrices: List[torch.Tensor], k: int):
        with torch.no_grad():
            for i, x in enumerate(data_matrices):
                u, s, v = ba_svd(x, nu=0, nv=k)
                if v.shape[1] < k: v = torch.cat([v, torch.randn(v.shape[0], k-v.shape[1], device=v.device)*1e-4], dim=1)
                self.linear_encoders[i].v_raw.copy_(v.to(x.dtype))
    def forward(self, x_list: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        shared_latents = [sh(l_enc(x)) for l_enc, sh, x in zip(self.linear_encoders, self.shared_heads, x_list)]
        private_latents = [pr(l_enc(x)) for l_enc, pr, x in zip(self.linear_encoders, self.private_heads, x_list)]
        if self.training:
            u_shared = torch.mean(torch.stack(shared_latents), dim=0)
            if self.mixing_algorithm == "newton": u_shared = _newton_step_ortho(u_shared)
        else:
            with torch.no_grad(): u_shared = calculate_u(shared_latents, mixing_algorithm=self.mixing_algorithm, k=self.shared_latent_dim)
        
        return shared_latents, private_latents, [dec(torch.cat([u_shared, pr], dim=1)) for dec, pr in zip(self.decoders, private_latents)], u_shared

class DeepSiMRModel(nn.Module):
    def __init__(self, input_dims: List[int], latent_dim: int, hidden_dims: List[int] = [128, 64], 
                 dropout: float = 0.1, mixing_algorithm: str = "avg"):
        super().__init__()
        self.encoders = nn.ModuleList([ModalityEncoder(dim, latent_dim, hidden_dims, dropout) for dim in input_dims])
        self.decoders = nn.ModuleList([ModalityDecoder(latent_dim, dim, hidden_dims[::-1], dropout) for dim in input_dims])
        
        self.mixing_algorithm, self.latent_dim = mixing_algorithm, latent_dim
    def forward(self, x_list: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        latents = [enc(x) for enc, x in zip(self.encoders, x_list)]
        if self.training:
            u_shared = torch.mean(torch.stack(latents), dim=0)
            if self.mixing_algorithm == "newton": u_shared = _newton_step_ortho(u_shared)
        else:
            with torch.no_grad(): u_shared = calculate_u(latents, mixing_algorithm=self.mixing_algorithm, k=self.latent_dim)
        
        return latents, [dec(u_shared) for dec in self.decoders], u_shared

def calculate_sim_loss(latents: List[torch.Tensor], u_shared: torch.Tensor, energy_type: str = "regression") -> torch.Tensor:
    sim_loss = torch.tensor(0.0, device=u_shared.device)
    n, u_target = u_shared.shape[0], u_shared.detach()
    
    # VICReg variance penalty to prevent dimensional collapse
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
    # Penalize redundancy (off-diagonals)
    collapse_loss = torch.sum((cov_u - torch.diag(torch.diag(cov_u)))**2) / u_shared.shape[1]
    
    return sim_loss + 0.1 * var_penalty + 0.1 * collapse_loss

def _train_loop(model, dataloader, optimizer, scheduler, mse_loss, epochs, sim_weight, energy_type, warmup_epochs, verbose, device, tol=1e-6, patience=10):
    loss_history, recon_history, sim_history = [], [], []
    best_loss = float('inf')
    patience_counter = 0
    converged_epoch = epochs

    for epoch in range(epochs):
        model.train(); epoch_loss, epoch_recon, epoch_sim = 0.0, 0.0, 0.0
        current_sim_weight = 0.0 if epoch < warmup_epochs else sim_weight
        for batch in dataloader:
            batch_mats = [b.to(device) for b in batch]; optimizer.zero_grad()
            res = model(batch_mats)
            shared_latents, reconstructions, u_shared = res[0], res[-2], res[-1]
            recon_loss = sum(mse_loss(r, x) for r, x in zip(reconstructions, batch_mats))
            sim_loss = calculate_sim_loss(shared_latents, u_shared, energy_type)
            l1_loss = 0.0
            if hasattr(model, 'linear_encoders'):
                for enc in model.linear_encoders:
                    if hasattr(enc, 'v_raw'): l1_loss += torch.sum(torch.abs(enc.v_raw))
            elif hasattr(model, 'encoders'):
                for enc in model.encoders:
                    if hasattr(enc, 'v_raw'): l1_loss += torch.sum(torch.abs(enc.v_raw))
            total_loss = recon_loss + current_sim_weight * sim_loss + 1e-6 * l1_loss
            if torch.isnan(total_loss): continue
            total_loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0); optimizer.step()
            epoch_loss += total_loss.item(); epoch_recon += recon_loss.item(); epoch_sim += sim_loss.item()
        
        epoch_loss /= len(dataloader); epoch_recon /= len(dataloader); epoch_sim /= len(dataloader)
        loss_history.append(epoch_loss); recon_history.append(epoch_recon); sim_history.append(epoch_sim)
        scheduler.step()
        
        # Check for convergence
        if epoch_loss < best_loss - tol:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience and epoch > warmup_epochs:
            if verbose: print(f"Converged at epoch {epoch}: Total Loss {epoch_loss:.4f}")
            converged_epoch = epoch + 1
            break

        if verbose and epoch % 10 == 0: print(f"Epoch {epoch}: Total={epoch_loss:.4f} (Recon={epoch_recon:.4f}, Sim={epoch_sim:.4f})")
        
    return loss_history, recon_history, sim_history, converged_epoch

def lend_simr(data_matrices: List[Union[torch.Tensor, np.ndarray]], k: int, epochs: int = 150, batch_size: int = 64, learning_rate: float = 5e-4, weight_decay: float = 1e-4, sim_weight: float = 1.0, warmup_epochs: int = 20, hidden_dims: List[int] = [64, 128], dropout: float = 0.1, sparseness_quantile: float = 0.0, positivity: str = "either", nsa_w: float = 0.5, energy_type: str = "regression", mixing_algorithm: str = "newton", device: Optional[str] = None, verbose: bool = False) -> Dict[str, Any]:
    if device is None: device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(device); torch_mats, provenance_list = _standardize_deep(data_matrices, ["centerAndScale"]); input_dims = [m.shape[1] for m in torch_mats]
    model = LENDSiMRModel(input_dims, k, hidden_dims, dropout, nsa_w, positivity, sparseness_quantile, mixing_algorithm).to(device)
    model.initialize_v(torch_mats, k)
    optimizer = optim.Adam([{'params': model.encoders.parameters(), 'lr': learning_rate * 0.1}, {'params': model.decoders.parameters()}], lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs); mse_loss = nn.MSELoss(); dataset = TensorDataset(*torch_mats); dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_h, recon_h, sim_h, conv_ep = _train_loop(model, dataloader, optimizer, scheduler, mse_loss, epochs, sim_weight, energy_type, warmup_epochs, verbose, device)
    model.eval(); 
    with torch.no_grad():
        final_latents, _, u_final = model([m.to(device) for m in torch_mats])
        v_mats = [torch.nan_to_num(enc.v.detach().cpu()) for enc in model.encoders]
    return {"model": model.cpu(), "model_type": "lend_simr", "u": torch.nan_to_num(u_final.cpu()), "v": v_mats, "latents": [torch.nan_to_num(l.cpu()) for l in final_latents], "loss_history": loss_h, "recon_history": recon_h, "sim_history": sim_h, "converged_iter": conv_ep, "scale_list": ["centerAndScale"], "provenance_list": provenance_list}

def ned_simr(data_matrices: List[Union[torch.Tensor, np.ndarray]], k: int, epochs: int = 150, batch_size: int = 64, learning_rate: float = 5e-4, weight_decay: float = 1e-4, sim_weight: float = 1.0, warmup_epochs: int = 20, hidden_dims: List[int] = [128, 64], dropout: float = 0.1, sparseness_quantile: float = 0.0, positivity: str = "either", nsa_w: float = 0.5, energy_type: str = "regression", mixing_algorithm: str = "newton", device: Optional[str] = None, verbose: bool = False) -> Dict[str, Any]:
    if device is None: device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(device); torch_mats, provenance_list = _standardize_deep(data_matrices, ["centerAndScale"]); input_dims = [m.shape[1] for m in torch_mats]
    model = NEDSiMRModel(input_dims, k, hidden_dims, dropout, nsa_w, positivity, sparseness_quantile, mixing_algorithm).to(device)
    model.initialize_v(torch_mats, k)
    optimizer = optim.Adam([{'params': model.linear_encoders.parameters(), 'lr': learning_rate * 0.1}, {'params': model.nonlinear_heads.parameters()}, {'params': model.decoders.parameters()}], lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs); mse_loss = nn.MSELoss(); dataset = TensorDataset(*torch_mats); dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_h, recon_h, sim_h, conv_ep = _train_loop(model, dataloader, optimizer, scheduler, mse_loss, epochs, sim_weight, energy_type, warmup_epochs, verbose, device)
    model.eval(); 
    with torch.no_grad():
        final_latents, _, u_final = model([m.to(device) for m in torch_mats])
        v_mats = [torch.nan_to_num(enc.v.detach().cpu()) for enc in model.linear_encoders]
    return {"model": model.cpu(), "model_type": "ned_simr", "u": torch.nan_to_num(u_final.cpu()), "v": v_mats, "latents": [torch.nan_to_num(l.cpu()) for l in final_latents], "loss_history": loss_h, "recon_history": recon_h, "sim_history": sim_h, "converged_iter": conv_ep, "scale_list": ["centerAndScale"], "provenance_list": provenance_list}

def ned_simr_shared_private(data_matrices: List[Union[torch.Tensor, np.ndarray]], k: int, private_k: Optional[int] = None, epochs: int = 150, batch_size: int = 64, learning_rate: float = 5e-4, weight_decay: float = 1e-4, sim_weight: float = 1.0, warmup_epochs: int = 20, sparseness_quantile: float = 0.0, positivity: str = "either", nsa_w: float = 0.5, hidden_dims: List[int] = [128, 64], dropout: float = 0.1, energy_type: str = "regression", mixing_algorithm: str = "newton", private_recon_weight: float = 1.0, private_orthogonality_weight: float = 0.05, private_variance_weight: float = 0.10, device: Optional[str] = None, verbose: bool = False) -> Dict[str, Any]:
    if private_k is None: private_k = max(1, k // 2)
    if device is None: device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(device); torch_mats, provenance_list = _standardize_deep(data_matrices, ["centerAndScale"]); input_dims = [m.shape[1] for m in torch_mats]
    model = NEDSharedPrivateSiMRModel(input_dims, k, private_k, hidden_dims, dropout, nsa_w, positivity, sparseness_quantile, mixing_algorithm).to(device)
    model.initialize_v(torch_mats, k)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs); mse_loss = nn.MSELoss(); dataset = TensorDataset(*torch_mats); dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_history, recon_history, sim_history, priv_history, orth_history = [], [], [], [], []
    best_loss = float('inf')
    patience_counter = 0
    converged_epoch = epochs
    for epoch in range(epochs):
        model.train(); epoch_loss, epoch_recon, epoch_sim, epoch_priv, epoch_orth = 0.0, 0.0, 0.0, 0.0, 0.0
        current_sim_weight = 0.0 if epoch < warmup_epochs else sim_weight
        for batch in dataloader:
            batch_mats = [b.to(device) for b in batch]; optimizer.zero_grad()
            shared_l, priv_l, recons, u_sh = model(batch_mats)
            recon_loss = sum(mse_loss(r, x) for r, x in zip(recons, batch_mats))
            sim_loss = calculate_sim_loss(shared_l, u_sh, energy_type)
            l1_loss = sum(torch.sum(torch.abs(enc.v_raw)) for enc in model.linear_encoders)
            priv_var_loss = sum(_variance_penalty(p) for p in priv_l)
            priv_orth_loss = sum(_cross_covariance_penalty(shared_l[i], priv_l[i]) for i in range(len(shared_l)))
            total_loss = private_recon_weight * recon_loss + current_sim_weight * sim_loss + private_variance_weight * priv_var_loss + private_orthogonality_weight * priv_orth_loss + 1e-6 * l1_loss
            if torch.isnan(total_loss): continue
            total_loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0); optimizer.step()
            epoch_loss += total_loss.item(); epoch_recon += recon_loss.item(); epoch_sim += sim_loss.item(); epoch_priv += priv_var_loss.item(); epoch_orth += priv_orth_loss.item()
        epoch_loss /= len(dataloader); epoch_recon /= len(dataloader); epoch_sim /= len(dataloader); epoch_priv /= len(dataloader); epoch_orth /= len(dataloader)
        loss_history.append(epoch_loss); recon_history.append(epoch_recon); sim_history.append(epoch_sim); priv_history.append(epoch_priv); orth_history.append(epoch_orth)
        scheduler.step()
        if epoch_loss < best_loss - 1e-6:
            best_loss = epoch_loss; patience_counter = 0
        else: patience_counter += 1
        if patience_counter >= 15 and epoch > warmup_epochs:
            if verbose: print(f"Converged at epoch {epoch}: Total Loss {epoch_loss:.4f}"); 
            converged_epoch = epoch + 1; break
        if verbose and epoch % 10 == 0: print(f"Epoch {epoch}: Total={epoch_loss:.4f} (Recon={epoch_recon:.4f}, Sim={epoch_sim:.4f}, CrossCov={epoch_orth:.4f})")
    model.eval(); 
    with torch.no_grad():
        all_torch_mats = [m.to(device) for m in torch_mats]
        sh_l, pr_l, _, u_final = model(all_torch_mats)
        v_mats = [torch.nan_to_num(enc.v.detach().cpu()) for enc in model.linear_encoders]
    return {"model": model.cpu(), "model_type": "ned_simr_shared_private", "u": torch.nan_to_num(u_final.cpu()), "v": v_mats, "shared_latents": [torch.nan_to_num(l.cpu()) for l in sh_l], "private_latents": [torch.nan_to_num(l.cpu()) for l in pr_l], "latents": [torch.nan_to_num(l.cpu()) for l in sh_l], "loss_history": loss_history, "recon_history": recon_history, "sim_history": sim_history, "private_variance_history": priv_history, "private_orthogonality_history": orth_history, "converged_iter": converged_epoch, "scale_list": ["centerAndScale"], "provenance_list": provenance_list}

def deep_simr(data_matrices: List[Union[torch.Tensor, np.ndarray]], k: int, epochs: int = 100, batch_size: int = 32, learning_rate: float = 1e-3, weight_decay: float = 1e-5, sim_weight: float = 1.0, warmup_epochs: int = 10, hidden_dims: List[int] = [128, 64], dropout: float = 0.1, energy_type: str = "regression", mixing_algorithm: str = "avg", device: Optional[str] = None, verbose: bool = False) -> Dict[str, Any]:
    if device is None: device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(device); torch_mats, provenance_list = _standardize_deep(data_matrices, ["centerAndScale"]); input_dims = [m.shape[1] for m in torch_mats]
    model = DeepSiMRModel(input_dims, k, hidden_dims, dropout, mixing_algorithm).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs); mse_loss = nn.MSELoss(); dataset = TensorDataset(*torch_mats); dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_h, recon_h, sim_h, conv_ep = _train_loop(model, dataloader, optimizer, scheduler, mse_loss, epochs, sim_weight, energy_type, warmup_epochs, verbose, device)
    model.eval(); 
    with torch.no_grad():
        final_latents, _, u_final = model([m.to(device) for m in torch_mats])
    return {"model": model.cpu(), "model_type": "deep_simr", "u": torch.nan_to_num(u_final.cpu()), "latents": [torch.nan_to_num(l.cpu()) for l in final_latents], "loss_history": loss_h, "recon_history": recon_h, "sim_history": sim_h, "converged_iter": conv_ep, "scale_list": ["centerAndScale"], "provenance_list": provenance_list}
