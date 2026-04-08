import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from typing import List, Optional, Union, Dict, Any, Tuple
from .sparsification import orthogonalize_and_q_sparsify
from .svd import ba_svd

try:
    import nsa_flow as nsa
except ImportError:
    nsa = None

class ModalityEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: List[int] = [128, 64], dropout: float = 0.1):
        super().__init__()
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, latent_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class LENDNSAEncoder(nn.Module):
    """
    A linear encoder optimized for LEND SiMR that delegates orthogonality 
    and positivity constraints to the NSAFlowLayer.
    """
    def __init__(self, input_dim: int, latent_dim: int, nsa_omega: float = 0.5, 
                 positivity: str = "either", sparseness_quantile: float = 0.0):
        super().__init__()
        self.v_raw = nn.Parameter(torch.randn(input_dim, latent_dim) * 0.01)
        self.positivity = positivity
        self.sparseness_quantile = sparseness_quantile
        
        apply_nonneg = 'none'
        if positivity == 'positive':
            apply_nonneg = 'hard'
            
        if nsa is not None:
            self.nsa_layer = nsa.NSAFlowLayer(
                k=latent_dim, 
                w_retract=nsa_omega, 
                retraction_type="soft_polar", 
                apply_nonneg=apply_nonneg,
                residual=False, # We want strict linear V, no extra MLP transform
                use_transform=False
            )
        else:
            self.nsa_layer = None

    @property
    def v(self):
        """
        Returns the orthogonal, positive, and sparsified projection matrix V.
        This allows backprop through the manifold geometry natively.
        """
        if self.nsa_layer is not None:
            v_out = self.nsa_layer(self.v_raw)
        else:
            # SVD fallback
            u_svd, _, v_svd = torch.linalg.svd(self.v_raw, full_matrices=False)
            v_out = u_svd @ v_svd
            if self.positivity == 'positive':
                v_out = torch.clamp(v_out, min=0.0)
                
        # Apply quantile sparsification (differentiably using mask, or just zero out)
        if self.sparseness_quantile > 0:
            v_sparse = v_out.clone()
            for col in range(v_sparse.shape[1]):
                col_vals = v_sparse[:, col]
                if self.positivity == "either":
                    abs_vals = torch.abs(col_vals)
                    q_val = torch.quantile(abs_vals, self.sparseness_quantile)
                    mask = abs_vals < q_val
                else:
                    q_val = torch.quantile(col_vals, self.sparseness_quantile)
                    if self.positivity == "positive":
                        mask = col_vals <= q_val
                    elif self.positivity == "negative":
                        mask = col_vals >= q_val
                    else:
                        mask = col_vals < q_val
                # Zero out elements below quantile without losing gradient flow for the non-zeroed elements
                v_sparse[:, col] = torch.where(mask, torch.zeros_like(col_vals), col_vals)
            return v_sparse
            
        return v_out

    def forward(self, x):
        return x @ self.v

class ModalityDecoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int, hidden_dims: List[int] = [64, 128], dropout: float = 0.1):
        super().__init__()
        layers = []
        curr_dim = latent_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, z):
        return self.network(z)

class DeepSiMRModel(nn.Module):
    def __init__(self, input_dims: List[int], latent_dim: int, hidden_dims: List[int] = [128, 64], dropout: float = 0.1):
        super().__init__()
        self.encoders = nn.ModuleList([
            ModalityEncoder(dim, latent_dim, hidden_dims, dropout) for dim in input_dims
        ])
        self.decoders = nn.ModuleList([
            ModalityDecoder(latent_dim, dim, hidden_dims[::-1], dropout) for dim in input_dims
        ])

    def forward(self, x_list: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        latents = [enc(x) for enc, x in zip(self.encoders, x_list)]
        reconstructions = [dec(z) for dec, z in zip(self.decoders, latents)]
        return latents, reconstructions

class LENDSiMRModel(nn.Module):
    def __init__(self, input_dims: List[int], latent_dim: int, hidden_dims: List[int] = [64, 128], 
                 dropout: float = 0.1, nsa_omega: float = 0.5, positivity: str = "either", sparseness_quantile: float = 0.0):
        super().__init__()
        self.encoders = nn.ModuleList([
            LENDNSAEncoder(dim, latent_dim, nsa_omega, positivity, sparseness_quantile) for dim in input_dims
        ])
        self.decoders = nn.ModuleList([
            ModalityDecoder(latent_dim, dim, hidden_dims, dropout) for dim in input_dims
        ])
        
    def initialize_v(self, data_matrices: List[torch.Tensor], k: int):
        with torch.no_grad():
            for i, x in enumerate(data_matrices):
                u, s, v = ba_svd(x, nu=0, nv=k)
                if v.shape[1] < k:
                    padding = torch.randn(v.shape[0], k - v.shape[1]) * 1e-4
                    v = torch.cat([v, padding], dim=1)
                self.encoders[i].v_raw.copy_(v)

    def forward(self, x_list: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        latents = [enc(x) for enc, x in zip(self.encoders, x_list)]
        reconstructions = [dec(z) for dec, z in zip(self.decoders, latents)]
        return latents, reconstructions

def calculate_sim_loss(latents: List[torch.Tensor], u_shared: torch.Tensor, energy_type: str = "regression") -> torch.Tensor:
    sim_loss = torch.tensor(0.0, device=u_shared.device)
    n = u_shared.shape[0]
    
    for z in latents:
        if energy_type == "regression":
            sim_loss += torch.sum((z - u_shared)**2) / n
        elif energy_type == "acc":
            z_centered = z - torch.mean(z, dim=0)
            u_centered = u_shared - torch.mean(u_shared, dim=0)
            cov = (u_centered.t() @ z_centered) / (n - 1)
            sim_loss -= torch.sum(torch.abs(cov))
        elif energy_type == "logcosh":
            s = u_shared.t() @ z
            abs_s = torch.abs(s)
            stable_logcosh = abs_s - np.log(2.0) + torch.log1p(torch.exp(-2.0 * abs_s))
            sim_loss -= torch.sum(stable_logcosh) / n
        elif energy_type == "exp":
            s = u_shared.t() @ z
            sim_loss -= torch.sum(-torch.exp(-s**2 / 2.0)) / n
        elif energy_type == "gauss":
            s = u_shared.t() @ z
            sim_loss -= torch.sum(-0.5 * torch.exp(-s**2)) / n
        elif energy_type == "kurtosis":
            s = u_shared.t() @ z
            s_clamped = torch.clamp(s, min=-20.0, max=20.0)
            sim_loss -= torch.sum((s_clamped**4.0) / 4.0) / n
            
    return sim_loss

def deep_simr(data_matrices: List[Union[torch.Tensor, np.ndarray]],
              k: int,
              epochs: int = 100,
              batch_size: int = 32,
              learning_rate: float = 1e-3,
              weight_decay: float = 1e-5,
              sim_weight: float = 0.1,
              hidden_dims: List[int] = [128, 64],
              dropout: float = 0.1,
              energy_type: str = "regression",
              device: Optional[str] = None,
              verbose: bool = False) -> Dict[str, Any]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(device)
    
    torch_mats = [m if isinstance(m, torch.Tensor) else torch.from_numpy(m).float() for m in data_matrices]
    input_dims = [m.shape[1] for m in torch_mats]
    
    model = DeepSiMRModel(input_dims, k, hidden_dims, dropout=dropout).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    mse_loss = nn.MSELoss()
    
    dataset = TensorDataset(*torch_mats)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    loss_history = []
    recon_history = []
    sim_history = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_sim = 0.0
        
        for batch in dataloader:
            batch_mats = [b.to(device) for b in batch]
            optimizer.zero_grad()
            
            latents, reconstructions = model(batch_mats)
            
            recon_loss = sum(mse_loss(r, x) for r, x in zip(reconstructions, batch_mats))
            u_shared = torch.mean(torch.stack(latents), dim=0)
            
            sim_loss = calculate_sim_loss(latents, u_shared, energy_type)
            
            total_loss = recon_loss + sim_weight * sim_loss
            
            if torch.isnan(total_loss):
                continue
                
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += total_loss.item()
            epoch_recon += recon_loss.item()
            epoch_sim += sim_loss.item()
            
        epoch_loss /= max(1, len(dataloader))
        epoch_recon /= max(1, len(dataloader))
        epoch_sim /= max(1, len(dataloader))
        
        loss_history.append(epoch_loss)
        recon_history.append(epoch_recon)
        sim_history.append(epoch_sim)
        
        scheduler.step()
        
        if verbose and epoch % max(1, epochs // 10) == 0:
            print(f"Epoch {epoch}: Total={epoch_loss:.4f} (Recon={epoch_recon:.4f}, Sim={epoch_sim:.4f})")
            
    model.eval()
    with torch.no_grad():
        all_torch_mats = [m.to(device) for m in torch_mats]
        final_latents, _ = model(all_torch_mats)
        u_final = torch.mean(torch.stack(final_latents), dim=0)
        
    return {
        "model": model.cpu(),
        "model_type": "deep_simr",
        "u": torch.nan_to_num(u_final.cpu()),
        "latents": [torch.nan_to_num(l.cpu()) for l in final_latents],
        "loss_history": loss_history,
        "recon_history": recon_history,
        "sim_history": sim_history
    }

def lend_simr(data_matrices: List[Union[torch.Tensor, np.ndarray]],
              k: int,
              epochs: int = 100,
              batch_size: int = 32,
              learning_rate: float = 1e-3,
              weight_decay: float = 1e-5,
              sim_weight: float = 0.1,
              sparseness_quantile: float = 0.0,
              positivity: str = "either",
              nsa_omega: float = 0.5,
              hidden_dims: List[int] = [64, 128],
              dropout: float = 0.1,
              energy_type: str = "regression",
              device: Optional[str] = None,
              verbose: bool = False) -> Dict[str, Any]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(device)
    
    torch_mats = [m if isinstance(m, torch.Tensor) else torch.from_numpy(m).float() for m in data_matrices]
    input_dims = [m.shape[1] for m in torch_mats]
    
    model = LENDSiMRModel(input_dims, k, hidden_dims, dropout=dropout, 
                          nsa_omega=nsa_omega, positivity=positivity, sparseness_quantile=sparseness_quantile)
    model.initialize_v(torch_mats, k)
    model = model.to(device)
    
    # We can now use AdamW for the entire model because NSAFlowLayer handles geometry natively in the forward pass!
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    mse_loss = nn.MSELoss()
    dataset = TensorDataset(*torch_mats)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    loss_history = []
    recon_history = []
    sim_history = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_sim = 0.0
        
        for batch in dataloader:
            batch_mats = [b.to(device) for b in batch]
            optimizer.zero_grad()
            
            # Forward pass inherently respects the Stiefel manifold and positivity constraints
            latents, reconstructions = model(batch_mats)
            
            recon_loss = sum(mse_loss(r, x) for r, x in zip(reconstructions, batch_mats))
            u_shared = torch.mean(torch.stack(latents), dim=0)
            sim_loss = calculate_sim_loss(latents, u_shared, energy_type)
            
            # Optional: L1 on the raw weights to encourage sparsity alongside the hard quantile threshold
            l1_loss = sum(torch.sum(torch.abs(enc.v_raw)) for enc in model.encoders)
            
            total_loss = recon_loss + sim_weight * sim_loss + 1e-4 * l1_loss
            
            if torch.isnan(total_loss):
                continue
                
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += total_loss.item()
            epoch_recon += recon_loss.item()
            epoch_sim += sim_loss.item()
            
        epoch_loss /= max(1, len(dataloader))
        epoch_recon /= max(1, len(dataloader))
        epoch_sim /= max(1, len(dataloader))
        
        loss_history.append(epoch_loss)
        recon_history.append(epoch_recon)
        sim_history.append(epoch_sim)
        
        scheduler.step()
        
        if verbose and epoch % max(1, epochs // 10) == 0:
            print(f"Epoch {epoch}: Total={epoch_loss:.4f} (Recon={epoch_recon:.4f}, Sim={epoch_sim:.4f})")
            
    model.eval()
    with torch.no_grad():
        all_torch_mats = [m.to(device) for m in torch_mats]
        final_latents, _ = model(all_torch_mats)
        u_final = torch.mean(torch.stack(final_latents), dim=0)
        # Extract the processed V matrices (not just the raw weights)
        v_mats = [torch.nan_to_num(enc.v.detach().cpu()) for enc in model.encoders]
        
    return {
        "model": model.cpu(),
        "model_type": "lend_simr",
        "u": torch.nan_to_num(u_final.cpu()),
        "v": v_mats,
        "latents": [torch.nan_to_num(l.cpu()) for l in final_latents],
        "loss_history": loss_history,
        "recon_history": recon_history,
        "sim_history": sim_history
    }
