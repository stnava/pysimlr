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
    """
    A flexible MLP encoder for a single modality.
    """
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

class LinearModalityEncoder(nn.Module):
    """
    A strict linear encoder (V matrix) for LEND SiMR.
    """
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.v = nn.Parameter(torch.randn(input_dim, latent_dim) * 0.01)

    def forward(self, x):
        return x @ self.v

class ModalityDecoder(nn.Module):
    """
    A flexible MLP decoder for a single modality.
    """
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
    """
    Multi-view Autoencoder for Deep SiMR.
    """
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
    """
    Linear Encode, Non-linear Decoder for SiMR.
    """
    def __init__(self, input_dims: List[int], latent_dim: int, hidden_dims: List[int] = [64, 128], dropout: float = 0.1):
        super().__init__()
        self.encoders = nn.ModuleList([
            LinearModalityEncoder(dim, latent_dim) for dim in input_dims
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
                self.encoders[i].v.copy_(v)

    def forward(self, x_list: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        latents = [enc(x) for enc, x in zip(self.encoders, x_list)]
        reconstructions = [dec(z) for dec, z in zip(self.decoders, latents)]
        return latents, reconstructions

def calculate_sim_loss(latents: List[torch.Tensor], u_shared: torch.Tensor, energy_type: str = "regression") -> torch.Tensor:
    """
    Calculates the similarity loss between modality latents (Z) and shared latent (U).
    """
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
            sim_loss -= torch.sum(torch.log(torch.cosh(s))) / n
        elif energy_type == "exp":
            s = u_shared.t() @ z
            sim_loss -= torch.sum(-torch.exp(-s**2 / 2.0)) / n
        elif energy_type == "gauss":
            s = u_shared.t() @ z
            sim_loss -= torch.sum(-0.5 * torch.exp(-s**2)) / n
        elif energy_type == "kurtosis":
            s = u_shared.t() @ z
            sim_loss -= torch.sum((s**4.0) / 4.0) / n
            
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
    """
    Deep SiMR: Non-linear Multi-modal Integration using PyTorch Autoencoders.
    """
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
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += total_loss.item()
            epoch_recon += recon_loss.item()
            epoch_sim += sim_loss.item()
            
        epoch_loss /= len(dataloader)
        epoch_recon /= len(dataloader)
        epoch_sim /= len(dataloader)
        
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
        "u": u_final.cpu(),
        "latents": [l.cpu() for l in final_latents],
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
              sparseness_quantile: float = 0.5,
              positivity: str = "either",
              nsa_omega: float = 0.5,
              hidden_dims: List[int] = [64, 128],
              dropout: float = 0.1,
              energy_type: str = "regression",
              device: Optional[str] = None,
              verbose: bool = False) -> Dict[str, Any]:
    """
    LEND SiMR: Linear Encode, Non-linear Decoder. Yields interpretable V matrices.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(device)
    
    torch_mats = [m if isinstance(m, torch.Tensor) else torch.from_numpy(m).float() for m in data_matrices]
    input_dims = [m.shape[1] for m in torch_mats]
    
    model = LENDSiMRModel(input_dims, k, hidden_dims, dropout=dropout)
    model.initialize_v(torch_mats, k)
    model = model.to(device)
    
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
            
            l1_loss = sum(torch.sum(torch.abs(enc.v)) for enc in model.encoders)
            
            total_loss = recon_loss + sim_weight * sim_loss + 1e-4 * l1_loss
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += total_loss.item()
            epoch_recon += recon_loss.item()
            epoch_sim += sim_loss.item()
            
        with torch.no_grad():
            for enc in model.encoders:
                v = enc.v.detach().cpu()
                if nsa is not None:
                    try:
                        v_retracted = nsa.nsa_flow_retract_auto(v, w_retract=nsa_omega, retraction_type="soft_polar")
                    except:
                        u_svd, _, v_svd = torch.linalg.svd(v, full_matrices=False)
                        v_retracted = u_svd @ v_svd
                else:
                    u_svd, _, v_svd = torch.linalg.svd(v, full_matrices=False)
                    v_retracted = u_svd @ v_svd
                    
                v_sparse = orthogonalize_and_q_sparsify(v_retracted, sparseness_quantile, positivity)
                enc.v.copy_(v_sparse.to(device))
        
        epoch_loss /= len(dataloader)
        epoch_recon /= len(dataloader)
        epoch_sim /= len(dataloader)
        
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
        v_mats = [enc.v.detach().cpu() for enc in model.encoders]
        
    return {
        "model": model.cpu(),
        "model_type": "lend_simr",
        "u": u_final.cpu(),
        "v": v_mats,
        "latents": [l.cpu() for l in final_latents],
        "loss_history": loss_history,
        "recon_history": recon_history,
        "sim_history": sim_history
    }
