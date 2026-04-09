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

try:
    import nsa_flow as nsa
except ImportError:
    nsa = None


def _normalize_rows(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Differentiable row normalization with stability epsilon.
    """
    x_clamped = torch.clamp(x, min=-1e10, max=1e10)
    norm = x_clamped.norm(p=2, dim=1, keepdim=True)
    return x_clamped / (norm + eps)


def _differentiable_consensus(latents: List[torch.Tensor], eps: float = 1e-8) -> torch.Tensor:
    """
    Differentiable mean of normalized latents.
    """
    latents_n = [_normalize_rows(z, eps=eps) for z in latents]
    u = torch.mean(torch.stack(latents_n), dim=0)
    return _normalize_rows(u, eps=eps)


def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    if n != m:
        raise ValueError("off_diagonal expects a square matrix")
    res = x.clone()
    res.diagonal().zero_()
    return res


def _variance_penalty(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    return torch.mean(torch.nn.functional.relu(gamma - std))


def _covariance_penalty(z: torch.Tensor) -> torch.Tensor:
    n = z.shape[0]
    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.T @ zc) / max(n - 1, 1)
    return _off_diagonal(cov).pow(2).sum() / z.shape[1]


def _standardize_modalities(data_matrices: List[Union[torch.Tensor, np.ndarray]]) -> List[torch.Tensor]:
    standardized = []
    for matrix in data_matrices:
        tensor = torch.as_tensor(matrix).float()
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        mean = tensor.mean(dim=0, keepdim=True)
        std = tensor.std(dim=0, keepdim=True, unbiased=False)
        tensor = (tensor - mean) / (std + 1e-6)
        standardized.append(tensor)
    return standardized


class ModalityEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: List[int] = [128, 64], dropout: float = 0.1):
        super().__init__()
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, latent_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class LENDNSAEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, nsa_w: float = 0.5,
                 positivity: str = "either", sparseness_quantile: float = 0.0):
        super().__init__()
        self.v_raw = nn.Parameter(torch.randn(input_dim, latent_dim) * 0.01)
        self.positivity = positivity
        self.sparseness_quantile = sparseness_quantile

        apply_nonneg = 'none'
        if positivity in ['positive', 'hard']:
            apply_nonneg = 'hard'

        if nsa is not None:
            self.nsa_layer = nsa.NSAFlowLayer(
                k=latent_dim,
                w_retract=nsa_w,
                retraction_type="polar",
                apply_nonneg=apply_nonneg,
                residual=False,
                use_transform=False
            )
        else:
            self.nsa_layer = None

    @property
    def v(self):
        if self.nsa_layer is not None:
            v_out = self.nsa_layer(self.v_raw)
        else:
            try:
                u_svd, _, v_svd = torch.linalg.svd(self.v_raw, full_matrices=False)
                v_out = u_svd @ v_svd
            except Exception:
                v_out = torch.nn.functional.normalize(self.v_raw, p=2, dim=0)

            if self.positivity in ['positive', 'hard']:
                v_out = torch.clamp(v_out, min=0.0)

        v_out = torch.nan_to_num(v_out, nan=0.0)

        if self.sparseness_quantile > 0:
            v_sparse = v_out.clone()
            with torch.no_grad():
                for col in range(v_sparse.shape[1]):
                    col_vals = v_sparse[:, col]
                    if self.positivity == "either":
                        abs_vals = torch.abs(col_vals)
                        q_val = torch.quantile(abs_vals, self.sparseness_quantile)
                        mask = abs_vals < q_val
                    else:
                        q_val = torch.quantile(col_vals, self.sparseness_quantile)
                        if self.positivity in ["positive", "hard"]:
                            mask = col_vals <= q_val
                        elif self.positivity == "negative":
                            mask = col_vals >= q_val
                        else:
                            mask = col_vals < q_val
                    v_sparse[mask, col] = 0.0
            return v_sparse

        return v_out

    def forward(self, x):
        return x @ self.v


class ModalityDecoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int, hidden_dims: List[int] = [64, 128], dropout: float = 0.1):
        super().__init__()
        if not hidden_dims:
            self.network = nn.Linear(latent_dim, output_dim)
        else:
            layers = []
            curr_dim = latent_dim
            for h_dim in hidden_dims:
                layers.append(nn.Linear(curr_dim, h_dim))
                layers.append(nn.LayerNorm(h_dim))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                curr_dim = h_dim
            layers.append(nn.Linear(curr_dim, output_dim))
            self.network = nn.Sequential(*layers)

    def forward(self, z):
        return self.network(z)


class LENDSiMRModel(nn.Module):
    def __init__(self, input_dims: List[int], latent_dim: int, hidden_dims: List[int] = [64, 128],
                 dropout: float = 0.1, nsa_w: float = 0.5, positivity: str = "either",
                 sparseness_quantile: float = 0.0, mixing_algorithm: str = "avg",
                 backprop_through_mixing: bool = True):
        super().__init__()
        self.encoders = nn.ModuleList([
            LENDNSAEncoder(dim, latent_dim, nsa_w, positivity, sparseness_quantile) for dim in input_dims
        ])
        self.decoders = nn.ModuleList([
            ModalityDecoder(latent_dim, dim, hidden_dims, dropout) for dim in input_dims
        ])
        self.mixing_algorithm = mixing_algorithm
        self.latent_dim = latent_dim
        self.backprop_through_mixing = backprop_through_mixing

    def initialize_v(self, data_matrices: List[torch.Tensor], k: int):
        with torch.no_grad():
            for i, x in enumerate(data_matrices):
                u, s, v = ba_svd(x, nu=0, nv=k)
                if v.shape[1] < k:
                    padding = torch.randn(v.shape[0], k - v.shape[1], device=v.device, dtype=v.dtype) * 1e-4
                    v = torch.cat([v, padding], dim=1)
                self.encoders[i].v_raw.copy_(v.to(x.dtype))

    def forward(self, x_list: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        latents = [enc(x) for enc, x in zip(self.encoders, x_list)]

        if self.training and self.backprop_through_mixing and self.mixing_algorithm in ["avg", "newton"]:
            u_shared = _differentiable_consensus(latents) if self.mixing_algorithm == "avg" else calculate_u(latents, mixing_algorithm="newton", k=self.latent_dim)
        else:
            with torch.no_grad():
                u_shared = calculate_u(latents, mixing_algorithm=self.mixing_algorithm, k=self.latent_dim)

        reconstructions = [dec(u_shared) for dec in self.decoders]
        return latents, reconstructions, u_shared


class NEDSiMRModel(nn.Module):
    """
    Non-linear Encoder Decoder SiMR.
    Architecture: Input -> Linear (V, constrained) -> Non-linear (MLP) -> Bottleneck -> Decoder.
    """
    def __init__(self, input_dims: List[int], latent_dim: int, hidden_dims: List[int] = [128, 64],
                 dropout: float = 0.1, nsa_w: float = 0.5, positivity: str = "either",
                 sparseness_quantile: float = 0.0, mixing_algorithm: str = "newton",
                 backprop_through_mixing: bool = True):
        super().__init__()
        # Each encoder is a sequence: Linear (V) -> Non-linear (MLP)
        self.linear_encoders = nn.ModuleList([
            LENDNSAEncoder(dim, latent_dim, nsa_w, positivity, sparseness_quantile) for dim in input_dims
        ])
        self.nonlinear_heads = nn.ModuleList([
            ModalityEncoder(latent_dim, latent_dim, hidden_dims, dropout) for _ in input_dims
        ])
        self.decoders = nn.ModuleList([
            ModalityDecoder(latent_dim, dim, hidden_dims[::-1], dropout) for dim in input_dims
        ])
        self.mixing_algorithm = mixing_algorithm
        self.latent_dim = latent_dim
        self.backprop_through_mixing = backprop_through_mixing

    def initialize_v(self, data_matrices: List[torch.Tensor], k: int):
        with torch.no_grad():
            for i, x in enumerate(data_matrices):
                u, s, v = ba_svd(x, nu=0, nv=k)
                if v.shape[1] < k:
                    padding = torch.randn(v.shape[0], k - v.shape[1], device=v.device, dtype=v.dtype) * 1e-4
                    v = torch.cat([v, padding], dim=1)
                self.linear_encoders[i].v_raw.copy_(v.to(x.dtype))

    def forward(self, x_list: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        # Input -> Linear V -> Non-linear Head Head
        latents = []
        for l_enc, nl_head, x in zip(self.linear_encoders, self.nonlinear_heads, x_list):
            h = l_enc(x)
            z = nl_head(h)
            latents.append(z)

        if self.training and self.backprop_through_mixing and self.mixing_algorithm in ["avg", "newton"]:
            u_shared = calculate_u(latents, mixing_algorithm=self.mixing_algorithm, k=self.latent_dim)
        else:
            with torch.no_grad():
                u_shared = calculate_u(latents, mixing_algorithm=self.mixing_algorithm, k=self.latent_dim)

        reconstructions = [dec(u_shared) for dec in self.decoders]
        return latents, reconstructions, u_shared


class DeepSiMRModel(nn.Module):
    def __init__(self, input_dims: List[int], latent_dim: int, hidden_dims: List[int] = [128, 64],
                 dropout: float = 0.1, mixing_algorithm: str = "avg", backprop_through_mixing: bool = True):
        super().__init__()
        self.encoders = nn.ModuleList([ModalityEncoder(dim, latent_dim, hidden_dims, dropout) for dim in input_dims])
        self.decoders = nn.ModuleList([ModalityDecoder(latent_dim, dim, hidden_dims[::-1], dropout) for dim in input_dims])
        self.mixing_algorithm = mixing_algorithm
        self.latent_dim = latent_dim
        self.backprop_through_mixing = backprop_through_mixing

    def forward(self, x_list: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        latents = [enc(x) for enc, x in zip(self.encoders, x_list)]
        if self.training and self.backprop_through_mixing and self.mixing_algorithm in ["avg", "newton"]:
            u_shared = calculate_u(latents, mixing_algorithm=self.mixing_algorithm, k=self.latent_dim)
        else:
            with torch.no_grad():
                u_shared = calculate_u(latents, mixing_algorithm=self.mixing_algorithm, k=self.latent_dim)
        reconstructions = [dec(u_shared) for dec in self.decoders]
        return latents, reconstructions, u_shared


def calculate_sim_loss(latents: List[torch.Tensor], u_shared: torch.Tensor, energy_type: str = "regression") -> torch.Tensor:
    sim_loss = torch.tensor(0.0, device=u_shared.device)
    n = u_shared.shape[0]
    u_target = u_shared.detach()

    for z in latents:
        if energy_type == "regression":
            z_n = _normalize_rows(z)
            u_n = _normalize_rows(u_target)
            align_loss = torch.mean((z_n - u_n) ** 2)
            var_loss = _variance_penalty(z)
            cov_loss = _covariance_penalty(z)
            sim_loss += align_loss + 0.1 * var_loss + 0.01 * cov_loss
        elif energy_type == "acc":
            z_centered = z - torch.mean(z, dim=0)
            u_centered = u_target - torch.mean(u_target, dim=0)
            cov = (u_centered.t() @ z_centered) / max(n - 1, 1)
            sim_loss -= torch.sum(torch.abs(cov))
        elif energy_type == "logcosh":
            s = u_target.t() @ z
            abs_s = torch.abs(s)
            stable_logcosh = abs_s - np.log(2.0) + torch.log1p(torch.exp(-2.0 * abs_s))
            sim_loss -= torch.sum(stable_logcosh) / max(n, 1)
    return sim_loss


def _train_loop(model, dataloader, optimizer, scheduler, mse_loss, epochs, sim_weight, energy_type, warmup_epochs, verbose, device):
    loss_history, recon_history, sim_history = [], [], []
    for epoch in range(epochs):
        model.train()
        epoch_loss, epoch_recon, epoch_sim = 0.0, 0.0, 0.0
        
        # Effective weight for similarity (warmup)
        current_sim_weight = sim_weight if epoch >= warmup_epochs else 0.0
        
        for batch in dataloader:
            batch_mats = [b.to(device) for b in batch]
            optimizer.zero_grad()
            latents, reconstructions, u_shared = model(batch_mats)
            recon_loss = sum(mse_loss(r, x) for r, x in zip(reconstructions, batch_mats))
            sim_loss = calculate_sim_loss(latents, u_shared, energy_type)
            
            # Additional sparsity penalty for NED/LEND linear weights
            l1_loss = 0.0
            if hasattr(model, 'linear_encoders'):
                l1_loss = sum(torch.sum(torch.abs(enc.v_raw)) for enc in model.linear_encoders)
            elif hasattr(model, 'encoders') and isinstance(model.encoders[0], LENDNSAEncoder):
                l1_loss = sum(torch.sum(torch.abs(enc.v_raw)) for enc in model.encoders)
                
            total_loss = recon_loss + current_sim_weight * sim_loss + 1e-4 * l1_loss
            if torch.isnan(total_loss):
                continue
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
    return loss_history, recon_history, sim_history


def deep_simr(data_matrices: List[Union[torch.Tensor, np.ndarray]],
              k: int,
              epochs: int = 100,
              batch_size: int = 32,
              learning_rate: float = 1e-4,
              weight_decay: float = 1e-5,
              sim_weight: float = 1.0,
              warmup_epochs: int = 10,
              hidden_dims: List[int] = [128, 64],
              dropout: float = 0.1,
              energy_type: str = "regression",
              mixing_algorithm: str = "newton",
              backprop_through_mixing: bool = True,
              device: Optional[str] = None,
              verbose: bool = False) -> Dict[str, Any]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(device)
    torch_mats = _standardize_modalities(data_matrices)
    input_dims = [m.shape[1] for m in torch_mats]
    model = DeepSiMRModel(input_dims, k, hidden_dims, dropout, mixing_algorithm, backprop_through_mixing).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    mse_loss = nn.MSELoss()
    dataset = TensorDataset(*torch_mats)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    loss_history, recon_history, sim_history = _train_loop(
        model, dataloader, optimizer, scheduler, mse_loss, epochs, sim_weight, energy_type, warmup_epochs, verbose, device
    )
    
    model.eval()
    with torch.no_grad():
        all_torch_mats = [m.to(device) for m in torch_mats]
        final_latents, _, u_final = model(all_torch_mats)
    return {
        "model": model.cpu(),
        "model_type": "deep_simr",
        "u": torch.nan_to_num(u_final.cpu()),
        "latents": [torch.nan_to_num(l.cpu()) for l in final_latents],
        "loss_history": loss_history,
        "recon_history": recon_history,
        "sim_history": sim_history,
    }


def lend_simr(data_matrices: List[Union[torch.Tensor, np.ndarray]],
              k: int,
              epochs: int = 100,
              batch_size: int = 32,
              learning_rate: float = 1e-4,
              weight_decay: float = 1e-5,
              sim_weight: float = 1.0,
              warmup_epochs: int = 10,
              sparseness_quantile: float = 0.0,
              positivity: str = "either",
              nsa_w: float = 0.5,
              hidden_dims: List[int] = [64, 128],
              dropout: float = 0.1,
              energy_type: str = "regression",
              mixing_algorithm: str = "newton",
              backprop_through_mixing: bool = True,
              device: Optional[str] = None,
              verbose: bool = False,
              **kwargs) -> Dict[str, Any]:
    if 'nsa_omega' in kwargs:
        nsa_w = kwargs.pop('nsa_omega')
    if device is None:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(device)
    torch_mats = _standardize_modalities(data_matrices)
    input_dims = [m.shape[1] for m in torch_mats]
    model = LENDSiMRModel(input_dims, k, hidden_dims, dropout, nsa_w, positivity, sparseness_quantile, mixing_algorithm, backprop_through_mixing)
    model.initialize_v(torch_mats, k)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    mse_loss = nn.MSELoss()
    dataset = TensorDataset(*torch_mats)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    loss_history, recon_history, sim_history = _train_loop(
        model, dataloader, optimizer, scheduler, mse_loss, epochs, sim_weight, energy_type, warmup_epochs, verbose, device
    )
    
    model.eval()
    with torch.no_grad():
        all_torch_mats = [m.to(device) for m in torch_mats]
        final_latents, _, u_final = model(all_torch_mats)
        v_mats = [torch.nan_to_num(enc.v.detach().cpu()) for enc in model.encoders]
    return {
        "model": model.cpu(),
        "model_type": "lend_simr",
        "u": torch.nan_to_num(u_final.cpu()),
        "v": v_mats,
        "latents": [torch.nan_to_num(l.cpu()) for l in final_latents],
        "loss_history": loss_history,
        "recon_history": recon_history,
        "sim_history": sim_history,
        "nsa_w": nsa_w,
        "sparseness_quantile": sparseness_quantile,
        "positivity": positivity,
    }


def ned_simr(data_matrices: List[Union[torch.Tensor, np.ndarray]],
              k: int,
              epochs: int = 100,
              batch_size: int = 32,
              learning_rate: float = 1e-4,
              weight_decay: float = 1e-5,
              sim_weight: float = 1.0,
              warmup_epochs: int = 10,
              sparseness_quantile: float = 0.0,
              positivity: str = "either",
              nsa_w: float = 0.5,
              hidden_dims: List[int] = [64, 128],
              dropout: float = 0.1,
              energy_type: str = "regression",
              mixing_algorithm: str = "newton",
              backprop_through_mixing: bool = True,
              device: Optional[str] = None,
              verbose: bool = False) -> Dict[str, Any]:
    """
    Non-linear Encoder Decoder SiMR.
    Retains interpretable V but adds non-linear refinement in the encoder.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(device)
    torch_mats = _standardize_modalities(data_matrices)
    input_dims = [m.shape[1] for m in torch_mats]
    
    model = NEDSiMRModel(input_dims, k, hidden_dims, dropout, nsa_w, positivity, 
                         sparseness_quantile, mixing_algorithm, backprop_through_mixing).to(device)
    model.initialize_v(torch_mats, k)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    mse_loss = nn.MSELoss()
    dataset = TensorDataset(*torch_mats)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    loss_history, recon_history, sim_history = _train_loop(
        model, dataloader, optimizer, scheduler, mse_loss, epochs, sim_weight, energy_type, warmup_epochs, verbose, device
    )
    
    model.eval()
    with torch.no_grad():
        all_torch_mats = [m.to(device) for m in torch_mats]
        final_latents, _, u_final = model(all_torch_mats)
        v_mats = [torch.nan_to_num(enc.v.detach().cpu()) for enc in model.linear_encoders]
        
    return {
        "model": model.cpu(),
        "model_type": "ned_simr",
        "u": torch.nan_to_num(u_final.cpu()),
        "v": v_mats,
        "latents": [torch.nan_to_num(l.cpu()) for l in final_latents],
        "loss_history": loss_history,
        "recon_history": recon_history,
        "sim_history": sim_history,
        "nsa_w": nsa_w,
        "sparseness_quantile": sparseness_quantile,
        "positivity": positivity,
    }
