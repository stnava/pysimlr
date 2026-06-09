import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
from .consensus import compute_shared_consensus

class AffineCouplingLayer(nn.Module):
    """
    Affine Coupling Layer for Normalizing Flows.
    
    Splits the input features, processes one half to predict scale (s)
    and translation (t), and applies them to the other half.
    """
    def __init__(self, dim: int, hidden_dim: int = 64, mask_type: str = "block", index: int = 0, scale_bound: float = 2.0):
        super().__init__()
        self.dim = dim
        self.mask_type = mask_type
        self.index = index
        self.split_dim = dim // 2
        self.scale_bound = scale_bound
        
        # Scale and translation neural network
        self.net = nn.Sequential(
            nn.Linear(self.split_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, (dim - self.split_dim) * 2)
        )
        
    def _get_mask(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = torch.zeros(self.dim, dtype=torch.bool, device=x.device)
        if self.mask_type == "alternate":
            mask[::2] = True
            if self.index % 2 == 1:
                mask = ~mask
        else:
            if self.index % 2 == 0:
                mask[:self.split_dim] = True
            else:
                mask[self.split_dim:] = True
        return mask, ~mask

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask, inv_mask = self._get_mask(x)
        x1 = x[:, mask]
        x2 = x[:, inv_mask]
        
        out = self.net(x1)
        s, t = out.chunk(2, dim=1)
        s = torch.tanh(s) * self.scale_bound  # bounded scale to prevent gradient explosion
        
        z2 = x2 * torch.exp(s) + t
        
        z = torch.zeros_like(x)
        z[:, mask] = x1
        z[:, inv_mask] = z2
        
        log_det = torch.sum(s, dim=1)
        return z, log_det

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        mask, inv_mask = self._get_mask(z)
        z1 = z[:, mask]
        z2 = z[:, inv_mask]
        
        out = self.net(z1)
        s, t = out.chunk(2, dim=1)
        s = torch.tanh(s) * self.scale_bound
        
        x2 = (z2 - t) * torch.exp(-s)
        
        x = torch.zeros_like(z)
        x[:, mask] = z1
        x[:, inv_mask] = x2
        return x

class PermutationLayer(nn.Module):
    """
    Fixed Permutation Layer to mix channels between coupling layers.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        perm = torch.randperm(dim)
        self.register_buffer("perm", perm)
        self.register_buffer("inv_perm", torch.argsort(perm))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return x[:, self.perm], torch.zeros(x.shape[0], device=x.device)
        
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        return z[:, self.inv_perm]

class NormalizingFlow(nn.Module):
    """
    Normalizing Flow model combining Affine Coupling and Permutation Layers.
    """
    def __init__(self, dim: int, num_layers: int = 4, hidden_dim: int = 64, scale_bound: float = 2.0):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(AffineCouplingLayer(dim, hidden_dim, mask_type="block", index=i, scale_bound=scale_bound))
            if i < num_layers - 1:
                self.layers.append(PermutationLayer(dim))
                
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = x
        total_log_det = torch.zeros(x.shape[0], device=x.device)
        for layer in self.layers:
            z, log_det = layer(z)
            total_log_det = total_log_det + log_det
        return z, total_log_det
          
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        x = z
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x

class FlowEncoderWrapper(nn.Module):
    """
    Wrapper to align a Normalizing Flow to a standard encoder interface.
    Extracts the first K components as the shared latent space.
    """
    def __init__(self, flow_model: NormalizingFlow, latent_dim: int):
        super().__init__()
        self.flow = flow_model
        self.latent_dim = latent_dim
        self.last_z = None
        self.last_log_det = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, log_det = self.flow(x)
        self.last_z = z
        self.last_log_det = log_det
        return z[:, :self.latent_dim]

class FlowDecoderWrapper(nn.Module):
    """
    Wrapper to align a Normalizing Flow to a standard decoder interface.
    Reconstructs the input from the shared latent space (padding private space with zeros).
    """
    def __init__(self, flow_model: NormalizingFlow, latent_dim: int):
        super().__init__()
        self.flow = flow_model
        self.latent_dim = latent_dim

    def forward(self, z_shared: torch.Tensor) -> torch.Tensor:
        device = z_shared.device
        batch_size = z_shared.shape[0]
        full_dim = self.flow.dim
        
        z_full = torch.zeros(batch_size, full_dim, device=device)
        z_full[:, :self.latent_dim] = z_shared
        
        return self.flow.inverse(z_full)

class FlowSiMRModel(nn.Module):
    """
    Normalizing Flow SiMR Model.
    """
    def __init__(self, input_dims: List[int], latent_dim: int, num_layers: int = 4, hidden_dim: int = 64, mixing_algorithm: str = "newton", scale_bound: float = 2.0):
        super().__init__()
        self.input_dims = input_dims
        self.latent_dim = latent_dim
        self.is_flow = True
        self.mixing_algorithm = mixing_algorithm
        
        self.flows = nn.ModuleList([NormalizingFlow(dim, num_layers, hidden_dim, scale_bound) for dim in input_dims])
        self.encoders = nn.ModuleList([FlowEncoderWrapper(flow, latent_dim) for flow in self.flows])
        self.decoders = nn.ModuleList([FlowDecoderWrapper(flow, latent_dim) for flow in self.flows])
        
        self.register_buffer("consensus_anchor", torch.zeros(len(input_dims) * latent_dim, latent_dim))
        
    def forward(self, x_list: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        latents = [enc(x) for enc, x in zip(self.encoders, x_list)]
        res_u = compute_shared_consensus(latents, mixing_algorithm=self.mixing_algorithm, k=self.latent_dim, training=self.training, anchor=self.consensus_anchor)
        if self.training:
            u_shared, new_anchor = res_u
            if new_anchor is not None: 
                self.consensus_anchor.copy_(0.9 * self.consensus_anchor + 0.1 * new_anchor)
        else:
            u_shared = res_u
            
        reconstructions = [dec(u_shared[i] if isinstance(u_shared, list) else u_shared) for i, dec in enumerate(self.decoders)]
        return latents, reconstructions, u_shared

class FlowConditionalInference:
    """
    Handles joint Gaussian parameterization and Woodbury-based conditional inference
    over the shared latent representations.
    """
    def __init__(self, num_modalities: int, latent_dim: int):
        self.num_modalities = num_modalities
        self.latent_dim = latent_dim
        self.joint_mean = None
        self.joint_cov = None

    def fit(self, shared_latents: List[torch.Tensor]):
        N = shared_latents[0].shape[0]
        device = shared_latents[0].device
        
        joint_z = torch.cat(shared_latents, dim=1) # N x (M*K)
        
        self.joint_mean = torch.mean(joint_z, dim=0)
        z_c = joint_z - self.joint_mean
        self.joint_cov = (z_c.t() @ z_c) / (N - 1) + torch.eye(z_c.shape[1], device=device) * 1e-4

    def predict_conditional(self, observed_idx: int, target_idx: int, observed_z: torch.Tensor) -> torch.Tensor:
        if self.joint_mean is None or self.joint_cov is None:
            raise ValueError("FlowConditionalInference has not been fit yet.")
            
        K = self.latent_dim
        obs_start, obs_end = observed_idx * K, (observed_idx + 1) * K
        tgt_start, tgt_end = target_idx * K, (target_idx + 1) * K
        
        mu_a = self.joint_mean[obs_start:obs_end]
        mu_b = self.joint_mean[tgt_start:tgt_end]
        
        cov_aa = self.joint_cov[obs_start:obs_end, obs_start:obs_end]
        cov_ba = self.joint_cov[tgt_start:tgt_end, obs_start:obs_end]
        
        diff = observed_z - mu_a
        solve_val = torch.linalg.solve(cov_aa, diff.t()).t()
        mu_cond = mu_b + (cov_ba @ solve_val.t()).t()
        
        return mu_cond

def _train_flow_loop(model: FlowSiMRModel, dataloader, optimizer, scheduler, epochs: int, sim_weight: float, energy_type: str, warmup_epochs: int, verbose: bool, device: torch.device, beta: float = 0.05, gamma: float = 2.0):
    loss_history, recon_history, sim_history = [], [], []
    penalty_weights = {
        "sim": sim_weight,
        "var": 1.0,
        "collapse": 1.0,
        "u_var": 1.0
    }
    
    from .deep import calculate_sim_loss
    mse_loss = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        epoch_loss, epoch_recon, epoch_sim = 0.0, 0.0, 0.0
        current_sim_weight = 0.0 if epoch < warmup_epochs else sim_weight
        penalty_weights["sim"] = current_sim_weight
        
        for batch in dataloader:
            batch_mats = [b.to(device) for b in batch]
            optimizer.zero_grad()
            
            latents, reconstructions, u_shared = model(batch_mats)
            
            # Flow negative log-likelihood (NLL) as reconstruction loss
            recon_loss = 0.0
            for enc in model.encoders:
                z = enc.last_z
                log_det = enc.last_log_det
                log_prior = -0.5 * torch.sum(z ** 2, dim=1) - 0.5 * z.shape[1] * np.log(2 * np.pi)
                nll = -torch.mean(log_prior + log_det)
                recon_loss += nll
                
            sim_loss_total, diagnostics = calculate_sim_loss(latents, u_shared, energy_type, weights=penalty_weights)
            
            # Hybrid zero-padded MSE reconstruction loss
            mse_recon_loss = sum(mse_loss(r, x) for r, x in zip(reconstructions, batch_mats))
            
            # Scaled total loss using beta and gamma parameters
            total_loss = beta * recon_loss + sim_loss_total + gamma * mse_recon_loss
            
            if torch.isnan(total_loss): 
                continue
                
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += total_loss.item()
            epoch_recon += recon_loss.item()
            epoch_sim += diagnostics["sim_loss"]
            
        epoch_loss /= len(dataloader)
        epoch_recon /= len(dataloader)
        epoch_sim /= len(dataloader)
        loss_history.append(epoch_loss)
        recon_history.append(epoch_recon)
        sim_history.append(epoch_sim)
        scheduler.step()
        
        if verbose and epoch % 10 == 0:
            print(f"Flow Epoch {epoch}: Total={epoch_loss:.4f} (NLL={epoch_recon:.4f}, Sim={epoch_sim:.4f})")
            
    return loss_history, recon_history, sim_history

def flow_simr(data_matrices: List[Union[torch.Tensor, np.ndarray]], 
              k: int, 
              epochs: int = 150, 
              batch_size: int = 64, 
              learning_rate: float = 1e-3, 
              weight_decay: float = 1e-4, 
              sim_weight: float = 1.0, 
              warmup_epochs: int = 20, 
              num_layers: int = 4, 
              hidden_dim: int = 64, 
              energy_type: str = "regression", 
              mixing_algorithm: str = "newton",
              device: Optional[str] = None,
              verbose: bool = False,
              beta: float = 0.05,
              scale_bound: float = 2.0,
              gamma: float = 2.0) -> Dict[str, Any]:
    """
    Perform Flow-based Similarity-driven Multi-view Representation (Flow-SiMLR).
    
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
        Warmup epochs before similarity loss.
    num_layers : int, default=4
        Number of flow coupling layers.
    hidden_dim : int, default=64
        Hidden size for coupling layer neural networks.
    energy_type : str, default="regression"
        Similarity objective type.
    mixing_algorithm : str, default="newton"
        Consensus mixing algorithm.
    device : Optional[str]
        PyTorch device.
    verbose : bool
        Print progress.
    beta : float, default=0.05
        Weight scaling factor for NLL reconstruction loss.
    scale_bound : float, default=2.0
        Bounded scaling coefficient for flow coupling transformations.
    gamma : float, default=2.0
        Weight scaling factor for zero-padded MSE reconstruction loss.
        
    Returns
    -------
    Dict[str, Any]
        Model results.
    """
    if device is None: 
        device = "cuda" if torch.cuda.is_available() else ("cpu")
    device = torch.device(device)
    
    from .deep import _standardize_deep, _get_optimizer
    from torch.utils.data import TensorDataset, DataLoader
    from torch.optim.lr_scheduler import CosineAnnealingLR
    
    torch_mats, provenance_list = _standardize_deep(data_matrices, ["centerAndScale"])
    input_dims = [m.shape[1] for m in torch_mats]
    
    model = FlowSiMRModel(input_dims, k, num_layers=num_layers, hidden_dim=hidden_dim, mixing_algorithm=mixing_algorithm, scale_bound=scale_bound).to(device)
    optimizer = _get_optimizer(model, "adam", learning_rate, weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    dataset = TensorDataset(*torch_mats)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    loss_h, recon_h, sim_h = _train_flow_loop(
        model, dataloader, optimizer, scheduler, epochs, sim_weight, energy_type, warmup_epochs, verbose, device, beta=beta, gamma=gamma
    )
    
    model.eval()
    with torch.no_grad():
        eval_mats = [m.to(device) for m in torch_mats]
        final_latents, recons, u_final = model(eval_mats)
        
    cond_inference = FlowConditionalInference(len(input_dims), k)
    cond_inference.fit(final_latents)
    
    result = {
        "model": model.cpu(),
        "model_type": "flow_simr",
        "u": u_final.cpu() if not isinstance(u_final, list) else [ui.cpu() for ui in u_final],
        "latents": [l.cpu() for l in final_latents],
        "reconstructions": [r.cpu() for r in recons],
        "loss_history": loss_h,
        "recon_history": recon_h,
        "sim_history": sim_h,
        "scale_list": ["centerAndScale"],
        "provenance_list": provenance_list,
        "cond_inference": cond_inference
    }
    
    result["errors"] = [
        torch.norm(x.cpu() - r.cpu(), p="fro").item() / (torch.norm(x.cpu(), p="fro").item() + 1e-10) 
        for x, r in zip(torch_mats, recons)
    ]
    
    return result
