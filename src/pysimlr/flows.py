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
        
        # Dynamically determine the input and output sizes to handle odd dimensions
        if index % 2 == 0:
            self.input_size = self.split_dim
            self.output_size = (dim - self.split_dim) * 2
        else:
            self.input_size = dim - self.split_dim
            self.output_size = self.split_dim * 2
            
        # Scale and translation neural network
        self.net = nn.Sequential(
            nn.Linear(self.input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_size)
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

class CustomRealNVP(nn.Module):
    def __init__(self, dim: int, num_layers: int = 4, hidden_dim: int = 64, scale_bound: float = 2.0):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(AffineCouplingLayer(dim, hidden_dim=hidden_dim, mask_type="block", index=i, scale_bound=scale_bound))
            if i < num_layers - 1:
                self.layers.append(PermutationLayer(dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_and_log_det(x)

    def forward_and_log_det(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = x
        total_log_det = torch.zeros(x.shape[0], device=x.device)
        for layer in self.layers:
            if isinstance(layer, PermutationLayer):
                z, _ = layer(z)
            else:
                z, log_det = layer(z)
                total_log_det = total_log_det + log_det
        return z, total_log_det

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        x = z
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x

    def inverse_and_log_det(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = z
        total_log_det = torch.zeros(z.shape[0], device=z.device)
        for layer in reversed(self.layers):
            if isinstance(layer, PermutationLayer):
                x = layer.inverse(x)
            else:
                # For AffineCouplingLayer: log_det_inv = -log_det
                mask, inv_mask = layer._get_mask(x)
                x1 = x[:, mask]
                out = layer.net(x1)
                s, t = out.chunk(2, dim=1)
                s = torch.tanh(s) * layer.scale_bound
                total_log_det = total_log_det - torch.sum(s, dim=1)
                x = layer.inverse(x)
        return x, total_log_det

class NormalizingFlow(nn.Module):
    """
    Normalizing Flow model wrapped around antstorch's create_real_nvp_normalizing_flow_model,
    with a pure PyTorch fallback if antstorch is not available.
    """
    def __init__(self, dim: int, num_layers: int = 4, hidden_dim: int = 64, scale_bound: float = 2.0):
        super().__init__()
        self.dim = dim
        try:
            from antstorch import create_real_nvp_normalizing_flow_model
            # Map num_layers to K, hidden_dim to mlp_width, scale_bound to scale_cap
            self.flow = create_real_nvp_normalizing_flow_model(
                latent_size=dim,
                K=num_layers,
                mlp_width=hidden_dim,
                scale_cap=scale_bound
            )
            self.use_fallback = False
        except ImportError:
            # Fallback to local implementation
            self.flow = CustomRealNVP(dim, num_layers, hidden_dim, scale_bound)
            self.use_fallback = True

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_fallback:
            return self.flow(x)
        else:
            return self.flow.forward_and_log_det(x)

    def forward_and_log_det(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.flow.forward_and_log_det(x)

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        return self.flow.inverse(z)

    def inverse_and_log_det(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.flow.inverse_and_log_det(z)

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
        if self.latent_dim > z.shape[1]:
            padding = torch.zeros(z.shape[0], self.latent_dim - z.shape[1], device=z.device, dtype=z.dtype)
            return torch.cat([z, padding], dim=1)
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
        
        z_full = torch.zeros(batch_size, full_dim, device=device, dtype=z_shared.dtype)
        min_dim = min(self.latent_dim, full_dim)
        z_full[:, :min_dim] = z_shared[:, :min_dim]
        
        return self.flow.inverse(z_full)

class FlowSiMRModel(nn.Module):
    """
    Normalizing Flow SiMR Model.
    """
    def __init__(self, input_dims: List[int], latent_dim: int, num_layers: int = 4, hidden_dim: int = 64, mixing_algorithm: str = "newton", scale_bound: float = 2.0, dynamic_weights: bool = False, mai_metric: str = "procrustes_r2", dynamic_weights_start: Optional[int] = None, use_rank_mai: bool = False):
        super().__init__()
        self.input_dims = input_dims
        self.latent_dim = latent_dim
        self.is_flow = True
        self.mixing_algorithm = mixing_algorithm
        self.dynamic_weights = dynamic_weights
        self.mai_metric = mai_metric
        self.dynamic_weights_start = dynamic_weights_start
        self.use_rank_mai = use_rank_mai
        
        self.flows = nn.ModuleList([NormalizingFlow(dim, num_layers, hidden_dim, scale_bound) for dim in input_dims])
        self.encoders = nn.ModuleList([FlowEncoderWrapper(flow, latent_dim) for flow in self.flows])
        self.decoders = nn.ModuleList([FlowDecoderWrapper(flow, latent_dim) for flow in self.flows])
        
        self.register_buffer("consensus_anchor", torch.zeros(len(input_dims) * latent_dim, latent_dim))
        self.register_buffer("mai", torch.ones(len(input_dims)) / len(input_dims))
        self.register_buffer("modality_weights", torch.ones(len(input_dims)) / len(input_dims))
        
    def update_mai(self, latents: List[torch.Tensor], epoch: int, total_epochs: int, dynamic_weights_start: Optional[int] = None):
        if not getattr(self, "dynamic_weights", False): return
        
        if dynamic_weights_start is None:
            dynamic_weights_start = getattr(self, "dynamic_weights_start", None)
            
        with torch.no_grad():
            if dynamic_weights_start is not None and epoch < dynamic_weights_start:
                # Keep weights uniform during training before the delay/warmup period ends
                uniform_w = torch.ones_like(self.modality_weights) / len(self.modality_weights)
                self.modality_weights.copy_(uniform_w)
                return
                
            if getattr(self, "use_rank_mai", False):
                ranked_latents = []
                for p in latents:
                    r = torch.argsort(torch.argsort(p, dim=0), dim=0).float()
                    ranked_latents.append(r)
                latents = ranked_latents
                
            norm_projs = []
            for p in latents:
                p_c = p - p.mean(dim=0, keepdim=True)
                p_norm = torch.norm(p_c, p='fro')
                if p_norm > 1e-8:
                    norm_projs.append(p_c / p_norm)
                else:
                    norm_projs.append(torch.zeros_like(p_c))
            
            mais = []
            for i in range(len(norm_projs)):
                loo_projs = [norm_projs[j] for j in range(len(norm_projs)) if j != i]
                if not loo_projs:
                    mais.append(1.0)
                    continue
                u_loo = torch.mean(torch.stack(loo_projs), dim=0)
                u_loo_norm = torch.norm(u_loo, p='fro')
                if u_loo_norm > 1e-8:
                    u_loo = u_loo / u_loo_norm
                    Z = norm_projs[i]
                    cross = Z.t() @ u_loo
                    try:
                        u_svd, s_svd, vh_svd = torch.linalg.svd(cross, full_matrices=False)
                        metric = getattr(self, "mai_metric", "procrustes_r2")
                        
                        if metric == "procrustes_r2" or metric == "procrustes_r2_sharp":
                            omega = u_svd @ vh_svd
                            aligned = Z @ omega
                            r2 = max(0.0, 1.0 - (torch.norm(aligned - u_loo, p='fro')**2 / (torch.norm(u_loo, p='fro')**2 + 1e-8)).item())
                            if metric == "procrustes_r2_sharp":
                                _, s_latent, _ = torch.linalg.svd(Z, full_matrices=False)
                                sharpness = s_latent[0] / (s_latent.sum() + 1e-8)
                                mais.append(r2 * sharpness.item())
                            else:
                                mais.append(r2)
                        elif metric == "cca":
                            mais.append(s_svd.mean().item())
                        elif metric == "rvcoef":
                            num = torch.norm(cross, p='fro')**2
                            den = torch.norm(Z.t() @ Z, p='fro') * torch.norm(u_loo.t() @ u_loo, p='fro')
                            mais.append((num / (den + 1e-8)).item())
                        else: # trace
                            mais.append(max(0.0, torch.trace(cross).item()))
                    except:
                        mais.append(0.0)
                else:
                    mais.append(0.0)
            
            mai_tensor = torch.tensor(mais, device=self.mai.device)
            # EMA for stability
            self.mai.copy_(0.9 * self.mai + 0.1 * mai_tensor)
            
            # Gating Logic
            center = self.mai.mean()
            progress = min(1.0, epoch / 30.0)
            steepness = 5.0 + 10.0 * progress
            gate = torch.sigmoid(steepness * (self.mai - center))
            
            # Target weights = Proportional MAI * Gating Signal
            raw_w = self.mai * gate
            target_w = raw_w / (raw_w.sum() + 1e-8)
            
            # Temporal transition (from uniform to target) starting relative to dynamic_weights_start
            if dynamic_weights_start is not None:
                rho = max(0.0, min(1.0, (epoch - dynamic_weights_start) / 20.0))
            else:
                rho = max(0.0, min(1.0, (epoch - 10) / 20.0))
            uniform_w = torch.ones_like(target_w) / len(target_w)
            self.modality_weights.copy_((1.0 - rho) * uniform_w + rho * target_w)

    def forward(self, x_list: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        latents = [enc(x) for enc, x in zip(self.encoders, x_list)]
        res_u = compute_shared_consensus(latents, mixing_algorithm=self.mixing_algorithm, k=self.latent_dim, training=self.training, anchor=self.consensus_anchor, modality_weights=self.modality_weights if getattr(self, "dynamic_weights", False) else None)
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

def _train_flow_loop(model: FlowSiMRModel, dataloader, optimizer, scheduler, epochs: int, sim_weight: float, energy_type: str, warmup_epochs: int, verbose: bool, device: torch.device, beta: float = 0.05, gamma: float = 2.0, dynamic_weights_start: Optional[int] = None, stabilization_start_epoch: Optional[int] = None, stabilization_ramp_epochs: Optional[int] = None):
    loss_history, recon_history, sim_history = [], [], []
    model.weight_history = []
    model.mai_history = []
    penalty_weights = {
        "sim": sim_weight,
        "var": 1.0,
        "collapse": 1.0,
        "u_var": 1.0
    }
    
    from .deep import calculate_sim_loss, _resolve_stabilization_schedule, _update_first_layer_schedule, invariant_orthogonality_defect
    mse_loss = nn.MSELoss()
    
    stabilization_start_epoch, stabilization_ramp_epochs = _resolve_stabilization_schedule(
        epochs,
        warmup_epochs,
        stabilization_start_epoch,
        stabilization_ramp_epochs,
    )
    
    for epoch in range(epochs):
        model.train()
        epoch_loss, epoch_recon, epoch_sim = 0.0, 0.0, 0.0
        
        # Update first-layer projection schedule (straight-through estimation)
        _update_first_layer_schedule(model, epoch, epochs, stabilization_start_epoch, stabilization_ramp_epochs)
        
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
            
            # Add orthogonality penalty for encoder basis V (both actual enc.v and raw enc.v_raw)
            if hasattr(model, 'linear_encoders'):
                total_loss += 0.05 * sum(invariant_orthogonality_defect(enc.v) for enc in model.linear_encoders)
                total_loss += 0.05 * sum(invariant_orthogonality_defect(enc.v_raw) for enc in model.linear_encoders)
                
            if torch.isnan(total_loss): 
                continue
                
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            if hasattr(model, 'update_mai'):
                model.update_mai(latents, epoch, epochs, dynamic_weights_start=dynamic_weights_start)
            
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
        
        if hasattr(model, 'modality_weights'):
            model.weight_history.append(model.modality_weights.detach().cpu().numpy().copy())
        if hasattr(model, 'mai'):
            model.mai_history.append(model.mai.detach().cpu().numpy().copy())
        
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
              gamma: float = 2.0,
              dynamic_weights: bool = False,
              mai_metric: str = "procrustes_r2",
              dynamic_weights_start: Optional[int] = None,
              use_rank_mai: bool = False) -> Dict[str, Any]:
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
    
    model = FlowSiMRModel(input_dims, k, num_layers=num_layers, hidden_dim=hidden_dim, mixing_algorithm=mixing_algorithm, scale_bound=scale_bound, dynamic_weights=dynamic_weights, mai_metric=mai_metric, dynamic_weights_start=dynamic_weights_start, use_rank_mai=use_rank_mai).to(device)
    optimizer = _get_optimizer(model, "adam", learning_rate, weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    dataset = TensorDataset(*torch_mats)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    loss_h, recon_h, sim_h = _train_flow_loop(
        model, dataloader, optimizer, scheduler, epochs, sim_weight, energy_type, warmup_epochs, verbose, device, beta=beta, gamma=gamma, dynamic_weights_start=dynamic_weights_start
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
        "cond_inference": cond_inference,
        "modality_weights": model.modality_weights.detach().cpu().numpy(),
        "mai": model.mai.detach().cpu().numpy(),
        "weight_history": getattr(model, "weight_history", [])
    }
    
    result["errors"] = [
        torch.norm(x.cpu() - r.cpu(), p="fro").item() / (torch.norm(x.cpu(), p="fro").item() + 1e-10) 
        for x, r in zip(torch_mats, recons)
    ]
    
    return result


class FlowSiMRVModel(nn.Module):
    """
    Normalizing Flow SiMR Model with Pre-pended Linear Encoder Matrix V (Flow-SiMLR-V).
    """
    def __init__(self, input_dims: List[int], latent_dim: int, num_layers: int = 4, 
                 hidden_dim: int = 64, mixing_algorithm: str = 'newton', scale_bound: float = 2.0,
                 nsa_w: float = 0.1, positivity: str = 'positive', 
                 sparseness_quantile: Union[float, List[float]] = 0.0, use_nsa: bool = True,
                 dynamic_weights: bool = False, mai_metric: str = "procrustes_r2",
                 dynamic_weights_start: Optional[int] = None, use_rank_mai: bool = False,
                 retraction_type: str = "soft_polar"):
        super().__init__()
        if positivity is True or (isinstance(positivity, str) and positivity.lower() == 'true'):
            positivity = 'positive'
        elif positivity is False or (isinstance(positivity, str) and positivity.lower() == 'false'):
            positivity = 'either'
        self.input_dims = input_dims
        self.latent_dim = latent_dim
        self.is_flow = True
        self.mixing_algorithm = mixing_algorithm
        self.dynamic_weights = dynamic_weights
        self.mai_metric = mai_metric
        self.dynamic_weights_start = dynamic_weights_start
        self.use_rank_mai = use_rank_mai
        self.retraction_type = retraction_type
        
        # Local import to prevent circular dependency
        from .deep import LENDNSAEncoder
        
        if isinstance(sparseness_quantile, (float, int)):
            sparseness_quantile = [float(sparseness_quantile)] * len(input_dims)
            
        self.linear_encoders = nn.ModuleList([
            LENDNSAEncoder(dim, latent_dim, nsa_w, positivity, sq, use_nsa=use_nsa, retraction_type=retraction_type) 
            for dim, sq in zip(input_dims, sparseness_quantile)
        ])
        
        # flows now operate on the low-dimensional projected space of size latent_dim (K)
        self.flows = nn.ModuleList([
            NormalizingFlow(latent_dim, num_layers, hidden_dim, scale_bound) 
            for _ in range(len(input_dims))
        ])
        self.encoders = nn.ModuleList([
            FlowEncoderWrapper(flow, latent_dim) 
            for flow in self.flows
        ])
        self.decoders = nn.ModuleList([
            FlowDecoderWrapper(flow, latent_dim) 
            for flow in self.flows
        ])
        
        self.register_buffer('consensus_anchor', torch.zeros(len(input_dims) * latent_dim, latent_dim))
        self.register_buffer("mai", torch.ones(len(input_dims)) / len(input_dims))
        self.register_buffer("modality_weights", torch.ones(len(input_dims)) / len(input_dims))
        
    def update_mai(self, latents: List[torch.Tensor], epoch: int, total_epochs: int, dynamic_weights_start: Optional[int] = None):
        if not getattr(self, "dynamic_weights", False): return
        
        if dynamic_weights_start is None:
            dynamic_weights_start = getattr(self, "dynamic_weights_start", None)
            
        with torch.no_grad():
            if dynamic_weights_start is not None and epoch < dynamic_weights_start:
                # Keep weights uniform during training before the delay/warmup period ends
                uniform_w = torch.ones_like(self.modality_weights) / len(self.modality_weights)
                self.modality_weights.copy_(uniform_w)
                return
                
            if getattr(self, "use_rank_mai", False):
                ranked_latents = []
                for p in latents:
                    r = torch.argsort(torch.argsort(p, dim=0), dim=0).float()
                    ranked_latents.append(r)
                latents = ranked_latents
                
            norm_projs = []
            for p in latents:
                p_c = p - p.mean(dim=0, keepdim=True)
                p_norm = torch.norm(p_c, p='fro')
                if p_norm > 1e-8:
                    norm_projs.append(p_c / p_norm)
                else:
                    norm_projs.append(torch.zeros_like(p_c))
            
            mais = []
            for i in range(len(norm_projs)):
                loo_projs = [norm_projs[j] for j in range(len(norm_projs)) if j != i]
                if not loo_projs:
                    mais.append(1.0)
                    continue
                u_loo = torch.mean(torch.stack(loo_projs), dim=0)
                u_loo_norm = torch.norm(u_loo, p='fro')
                if u_loo_norm > 1e-8:
                    u_loo = u_loo / u_loo_norm
                    Z = norm_projs[i]
                    cross = Z.t() @ u_loo
                    try:
                        u_svd, s_svd, vh_svd = torch.linalg.svd(cross, full_matrices=False)
                        metric = getattr(self, "mai_metric", "procrustes_r2")
                        
                        if metric == "procrustes_r2" or metric == "procrustes_r2_sharp":
                            omega = u_svd @ vh_svd
                            aligned = Z @ omega
                            r2 = max(0.0, 1.0 - (torch.norm(aligned - u_loo, p='fro')**2 / (torch.norm(u_loo, p='fro')**2 + 1e-8)).item())
                            if metric == "procrustes_r2_sharp":
                                _, s_latent, _ = torch.linalg.svd(Z, full_matrices=False)
                                sharpness = s_latent[0] / (s_latent.sum() + 1e-8)
                                mais.append(r2 * sharpness.item())
                            else:
                                mais.append(r2)
                        elif metric == "cca":
                            mais.append(s_svd.mean().item())
                        elif metric == "rvcoef":
                            num = torch.norm(cross, p='fro')**2
                            den = torch.norm(Z.t() @ Z, p='fro') * torch.norm(u_loo.t() @ u_loo, p='fro')
                            mais.append((num / (den + 1e-8)).item())
                        else: # trace
                            mais.append(max(0.0, torch.trace(cross).item()))
                    except:
                        mais.append(0.0)
                else:
                    mais.append(0.0)
            
            mai_tensor = torch.tensor(mais, device=self.mai.device)
            # EMA for stability
            self.mai.copy_(0.9 * self.mai + 0.1 * mai_tensor)
            
            # Gating Logic
            center = self.mai.mean()
            progress = min(1.0, epoch / 30.0)
            steepness = 5.0 + 10.0 * progress
            gate = torch.sigmoid(steepness * (self.mai - center))
            
            # Target weights = Proportional MAI * Gating Signal
            raw_w = self.mai * gate
            target_w = raw_w / (raw_w.sum() + 1e-8)
            
            # Temporal transition (from uniform to target) starting relative to dynamic_weights_start
            if dynamic_weights_start is not None:
                rho = max(0.0, min(1.0, (epoch - dynamic_weights_start) / 20.0))
            else:
                rho = max(0.0, min(1.0, (epoch - 10) / 20.0))
            uniform_w = torch.ones_like(target_w) / len(target_w)
            self.modality_weights.copy_((1.0 - rho) * uniform_w + rho * target_w)

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
        # 1. Linear projection layer
        projected_zs = [enc(x) for enc, x in zip(self.linear_encoders, x_list)]
        
        # 2. Flow encoders wrapper on projected zs
        latents = [enc(z) for enc, z in zip(self.encoders, projected_zs)]
        
        # 3. Consensus mixing
        res_u = compute_shared_consensus(latents, mixing_algorithm=self.mixing_algorithm, k=self.latent_dim, training=self.training, anchor=self.consensus_anchor, modality_weights=self.modality_weights if getattr(self, "dynamic_weights", False) else None)
        if self.training:
            u_shared, new_anchor = res_u
            if new_anchor is not None: 
                self.consensus_anchor.copy_(0.9 * self.consensus_anchor + 0.1 * new_anchor)
        else:
            u_shared = res_u
            
        # 4. Decoders: invert normalizing flows, then project back using V_m.t()
        reconstructions = []
        for i, (flow, enc) in enumerate(zip(self.flows, self.linear_encoders)):
            ui = u_shared[i] if isinstance(u_shared, list) else u_shared
            z_recon = flow.inverse(ui)
            v = enc.v
            x_recon = z_recon @ v.t()
            reconstructions.append(x_recon)
            
        return latents, reconstructions, u_shared
        
    def initialize_weights(self, data_matrices: List[torch.Tensor]):
        from .simlr import ba_svd
        with torch.no_grad():
            k = self.latent_dim
            for i, x in enumerate(data_matrices):
                u, s, v = ba_svd(x, nu=0, nv=k)
                if v.shape[1] < k: 
                    v = torch.cat([v, torch.randn(v.shape[0], k-v.shape[1], device=v.device)*1e-4], dim=1)
                if self.linear_encoders[i].positivity in {'positive', 'hard', 'softplus'}:
                    for j in range(v.shape[1]):
                        if v[:, j].sum() < 0: v[:, j] *= -1
                self.linear_encoders[i].v_raw.copy_(v.to(x.dtype))

def flow_simr_v(data_matrices: List[Union[torch.Tensor, np.ndarray]], 
                k: int, 
                epochs: int = 150, 
                batch_size: int = 64, 
                learning_rate: float = 1e-3, 
                weight_decay: float = 1e-4, 
                sim_weight: float = 1.0, 
                warmup_epochs: int = 20, 
                num_layers: int = 4, 
                hidden_dim: int = 64, 
                energy_type: str = 'regression', 
                mixing_algorithm: str = 'newton',
                device: Optional[str] = None,
                verbose: bool = False,
                beta: float = 0.05,
                scale_bound: float = 2.0,
                gamma: float = 2.0,
                nsa_w: float = 0.1,
                positivity: str = 'positive',
                sparseness_quantile: Union[float, List[float]] = 0.0,
                use_nsa: bool = True,
                dynamic_weights: bool = False,
                mai_metric: str = 'procrustes_r2',
                dynamic_weights_start: Optional[int] = None,
                use_rank_mai: bool = False,
                stabilization_start_epoch: Optional[int] = None,
                stabilization_ramp_epochs: Optional[int] = None,
                retraction_type: str = 'soft_polar') -> Dict[str, Any]:
    """
    Perform Flow-based Similarity-driven Multi-view Representation with Linear Encoder (Flow-SiMLR-V).
    """
    if positivity is True or (isinstance(positivity, str) and positivity.lower() == 'true'):
        positivity = 'positive'
    elif positivity is False or (isinstance(positivity, str) and positivity.lower() == 'false'):
        positivity = 'either'
        
    if device is None: 
        device = 'cuda' if torch.cuda.is_available() else ('cpu')
    device = torch.device(device)
    
    from .deep import _standardize_deep, _get_optimizer
    from torch.utils.data import TensorDataset, DataLoader
    from torch.optim.lr_scheduler import CosineAnnealingLR
    
    torch_mats, provenance_list = _standardize_deep(data_matrices, ['centerAndScale'])
    input_dims = [m.shape[1] for m in torch_mats]
    
    model = FlowSiMRVModel(
        input_dims, k, num_layers=num_layers, hidden_dim=hidden_dim, 
        mixing_algorithm=mixing_algorithm, scale_bound=scale_bound,
        nsa_w=nsa_w, positivity=positivity, 
        sparseness_quantile=sparseness_quantile, use_nsa=use_nsa,
        dynamic_weights=dynamic_weights, mai_metric=mai_metric,
        dynamic_weights_start=dynamic_weights_start,
        use_rank_mai=use_rank_mai,
        retraction_type=retraction_type
    ).to(device)
    
    # Initialize linear encoders via SVD
    model.initialize_weights(torch_mats)
    
    optimizer = _get_optimizer(model, 'adam', learning_rate, weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    dataset = TensorDataset(*torch_mats)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    loss_h, recon_h, sim_h = _train_flow_loop(
        model, dataloader, optimizer, scheduler, epochs, sim_weight, energy_type, warmup_epochs, verbose, device, 
        beta=beta, gamma=gamma, dynamic_weights_start=dynamic_weights_start,
        stabilization_start_epoch=stabilization_start_epoch, stabilization_ramp_epochs=stabilization_ramp_epochs
    )
    
    model.eval()
    with torch.no_grad():
        eval_mats = [m.to(device) for m in torch_mats]
        final_latents, recons, u_final = model(eval_mats)
        
    cond_inference = FlowConditionalInference(len(input_dims), k)
    cond_inference.fit(final_latents)
    
    # Extract linear projection matrices V
    v_mats = [enc.v.detach().cpu() for enc in model.linear_encoders]
    
    result = {
        'model': model.cpu(),
        'model_type': 'flow_simr_v',
        'u': u_final.cpu() if not isinstance(u_final, list) else [ui.cpu() for ui in u_final],
        'latents': [l.cpu() for l in final_latents],
        'reconstructions': [r.cpu() for r in recons],
        'v': v_mats,
        'loss_history': loss_h,
        'recon_history': recon_h,
        'sim_history': sim_h,
        'scale_list': ['centerAndScale'],
        'provenance_list': provenance_list,
        'cond_inference': cond_inference,
        "modality_weights": model.modality_weights.detach().cpu().numpy(),
        "mai": model.mai.detach().cpu().numpy(),
        "weight_history": getattr(model, "weight_history", []),
        "mai_history": getattr(model, "mai_history", [])
    }
    
    result['errors'] = [
        torch.norm(x.cpu() - r.cpu(), p='fro').item() / (torch.norm(x.cpu(), p='fro').item() + 1e-10) 
        for x, r in zip(torch_mats, recons)
    ]
    
    return result
