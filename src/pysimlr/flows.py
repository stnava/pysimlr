import warnings
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
from .utils import basis_rank_report
from .defaults import DEFAULT_ENERGY_TYPE, DEFAULT_NSA_W, DEFAULT_TOPOLOGY, DEFAULT_MIXING_ALGORITHM, DEFAULT_TOL, DEFAULT_COLLAPSE_THRESHOLD, DEFAULT_DEEP_ITERATIONS, check_collapse_threshold, check_nsa_w


#: MAI similarity metrics. `trace` was removed: tr(Z' u_loo) is the only one
#: not computed through the SVD and it is not invariant to the arbitrary
#: rotation of the k latent coordinates -- measured, the same data under three
#: rotations gave 0.0348, 0.1319 and 0.0000, while the others were constant to
#: four decimals. It ranked a planted pure-noise view above a signal view.
from .deep import _ModalityWeightingMixin, _MultiViewModelMixin, _foreach, _linear_encoders, _sum_on_host


class preserve_matplotlib_backend:
    """
    Context manager restoring the active matplotlib backend on exit.

    The optional `antstorch` dependency calls ``matplotlib.use("Agg")`` at
    module import time (in ``lamnr_flows/core/train_lamnr_glow_base.py`` and
    ``lamnr_glow_tool_base.py``). Because this module imports antstorch lazily
    inside the flow constructors, merely *fitting* a flow model switched the
    process-wide backend to Agg.

    In a notebook or Quarto render that silently breaks inline figure capture:
    the cell still runs to completion, but nothing is emitted, so figures
    disappear from the document without any error. Three figures in the
    Flow-SiMR-V appendix were missing for exactly this reason.

    Restoring the backend afterwards leaves antstorch's own behaviour intact
    while keeping the side effect out of the caller's session. Only used where
    antstorch is imported; a no-op when matplotlib is absent.
    """

    def __enter__(self):
        self._backend = None
        try:
            import matplotlib
        except ImportError:   # nothing to preserve without matplotlib
            return self
        self._backend = matplotlib.get_backend()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._backend is None:
            return False
        import matplotlib
        if matplotlib.get_backend() != self._backend:
            matplotlib.use(self._backend, force=False)
        return False


class AffineCouplingLayer(nn.Module):
    """
    Affine Coupling Layer for Normalizing Flows.
    
    Splits the input features, processes one half to predict scale (s)
    and translation (t), and applies them to the other half. When ``dim == 1``
    (``split_dim == 0``), the empty partition acts as a bias-only conditioner
    so the layer reduces to a global learnable 1D affine bijection
    ``z = x * exp(s) + t``; use ``dim >= 2`` for conditional feature coupling.
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

_ANTSTORCH_REAL_NVP_BUILDER = None


def _load_antstorch_real_nvp_builder():
    """Load ANTsTorch's `create_real_nvp_normalizing_flow_model` directly from
    its pure-PyTorch submodule when present, avoiding `antstorch.__init__`'s
    ~2.6s eager import of unrelated ITK/ANTsPy/matplotlib modules."""
    global _ANTSTORCH_REAL_NVP_BUILDER
    import importlib.util
    import os
    import sys

    if "antstorch" in sys.modules:
        mod = sys.modules["antstorch"]
        if mod is None:
            raise ImportError("No module named 'antstorch'")
        fn = getattr(mod, "create_real_nvp_normalizing_flow_model", None)
        if fn is not None:
            return fn
    if _ANTSTORCH_REAL_NVP_BUILDER is not None:
        return _ANTSTORCH_REAL_NVP_BUILDER

    spec = importlib.util.find_spec("antstorch")
    if spec is None:
        raise ImportError("No module named 'antstorch'")
    for loc in spec.submodule_search_locations or ():
        candidate = os.path.join(
            loc, "lamnr_flows", "architectures", "create_normalizing_flow_model.py"
        )
        if os.path.isfile(candidate):
            mspec = importlib.util.spec_from_file_location(
                "_antstorch_real_nvp_builder", candidate
            )
            if mspec is not None and mspec.loader is not None:
                mod = importlib.util.module_from_spec(mspec)
                mspec.loader.exec_module(mod)
                fn = getattr(mod, "create_real_nvp_normalizing_flow_model", None)
                if fn is not None:
                    _ANTSTORCH_REAL_NVP_BUILDER = fn
                    return fn
    with preserve_matplotlib_backend():
        from antstorch import create_real_nvp_normalizing_flow_model
    _ANTSTORCH_REAL_NVP_BUILDER = create_real_nvp_normalizing_flow_model
    return create_real_nvp_normalizing_flow_model


class NormalizingFlow(nn.Module):
    """
    Normalizing Flow model wrapped around antstorch's create_real_nvp_normalizing_flow_model,
    with a pure PyTorch fallback if antstorch is not available.
    """
    def __init__(self, dim: int, num_layers: int = 4, hidden_dim: int = 64, scale_bound: float = 2.0, force_fallback: bool = False):
        super().__init__()
        self.dim = dim
        if force_fallback:
            self.flow = CustomRealNVP(dim, num_layers, hidden_dim, scale_bound)
            self.use_fallback = True
            warnings.warn("Using local fallback CustomRealNVP because force_fallback is True.", UserWarning, stacklevel=2)
        else:
            try:
                create_real_nvp_normalizing_flow_model = _load_antstorch_real_nvp_builder()
                import antsnormflows as nf
                q0 = nf.distributions.DiagGaussian(dim)
                # Map num_layers to K, hidden_dim to mlp_width, scale_bound to scale_cap
                self.flow = create_real_nvp_normalizing_flow_model(
                    latent_size=dim,
                    K=num_layers,
                    mlp_width=hidden_dim,
                    scale_cap=scale_bound,
                    q0=q0
                )
                self.use_fallback = False
            except ImportError as e:
                # ANTsTorch is optional: without it, the local RealNVP is used,
                # with a warning and `use_fallback=True` on the model. Any other
                # failure of the ANTsTorch constructor propagates.
                self.flow = CustomRealNVP(dim, num_layers, hidden_dim, scale_bound)
                self.use_fallback = True
                warnings.warn(f"Using local fallback CustomRealNVP because ANTsTorch flow model initialization failed: {e}", UserWarning, stacklevel=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_fallback:
            return self.flow(x)
        return self.forward_and_log_det(x)

    def forward_and_log_det(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z, log_det = self.flow.forward_and_log_det(x)
        if not self.use_fallback and torch.isfinite(x).all() and (not torch.isfinite(z).all() or not torch.isfinite(log_det).all()):
            raise RuntimeError("[ActNorm] non-finite values in 'std' during flow forward pass (requires N >= 2 on fresh initialization).")
        return z, log_det

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

class FlowSiMRModel(_ModalityWeightingMixin, nn.Module):
    """
    Normalizing Flow SiMR Model.
    """
    def __init__(self, input_dims: List[int], latent_dim: int, num_layers: int = 4, hidden_dim: int = 64, mixing_algorithm: str = DEFAULT_MIXING_ALGORITHM, scale_bound: float = 2.0, topology: str = DEFAULT_TOPOLOGY, path_graph: Optional[Dict[int, List[int]]] = None, prune_threshold: Optional[float] = None, dynamic_weights: bool = False, mai_metric: str = "procrustes_r2", dynamic_weights_start: Optional[int] = None, use_rank_mai: bool = False, force_fallback: bool = False):
        super().__init__()
        self._init_consensus_state(len(input_dims), latent_dim, mixing_algorithm, topology, path_graph,
                                    prune_threshold, dynamic_weights, mai_metric, use_rank_mai, dynamic_weights_start)
        self.input_dims = input_dims
        self.latent_dim = latent_dim
        self.is_flow = True
        self.force_fallback = force_fallback
        
        self.flows = nn.ModuleList([NormalizingFlow(dim, num_layers, hidden_dim, scale_bound, force_fallback=force_fallback) for dim in input_dims])
        self.encoders = nn.ModuleList([FlowEncoderWrapper(flow, latent_dim) for flow in self.flows])
        self.decoders = nn.ModuleList([FlowDecoderWrapper(flow, latent_dim) for flow in self.flows])
        
        
    def forward(self, x_list: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        latents = [enc(x) for enc, x in zip(self.encoders, x_list)]
        u_shared = self._training_consensus(latents, self.latent_dim)
            
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

def _train_flow_loop(model: FlowSiMRModel, dataloader, optimizer, scheduler, epochs: int, sim_weight: float, energy_type: str, warmup_epochs: int, verbose: bool, device: torch.device, beta: float = 0.05, gamma: float = 2.0, dynamic_weights_start: Optional[int] = None, stabilization_start_epoch: Optional[int] = None, stabilization_ramp_epochs: Optional[int] = None,
                     tol: float = DEFAULT_TOL, patience: int = 10,
                     collapse_threshold: float = DEFAULT_COLLAPSE_THRESHOLD):
    loss_history, recon_history, sim_history = [], [], []
    model._n_fit = getattr(dataloader, "n", None)       # rows the bases are fitted on (MAI dof)
    nonfinite_grad_steps = 0
    model.weight_history = []
    model.mai_history = []
    penalty_weights = {
        "sim": sim_weight,
        "var": 1.0,
        "collapse": 1.0,
        "u_var": 1.0
    }
    
    from .deep import (LENDNSAEncoder, calculate_sim_loss, _resolve_stabilization_schedule,
                       _update_first_layer_schedule, orthogonality_defect,
                       resolve_warmup_epochs)
    warmup_epochs = resolve_warmup_epochs(warmup_epochs, epochs)
    mse_loss = nn.MSELoss()
    if hasattr(dataloader, "bind_device"):
        dataloader.bind_device(device)
    first_layer_encs = [
        enc for enc in getattr(model, "linear_encoders", ())
        if isinstance(enc, LENDNSAEncoder)
    ]
    
    stabilization_start_epoch, stabilization_ramp_epochs = _resolve_stabilization_schedule(
        epochs,
        warmup_epochs,
        stabilization_start_epoch,
        stabilization_ramp_epochs,
    )
    
    has_stepped = False
    from .utils import ConvergenceMonitor
    _monitor = ConvergenceMonitor(tol=float(tol), patience=int(patience),
                                  max_steps=epochs)
    _best_loss, _best_state, _best_epoch = float("inf"), None, None
    _warm = int(warmup_epochs or 0)
    _all_params = list(model.parameters())
    for epoch in range(epochs):
        model.train()
        step_loss, step_recon, step_sim = [], [], []
        valid_batches = 0
        
        # Update first-layer projection schedule (straight-through estimation)
        _update_first_layer_schedule(model, epoch, epochs, stabilization_start_epoch, stabilization_ramp_epochs)
        
        current_sim_weight = 0.0 if epoch < warmup_epochs else sim_weight
        penalty_weights["sim"] = current_sim_weight
        
        pre_staged = getattr(dataloader, "_device", None) == device
        for batch in dataloader:
            batch_mats = batch if pre_staged else [b.to(device) for b in batch]
            optimizer.zero_grad()
            try:
                for enc in first_layer_encs:
                    enc._step_v_cache = enc.v
                latents, reconstructions, u_shared = model(batch_mats)
                
                recon_loss = 0.0
                for enc in model.encoders:
                    z = enc.last_z
                    log_det = enc.last_log_det
                    log_prior = -0.5 * torch.sum(z ** 2, dim=1) - 0.5 * z.shape[1] * np.log(2 * np.pi)
                    nll = -torch.mean(log_prior + log_det)
                    recon_loss += nll

                sim_loss_total, diagnostics = calculate_sim_loss(latents, u_shared, energy_type, weights=penalty_weights, compute_diagnostics=False, collapse_threshold=collapse_threshold)

                # Hybrid zero-padded MSE reconstruction loss
                mse_recon_loss = sum(mse_loss(r, x) for r, x in zip(reconstructions, batch_mats))

                total_loss = beta * recon_loss + sim_loss_total + gamma * mse_recon_loss
                
                # Orthogonality penalty on the activated basis enc.v and the raw weights
                # enc.v_raw (a regulariser; see the same term in deep._train_loop).
                if hasattr(model, 'linear_encoders'):
                    total_loss += 0.05 * sum(orthogonality_defect(enc.v) for enc in model.linear_encoders)
                    total_loss += 0.05 * sum(orthogonality_defect(enc.v_raw) for enc in model.linear_encoders)
            finally:
                for enc in first_layer_encs:
                    enc._step_v_cache = None
                
            if torch.isnan(total_loss): 
                continue
                
            total_loss.backward()

            total_norm = torch.nn.utils.clip_grad_norm_(_all_params, max_norm=1.0, foreach=_foreach(_all_params))
            if not bool(torch.isfinite(total_norm)):
                nonfinite_grad_steps += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.step()
            has_stepped = True
            valid_batches += 1
            
            if hasattr(model, 'update_mai'):
                model.update_mai(latents, epoch, epochs, dynamic_weights_start=dynamic_weights_start)
            
            step_loss.append(total_loss.detach())
            step_recon.append(recon_loss.detach())
            step_sim.append(diagnostics["sim_loss"])

        denom = max(1, valid_batches)
        epoch_loss = _sum_on_host(step_loss) / denom
        epoch_recon = _sum_on_host(step_recon) / denom
        epoch_sim = _sum_on_host(step_sim) / denom
        loss_history.append(epoch_loss)
        if epoch >= _warm and np.isfinite(epoch_loss) and epoch_loss < _best_loss:
            _best_loss, _best_epoch = epoch_loss, epoch
            _best_state = {k_: v_.detach().clone() for k_, v_ in model.state_dict().items()}
        recon_history.append(epoch_recon)
        sim_history.append(epoch_sim)
        if has_stepped and scheduler is not None:
            scheduler.step(epoch_loss)
        
        if hasattr(model, 'modality_weights'):
            model.weight_history.append(model.modality_weights.detach().cpu().numpy().copy())
        if hasattr(model, 'mai'):
            model.mai_history.append(model.mai.detach().cpu().numpy().copy())
        
        if verbose and epoch % 10 == 0:
            print(f"Flow Epoch {epoch}: Total={epoch_loss:.4f} (NLL={epoch_recon:.4f}, Sim={epoch_sim:.4f})")

        if epoch >= _warm and _monitor.update(epoch_loss):
            if verbose:
                print(f"Flow stop at epoch {epoch}: "
                      f"{_monitor.report()['stop_reason']}")
            break

    if _best_state is not None:
        model.load_state_dict(_best_state)
    _conv_report = _monitor.report()
    _conv_report.update(returned="best" if _best_state is not None else "final",
                        best_epoch=_best_epoch, warmup_epochs=_warm,
                        best_loss=_best_loss if np.isfinite(_best_loss) else float("nan"))

    if nonfinite_grad_steps:
        warnings.warn(
            f"{nonfinite_grad_steps} optimizer step(s) were skipped because the "
            f"gradient was not finite while the loss was. The fit completed but "
            f"used fewer updates than requested.",
            RuntimeWarning, stacklevel=2,
        )

    return loss_history, recon_history, sim_history, _conv_report

def _fit_flow(model_type: str, build_model, data_matrices, k: int, *, iterations: int,
              batch_size: int, learning_rate: float, weight_decay: float, sim_weight: float,
              warmup_epochs: Optional[int], energy_type: str, optimizer_type: str,
              device: Optional[str], verbose: bool, beta: float, gamma: float,
              dynamic_weights_start: Optional[int], stabilization_start_epoch: Optional[int],
              stabilization_ramp_epochs: Optional[int], random_state: Optional[int],
              tol: float, patience: int, positivity: str = "either", nsa_w: float = DEFAULT_NSA_W,
              sparseness=None, init_kwargs: Optional[Dict[str, Any]] = None,
              collapse_threshold: float = DEFAULT_COLLAPSE_THRESHOLD) -> Dict[str, Any]:
    """Shared body of `flow_simr` and `flow_simr_v`.

    ``init_kwargs`` is given only for the models with a linear first layer
    (`flow_simr_v`); it is passed to ``model.initialize_weights``.
    """
    from .deep import (DEEP_OPTIMIZER_TYPES, _standardize_deep, _get_optimizer,
                       resolve_warmup_epochs, _aggregate_shared_consensus, _plateau_scheduler,
                       _finalize_bases, _resolve_device, _resolve_seed, _seeded_rng,
                       _shuffled_loader, _init_generator, _freeze_consensus)
    nsa_w = check_nsa_w(nsa_w)
    collapse_threshold = check_collapse_threshold(collapse_threshold)
    from .similarity import check_fit_energy
    check_fit_energy(energy_type, path="deep", positivity=positivity)
    if optimizer_type not in DEEP_OPTIMIZER_TYPES:
        raise ValueError(f"optimizer_type={optimizer_type!r} is not a flow-path optimizer; "
                         f"choose one of {DEEP_OPTIMIZER_TYPES}.")
    iterations = int(iterations)
    device = _resolve_device(device)
    seed = _resolve_seed(random_state)
    has_linear = init_kwargs is not None
    with _seeded_rng(seed, device):
        torch_mats, provenance_list = _standardize_deep(data_matrices)
        input_dims = [m.shape[1] for m in torch_mats]
        model = build_model(input_dims).to(device)
        if has_linear:
            init_type = init_kwargs.get("initialization_type", "pca")
            model.initialize_weights(torch_mats, generator=_init_generator(init_type, seed),
                                     **init_kwargs)
        optimizer = _get_optimizer(model, optimizer_type, learning_rate, weight_decay)
        scheduler = _plateau_scheduler(optimizer)
        dataloader = _shuffled_loader(torch_mats, batch_size, seed)
        warmup_epochs = resolve_warmup_epochs(warmup_epochs, iterations)
        loss_h, recon_h, sim_h, conv_report = _train_flow_loop(
            model, dataloader, optimizer, scheduler, iterations, sim_weight, energy_type,
            warmup_epochs, verbose, device, beta=beta, gamma=gamma,
            dynamic_weights_start=dynamic_weights_start,
            stabilization_start_epoch=stabilization_start_epoch,
            stabilization_ramp_epochs=stabilization_ramp_epochs, tol=tol, patience=patience,
            collapse_threshold=collapse_threshold)
        from .utils import warn_if_unconverged
        warn_if_unconverged(conv_report, model_type)
        model.eval()
        with torch.no_grad():
            v_mats, retraction_diagnostics = None, None
            if has_linear:
                v_mats = [enc.v.detach().cpu() for enc in model.linear_encoders]
                v_mats, retraction_diagnostics = _finalize_bases(
                    v_mats, positivity, nsa_w, energy_type, encoders=model.linear_encoders)
            eval_mats = [m.to(device) for m in torch_mats]
            final_latents, recons, u_final = model(eval_mats)
            u_aggregate = (_aggregate_shared_consensus(model, final_latents)
                           if isinstance(u_final, list) else u_final)
            consensus_residual = _freeze_consensus(model, final_latents, u_final, u_aggregate)
        cond_inference = FlowConditionalInference(len(input_dims), k)
        cond_inference.fit(final_latents)

    result = {
        "model": model.cpu(),
        "model_type": model_type,
        "u": u_aggregate.cpu(),
        "u_per_modality": ([ui.cpu() for ui in u_final] if isinstance(u_final, list) else None),
        "latents": [l.cpu() for l in final_latents],
        "reconstructions": [r.cpu() for r in recons],
        "loss_history": loss_h,
        "recon_history": recon_h,
        "sim_history": sim_h,
        "convergence": conv_report,
        "scale_list": ["centerAndScale"],
        "provenance_list": provenance_list,
        "sparseness": sparseness,
        "random_state": seed,
        "consensus_map_residual": consensus_residual,
        "cond_inference": cond_inference,
        "modality_weights": model.modality_weights.detach().cpu().numpy(),
        "mai": model.mai.detach().cpu().numpy(),
        "weight_history": getattr(model, "weight_history", []),
        "mai_history": getattr(model, "mai_history", []),
    }
    if has_linear:
        result["v"] = v_mats
        result["retraction_diagnostics"] = retraction_diagnostics
        result["initialization_type"] = init_kwargs.get("initialization_type", "pca")
        # How many components actually came back; see `utils.basis_rank_report`.
        result.update(basis_rank_report(v_mats, k))
    result["errors"] = [
        torch.norm(x.cpu() - r.cpu(), p="fro").item() / (torch.norm(x.cpu(), p="fro").item() + 1e-10)
        for x, r in zip(torch_mats, recons)
    ]
    return result


def flow_simr(data_matrices: List[Union[torch.Tensor, np.ndarray]],
              k: int,
              iterations: int = DEFAULT_DEEP_ITERATIONS,
              batch_size: int = 64,
              learning_rate: float = 1e-3,
              weight_decay: float = 1e-4,
              sim_weight: float = 1.0,
              warmup_epochs: Optional[int] = None,
              num_layers: int = 4,
              hidden_dim: int = 64,
              energy_type: str = DEFAULT_ENERGY_TYPE,
              mixing_algorithm: str = DEFAULT_MIXING_ALGORITHM,
              topology: str = DEFAULT_TOPOLOGY,
              path_graph: Optional[Dict[int, List[int]]] = None,
              prune_threshold: Optional[float] = None,
              optimizer_type: str = "adam",
              device: Optional[str] = None,
              verbose: bool = False,
              beta: float = 0.05,
              scale_bound: float = 2.0,
              gamma: float = 2.0,
              dynamic_weights: bool = False,
              mai_metric: str = "procrustes_r2",
              dynamic_weights_start: Optional[int] = None,
              use_rank_mai: bool = False,
              force_fallback: bool = False,
              random_state: Optional[int] = None,
              tol: float = DEFAULT_TOL,
              patience: int = 10,
              collapse_threshold: float = DEFAULT_COLLAPSE_THRESHOLD) -> Dict[str, Any]:
    """
    Flow-based Similarity-driven Multi-view Representation (Flow-SiMR).

    A RealNVP flow per view with no linear first layer, so there is no per-view
    basis ``V``: the result carries ``u``, ``latents``, ``reconstructions``,
    ``convergence`` and ``random_state`` but no ``v`` or rank diagnostics, and
    the method is not part of the `pysimlr.api` basis contract. Use
    `flow_simr_v` for the flow model with an interpretable linear entry.

    Parameters are those of `flow_simr_v` without the first-layer settings.
    ``iterations`` is the budget in epochs; ``random_state`` seeds the whole
    fit on a private RNG stream (see `lend_simr`).
    """
    return _fit_flow(
        "flow_simr",
        lambda dims: FlowSiMRModel(
            dims, k, num_layers=num_layers, hidden_dim=hidden_dim,
            mixing_algorithm=mixing_algorithm, scale_bound=scale_bound,
            topology=topology, path_graph=path_graph, prune_threshold=prune_threshold,
            dynamic_weights=dynamic_weights, mai_metric=mai_metric,
            dynamic_weights_start=dynamic_weights_start, use_rank_mai=use_rank_mai,
            force_fallback=force_fallback),
        data_matrices, k, iterations=iterations, batch_size=batch_size,
        learning_rate=learning_rate, weight_decay=weight_decay, sim_weight=sim_weight,
        warmup_epochs=warmup_epochs, energy_type=energy_type, optimizer_type=optimizer_type,
        device=device, verbose=verbose, beta=beta, gamma=gamma,
        dynamic_weights_start=dynamic_weights_start, stabilization_start_epoch=None,
        stabilization_ramp_epochs=None, random_state=random_state, tol=tol, patience=patience,
        collapse_threshold=collapse_threshold)


class FlowSiMRVModel(_MultiViewModelMixin, nn.Module):
    """
    Normalizing Flow SiMR Model with Pre-pended Linear Encoder Matrix V (Flow-SiMR-V).
    """
    def __init__(self, input_dims: List[int], latent_dim: int, num_layers: int = 4, 
                 hidden_dim: int = 64, mixing_algorithm: str = DEFAULT_MIXING_ALGORITHM, scale_bound: float = 2.0,
                 nsa_w: float = DEFAULT_NSA_W, positivity: str = 'either', 
                 sparseness: Optional[Union[float, List[Optional[float]]]] = 0.5, use_nsa: bool = True,
                 topology: str = DEFAULT_TOPOLOGY, path_graph: Optional[Dict[int, List[int]]] = None,
                 prune_threshold: Optional[float] = None,
                 dynamic_weights: bool = False, mai_metric: str = "procrustes_r2",
                 dynamic_weights_start: Optional[int] = None, use_rank_mai: bool = False,
                 first_layer_mode: str = "scheduled", nsa_iterations: int = 1,
                 force_fallback: bool = False):
        super().__init__()
        self._init_consensus_state(len(input_dims), latent_dim, mixing_algorithm, topology, path_graph,
                                    prune_threshold, dynamic_weights, mai_metric, use_rank_mai, dynamic_weights_start)
        from .positivity import resolve_deep_positivity
        positivity = resolve_deep_positivity(positivity)
        self.input_dims = input_dims
        self.latent_dim = latent_dim
        self.is_flow = True
        self.first_layer_mode = first_layer_mode
        self.nsa_iterations = nsa_iterations
        self.force_fallback = force_fallback
        
        self.linear_encoders = _linear_encoders(input_dims, latent_dim, nsa_w, positivity, sparseness,
                                                use_nsa, first_layer_mode, nsa_iterations)
        
        # flows now operate on the low-dimensional projected space of size latent_dim (K)
        self.flows = nn.ModuleList([
            NormalizingFlow(latent_dim, num_layers, hidden_dim, scale_bound, force_fallback=force_fallback) 
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
        
        
    def encode_first_layer(self, x_list: List[torch.Tensor], use_projected: Optional[bool] = None) -> List[torch.Tensor]:
        return [enc.encode_first_layer(x, use_projected=use_projected) for enc, x in zip(self.linear_encoders, x_list)]
        
    def retract_linear_encoders(self) -> None:
        """Retract the linear encoders onto the feasible set.

        Prefers ``NSAFlowLinear.project_()``, the backend's in-place no-grad
        prox, which is the API provided for proximal-gradient training. The
        fallback copies ``enc.v`` into the raw weight, which is the same idea
        done by hand.

        Why the preference matters: the hand-rolled path writes the *projected*
        basis into the weight that the next forward pass will project again, so
        the graph differentiates the projection at a point already on the
        constraint boundary. Measured on a 2-view flow, one Adam step followed
        by that retraction was enough to make
        ``linear_encoders.0.nsa_linear.weight`` produce a non-finite gradient
        while every parameter and the loss itself stayed finite -- after which
        119 of 120 optimizer steps were skipped by the gradient guard and the
        model trained on a single update. ``project_`` keeps the projection out
        of the graph entirely.
        """
        with torch.no_grad():
            for enc in self.linear_encoders:
                nsa = getattr(enc, 'nsa_linear', None)
                if nsa is not None and hasattr(nsa, 'project_'):
                    nsa.project_()
                    continue
                target_v = enc.v.detach()
                if getattr(enc, 'v_raw', None) is not None:
                    enc.v_raw.data.copy_(target_v)
                elif nsa is not None and hasattr(nsa, 'weight'):
                    nsa.weight.data.copy_(target_v.t())

    def forward(self, x_list: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        # 1. Linear projection layer
        projected_zs = [enc(x) for enc, x in zip(self.linear_encoders, x_list)]
        
        # 2. Flow encoders wrapper on projected zs
        latents = [enc(z) for enc, z in zip(self.encoders, projected_zs)]
        if self.training:
            # Evaluate the flow NLL term on detached first-layer projections so
            # that the Gaussian prior + log|det J_flow| regularizes the flow
            # bijection without driving unit-column V_m (which has no log|det|
            # term) into low-variance noise directions.
            for enc, z in zip(self.encoders, projected_zs):
                z_det, ld_det = enc.flow(z.detach())
                enc.last_z = z_det
                enc.last_log_det = ld_det
        
        # 3. Consensus mixing
        u_shared = self._training_consensus(latents, self.latent_dim)
            
        # 4. Decoders: invert normalizing flows, then project back using V_m.t()
        reconstructions = []
        for i, (flow, enc) in enumerate(zip(self.flows, self.linear_encoders)):
            ui = u_shared[i] if isinstance(u_shared, list) else u_shared
            z_recon = flow.inverse(ui)
            v = enc.active_training_basis() if self.training else enc.v
            x_recon = z_recon @ v.t()
            reconstructions.append(x_recon)
            
        return latents, reconstructions, u_shared
        
    def initialize_weights(self, data_matrices: List[torch.Tensor], **init_kwargs):
        """Seed each encoder's basis.

        Delegates to `initialize_deep_encoders`, so any `INITIALIZATION_TYPES`
        strategy is available here, not only PCA -- see `initialize_simlr`'s
        docstring for what each does.
        """
        from .simlr import initialize_deep_encoders
        initialize_deep_encoders(self.linear_encoders, data_matrices, self.latent_dim, **init_kwargs)

def flow_simr_v(data_matrices: List[Union[torch.Tensor, np.ndarray]],
                k: int,
                iterations: int = DEFAULT_DEEP_ITERATIONS,
                batch_size: int = 64,
                learning_rate: float = 5e-4,
                weight_decay: float = 1e-4,
                sim_weight: float = 1.0,
                warmup_epochs: Optional[int] = None,
                num_layers: int = 4,
                hidden_dim: int = 64,
                energy_type: str = DEFAULT_ENERGY_TYPE,
                mixing_algorithm: str = DEFAULT_MIXING_ALGORITHM,
                topology: str = DEFAULT_TOPOLOGY,
                path_graph: Optional[Dict[int, List[int]]] = None,
                prune_threshold: Optional[float] = None,
                optimizer_type: str = 'adam',
                device: Optional[str] = None,
                verbose: bool = False,
                beta: float = 0.05,
                scale_bound: float = 2.0,
                gamma: float = 2.0,
                nsa_w: float = DEFAULT_NSA_W,
                positivity: str = 'either',
                sparseness: Optional[Union[float, List[Optional[float]]]] = 0.5,
                use_nsa: bool = True,
                dynamic_weights: bool = False,
                mai_metric: str = 'procrustes_r2',
                dynamic_weights_start: Optional[int] = None,
                use_rank_mai: bool = False,
                first_layer_mode: str = 'scheduled',
                nsa_iterations: int = 1,
                stabilization_start_epoch: Optional[int] = None,
                stabilization_ramp_epochs: Optional[int] = None,
                force_fallback: bool = False,
                initialization_type: str = 'pca',
                random_state: Optional[int] = None,
                domain_matrices: Optional[List[Optional[torch.Tensor]]] = None,
                n_candidates: int = 5,
                perturbation_scale: float = 0.05,
                varimax: Optional[bool] = None,
                tol: float = DEFAULT_TOL,
                patience: int = 10,
              collapse_threshold: float = DEFAULT_COLLAPSE_THRESHOLD) -> Dict[str, Any]:
    """
    Flow-SiMR-V: a linear, NSA-Flow-projected first layer ``V_m`` per view
    followed by a RealNVP flow.

    Parameters
    ----------
    iterations : int, default=150
        Training budget in epochs.
    positivity : str, default="either"
        Sign constraint on the linear encoder `V_m`. Was `"positive"`;
        changed because that silently forced every encoder non-negative even
        on standardized (signed, zero-mean) input. Measured on two real
        datasets (`scripts/real_data_method_comparison.py`, epochs=60, k=3):
        switching `"positive"` -> `"either"` moved a downstream random-forest's
        accuracy/R^2 from 0.7133 -> 0.7800 on mfeat and 0.2794 -> 0.2928 on
        Diabetes. Pass `"positive"` when the input is genuinely non-negative.
        `"convex"` puts each column of `V_m` on the probability simplex inside
        training (the shared first layer, `deep.LENDNSAEncoder`);
        `"linear"` / `"conical"` alias `"either"` / `"positive"`.
    sparseness : float, list of float or None, default=0.5
        Column-wise quantile sparsification target of `V_m`.
    initialization_type : str, default="pca"
        One of `INITIALIZATION_TYPES` (`pysimlr.simlr`); see `initialize_simlr`.
    random_state : int, optional
        Seed for the whole fit, run on a private RNG stream; the caller's
        global RNG state is restored afterwards. ``None`` draws one seed from
        the global stream; the seed used is ``result["random_state"]``.
    domain_matrices, n_candidates, perturbation_scale :
        Settings of the ``"domain"``, ``"best_of_n"`` and ``"perturbed_pca"``
        initializers.
    varimax : bool or None, default=None
        Orthogonal Varimax at linear-encoder initialization only (see
        `pysimlr.simlr.resolve_varimax`). The trained ``V`` is never rotated
        afterwards: the RealNVP couplings act coordinate-wise on ``X_m V_m``
        and are not O(k)-equivariant.
    tol, patience : early-stopping rule (`utils.ConvergenceMonitor`). Both
        were previously accepted and ignored here.

    The remaining parameters are the flow architecture (`num_layers`,
    `hidden_dim`, `scale_bound`, `force_fallback`), the loss weights (`beta`
    on the flow NLL, `gamma` on the MSE reconstruction, `sim_weight`), and the
    consensus settings shared with `simlr`.
    """
    return _fit_flow(
        "flow_simr_v",
        lambda dims: FlowSiMRVModel(
            dims, k, num_layers=num_layers, hidden_dim=hidden_dim,
            mixing_algorithm=mixing_algorithm, scale_bound=scale_bound,
            nsa_w=nsa_w, positivity=positivity, sparseness=sparseness,
            use_nsa=use_nsa, topology=topology, path_graph=path_graph,
            prune_threshold=prune_threshold, dynamic_weights=dynamic_weights,
            mai_metric=mai_metric, dynamic_weights_start=dynamic_weights_start,
            use_rank_mai=use_rank_mai, first_layer_mode=first_layer_mode,
            nsa_iterations=nsa_iterations,
            force_fallback=force_fallback),
        data_matrices, k, iterations=iterations, batch_size=batch_size,
        learning_rate=learning_rate, weight_decay=weight_decay, sim_weight=sim_weight,
        warmup_epochs=warmup_epochs, energy_type=energy_type, optimizer_type=optimizer_type,
        device=device, verbose=verbose, beta=beta, gamma=gamma,
        dynamic_weights_start=dynamic_weights_start,
        stabilization_start_epoch=stabilization_start_epoch,
        stabilization_ramp_epochs=stabilization_ramp_epochs, random_state=random_state,
        tol=tol, patience=patience, positivity=positivity, nsa_w=nsa_w, sparseness=sparseness,
        init_kwargs=dict(initialization_type=initialization_type,
                         domain_matrices=domain_matrices, n_candidates=n_candidates,
                         perturbation_scale=perturbation_scale, varimax=varimax),
        collapse_threshold=collapse_threshold)
