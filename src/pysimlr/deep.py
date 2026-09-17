import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
import math
class larslow(optim.Optimizer):
    """Layer-wise Adaptive Rate Scaling (LARS) optimizer for deep SiMLR."""
    def __init__(self, params, lr, weight_decay=1e-4, momentum=0.9, trust_coefficient=0.001):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, trust_coefficient=trust_coefficient)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                v_norm = torch.norm(p); g_norm = torch.norm(grad)
                # The fallback must also carry `trust_coefficient`; returning a
                # bare 1.0 gave any zero-initialised parameter (v_norm == 0,
                # e.g. a LayerNorm bias) 1/trust_coefficient = 1000x the
                # intended learning rate on that step.
                if v_norm > 0 and g_norm > 0:
                    trust_ratio = group['trust_coefficient'] * v_norm / (
                        g_norm + group['weight_decay'] * v_norm + 1e-10)
                else:
                    trust_ratio = group['trust_coefficient']
                state = self.state[p]
                if 'momentum_buffer' not in state: state['momentum_buffer'] = torch.zeros_like(p)
                buf = state['momentum_buffer']
                buf.mul_(group['momentum']).add_(grad + group['weight_decay'] * p, alpha=trust_ratio * group['lr'])
                p.sub_(buf)
        return loss

def _get_optimizer(model, optimizer_type, learning_rate, weight_decay):
    if optimizer_type == "larslow":
        return larslow(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == "lars":
        try:
            import torch_optimizer as t_opt
            return t_opt.LARS(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        except ImportError:
            return larslow(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

from .simlr import ba_svd
from .consensus import compute_shared_consensus
from .utils import preprocess_data, invariant_orthogonality_defect, safe_svd
from .interpretability import build_first_layer_contract, build_interpretability_report

from .nsa_backend import load_nsa_backend

def _aggregate_shared_consensus(model, latents):
    """
    Single (n, k) shared embedding for a model whose topology yields one
    consensus per modality.

    With ``topology="loo"`` (the default) or ``"graph"``, each modality is
    aligned to a *different* consensus, so ``forward`` returns a list. Users
    and downstream code still need one shared embedding, which is the star
    consensus taken over all modalities at once.

    This is computed explicitly here rather than being inherited from a
    prediction-time short-circuit inside `compute_shared_consensus`. That
    short-circuit silently returned the star consensus whenever an anchor was
    present, so a leave-one-out model's reported ``u`` came from a different
    estimator than the one it was trained against -- and only at eval time.

    Parameters
    ----------
    model : torch.nn.Module
        A fitted deep SiMR model.
    latents : List[torch.Tensor]
        Per-modality latent scores, as returned by ``model.forward``.

    Returns
    -------
    torch.Tensor
        The star consensus of shape (n_samples, k).
    """
    k = getattr(model, "latent_dim", None) or getattr(model, "shared_dim", None)
    return compute_shared_consensus(
        latents,
        mixing_algorithm=model.mixing_algorithm,
        k=k,
        training=False,
        anchor=getattr(model, "consensus_anchor", None),
        topology="star",
        prune_threshold=getattr(model, "prune_threshold", None),
        modality_weights=(model.modality_weights
                          if getattr(model, "dynamic_weights", False) else None),
    )


def _svd_project_columns(u: torch.Tensor) -> torch.Tensor:
    """Project towards the Stiefel manifold using SVD."""
    with torch.no_grad():
        try:
            u_svd, _, vh_svd = safe_svd(u, full_matrices=False)
            return u_svd @ vh_svd
        except Exception:
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


def _construct_nsa(factory, modern_kwargs: dict, legacy_kwargs: dict):
    """
    Build an NSA-Flow module, tolerating the backend's keyword rename.

    The backend renamed ``w_retract`` to ``w`` and ``apply_nonneg`` (a string)
    to ``nonneg`` (a boolean), and dropped ``retraction_type`` entirely.
    Pinning either spelling makes pysimlr fail against half the released
    versions, so the modern signature is tried first and the legacy one second.

    Parameters
    ----------
    factory : type
        The backend class to instantiate.
    modern_kwargs, legacy_kwargs : dict
        Keyword sets for the current and previous signatures.

    Returns
    -------
    object
        The constructed module.

    Raises
    ------
    TypeError
        If neither keyword set is accepted, since that means the backend has
        changed in a way this shim does not cover.
    """
    try:
        return factory(**modern_kwargs)
    except TypeError:
        return factory(**legacy_kwargs)


def _orient_first_layer(weight, input_dim: int, latent_dim: int):
    """
    Return ``weight`` oriented as the ``(features, latent)`` basis pysimlr uses.

    The NSA-Flow backend's linear module follows the ``torch.nn.Linear``
    convention and stores its weight as ``(out_features, in_features)``, i.e.
    ``(latent, features)``. Everything in pysimlr treats the first-layer basis
    as ``(features, latent)`` -- ``x @ v`` , column-wise normalization,
    ``v.copy_(v_init)``. Transposing here rather than at each use site keeps a
    single place responsible for the convention.

    A transposed view shares storage with the parameter, so gradients flow back
    and in-place writes (the ``v_raw.copy_(...)`` initialization paths) reach
    the backend parameter.

    Encoders only use the backend when ``input_dim > latent_dim``, so the two
    orientations are never the same shape and the choice is unambiguous.
    """
    if weight.shape == (input_dim, latent_dim):
        return weight
    if weight.shape == (latent_dim, input_dim):
        return weight.transpose(0, 1)
    raise ValueError(
        f"NSA-Flow weight has shape {tuple(weight.shape)}, which matches "
        f"neither ({input_dim}, {latent_dim}) nor ({latent_dim}, {input_dim})."
    )


def _nsa_raw_parameter(module):
    """
    Return the backend module's trainable weight under either of its names.

    The backend renamed ``weight_raw`` to ``weight``; both spellings are
    accepted so pysimlr works against released versions on either side of the
    rename.
    """
    for name in ("weight", "weight_raw"):
        candidate = getattr(module, name, None)
        if isinstance(candidate, torch.Tensor):
            return candidate
    raise AttributeError(
        f"{type(module).__name__} exposes neither 'weight' nor 'weight_raw'; "
        "the NSA-Flow backend interface has changed."
    )


def _nsa_effective_weight(module):
    """
    Return the backend module's retracted weight under either of its names.

    ``get_manifold_weight`` was renamed to ``effective_weight``.
    """
    for name in ("effective_weight", "get_manifold_weight"):
        accessor = getattr(module, name, None)
        if callable(accessor):
            return accessor()
    raise AttributeError(
        f"{type(module).__name__} exposes neither 'effective_weight()' nor "
        "'get_manifold_weight()'; the NSA-Flow backend interface has changed."
    )


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
    positivity : str, default="positive"
        Positivity constraint on weights: 
        - 'either': Unconstrained (standard orthogonal basis).
        - 'positive': Strictly non-negative via clamping (legacy behavior).
        - 'softplus': Smoothly non-negative via Softplus (advanced behavior).
        - 'hard': Strictly non-negative via clamping.
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
    """
    def __init__(self, input_dim: int, latent_dim: int, nsa_w: float = 0.1, 
                 positivity: str = "positive", sparseness_quantile: float = 0.0,
                 soft_thresholding: bool = False, use_nsa: bool = True,
                 first_layer_mode: str = "scheduled", nsa_iterations: int = 1,
                 retraction_type: str = "soft_polar"):
        super().__init__()
        if positivity is True or (isinstance(positivity, str) and positivity.lower() == 'true'):
            positivity = 'positive'
        elif positivity is False or (isinstance(positivity, str) and positivity.lower() == 'false'):
            positivity = 'either'
        self.positivity = positivity
        self.sparseness_quantile = sparseness_quantile
        self.soft_thresholding = soft_thresholding
        self.use_nsa = use_nsa
        if first_layer_mode not in {"raw", "projected", "scheduled"}:
            raise ValueError(f"Unsupported first_layer_mode: {first_layer_mode}")
        self.first_layer_mode = first_layer_mode
        self.topology = "loo"
        self.nsa_iterations = nsa_iterations
        # Absent a driven schedule, train on the same basis inference uses.
        #
        # `first_layer_mode="scheduled"` blends the raw and projected bases by
        # `projection_alpha`, which the training loops set every epoch via
        # `set_projection_schedule`. A caller who uses the encoder directly
        # never advances it, so an initial 0.0 meant training ran on the raw
        # signed basis while `eval()` switched to the projected one. Under the
        # default `positivity="positive"` those differ by a sign clamp, and a
        # k=1 encoder fitted to correlation 0.99 on its training basis scored
        # 0.51 at inference. The trainers still start their own ramp at 0.0, so
        # this only changes the undriven case.
        self.projection_alpha = 1.0
        self.stabilization_epoch = 0
        self.stabilization_ramp_epochs = 1
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.is_low_dim = (input_dim <= latent_dim)
        if self.is_low_dim:
            # A basis cannot be near-orthogonal with more columns than rows, so
            # the retraction backend is bypassed entirely in that regime. This
            # also guarantees input_dim > latent_dim wherever the backend runs,
            # which is what makes `_orient_first_layer` unambiguous.
            self.use_nsa = False
        
        # Determine internal NSA constraint mode.
        #
        # The backend's layers and its solver spell non-negativity differently,
        # and they do not agree on what `True` means. `nsa_flow(V, nonneg=True)`
        # applies a hard non-negativity constraint, but `NSAFlowLinear` maps
        # `nonneg=True` to *softplus*: `_nonneg(W, True)` returns
        # `F.softplus(W)`. Passing the boolean here therefore turned a sparse
        # basis into a dense, near-uniform one -- softplus sends every zero to
        # log(2) = 0.693 -- taking the encoder's effective basis from a
        # normalized Stiefel defect of 0.10 to 2.38 with every one of its
        # entries above 0.689. The layer wants the string.
        #
        # 'softplus' positivity is excluded deliberately: the `v` property
        # applies its own `softplus(v - 4)`, so asking the layer for softplus
        # too would apply it twice.
        # The layer's own `w` is a retraction weight like any other, so it
        # takes the same bound: at w=1 the fidelity term drops out and every
        # scaled Stiefel matrix is optimal, so nothing selects among them.
        # `_nsa_retract` clamps its own calls, but the layer is constructed
        # with this value directly and would otherwise bypass the cap --
        # `lend_simr(nsa_w=1.0)` reached the degenerate case.
        from .sparsification import _clamp_retraction_weight
        self.nsa_w = _clamp_retraction_weight(nsa_w)
        nsa_w = self.nsa_w
        nsa_nonneg = positivity in {'positive', 'hard', 'nonnegative', 'nonneg'}
        nsa_layer_nonneg = 'hard' if nsa_nonneg else None
        
        self.nsa_linear = None
        self.nsa_layer = None
        
        nsa = load_nsa_backend() if use_nsa else None
        if nsa is not None and use_nsa:
            # Prefer the new NSAFlowLinear for parameter-squeeze
            if hasattr(nsa, 'NSAFlowLinear'):
                self.nsa_linear = _construct_nsa(
                    nsa.NSAFlowLinear,
                    dict(in_features=input_dim, out_features=latent_dim,
                         bias=False, w=nsa_w, nonneg=nsa_layer_nonneg),
                    dict(in_features=input_dim, out_features=latent_dim,
                         bias=False, w_retract=nsa_w,
                         retraction_type=retraction_type,
                         apply_nonneg='hard' if nsa_nonneg else 'none'),
                )
            else:
                # Fallback to NSAFlowLayer (activation-squeeze) applied to weights
                self.v_raw = nn.Parameter(torch.randn(input_dim, latent_dim) * 0.01)
                self.nsa_layer = _construct_nsa(
                    nsa.NSAFlowLayer,
                    dict(w=nsa_w, nonneg=nsa_layer_nonneg),
                    dict(k=latent_dim, w_retract=nsa_w,
                         retraction_type=retraction_type,
                         apply_nonneg='hard' if nsa_nonneg else 'none',
                         residual=False, use_transform=False),
                )
        else:
            self.v_raw = nn.Parameter(torch.randn(input_dim, latent_dim) * 0.01)

    @property
    def v_raw(self):
        """
        The unretracted ``(features, latent)`` first-layer basis.

        When the NSA-Flow backend owns the parameter it lives inside
        ``nsa_linear`` under the ``torch.nn.Linear`` orientation, so it is
        transposed into pysimlr's convention on access. Aliasing it as a second
        registered parameter (the previous approach) both duplicated it in the
        state dict and handed out a ``(latent, features)`` tensor to call sites
        that index it as ``(features, latent)``.

        Otherwise the encoder owns the parameter directly, registered under
        this same name by ``nn.Module.__setattr__``.
        """
        own = self._parameters.get("v_raw")
        if own is not None:
            return own
        backend = getattr(self, "nsa_linear", None)
        if backend is not None:
            return _orient_first_layer(
                _nsa_raw_parameter(backend), self.input_dim, self.latent_dim)
        raise AttributeError("v_raw has not been initialized")

    @property
    def v(self):
        if getattr(self, 'is_low_dim', False):
            v_out = self.v_raw
            if self.positivity in {'positive', 'hard', 'softplus'}:
                magnitudes = torch.abs(v_out)
                if self.positivity == 'softplus': magnitudes = torch.nn.functional.softplus(v_out - 4.0)
                # Shift by the global max before exponentiating. The shift is a
                # single multiplicative constant on `scores`, which cancels in
                # the first Sinkhorn normalization, so this is exact rather than
                # an approximation.
                logits = v_out * 20.0
                scores = torch.exp(logits - logits.max().detach())
                for _ in range(10):
                    scores = scores / (torch.sum(scores, dim=0, keepdim=True) + 1e-8)
                    scores = scores / (torch.sum(scores, dim=1, keepdim=True) + 1e-8)
                v_out = scores * magnitudes
            v_out = torch.nn.functional.normalize(v_out, p=2, dim=0)
        else:
            if self.nsa_linear is not None:
                # Routing this through the full solver instead
                # (`_nsa_retract(self.v_raw, ...)`) was tried and dropped: it
                # produces a better-conditioned basis in isolation (defect
                # 0.08 against 0.09) but leaves NED's latents just as
                # correlated, costs 26x per access, and moved real-data
                # accuracy by +0.005 on average -- within seed noise.
                v_out = _orient_first_layer(
                    _nsa_effective_weight(self.nsa_linear),
                    self.input_dim, self.latent_dim)
                if self.nsa_iterations > 1:
                    try: v_out = _svd_project_columns(v_out)
                    except Exception: pass
            elif self.nsa_layer is not None:
                v_out = self.nsa_layer(self.v_raw)
                if self.nsa_iterations > 1:
                    try: v_out = _svd_project_columns(v_out)
                    except Exception: pass
            else:
                try: v_out = _svd_project_columns(self.v_raw)
                except Exception: v_out = torch.nn.functional.normalize(self.v_raw, p=2, dim=0)
            if self.positivity in {'positive', 'hard'}: v_out = torch.clamp(v_out, min=0.0)
            elif self.positivity == 'softplus': v_out = torch.nn.functional.softplus(v_out - 4.0)
        if torch.isnan(v_out).any(): v_out = torch.nan_to_num(v_out, nan=0.0)
        if self.sparseness_quantile > 0:
            v_abs = torch.abs(v_out) if self.positivity == "either" else v_out
            q_vals = torch.quantile(v_abs, self.sparseness_quantile, dim=0, keepdim=True)
            if self.soft_thresholding: v_out = torch.sign(v_out) * torch.clamp(v_abs - q_vals, min=0.0)
            else: v_out = v_out * (v_abs >= q_vals).float()
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
    """
    def __init__(self, input_dims: List[int], latent_dim: int, hidden_dims: List[int] = [128, 64], 
                 dropout: float = 0.1, nsa_w: float = 0.1, positivity: str = "positive", 
                 sparseness_quantile: Union[float, List[float]] = 0.0, mixing_algorithm: str = "newton",
                 topology: str = "loo", path_graph: Optional[Dict[int, List[int]]] = None, prune_threshold: Optional[float] = None, dynamic_weights: bool = False, mai_metric: str = "procrustes_r2",
                 use_nsa: bool = True, first_layer_mode: str = "scheduled", nsa_iterations: int = 1, use_rank_mai: bool = False,
                 retraction_type: str = "soft_polar"):
        super().__init__()
        if isinstance(sparseness_quantile, (float, int)):
            sparseness_quantile = [float(sparseness_quantile)] * len(input_dims)
        self.encoders = nn.ModuleList([LENDNSAEncoder(dim, latent_dim, nsa_w, positivity, sq, use_nsa=use_nsa, first_layer_mode=first_layer_mode, nsa_iterations=nsa_iterations, retraction_type=retraction_type) for dim, sq in zip(input_dims, sparseness_quantile)])
        self.decoders = nn.ModuleList([ModalityDecoder(latent_dim, dim, hidden_dims, dropout) for dim in input_dims])
        self.mixing_algorithm, self.latent_dim = mixing_algorithm, latent_dim
        self.topology = topology
        self.path_graph = path_graph
        self.path_graph = path_graph
        self.prune_threshold = prune_threshold
        self.dynamic_weights = dynamic_weights
        self.mai_metric = mai_metric
        self.use_rank_mai = use_rank_mai
        self.register_buffer("mai", torch.ones(len(input_dims)) / len(input_dims))
        self.register_buffer("modality_weights", torch.ones(len(input_dims)) / len(input_dims))
        self.register_buffer("consensus_anchor", torch.zeros(len(input_dims) * latent_dim, latent_dim))
    def initialize_v(self, data_matrices: List[torch.Tensor], k: int):
        """
        Seed each encoder's first-layer basis.

        `initial_basis_for_view` fits a non-negative basis from the data when
        the encoder rectifies (its `v` property clamps negatives), so the first
        forward pass does not see ``clamp(V_pca)``; otherwise it returns the
        signed PCA loadings with column signs resolved. Sharing it with
        `initialize_simlr` keeps the linear and deep paths on the same start.
        """
        from .simlr import initial_basis_for_view
        with torch.no_grad():
            for i, x in enumerate(data_matrices):
                v = initial_basis_for_view(
                    x, k, positivity=self.encoders[i].positivity)
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

    def update_mai(self, latents: List[torch.Tensor], epoch: int, total_epochs: int):
        if not getattr(self, "dynamic_weights", False): return
        with torch.no_grad():
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
                        metric = getattr(self, "mai_metric", "procrustes_r2_sharp")
                        
                        if metric == "procrustes_r2" or metric == "procrustes_r2_sharp":
                            omega = u_svd @ vh_svd
                            aligned = Z @ omega
                            r2 = max(0.0, 1.0 - (torch.norm(aligned - u_loo, p='fro')**2 / (torch.norm(u_loo, p='fro')**2 + 1e-8)).item())
                            if metric == "procrustes_r2_sharp":
                                # Spectral Sharpness of the modality latent itself
                                # Isotropic noise has flat singular values; structured signal is top-heavy.
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
                    except Exception:
                        mais.append(0.0)
                else:
                    mais.append(0.0)
            
            mai_tensor = torch.tensor(mais, device=self.mai.device)
            # EMA for stability
            self.mai.copy_(0.9 * self.mai + 0.1 * mai_tensor)
            
            # Gating Logic: Only squashing severe outliers
            # Use mean similarity as the soft threshold center
            center = self.mai.mean()
            progress = min(1.0, epoch / 30.0)
            steepness = 5.0 + 10.0 * progress
            gate = torch.sigmoid(steepness * (self.mai - center))
            
            # Target weights = Proportional MAI * Gating Signal
            raw_w = self.mai * gate
            target_w = raw_w / (raw_w.sum() + 1e-8)
            
            # Temporal transition (from uniform to target) over first 30 epochs
            rho = max(0.0, min(1.0, (epoch - 10) / 20.0))
            uniform_w = torch.ones_like(target_w) / len(target_w)
            self.modality_weights.copy_((1.0 - rho) * uniform_w + rho * target_w)
    def forward(self, x_list: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        latents = self.encode_first_layer(x_list)
        res_u = compute_shared_consensus(latents, mixing_algorithm=self.mixing_algorithm, k=self.latent_dim, training=self.training, anchor=self.consensus_anchor, topology=self.topology, path_graph=getattr(self, "path_graph", None), prune_threshold=self.prune_threshold, modality_weights=self.modality_weights if getattr(self, "dynamic_weights", False) else None)
        if self.training:
            u_shared, new_anchor = res_u
            # Update anchor using EMA for stability
            if new_anchor is not None: self.consensus_anchor.copy_(0.9 * self.consensus_anchor + 0.1 * new_anchor)
        else:
            u_shared = res_u
        return latents, [dec(u_shared[i] if isinstance(u_shared, list) else u_shared) for i, dec in enumerate(self.decoders)], u_shared

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
    """
    def __init__(self, input_dims: List[int], latent_dim: int, hidden_dims: List[int] = [128, 64], 
                 dropout: float = 0.1, nsa_w: float = 0.1, positivity: str = "positive", 
                 sparseness_quantile: Union[float, List[float]] = 0.0, mixing_algorithm: str = "newton",
                 topology: str = "loo", path_graph: Optional[Dict[int, List[int]]] = None, prune_threshold: Optional[float] = None, dynamic_weights: bool = False, mai_metric: str = "procrustes_r2",
                 use_nsa: bool = True, first_layer_mode: str = "scheduled", nsa_iterations: int = 1, use_rank_mai: bool = False,
                 retraction_type: str = "soft_polar"):
        super().__init__()
        if isinstance(sparseness_quantile, (float, int)):
            sparseness_quantile = [float(sparseness_quantile)] * len(input_dims)
        self.linear_encoders = nn.ModuleList([LENDNSAEncoder(dim, latent_dim, nsa_w, positivity, sq, use_nsa=use_nsa, first_layer_mode=first_layer_mode, nsa_iterations=nsa_iterations, retraction_type=retraction_type) for dim, sq in zip(input_dims, sparseness_quantile)])
        self.nonlinear_heads = nn.ModuleList([ModalityDecoder(latent_dim, latent_dim, hidden_dims, dropout) for _ in input_dims])
        self.decoders = nn.ModuleList([ModalityDecoder(latent_dim, dim, hidden_dims, dropout) for dim in input_dims])
        self.mixing_algorithm, self.latent_dim = mixing_algorithm, latent_dim
        self.topology = topology
        self.path_graph = path_graph
        self.path_graph = path_graph
        self.prune_threshold = prune_threshold
        self.dynamic_weights = dynamic_weights
        self.mai_metric = mai_metric
        self.use_rank_mai = use_rank_mai
        self.register_buffer("mai", torch.ones(len(input_dims)) / len(input_dims))
        self.register_buffer("modality_weights", torch.ones(len(input_dims)) / len(input_dims))
        self.register_buffer("consensus_anchor", torch.zeros(len(input_dims) * latent_dim, latent_dim))
    def initialize_v(self, data_matrices: List[torch.Tensor], k: int):
        """
        Seed each encoder's first-layer basis.

        `initial_basis_for_view` fits a non-negative basis from the data when
        the encoder rectifies (its `v` property clamps negatives), so the first
        forward pass does not see ``clamp(V_pca)``; otherwise it returns the
        signed PCA loadings with column signs resolved. Sharing it with
        `initialize_simlr` keeps the linear and deep paths on the same start.
        """
        from .simlr import initial_basis_for_view
        with torch.no_grad():
            for i, x in enumerate(data_matrices):
                v = initial_basis_for_view(
                    x, k, positivity=self.linear_encoders[i].positivity)
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

    def update_mai(self, latents: List[torch.Tensor], epoch: int, total_epochs: int):
        if not getattr(self, "dynamic_weights", False): return
        with torch.no_grad():
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
                        metric = getattr(self, "mai_metric", "procrustes_r2_sharp")
                        
                        if metric == "procrustes_r2" or metric == "procrustes_r2_sharp":
                            omega = u_svd @ vh_svd
                            aligned = Z @ omega
                            r2 = max(0.0, 1.0 - (torch.norm(aligned - u_loo, p='fro')**2 / (torch.norm(u_loo, p='fro')**2 + 1e-8)).item())
                            if metric == "procrustes_r2_sharp":
                                # Spectral Sharpness of the modality latent itself
                                # Isotropic noise has flat singular values; structured signal is top-heavy.
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
                    except Exception:
                        mais.append(0.0)
                else:
                    mais.append(0.0)
            
            mai_tensor = torch.tensor(mais, device=self.mai.device)
            # EMA for stability
            self.mai.copy_(0.9 * self.mai + 0.1 * mai_tensor)
            
            # Gating Logic: Only squashing severe outliers
            # Use mean similarity as the soft threshold center
            center = self.mai.mean()
            progress = min(1.0, epoch / 30.0)
            steepness = 5.0 + 10.0 * progress
            gate = torch.sigmoid(steepness * (self.mai - center))
            
            # Target weights = Proportional MAI * Gating Signal
            raw_w = self.mai * gate
            target_w = raw_w / (raw_w.sum() + 1e-8)
            
            # Temporal transition (from uniform to target) over first 30 epochs
            rho = max(0.0, min(1.0, (epoch - 10) / 20.0))
            uniform_w = torch.ones_like(target_w) / len(target_w)
            self.modality_weights.copy_((1.0 - rho) * uniform_w + rho * target_w)
    def forward(self, x_list: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        first_layer_scores = self.encode_first_layer(x_list)
        latents = [head(z0) for head, z0 in zip(self.nonlinear_heads, first_layer_scores)]
        res_u = compute_shared_consensus(latents, mixing_algorithm=self.mixing_algorithm, k=self.latent_dim, training=self.training, anchor=self.consensus_anchor, topology=self.topology, path_graph=getattr(self, "path_graph", None), prune_threshold=self.prune_threshold, modality_weights=self.modality_weights if getattr(self, "dynamic_weights", False) else None)
        if self.training:
            u_shared, new_anchor = res_u
            # Update anchor using EMA for stability
            if new_anchor is not None: self.consensus_anchor.copy_(0.9 * self.consensus_anchor + 0.1 * new_anchor)
        else:
            u_shared = res_u
        return latents, [dec(u_shared[i] if isinstance(u_shared, list) else u_shared) for i, dec in enumerate(self.decoders)], u_shared

    def transform(self, x_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Unify downstream prediction API: return the shared consensus U as an
        (n_samples, latent_dim) tensor.

        ``forward`` yields one consensus per modality under the "loo" and
        "graph" topologies; this collapses those to the single star consensus,
        so the public transform contract is a tensor regardless of topology.
        """
        self.eval()
        with torch.no_grad():
            latents, _, u = self.forward(x_list)
            if isinstance(u, list):
                return _aggregate_shared_consensus(self, latents)
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
    """
    def __init__(self, input_dims: List[int], shared_latent_dim: int, private_latent_dim: int,
                 hidden_dims: List[int] = [128, 64], dropout: float = 0.1, nsa_w: float = 0.1,
                 positivity: str = "positive", sparseness_quantile: Union[float, List[float]] = 0.0, mixing_algorithm: str = "newton",
                 topology: str = "loo", path_graph: Optional[Dict[int, List[int]]] = None, prune_threshold: Optional[float] = None, dynamic_weights: bool = False, mai_metric: str = "procrustes_r2",
                 use_nsa: bool = True, first_layer_mode: str = "scheduled", nsa_iterations: int = 1, use_rank_mai: bool = False,
                 retraction_type: str = "soft_polar"):
        super().__init__()
        if isinstance(sparseness_quantile, (float, int)):
            sparseness_quantile = [float(sparseness_quantile)] * len(input_dims)
        self.linear_encoders = nn.ModuleList([LENDNSAEncoder(dim, shared_latent_dim, nsa_w, positivity, sq, use_nsa=use_nsa, first_layer_mode=first_layer_mode, nsa_iterations=nsa_iterations, retraction_type=retraction_type) for dim, sq in zip(input_dims, sparseness_quantile)])
        self.shared_heads = nn.ModuleList([ModalityDecoder(shared_latent_dim, shared_latent_dim, hidden_dims, dropout) for _ in input_dims])
        self.register_buffer("consensus_anchor", torch.zeros(len(input_dims) * shared_latent_dim, shared_latent_dim))
        self.private_encoders = nn.ModuleList([ModalityEncoder(dim, private_latent_dim, hidden_dims, dropout) for dim in input_dims])
        self.decoders = nn.ModuleList([ModalityDecoder(shared_latent_dim + private_latent_dim, dim, hidden_dims, dropout) for dim in input_dims])
        self.mixing_algorithm, self.shared_dim = mixing_algorithm, shared_latent_dim
        self.topology = topology
        self.path_graph = path_graph
        self.path_graph = path_graph
        self.prune_threshold = prune_threshold
        self.dynamic_weights = dynamic_weights
        self.mai_metric = mai_metric
        self.use_rank_mai = use_rank_mai
        self.register_buffer("mai", torch.ones(len(input_dims)) / len(input_dims))
        self.register_buffer("modality_weights", torch.ones(len(input_dims)) / len(input_dims))
    def initialize_v(self, data_matrices: List[torch.Tensor], k: int):
        """
        Seed each encoder's first-layer basis.

        `initial_basis_for_view` fits a non-negative basis from the data when
        the encoder rectifies (its `v` property clamps negatives), so the first
        forward pass does not see ``clamp(V_pca)``; otherwise it returns the
        signed PCA loadings with column signs resolved. Sharing it with
        `initialize_simlr` keeps the linear and deep paths on the same start.
        """
        from .simlr import initial_basis_for_view
        with torch.no_grad():
            for i, x in enumerate(data_matrices):
                v = initial_basis_for_view(
                    x, k, positivity=self.linear_encoders[i].positivity)
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

    def update_mai(self, latents: List[torch.Tensor], epoch: int, total_epochs: int):
        if not getattr(self, "dynamic_weights", False): return
        with torch.no_grad():
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
                        metric = getattr(self, "mai_metric", "procrustes_r2_sharp")
                        
                        if metric == "procrustes_r2" or metric == "procrustes_r2_sharp":
                            omega = u_svd @ vh_svd
                            aligned = Z @ omega
                            r2 = max(0.0, 1.0 - (torch.norm(aligned - u_loo, p='fro')**2 / (torch.norm(u_loo, p='fro')**2 + 1e-8)).item())
                            if metric == "procrustes_r2_sharp":
                                # Spectral Sharpness of the modality latent itself
                                # Isotropic noise has flat singular values; structured signal is top-heavy.
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
                    except Exception:
                        mais.append(0.0)
                else:
                    mais.append(0.0)
            
            mai_tensor = torch.tensor(mais, device=self.mai.device)
            # EMA for stability
            self.mai.copy_(0.9 * self.mai + 0.1 * mai_tensor)
            
            # Gating Logic: Only squashing severe outliers
            # Use mean similarity as the soft threshold center
            center = self.mai.mean()
            progress = min(1.0, epoch / 30.0)
            steepness = 5.0 + 10.0 * progress
            gate = torch.sigmoid(steepness * (self.mai - center))
            
            # Target weights = Proportional MAI * Gating Signal
            raw_w = self.mai * gate
            target_w = raw_w / (raw_w.sum() + 1e-8)
            
            # Temporal transition (from uniform to target) over first 30 epochs
            rho = max(0.0, min(1.0, (epoch - 10) / 20.0))
            uniform_w = torch.ones_like(target_w) / len(target_w)
            self.modality_weights.copy_((1.0 - rho) * uniform_w + rho * target_w)
    def forward(self, x_list: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, List[torch.Tensor]]:
        first_layer_scores = self.encode_first_layer(x_list)
        shared_l = [head(z0) for head, z0 in zip(self.shared_heads, first_layer_scores)]
        private_l = [p_enc(x) for p_enc, x in zip(self.private_encoders, x_list)]
        res_u = compute_shared_consensus(shared_l, mixing_algorithm=self.mixing_algorithm, k=self.shared_dim, training=self.training, anchor=self.consensus_anchor, topology=self.topology, path_graph=getattr(self, "path_graph", None), prune_threshold=self.prune_threshold, modality_weights=self.modality_weights if getattr(self, "dynamic_weights", False) else None)
        if self.training:
            u_shared, new_anchor = res_u
            if new_anchor is not None: self.consensus_anchor.copy_(0.9 * self.consensus_anchor + 0.1 * new_anchor)
        else:
            u_shared = res_u
        recons = [dec(torch.cat([u_shared[i] if isinstance(u_shared, list) else u_shared, p], dim=1)) for i, (dec, p) in enumerate(zip(self.decoders, private_l))]
        return shared_l, recons, u_shared, private_l

    def transform(self, x_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Return the shared consensus U as an (n_samples, shared_dim) tensor.

        See :meth:`NEDSiMRModel.transform` -- the "loo"/"graph" topologies give
        one consensus per modality, which is collapsed to the star consensus
        here so the transform contract stays a tensor.
        """
        self.eval()
        with torch.no_grad():
            shared_l, _, u, _ = self.forward(x_list)
            if isinstance(u, list):
                return _aggregate_shared_consensus(self, shared_l)
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
                       u_shared: Union[torch.Tensor, List[torch.Tensor]], 
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
        - "nc" or "normalized_correlation": Negative Procrustes Correlation (tr(U'Z)/||U'Z||_F).
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
    """
    is_loo = isinstance(u_shared, list)
    device = u_shared[0].device if is_loo else u_shared.device
    n = u_shared[0].shape[0] if is_loo else u_shared.shape[0]
    sim_loss = torch.tensor(0.0, device=device)
    
    var_penalty = sum(_variance_penalty(z) for z in latents)
    
    for i, z in enumerate(latents):
        z_c = z - z.mean(dim=0)
        
        # Get target for this modality
        u_target = u_shared[i].detach() if is_loo else u_shared.detach()
        u_target_c = u_target - u_target.mean(dim=0)
        u_std = _safe_std(u_target_c, dim=0, keepdim=True)
        
        if energy_type == "regression":
            z_std = _safe_std(z_c, dim=0, keepdim=True)
            sim_loss += torch.mean(((z_c / z_std) - (u_target_c / u_std))**2)
        elif energy_type == "acc":
            cov = (u_target_c.t() @ z_c) / (n - 1)
            sim_loss -= torch.sum(torch.abs(cov))
        elif energy_type in ["nc", "normalized_correlation"]:
            cross_cov = u_target_c.t() @ z_c
            numerator = torch.trace(cross_cov)
            frobenius_norm = torch.norm(cross_cov, p='fro')
            if frobenius_norm > 1e-8:
                sim_loss -= (numerator / frobenius_norm)
        elif energy_type == "logcosh":
            z_std = _safe_std(z_c, dim=0, keepdim=True)
            z_norm = z_c / z_std
            u_norm = u_target_c / u_std
            s = u_norm.t() @ z_norm / n
            abs_s = torch.abs(s)
            sim_loss -= torch.sum(abs_s - np.log(2.0) + torch.log1p(torch.exp(-2.0 * abs_s)))
    
    if is_loo:
        # For LOO, average the collapse and variance penalties across all leave-one-out consensus spaces
        collapse_loss = torch.tensor(0.0, device=device)
        u_var_penalty = torch.tensor(0.0, device=device)
        for u in u_shared:
            u_c = u - torch.mean(u, dim=0)
            cov_u = (u_c.t() @ u_c) / (n - 1)
            mask = ~torch.eye(u.shape[1], device=device).bool()
            collapse_loss += torch.mean(cov_u[mask]**2)
            u_var_penalty += _variance_penalty(u)
        collapse_loss /= len(u_shared)
        u_var_penalty /= len(u_shared)
        
        u_std_mean = torch.mean(torch.stack([torch.mean(_safe_std(u, dim=0)) for u in u_shared])).item()
        
        # approximate off diag cov for diagnostics
        u_c_first = u_shared[0] - torch.mean(u_shared[0], dim=0)
        cov_u_first = (u_c_first.t() @ u_c_first) / (n - 1)
        u_off_diag_cov = torch.norm(cov_u_first[~torch.eye(u_shared[0].shape[1], device=device).bool()]).item()
    else:
        u_c = u_shared - torch.mean(u_shared, dim=0)
        cov_u = (u_c.t() @ u_c) / (n - 1)
        mask = ~torch.eye(u_shared.shape[1], device=device).bool()
        collapse_loss = torch.mean(cov_u[mask]**2)
        u_var_penalty = _variance_penalty(u_shared)
        u_std_mean = torch.mean(_safe_std(u_shared, dim=0)).item()
        u_off_diag_cov = torch.norm(cov_u[mask]).item()
    
    total_loss = (weights.get("sim", 1.0) * sim_loss + 
                  weights.get("var", 1.0) * var_penalty + 
                  weights.get("collapse", 1.0) * collapse_loss + 
                  weights.get("u_var", 1.0) * u_var_penalty)
    
    diagnostics = {
        "sim_loss": sim_loss.item(),
        "var_penalty": var_penalty.item(),
        "collapse_loss": collapse_loss.item(),
        "u_var_penalty": u_var_penalty.item(),
        "u_std_mean": u_std_mean,
        "u_off_diag_cov": u_off_diag_cov
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
    last_sim_weight = None
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
        # The similarity term switches on at `warmup_epochs`, which discontinuously
        # raises the objective. Comparing post-warmup losses against a best_loss
        # recorded while the term was switched off makes the patience counter
        # increment every epoch, stopping training at warmup+patience regardless
        # of actual convergence. Restart the baseline when the objective changes.
        if current_sim_weight != last_sim_weight:
            best_loss = float('inf')
            patience_counter = 0
        last_sim_weight = current_sim_weight
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
                if isinstance(u_shared, list):
                    cross_cov_loss = sum(_cross_covariance_penalty(u_shared[i], p) for i, p in enumerate(private_l))
                else:
                    cross_cov_loss = sum(_cross_covariance_penalty(u_shared, p) for p in private_l)
                p_var_loss = sum(_variance_penalty(p) for p in private_l)
                # These attributes are typically set during the ned_simr_shared_private call
                # We default to 0 if not provided by opt loop structure
                total_loss += getattr(model, 'private_ortho_w', 0.05) * cross_cov_loss
                total_loss += getattr(model, 'private_var_w', 0.10) * p_var_loss

            # Add orthogonality penalty for encoder basis V (post-activation) during training
            # EXPERT STRATEGY: Penalize the defect of the actual activated basis enc.v 
            # to encourage emergent disjoint sparsity.
            # We also penalize enc.v_raw for backwards compatibility with test scripts.
            if hasattr(model, 'encoders'):
                for enc in model.encoders:
                    # Apply a stronger soft disjoint penalty to low-dimensional modalities 
                    # since they bypassed the hard Stiefel projection
                    penalty_weight = 0.5 if getattr(enc, 'is_low_dim', False) else 0.05
                    total_loss += penalty_weight * invariant_orthogonality_defect(enc.v)
                    total_loss += penalty_weight * invariant_orthogonality_defect(enc.v_raw)
            elif hasattr(model, 'linear_encoders'):
                total_loss += 0.05 * sum(invariant_orthogonality_defect(enc.v) for enc in model.linear_encoders)
                total_loss += 0.05 * sum(invariant_orthogonality_defect(enc.v_raw) for enc in model.linear_encoders)
            
            if torch.isnan(total_loss): continue
            total_loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0); optimizer.step()
            if hasattr(model, 'update_mai'):
                model.update_mai(shared_latents, epoch, epochs)
            epoch_loss += total_loss.item(); epoch_recon += recon_loss.item(); epoch_sim += diagnostics["sim_loss"]
            
        epoch_loss /= len(dataloader); epoch_recon /= len(dataloader); epoch_sim /= len(dataloader)
        loss_history.append(epoch_loss); recon_history.append(epoch_recon); sim_history.append(epoch_sim)
        scheduler.step()
        
        if epoch_loss < best_loss - tol: 
            best_loss = epoch_loss; patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience and epoch > warmup_epochs:
            if verbose: print(f"Converged at epoch {epoch}: Total Loss {epoch_loss:.4f}")
            converged_epoch = epoch + 1; break
        if verbose and epoch % 10 == 0: 
            print(f"Epoch {epoch}: Total={epoch_loss:.4f} (Recon={epoch_recon:.4f}, Sim={epoch_sim:.4f})")
            
    first_layer_training = {"mode": getattr(getattr(model, "encoders", getattr(model, "linear_encoders", [None]))[0], "first_layer_mode", None) if (hasattr(model, "encoders") or hasattr(model, "linear_encoders")) else None, "stabilization_start_epoch": stabilization_start_epoch, "stabilization_ramp_epochs": stabilization_ramp_epochs, "projection_alpha_history": projection_alpha_history, "basis_drift_history": basis_drift_history}
    return loss_history, recon_history, sim_history, converged_epoch, first_layer_training

def lend_simr(data_matrices: List[Union[torch.Tensor, np.ndarray]], k: int, epochs: int = 150, batch_size: int = 64, learning_rate: float = 5e-4, weight_decay: float = 1e-4, sim_weight: float = 1.0, warmup_epochs: int = 20, hidden_dims: List[int] = [128, 64], dropout: float = 0.1, sparseness_quantile: Union[float, List[float]] = 0.0, positivity: str = "positive", nsa_w: float = 0.1, energy_type: str = "regression", mixing_algorithm: str = "newton",
                 topology: str = "loo", path_graph: Optional[Dict[int, List[int]]] = None, prune_threshold: Optional[float] = None, dynamic_weights: bool = False, mai_metric: str = "procrustes_r2", device: Optional[str] = None, verbose: bool = False, use_nsa: bool = True, first_layer_mode: str = "scheduled", nsa_iterations: int = 1, stabilization_start_epoch: Optional[int] = None, stabilization_ramp_epochs: Optional[int] = None, optimizer_type: str = 'adam', use_rank_mai: bool = False, **kwargs) -> Dict[str, Any]:
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
    """
    if 'sparsity' in kwargs: sparseness_quantile = kwargs.pop('sparsity')
    if 'sparseness' in kwargs: sparseness_quantile = kwargs.pop('sparseness')
    if device is None: device = "cuda" if torch.cuda.is_available() else ("cpu")
    device = torch.device(device); torch_mats, provenance_list = _standardize_deep(data_matrices, ["centerAndScale"]); input_dims = [m.shape[1] for m in torch_mats]
    retraction_type = kwargs.pop('retraction_type', 'soft_polar')
    model = LENDSiMRModel(input_dims, k, hidden_dims, dropout, nsa_w, positivity, sparseness_quantile, mixing_algorithm, topology=topology, path_graph=path_graph, prune_threshold=prune_threshold, dynamic_weights=dynamic_weights, mai_metric=mai_metric, use_nsa=use_nsa, first_layer_mode=first_layer_mode, nsa_iterations=nsa_iterations, use_rank_mai=use_rank_mai, retraction_type=retraction_type).to(device)
    model.initialize_v(torch_mats, k)
    optimizer = _get_optimizer(model, optimizer_type, learning_rate, weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs); mse_loss = nn.MSELoss(); dataset = TensorDataset(*torch_mats); dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_h, recon_h, sim_h, conv_ep, first_layer_training = _train_loop(model, dataloader, optimizer, scheduler, mse_loss, epochs, sim_weight, energy_type, warmup_epochs, verbose, device, tol=kwargs.get('tol', 1e-6), patience=kwargs.get('patience', 10), stabilization_start_epoch=stabilization_start_epoch, stabilization_ramp_epochs=stabilization_ramp_epochs)
    model.eval(); 
    with torch.no_grad():
        eval_mats = [m.to(device) for m in torch_mats]
        final_latents, recons, u_final = model(eval_mats)
        u_aggregate = (_aggregate_shared_consensus(model, final_latents)
                       if isinstance(u_final, list) else u_final)
        v_mats = [torch.nan_to_num(enc.v.detach().cpu(), nan=0.0, posinf=0.0, neginf=0.0) for enc in model.encoders]
        first_layer_scores = [torch.nan_to_num(z.detach().cpu(), nan=0.0, posinf=0.0, neginf=0.0) for z in model.encode_first_layer(eval_mats, use_projected=True)]
        first_layer = build_first_layer_contract(v_mats, first_layer_scores)
    result = {"model": model.cpu(), "model_type": "lend_simr", "u": torch.nan_to_num(u_aggregate.cpu(), nan=0.0, posinf=0.0, neginf=0.0), "u_per_modality": ([torch.nan_to_num(ux.cpu(), nan=0.0, posinf=0.0, neginf=0.0) for ux in u_final] if isinstance(u_final, list) else None), "v": v_mats, "first_layer_scores": first_layer_scores, "first_layer": first_layer, "first_layer_training": first_layer_training, "latents": [torch.nan_to_num(l.cpu(), nan=0.0, posinf=0.0, neginf=0.0) for l in final_latents], "loss_history": loss_h, "recon_history": recon_h, "sim_history": sim_h, "converged_iter": conv_ep, "scale_list": ["centerAndScale"], "provenance_list": provenance_list}
    result["errors"] = [torch.norm(x.cpu() - r.cpu(), p="fro").item() / (torch.norm(x.cpu(), p="fro").item() + 1e-10) for x, r in zip(torch_mats, recons)]
    result["interpretability"] = build_interpretability_report(result)
    result["deep_layer"] = {"alignment_to_first_layer": result["interpretability"]["deep_layer_alignment"]}
    if getattr(model, "dynamic_weights", False):
        result["modality_weights"] = model.modality_weights.detach().cpu().numpy()
        result["mai"] = model.mai.detach().cpu().numpy()
    return result

def ned_simr(data_matrices: List[Union[torch.Tensor, np.ndarray]], k: int, epochs: int = 150, batch_size: int = 64, learning_rate: float = 5e-4, weight_decay: float = 1e-4, sim_weight: float = 1.0, warmup_epochs: int = 20, hidden_dims: List[int] = [128, 64], dropout: float = 0.1, sparseness_quantile: Union[float, List[float]] = 0.0, positivity: str = "positive", nsa_w: float = 0.1, energy_type: str = "regression", mixing_algorithm: str = "newton",
                 topology: str = "loo", path_graph: Optional[Dict[int, List[int]]] = None, prune_threshold: Optional[float] = None, dynamic_weights: bool = False, mai_metric: str = "procrustes_r2", device: Optional[str] = None, verbose: bool = False, use_nsa: bool = True, first_layer_mode: str = "scheduled", nsa_iterations: int = 1, stabilization_start_epoch: Optional[int] = None, stabilization_ramp_epochs: Optional[int] = None, optimizer_type: str = 'adam', use_rank_mai: bool = False, **kwargs) -> Dict[str, Any]:
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
    """
    if 'sparsity' in kwargs: sparseness_quantile = kwargs.pop('sparsity')
    if 'sparseness' in kwargs: sparseness_quantile = kwargs.pop('sparseness')
    if device is None: device = "cuda" if torch.cuda.is_available() else ("cpu")
    device = torch.device(device); torch_mats, provenance_list = _standardize_deep(data_matrices, ["centerAndScale"]); input_dims = [m.shape[1] for m in torch_mats]
    retraction_type = kwargs.pop('retraction_type', 'soft_polar')
    model = NEDSiMRModel(input_dims, k, hidden_dims, dropout, nsa_w, positivity, sparseness_quantile, mixing_algorithm, topology=topology, path_graph=path_graph, prune_threshold=prune_threshold, dynamic_weights=dynamic_weights, mai_metric=mai_metric, use_nsa=use_nsa, first_layer_mode=first_layer_mode, nsa_iterations=nsa_iterations, use_rank_mai=use_rank_mai, retraction_type=retraction_type).to(device)
    model.initialize_v(torch_mats, k)
    optimizer = _get_optimizer(model, optimizer_type, learning_rate, weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs); mse_loss = nn.MSELoss(); dataset = TensorDataset(*torch_mats); dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_h, recon_h, sim_h, conv_ep, first_layer_training = _train_loop(model, dataloader, optimizer, scheduler, mse_loss, epochs, sim_weight, energy_type, warmup_epochs, verbose, device, tol=kwargs.get('tol', 1e-6), patience=kwargs.get('patience', 10), stabilization_start_epoch=stabilization_start_epoch, stabilization_ramp_epochs=stabilization_ramp_epochs)
    model.eval(); 
    with torch.no_grad():
        eval_mats = [m.to(device) for m in torch_mats]
        final_latents, recons, u_final = model(eval_mats)
        u_aggregate = (_aggregate_shared_consensus(model, final_latents)
                       if isinstance(u_final, list) else u_final)
        v_mats = [torch.nan_to_num(enc.v.detach().cpu(), nan=0.0, posinf=0.0, neginf=0.0) for enc in model.linear_encoders]
        first_layer_scores = [torch.nan_to_num(z.detach().cpu(), nan=0.0, posinf=0.0, neginf=0.0) for z in model.encode_first_layer(eval_mats, use_projected=True)]
        first_layer = build_first_layer_contract(v_mats, first_layer_scores)
    result = {"model": model.cpu(), "model_type": "ned_simr", "u": torch.nan_to_num(u_aggregate.cpu(), nan=0.0, posinf=0.0, neginf=0.0), "u_per_modality": ([torch.nan_to_num(ux.cpu(), nan=0.0, posinf=0.0, neginf=0.0) for ux in u_final] if isinstance(u_final, list) else None), "v": v_mats, "first_layer_scores": first_layer_scores, "first_layer": first_layer, "first_layer_training": first_layer_training, "latents": [torch.nan_to_num(l.cpu(), nan=0.0, posinf=0.0, neginf=0.0) for l in final_latents], "loss_history": loss_h, "recon_history": recon_h, "sim_history": sim_h, "converged_iter": conv_ep, "scale_list": ["centerAndScale"], "provenance_list": provenance_list}
    result["errors"] = [torch.norm(x.cpu() - r.cpu(), p="fro").item() / (torch.norm(x.cpu(), p="fro").item() + 1e-10) for x, r in zip(torch_mats, recons)]
    result["interpretability"] = build_interpretability_report(result)
    result["deep_layer"] = {"alignment_to_first_layer": result["interpretability"]["deep_layer_alignment"]}
    if getattr(model, "dynamic_weights", False):
        result["modality_weights"] = model.modality_weights.detach().cpu().numpy()
        result["mai"] = model.mai.detach().cpu().numpy()
    return result

def ned_simr_shared_private(data_matrices: List[Union[torch.Tensor, np.ndarray]], k: int, private_k: Optional[int] = None, epochs: int = 150, batch_size: int = 64, learning_rate: float = 5e-4, weight_decay: float = 1e-4, sim_weight: float = 1.0, warmup_epochs: int = 20, sparseness_quantile: Union[float, List[float]] = 0.0, positivity: str = "positive", nsa_w: float = 0.1, hidden_dims: List[int] = [128, 64], dropout: float = 0.1, energy_type: str = "regression", mixing_algorithm: str = "newton",
                 topology: str = "loo", path_graph: Optional[Dict[int, List[int]]] = None, prune_threshold: Optional[float] = None, dynamic_weights: bool = False, mai_metric: str = "procrustes_r2", private_recon_weight: float = 1.0, private_orthogonality_weight: float = 0.05, private_variance_weight: float = 0.10, device: Optional[str] = None, verbose: bool = False, tol: float = 1e-6, patience: int = 10, use_nsa: bool = True, first_layer_mode: str = "scheduled", nsa_iterations: int = 1, stabilization_start_epoch: Optional[int] = None, stabilization_ramp_epochs: Optional[int] = None, shared_warmup_epochs: int = 20, optimizer_type: str = 'adam', use_rank_mai: bool = False, **kwargs) -> Dict[str, Any]:
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
    """
    if 'sparsity' in kwargs: sparseness_quantile = kwargs.pop('sparsity')
    if 'sparseness' in kwargs: sparseness_quantile = kwargs.pop('sparseness')
    if private_k is None: private_k = max(1, k // 2)
    if device is None: device = "cuda" if torch.cuda.is_available() else ("cpu")
    device = torch.device(device); torch_mats, provenance_list = _standardize_deep(data_matrices, ["centerAndScale"]); input_dims = [m.shape[1] for m in torch_mats]
    retraction_type = kwargs.pop('retraction_type', 'soft_polar')
    model = NEDSharedPrivateSiMRModel(input_dims, k, private_k, hidden_dims, dropout, nsa_w, positivity, sparseness_quantile, mixing_algorithm, topology=topology, path_graph=path_graph, prune_threshold=prune_threshold, dynamic_weights=dynamic_weights, mai_metric=mai_metric, use_nsa=use_nsa, first_layer_mode=first_layer_mode, nsa_iterations=nsa_iterations, use_rank_mai=use_rank_mai, retraction_type=retraction_type).to(device)
    model.initialize_v(torch_mats, k)
    
    # Store weights on model for train loop access
    model.private_ortho_w = private_orthogonality_weight
    model.private_var_w = private_variance_weight
    
    optimizer = _get_optimizer(model, optimizer_type, learning_rate, weight_decay)
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
        u_aggregate = (_aggregate_shared_consensus(model, final_shared)
                       if isinstance(u_final, list) else u_final)
        v_mats = [torch.nan_to_num(enc.v.detach().cpu(), nan=0.0, posinf=0.0, neginf=0.0) for enc in model.linear_encoders]
        first_layer_scores = [torch.nan_to_num(z.detach().cpu(), nan=0.0, posinf=0.0, neginf=0.0) for z in model.encode_first_layer(eval_mats, use_projected=True)]
        first_layer = build_first_layer_contract(v_mats, first_layer_scores)
    result = {"model": model.cpu(), "model_type": "ned_shared_private", "u": torch.nan_to_num(u_aggregate.cpu(), nan=0.0, posinf=0.0, neginf=0.0), "u_per_modality": ([torch.nan_to_num(ux.cpu(), nan=0.0, posinf=0.0, neginf=0.0) for ux in u_final] if isinstance(u_final, list) else None), "v": v_mats, "first_layer_scores": first_layer_scores, "first_layer": first_layer, "first_layer_training": first_layer_training, "latents": [torch.nan_to_num(l.cpu(), nan=0.0, posinf=0.0, neginf=0.0) for l in final_shared], "private_latents": [torch.nan_to_num(p.cpu(), nan=0.0, posinf=0.0, neginf=0.0) for p in final_private], "loss_history": loss_h, "recon_history": recon_h, "sim_history": sim_h, "converged_iter": conv_ep, "scale_list": ["centerAndScale"], "provenance_list": provenance_list}
    result["errors"] = [torch.norm(x.cpu() - r.cpu(), p="fro").item() / (torch.norm(x.cpu(), p="fro").item() + 1e-10) for x, r in zip(torch_mats, final_recons)]
    result["interpretability"] = build_interpretability_report(result)
    result["deep_layer"] = {"alignment_to_first_layer": result["interpretability"]["deep_layer_alignment"]}
    if getattr(model, "dynamic_weights", False):
        result["modality_weights"] = model.modality_weights.detach().cpu().numpy()
        result["mai"] = model.mai.detach().cpu().numpy()
    return result

def deep_simr(data_matrices: List[Union[torch.Tensor, np.ndarray]], k: int, epochs: int = 150, batch_size: int = 64, learning_rate: float = 5e-4, sim_weight: float = 1.0, warmup_epochs: int = 20, energy_type: str = "regression", device: Optional[str] = None, verbose: bool = False, optimizer_type: str = 'adam', **kwargs) -> Dict[str, Any]:
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
    """
    model = model_res["model"]; model_type = model_res.get("model_type", "lend_simr")
    if device is None: device = "cuda" if torch.cuda.is_available() else ("cpu")
    device = torch.device(device); model.to(device).eval()
    torch_mats = [preprocess_data(torch.as_tensor(m).float(), model_res["scale_list"], prov) for m, prov in zip(data_matrices, model_res["provenance_list"])]
    with torch.no_grad():
        res = model([m.to(device) for m in torch_mats])
        first_layer_scores = [z.cpu() for z in model.encode_first_layer([m.to(device) for m in torch_mats], use_projected=True)] if hasattr(model, "encode_first_layer") else None
        v_list = None
        if hasattr(model, "encoders") and not getattr(model, "is_flow", False):
            v_list = [enc.v.detach().cpu() for enc in model.encoders]
        elif hasattr(model, "linear_encoders"):
            v_list = [enc.v.detach().cpu() for enc in model.linear_encoders]
        first_layer = build_first_layer_contract(v_list, first_layer_scores) if v_list is not None and first_layer_scores is not None else None
        if model_type == "ned_shared_private":
            shared_l, recons, u, private_l = res
            u_agg = _aggregate_shared_consensus(model, shared_l) if isinstance(u, list) else u
            result = {"u": u_agg.cpu(), "u_per_modality": ([ui.cpu() for ui in u] if isinstance(u, list) else None), "latents": [l.cpu() for l in shared_l], "reconstructions": [r.cpu() for r in recons], "private_latents": [p.cpu() for p in private_l], "first_layer_scores": first_layer_scores, "first_layer": first_layer, "v": v_list}
        else:
            latents, recons, u = res
            u_agg = _aggregate_shared_consensus(model, latents) if isinstance(u, list) else u
            result = {"u": u_agg.cpu(), "u_per_modality": ([ui.cpu() for ui in u] if isinstance(u, list) else None), "latents": [l.cpu() for l in latents], "reconstructions": [r.cpu() for r in recons], "first_layer_scores": first_layer_scores, "first_layer": first_layer, "v": v_list}
        result["interpretability"] = build_interpretability_report(result) if first_layer is not None else None
        result["deep_layer"] = None if result["interpretability"] is None else {"alignment_to_first_layer": result["interpretability"]["deep_layer_alignment"]}
        result["errors"] = [torch.norm(x.cpu() - r.cpu(), p="fro").item() / (torch.norm(x.cpu(), p="fro").item() + 1e-10) for x, r in zip(torch_mats, recons)]
        if getattr(model, "dynamic_weights", False):
            result["modality_weights"] = model.modality_weights.detach().cpu().numpy()
            result["mai"] = model.mai.detach().cpu().numpy()
        
        # Calculate reconstruction errors
        errors = [torch.norm(x.cpu() - r.cpu(), p='fro').item() / (torch.norm(x.cpu(), p='fro').item() + 1e-10) for x, r in zip(torch_mats, recons)]
        result["errors"] = errors
        
        return result
