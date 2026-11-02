import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import List, Dict, Any, Optional, Sequence, Union, Tuple, Callable
import contextlib
import warnings
from .defaults import DEFAULT_ENERGY_TYPE, DEFAULT_NSA_W, DEFAULT_TOPOLOGY, DEFAULT_COLLAPSE_WEIGHT, DEFAULT_MIXING_ALGORITHM
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
    if optimizer_type == "adam":
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if optimizer_type == "larslow":
        return larslow(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if optimizer_type == "lars":
        try:
            import torch_optimizer as t_opt
        except ImportError:
            warnings.warn(
                "optimizer_type='lars' needs the optional torch_optimizer package; "
                "using pysimlr's own 'larslow' implementation instead.",
                RuntimeWarning, stacklevel=3)
            return larslow(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        return t_opt.LARS(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    raise ValueError(f"optimizer_type={optimizer_type!r} is not a deep-path optimizer; "
                     f"choose one of ('adam', 'lars', 'larslow').")

from .consensus import compute_shared_consensus
from .utils import (preprocess_data, correlation_penalty,
                    orthogonality_defect, safe_svd, basis_rank_report)
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
        frozen_map=getattr(model, "aggregate_map", None),
    )


def _svd_project_columns(u: torch.Tensor) -> torch.Tensor:
    """Project towards the Stiefel manifold using SVD."""
    with torch.no_grad():
        # `safe_svd` already retries with LAPACK gesvd. A failure beyond that
        # propagates; it used to return column-normalized `u`, which is not on
        # the Stiefel manifold this function promises.
        u_svd, _, vh_svd = safe_svd(u, full_matrices=False)
        return u_svd @ vh_svd

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

def _standardize_deep(data_matrices, scale_list=("centerAndScale",)):
    torch_mats = [torch.as_tensor(m).float() for m in data_matrices]
    provenance_list = []
    scaled_mats = []
    for m in torch_mats:
        m_scaled, prov = preprocess_data(m, list(scale_list))
        # Add small noise to prevent constant columns if necessary
        if torch.any(torch.std(m_scaled, dim=0) < 1e-6):
            m_scaled = m_scaled + torch.randn_like(m_scaled) * 1e-6
        scaled_mats.append(m_scaled)
        provenance_list.append(prov)
    return scaled_mats, provenance_list


def nonneg_orthogonal_assignment(v: torch.Tensor, iters: int = 10,
                                 temperature: float = 20.0) -> torch.Tensor:
    r"""The non-negative orthogonal basis for ``p <= k``, by balanced assignment.

    When a view has no more features than components the constrained problem
    is not merely hard, it is forced. With :math:`V \ge 0`, orthogonality
    :math:`\langle v_i, v_j \rangle = 0` requires *disjoint supports*; with
    :math:`p \le k` disjoint supports over :math:`p` coordinates means one
    coordinate per column. The feasible set is therefore the scaled
    permutations, and the operator that lands on it is an assignment.

    Sinkhorn is used rather than `consolidate_supports` because the constraint
    is two-sided. `consolidate_supports` assigns each *row* to its best column
    and does not guarantee that every *column* receives one: on the 2x2 case
    in ``test_nsa_nonnegative_stiefel_collapse`` it puts both coordinates in
    column 0 and leaves column 1 empty, `enforce_column_floor` then revives
    the dead column, and the result has column overlap 0.707. Sinkhorn's
    alternating row/column normalisation enforces both margins, which at
    ``p == k`` is exactly a permutation.

    This is a genuinely different operator from the one every other shape
    uses, and that is a property of the regime rather than a shortcut: the
    NSA-Flow retraction is bypassed entirely when ``p <= k`` (see
    `LENDNSAEncoder.__init__`), because a basis cannot be near-orthogonal
    with more columns than rows. It lives here, named and documented, rather
    than inline in a property, so that a reader of `LENDNSAEncoder.v` can see
    that the regime switches and why.

    Parameters
    ----------
    v : torch.Tensor
        ``(features, latent)`` basis, ``features <= latent``.
    iters : int
        Sinkhorn iterations.
    temperature : float
        Logit scale. Larger drives harder toward a permutation.
    """
    magnitudes = torch.abs(v)
    logits = v * temperature
    # Shift by the global max before exponentiating: a single multiplicative
    # constant on `scores`, which cancels in the first normalisation, so the
    # shift is exact rather than an approximation.
    scores = torch.exp(logits - logits.max().detach())
    for _ in range(iters):
        scores = scores / (torch.sum(scores, dim=0, keepdim=True) + 1e-8)
        scores = scores / (torch.sum(scores, dim=1, keepdim=True) + 1e-8)
    return scores * magnitudes


from torch.optim.lr_scheduler import ReduceLROnPlateau


#: LR schedule for the deep trainers. `CosineAnnealingLR(T_max=epochs)` tied
#: the schedule to the epoch *budget*, which made `epochs` a hyperparameter
#: rather than a cap: at the same step and seed, a 120-epoch run sits well
#: down the cosine while a 600-epoch run is still near its initial rate, so
#: two runs stopping at identical step counts landed at different losses.
#: That is incompatible with running to convergence, where the cap is supposed
#: to be a safety limit. ReduceLROnPlateau responds to the loss instead of the
#: calendar, so the path no longer depends on the budget.
#:
#: Its patience is deliberately shorter than `ConvergenceMonitor`'s: the LR
#: should drop and buy more progress *before* the monitor concludes there is
#: none left.
def _plateau_scheduler(optimizer, factor: float = 0.5, patience: int = 5,
                       threshold: float = 1e-4):
    return ReduceLROnPlateau(optimizer, mode="min", factor=factor,
                             patience=patience, threshold=threshold,
                             threshold_mode="rel")


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


#: MAI similarity metrics. `trace` was removed: tr(Z' u_loo) is the only one
#: not computed through the SVD and it is not invariant to the arbitrary
#: rotation of the k latent coordinates -- measured, the same data under three
#: rotations gave 0.0348, 0.1319 and 0.0000, while the others were constant to
#: four decimals. It ranked a planted pure-noise view above a signal view.
from .consensus import (check_mai_metric,
                        loo_mai_scores, mai_gated_target)


def _mlp(in_dim: int, out_dim: int, hidden_dims: Sequence[int], dropout: float) -> nn.Sequential:
    """Linear -> LayerNorm -> ReLU (-> Dropout) per hidden width, then a linear
    output layer. The body of every `ModalityEncoder` and `ModalityDecoder`."""
    layers, curr = [], in_dim
    for h in hidden_dims:
        layers += [nn.Linear(curr, h), nn.LayerNorm(h), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        curr = h
    layers.append(nn.Linear(curr, out_dim))
    return nn.Sequential(*layers)


def _linear_encoders(input_dims, latent_dim, nsa_w, positivity, sparseness, use_nsa,
                     first_layer_mode, nsa_iterations) -> nn.ModuleList:
    """One `LENDNSAEncoder` per view: the linear first layer shared by LEND,
    NED, NEDPP and Flow-SiMR-V. A scalar or None `sparseness` applies
    to every view."""
    if sparseness is None or isinstance(sparseness, (float, int)):
        sparseness = [None if sparseness is None else float(sparseness)] * len(input_dims)
    return nn.ModuleList([
        LENDNSAEncoder(dim, latent_dim, nsa_w, positivity, sq, use_nsa=use_nsa,
                       first_layer_mode=first_layer_mode, nsa_iterations=nsa_iterations)
        for dim, sq in zip(input_dims, sparseness)])


class _ModalityWeightingMixin:
    """`update_mai`, shared by every deep and flow multi-view model: reweights
    the views by leave-one-out agreement (`consensus.loo_mai_scores`). It was
    copied into each model class; one definition keeps the copies from drifting.
    """

    def _init_consensus_state(self, n_views: int, latent_dim: int, mixing_algorithm: str,
                              topology: str, path_graph, prune_threshold, dynamic_weights: bool,
                              mai_metric: str, use_rank_mai: bool,
                              dynamic_weights_start: Optional[int] = None) -> None:
        """Consensus settings and the three buffers every multi-view model
        carries: MAI scores, modality weights (both uniform), and the
        consensus anchor."""
        self.mixing_algorithm = mixing_algorithm
        self.topology = topology
        self.path_graph = path_graph
        self.prune_threshold = prune_threshold
        self.dynamic_weights = dynamic_weights
        self.mai_metric = mai_metric
        self.use_rank_mai = use_rank_mai
        self.dynamic_weights_start = dynamic_weights_start
        self.register_buffer("mai", torch.ones(n_views) / n_views)
        self.register_buffer("modality_weights", torch.ones(n_views) / n_views)
        self.register_buffer("consensus_anchor", torch.zeros(n_views * latent_dim, latent_dim))

    def _training_consensus(self, latents: List[torch.Tensor], k: int):
        """Shared latent for a forward pass under the model's consensus settings.

        In training mode the consensus anchor follows the batch anchor by an
        EMA (0.9) for stability; in eval mode the frozen training map, when
        one has been captured, makes the result batch-invariant.
        """
        res_u = compute_shared_consensus(
            latents, mixing_algorithm=self.mixing_algorithm, k=k, training=self.training,
            anchor=self.consensus_anchor, topology=self.topology, path_graph=self.path_graph,
            prune_threshold=self.prune_threshold,
            modality_weights=self.modality_weights if self.dynamic_weights else None,
            frozen_map=getattr(self, "consensus_map", None))
        if not self.training:
            return res_u
        u_shared, new_anchor = res_u
        if new_anchor is not None:
            self.consensus_anchor.copy_(0.9 * self.consensus_anchor + 0.1 * new_anchor)
        return u_shared

    def update_mai(self, latents: List[torch.Tensor], epoch: int, total_epochs: int,
                   dynamic_weights_start: Optional[int] = None) -> None:
        """Move `modality_weights` from uniform toward MAI-gated targets.

        Scores are smoothed by an EMA (0.9); the gate steepens over 30 epochs.
        The transition from uniform runs over 20 epochs from
        `dynamic_weights_start` when one is set, otherwise from 33% to 66% of
        the run (it was `(epoch - 10) / 20`, which left MAI inert on runs of
        10 epochs or fewer).
        """
        if not getattr(self, "dynamic_weights", False):
            return
        metric = check_mai_metric(getattr(self, "mai_metric", "procrustes_r2"))
        if dynamic_weights_start is None:
            dynamic_weights_start = getattr(self, "dynamic_weights_start", None)
        with torch.no_grad():
            uniform_w = torch.ones_like(self.modality_weights) / len(self.modality_weights)
            if dynamic_weights_start is not None and epoch < dynamic_weights_start:
                self.modality_weights.copy_(uniform_w)
                return
            mais = loo_mai_scores(latents, metric, use_rank=getattr(self, "use_rank_mai", False))
            self.mai.copy_(0.9 * self.mai + 0.1 * torch.tensor(mais, device=self.mai.device))
            steepness = 5.0 + 10.0 * min(1.0, epoch / 30.0)
            target_w = mai_gated_target(self.mai, steepness)
            if dynamic_weights_start is not None:
                rho = max(0.0, min(1.0, (epoch - dynamic_weights_start) / 20.0))
            else:
                warm = 0.33 * max(1, total_epochs)
                rho = max(0.0, min(1.0, (epoch - warm) / max(1.0, warm)))
            self.modality_weights.copy_((1.0 - rho) * uniform_w + rho * target_w)


class _MultiViewModelMixin(_ModalityWeightingMixin):
    """Adds the first-layer schedule and its diagnostics, which act on the
    per-view linear encoders (`_first_layer_encoders`), for the models that
    have a linear first layer (LEND, NED, NEDPP, Flow-SiMR-V).
    """

    def _first_layer_encoders(self):
        return self.linear_encoders

    def initialize_v(self, data_matrices: List[torch.Tensor], k: int, **init_kwargs):
        """
        Seed each encoder's first-layer basis.

        Delegates to `initialize_deep_encoders`, so any `INITIALIZATION_TYPES`
        strategy is available here, not only PCA -- see `initialize_simlr`'s
        docstring for what each does.
        """
        from .simlr import initialize_deep_encoders
        initialize_deep_encoders(self._first_layer_encoders(), data_matrices, k, **init_kwargs)

    def set_projection_schedule(self, epoch: int, total_epochs: int, stabilization_start_epoch: int,
                                stabilization_ramp_epochs: int) -> None:
        for enc in self._first_layer_encoders():
            enc.set_projection_schedule(epoch, total_epochs, stabilization_start_epoch, stabilization_ramp_epochs)

    def first_layer_diagnostics(self) -> Dict[str, float]:
        encs = list(self._first_layer_encoders())
        drifts = [float(enc.basis_drift().cpu()) for enc in encs]
        alphas = [float(enc.projection_alpha) for enc in encs]
        return {
            "basis_drift": float(sum(drifts) / max(1, len(drifts))),
            "projection_alpha": float(sum(alphas) / max(1, len(alphas))),
        }


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
    nsa_w : float, default=DEFAULT_NSA_W (0.5)
        NSA-Flow weight (`pysimlr.defaults`).
    positivity : str, default="either"
        Positivity constraint on weights:
        - 'either': Unconstrained (standard orthogonal basis).
        - 'positive': Strictly non-negative via clamping (legacy behavior).
        - 'softplus': Smoothly non-negative via Softplus (advanced behavior).
        - 'hard': Strictly non-negative via clamping.
    sparseness : float, default=0.0
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
    def __init__(self, input_dim: int, latent_dim: int, nsa_w: float = DEFAULT_NSA_W, 
                 positivity: str = "either", sparseness: Optional[float] = 0.5,
                 soft_thresholding: bool = False, use_nsa: bool = True,
                 first_layer_mode: str = "scheduled", nsa_iterations: int = 1):
        super().__init__()
        if positivity is True or (isinstance(positivity, str) and positivity.lower() == 'true'):
            positivity = 'positive'
        elif positivity is False or (isinstance(positivity, str) and positivity.lower() == 'false'):
            positivity = 'either'
        self.positivity = positivity
        self.sparseness = sparseness
        self.soft_thresholding = soft_thresholding
        self.use_nsa = use_nsa
        if first_layer_mode not in {"raw", "projected", "scheduled"}:
            raise ValueError(f"Unsupported first_layer_mode: {first_layer_mode}")
        self.first_layer_mode = first_layer_mode
        self.topology = DEFAULT_TOPOLOGY
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
        
        nsa = load_nsa_backend() if use_nsa else None
        if nsa is not None and use_nsa:
            self.nsa_linear = nsa.NSAFlowLinear(in_features=input_dim, out_features=latent_dim,
                                                bias=False, w=nsa_w, nonneg=nsa_layer_nonneg)
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
        if not self.training and getattr(self, "_finalized_v", None) is not None:
            return self._finalized_v.to(device=self.v_raw.device, dtype=self.v_raw.dtype)
        if getattr(self, 'is_low_dim', False):
            # p <= k. A basis cannot be near-orthogonal with more columns than
            # rows, so the retraction backend is bypassed here (see __init__).
            #
            # What used to occupy this branch was a Sinkhorn assignment --
            # exp(20 * V) followed by ten alternating row/column
            # normalisations -- which is a different algorithm from the one
            # every other code path runs, selected silently by a shape test
            # and documented nowhere in the public signature. At that
            # temperature it drives the basis to a near-permutation matrix
            # whatever `nsa_w` is, so a view with p <= k answered a different
            # optimisation problem than its siblings in the same fit, and no
            # sweep over the constraint could move it.
            #
            # What replaces it is not "do nothing": at p <= k the constrained
            # problem still has an answer, and it is forced. Two non-negative
            # orthogonal unit columns in R^2 must be [1,0] and [0,1] -- with
            # V >= 0, <v_i, v_j> = 0 requires disjoint supports, and at p <= k
            # disjoint supports over p coordinates means one coordinate per
            # column. So the feasible set is the scaled permutations, and the
            # right operator is an assignment. That is what the Sinkhorn was
            # approximating, and `test_nsa_nonnegative_stiefel_collapse` is the
            # case it existed for: without it a 2x2 basis comes back with
            # column overlap 0.60 and orthogonality defect 0.18.
            #
            # See `nonneg_orthogonal_assignment` for why this needs a
            # two-sided (Sinkhorn) assignment and why `consolidate_supports`
            # is not a substitute.
            v_out = self.v_raw
            if self.positivity in {'positive', 'hard', 'softplus'}:
                if self.positivity == 'softplus':
                    v_out = torch.nn.functional.softplus(v_out - 4.0)
                v_out = nonneg_orthogonal_assignment(v_out)
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
                    self.nsa_linear.effective_weight(),
                    self.input_dim, self.latent_dim)
                if self.nsa_iterations > 1:
                    v_out = _svd_project_columns(v_out)
            else:
                v_out = _svd_project_columns(self.v_raw)
        from .sparsification import _resolve_quantile, _apply_quantile_sparseness
        q_enc = _resolve_quantile(self.sparseness)
        if self.positivity in {'positive', 'hard'}:
            # `consolidate_supports` used to run here, unconditionally, and it
            # is why `nsa_w` did nothing to any deep model. It forces strictly
            # disjoint supports, so whatever the constrained layer produced was
            # overwritten by a hard partition before anyone saw it.
            v_out = torch.clamp(v_out, min=0.0)
            if q_enc is not None and not getattr(self, 'is_low_dim', False):
                v_out = _apply_quantile_sparseness(v_out, q_enc, nonneg=True, alg="soft")
            v_out = torch.nn.functional.normalize(v_out, p=2, dim=0, eps=1e-8)
        elif self.positivity == 'softplus':
            if q_enc is not None and not getattr(self, 'is_low_dim', False):
                v_out = _apply_quantile_sparseness(v_out, q_enc, nonneg=True, alg="soft")
            v_out = torch.nn.functional.normalize(v_out, p=2, dim=0, eps=1e-8)
        elif self.positivity == 'either':
            if self.nsa_linear is None:
                from .nsa_backend import load_polar_factor
                pf = load_polar_factor()
                if pf is not None:
                    # A backend failure propagates (see D2: no silent switch
                    # between the nsa_flow and PyTorch implementations).
                    v_out = pf(v_out)
                else:
                    v_out = _svd_project_columns(v_out)
            if q_enc is not None and not getattr(self, 'is_low_dim', False):
                v_out = _apply_quantile_sparseness(v_out, q_enc, nonneg=False, alg="soft")
            v_out = torch.nn.functional.normalize(v_out, p=2, dim=0, eps=1e-8)
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
    def __init__(self, latent_dim: int, output_dim: int, hidden_dims: Sequence[int] = (128, 64), dropout: float = 0.1):
        super().__init__()
        self.network = _mlp(latent_dim, output_dim, hidden_dims, dropout)
    def forward(self, z): return self.network(z)

class LENDSiMRModel(_MultiViewModelMixin, nn.Module):
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
    nsa_w : float, default=DEFAULT_NSA_W (0.5)
        NSA-Flow weight (`pysimlr.defaults`).
    positivity : str, default="either"
        Positivity constraint on first-layer weights.
    sparseness : Union[float, List[float]], default=0.0
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
    def __init__(self, input_dims: List[int], latent_dim: int, hidden_dims: Sequence[int] = (128, 64), 
                 dropout: float = 0.1, nsa_w: float = DEFAULT_NSA_W, positivity: str = "either", 
                 sparseness: Optional[Union[float, List[Optional[float]]]] = 0.5, mixing_algorithm: str = DEFAULT_MIXING_ALGORITHM,
                 topology: str = DEFAULT_TOPOLOGY, path_graph: Optional[Dict[int, List[int]]] = None, prune_threshold: Optional[float] = None, dynamic_weights: bool = False, mai_metric: str = "procrustes_r2",
                 use_nsa: bool = True, first_layer_mode: str = "scheduled", nsa_iterations: int = 1, use_rank_mai: bool = False):
        super().__init__()
        self._init_consensus_state(len(input_dims), latent_dim, mixing_algorithm, topology, path_graph,
                                    prune_threshold, dynamic_weights, mai_metric, use_rank_mai)
        self.encoders = _linear_encoders(input_dims, latent_dim, nsa_w, positivity, sparseness, use_nsa, first_layer_mode, nsa_iterations)
        self.decoders = nn.ModuleList([ModalityDecoder(latent_dim, dim, hidden_dims, dropout) for dim in input_dims])
        self.latent_dim = latent_dim

    def _first_layer_encoders(self):
        return self.encoders
    def encode_first_layer(self, x_list: List[torch.Tensor], use_projected: Optional[bool] = None) -> List[torch.Tensor]:
        return [enc.encode_first_layer(x, use_projected=use_projected) for enc, x in zip(self.encoders, x_list)]

    def forward(self, x_list: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        latents = self.encode_first_layer(x_list)
        u_shared = self._training_consensus(latents, self.latent_dim)
        return latents, [dec(u_shared[i] if isinstance(u_shared, list) else u_shared) for i, dec in enumerate(self.decoders)], u_shared

    def get_projectors(self) -> List[Callable[[torch.Tensor], torch.Tensor]]:
        return [enc.get_projector() for enc in self.encoders]

class NEDSiMRModel(_MultiViewModelMixin, nn.Module):
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
    nsa_w : float, default=DEFAULT_NSA_W (0.5)
        NSA-Flow weight (`pysimlr.defaults`).
    positivity : str, default="either"
        Positivity constraint on first-layer weights.
    sparseness : Union[float, List[float]], default=0.0
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
    def __init__(self, input_dims: List[int], latent_dim: int, hidden_dims: Sequence[int] = (128, 64), 
                 dropout: float = 0.1, nsa_w: float = DEFAULT_NSA_W, positivity: str = "either", 
                 sparseness: Optional[Union[float, List[Optional[float]]]] = 0.5, mixing_algorithm: str = DEFAULT_MIXING_ALGORITHM,
                 topology: str = DEFAULT_TOPOLOGY, path_graph: Optional[Dict[int, List[int]]] = None, prune_threshold: Optional[float] = None, dynamic_weights: bool = False, mai_metric: str = "procrustes_r2",
                 use_nsa: bool = True, first_layer_mode: str = "scheduled", nsa_iterations: int = 1, use_rank_mai: bool = False):
        super().__init__()
        self._init_consensus_state(len(input_dims), latent_dim, mixing_algorithm, topology, path_graph,
                                    prune_threshold, dynamic_weights, mai_metric, use_rank_mai)
        self.linear_encoders = _linear_encoders(input_dims, latent_dim, nsa_w, positivity, sparseness, use_nsa, first_layer_mode, nsa_iterations)
        self.nonlinear_heads = nn.ModuleList([ModalityDecoder(latent_dim, latent_dim, hidden_dims, dropout) for _ in input_dims])
        self.decoders = nn.ModuleList([ModalityDecoder(latent_dim, dim, hidden_dims, dropout) for dim in input_dims])
        self.latent_dim = latent_dim
    def encode_first_layer(self, x_list: List[torch.Tensor], use_projected: Optional[bool] = None) -> List[torch.Tensor]:
        return [enc.encode_first_layer(x, use_projected=use_projected) for enc, x in zip(self.linear_encoders, x_list)]

    def forward(self, x_list: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        first_layer_scores = self.encode_first_layer(x_list)
        latents = [head(z0) for head, z0 in zip(self.nonlinear_heads, first_layer_scores)]
        u_shared = self._training_consensus(latents, self.latent_dim)
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

class NEDSharedPrivateSiMRModel(_MultiViewModelMixin, nn.Module):
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
    nsa_w : float, default=DEFAULT_NSA_W (0.5)
        NSA-Flow weight (`pysimlr.defaults`).
    positivity : str, default="either"
        Positivity constraint on shared first-layer weights.
    sparseness : Union[float, List[float]], default=0.0
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
                 hidden_dims: Sequence[int] = (128, 64), dropout: float = 0.1, nsa_w: float = DEFAULT_NSA_W,
                 positivity: str = "either", sparseness: Optional[Union[float, List[Optional[float]]]] = 0.5, mixing_algorithm: str = DEFAULT_MIXING_ALGORITHM,
                 topology: str = DEFAULT_TOPOLOGY, path_graph: Optional[Dict[int, List[int]]] = None, prune_threshold: Optional[float] = None, dynamic_weights: bool = False, mai_metric: str = "procrustes_r2",
                 use_nsa: bool = True, first_layer_mode: str = "scheduled", nsa_iterations: int = 1, use_rank_mai: bool = False):
        super().__init__()
        self._init_consensus_state(len(input_dims), shared_latent_dim, mixing_algorithm, topology, path_graph,
                                    prune_threshold, dynamic_weights, mai_metric, use_rank_mai)
        self.linear_encoders = _linear_encoders(input_dims, shared_latent_dim, nsa_w, positivity, sparseness, use_nsa, first_layer_mode, nsa_iterations)
        self.shared_heads = nn.ModuleList([ModalityDecoder(shared_latent_dim, shared_latent_dim, hidden_dims, dropout) for _ in input_dims])
        self.private_encoders = nn.ModuleList([ModalityEncoder(dim, private_latent_dim, hidden_dims, dropout) for dim in input_dims])
        self.decoders = nn.ModuleList([ModalityDecoder(shared_latent_dim + private_latent_dim, dim, hidden_dims, dropout) for dim in input_dims])
        self.shared_dim = shared_latent_dim
    def encode_first_layer(self, x_list: List[torch.Tensor], use_projected: Optional[bool] = None) -> List[torch.Tensor]:
        return [enc.encode_first_layer(x, use_projected=use_projected) for enc, x in zip(self.linear_encoders, x_list)]

    def forward(self, x_list: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, List[torch.Tensor]]:
        first_layer_scores = self.encode_first_layer(x_list)
        shared_l = [head(z0) for head, z0 in zip(self.shared_heads, first_layer_scores)]
        private_l = [p_enc(x) for p_enc, x in zip(self.private_encoders, x_list)]
        u_shared = self._training_consensus(shared_l, self.shared_dim)
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
    def __init__(self, input_dim: int, k: int, hidden_dims: Sequence[int] = (128, 64), dropout: float = 0.1):
        super().__init__()
        self.network = _mlp(input_dim, k, hidden_dims, dropout)
    def forward(self, x): return self.network(x)

def resolve_warmup_epochs(warmup_epochs, epochs: int) -> int:
    """How many epochs to hold the similarity term at zero.

    ``None`` means ``min(20, max(1, epochs // 4))`` rather than a flat 20.

    The flat default silently disabled the multi-view objective on any short
    run: the training loops gate it with
    ``sim_weight = 0 if epoch < warmup_epochs else sim_weight``, no benchmark
    passed ``warmup_epochs``, and at ``epochs=8`` the alignment term -- the
    entire reason these models exist -- never activated once. Even at the full
    60-epoch budget it was off for the first third. Measured on mfeat6,
    balanced accuracy, 8ep/warmup20 vs 60ep/warmup0: Flow-SiMLR-V 0.558 ->
    0.750, NED 0.617 -> 0.817, LEND 0.625 -> 0.667.
    """
    if warmup_epochs is None:
        return min(20, max(1, int(epochs) // 4))
    return max(0, int(warmup_epochs))


def _drop_last(dataset, batch_size: int) -> bool:
    """Whether the final partial batch is too small to form a cross-moment.

    Every centred similarity needs at least `similarity.MIN_ROWS_FOR_MOMENT`
    rows; a DataLoader remainder of one or two rows does not, and it is a
    property of the batching rather than of the data. Dropping it is correct --
    the alternative is a batch whose similarity term is identically constant.
    Only the remainder is dropped, and only when it is too small.
    """
    from .similarity import MIN_ROWS_FOR_MOMENT
    n = len(dataset)
    rem = n % int(batch_size)
    return 0 < rem < MIN_ROWS_FOR_MOMENT and n > int(batch_size)


def calculate_sim_loss(latents: List[torch.Tensor], 
                       u_shared: Union[torch.Tensor, List[torch.Tensor]], 
                       energy_type: str = DEFAULT_ENERGY_TYPE,
                       weights: Optional[Dict[str, float]] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
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
        Any key of `pysimlr.similarity.SIMILARITY` that does not need the data
        (e.g. "align", "acc", "nc", "procrustes", "logcosh"). Default
        `defaults.DEFAULT_ENERGY_TYPE`.
    weights : Dict[str, float], optional
        Weights for the different loss components:
        - "sim": Similarity loss weight.
        - "var": Variance penalty weight (VICReg-style).
        - "collapse": Weight of the latent correlation penalty
          (`utils.correlation_penalty`) against dimensional collapse.
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
    
    # Fix the gauge before anything is scored, as the linear path does.
    #
    # `acc` is degree-1 homogeneous and unbounded below, so an optimiser
    # reduces it forever by inflating the representation. `simlr` prevents
    # that by returning V with unit columns
    # (`sparsification.energy_needs_unit_columns`); the deep path had no
    # equivalent for the *latent*, which is what the similarity actually sees.
    # Measured with `ned_simr` on a planted two-factor signal, 25 -> 400 steps:
    #
    #     ||latent||   27.5 -> 73.3 (seed 0),  28.1 -> 125.2 (seed 1)
    #     condition    11.5 -> 24.0         ,   7.6 ->  52.1
    #     u -> y R^2   0.795 -> 0.727       , 0.400 ->  0.345
    #
    # with the energy improving from -9.9 to -59.4 throughout. The two latent
    # columns end 98% correlated, so a k=2 representation is effectively rank
    # 1 and training longer makes the answer worse. Gauged, the same runs give
    # 0.909 and 0.891 with condition numbers 1.08 and 1.33, and more training
    # helps again.
    #
    # Applied to `latents` as a whole rather than only inside the similarity:
    # the variance penalty is computed from them too, and gauging just one
    # term left ned_simr at 0.726 mean / 0.446 worst against 0.902 / 0.891
    # when every term sees the same gauged latent.
    #
    # A no-op for the scale-invariant terms -- `align`, `nc` and the ICA
    # contrasts standardise internally -- so it is applied only where the
    # energy is documented as needing a gauge.
    # Decided from the *resolved* registry name, never the raw argument: the
    # deep path's 'regression' is the registry's 'align', and keying the gauge
    # off the alias gauged one and not the other -- the same energy behaving
    # two ways depending on which name the caller typed.
    from .similarity import (SimilarityContext, resolve_energy_name,
                             similarity_energy)
    from .sparsification import energy_needs_unit_columns
    sim_name = resolve_energy_name(energy_type, path="deep")
    if energy_needs_unit_columns(sim_name):
        latents = [(z - z.mean(dim=0, keepdim=True))
                   / (z.std(dim=0, keepdim=True) + 1e-6) for z in latents]

    var_penalty = sum(_variance_penalty(z) for z in latents)
    
    # Similarity terms come from `pysimlr.similarity`, which is the single
    # definition of each energy. This block previously reimplemented them, and
    # the two copies had drifted: "regression" meant ||X - uV'||^2 in the linear
    # path and standardised latent MSE here, and "nc" meant two different
    # formulas. A benchmark that held energy_type fixed while varying the model
    # was therefore varying the objective at the same time.
    #
    # `latents` are already the representation entering the comparison -- the
    # warp output for a flow model, the linear projection otherwise -- so this
    # is exactly the registry's ``s``.
    ctx = SimilarityContext()

    for i, z in enumerate(latents):
        u_target = u_shared[i].detach() if is_loo else u_shared.detach()
        sim_loss = sim_loss + similarity_energy(sim_name, z, u_target, ctx)

    # Dimensional collapse is penalised on the latents themselves
    # (`utils.correlation_penalty`). It was measured on the mixed consensus
    # `u`, which `newton` (and `svd`) mixing orthogonalise, so it read ~0 while
    # the latents had collapsed. Without any collapse penalty NED still
    # collapses: 6 of 20 (data seed, start seed) pairs on the
    # tests/test_collapse_penalty fixture (|r| 0.76-0.98), against 0/20 with
    # this one. (An earlier 0/5 without it used one start per data seed and
    # was not representative.)
    collapse_loss = torch.stack([correlation_penalty(z) for z in latents]).mean()
    if is_loo:
        u_var_penalty = torch.tensor(0.0, device=device)
        for u in u_shared:
            u_var_penalty += _variance_penalty(u)
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
        u_var_penalty = _variance_penalty(u_shared)
        u_std_mean = torch.mean(_safe_std(u_shared, dim=0)).item()
        u_off_diag_cov = torch.norm(cov_u[mask]).item()
    
    weights = {} if weights is None else weights
    unknown = sorted(set(weights) - {"sim", "var", "collapse", "u_var"})
    if unknown:
        raise ValueError(
            f"unknown loss weight(s) {unknown}; calculate_sim_loss uses 'sim', "
            f"'var', 'collapse' and 'u_var'.")
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
        # Absolute, not a fraction of the budget. As `0.6 * epochs` this made
        # the epoch cap a hyperparameter in the same way the cosine LR
        # schedule did: at an 80-epoch cap the projection ramp engaged at 48,
        # inside the run, while at a 400-epoch cap it engaged at 240 and a run
        # converging near epoch 60 never reached it at all. Two "generous"
        # budgets then optimised different problems -- measured, best losses
        # of 1.01384 and 1.01436 from an identical seed.
        stabilization_start_epoch = warmup_epochs + 20
    if stabilization_ramp_epochs is None:
        stabilization_ramp_epochs = 20
    stabilization_start_epoch = int(max(0, min(stabilization_start_epoch, max(0, epochs - 1))))
    stabilization_ramp_epochs = int(max(1, stabilization_ramp_epochs))
    return stabilization_start_epoch, stabilization_ramp_epochs


def _update_first_layer_schedule(model, epoch: int, epochs: int, stabilization_start_epoch: int, stabilization_ramp_epochs: int) -> Dict[str, float]:
    if hasattr(model, "set_projection_schedule"):
        model.set_projection_schedule(epoch, epochs, stabilization_start_epoch, stabilization_ramp_epochs)
    if hasattr(model, "first_layer_diagnostics"):
        return model.first_layer_diagnostics()
    return {"basis_drift": 0.0, "projection_alpha": 1.0}


def _train_loop(model, dataloader, optimizer, scheduler, mse_loss, epochs, sim_weight, energy_type, warmup_epochs, verbose, device, tol=1e-6, patience=10, stabilization_start_epoch: Optional[int] = None, stabilization_ramp_epochs: Optional[int] = None, freeze_private_epochs: int = 0):
    warmup_epochs = resolve_warmup_epochs(warmup_epochs, epochs)
    loss_history, recon_history, sim_history = [], [], []
    projection_alpha_history, basis_drift_history = [], []
    # One criterion for every method in the package; see
    # `utils.ConvergenceMonitor`. The rule this replaces compared
    # `epoch_loss < best_loss - tol` with an *absolute* tol of 1e-6 against a
    # loss of order 10, so essentially any decrease reset the counter and the
    # trainer never stopped early: measured on ADNI it ran its full budget at
    # 8, 30 and 80 epochs alike, with the loss still falling at -0.37/epoch at
    # the 8-epoch cap. The monitor's tolerance is relative, so it means the
    # same thing here as it does in `simlr`.
    from .utils import ConvergenceMonitor
    monitor = ConvergenceMonitor(tol=tol, patience=patience, max_steps=epochs)
    converged_epoch = epochs
    # Best-iterate tracking, to match `simlr`, which returns `best_v_mats`.
    # Without it these trainers returned their *last* weights, and the last
    # is not close to the best: measured on ADNI at a 200-epoch cap, final
    # minus best was +7.63 for LEND (10.51 against 2.88, more than three
    # times worse), +2.78 for NED, +1.91 for NEDPP and Flow-V, against -0.001
    # for simlr. Every deep result was being read off a badly sub-optimal
    # iterate, and the convergence monitor cannot prevent that -- it stops the
    # run, it does not choose what the run returns.
    import copy as _copy
    best_loss, best_state, best_epoch_idx = float("inf"), None, None
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
        "collapse": DEFAULT_COLLAPSE_WEIGHT,
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

            # Orthogonality penalty on the activated basis enc.v and on the raw
            # weights enc.v_raw. The raw-weight term is a regulariser, not a
            # compatibility shim (its old comment said otherwise): removing it
            # dropped NEDPP's shared-branch R^2 below 0.4 on
            # test_shared_private_no_shared_starvation (0.384).
            if hasattr(model, 'encoders'):
                for enc in model.encoders:
                    # Apply a stronger soft disjoint penalty to low-dimensional modalities 
                    # since they bypassed the hard Stiefel projection
                    penalty_weight = 0.5 if getattr(enc, 'is_low_dim', False) else 0.05
                    total_loss += penalty_weight * orthogonality_defect(enc.v)
                    total_loss += penalty_weight * orthogonality_defect(enc.v_raw)
            elif hasattr(model, 'linear_encoders'):
                total_loss += 0.05 * sum(orthogonality_defect(enc.v) for enc in model.linear_encoders)
                total_loss += 0.05 * sum(orthogonality_defect(enc.v_raw) for enc in model.linear_encoders)
            
            if torch.isnan(total_loss): continue
            total_loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0); optimizer.step()
            if hasattr(model, 'update_mai'):
                model.update_mai(shared_latents, epoch, epochs)
            epoch_loss += total_loss.item(); epoch_recon += recon_loss.item(); epoch_sim += diagnostics["sim_loss"]
            
        epoch_loss /= len(dataloader); epoch_recon /= len(dataloader); epoch_sim /= len(dataloader)
        loss_history.append(epoch_loss); recon_history.append(epoch_recon); sim_history.append(epoch_sim)
        # ReduceLROnPlateau is metric-driven, unlike the cosine it replaced.
        scheduler.step(epoch_loss)
        
        # Monitored only after warmup. During warmup the similarity term is
        # switched off, so those epochs optimise a *different* objective and
        # feeding them to the monitor lets the stale counter fill up before
        # the real objective has been seen at all: measured, every deep model
        # stopped at exactly warmup + patience = 20 epochs with the relative
        # range over the last window still 0.56, i.e. nowhere near flat.
        if epoch >= warmup_epochs and np.isfinite(epoch_loss) and epoch_loss < best_loss:
            best_loss, best_epoch_idx = epoch_loss, epoch
            best_state = _copy.deepcopy(model.state_dict())
        stop = monitor.update(epoch_loss) if epoch >= warmup_epochs else False
        if stop:
            if verbose:
                print(f"Stop at epoch {epoch} ({monitor.report()['stop_reason']}): "
                      f"Total Loss {epoch_loss:.4f}")
            converged_epoch = epoch + 1; break
        if verbose and epoch % 10 == 0: 
            print(f"Epoch {epoch}: Total={epoch_loss:.4f} (Recon={epoch_recon:.4f}, Sim={epoch_sim:.4f})")
            
    first_layer_training = {"mode": getattr(getattr(model, "encoders", getattr(model, "linear_encoders", [None]))[0], "first_layer_mode", None) if (hasattr(model, "encoders") or hasattr(model, "linear_encoders")) else None, "stabilization_start_epoch": stabilization_start_epoch, "stabilization_ramp_epochs": stabilization_ramp_epochs, "projection_alpha_history": projection_alpha_history, "basis_drift_history": basis_drift_history}
    if best_state is not None:
        model.load_state_dict(best_state)
    report = monitor.report()
    report["returned"] = "best" if best_state is not None else "final"
    # The epoch of the state actually restored, not the global argmin. Those
    # differ: the similarity term is off during warmup, so pre-warmup epochs
    # optimise a different and easier objective and always look best -- on
    # ADNI the loss goes 2.96 at epoch 19 to 10.49 at epoch 20, the warmup
    # boundary, and a global argmin would name epoch 19 while the restored
    # weights are the best of the real objective.
    report["best_epoch"] = best_epoch_idx
    # `report["best"]` is the monitor's smoothed criterion value; this is the
    # raw loss of the iterate actually restored. They are different
    # quantities and conflating them reads as a reporting bug.
    report["best_loss"] = best_loss if np.isfinite(best_loss) else float("nan")
    report["warmup_epochs"] = int(warmup_epochs)
    return (loss_history, recon_history, sim_history, converged_epoch,
            first_layer_training, report)


def _finalize_bases(v_mats, positivity, nsa_w, energy_type, encoders=None):
    """Project the returned bases with the SAME operator the linear method uses.

    During training the deep encoders hold their basis on the
    ``NSAFlowLinear`` blend plus a clamp and a column normalisation -- a cheap
    surrogate that is fine for gradient steps but is not the feasible set
    ``simlr`` returns its bases on.  Routing every access through the solver
    was measured at 26x per access and dropped; projecting ONCE at the end costs
    one solve per view and puts every method's ``result["v"]`` on the same set,
    with the same certificate.  Scores are recomputed as ``X @ V`` from the
    projected basis so the first-layer contract stays exact.  ``unit_columns``
    keeps the encoders' unit-norm convention (a gauge; the energy is even in it).
    Returns ``(bases, diagnostics)``; on a missing backend the inputs pass
    through and the diagnostics say so.
    """
    from .sparsification import simlr_sparseness, DEFAULT_SPARSENESS
    out, diags = [], []
    for idx, v in enumerate(v_mats):
        d = {}
        sq = (getattr(encoders[idx], "sparseness", DEFAULT_SPARSENESS)
              if encoders is not None and idx < len(encoders)
              else DEFAULT_SPARSENESS)
        try:
            v_f = simlr_sparseness(v.double(), constraint_type="orth",
                                   positivity=positivity,
                                   sparseness=sq,
                                   constraint_weight=nsa_w,
                                   energy_type=energy_type,
                                   modality_index=idx,
                                   unit_columns=True,
                                   retraction_diagnostics=d).to(v.dtype)
        except (ImportError, RuntimeError) as exc:
            v_f, d = v, {"error": repr(exc)[:200]}
        out.append(torch.nan_to_num(v_f, nan=0.0, posinf=0.0, neginf=0.0)); diags.append(d)
    if encoders is not None:
        for enc, v_f in zip(encoders, out):
            enc.register_buffer("_finalized_v", v_f.detach().clone().to(device=enc.v_raw.device, dtype=enc.v_raw.dtype))
    return out, diags


#: Optimizers the deep and flow trainers accept. Anything else raises: the
#: previous dispatcher turned every unrecognised name, typos included, into Adam.
DEEP_OPTIMIZER_TYPES = ("adam", "lars", "larslow")

#: Default decoder widths for the deep entry points (a tuple, so no call can
#: mutate the shared default).
DEFAULT_HIDDEN_DIMS = (128, 64)


def _resolve_device(device: Optional[str]) -> torch.device:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def _resolve_seed(random_state: Optional[int]) -> int:
    """The seed a stochastic fit runs under.

    ``None`` draws one value from the caller's global torch stream, so two
    unseeded calls still differ while each is reproducible from the seed it
    reports in ``result["random_state"]``.
    """
    if random_state is None:
        return int(torch.randint(0, 2**31 - 1, (1,)).item())
    return int(random_state)


@contextlib.contextmanager
def _seeded_rng(seed: int, device: torch.device):
    """Run a fit on its own RNG stream and leave the caller's untouched.

    Covers every global-RNG consumer in a deep or flow fit: the jitter
    `_standardize_deep` adds to near-constant columns, module weight
    initialisation, dropout and the DataLoader shuffle. The caller's CPU (and,
    for a CUDA fit, that device's) generator state is restored on exit.
    """
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        fork = torch.random.fork_rng(devices=[idx], device_type="cuda")
    else:
        fork = torch.random.fork_rng(devices=[])
    with fork:
        torch.random.default_generator.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed(int(seed))
        yield


def _shuffled_loader(torch_mats: List[torch.Tensor], batch_size: int, seed: int) -> DataLoader:
    dataset = TensorDataset(*torch_mats)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      drop_last=_drop_last(dataset, batch_size),
                      generator=torch.Generator().manual_seed(int(seed)))


def _init_generator(initialization_type: str, seed: int) -> torch.Generator:
    """A generator seeded from the fit's seed, for every initializer; the
    deterministic ones ignore it (`initialization_type` is kept for the
    call sites' readability)."""
    return torch.Generator().manual_seed(int(seed))


def _freeze_consensus(model: nn.Module, latents, u_final, u_aggregate) -> float:
    """Freeze the trained model's consensus as the training batch's affine map
    (`consensus.capture_consensus_map`), so prediction does not depend on which
    rows share a batch. Returns the largest absolute residual of the fit."""
    from .consensus import capture_consensus_map
    model.consensus_map = capture_consensus_map(latents, u_final)
    resid = model.consensus_map["max_residual"]
    if isinstance(u_final, list):
        model.aggregate_map = capture_consensus_map(latents, u_aggregate)
        resid = max(resid, model.aggregate_map["max_residual"])
    return float(resid)


def _clean(t: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(t.cpu(), nan=0.0, posinf=0.0, neginf=0.0)


def _fit_deep(model_type: str, build_model: Callable[[List[int]], nn.Module],
              data_matrices, k: int, *, iterations: int, batch_size: int,
              learning_rate: float, weight_decay: float, sim_weight: float,
              warmup_epochs: Optional[int], sparseness, positivity: str,
              nsa_w: float, energy_type: str, device: Optional[str], verbose: bool,
              stabilization_start_epoch: Optional[int],
              stabilization_ramp_epochs: Optional[int], optimizer_type: str,
              initialization_type: str, random_state: Optional[int],
              domain_matrices, n_candidates: int, perturbation_scale: float,
              varimax: Optional[bool], tol: float, patience: int,
              freeze_private_epochs: int = 0) -> Dict[str, Any]:
    """Shared body of `lend_simr`, `ned_simr` and `ned_simr_shared_private`."""
    if optimizer_type not in DEEP_OPTIMIZER_TYPES:
        raise ValueError(f"optimizer_type={optimizer_type!r} is not a deep-path optimizer; "
                         f"choose one of {DEEP_OPTIMIZER_TYPES}.")
    iterations = int(iterations)
    device = _resolve_device(device)
    seed = _resolve_seed(random_state)
    shared_private = model_type == "ned_shared_private"
    with _seeded_rng(seed, device):
        torch_mats, provenance_list = _standardize_deep(data_matrices)
        model = build_model([m.shape[1] for m in torch_mats]).to(device)
        encoders = model.encoders if hasattr(model, "encoders") else model.linear_encoders
        model.initialize_v(torch_mats, k, initialization_type=initialization_type,
                           generator=_init_generator(initialization_type, seed),
                           domain_matrices=domain_matrices, n_candidates=n_candidates,
                           perturbation_scale=perturbation_scale, varimax=varimax)
        optimizer = _get_optimizer(model, optimizer_type, learning_rate, weight_decay)
        scheduler = _plateau_scheduler(optimizer)
        dataloader = _shuffled_loader(torch_mats, batch_size, seed)
        warmup_epochs = resolve_warmup_epochs(warmup_epochs, iterations)
        loss_h, recon_h, sim_h, conv_ep, first_layer_training, conv_report = _train_loop(
            model, dataloader, optimizer, scheduler, nn.MSELoss(), iterations, sim_weight,
            energy_type, warmup_epochs, verbose, device, tol=tol, patience=patience,
            stabilization_start_epoch=stabilization_start_epoch,
            stabilization_ramp_epochs=stabilization_ramp_epochs,
            freeze_private_epochs=freeze_private_epochs)
        model.eval()
        with torch.no_grad():
            eval_mats = [m.to(device) for m in torch_mats]
            v_mats = [_clean(enc.v.detach()) for enc in encoders]
            v_mats, retraction_diagnostics = _finalize_bases(v_mats, positivity, nsa_w, energy_type,
                                                             encoders=encoders)
            out = model(eval_mats)
            if shared_private:
                final_latents, recons, u_final, final_private = out
            else:
                final_latents, recons, u_final = out
            u_aggregate = (_aggregate_shared_consensus(model, final_latents)
                           if isinstance(u_final, list) else u_final)
            consensus_residual = _freeze_consensus(model, final_latents, u_final, u_aggregate)
            first_layer_scores = [_clean((m_.cpu().to(v_.dtype) @ v_).detach())
                                  for m_, v_ in zip(eval_mats, v_mats)]
            first_layer = build_first_layer_contract(v_mats, first_layer_scores)
    result = {
        "model": model.cpu(), "model_type": model_type,
        "retraction_diagnostics": retraction_diagnostics,
        "u": _clean(u_aggregate),
        "u_per_modality": ([_clean(ux) for ux in u_final] if isinstance(u_final, list) else None),
        "v": v_mats, "first_layer_scores": first_layer_scores, "first_layer": first_layer,
        "first_layer_training": first_layer_training,
        "latents": [_clean(l) for l in final_latents],
        "loss_history": loss_h, "recon_history": recon_h, "sim_history": sim_h,
        "converged_iter": conv_ep, "convergence": conv_report,
        "scale_list": ["centerAndScale"], "provenance_list": provenance_list,
        "sparseness": sparseness, "random_state": seed,
        "initialization_type": initialization_type,
        "consensus_map_residual": consensus_residual,
        **basis_rank_report(v_mats, k),
    }
    if shared_private:
        result["private_latents"] = [_clean(p) for p in final_private]
    result["errors"] = [torch.norm(x.cpu() - r.cpu(), p="fro").item()
                        / (torch.norm(x.cpu(), p="fro").item() + 1e-10)
                        for x, r in zip(torch_mats, recons)]
    result["interpretability"] = build_interpretability_report(result)
    result["deep_layer"] = {"alignment_to_first_layer": result["interpretability"]["deep_layer_alignment"]}
    if getattr(model, "dynamic_weights", False):
        result["modality_weights"] = model.modality_weights.detach().cpu().numpy()
        result["mai"] = model.mai.detach().cpu().numpy()
    return result


_DEEP_COMMON_DOC = """
    Parameters
    ----------
    data_matrices : list of torch.Tensor or np.ndarray
        One ``(n, p_m)`` matrix per view; rows are matched samples.
    k : int
        Dimensionality of the shared latent space.
    iterations : int, default=150
        Training budget in epochs (the shared budget name across pysimlr;
        ``api.describe_method`` reports the unit).
    batch_size, learning_rate, weight_decay : training hyperparameters.
    sim_weight : float, default=1.0
        Weight of the similarity term after warmup.
    warmup_epochs : int, optional
        Epochs before the similarity term switches on (`resolve_warmup_epochs`).
    hidden_dims : sequence of int, optional
        Decoder widths; ``None`` means ``DEFAULT_HIDDEN_DIMS``.
    dropout : float, default=0.1
    sparseness : float, list of float or None, default=0.5
        Column-wise quantile sparsification target of the linear first layer.
    positivity : str, default="either"
        Sign constraint on the first-layer basis. Pass ``"positive"`` only for
        genuinely non-negative input; on standardized data it forces a
        non-negative basis onto signed features.
    nsa_w : float, default=DEFAULT_NSA_W (0.5)
        NSA-Flow weight (`pysimlr.defaults`).
    energy_type, mixing_algorithm, topology, path_graph, prune_threshold,
    dynamic_weights, mai_metric, use_rank_mai : consensus settings, as in `simlr`.
    device : str, optional
        ``None`` picks CUDA when available, else CPU.
    use_nsa, first_layer_mode, nsa_iterations,
    stabilization_start_epoch, stabilization_ramp_epochs :
        First-layer projection schedule.
    optimizer_type : {"adam", "lars", "larslow"}, default="adam"
        Unknown names raise `ValueError`.
    initialization_type : str, default="pca"
        One of `pysimlr.simlr.INITIALIZATION_TYPES` (see `initialize_simlr`).
    random_state : int, optional
        Seed for the whole fit: initialization, weight init, dropout and batch
        order run on a private RNG stream seeded with it, and the caller's
        global RNG state is restored afterwards. ``None`` draws one seed from
        the global stream; the seed used is returned as
        ``result["random_state"]``.
    domain_matrices, n_candidates, perturbation_scale :
        Settings of the ``"domain"``, ``"best_of_n"`` and ``"perturbed_pca"``
        initializers.
    varimax : bool or None, default=None
        Orthogonal Varimax at first-layer initialization only (see
        `pysimlr.simlr.resolve_varimax`). The trained basis is never rotated
        afterwards, because the nonlinear decoder is not O(k)-equivariant.
    tol : float, default=1e-6
    patience : int, default=10
        Early-stopping rule (`utils.ConvergenceMonitor`).

    Returns
    -------
    dict
        The deep result contract: ``u``, ``v``, ``latents``, ``convergence``,
        rank diagnostics, ``random_state`` and the trained ``model``.
"""


def _fit_first_layer_model(name: str, model_cls, data_matrices, k, hidden_dims, dropout, nsa_w,
                           positivity, sparseness, mixing_algorithm, topology, path_graph,
                           prune_threshold, dynamic_weights, mai_metric, use_nsa, first_layer_mode,
                           nsa_iterations, use_rank_mai, **fit_kwargs) -> Dict[str, Any]:
    """Shared body of `lend_simr` and `ned_simr`, which take identical
    parameters and differ only in the model class they train."""
    hidden_dims = tuple(DEFAULT_HIDDEN_DIMS if hidden_dims is None else hidden_dims)
    return _fit_deep(
        name,
        lambda dims: model_cls(dims, k, hidden_dims, dropout, nsa_w, positivity, sparseness,
                               mixing_algorithm, topology=topology, path_graph=path_graph,
                               prune_threshold=prune_threshold, dynamic_weights=dynamic_weights,
                               mai_metric=mai_metric, use_nsa=use_nsa,
                               first_layer_mode=first_layer_mode, nsa_iterations=nsa_iterations,
                               use_rank_mai=use_rank_mai),
        data_matrices, k, sparseness=sparseness, positivity=positivity, nsa_w=nsa_w,
        **fit_kwargs)


def lend_simr(data_matrices: List[Union[torch.Tensor, np.ndarray]], k: int, iterations: int = 150,
              batch_size: int = 64, learning_rate: float = 5e-4, weight_decay: float = 1e-4,
              sim_weight: float = 1.0, warmup_epochs: Optional[int] = None,
              hidden_dims: Optional[Sequence[int]] = None, dropout: float = 0.1,
              sparseness: Optional[Union[float, List[Optional[float]]]] = 0.5,
              positivity: str = "either", nsa_w: float = DEFAULT_NSA_W, energy_type: str = DEFAULT_ENERGY_TYPE,
              mixing_algorithm: str = DEFAULT_MIXING_ALGORITHM, topology: str = DEFAULT_TOPOLOGY,
              path_graph: Optional[Dict[int, List[int]]] = None,
              prune_threshold: Optional[float] = None, dynamic_weights: bool = False,
              mai_metric: str = "procrustes_r2", device: Optional[str] = None,
              verbose: bool = False, use_nsa: bool = True, first_layer_mode: str = "scheduled",
              nsa_iterations: int = 1, stabilization_start_epoch: Optional[int] = None,
              stabilization_ramp_epochs: Optional[int] = None, optimizer_type: str = "adam",
              use_rank_mai: bool = False, initialization_type: str = "pca",
              random_state: Optional[int] = None,
              domain_matrices: Optional[List[Optional[torch.Tensor]]] = None,
              n_candidates: int = 5, perturbation_scale: float = 0.05,
              varimax: Optional[bool] = None,
              tol: float = 1e-6, patience: int = 10) -> Dict[str, Any]:
    return _fit_first_layer_model("lend_simr", LENDSiMRModel, **locals())


lend_simr.__doc__ = ("Fit a Linear Encoded Nonlinear Decoding (LEND) model: a linear,\n"
                     "NSA-Flow-projected first layer per view and a nonlinear decoder.\n"
                     + _DEEP_COMMON_DOC)


def ned_simr(data_matrices: List[Union[torch.Tensor, np.ndarray]], k: int, iterations: int = 150,
             batch_size: int = 64, learning_rate: float = 5e-4, weight_decay: float = 1e-4,
             sim_weight: float = 1.0, warmup_epochs: Optional[int] = None,
             hidden_dims: Optional[Sequence[int]] = None, dropout: float = 0.1,
             sparseness: Optional[Union[float, List[Optional[float]]]] = 0.5,
             positivity: str = "either", nsa_w: float = DEFAULT_NSA_W, energy_type: str = DEFAULT_ENERGY_TYPE,
             mixing_algorithm: str = DEFAULT_MIXING_ALGORITHM, topology: str = DEFAULT_TOPOLOGY,
             path_graph: Optional[Dict[int, List[int]]] = None,
             prune_threshold: Optional[float] = None, dynamic_weights: bool = False,
             mai_metric: str = "procrustes_r2", device: Optional[str] = None,
             verbose: bool = False, use_nsa: bool = True, first_layer_mode: str = "scheduled",
             nsa_iterations: int = 1, stabilization_start_epoch: Optional[int] = None,
             stabilization_ramp_epochs: Optional[int] = None, optimizer_type: str = "adam",
             use_rank_mai: bool = False, initialization_type: str = "pca",
             random_state: Optional[int] = None,
             domain_matrices: Optional[List[Optional[torch.Tensor]]] = None,
             n_candidates: int = 5, perturbation_scale: float = 0.05,
             varimax: Optional[bool] = None,
             tol: float = 1e-6, patience: int = 10) -> Dict[str, Any]:
    return _fit_first_layer_model("ned_simr", NEDSiMRModel, **locals())


ned_simr.__doc__ = ("Fit a Nonlinear Encoded Decoding (NED) model: a linear first layer\n"
                    "followed by a nonlinear head per view, and nonlinear decoders.\n"
                    + _DEEP_COMMON_DOC)


def ned_simr_shared_private(data_matrices: List[Union[torch.Tensor, np.ndarray]], k: int,
                            private_k: Optional[int] = None, iterations: int = 150,
                            batch_size: int = 64, learning_rate: float = 5e-4,
                            weight_decay: float = 1e-4, sim_weight: float = 1.0,
                            warmup_epochs: Optional[int] = None,
                            sparseness: Optional[Union[float, List[Optional[float]]]] = 0.5,
                            positivity: str = "either", nsa_w: float = DEFAULT_NSA_W,
                            hidden_dims: Optional[Sequence[int]] = None, dropout: float = 0.1,
                            energy_type: str = DEFAULT_ENERGY_TYPE, mixing_algorithm: str = DEFAULT_MIXING_ALGORITHM,
                            topology: str = DEFAULT_TOPOLOGY, path_graph: Optional[Dict[int, List[int]]] = None,
                            prune_threshold: Optional[float] = None, dynamic_weights: bool = False,
                            mai_metric: str = "procrustes_r2",
                            private_orthogonality_weight: float = 0.05,
                            private_variance_weight: float = 0.10, device: Optional[str] = None,
                            verbose: bool = False, tol: float = 1e-6, patience: int = 10,
                            use_nsa: bool = True, first_layer_mode: str = "scheduled",
                            nsa_iterations: int = 1,
                            stabilization_start_epoch: Optional[int] = None,
                            stabilization_ramp_epochs: Optional[int] = None,
                            shared_warmup_epochs: int = 20, optimizer_type: str = "adam",
                            use_rank_mai: bool = False, initialization_type: str = "pca",
                            random_state: Optional[int] = None,
                            domain_matrices: Optional[List[Optional[torch.Tensor]]] = None,
                            n_candidates: int = 5, perturbation_scale: float = 0.05,
                            varimax: Optional[bool] = None) -> Dict[str, Any]:
    hidden_dims = tuple(DEFAULT_HIDDEN_DIMS if hidden_dims is None else hidden_dims)
    if private_k is None:
        private_k = max(1, k // 2)

    def build(dims):
        model = NEDSharedPrivateSiMRModel(
            dims, k, private_k, hidden_dims, dropout, nsa_w, positivity, sparseness,
            mixing_algorithm, topology=topology, path_graph=path_graph,
            prune_threshold=prune_threshold, dynamic_weights=dynamic_weights,
            mai_metric=mai_metric, use_nsa=use_nsa, first_layer_mode=first_layer_mode,
            nsa_iterations=nsa_iterations, use_rank_mai=use_rank_mai)
        # Read by `_train_loop`.
        model.private_ortho_w = private_orthogonality_weight
        model.private_var_w = private_variance_weight
        return model

    return _fit_deep(
        "ned_shared_private", build, data_matrices, k, iterations=iterations,
        batch_size=batch_size, learning_rate=learning_rate, weight_decay=weight_decay,
        sim_weight=sim_weight, warmup_epochs=warmup_epochs, sparseness=sparseness,
        positivity=positivity, nsa_w=nsa_w, energy_type=energy_type, device=device,
        verbose=verbose, stabilization_start_epoch=stabilization_start_epoch,
        stabilization_ramp_epochs=stabilization_ramp_epochs, optimizer_type=optimizer_type,
        initialization_type=initialization_type, random_state=random_state,
        domain_matrices=domain_matrices, n_candidates=n_candidates,
        perturbation_scale=perturbation_scale, varimax=varimax, tol=tol, patience=patience,
        freeze_private_epochs=shared_warmup_epochs)


ned_simr_shared_private.__doc__ = (
    "Fit a NED model with shared and per-view private latents (NED++).\n\n"
    "Additional parameters: ``private_k`` (private latent width, default ``max(1, k // 2)``),\n"
    "``private_orthogonality_weight``, ``private_variance_weight``\n"
    "and ``shared_warmup_epochs`` (epochs the private encoders stay frozen).\n"
    + _DEEP_COMMON_DOC)


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
