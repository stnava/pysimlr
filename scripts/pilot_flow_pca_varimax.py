#!/usr/bin/env python3
"""Pilot benchmark for PCA-based, Monotone-Spline, and Varimax Normalizing Flows in pysimlr.

Evaluates experimental flow architectures and regularizers WITHOUT modifying
``src/pysimlr/flows.py`` across synthetic nonlinear regimes, ``Diabetes``, and
``ADNI-MCI`` (Protocol B):

Track A -- ``flow_v`` (OLEE linear first layer V_m + k-dim flow):
  * ``simlr``             : Linear OLEE reference baseline
  * ``lend``              : Linear OLEE + MLP decoder reference baseline
  * ``flow_v_base``       : Production Flow-SiMR-V (k-dim RealNVP, DiagGaussian)
  * ``flow_v_pcf_01``     : Idea 1 -- Flow-SiMR-V + exact k-dim PCF Jacobian penalty (alpha=0.10)
  * ``flow_v_pcf_05``     : Idea 1 -- Flow-SiMR-V + exact k-dim PCF Jacobian penalty (alpha=0.50)
  * ``flow_v_spline1d``   : Idea 3a -- Flow-SiMR-V with 1D elementwise monotone RQS splines
                            (exact PCF=0, diagonal Jacobian, genuine k=1 nonlinearity)
  * ``flow_v_spline_pcf`` : Idea 3b -- Hybrid 1D RQS spline + RealNVP couplings + PCF (alpha=0.10)
  * ``flow_v_gpca``       : Control -- Flow-SiMR-V with k-dim GaussianPCA base distribution

Track B -- ``flow_simr`` (ambient p_m -> p_m flow, no linear V_m):
  * ``flow_simr_base``           : Production Flow-SiMR (ambient RealNVP, slice z[:, :k])
  * ``flow_simr_tier1_struct``   : Tier 1a -- Exact post-flow shared O(k) Structure-Varimax gauge
  * ``flow_simr_tier1_jac``      : Tier 1b -- Exact post-flow shared O(k) Jacobian-Varimax gauge
  * ``flow_simr_orthseed``       : Ablation -- Exact O(p_m) Varimax-PCA front rotation + RealNVP
  * ``flow_simr_tier2_jvarimax`` : Tier 2a -- Smooth SiLU RealNVP + post-warmup Jacobian-Varimax & IMA
  * ``flow_simr_tier2_orthseed`` : Tier 2b -- O(p_m) Varimax-PCA front rotation + Tier 2 Jacobian-Varimax & IMA
  * ``flow_simr_gpca``           : Idea 2 -- Ambient Flow-SiMR with GaussianPCA(p_m, latent_dim=k) base

Usage:
    python scripts/pilot_flow_pca_varimax.py --smoke
    python scripts/pilot_flow_pca_varimax.py --iters 100 --n-seeds 5 --n-adni-seeds 3
"""
from __future__ import annotations

import argparse
import copy
import datetime
import html
import json
import math
import os
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr, wilcoxon
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

import antsnormflows as nf
from antsnormflows.distributions import GaussianPCA
from antsnormflows.utils.splines import rational_quadratic_spline

import pysimlr
from pysimlr.benchmarks.deep_ranking import exact_sign_flip_permutation
from pysimlr.benchmarks.metrics import (
    axis_sensitive_cross_val_metrics,
    column_coherence,
    cross_val_metrics,
    latent_recovery_score,
    support_overlap,
    support_recovery_score,
)
from pysimlr.benchmarks.runner import split_indices
from pysimlr.benchmarks.synthetic_cases import _sparse_signed_basis, build_nonlinear_shared_case
from pysimlr.defaults import (
    DEFAULT_COLLAPSE_THRESHOLD,
    DEFAULT_ENERGY_TYPE,
    DEFAULT_MIXING_ALGORITHM,
    DEFAULT_NSA_W,
    DEFAULT_SIGN_ALIGNMENT,
    DEFAULT_TOL,
    DEFAULT_TOPOLOGY,
)
from pysimlr.deep import (
    LENDNSAEncoder,
    _ModalityWeightingMixin,
    _MultiViewModelMixin,
    _aggregate_shared_consensus,
    _finalize_bases,
    _foreach,
    _freeze_consensus,
    _get_optimizer,
    _init_generator,
    _linear_encoders,
    _plateau_scheduler,
    _prepare_deep_inputs,
    _recon_errors,
    _resolve_device,
    _resolve_seed,
    _resolve_stabilization_schedule,
    _seeded_rng,
    _shuffled_loader,
    _sum_on_host,
    _update_first_layer_schedule,
    calculate_sim_loss,
    orthogonality_defect,
    predict_deep,
    resolve_warmup_epochs,
)
from pysimlr.flows import (
    FlowConditionalInference,
    FlowDecoderWrapper,
    FlowEncoderWrapper,
    FlowSiMRModel,
    FlowSiMRVModel,
    NormalizingFlow,
    flow_simr,
    flow_simr_v,
)
from pysimlr.loadings import olee_loadings, structure_coefficients, varimax_rotation
from pysimlr.positivity import resolve_deep_positivity
from pysimlr.utils import ConvergenceMonitor, basis_rank_report

from paper_adni_ablations import PATH_GRAPH as ADNI_PATH_GRAPH
from paper_adni_ablations import protocol_b


# ---------------------------------------------------------------------------
# 1. Experimental Flow Primitives (Pure PyTorch + antsnormflows, no src edits)
# ---------------------------------------------------------------------------

def _eager_rational_quadratic_spline(
    inputs: torch.Tensor,
    unnormalized_widths: torch.Tensor,
    unnormalized_heights: torch.Tensor,
    unnormalized_derivatives: torch.Tensor,
    *,
    inverse: bool = False,
    left: float = -3.5,
    right: float = 3.5,
    bottom: float = -3.5,
    top: float = 3.5,
    min_bin_width: float = 1e-3,
    min_bin_height: float = 1e-3,
    min_derivative: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eager-mode Durkan et al. (2019) Rational-Quadratic Spline matching antsnormflows.utils.splines
    without triggering torch._dynamo C++ recompilations on search_sorted across grad modes."""
    num_bins = unnormalized_widths.shape[-1]
    widths = min_bin_width + (1.0 - min_bin_width * num_bins) * F.softmax(unnormalized_widths, dim=-1)
    cumwidths = F.pad(torch.cumsum(widths, dim=-1), pad=(1, 0), mode="constant", value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    heights = min_bin_height + (1.0 - min_bin_height * num_bins) * F.softmax(unnormalized_heights, dim=-1)
    cumheights = F.pad(torch.cumsum(heights, dim=-1), pad=(1, 0), mode="constant", value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    bins_ref = cumheights if inverse else cumwidths
    adjust = torch.zeros(bins_ref.shape[-1], device=bins_ref.device, dtype=bins_ref.dtype)
    adjust[-1] = 1e-6
    bin_idx = (torch.sum(inputs[..., None] >= (bins_ref + adjust), dim=-1) - 1).clamp(0, num_bins - 1)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]
    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]
    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]
    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        a = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2.0 * input_delta
        ) + input_heights * (input_delta - input_derivatives)
        b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2.0 * input_delta
        )
        c = -input_delta * (inputs - input_cumheights)
        discriminant = (b.pow(2) - 4.0 * a * c).clamp_min(0.0)
        root = (2.0 * c) / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths
        t1mt = root * (1.0 - root)
        denom = input_delta + (input_derivatives + input_derivatives_plus_one - 2.0 * input_delta) * t1mt
        deriv_num = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2.0 * input_delta * t1mt
            + input_derivatives * (1.0 - root).pow(2)
        )
        logabsdet = torch.log(deriv_num) - 2.0 * torch.log(denom)
        return outputs, -logabsdet

    theta = (inputs - input_cumwidths) / input_bin_widths
    t1mt = theta * (1.0 - theta)
    num = input_heights * (input_delta * theta.pow(2) + input_derivatives * t1mt)
    denom = input_delta + (input_derivatives + input_derivatives_plus_one - 2.0 * input_delta) * t1mt
    outputs = input_cumheights + num / denom
    deriv_num = input_delta.pow(2) * (
        input_derivatives_plus_one * theta.pow(2)
        + 2.0 * input_delta * t1mt
        + input_derivatives * (1.0 - theta).pow(2)
    )
    logabsdet = torch.log(deriv_num) - 2.0 * torch.log(denom)
    return outputs, logabsdet


class ElementwiseRQSplineLayer(nn.Module):
    """Coordinate-wise strictly monotone Rational-Quadratic Spline + positive affine layer.

    Maps R^dim -> R^dim independently on each coordinate j = 0..dim-1:
      1. Unconstrained monotone rational-quadratic spline on [-tail_bound, tail_bound]
         with exact identity linear tails outside [-tail_bound, tail_bound];
      2. Strictly positive per-coordinate scaling exp(scale_bound * tanh(log_scale)) + shift.

    Properties:
      * Exact identity at initialization (widths=0, heights=0, deriv=1, scale=0, shift=0);
      * Strictly diagonal positive Jacobian (dz_j / dx_l == 0 for j != l, dz_j / dx_j > 0);
      * PCF Hadamard penalty == 0 identically;
      * Genuine nonlinear density warping even when dim == 1.
    """

    def __init__(
        self,
        dim: int,
        num_bins: int = 8,
        tail_bound: float = 3.5,
        scale_bound: float = 2.0,
        min_derivative: float = 1e-3,
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_bins = int(num_bins)
        self.tail_bound = float(tail_bound)
        self.scale_bound = float(scale_bound)
        self.min_derivative = float(min_derivative)

        init_deriv = math.log(math.exp(1.0 - self.min_derivative) - 1.0)
        self.register_buffer("_tail_deriv", torch.tensor(init_deriv, dtype=torch.float32))
        self.unnorm_widths = nn.Parameter(torch.zeros(1, self.dim, self.num_bins))
        self.unnorm_heights = nn.Parameter(torch.zeros(1, self.dim, self.num_bins))
        # Interior knots only (num_bins - 1); boundary derivatives fixed to 1.0 for C^1 linear tails
        self.unnorm_derivs = nn.Parameter(
            torch.full((1, self.dim, self.num_bins - 1), init_deriv, dtype=torch.float32)
        )
        self.raw_log_scale = nn.Parameter(torch.zeros(1, self.dim))
        self.shift = nn.Parameter(torch.zeros(1, self.dim))

    def _spline_params(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        w = self.unnorm_widths.expand(batch_size, self.dim, self.num_bins)
        h = self.unnorm_heights.expand(batch_size, self.dim, self.num_bins)
        pad = self._tail_deriv.view(1, 1, 1).expand(batch_size, self.dim, 1)
        d = torch.cat([pad, self.unnorm_derivs.expand(batch_size, self.dim, self.num_bins - 1), pad], dim=-1)
        return w, h, d

    def _apply_spline(self, x: torch.Tensor, inverse: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        tb = self.tail_bound
        inside = (x >= -tb) & (x <= tb)
        x_clamped = torch.clamp(x, min=-tb, max=tb)
        w, h, d = self._spline_params(x.shape[0])
        y_sp, ld_sp = _eager_rational_quadratic_spline(
            inputs=x_clamped,
            unnormalized_widths=w,
            unnormalized_heights=h,
            unnormalized_derivatives=d,
            inverse=inverse,
            left=-tb,
            right=tb,
            bottom=-tb,
            top=tb,
            min_bin_width=1e-3,
            min_bin_height=1e-3,
            min_derivative=self.min_derivative,
        )
        y = torch.where(inside, y_sp.to(x.dtype), x)
        ld = torch.where(inside, ld_sp.to(x.dtype), torch.zeros_like(x))
        return y, ld

    def forward_and_log_det(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y, ld_sp = self._apply_spline(x, inverse=False)
        log_s = self.scale_bound * torch.tanh(self.raw_log_scale)
        z = y * torch.exp(log_s) + self.shift
        total_ld = ld_sp.sum(dim=1) + log_s.sum(dim=1).expand(x.shape[0])
        return z, total_ld

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_and_log_det(x)

    def inverse_and_log_det(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        log_s = self.scale_bound * torch.tanh(self.raw_log_scale)
        y = (z - self.shift) * torch.exp(-log_s)
        x, ld_sp = self._apply_spline(y, inverse=True)
        total_ld = ld_sp.sum(dim=1) - log_s.sum(dim=1).expand(z.shape[0])
        return x, total_ld

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        x, _ = self.inverse_and_log_det(z)
        return x


class ElementwiseRQSplineFlow(nn.Module):
    """Stack of coordinate-wise monotone RQS layers matching NormalizingFlow's interface."""

    def __init__(self, dim: int, num_layers: int = 2, num_bins: int = 8,
                 tail_bound: float = 3.5, scale_bound: float = 2.0):
        super().__init__()
        self.dim = int(dim)
        self.layers = nn.ModuleList([
            ElementwiseRQSplineLayer(dim, num_bins=num_bins, tail_bound=tail_bound, scale_bound=scale_bound)
            for _ in range(num_layers)
        ])

    def forward_and_log_det(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = x
        total_ld = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for layer in self.layers:
            z, ld = layer.forward_and_log_det(z)
            total_ld = total_ld + ld
        return z, total_ld

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_and_log_det(x)

    def inverse_and_log_det(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = z
        total_ld = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        for layer in reversed(self.layers):
            x, ld = layer.inverse_and_log_det(x)
            total_ld = total_ld + ld
        return x, total_ld

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        x, _ = self.inverse_and_log_det(z)
        return x


class HybridSplineRealNVPFlow(nn.Module):
    """1D Elementwise Monotone Spline followed by RealNVP couplings (for Idea 3b)."""

    def __init__(self, dim: int, num_layers: int = 3, hidden_dim: int = 64,
                 num_bins: int = 8, scale_bound: float = 2.0):
        super().__init__()
        self.dim = int(dim)
        self.spline = ElementwiseRQSplineLayer(dim, num_bins=num_bins, scale_bound=scale_bound)
        self.rnvp = NormalizingFlow(dim, num_layers=max(1, num_layers - 1),
                                    hidden_dim=hidden_dim, scale_bound=scale_bound)

    def forward_and_log_det(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h, ld1 = self.spline.forward_and_log_det(x)
        z, ld2 = self.rnvp.forward_and_log_det(h)
        return z, ld1 + ld2

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_and_log_det(x)

    def inverse_and_log_det(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h, ld2 = self.rnvp.inverse_and_log_det(z)
        x, ld1 = self.spline.inverse_and_log_det(h)
        return x, ld1 + ld2

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        h = self.rnvp.inverse(z)
        return self.spline.inverse(h)


class SmoothAffineCouplingLayer(nn.Module):
    """Smooth SiLU-conditioned Affine Coupling Layer with zero-initialized output (identity start).

    Unlike ReLU conditioners (whose Jacobian wrt x is piecewise constant with zero
    second derivative wrt x), SiLU gives C^infinity Jacobians for Tier 2 training-time
    Jacobian-Varimax regularization while starting at exact identity (z == x, log_det == 0).
    """

    def __init__(self, dim: int, hidden_dim: int = 64, index: int = 0, scale_bound: float = 2.0):
        super().__init__()
        self.dim = int(dim)
        self.index = int(index)
        self.scale_bound = float(scale_bound)
        all_idx = torch.arange(self.dim, dtype=torch.long)
        even = (all_idx % 2 == 0)
        sel = ~even if (self.index % 2 == 1) else even
        idx1 = all_idx[sel]
        idx2 = all_idx[~sel]
        inv_order = torch.argsort(torch.cat([idx1, idx2]))
        self.register_buffer("_idx1", idx1, persistent=False)
        self.register_buffer("_idx2", idx2, persistent=False)
        self.register_buffer("_inv", inv_order, persistent=False)

        in_sz = int(idx1.numel())
        out_sz = int(idx2.numel()) * 2
        fc_out = nn.Linear(hidden_dim, out_sz)
        nn.init.zeros_(fc_out.weight)
        nn.init.zeros_(fc_out.bias)
        self.net = nn.Sequential(
            nn.Linear(in_sz, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            fc_out,
        )

    def forward_and_log_det(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._idx2.numel() == 0:
            return x, torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        x1, x2 = x[:, self._idx1], x[:, self._idx2]
        s, t = self.net(x1).chunk(2, dim=1)
        s = torch.tanh(s) * self.scale_bound
        z2 = x2 * torch.exp(s) + t
        z = torch.cat([x1, z2], dim=1)[:, self._inv]
        return z, s.sum(dim=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_and_log_det(x)

    def inverse_and_log_det(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._idx2.numel() == 0:
            return z, torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        z1, z2 = z[:, self._idx1], z[:, self._idx2]
        s, t = self.net(z1).chunk(2, dim=1)
        s = torch.tanh(s) * self.scale_bound
        x2 = (z2 - t) * torch.exp(-s)
        x = torch.cat([z1, x2], dim=1)[:, self._inv]
        return x, -s.sum(dim=1)

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        x, _ = self.inverse_and_log_det(z)
        return x


class SmoothRealNVPFlow(nn.Module):
    """Smooth SiLU RealNVP with optional exact O(p) front orthogonal basis Q_init."""

    def __init__(self, dim: int, num_layers: int = 4, hidden_dim: int = 64,
                 scale_bound: float = 2.0, q_init: Optional[torch.Tensor] = None):
        super().__init__()
        self.dim = int(dim)
        if q_init is not None:
            self.register_buffer("q_orth", q_init.float().clone())
        else:
            self.q_orth = None
        self.layers = nn.ModuleList([
            SmoothAffineCouplingLayer(dim, hidden_dim=hidden_dim, index=i, scale_bound=scale_bound)
            for i in range(num_layers)
        ])

    def forward_and_log_det(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = x @ self.q_orth if self.q_orth is not None else x
        total_ld = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for layer in self.layers:
            z, ld = layer.forward_and_log_det(z)
            total_ld = total_ld + ld
        return z, total_ld

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_and_log_det(x)

    def inverse_and_log_det(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = z
        total_ld = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        for layer in reversed(self.layers):
            x, ld = layer.inverse_and_log_det(x)
            total_ld = total_ld + ld
        if self.q_orth is not None:
            x = x @ self.q_orth.t()
        return x, total_ld

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        x, _ = self.inverse_and_log_det(z)
        return x


class PreOrthogonalFlow(nn.Module):
    """Wraps any square flow f: R^p -> R^p with an exact orthogonal input frame Q in O(p):
    z = f(x @ Q), x_hat = f^{-1}(z) @ Q^T, log|det J| = log|det J_f(x @ Q)| + 0.
    """

    def __init__(self, base_flow: nn.Module, q_orth: torch.Tensor):
        super().__init__()
        self.dim = int(base_flow.dim)
        self.base_flow = base_flow
        self.register_buffer("q_orth", q_orth.float().clone())

    def forward_and_log_det(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.base_flow.forward_and_log_det(x @ self.q_orth)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_and_log_det(x)

    def inverse_and_log_det(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_q, ld = self.base_flow.inverse_and_log_det(z)
        return x_q @ self.q_orth.t(), ld

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        return self.base_flow.inverse(z) @ self.q_orth.t()


class PostRotatedFlowWrapper(nn.Module):
    """Exact O(k) gauge rotation R on the top-k shared coordinates of a square flow f: R^p -> R^p.

    Forward:  z_full = f(x);  z_full[:, :k] <- z_full[:, :k] @ R;  log_det unchanged (|det R| == 1).
    Inverse:  z_unrot = z_full.clone();  z_unrot[:, :k] <- z_full[:, :k] @ R^T;  return f^{-1}(z_unrot).
    """

    def __init__(self, base_flow: nn.Module, r_orth: torch.Tensor):
        super().__init__()
        self.dim = int(base_flow.dim)
        self.k = int(r_orth.shape[0])
        self.base_flow = base_flow
        self.register_buffer("r_orth", r_orth.float().clone())

    def forward_and_log_det(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z, ld = self.base_flow.forward_and_log_det(x)
        if self.k <= z.shape[1]:
            z = torch.cat([z[:, :self.k] @ self.r_orth, z[:, self.k:]], dim=1)
        return z, ld

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_and_log_det(x)

    def inverse_and_log_det(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.k <= z.shape[1]:
            z = torch.cat([z[:, :self.k] @ self.r_orth.t(), z[:, self.k:]], dim=1)
        return self.base_flow.inverse_and_log_det(z)

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        if self.k <= z.shape[1]:
            z = torch.cat([z[:, :self.k] @ self.r_orth.t(), z[:, self.k:]], dim=1)
        return self.base_flow.inverse(z)


class GaussianPCAEncoderWrapper(nn.Module):
    """Encoder wrapper for GaussianPCA base q0: extracts whitened k-dim posterior mean."""

    def __init__(self, flow_model: nn.Module, gpca: GaussianPCA, latent_dim: int):
        super().__init__()
        self.flow = flow_model
        self.gpca = gpca
        self.latent_dim = int(latent_dim)
        self.last_z = None
        self.last_log_det = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, ld = self.flow(x)
        self.last_z = z
        self.last_log_det = ld
        zc = z - self.gpca.loc
        w = self.gpca.W  # (k, D)
        sigma2 = torch.exp(2.0 * self.gpca.log_sigma)
        a_mat = w @ w.t() + (sigma2 + 1e-5) * torch.eye(w.shape[0], device=w.device, dtype=w.dtype)
        proj = zc @ w.t()
        return torch.linalg.solve(a_mat, proj.t()).t()


class GaussianPCADecoderWrapper(nn.Module):
    """Decoder wrapper for GaussianPCA base q0: maps k-dim latent eps -> eps @ W + loc -> f^{-1}."""

    def __init__(self, flow_model: nn.Module, gpca: GaussianPCA):
        super().__init__()
        self.flow = flow_model
        self.gpca = gpca

    def forward(self, eps: torch.Tensor) -> torch.Tensor:
        z_hat = eps @ self.gpca.W + self.gpca.loc
        return self.flow.inverse(z_hat)


# ---------------------------------------------------------------------------
# 2. Exact PCF, Jacobian-Varimax, and Orthogonal Varimax-PCA Initializer
# ---------------------------------------------------------------------------

def exact_pcf_penalty_bpd(flow_module: nn.Module, x_detached: torch.Tensor) -> torch.Tensor:
    """Exact Hadamard-inequality PCF penalty in bits/dim on encoder direction (x -> z):

        PCF(f; x) = [ sum_{j=1}^d log || dz_j / dx ||_2 - log |det dz/dx| ] / (d * ln 2) >= 0.

    Zero iff the rows of the encoder Jacobian J_f(x) are mutually orthogonal.
    """
    d = int(x_detached.shape[1])
    if d <= 1 or x_detached.shape[0] == 0:
        return torch.tensor(0.0, device=x_detached.device, dtype=torch.float32)
    xr = x_detached.detach().float().requires_grad_(True)
    z, log_det = flow_module.forward_and_log_det(xr)
    row_log_norms = []
    for j in range(d):
        g_j = torch.autograd.grad(
            z[:, j].sum(), xr, create_graph=True, retain_graph=True
        )[0]
        row_log_norms.append(0.5 * torch.log(g_j.pow(2).sum(dim=1).clamp_min(1e-24)))
    sum_log_norms = torch.stack(row_log_norms, dim=1).sum(dim=1)
    # Clamp numerical roundoff below 0 to -1e-7 in forward diagnostics while keeping exact gradient
    pen = (sum_log_norms - log_det).mean() / (d * math.log(2.0))
    return pen


def shared_jacobian_ima_and_varimax(
    flow_module: nn.Module,
    x_batch: torch.Tensor,
    k: int,
    max_samples: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute (ima_defect, neg_varimax) on the p x k shared encoder Jacobian G(x) = (dz_{1:k}/dx)^T.

    Costs exactly k vector-Jacobian products regardless of ambient feature width p.
    Returns:
      * ima_defect : mean normalized off-diagonal squared cosine among the k shared sensitivity vectors;
      * neg_varimax: negative Kaiser (1958) Varimax criterion across the p features (minimize to localize).
    """
    n_sub = min(int(max_samples), int(x_batch.shape[0]))
    if n_sub == 0 or k <= 0:
        zero = torch.tensor(0.0, device=x_batch.device, dtype=torch.float32)
        return zero, zero
    xr = x_batch[:n_sub].detach().float().requires_grad_(True)
    z, _ = flow_module.forward_and_log_det(xr)
    k_eff = min(int(k), int(z.shape[1]))
    cols = []
    for j in range(k_eff):
        g_j = torch.autograd.grad(
            z[:, j].sum(), xr, create_graph=True, retain_graph=True
        )[0]  # (n_sub, p)
        cols.append(g_j)
    # G has shape (n_sub, p, k_eff)
    g_mat = torch.stack(cols, dim=-1)
    if k_eff == 1:
        # At k=1, off-diagonal defect is 0; unnormalized fourth-moment kurtosis across p localizes support
        col_unit = g_mat[..., 0] / g_mat[..., 0].norm(dim=1, keepdim=True).clamp_min(1e-12)
        u2 = col_unit.pow(2)
        vmax = (u2.pow(2).mean(dim=1) - u2.mean(dim=1).pow(2)).mean() * float(g_mat.shape[1])
        return torch.tensor(0.0, device=x_batch.device), -vmax

    # 1. Local IMA off-diagonal cosine defect among the k columns of G(x)
    col_norms = g_mat.norm(dim=1, keepdim=True).clamp_min(1e-12)
    g_col_unit = g_mat / col_norms  # (n_sub, p, k)
    gram = torch.bmm(g_col_unit.transpose(1, 2), g_col_unit)  # (n_sub, k, k)
    eye = torch.eye(k_eff, device=gram.device, dtype=gram.dtype).unsqueeze(0)
    off_diag = (gram - eye).pow(2).sum(dim=(1, 2)) / float(k_eff * (k_eff - 1))
    ima_defect = off_diag.mean()

    # 2. Local Kaiser row-normalized Varimax criterion across features i = 1..p
    # Guard zero-sensitivity rows smoothly so row normalization only acts on active features
    row_sq = g_mat.pow(2).sum(dim=-1, keepdim=True)
    med_sq = row_sq.detach().mean(dim=1, keepdim=True).clamp_min(1e-12)
    g_row_norm = g_mat / (row_sq + 0.05 * med_sq).sqrt()
    u2 = g_row_norm.pow(2)
    vmax_per_sample = (u2.pow(2).mean(dim=1) - u2.mean(dim=1).pow(2)).mean(dim=-1)
    return ima_defect, -vmax_per_sample.mean()


def build_varimax_pca_orthogonal_frames(
    data_matrices: Sequence[torch.Tensor],
    k: int,
) -> List[torch.Tensor]:
    """Build exact square orthogonal matrices Q_m in O(p_m) for each view m:
      * First k columns: Kaiser-Varimax-rotated top-k right singular vectors of X_m,
        column- and sign-aligned across views to view 0 in sample space;
      * Remaining p_m - k columns: exact orthogonal complement singular vectors k..p_m-1.
    """
    frames: List[torch.Tensor] = []
    ref_scores: Optional[np.ndarray] = None
    for x in data_matrices:
        x64 = x.detach().cpu().double()
        x64 = x64 - x64.mean(dim=0, keepdim=True)
        p = int(x64.shape[1])
        k_eff = min(int(k), p)
        _, _, vh = torch.linalg.svd(x64, full_matrices=True)
        q_full = vh.t()  # (p, p), exact O(p)
        e_top = q_full[:, :k_eff]  # (p, k_eff)
        if k_eff > 1:
            r_vm = varimax_rotation([e_top], max_iter=500)  # (k_eff, k_eff)
            e_rot = e_top @ r_vm.double()
        else:
            s3 = float((e_top ** 3).sum().item())
            e_rot = e_top * (1.0 if s3 >= 0 else -1.0)

        scores = (x64 @ e_rot).numpy()
        norms = np.linalg.norm(scores, axis=0, keepdims=True)
        scores_u = np.where(norms > 1e-12, scores / np.where(norms > 1e-12, norms, 1.0), 0.0)
        if ref_scores is None:
            ref_scores = scores_u[:, :k_eff]
        elif scores_u.shape[1] == ref_scores.shape[1]:
            c_mat = ref_scores.T @ scores_u
            _, col_ind = linear_sum_assignment(-np.abs(c_mat))
            signs = np.sign(c_mat[np.arange(k_eff), col_ind])
            signs[signs == 0] = 1.0
            perm_t = torch.as_tensor(col_ind, dtype=torch.long)
            sign_t = torch.as_tensor(signs, dtype=torch.float64).unsqueeze(0)
            e_rot = e_rot[:, perm_t] * sign_t

        q_out = torch.cat([e_rot, q_full[:, k_eff:]], dim=1)
        frames.append(q_out.float())
    return frames


@torch.no_grad()
def apply_tier1_post_flow_varimax(
    base_res: Dict[str, Any],
    train_views_raw: Sequence[torch.Tensor],
    k: int,
    eval_view_indices: Optional[Sequence[int]] = None,
    mode: str = "structure",
) -> Dict[str, Any]:
    """Construct an exact likelihood-preserving O(k) gauge-rotated copy of a fitted flow_simr result.

    ``mode="structure"`` rotates by Kaiser Varimax of stacked training structure coefficients S_m;
    ``mode="jacobian"`` rotates by Kaiser Varimax of stacked training mean encoder Jacobians G_m.
    """
    res = copy.deepcopy(base_res)
    model: FlowSiMRModel = res["model"]
    model.eval()

    # Standardize training views using the fit's own provenance
    from pysimlr.api import _preprocess_views
    xs_tr, _ = _preprocess_views(train_views_raw, res["scale_list"], res["provenance_list"])
    use_idx = list(range(len(xs_tr))) if eval_view_indices is None else list(eval_view_indices)

    if mode == "structure":
        u_tr = res["u"].detach().cpu()
        s_list = structure_coefficients([xs_tr[i] for i in use_idx], u_tr)
        r_orth = varimax_rotation(s_list).float()
    elif mode == "jacobian":
        g_means = []
        with torch.enable_grad():
            for i in use_idx:
                xm = xs_tr[i].detach().cpu().float().requires_grad_(True)
                zm = model.encoders[i](xm)
                cols = []
                for j in range(min(k, zm.shape[1])):
                    gj = torch.autograd.grad(zm[:, j].sum(), xm, retain_graph=True)[0]
                    cols.append(gj.detach().mean(dim=0))
                g_means.append(torch.stack(cols, dim=-1))  # (p_i, k)
        r_orth = varimax_rotation(g_means).float()
    else:
        raise ValueError(f"Unknown Tier 1 mode: {mode!r}")

    # Wrap each view's flow with the shared orthogonal R in O(k)
    new_flows = nn.ModuleList([PostRotatedFlowWrapper(fl, r_orth) for fl in model.flows])
    model.flows = new_flows
    model.encoders = nn.ModuleList([FlowEncoderWrapper(fl, k) for fl in model.flows])
    model.decoders = nn.ModuleList([FlowDecoderWrapper(fl, k) for fl in model.flows])

    # Conjugate training latents & consensus by R so the frozen affine consensus map is exact in the rotated gauge
    eval_mats = [m.float() for m in xs_tr]
    rot_latents = [enc(xm) for enc, xm in zip(model.encoders, eval_mats)]
    u_agg = base_res["u"].detach().cpu().float() @ r_orth
    if base_res.get("u_per_modality") is not None:
        u_final = [ui.detach().cpu().float() @ r_orth for ui in base_res["u_per_modality"]]
    else:
        u_final = u_agg
    consensus_res = _freeze_consensus(model, rot_latents, u_final, u_agg, obs_masks=None)
    _, recons, _ = model(eval_mats)
    cond_inf = FlowConditionalInference(len(xs_tr), k)
    cond_inf.fit(rot_latents)

    res["model"] = model
    res["model_type"] = f"flow_simr_tier1_{mode}"
    res["u"] = u_agg.cpu()
    res["u_per_modality"] = [ui.cpu() for ui in u_final] if isinstance(u_final, list) else None
    res["latents"] = [l.cpu() for l in rot_latents]
    res["reconstructions"] = [r.cpu() for r in recons]
    res["consensus_map_residual"] = consensus_res
    res["cond_inference"] = cond_inf
    res["tier1_rotation"] = r_orth.cpu()
    return res


# ---------------------------------------------------------------------------
# 3. Extensible Track A (flow_v) and Track B (flow_simr) Pilot Trainers
# ---------------------------------------------------------------------------

class PilotFlowVModel(_MultiViewModelMixin, nn.Module):
    """FlowSiMRVModel parameterized by flow_kind ('realnvp', 'spline1d', 'spline_pcf', 'gpca')."""

    def __init__(
        self,
        input_dims: List[int],
        latent_dim: int,
        *,
        flow_kind: str = "realnvp",
        num_layers: int = 4,
        hidden_dim: int = 64,
        scale_bound: float = 2.0,
        nsa_w: float = DEFAULT_NSA_W,
        positivity: str = "either",
        sparseness: Optional[Union[float, List[Optional[float]]]] = 0.5,
        use_nsa: bool = True,
        mixing_algorithm: str = DEFAULT_MIXING_ALGORITHM,
        topology: str = DEFAULT_TOPOLOGY,
        path_graph: Optional[Dict[int, List[int]]] = None,
        sign_alignment: str = DEFAULT_SIGN_ALIGNMENT,
        first_layer_mode: str = "scheduled",
        nsa_iterations: int = 1,
    ):
        super().__init__()
        self._init_consensus_state(
            len(input_dims), latent_dim, mixing_algorithm, topology, path_graph,
            None, False, "procrustes_r2", False, None, sign_alignment=sign_alignment,
        )
        positivity = resolve_deep_positivity(positivity)
        self.input_dims = input_dims
        self.latent_dim = latent_dim
        self.is_flow = True
        self.flow_kind = flow_kind
        self.first_layer_mode = first_layer_mode
        self.nsa_iterations = nsa_iterations

        self.linear_encoders = _linear_encoders(
            input_dims, latent_dim, nsa_w, positivity, sparseness,
            use_nsa, first_layer_mode, nsa_iterations,
        )

        if flow_kind == "spline1d":
            self.flows = nn.ModuleList([
                ElementwiseRQSplineFlow(latent_dim, num_layers=2, num_bins=8, scale_bound=scale_bound)
                for _ in input_dims
            ])
        elif flow_kind == "spline_pcf":
            self.flows = nn.ModuleList([
                HybridSplineRealNVPFlow(latent_dim, num_layers=3, hidden_dim=hidden_dim, scale_bound=scale_bound)
                for _ in input_dims
            ])
        else:
            self.flows = nn.ModuleList([
                NormalizingFlow(latent_dim, num_layers, hidden_dim, scale_bound)
                for _ in input_dims
            ])

        if flow_kind == "gpca":
            self.gpca_bases = nn.ModuleList([
                GaussianPCA(latent_dim, latent_dim=latent_dim, sigma=0.2)
                for _ in input_dims
            ])
            self.encoders = nn.ModuleList([
                GaussianPCAEncoderWrapper(fl, gp, latent_dim)
                for fl, gp in zip(self.flows, self.gpca_bases)
            ])
            self.decoders = nn.ModuleList([
                GaussianPCADecoderWrapper(fl, gp)
                for fl, gp in zip(self.flows, self.gpca_bases)
            ])
        else:
            self.gpca_bases = None
            self.encoders = nn.ModuleList([FlowEncoderWrapper(fl, latent_dim) for fl in self.flows])
            self.decoders = nn.ModuleList([FlowDecoderWrapper(fl, latent_dim) for fl in self.flows])

    def retract_linear_encoders(self) -> None:
        with torch.no_grad():
            for enc in self.linear_encoders:
                nsa = getattr(enc, "nsa_linear", None)
                if nsa is not None and hasattr(nsa, "project_"):
                    nsa.project_()
                    continue
                target_v = enc.v.detach()
                if getattr(enc, "v_raw", None) is not None:
                    enc.v_raw.data.copy_(target_v)
                elif nsa is not None and hasattr(nsa, "weight"):
                    nsa.weight.data.copy_(target_v.t())

    def forward(self, x_list: List[torch.Tensor], obs_masks=None):
        projected_zs = [enc(x) for enc, x in zip(self.linear_encoders, x_list)]
        latents = [enc(z) for enc, z in zip(self.encoders, projected_zs)]
        self._last_projected_zs = projected_zs
        if self.training:
            for enc, z in zip(self.encoders, projected_zs):
                z_det, ld_det = enc.flow(z.detach())
                enc.last_z = z_det
                enc.last_log_det = ld_det
        u_shared = self._training_consensus(latents, self.latent_dim)
        reconstructions = []
        for i, (dec, enc) in enumerate(zip(self.decoders, self.linear_encoders)):
            ui = u_shared[i] if isinstance(u_shared, list) else u_shared
            z_recon = dec(ui)
            v = enc.active_training_basis() if self.training else enc.v
            reconstructions.append(z_recon @ v.t())
        return latents, reconstructions, u_shared

    def initialize_weights(self, data_matrices: List[torch.Tensor], **init_kwargs):
        self.initialize_v(data_matrices, self.latent_dim, **init_kwargs)


class PilotFlowSiMRModel(_ModalityWeightingMixin, nn.Module):
    """Ambient p_m -> p_m FlowSiMRModel supporting orthseed, smooth SiLU couplings, and GaussianPCA."""

    def __init__(
        self,
        input_dims: List[int],
        latent_dim: int,
        *,
        flow_kind: str = "realnvp",
        q_frames: Optional[List[torch.Tensor]] = None,
        num_layers: int = 4,
        hidden_dim: int = 64,
        scale_bound: float = 2.0,
        mixing_algorithm: str = DEFAULT_MIXING_ALGORITHM,
        topology: str = DEFAULT_TOPOLOGY,
        path_graph: Optional[Dict[int, List[int]]] = None,
        sign_alignment: str = DEFAULT_SIGN_ALIGNMENT,
    ):
        super().__init__()
        self._init_consensus_state(
            len(input_dims), latent_dim, mixing_algorithm, topology, path_graph,
            None, False, "procrustes_r2", False, None, sign_alignment=sign_alignment,
        )
        self.input_dims = input_dims
        self.latent_dim = latent_dim
        self.is_flow = True
        self.flow_kind = flow_kind

        flows = []
        for idx, dim in enumerate(input_dims):
            qm = q_frames[idx] if q_frames is not None else None
            if flow_kind == "smooth":
                flows.append(SmoothRealNVPFlow(dim, num_layers=num_layers, hidden_dim=hidden_dim,
                                               scale_bound=scale_bound, q_init=qm))
            else:
                base_fl = NormalizingFlow(dim, num_layers=num_layers, hidden_dim=hidden_dim,
                                          scale_bound=scale_bound)
                flows.append(PreOrthogonalFlow(base_fl, qm) if qm is not None else base_fl)
        self.flows = nn.ModuleList(flows)

        if flow_kind == "gpca":
            self.gpca_bases = nn.ModuleList([
                GaussianPCA(dim, latent_dim=min(latent_dim, dim), sigma=0.2)
                for dim in input_dims
            ])
            self.encoders = nn.ModuleList([
                GaussianPCAEncoderWrapper(fl, gp, min(latent_dim, dim))
                for fl, gp, dim in zip(self.flows, self.gpca_bases, input_dims)
            ])
            self.decoders = nn.ModuleList([
                GaussianPCADecoderWrapper(fl, gp)
                for fl, gp in zip(self.flows, self.gpca_bases)
            ])
        else:
            self.gpca_bases = None
            self.encoders = nn.ModuleList([FlowEncoderWrapper(fl, latent_dim) for fl in self.flows])
            self.decoders = nn.ModuleList([FlowDecoderWrapper(fl, latent_dim) for fl in self.flows])

    def forward(self, x_list: List[torch.Tensor], obs_masks=None):
        latents = [enc(x) for enc, x in zip(self.encoders, x_list)]
        u_shared = self._training_consensus(latents, self.latent_dim)
        recons = [dec(u_shared[i] if isinstance(u_shared, list) else u_shared)
                  for i, dec in enumerate(self.decoders)]
        return latents, recons, u_shared


def _train_pilot_flow_loop(
    model: nn.Module,
    dataloader,
    optimizer,
    scheduler,
    *,
    iterations: int,
    sim_weight: float = 1.0,
    energy_type: str = DEFAULT_ENERGY_TYPE,
    warmup_epochs: int = 10,
    beta: float = 0.05,
    gamma: float = 2.0,
    pcf_alpha: float = 0.0,
    jac_varimax_weight: float = 0.0,
    jac_ima_weight: float = 0.0,
    tol: float = DEFAULT_TOL,
    patience: int = 10,
    device: torch.device = torch.device("cpu"),
) -> Tuple[List[float], Dict[str, Any]]:
    """Unified training loop matching pysimlr.flows._train_flow_loop with optional PCF / Jacobian-Varimax."""
    loss_history: List[float] = []
    model._n_fit = getattr(dataloader, "n", None)
    penalty_weights = {"sim": sim_weight, "var": 1.0, "collapse": 1.0, "u_var": 1.0}
    warmup_epochs = resolve_warmup_epochs(warmup_epochs, iterations)
    mse_loss = nn.MSELoss()
    if hasattr(dataloader, "bind_device"):
        dataloader.bind_device(device)

    first_layer_encs = [
        enc for enc in getattr(model, "linear_encoders", ())
        if isinstance(enc, LENDNSAEncoder)
    ]
    stab_start, stab_ramp = _resolve_stabilization_schedule(iterations, warmup_epochs, None, None)
    ramp_epochs = max(1, min(15, max(1, (iterations - warmup_epochs) // 3)))

    monitor = ConvergenceMonitor(tol=float(tol), patience=int(patience), max_steps=iterations)
    best_loss, best_state, best_step = float("inf"), None, None
    all_params = list(model.parameters())
    has_stepped = False

    for step_idx in range(iterations):
        model.train()
        step_losses: List[torch.Tensor] = []
        valid_batches = 0
        _update_first_layer_schedule(model, step_idx, iterations, stab_start, stab_ramp)
        penalty_weights["sim"] = 0.0 if step_idx < warmup_epochs else sim_weight

        reg_ramp = 0.0
        if step_idx >= warmup_epochs:
            reg_ramp = min(1.0, float(step_idx - warmup_epochs + 1) / float(ramp_epochs))

        for batch in dataloader:
            batch_mats = [b.to(device) for b in batch]
            optimizer.zero_grad()
            try:
                for enc in first_layer_encs:
                    enc._step_v_cache = enc.v
                latents, reconstructions, u_shared = model(batch_mats)

                # Flow NLL term (DiagGaussian or GaussianPCA)
                recon_loss = torch.tensor(0.0, device=device)
                gpca_list = getattr(model, "gpca_bases", None)
                for idx_m, enc in enumerate(model.encoders):
                    z = enc.last_z
                    ld = enc.last_log_det
                    if z is None or z.shape[0] == 0:
                        continue
                    if gpca_list is not None:
                        log_prior = gpca_list[idx_m].log_prob(z)
                    else:
                        log_prior = -0.5 * torch.sum(z ** 2, dim=1) - 0.5 * z.shape[1] * math.log(2.0 * math.pi)
                    recon_loss = recon_loss - torch.mean(log_prior + ld)

                sim_loss_total, _ = calculate_sim_loss(
                    latents, u_shared, energy_type, weights=penalty_weights,
                    compute_diagnostics=False, collapse_threshold=DEFAULT_COLLAPSE_THRESHOLD,
                )
                mse_recon_loss = sum(mse_loss(r, x) for r, x in zip(reconstructions, batch_mats))
                total_loss = beta * recon_loss + sim_loss_total + gamma * mse_recon_loss

                if hasattr(model, "linear_encoders"):
                    total_loss = total_loss + 0.05 * sum(orthogonality_defect(enc.v) for enc in model.linear_encoders)
                    total_loss = total_loss + 0.05 * sum(orthogonality_defect(enc.v_raw) for enc in model.linear_encoders)

                # Idea 1: Exact PCF Jacobian row-orthogonality penalty on detached z_lin
                if pcf_alpha > 0.0 and reg_ramp > 0.0 and hasattr(model, "_last_projected_zs"):
                    pcf_sum = torch.tensor(0.0, device=device)
                    for fl_m, z_lin in zip(model.flows, model._last_projected_zs):
                        pcf_sum = pcf_sum + exact_pcf_penalty_bpd(fl_m, z_lin.detach())
                    total_loss = total_loss + (pcf_alpha * reg_ramp) * pcf_sum

                # Tier 2: Shared-subspace IMA + Kaiser Jacobian-Varimax on ambient flow_simr
                if (jac_varimax_weight > 0.0 or jac_ima_weight > 0.0) and reg_ramp > 0.0:
                    for fl_m, x_m in zip(model.flows, batch_mats):
                        ima_def, neg_vmax = shared_jacobian_ima_and_varimax(
                            fl_m, x_m, model.latent_dim, max_samples=32
                        )
                        total_loss = total_loss + reg_ramp * (
                            jac_ima_weight * ima_def + jac_varimax_weight * neg_vmax
                        )
            finally:
                for enc in first_layer_encs:
                    enc._step_v_cache = None

            if not bool(torch.isfinite(total_loss)):
                continue
            total_loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0, foreach=_foreach(all_params))
            if not bool(torch.isfinite(total_norm)):
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.step()
            has_stepped = True
            valid_batches += 1
            step_losses.append(total_loss.detach())

        denom = max(1, valid_batches)
        ep_loss = _sum_on_host(step_losses) / denom
        loss_history.append(ep_loss)

        # Monitor unpenalized / ramped objective after warmup & reg_ramp stabilization
        ready_for_best = step_idx >= (warmup_epochs + (ramp_epochs if (pcf_alpha > 0 or jac_varimax_weight > 0) else 0))
        if ready_for_best and np.isfinite(ep_loss) and ep_loss < best_loss:
            best_loss, best_step = ep_loss, step_idx
            best_state = {k_: v_.detach().clone() for k_, v_ in model.state_dict().items()}
        elif best_state is None and step_idx >= warmup_epochs and np.isfinite(ep_loss):
            best_loss, best_step = ep_loss, step_idx
            best_state = {k_: v_.detach().clone() for k_, v_ in model.state_dict().items()}

        if has_stepped and scheduler is not None:
            scheduler.step(ep_loss)
        if ready_for_best and monitor.update(ep_loss):
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    rep = monitor.report()
    rep.update(returned="best" if best_state is not None else "final", best_step=best_step)
    return loss_history, rep


def fit_pilot_flow_v(
    data_matrices: Sequence[torch.Tensor],
    k: int,
    *,
    variant: str,
    iterations: int = 100,
    batch_size: int = 64,
    learning_rate: float = 5e-4,
    positivity: str = "either",
    sparseness: Any = 0.5,
    topology: str = DEFAULT_TOPOLOGY,
    path_graph: Optional[Dict[int, List[int]]] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Fit a Track A Flow-SiMR-V variant ('base', 'pcf_01', 'pcf_05', 'spline1d', 'spline_pcf', 'gpca')."""
    flow_kind_map = {
        "base": ("realnvp", 0.0),
        "pcf_01": ("realnvp", 0.10),
        "pcf_05": ("realnvp", 0.50),
        "spline1d": ("spline1d", 0.0),
        "spline_pcf": ("spline_pcf", 0.10),
        "gpca": ("gpca", 0.0),
    }
    if variant not in flow_kind_map:
        raise ValueError(f"Unknown Track A variant {variant!r}")
    flow_kind, pcf_alpha = flow_kind_map[variant]

    device = _resolve_device("cpu")
    seed = _resolve_seed(random_state)
    with _seeded_rng(seed, device):
        torch_mats, provenance_list, init_mats, _ = _prepare_deep_inputs(data_matrices, k)
        input_dims = [m.shape[1] for m in torch_mats]
        model = PilotFlowVModel(
            input_dims, k, flow_kind=flow_kind, positivity=positivity, sparseness=sparseness,
            topology=topology, path_graph=path_graph,
        ).to(device)
        model.initialize_weights(
            init_mats, generator=_init_generator("pca", seed),
            initialization_type="pca", varimax=None,
        )
        optimizer = _get_optimizer(model, "adam", learning_rate, 1e-4)
        scheduler = _plateau_scheduler(optimizer)
        dataloader = _shuffled_loader(torch_mats, batch_size, seed)
        warmup = resolve_warmup_epochs(None, iterations)
        loss_h, conv = _train_pilot_flow_loop(
            model, dataloader, optimizer, scheduler, iterations=iterations,
            warmup_epochs=warmup, pcf_alpha=pcf_alpha, device=device,
        )
        model.eval()
        with torch.no_grad():
            v_mats = [enc.v.detach().cpu() for enc in model.linear_encoders]
            v_mats, retr = _finalize_bases(v_mats, positivity, DEFAULT_NSA_W, DEFAULT_ENERGY_TYPE,
                                           encoders=model.linear_encoders)
            final_latents, recons, u_final = model(torch_mats)
            u_agg = _aggregate_shared_consensus(model, final_latents) if isinstance(u_final, list) else u_final
            cons_res = _freeze_consensus(model, final_latents, u_final, u_agg, obs_masks=None)
        cond_inf = FlowConditionalInference(len(input_dims), k)
        cond_inf.fit(final_latents)

    res = {
        "model": model.cpu(),
        "model_type": f"flow_v_{variant}",
        "u": u_agg.cpu(),
        "u_per_modality": [ui.cpu() for ui in u_final] if isinstance(u_final, list) else None,
        "latents": [l.cpu() for l in final_latents],
        "missing_rows": "impute",
        "obs_mask": None,
        "reconstructions": [r.cpu() for r in recons],
        "loss_history": loss_h,
        "convergence": conv,
        "scale_list": ["centerAndScale"],
        "provenance_list": provenance_list,
        "sparseness": sparseness,
        "random_state": seed,
        "consensus_map_residual": cons_res,
        "cond_inference": cond_inf,
        "v": v_mats,
        "retraction_diagnostics": retr,
        "errors": _recon_errors(torch_mats, recons),
    }
    res.update(basis_rank_report(v_mats, k, warn=False))
    # Populate OLEE structure, loadings, and loading_rotation using pysimlr.loadings
    s_tr = structure_coefficients(torch_mats, res["u"])
    r_load = varimax_rotation(s_tr)
    s_rot = [(s.double() @ r_load).float() for s in s_tr]
    loads, _ = olee_loadings(s_rot, positivity=positivity)
    res["structure"] = s_rot
    res["loadings"] = loads
    res["loading_rotation"] = r_load.float()
    return res


def fit_pilot_flow_simr(
    data_matrices: Sequence[torch.Tensor],
    k: int,
    *,
    variant: str,
    iterations: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    topology: str = DEFAULT_TOPOLOGY,
    path_graph: Optional[Dict[int, List[int]]] = None,
    eval_view_indices: Optional[Sequence[int]] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Fit a Track B Flow-SiMR variant ('base', 'orthseed', 'tier2_jvarimax', 'tier2_orthseed', 'gpca')."""
    cfg = {
        "base": ("realnvp", False, 0.0, 0.0, False),
        "orthseed": ("realnvp", True, 0.0, 0.0, True),
        "tier2_jvarimax": ("smooth", False, 1.5, 0.25, True),
        "tier2_orthseed": ("smooth", True, 1.5, 0.25, True),
        "gpca": ("gpca", False, 0.0, 0.0, True),
    }
    if variant not in cfg:
        raise ValueError(f"Unknown Track B variant {variant!r}")
    flow_kind, use_orth_init, jvmax_w, jima_w, post_align = cfg[variant]

    device = _resolve_device("cpu")
    seed = _resolve_seed(random_state)
    with _seeded_rng(seed, device):
        torch_mats, provenance_list, _, _ = _prepare_deep_inputs(data_matrices, k)
        input_dims = [m.shape[1] for m in torch_mats]
        q_frames = build_varimax_pca_orthogonal_frames(torch_mats, k) if use_orth_init else None
        model = PilotFlowSiMRModel(
            input_dims, k, flow_kind=flow_kind, q_frames=q_frames,
            topology=topology, path_graph=path_graph,
        ).to(device)
        optimizer = _get_optimizer(model, "adam", learning_rate, 1e-4)
        scheduler = _plateau_scheduler(optimizer)
        dataloader = _shuffled_loader(torch_mats, batch_size, seed)
        warmup = resolve_warmup_epochs(None, iterations)
        loss_h, conv = _train_pilot_flow_loop(
            model, dataloader, optimizer, scheduler, iterations=iterations,
            warmup_epochs=warmup, jac_varimax_weight=jvmax_w, jac_ima_weight=jima_w,
            device=device,
        )
        model.eval()
        with torch.no_grad():
            final_latents, recons, u_final = model(torch_mats)
            u_agg = _aggregate_shared_consensus(model, final_latents) if isinstance(u_final, list) else u_final
            cons_res = _freeze_consensus(model, final_latents, u_final, u_agg, obs_masks=None)
        cond_inf = FlowConditionalInference(len(input_dims), k)
        cond_inf.fit(final_latents)

    res = {
        "model": model.cpu(),
        "model_type": f"flow_simr_{variant}",
        "u": u_agg.cpu(),
        "u_per_modality": [ui.cpu() for ui in u_final] if isinstance(u_final, list) else None,
        "latents": [l.cpu() for l in final_latents],
        "missing_rows": "impute",
        "obs_mask": None,
        "reconstructions": [r.cpu() for r in recons],
        "loss_history": loss_h,
        "convergence": conv,
        "scale_list": ["centerAndScale"],
        "provenance_list": provenance_list,
        "sparseness": None,
        "random_state": seed,
        "consensus_map_residual": cons_res,
        "cond_inference": cond_inf,
        "errors": _recon_errors(torch_mats, recons),
    }
    if post_align and flow_kind != "gpca" and k > 1:
        res = apply_tier1_post_flow_varimax(
            res, data_matrices, k=k, eval_view_indices=eval_view_indices, mode="structure"
        )
        res["model_type"] = f"flow_simr_{variant}"
    return res


# ---------------------------------------------------------------------------
# 4. Datasets & Diagnostic Metrics (oos_r2, Coh(L), Coh(S), Coh(J), Spearman rho)
# ---------------------------------------------------------------------------

def build_localized_warp_case(n_samples: int = 450, seed: int = 42) -> Dict[str, Any]:
    """3-view nonlinear manifold with sparse localized ground-truth bases V_m* (k=3)."""
    gen = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    k = 3
    p_list = [60, 48, 36]
    u_true = torch.from_numpy(rng.standard_normal((n_samples, k))).float()
    # Outcome concentrated primarily on factor 0 and factor 1 so axis rotation matters for trees/single-axis
    y = (1.25 * u_true[:, 0] - 0.75 * u_true[:, 1] + 0.15 * torch.from_numpy(rng.standard_normal(n_samples)).float())

    data_mats, v_true = [], []
    for m_idx, p in enumerate(p_list):
        v_sp = _sparse_signed_basis(k, p, true_sparsity=0.75, generator=gen)  # (k, p), 25% active per factor
        # Normalize columns
        v_sp = v_sp / v_sp.norm(dim=1, keepdim=True).clamp_min(1e-8)
        u_next = torch.roll(u_true, shifts=1, dims=1)
        if m_idx == 0:
            z_warp = torch.sinh(0.85 * u_true) + 0.30 * (u_true.pow(2) - 1.0) + 0.18 * (u_true * u_next)
        elif m_idx == 1:
            z_warp = 0.75 * u_true + 0.25 * u_true.pow(3) + 0.22 * torch.sin(1.5 * u_next)
        else:
            z_warp = (torch.exp(0.65 * u_true) - math.exp(0.65 ** 2 / 2.0)) / 0.65 + 0.15 * (u_next.pow(2) - 1.0)
        u_priv = torch.from_numpy(rng.standard_normal((n_samples, 2))).float() * 0.45
        v_priv = torch.from_numpy(rng.standard_normal((2, p))).float() / math.sqrt(p)
        noise = torch.from_numpy(rng.standard_normal((n_samples, p))).float() * 0.30
        x = z_warp @ v_sp + u_priv @ v_priv + noise
        data_mats.append(x.float())
        v_true.append(v_sp.t().float())

    return {
        "name": "Synth_LocalWarp_k3",
        "data": data_mats,
        "true_u": u_true,
        "true_v": v_true,
        "outcome": y.numpy(),
        "shared_k": k,
        "needs_scaling": True,
    }


def build_skew_k1_case(n_samples: int = 400, seed: int = 42) -> Dict[str, Any]:
    """2-view rank-1 heavy-tailed/skewed monotone nonlinear benchmark (tests k=1 flow capacity)."""
    gen = torch.Generator().manual_seed(seed + 100)
    rng = np.random.default_rng(seed + 100)
    p_list = [40, 30]
    u_true = torch.from_numpy(rng.standard_normal((n_samples, 1))).float()
    y = (u_true[:, 0] + 0.15 * torch.from_numpy(rng.standard_normal(n_samples)).float()).numpy()

    data_mats, v_true = [], []
    for m_idx, p in enumerate(p_list):
        v_sp = _sparse_signed_basis(1, p, true_sparsity=0.75, generator=gen)
        v_sp = v_sp / v_sp.norm(dim=1, keepdim=True).clamp_min(1e-8)
        if m_idx == 0:
            # Heavy exponential right-skewed sensor (log-normal biomarker)
            w_u = (torch.exp(0.95 * u_true) - math.exp(0.95 ** 2 / 2.0)) / 1.2
        else:
            # Cubic/sinh heavy-tailed sensor
            w_u = torch.sinh(1.05 * u_true) / 1.4
        u_priv = torch.from_numpy(rng.standard_normal((n_samples, 2))).float() * 0.40
        v_priv = torch.from_numpy(rng.standard_normal((2, p))).float() / math.sqrt(p)
        noise = torch.from_numpy(rng.standard_normal((n_samples, p))).float() * 0.25
        x = w_u @ v_sp + u_priv @ v_priv + noise
        data_mats.append(x.float())
        v_true.append(v_sp.t().float())

    return {
        "name": "Synth_Skew_k1",
        "data": data_mats,
        "true_u": u_true,
        "true_v": v_true,
        "outcome": y,
        "shared_k": 1,
        "needs_scaling": True,
    }


def load_diabetes_case(k: int = 2) -> Dict[str, Any]:
    """Two-view Diabetes clinical benchmark (Vitals [:, :5] + Blood Serums [:, 5:])."""
    d = load_diabetes()
    x_raw = d.data.astype(np.float32)
    y_raw = d.target.astype(np.float64)
    return {
        "name": f"Diabetes_k{k}",
        "data": [torch.tensor(x_raw[:, :5]), torch.tensor(x_raw[:, 5:])],
        "true_u": None,
        "true_v": None,
        "outcome": y_raw,
        "shared_k": k,
        "needs_scaling": True,
    }


def _hoyer_columns(mat: np.ndarray) -> float:
    p, k = mat.shape
    if p <= 1:
        return 0.0
    vals = []
    for j in range(k):
        c = np.abs(mat[:, j])
        l2 = float(np.linalg.norm(c))
        vals.append(0.0 if l2 < 1e-12 else float((math.sqrt(p) - c.sum() / l2) / (math.sqrt(p) - 1.0)))
    return float(np.mean(vals))


def _flow_jacobian_coherence(flow_mod: nn.Module, x_in: torch.Tensor, k: int, max_n: int = 48) -> float:
    """Mean off-diagonal |cosine| among the first k encoder Jacobian rows on test inputs."""
    if k < 2 or x_in.shape[0] == 0:
        return 0.0
    n_s = min(int(max_n), int(x_in.shape[0]))
    with torch.enable_grad():
        xr = x_in[:n_s].detach().float().requires_grad_(True)
        out = flow_mod(xr)
        z = out[0] if isinstance(out, tuple) else out
        k_eff = min(int(k), int(z.shape[1]))
        if k_eff < 2:
            return 0.0
        rows = []
        for j in range(k_eff):
            gj = torch.autograd.grad(z[:, j].sum(), xr, retain_graph=True)[0].detach()
            gj = gj / gj.norm(dim=1, keepdim=True).clamp_min(1e-12)
            rows.append(gj)
        # (n_s, k_eff, dim)
        r_mat = torch.stack(rows, dim=1)
        gram = torch.bmm(r_mat, r_mat.transpose(1, 2)).abs()
        eye = torch.eye(k_eff, device=gram.device).unsqueeze(0)
        off = (gram * (1.0 - eye)).amax(dim=(1, 2))
        return float(off.mean().item())


def _best_single_axis_r2(z_tr: np.ndarray, y_tr: np.ndarray, z_te: np.ndarray, y_te: np.ndarray) -> float:
    best = -float("inf")
    for j in range(z_tr.shape[1]):
        ols = LinearRegression().fit(z_tr[:, [j]], y_tr)
        best = max(best, float(r2_score(y_te, ols.predict(z_te[:, [j]]))))
    return float(best)


def score_fit_on_split(
    res: Dict[str, Any],
    train_mats: List[torch.Tensor],
    test_mats: List[torch.Tensor],
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    k: int,
    eval_view_indices: Optional[Sequence[int]] = None,
    true_u_test: Optional[torch.Tensor] = None,
    true_v: Optional[List[torch.Tensor]] = None,
    y_multi_train: Optional[np.ndarray] = None,
    y_multi_test: Optional[np.ndarray] = None,
    positivity: str = "either",
    track: str = "A",
    rf_trees: int = 60,
) -> Dict[str, float]:
    """Compute uniform out-of-sample R^2, Axis RF R^2, Coh(L), Coh(S), Coh(J), and Spearman rho."""
    eval_idx = list(range(len(train_mats))) if eval_view_indices is None else list(eval_view_indices)

    is_simlr_linear = "model" not in res
    if is_simlr_linear:
        pred_tr = pysimlr.predict(res, train_mats)
        pred_te = pysimlr.predict(res, test_mats)
        z_tr_views = [pred_tr["latents"][i].detach().cpu().float() for i in eval_idx]
        z_te_views = [pred_te["latents"][i].detach().cpu().float() for i in eval_idx]
        u_tr_eval = pred_tr["u"].detach().cpu().float()
        u_te_eval = pred_te["u"].detach().cpu().float()
    else:
        # Score evaluated views directly through each view's trained encoder (works even when outcome view 3 is withheld!)
        from pysimlr.api import _preprocess_views
        xs_tr_sc, _ = _preprocess_views(
            [train_mats[i] for i in eval_idx],
            res["scale_list"],
            [res["provenance_list"][i] for i in eval_idx],
        )
        xs_te_sc, _ = _preprocess_views(
            [test_mats[i] for i in eval_idx],
            res["scale_list"],
            [res["provenance_list"][i] for i in eval_idx],
        )
        model = res["model"]
        model.eval()
        with torch.no_grad():
            if hasattr(model, "linear_encoders") and getattr(model, "is_flow", False):
                lin_tr = [model.linear_encoders[i](xs_tr_sc[idx]) for idx, i in enumerate(eval_idx)]
                lin_te = [model.linear_encoders[i](xs_te_sc[idx]) for idx, i in enumerate(eval_idx)]
                z_tr_views = [model.encoders[i](lin_tr[idx]).cpu() for idx, i in enumerate(eval_idx)]
                z_te_views = [model.encoders[i](lin_te[idx]).cpu() for idx, i in enumerate(eval_idx)]
            elif not getattr(model, "is_flow", False):  # LEND (linear encoders stored on model.encoders)
                z_tr_views = [model.encoders[i](xs_tr_sc[idx]).cpu() for idx, i in enumerate(eval_idx)]
                z_te_views = [model.encoders[i](xs_te_sc[idx]).cpu() for idx, i in enumerate(eval_idx)]
                lin_tr, lin_te = z_tr_views, z_te_views
            else:  # Track B flow_simr
                z_tr_views = [model.encoders[i](xs_tr_sc[idx]).cpu() for idx, i in enumerate(eval_idx)]
                z_te_views = [model.encoders[i](xs_te_sc[idx]).cpu() for idx, i in enumerate(eval_idx)]
                lin_tr, lin_te = None, None

        if eval_view_indices is None:
            pred_tr_full = predict_deep(train_mats, res)
            pred_te_full = predict_deep(test_mats, res)
            u_tr_eval = pred_tr_full["u"].detach().cpu().float()
            u_te_eval = pred_te_full["u"].detach().cpu().float()
        else:
            u_tr_eval = torch.cat(z_tr_views, dim=1)
            u_te_eval = torch.cat(z_te_views, dim=1)

    # Concatenated per-view post-flow latents (standardized on train split)
    z_cat_tr = torch.cat(z_tr_views, dim=1)
    z_cat_te = torch.cat(z_te_views, dim=1)
    mu_z = z_cat_tr.mean(0, keepdim=True)
    sd_z = z_cat_tr.std(0, keepdim=True).clamp_min(1e-6)
    z_cat_tr_std = (z_cat_tr - mu_z) / sd_z
    z_cat_te_std = (z_cat_te - mu_z) / sd_z

    # Primary OOS R^2 (on concatenated evaluated-view latents for Protocol B; consensus u AND concat on full-view tasks)
    probe_tr = z_cat_tr_std if eval_view_indices is not None else u_tr_eval
    probe_te = z_cat_te_std if eval_view_indices is not None else u_te_eval
    lin_cv = cross_val_metrics(probe_tr, y_train, probe_te, y_test, is_classification=False)
    rf_cv = axis_sensitive_cross_val_metrics(
        z_cat_tr_std, y_train, z_cat_te_std, y_test, is_classification=False, n_estimators=rf_trees, random_state=0
    )
    best_1d_r2 = _best_single_axis_r2(
        z_cat_tr_std.numpy(), y_train, z_cat_te_std.numpy(), y_test
    )
    concat_oos_r2 = float(cross_val_metrics(z_cat_tr_std, y_train, z_cat_te_std, y_test)["test"])

    mean11_r2 = float("nan")
    if y_multi_train is not None and y_multi_test is not None:
        rcv = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(z_cat_tr_std.numpy(), y_multi_train)
        mean11_r2 = float(r2_score(y_multi_test, rcv.predict(z_cat_te_std.numpy()), multioutput="uniform_average"))

    # Strictly linear X_m V_m probe (Track A models with V_m)
    xv_oos_r2 = float("nan")
    frame_def_v = float("nan")
    sup_rec_v = float("nan")
    if res.get("v") is not None:
        v_eval = [torch.as_tensor(res["v"][i]).detach().cpu().float() for i in eval_idx]
        xv_tr = torch.cat([train_mats[i].cpu().float() @ v_eval[idx] for idx, i in enumerate(eval_idx)], dim=1)
        xv_te = torch.cat([test_mats[i].cpu().float() @ v_eval[idx] for idx, i in enumerate(eval_idx)], dim=1)
        mu_xv = xv_tr.mean(0, keepdim=True)
        sd_xv = xv_tr.std(0, keepdim=True).clamp_min(1e-6)
        xv_oos_r2 = float(cross_val_metrics((xv_tr - mu_xv) / sd_xv, y_train, (xv_te - mu_xv) / sd_xv, y_test)["test"])
        frame_def_v = float(np.mean([float(orthogonality_defect(v, normalized=True)) for v in v_eval]))
        if true_v is not None:
            sup_rec_v = support_recovery_score(v_eval, [true_v[i] for i in eval_idx])

    # Structure coefficients S_m on held-out test views vs per-view test latents Z_m_te
    s_unrot = [structure_coefficients([test_mats[i].cpu().float()], z_te_views[idx])[0]
               for idx, i in enumerate(eval_idx)]
    r_vm = varimax_rotation(s_unrot)
    s_rot = [(s.double() @ r_vm).float() for s in s_unrot]
    loads_te, _ = olee_loadings(s_rot, positivity=positivity)

    coh_L = float(np.mean([column_coherence(lm) for lm in loads_te]))
    coh_S_unrot = float(np.mean([column_coherence(sm) for sm in s_unrot]))
    coh_S_rot = float(np.mean([column_coherence(sm) for sm in s_rot]))
    hoyer_S_unrot = float(np.mean([_hoyer_columns(sm.numpy()) for sm in s_unrot]))

    # Support recovery from top-quartile thresholded structure coefficients (comparable across Track A & Track B!)
    sup_rec_S = float("nan")
    if true_v is not None:
        s_thresh = []
        for sm in s_unrot:
            sm_abs = sm.abs()
            q75 = torch.quantile(sm_abs, 0.75, dim=0, keepdim=True)
            s_thresh.append(torch.where(sm_abs >= q75, sm, torch.zeros_like(sm)))
        sup_rec_S = support_recovery_score(s_thresh, [true_v[i] for i in eval_idx])

    # Coordinate Spearman rho & Flow Jacobian Coherence Coh(J)
    rhos: List[float] = []
    coh_j_list: List[float] = []
    if is_simlr_linear or not getattr(res["model"], "is_flow", False):
        rhos = [1.0]
        coh_j_list = [0.0]
    elif hasattr(res["model"], "linear_encoders"):
        r_frame = res.get("loading_rotation", torch.eye(k)).detach().cpu().numpy()
        for idx, i in enumerate(eval_idx):
            a_mat = lin_te[idx].detach().cpu().numpy() @ r_frame
            b_mat = z_te_views[idx].detach().cpu().numpy() @ r_frame
            for j in range(a_mat.shape[1]):
                val = spearmanr(a_mat[:, j], b_mat[:, j]).correlation
                if np.isfinite(val):
                    rhos.append(float(val))
            if getattr(res["model"], "is_flow", False):
                coh_j_list.append(_flow_jacobian_coherence(res["model"].flows[i], lin_te[idx], k))
            else:
                coh_j_list.append(0.0)
    else:
        # Track B: Spearman rho between tangent projection X_te @ G_mean_tr and post-flow Z_te, plus Jacobian coherence
        from pysimlr.api import _preprocess_views
        xs_tr_sc, _ = _preprocess_views(
            [train_mats[i] for i in eval_idx], res["scale_list"], [res["provenance_list"][i] for i in eval_idx]
        )
        xs_te_sc, _ = _preprocess_views(
            [test_mats[i] for i in eval_idx], res["scale_list"], [res["provenance_list"][i] for i in eval_idx]
        )
        for idx, i in enumerate(eval_idx):
            with torch.enable_grad():
                xm_tr = xs_tr_sc[idx][:64].detach().float().requires_grad_(True)
                zm_tr = res["model"].encoders[i](xm_tr)
                g_cols = [
                    torch.autograd.grad(zm_tr[:, j].sum(), xm_tr, retain_graph=True)[0].detach().mean(0)
                    for j in range(zm_tr.shape[1])
                ]
                g_mean = torch.stack(g_cols, dim=-1)  # (p_i, k)
            lin_tan = (xs_te_sc[idx] @ g_mean).detach().cpu().numpy()
            z_np = z_te_views[idx].detach().cpu().numpy()
            for j in range(z_np.shape[1]):
                val = spearmanr(lin_tan[:, j], z_np[:, j]).correlation
                if np.isfinite(val):
                    rhos.append(float(val))
            coh_j_list.append(_flow_jacobian_coherence(res["model"].flows[i], xs_te_sc[idx], k))

    cmc_u = float("nan")
    if true_u_test is not None:
        cmc_u = float(latent_recovery_score(u_te_eval, true_u_test.cpu().float()))

    # Conditional cross-view imputation R^2 (view 0 -> view 1 on 2-view cohorts with FlowConditionalInference)
    cond_imp_r2 = float("nan")
    if res.get("cond_inference") is not None and len(eval_idx) == 2 and eval_view_indices is None:
        from pysimlr.api import _preprocess_views
        xs_te_sc, _ = _preprocess_views(test_mats, res["scale_list"], res["provenance_list"])
        cond = res["cond_inference"]
        model = res["model"]
        with torch.no_grad():
            pred_z1 = cond.predict_conditional(observed_idx=0, target_idx=1, observed_z=z_te_views[0])
            if hasattr(model, "linear_encoders"):
                x1_imp = model.decoders[1](pred_z1) @ model.linear_encoders[1].v.t()
            else:
                x1_imp = model.decoders[1](pred_z1)
        # Score imputation on last feature (e.g., s5 serum triglycerides on Diabetes) & mean across view 1
        y_s5_true = xs_te_sc[1][:, -1].numpy()
        y_s5_pred = x1_imp[:, -1].cpu().numpy()
        cond_imp_r2 = float(r2_score(y_s5_true, y_s5_pred))

    return {
        "oos_r2": float(lin_cv["test"]),
        "concat_oos_r2": concat_oos_r2,
        "xv_oos_r2": xv_oos_r2,
        "axis_oos_r2": float(rf_cv["test"]),
        "best_single_axis_r2": best_1d_r2,
        "mean11_r2": mean11_r2,
        "coh_L": coh_L,
        "coh_S_unrot": coh_S_unrot,
        "coh_S_rot": coh_S_rot,
        "coh_J_flow": float(np.mean(coh_j_list)) if coh_j_list else float("nan"),
        "coord_spearman_rho": float(np.mean(rhos)) if rhos else float("nan"),
        "min_coord_spearman_rho": float(np.min(rhos)) if rhos else float("nan"),
        "hoyer_S": hoyer_S_unrot,
        "frame_defect_v": frame_def_v,
        "support_recovery_v": sup_rec_v,
        "support_recovery_S": sup_rec_S,
        "cmc_latent_u": cmc_u,
        "cond_impute_r2": cond_imp_r2,
    }


# ---------------------------------------------------------------------------
# 5. Stage 0 Self-Checks & Stage 1 Noise Floor
# ---------------------------------------------------------------------------

def run_stage0_contract_checks() -> Dict[str, Any]:
    """Verify exact mathematical contracts before running any benchmark experiment."""
    torch.manual_seed(7)
    checks: Dict[str, Any] = {}

    # 1. ElementwiseRQSplineFlow identity init, exact inverse, exact autograd log-det, PCF==0, and k=1 nonlinearity
    sp3 = ElementwiseRQSplineFlow(dim=3, num_layers=2, num_bins=8)
    x3 = torch.linspace(-2.8, 2.8, 15).unsqueeze(1).repeat(1, 3) + 0.2 * torch.randn(15, 3)
    y3_id, ld3_id = sp3(x3)
    id_err = float((y3_id - x3).abs().max().item())
    assert id_err < 1e-5, f"Spline identity init error {id_err}"

    for p in sp3.parameters():
        p.data.add_(0.35 * torch.randn_like(p))
    z3, ld3 = sp3(x3)
    x3_rec, ld3_inv = sp3.inverse_and_log_det(z3)
    inv_err = float((x3_rec - x3).abs().max().item())
    ld_roundtrip = float((ld3 + ld3_inv).abs().max().item())
    # Autograd Jacobian determinant check on sample 0
    jac0 = torch.autograd.functional.jacobian(lambda inp: sp3(inp.unsqueeze(0))[0].squeeze(0), x3[0])
    ld_exact0 = float(torch.linalg.slogdet(jac0)[1].item())
    ld_diff0 = abs(float(ld3[0].item()) - ld_exact0)
    off_diag_jac0 = float((jac0 - torch.diag(torch.diag(jac0))).abs().max().item())
    pcf_sp3 = float(exact_pcf_penalty_bpd(sp3, x3).item())
    assert inv_err < 1e-4 and ld_diff0 < 1e-4 and off_diag_jac0 < 1e-6 and abs(pcf_sp3) < 1e-5

    # k=1 nonlinearity check: Spline vs RealNVP
    sp1 = ElementwiseRQSplineFlow(dim=1, num_layers=2, num_bins=8)
    nf1 = NormalizingFlow(dim=1, num_layers=4, hidden_dim=16)
    for p in sp1.parameters():
        p.data.add_(0.45 * torch.randn_like(p))
    for p in nf1.parameters():
        p.data.normal_(0.0, 0.45)
    grid1 = torch.linspace(-2.5, 2.5, 25).unsqueeze(1)
    d2_sp1 = float(torch.diff(sp1(grid1)[0].squeeze(), n=2).abs().max().item())
    d2_nf1 = float(torch.diff(nf1(grid1)[0].squeeze(), n=2).abs().max().item())
    assert d2_sp1 > 0.02 and d2_nf1 < 1e-5, f"Expected nonlinear spline ({d2_sp1}) vs linear RealNVP ({d2_nf1}) at k=1"

    # 2. PCF non-negativity on sheared RealNVP
    nf3 = NormalizingFlow(dim=3, num_layers=3, hidden_dim=16)
    for m in nf3.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.15)
    pcf_nf3 = float(exact_pcf_penalty_bpd(nf3, x3).item())
    assert pcf_nf3 >= -1e-5, f"Hadamard PCF must be >= 0, got {pcf_nf3}"

    # 3. Tier 1 exact NLL, reconstruction, and subspace R^2 invariance
    toy_views = [torch.randn(36, 7), torch.randn(36, 6)]
    toy_y = np.random.default_rng(0).standard_normal(36)
    base_fs = flow_simr(toy_views, k=2, iterations=3, batch_size=18, random_state=11)
    t1_struct = apply_tier1_post_flow_varimax(base_fs, toy_views, k=2, mode="structure")
    t1_jac = apply_tier1_post_flow_varimax(base_fs, toy_views, k=2, mode="jacobian")

    recon_diff_t1 = float(max(
        (r_a - r_b).abs().max().item()
        for r_a, r_b in zip(base_fs["reconstructions"], t1_struct["reconstructions"])
    ))
    ols_base = LinearRegression().fit(base_fs["u"].numpy(), toy_y).score(base_fs["u"].numpy(), toy_y)
    ols_t1 = LinearRegression().fit(t1_struct["u"].numpy(), toy_y).score(t1_struct["u"].numpy(), toy_y)
    ols_t1j = LinearRegression().fit(t1_jac["u"].numpy(), toy_y).score(t1_jac["u"].numpy(), toy_y)
    assert recon_diff_t1 < 1e-4 and abs(ols_base - ols_t1) < 1e-5 and abs(ols_base - ols_t1j) < 1e-5

    # 4. Orthogonal Varimax-PCA frame check
    q_frames = build_varimax_pca_orthogonal_frames(toy_views, k=2)
    q_orth_err = float(max((q.t() @ q - torch.eye(q.shape[0])).abs().max().item() for q in q_frames))
    assert q_orth_err < 1e-5

    checks.update({
        "spline_identity_err": id_err,
        "spline_inverse_err": inv_err,
        "spline_logdet_autograd_err": ld_diff0,
        "spline_offdiag_jacobian": off_diag_jac0,
        "spline_pcf_bpd": pcf_sp3,
        "k1_spline_max_2nd_diff": d2_sp1,
        "k1_realnvp_max_2nd_diff": d2_nf1,
        "sheared_realnvp_pcf_bpd": pcf_nf3,
        "tier1_recon_max_diff": recon_diff_t1,
        "tier1_ols_r2_abs_diff": float(abs(ols_base - ols_t1)),
        "orth_frame_max_err": q_orth_err,
        "all_passed": True,
    })
    return checks


def run_stage1_noise_floor(iters: int, n_repeats: int = 3) -> Dict[str, Any]:
    """Measure same-seed determinism & wall-clock timing floor (min/median/max over isolated repeats)."""
    case = load_diabetes_case(k=2)
    tr_idx, te_idx = split_indices(len(case["outcome"]), seed=42)
    sc0 = StandardScaler().fit(case["data"][0][tr_idx].numpy())
    sc1 = StandardScaler().fit(case["data"][1][tr_idx].numpy())
    tr_views = [
        torch.tensor(sc0.transform(case["data"][0][tr_idx].numpy())).float(),
        torch.tensor(sc1.transform(case["data"][1][tr_idx].numpy())).float(),
    ]
    te_views = [
        torch.tensor(sc0.transform(case["data"][0][te_idx].numpy())).float(),
        torch.tensor(sc1.transform(case["data"][1][te_idx].numpy())).float(),
    ]
    y_tr, y_te = case["outcome"][tr_idx.numpy()], case["outcome"][te_idx.numpy()]

    times_fv, r2s_fv = [], []
    times_sp, r2s_sp = [], []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        res_b = fit_pilot_flow_v(tr_views, k=2, variant="base", iterations=iters, sparseness=None, random_state=42)
        times_fv.append(time.perf_counter() - t0)
        r2s_fv.append(score_fit_on_split(res_b, tr_views, te_views, y_tr, y_te, k=2)["oos_r2"])

        t0 = time.perf_counter()
        res_s = fit_pilot_flow_v(tr_views, k=2, variant="spline1d", iterations=iters, sparseness=None, random_state=42)
        times_sp.append(time.perf_counter() - t0)
        r2s_sp.append(score_fit_on_split(res_s, tr_views, te_views, y_tr, y_te, k=2)["oos_r2"])

    # Also check bit-parity against native pysimlr.flows.flow_simr_v on same inputs
    res_native = flow_simr_v(tr_views, k=2, iterations=iters, sparseness=None, random_state=42)
    parity_u_diff = float((res_native["u"] - res_b["u"]).abs().max().item())

    return {
        "n_repeats": n_repeats,
        "flow_v_base_sec_min_med_max": [float(np.min(times_fv)), float(np.median(times_fv)), float(np.max(times_fv))],
        "flow_v_spline1d_sec_min_med_max": [float(np.min(times_sp)), float(np.median(times_sp)), float(np.max(times_sp))],
        "same_seed_r2_span_base": float(np.max(r2s_fv) - np.min(r2s_fv)),
        "same_seed_r2_span_spline1d": float(np.max(r2s_sp) - np.min(r2s_sp)),
        "native_flow_simr_v_u_max_diff": parity_u_diff,
    }


# ---------------------------------------------------------------------------
# 6. Main Benchmark Suite (Track A & Track B across Synthetic + Diabetes + ADNI-MCI)
# ---------------------------------------------------------------------------

def _prepare_standard_split(case: Dict[str, Any], seed: int, max_subjects: int = 0):
    data_all = case["data"]
    y_all = np.asarray(case["outcome"], dtype=np.float64)
    u_all = case.get("true_u")
    n_tot = data_all[0].shape[0]
    if max_subjects > 0 and n_tot > max_subjects:
        sub = np.random.default_rng(seed + 999).choice(n_tot, size=max_subjects, replace=False)
        sub.sort()
        data_all = [m[sub] for m in data_all]
        y_all = y_all[sub]
        if u_all is not None:
            u_all = u_all[sub]
        n_tot = max_subjects

    tr_idx, te_idx = split_indices(n_tot, seed=seed, train_frac=0.7)
    tr_mats = [m[tr_idx] for m in data_all]
    te_mats = [m[te_idx] for m in data_all]
    if case.get("needs_scaling", True):
        sc_tr, sc_te = [], []
        for m_tr, m_te in zip(tr_mats, te_mats):
            sc = StandardScaler().fit(m_tr.numpy())
            sc_tr.append(torch.tensor(sc.transform(m_tr.numpy())).float())
            sc_te.append(torch.tensor(sc.transform(m_te.numpy())).float())
        tr_mats, te_mats = sc_tr, sc_te
    u_te = u_all[te_idx] if u_all is not None else None
    return tr_mats, te_mats, y_all[tr_idx.numpy()], y_all[te_idx.numpy()], u_te


def run_benchmark_suite(
    *,
    iters: int,
    n_seeds: int,
    n_adni_seeds: int,
    max_subjects: int,
    rf_trees: int = 60,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    # --- A. Standard & Synthetic + Diabetes Tasks ---
    tasks = [
        ("Synth_Poly_k3", lambda s: {
            **build_nonlinear_shared_case(
                n_samples=max_subjects if max_subjects > 0 else 450,
                shared_k=3, p_list=[80, 60, 40], noise_scale=0.35, seed=s,
                regime="polynomial", private_scale=0.5,
            ),
            "name": "Synth_Poly_k3",
            "needs_scaling": True,
        }, True),
        ("Synth_LocalWarp_k3", lambda s: build_localized_warp_case(
            n_samples=max_subjects if max_subjects > 0 else 450, seed=s
        ), True),
        ("Synth_Skew_k1", lambda s: build_skew_k1_case(
            n_samples=max_subjects if max_subjects > 0 else 400, seed=s
        ), False),
        ("Diabetes_k2", lambda _s: load_diabetes_case(k=2), True),
        ("Diabetes_k1", lambda _s: load_diabetes_case(k=1), False),
    ]

    for task_name, builder, run_track_b in tasks:
        for s_off in range(n_seeds):
            seed = 42 + s_off
            case = builder(seed)
            k = int(case["shared_k"])
            tr_mats, te_mats, y_tr, y_te, u_te = _prepare_standard_split(case, seed, max_subjects=max_subjects)
            sp_list = [0.5 if m.shape[1] > 15 else None for m in tr_mats]
            lin_iters = min(60, max(2, iters))

            # Track A: Anchors + Flow-SiMR-V variants
            track_a_specs: List[Tuple[str, Callable[[], Dict[str, Any]]]] = [
                ("simlr", lambda: pysimlr.fit("simlr", tr_mats, k, sparseness=sp_list,
                                              positivity="either", iterations=lin_iters, random_state=seed)),
                ("lend", lambda: pysimlr.fit("lend", tr_mats, k, sparseness=sp_list,
                                             positivity="either", iterations=iters, random_state=seed)),
                ("flow_v_base", lambda: fit_pilot_flow_v(tr_views := tr_mats, k, variant="base",
                                                         iterations=iters, sparseness=sp_list, random_state=seed)),
                ("flow_v_pcf_01", lambda: fit_pilot_flow_v(tr_mats, k, variant="pcf_01",
                                                           iterations=iters, sparseness=sp_list, random_state=seed)),
                ("flow_v_pcf_05", lambda: fit_pilot_flow_v(tr_mats, k, variant="pcf_05",
                                                           iterations=iters, sparseness=sp_list, random_state=seed)),
                ("flow_v_spline1d", lambda: fit_pilot_flow_v(tr_mats, k, variant="spline1d",
                                                             iterations=iters, sparseness=sp_list, random_state=seed)),
                ("flow_v_spline_pcf", lambda: fit_pilot_flow_v(tr_mats, k, variant="spline_pcf",
                                                               iterations=iters, sparseness=sp_list, random_state=seed)),
                ("flow_v_gpca", lambda: fit_pilot_flow_v(tr_mats, k, variant="gpca",
                                                         iterations=iters, sparseness=sp_list, random_state=seed)),
            ]

            for m_name, fn in track_a_specs:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    t0 = time.perf_counter()
                    res = fn()
                    sec = time.perf_counter() - t0
                met = score_fit_on_split(
                    res, tr_mats, te_mats, y_tr, y_te, k=k,
                    true_u_test=u_te, true_v=case.get("true_v"), track="A", rf_trees=rf_trees,
                )
                rows.append({"Dataset": task_name, "Track": "A_flow_v", "Model": m_name,
                             "Seed": seed, "k": k, "fit_seconds": sec, **met})

            # Track B: Ambient flow_simr variants (for k >= 2 tasks)
            if run_track_b and k >= 2:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    t0 = time.perf_counter()
                    res_fs_base = fit_pilot_flow_simr(tr_mats, k, variant="base", iterations=iters, random_state=seed)
                    sec_base = time.perf_counter() - t0

                    t0 = time.perf_counter()
                    res_t1_s = apply_tier1_post_flow_varimax(res_fs_base, tr_mats, k=k, mode="structure")
                    sec_t1s = sec_base + (time.perf_counter() - t0)

                    t0 = time.perf_counter()
                    res_t1_j = apply_tier1_post_flow_varimax(res_fs_base, tr_mats, k=k, mode="jacobian")
                    sec_t1j = sec_base + (time.perf_counter() - t0)

                for m_name, r_obj, s_val in [
                    ("flow_simr_base", res_fs_base, sec_base),
                    ("flow_simr_tier1_struct", res_t1_s, sec_t1s),
                    ("flow_simr_tier1_jac", res_t1_j, sec_t1j),
                ]:
                    met = score_fit_on_split(
                        r_obj, tr_mats, te_mats, y_tr, y_te, k=k,
                        true_u_test=u_te, true_v=case.get("true_v"), track="B", rf_trees=rf_trees,
                    )
                    rows.append({"Dataset": task_name, "Track": "B_flow_simr", "Model": m_name,
                                 "Seed": seed, "k": k, "fit_seconds": s_val, **met})

                for v_tag in ("orthseed", "tier2_jvarimax", "tier2_orthseed", "gpca"):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        t0 = time.perf_counter()
                        r_v = fit_pilot_flow_simr(tr_mats, k, variant=v_tag, iterations=iters, random_state=seed)
                        sec_v = time.perf_counter() - t0
                    met = score_fit_on_split(
                        r_v, tr_mats, te_mats, y_tr, y_te, k=k,
                        true_u_test=u_te, true_v=case.get("true_v"), track="B", rf_trees=rf_trees,
                    )
                    rows.append({"Dataset": task_name, "Track": "B_flow_simr", "Model": f"flow_simr_{v_tag}",
                                 "Seed": seed, "k": k, "fit_seconds": sec_v, **met})

        print(f"[done] {task_name} ({n_seeds} seeds)", flush=True)

    # --- B. Real Neuroimaging: ADNI-MCI Protocol B (k=3, MRI + Cog train -> MRI-only test) ---
    adni_iters = min(90, iters) if iters > 10 else iters
    for s_off in range(n_adni_seeds):
        seed = 42 + s_off
        d_adni = protocol_b(seed=seed, row_center=False)
        x_tr_np = d_adni["x_tr"]
        x_te_np = d_adni["x_te"]
        y_tr_multi = d_adni["y_tr"]
        y_te_multi = d_adni["y_te"]
        pc1_tr = d_adni["pc1_tr"]
        pc1_te = d_adni["pc1_te"]

        if max_subjects > 0:
            n_tr_cap = max(24, int(0.7 * max_subjects))
            n_te_cap = max(12, max_subjects - n_tr_cap)
            x_tr_np = [m[:n_tr_cap, :min(m.shape[1], 32)] for m in x_tr_np]
            x_te_np = [m[:n_te_cap, :min(m.shape[1], 32)] for m in x_te_np]
            y_tr_multi = y_tr_multi[:n_tr_cap]
            y_te_multi = y_te_multi[:n_te_cap]
            pc1_tr = pc1_tr[:n_tr_cap]
            pc1_te = pc1_te[:n_te_cap]

        tr_4views = [torch.tensor(m).float() for m in x_tr_np] + [torch.tensor(y_tr_multi).float()]
        te_4views = [torch.tensor(m).float() for m in x_te_np] + [torch.tensor(y_te_multi).float()]
        sp_adni = [0.5, 0.5, 0.5, None]
        eval_mri = (0, 1, 2)
        k_adni = 3
        lin_iters_adni = min(60, max(2, adni_iters))

        adni_track_a: List[Tuple[str, Callable[[], Dict[str, Any]], str]] = [
            ("simlr", lambda: pysimlr.fit("simlr", tr_4views, k_adni, sparseness=sp_adni, positivity="positive",
                                          topology="graph", path_graph=ADNI_PATH_GRAPH,
                                          iterations=lin_iters_adni, random_state=seed), "positive"),
            ("lend", lambda: pysimlr.fit("lend", tr_4views, k_adni, sparseness=sp_adni, positivity="positive",
                                         topology="graph", path_graph=ADNI_PATH_GRAPH,
                                         iterations=adni_iters, random_state=seed), "positive"),
            ("flow_v_base", lambda: fit_pilot_flow_v(
                tr_4views, k_adni, variant="base", iterations=adni_iters, sparseness=sp_adni,
                positivity="positive", topology="graph", path_graph=ADNI_PATH_GRAPH, random_state=seed
            ), "positive"),
            ("flow_v_pcf_01", lambda: fit_pilot_flow_v(
                tr_4views, k_adni, variant="pcf_01", iterations=adni_iters, sparseness=sp_adni,
                positivity="positive", topology="graph", path_graph=ADNI_PATH_GRAPH, random_state=seed
            ), "positive"),
            ("flow_v_pcf_05", lambda: fit_pilot_flow_v(
                tr_4views, k_adni, variant="pcf_05", iterations=adni_iters, sparseness=sp_adni,
                positivity="positive", topology="graph", path_graph=ADNI_PATH_GRAPH, random_state=seed
            ), "positive"),
            ("flow_v_spline1d", lambda: fit_pilot_flow_v(
                tr_4views, k_adni, variant="spline1d", iterations=adni_iters, sparseness=sp_adni,
                positivity="positive", topology="graph", path_graph=ADNI_PATH_GRAPH, random_state=seed
            ), "positive"),
            ("flow_v_spline_pcf", lambda: fit_pilot_flow_v(
                tr_4views, k_adni, variant="spline_pcf", iterations=adni_iters, sparseness=sp_adni,
                positivity="positive", topology="graph", path_graph=ADNI_PATH_GRAPH, random_state=seed
            ), "positive"),
            ("flow_v_gpca", lambda: fit_pilot_flow_v(
                tr_4views, k_adni, variant="gpca", iterations=adni_iters, sparseness=sp_adni,
                positivity="positive", topology="graph", path_graph=ADNI_PATH_GRAPH, random_state=seed
            ), "positive"),
        ]

        for m_name, fn, pos_mode in adni_track_a:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t0 = time.perf_counter()
                res = fn()
                sec = time.perf_counter() - t0
            met = score_fit_on_split(
                res, tr_4views, te_4views, pc1_tr, pc1_te, k=k_adni,
                eval_view_indices=eval_mri, y_multi_train=y_tr_multi, y_multi_test=y_te_multi,
                positivity=pos_mode, track="A", rf_trees=rf_trees,
            )
            rows.append({"Dataset": "ADNI_MCI_ProtocolB_k3", "Track": "A_flow_v", "Model": m_name,
                         "Seed": seed, "k": k_adni, "fit_seconds": sec, **met})

        # Track B on ADNI-MCI Protocol B (MRI-only deployment via standalone MRI view flow encoders 0, 1, 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t0 = time.perf_counter()
            res_b = fit_pilot_flow_simr(
                tr_4views, k_adni, variant="base", iterations=adni_iters,
                topology="graph", path_graph=ADNI_PATH_GRAPH, eval_view_indices=eval_mri, random_state=seed,
            )
            sec_b = time.perf_counter() - t0

            t0 = time.perf_counter()
            res_t1s = apply_tier1_post_flow_varimax(res_b, tr_4views, k=k_adni, eval_view_indices=eval_mri, mode="structure")
            sec_t1s = sec_b + (time.perf_counter() - t0)

            t0 = time.perf_counter()
            res_t1j = apply_tier1_post_flow_varimax(res_b, tr_4views, k=k_adni, eval_view_indices=eval_mri, mode="jacobian")
            sec_t1j = sec_b + (time.perf_counter() - t0)

        for m_name, r_obj, s_val in [
            ("flow_simr_base", res_b, sec_b),
            ("flow_simr_tier1_struct", res_t1s, sec_t1s),
            ("flow_simr_tier1_jac", res_t1j, sec_t1j),
        ]:
            met = score_fit_on_split(
                r_obj, tr_4views, te_4views, pc1_tr, pc1_te, k=k_adni,
                eval_view_indices=eval_mri, y_multi_train=y_tr_multi, y_multi_test=y_te_multi,
                track="B", rf_trees=rf_trees,
            )
            rows.append({"Dataset": "ADNI_MCI_ProtocolB_k3", "Track": "B_flow_simr", "Model": m_name,
                         "Seed": seed, "k": k_adni, "fit_seconds": s_val, **met})

        for v_tag in ("orthseed", "tier2_jvarimax", "tier2_orthseed", "gpca"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t0 = time.perf_counter()
                r_v = fit_pilot_flow_simr(
                    tr_4views, k_adni, variant=v_tag, iterations=adni_iters,
                    topology="graph", path_graph=ADNI_PATH_GRAPH, eval_view_indices=eval_mri, random_state=seed,
                )
                sec_v = time.perf_counter() - t0
            met = score_fit_on_split(
                r_v, tr_4views, te_4views, pc1_tr, pc1_te, k=k_adni,
                eval_view_indices=eval_mri, y_multi_train=y_tr_multi, y_multi_test=y_te_multi,
                track="B", rf_trees=rf_trees,
            )
            rows.append({"Dataset": "ADNI_MCI_ProtocolB_k3", "Track": "B_flow_simr", "Model": f"flow_simr_{v_tag}",
                         "Seed": seed, "k": k_adni, "fit_seconds": sec_v, **met})

        print(f"[done] ADNI_MCI_ProtocolB_k3 seed={seed}", flush=True)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 7. Paired Statistical Inference, Visualizations, and HTML Report
# ---------------------------------------------------------------------------

def compute_paired_contrasts(df: pd.DataFrame, n_perms: int) -> pd.DataFrame:
    """Compute within-dataset paired contrasts (Delta, exact sign-flip p, Wilcoxon p) against baselines."""
    contrasts = [
        # Track A contrasts vs flow_v_base
        ("A_flow_v", "flow_v_pcf_01", "flow_v_base", "oos_r2"),
        ("A_flow_v", "flow_v_pcf_05", "flow_v_base", "oos_r2"),
        ("A_flow_v", "flow_v_spline1d", "flow_v_base", "oos_r2"),
        ("A_flow_v", "flow_v_spline_pcf", "flow_v_base", "oos_r2"),
        ("A_flow_v", "flow_v_gpca", "flow_v_base", "oos_r2"),
        ("A_flow_v", "flow_v_pcf_01", "flow_v_base", "coord_spearman_rho"),
        ("A_flow_v", "flow_v_pcf_05", "flow_v_base", "coord_spearman_rho"),
        ("A_flow_v", "flow_v_spline1d", "flow_v_base", "coord_spearman_rho"),
        ("A_flow_v", "flow_v_pcf_05", "flow_v_base", "coh_J_flow"),
        ("A_flow_v", "flow_v_spline1d", "flow_v_base", "coh_J_flow"),
        ("A_flow_v", "flow_v_spline1d", "simlr", "oos_r2"),
        # Track B contrasts vs flow_simr_base
        ("B_flow_simr", "flow_simr_tier1_struct", "flow_simr_base", "axis_oos_r2"),
        ("B_flow_simr", "flow_simr_tier1_struct", "flow_simr_base", "best_single_axis_r2"),
        ("B_flow_simr", "flow_simr_tier1_struct", "flow_simr_base", "hoyer_S"),
        ("B_flow_simr", "flow_simr_tier1_jac", "flow_simr_base", "axis_oos_r2"),
        ("B_flow_simr", "flow_simr_orthseed", "flow_simr_base", "oos_r2"),
        ("B_flow_simr", "flow_simr_orthseed", "flow_simr_base", "axis_oos_r2"),
        ("B_flow_simr", "flow_simr_tier2_jvarimax", "flow_simr_base", "oos_r2"),
        ("B_flow_simr", "flow_simr_tier2_orthseed", "flow_simr_base", "oos_r2"),
        ("B_flow_simr", "flow_simr_tier2_orthseed", "flow_simr_base", "axis_oos_r2"),
        ("B_flow_simr", "flow_simr_tier2_orthseed", "flow_simr_orthseed", "oos_r2"),
        ("B_flow_simr", "flow_simr_gpca", "flow_simr_base", "oos_r2"),
    ]
    out_rows = []
    for track, m_a, m_b, metric in contrasts:
        sub = df[df["Track"] == track]
        for ds, grp in sub.groupby("Dataset"):
            piv = grp.pivot_table(index="Seed", columns="Model", values=metric, aggfunc="mean")
            if m_a not in piv.columns or m_b not in piv.columns:
                continue
            pair = piv[[m_a, m_b]].dropna()
            if len(pair) == 0:
                continue
            diffs = (pair[m_a] - pair[m_b]).to_numpy(dtype=float)
            nz = diffs[np.abs(diffs) > 1e-12]
            p_wilc = float(wilcoxon(diffs).pvalue) if len(nz) >= 3 else 1.0
            p_perm = float(exact_sign_flip_permutation(diffs, n_mc=max(2, n_perms)))
            out_rows.append({
                "Dataset": ds,
                "Track": track,
                "Metric": metric,
                "Candidate": m_a,
                "Reference": m_b,
                "N_Seeds": int(len(pair)),
                "Mean_Candidate": float(pair[m_a].mean()),
                "Mean_Reference": float(pair[m_b].mean()),
                "Mean_Diff": float(diffs.mean()),
                "SD_Diff": float(diffs.std(ddof=0)),
                "Wins": int(np.sum(diffs > 1e-9)),
                "Losses": int(np.sum(diffs < -1e-9)),
                "P_SignFlip": p_perm,
                "P_Wilcoxon": p_wilc,
            })

    # Also cross-dataset paired summary (one block per dataset, Demsar 2006)
    for track, m_a, m_b, metric in contrasts:
        sub = df[df["Track"] == track]
        piv = sub.pivot_table(index="Dataset", columns="Model", values=metric, aggfunc="mean")
        if m_a not in piv.columns or m_b not in piv.columns:
            continue
        pair = piv[[m_a, m_b]].dropna()
        if len(pair) < 2:
            continue
        diffs = (pair[m_a] - pair[m_b]).to_numpy(dtype=float)
        nz = diffs[np.abs(diffs) > 1e-12]
        p_wilc = float(wilcoxon(diffs).pvalue) if len(nz) >= 3 else 1.0
        p_perm = float(exact_sign_flip_permutation(diffs, n_mc=max(2, n_perms)))
        out_rows.append({
            "Dataset": "ALL_DATASETS_BLOCK_MEAN",
            "Track": track,
            "Metric": metric,
            "Candidate": m_a,
            "Reference": m_b,
            "N_Seeds": int(len(pair)),
            "Mean_Candidate": float(pair[m_a].mean()),
            "Mean_Reference": float(pair[m_b].mean()),
            "Mean_Diff": float(diffs.mean()),
            "SD_Diff": float(diffs.std(ddof=0)),
            "Wins": int(np.sum(diffs > 1e-9)),
            "Losses": int(np.sum(diffs < -1e-9)),
            "P_SignFlip": p_perm,
            "P_Wilcoxon": p_wilc,
        })
    return pd.DataFrame(out_rows)


def render_qc_figures(df: pd.DataFrame, out_dir: Path, suffix: str) -> List[Path]:
    """Render publication-grade diagnostic panels for Track A, Track B, and k=1 warping."""
    saved: List[Path] = []
    plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False})

    # Figure 1: Track A (flow_v) Out-of-Sample R^2, Coordinate Spearman rho, and Jacobian Shear Coh(J)
    sub_a = df[df["Track"] == "A_flow_v"]
    dsets_a = ["Synth_Poly_k3", "Synth_LocalWarp_k3", "Synth_Skew_k1", "Diabetes_k2", "ADNI_MCI_ProtocolB_k3"]
    dsets_a = [d for d in dsets_a if d in sub_a["Dataset"].unique()]
    models_a = ["simlr", "lend", "flow_v_base", "flow_v_pcf_01", "flow_v_pcf_05",
                "flow_v_spline1d", "flow_v_spline_pcf", "flow_v_gpca"]

    fig1, axes1 = plt.subplots(1, 3, figsize=(15.5, 4.8), dpi=(100 if suffix else 180))
    x_pos = np.arange(len(dsets_a))
    width = 0.10
    palette_a = ["#4c566a", "#81a1c1", "#5e81ac", "#ebcb8b", "#d08770", "#2a9d8f", "#88c0d0", "#bf616a"]

    for m_i, m_name in enumerate(models_a):
        offset = (m_i - (len(models_a) - 1) / 2.0) * width
        for ax_i, (col, title) in enumerate([
            ("oos_r2", "Out-of-Sample Linear Probe R² (↑)"),
            ("coord_spearman_rho", "Coordinate Monotonicity Spearman ρ (↑)"),
            ("coh_J_flow", "Flow Jacobian Off-Diag Coherence Coh(J) (↓)"),
        ]):
            means, sds = [], []
            for ds in dsets_a:
                vals = sub_a[(sub_a["Dataset"] == ds) & (sub_a["Model"] == m_name)][col].dropna()
                means.append(float(vals.mean()) if len(vals) else 0.0)
                sds.append(float(vals.std(ddof=0)) if len(vals) > 1 else 0.0)
            axes1[ax_i].bar(
                x_pos + offset, means, width=width * 0.92, yerr=sds, capsize=1.8,
                label=m_name if ax_i == 0 else None, color=palette_a[m_i], edgecolor="white", linewidth=0.4,
            )
            axes1[ax_i].set_xticks(x_pos)
            axes1[ax_i].set_xticklabels([d.replace("_ProtocolB_k3", "").replace("Synth_", "S:") for d in dsets_a], rotation=20, ha="right")
            axes1[ax_i].set_title(title, fontsize=10.5, fontweight="bold")
            axes1[ax_i].grid(axis="y", alpha=0.25, linestyle="--")
    axes1[0].legend(loc="lower left", fontsize=7.8, ncol=2, framealpha=0.9)
    fig1.suptitle("Track A (flow_v): PCF Regularizer (Idea 1) vs. 1D Monotone RQS Splines (Idea 3) vs. GaussianPCA",
                  fontsize=11.5, fontweight="bold", y=1.02)
    fig1.tight_layout()
    p1 = out_dir / f"fig_track_a_flow_v{suffix}.png"
    fig1.savefig(p1, bbox_inches="tight")
    plt.close(fig1)
    saved.append(p1)

    # Figure 2: Track B (flow_simr) Subspace R^2, Axis-Sensitive RF R^2, Best Single-Axis R^2, & Hoyer(S)
    sub_b = df[df["Track"] == "B_flow_simr"]
    dsets_b = [d for d in ["Synth_Poly_k3", "Synth_LocalWarp_k3", "Diabetes_k2", "ADNI_MCI_ProtocolB_k3"]
               if d in sub_b["Dataset"].unique()]
    models_b = [
        "flow_simr_base", "flow_simr_tier1_struct", "flow_simr_tier1_jac",
        "flow_simr_orthseed", "flow_simr_tier2_jvarimax", "flow_simr_tier2_orthseed", "flow_simr_gpca"
    ]
    fig2, axes2 = plt.subplots(1, 3, figsize=(15.5, 4.8), dpi=(100 if suffix else 180))
    x_pos_b = np.arange(len(dsets_b))
    width_b = 0.115
    palette_b = ["#5e81ac", "#2a9d8f", "#52b788", "#e9c46a", "#f4a261", "#e76f51", "#b56576"]

    for m_i, m_name in enumerate(models_b):
        offset = (m_i - (len(models_b) - 1) / 2.0) * width_b
        for ax_i, (col, title) in enumerate([
            ("oos_r2", "Subspace OOS R² (↑)"),
            ("axis_oos_r2", "Axis-Sensitive RF OOS R² (↑)"),
            ("hoyer_S", "Structure Simple-Structure Hoyer(S) (↑)"),
        ]):
            means, sds = [], []
            for ds in dsets_b:
                vals = sub_b[(sub_b["Dataset"] == ds) & (sub_b["Model"] == m_name)][col].dropna()
                means.append(float(vals.mean()) if len(vals) else 0.0)
                sds.append(float(vals.std(ddof=0)) if len(vals) > 1 else 0.0)
            axes2[ax_i].bar(
                x_pos_b + offset, means, width=width_b * 0.92, yerr=sds, capsize=1.8,
                label=m_name.replace("flow_simr_", "") if ax_i == 0 else None,
                color=palette_b[m_i], edgecolor="white", linewidth=0.4,
            )
            axes2[ax_i].set_xticks(x_pos_b)
            axes2[ax_i].set_xticklabels([d.replace("_ProtocolB_k3", "").replace("Synth_", "S:") for d in dsets_b], rotation=20, ha="right")
            axes2[ax_i].set_title(title, fontsize=10.5, fontweight="bold")
            axes2[ax_i].grid(axis="y", alpha=0.25, linestyle="--")
    axes2[0].legend(loc="best", fontsize=8.0, framealpha=0.9)
    fig2.suptitle("Track B (flow_simr): Tier 1 Post-Flow Varimax vs. Orthogonal Varimax-PCA Seed vs. Tier 2 Jacobian-Varimax",
                  fontsize=11.5, fontweight="bold", y=1.02)
    fig2.tight_layout()
    p2 = out_dir / f"fig_track_b_flow_simr{suffix}.png"
    fig2.savefig(p2, bbox_inches="tight")
    plt.close(fig2)
    saved.append(p2)

    # Figure 3: Visual Warping Curve Verification at k=1 (RealNVP linear trap vs 1D Monotone RQS Spline)
    case_k1 = build_skew_k1_case(n_samples=160, seed=42)
    tr_m, te_m, _, _, u_te = _prepare_standard_split(case_k1, seed=42, max_subjects=0)
    w_iters = 2 if suffix else 60
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_k1_base = fit_pilot_flow_v(tr_m, 1, variant="base", iterations=w_iters, sparseness=[0.5, 0.5], random_state=42)
        res_k1_spl = fit_pilot_flow_v(tr_m, 1, variant="spline1d", iterations=w_iters, sparseness=[0.5, 0.5], random_state=42)

    fig3, axes3 = plt.subplots(1, 2, figsize=(10.2, 4.1), dpi=(100 if suffix else 180))
    from pysimlr.api import _preprocess_views
    xs_te_b, _ = _preprocess_views(te_m, res_k1_base["scale_list"], res_k1_base["provenance_list"])
    xs_te_s, _ = _preprocess_views(te_m, res_k1_spl["scale_list"], res_k1_spl["provenance_list"])
    with torch.no_grad():
        lin_b = res_k1_base["model"].linear_encoders[0](xs_te_b[0]).squeeze().numpy()
        lat_b = res_k1_base["model"].encoders[0](torch.tensor(lin_b).unsqueeze(1)).squeeze().numpy()
        lin_s = res_k1_spl["model"].linear_encoders[0](xs_te_s[0]).squeeze().numpy()
        lat_s = res_k1_spl["model"].encoders[0](torch.tensor(lin_s).unsqueeze(1)).squeeze().numpy()

    for ax, lx, lz, title_str, col in [
        (axes3[0], lin_b, lat_b, "flow_v_base (RealNVP at k=1: strictly affine)", "#5e81ac"),
        (axes3[1], lin_s, lat_s, "flow_v_spline1d (1D Monotone RQS at k=1: log-like warp)", "#2a9d8f"),
    ]:
        order = np.argsort(lx)
        ax.scatter(lx, lz, s=22, alpha=0.65, color=col, edgecolor="none")
        ax.plot(lx[order], lz[order], color="#2e3440", lw=1.8)
        rho_v = spearmanr(lx, lz).correlation
        ax.set_title(f"{title_str}\nSpearman ρ = {rho_v:.4f}", fontsize=9.8, fontweight="bold")
        ax.set_xlabel("First-layer projection (X₀ V₀)")
        ax.set_ylabel("Post-flow latent z₀")
        ax.grid(alpha=0.25, linestyle="--")
    fig3.suptitle("Direct Visual Check of k=1 Flow Warping on Skewed Log-Normal Synthetic View 0",
                  fontsize=11.0, fontweight="bold", y=1.03)
    fig3.tight_layout()
    p3 = out_dir / f"fig_k1_warping{suffix}.png"
    fig3.savefig(p3, bbox_inches="tight")
    plt.close(fig3)
    saved.append(p3)

    return saved


def write_html_report(
    summary_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    provenance: Dict[str, Any],
    fig_paths: List[Path],
    out_html: Path,
) -> None:
    """Write standalone engineering HTML report embedding tables, provenance, and figures."""
    import base64
    imgs_html = []
    for fp in fig_paths:
        b64 = base64.b64encode(fp.read_bytes()).decode("ascii")
        imgs_html.append(
            f'<div style="margin:18px 0;"><img src="data:image/png;base64,{b64}" '
            f'style="max-width:100%;border:1px solid #ddd;border-radius:4px;"/></div>'
        )
    sum_html = summary_df.to_html(index=False, float_format=lambda v: f"{v:.4f}")
    st_html = stats_df.to_html(index=False, float_format=lambda v: f"{v:.4f}")
    prov_pretty = html.escape(json.dumps(provenance, indent=2, default=str))
    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>pysimlr Pilot: PCA, Monotone Spline &amp; Varimax Normalizing Flows</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 28px auto; max-width: 1280px; color: #1f242b; line-height: 1.45; padding: 0 18px; }}
h1, h2, h3 {{ color: #1a2a3a; }}
table {{ border-collapse: collapse; width: 100%; font-size: 12.5px; margin: 14px 0; }}
th, td {{ border: 1px solid #d8dee9; padding: 6px 8px; text-align: right; }}
th {{ background: #eceff4; text-align: center; }}
td:nth-child(1), td:nth-child(2), td:nth-child(3) {{ text-align: left; }}
pre {{ background: #f6f8fa; padding: 12px; border-radius: 4px; overflow-x: auto; font-size: 12px; }}
</style></head><body>
<h1>Pilot Study: PCA-Based, Monotone-Spline &amp; Varimax Normalizing Flows in <code>pysimlr</code></h1>
<p><strong>Smoke mode:</strong> <code>{provenance['smoke']}</code> &nbsp;|&nbsp;
<strong>UTC:</strong> <code>{provenance['utc_start']}</code> &rarr; <code>{provenance['utc_end']}</code> &nbsp;|&nbsp;
<strong>Wall time:</strong> <code>{provenance['wall_seconds']:.2f} s</code></p>
<h2>1. Diagnostic Figures</h2>
{''.join(imgs_html)}
<h2>2. Dataset &times; Model Mean Metrics</h2>
{sum_html}
<h2>3. Paired Statistical Contrasts (Sign-Flip Permutation &amp; Wilcoxon)</h2>
{st_html}
<h2>4. Provenance &amp; Stage 0/1 Contract Verification</h2>
<pre>{prov_pretty}</pre>
</body></html>"""
    out_html.write_text(doc, encoding="utf-8")


def _git_sha(repo_path: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_path).decode().strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=100, help="Training iterations budget (2 under --smoke)")
    ap.add_argument("--n-seeds", type=int, default=5, help="Replicate seeds for synthetic & Diabetes (2 under --smoke)")
    ap.add_argument("--n-adni-seeds", type=int, default=3, help="Replicate seeds for ADNI-MCI Protocol B (2 under --smoke)")
    ap.add_argument("--max-subjects", type=int, default=0, help="Subject cap (0 = full data; 48 under --smoke)")
    ap.add_argument("--n-perms", type=int, default=10000, help="Sign-flip permutation draws (2 under --smoke)")
    ap.add_argument("--n-jobs", type=int, default=1, help="Worker count parameter (2 under --smoke)")
    ap.add_argument("--out-dir", type=str, default=str(REPO / "reports" / "pilot_flow_pca_varimax"))
    ap.add_argument("--smoke", action="store_true", help="Fast smoke gate: counts=2, 100%% of stages in <10s")
    args = ap.parse_args(argv)

    utc_start = datetime.datetime.now(datetime.timezone.utc).isoformat()
    t_wall0 = time.perf_counter()

    iters = 2 if args.smoke else int(args.iters)
    n_seeds = 2 if args.smoke else int(args.n_seeds)
    n_adni_seeds = 2 if args.smoke else int(args.n_adni_seeds)
    max_subjects = 36 if args.smoke else int(args.max_subjects)
    n_perms = 2 if args.smoke else int(args.n_perms)
    n_jobs = 2 if args.smoke else int(args.n_jobs)
    n_repeats = 2 if args.smoke else 3
    rf_trees = 5 if args.smoke else 60

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if args.smoke else ""

    print(f"[Stage 0] Running mathematical bijectivity, log-det, PCF, k=1, and Tier-1 gauge checks...", flush=True)
    stage0 = run_stage0_contract_checks()

    print(f"[Stage 1] Measuring same-engine noise floor ({n_repeats} repeats)...", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stage1 = run_stage1_noise_floor(iters=iters, n_repeats=n_repeats)

    print(f"[Stage 2-3] Running Track A (flow_v) & Track B (flow_simr) pilot suite "
          f"(smoke={bool(args.smoke)}, iters={iters}, n_seeds={n_seeds}, n_adni_seeds={n_adni_seeds})...", flush=True)
    raw_df = run_benchmark_suite(
        iters=iters, n_seeds=n_seeds, n_adni_seeds=n_adni_seeds,
        max_subjects=max_subjects, rf_trees=rf_trees,
    )

    metric_cols = [
        "oos_r2", "xv_oos_r2", "axis_oos_r2", "best_single_axis_r2", "mean11_r2",
        "coh_L", "coh_S_unrot", "coh_S_rot", "coh_J_flow",
        "coord_spearman_rho", "min_coord_spearman_rho",
        "hoyer_S", "frame_defect_v", "support_recovery_v", "support_recovery_S",
        "cmc_latent_u", "cond_impute_r2", "fit_seconds",
    ]
    summary_df = (
        raw_df.groupby(["Dataset", "Track", "Model"], sort=False)[metric_cols]
        .mean()
        .reset_index()
    )
    stats_df = compute_paired_contrasts(raw_df, n_perms=n_perms)

    raw_csv = out_dir / f"pilot_flow_raw{suffix}.csv"
    sum_csv = out_dir / f"pilot_flow_summary{suffix}.csv"
    stats_csv = out_dir / f"pilot_flow_contrasts{suffix}.csv"
    raw_df.to_csv(raw_csv, index=False)
    summary_df.to_csv(sum_csv, index=False)
    stats_df.to_csv(stats_csv, index=False)

    fig_paths = render_qc_figures(raw_df, out_dir=out_dir, suffix=suffix)

    wall_seconds = time.perf_counter() - t_wall0
    utc_end = datetime.datetime.now(datetime.timezone.utc).isoformat()

    import antstorch
    antstorch_repo = Path(antstorch.__file__).resolve().parents[1]
    provenance = {
        "smoke": bool(args.smoke),
        "cli": [sys.executable, *sys.argv],
        "utc_start": utc_start,
        "utc_end": utc_end,
        "wall_seconds": round(wall_seconds, 3),
        "host": platform.node(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "device": "cpu",
        "versions": {
            "pysimlr": getattr(pysimlr, "__version__", "unknown"),
            "pysimlr_git_sha": _git_sha(REPO),
            "antstorch_git_sha": _git_sha(antstorch_repo),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "antsnormflows": getattr(nf, "__version__", "installed"),
        },
        "parameters": {
            "iters": iters,
            "n_seeds": n_seeds,
            "n_adni_seeds": n_adni_seeds,
            "max_subjects": max_subjects,
            "n_perms": n_perms,
            "n_jobs": n_jobs,
        },
        "contract_checks": stage0,
        "noise_floor": stage1,
        "outputs": {
            "raw_csv": str(raw_csv),
            "summary_csv": str(sum_csv),
            "contrasts_csv": str(stats_csv),
            "figures": [str(p) for p in fig_paths],
        },
    }
    prov_path = out_dir / f"provenance_pilot_flow{suffix}.json"
    html_path = out_dir / f"pilot_flow_report{suffix}.html"
    provenance["outputs"]["html_report"] = str(html_path)
    prov_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    write_html_report(summary_df, stats_df, provenance, fig_paths, html_path)

    print(f"[Done] Wrote {raw_csv.name}, {sum_csv.name}, {stats_csv.name}, {prov_path.name}, "
          f"and {html_path.name} in {wall_seconds:.2f}s.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
