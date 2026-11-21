#!/usr/bin/env python3
"""Targeted follow-up experiments resolving the two root causes identified in pilot_flow_pca_varimax.py:

1. Track A (``flow_v``): ``flow_v_spline1d`` lacked per-coordinate ``ActNorm1d`` before the
   fixed ``[-3.5, +3.5]`` RQS spline box. On ``ADNI-MCI`` (``p_m = 160``, ``positivity="positive"``),
   ``SD(X_m V_m) ~ 6-10`` placed most samples on the linear identity tails. Here we test
   ``flow_v_actnorm_spline1d`` (data-initialized positive-scale ``ActNorm1d`` + ``ElementwiseRQSplineFlow``),
   which preserves exact ``Coh(J) == 0`` and ``Spearman rho == +1.0`` while keeping inputs inside
   the active spline knots.

2. Track B (``flow_simr``):
   (a) ``flow_simr_tier2_orthseed`` achieved ``100%`` support recovery and ``0.879`` RF R^2 on
       ``Synth_LocalWarp_k3``, but ``scale_bound=2.0`` across 4 coupling layers (``exp(8)=2981``)
       produced single-sample test outliers (``|z_te| = 76.4``) that broke linear OLS. We test
       ``scale_bound=0.35`` (``flow_simr_tier2_orthseed_sb035`` and ``flow_simr_base_sb035``).
   (b) ``flow_simr_gpca`` led Track B subspace OOS R^2 (``0.942`` Poly, ``0.869`` LocalWarp)
       without Varimax axis alignment. We test ``flow_simr_gpca_varimax`` (exact O(k) Kaiser
       Varimax rotation applied to the GaussianPCA whitened space ``W <- R^T W``).

Usage:
    python scripts/pilot_flow_targeted_followup.py --smoke
    python scripts/pilot_flow_targeted_followup.py --iters 100 --n-seeds 5 --n-adni-seeds 3
"""
from __future__ import annotations

import argparse
import copy
import datetime
import json
import math
import platform
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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
from scipy.stats import wilcoxon

import pysimlr
from pysimlr.benchmarks.deep_ranking import exact_sign_flip_permutation
from pysimlr.defaults import (
    DEFAULT_ENERGY_TYPE,
    DEFAULT_MIXING_ALGORITHM,
    DEFAULT_NSA_W,
    DEFAULT_SIGN_ALIGNMENT,
    DEFAULT_TOPOLOGY,
)
from pysimlr.deep import (
    _MultiViewModelMixin,
    _aggregate_shared_consensus,
    _finalize_bases,
    _freeze_consensus,
    _get_optimizer,
    _init_generator,
    _linear_encoders,
    _plateau_scheduler,
    _prepare_deep_inputs,
    _recon_errors,
    _resolve_device,
    _resolve_seed,
    _seeded_rng,
    _shuffled_loader,
    resolve_warmup_epochs,
)
from pysimlr.flows import FlowConditionalInference, FlowDecoderWrapper, FlowEncoderWrapper
from pysimlr.loadings import olee_loadings, structure_coefficients, varimax_rotation
from pysimlr.positivity import resolve_deep_positivity
from pysimlr.utils import basis_rank_report

from paper_adni_ablations import PATH_GRAPH as ADNI_PATH_GRAPH
from paper_adni_ablations import protocol_b
from pilot_flow_pca_varimax import (
    ElementwiseRQSplineLayer,
    PilotFlowSiMRModel,
    _git_sha,
    _prepare_standard_split,
    _train_pilot_flow_loop,
    build_localized_warp_case,
    build_nonlinear_shared_case,
    build_skew_k1_case,
    build_varimax_pca_orthogonal_frames,
    exact_pcf_penalty_bpd,
    fit_pilot_flow_simr,
    fit_pilot_flow_v,
    load_diabetes_case,
    score_fit_on_split,
)


class PositiveActNorm1d(nn.Module):
    """Data-dependent per-coordinate standardization with strictly positive scale.

    On the first forward pass with N >= 2 rows, initializes ``loc = -mean(x)`` and
    ``raw_log_scale`` so ``(x + loc) * exp(log_scale)`` has zero mean and unit SD per
    coordinate, bringing arbitrary-variance ``X_m V_m`` projections cleanly inside the
    ``[-tail_bound, +tail_bound]`` spline knot domain while preserving:
      * strictly positive derivative (> 0) -> Spearman rho == +1.0 guaranteed;
      * strictly diagonal Jacobian -> Coh(J) == 0.0 and PCF == 0.0 identically;
      * exact analytical inverse and log-determinant.
    """

    def __init__(self, dim: int, log_scale_cap: float = 4.0):
        super().__init__()
        self.dim = int(dim)
        self.log_scale_cap = float(log_scale_cap)
        self.loc = nn.Parameter(torch.zeros(1, self.dim))
        self.raw_log_scale = nn.Parameter(torch.zeros(1, self.dim))
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))

    @torch.no_grad()
    def _maybe_init(self, x: torch.Tensor) -> None:
        if bool(self.initialized.item()) or not self.training or x.shape[0] < 2:
            return
        mu = x.detach().mean(dim=0, keepdim=True)
        sd = x.detach().std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-3)
        target_ls = (-torch.log(sd)).clamp(-self.log_scale_cap * 0.95, self.log_scale_cap * 0.95)
        # Invert log_scale_cap * tanh(raw) = target_ls
        ratio = (target_ls / self.log_scale_cap).clamp(-0.999, 0.999)
        self.loc.data.copy_(-mu)
        self.raw_log_scale.data.copy_(torch.atanh(ratio))
        self.initialized.fill_(True)

    def _log_scale(self) -> torch.Tensor:
        return self.log_scale_cap * torch.tanh(self.raw_log_scale)

    def forward_and_log_det(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self._maybe_init(x)
        ls = self._log_scale()
        z = (x + self.loc) * torch.exp(ls)
        ld = ls.sum(dim=1).expand(x.shape[0])
        return z, ld

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_and_log_det(x)

    def inverse_and_log_det(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ls = self._log_scale()
        x = z * torch.exp(-ls) - self.loc
        ld = -ls.sum(dim=1).expand(z.shape[0])
        return x, ld

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        x, _ = self.inverse_and_log_det(z)
        return x


class ActNormElementwiseRQSplineFlow(nn.Module):
    """PositiveActNorm1d followed by monotone ElementwiseRQSplineLayers."""

    def __init__(self, dim: int, num_layers: int = 2, num_bins: int = 8,
                 tail_bound: float = 3.5, scale_bound: float = 1.5):
        super().__init__()
        self.dim = int(dim)
        self.actnorm = PositiveActNorm1d(dim)
        self.layers = nn.ModuleList([
            ElementwiseRQSplineLayer(dim, num_bins=num_bins, tail_bound=tail_bound, scale_bound=scale_bound)
            for _ in range(num_layers)
        ])

    def forward_and_log_det(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z, total_ld = self.actnorm.forward_and_log_det(x)
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
        x, ld0 = self.actnorm.inverse_and_log_det(x)
        return x, total_ld + ld0

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        x, _ = self.inverse_and_log_det(z)
        return x


class ActNormSplineFlowVModel(_MultiViewModelMixin, nn.Module):
    """Flow-SiMR-V with ActNormElementwiseRQSplineFlow per view."""

    def __init__(
        self,
        input_dims: List[int],
        latent_dim: int,
        *,
        positivity: str = "either",
        sparseness: Any = 0.5,
        nsa_w: float = DEFAULT_NSA_W,
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
        positivity = resolve_deep_positivity(positivity)
        self.input_dims = input_dims
        self.latent_dim = latent_dim
        self.is_flow = True
        self.first_layer_mode = "scheduled"
        self.nsa_iterations = 1

        self.linear_encoders = _linear_encoders(
            input_dims, latent_dim, nsa_w, positivity, sparseness, True, "scheduled", 1
        )
        self.flows = nn.ModuleList([
            ActNormElementwiseRQSplineFlow(latent_dim, num_layers=2, num_bins=8)
            for _ in input_dims
        ])
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

    def forward(self, x_list: List[torch.Tensor], obs_masks=None):
        projected_zs = [enc(x) for enc, x in zip(self.linear_encoders, x_list)]
        if self.training:
            # Initialize ActNorm on detached first-layer projections first so grad graph stays clean
            for enc, z in zip(self.encoders, projected_zs):
                z_det, ld_det = enc.flow(z.detach())
                latents_noop = None
        latents = [enc(z) for enc, z in zip(self.encoders, projected_zs)]
        if self.training:
            for enc, z in zip(self.encoders, projected_zs):
                z_det, ld_det = enc.flow(z.detach())
                enc.last_z = z_det
                enc.last_log_det = ld_det
        u_shared = self._training_consensus(latents, self.latent_dim)
        recons = []
        for dec, enc_lin, idx in zip(self.decoders, self.linear_encoders, range(len(x_list))):
            ui = u_shared[idx] if isinstance(u_shared, list) else u_shared
            z_rec = dec(ui)
            v = enc_lin.active_training_basis() if self.training else enc_lin.v
            recons.append(z_rec @ v.t())
        return latents, recons, u_shared

    def initialize_weights(self, data_matrices: List[torch.Tensor], **init_kwargs):
        self.initialize_v(data_matrices, self.latent_dim, **init_kwargs)


def fit_actnorm_spline_flow_v(
    data_matrices: Sequence[torch.Tensor],
    k: int,
    *,
    iterations: int = 100,
    batch_size: int = 64,
    learning_rate: float = 5e-4,
    positivity: str = "either",
    sparseness: Any = 0.5,
    topology: str = DEFAULT_TOPOLOGY,
    path_graph: Optional[Dict[int, List[int]]] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    device = _resolve_device("cpu")
    seed = _resolve_seed(random_state)
    with _seeded_rng(seed, device):
        torch_mats, provenance_list, init_mats, _ = _prepare_deep_inputs(data_matrices, k)
        input_dims = [m.shape[1] for m in torch_mats]
        model = ActNormSplineFlowVModel(
            input_dims, k, positivity=positivity, sparseness=sparseness,
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
            warmup_epochs=warmup, pcf_alpha=0.0, device=device,
        )
        model.eval()
        with torch.no_grad():
            v_mats = [enc.v.detach().cpu() for enc in model.linear_encoders]
            v_mats, retr = _finalize_bases(
                v_mats, positivity, DEFAULT_NSA_W, DEFAULT_ENERGY_TYPE, encoders=model.linear_encoders
            )
            final_latents, recons, u_final = model(torch_mats)
            u_agg = _aggregate_shared_consensus(model, final_latents) if isinstance(u_final, list) else u_final
            cons_res = _freeze_consensus(model, final_latents, u_final, u_agg, obs_masks=None)
        cond_inf = FlowConditionalInference(len(input_dims), k)
        cond_inf.fit(final_latents)

    res = {
        "model": model.cpu(),
        "model_type": "flow_v_actnorm_spline1d",
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
    s_tr = structure_coefficients(torch_mats, res["u"])
    r_load = varimax_rotation(s_tr)
    s_rot = [(s.double() @ r_load).float() for s in s_tr]
    loads, _ = olee_loadings(s_rot, positivity=positivity)
    res["structure"] = s_rot
    res["loadings"] = loads
    res["loading_rotation"] = r_load.float()
    return res


@torch.no_grad()
def apply_gpca_tier1_varimax(
    gpca_res: Dict[str, Any],
    train_views_raw: Sequence[torch.Tensor],
    k: int,
    eval_view_indices: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Exact O(k) Kaiser Varimax gauge rotation of GaussianPCA base factor matrix W <- R^T W.

    Because Sigma = W^T W + sigma^2 I == (R^T W)^T (R^T W) + sigma^2 I for any R in O(k),
    replacing W <- R^T W leaves GaussianPCA.log_prob(z) BIT-FOR-BIT IDENTICAL while rotating
    the extracted whitened latent coordinates eps <- eps @ R into simple structure!
    """
    res = copy.deepcopy(gpca_res)
    model = res["model"]
    model.eval()
    from pysimlr.api import _preprocess_views
    xs_tr, _ = _preprocess_views(train_views_raw, res["scale_list"], res["provenance_list"])
    use_idx = list(range(len(xs_tr))) if eval_view_indices is None else list(eval_view_indices)

    u_tr = res["u"].detach().cpu()
    s_list = structure_coefficients([xs_tr[i] for i in use_idx], u_tr)
    r_orth64 = varimax_rotation(s_list).double()  # (k, k) in float64 before cast
    r_orth = r_orth64.float()

    for gp in model.gpca_bases:
        if gp.W.shape[0] == k:
            gp.W.data.copy_((r_orth64.t() @ gp.W.data.double()).float())

    eval_mats = [m.float() for m in xs_tr]
    rot_latents = [enc(xm) for enc, xm in zip(model.encoders, eval_mats)]
    u_agg = gpca_res["u"].detach().cpu().float() @ r_orth
    u_final = (
        [ui.detach().cpu().float() @ r_orth for ui in gpca_res["u_per_modality"]]
        if gpca_res.get("u_per_modality") is not None else u_agg
    )
    cons_res = _freeze_consensus(model, rot_latents, u_final, u_agg, obs_masks=None)
    _, recons, _ = model(eval_mats)
    cond_inf = FlowConditionalInference(len(xs_tr), k)
    cond_inf.fit(rot_latents)

    res["model"] = model
    res["model_type"] = "flow_simr_gpca_varimax"
    res["u"] = u_agg.cpu()
    res["u_per_modality"] = [ui.cpu() for ui in u_final] if isinstance(u_final, list) else None
    res["latents"] = [l.cpu() for l in rot_latents]
    res["reconstructions"] = [r.cpu() for r in recons]
    res["consensus_map_residual"] = cons_res
    res["cond_inference"] = cond_inf
    return res


def fit_bounded_flow_simr(
    data_matrices: Sequence[torch.Tensor],
    k: int,
    *,
    use_tier2_orthseed: bool,
    scale_bound: float = 0.35,
    iterations: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    topology: str = DEFAULT_TOPOLOGY,
    path_graph: Optional[Dict[int, List[int]]] = None,
    eval_view_indices: Optional[Sequence[int]] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Fit ambient flow_simr with tight scale_bound=0.35 to test whether exponential scale capping eliminates OOS blow-up."""
    device = _resolve_device("cpu")
    seed = _resolve_seed(random_state)
    with _seeded_rng(seed, device):
        torch_mats, provenance_list, _, _ = _prepare_deep_inputs(data_matrices, k)
        input_dims = [m.shape[1] for m in torch_mats]
        q_frames = build_varimax_pca_orthogonal_frames(torch_mats, k) if use_tier2_orthseed else None
        model = PilotFlowSiMRModel(
            input_dims, k, flow_kind="smooth" if use_tier2_orthseed else "realnvp",
            q_frames=q_frames, scale_bound=scale_bound, topology=topology, path_graph=path_graph,
        ).to(device)
        optimizer = _get_optimizer(model, "adam", learning_rate, 5e-4)
        scheduler = _plateau_scheduler(optimizer)
        dataloader = _shuffled_loader(torch_mats, batch_size, seed)
        warmup = resolve_warmup_epochs(None, iterations)
        loss_h, conv = _train_pilot_flow_loop(
            model, dataloader, optimizer, scheduler, iterations=iterations, warmup_epochs=warmup,
            jac_varimax_weight=1.5 if use_tier2_orthseed else 0.0,
            jac_ima_weight=0.25 if use_tier2_orthseed else 0.0,
            device=device,
        )
        model.eval()
        with torch.no_grad():
            final_latents, recons, u_final = model(torch_mats)
            u_agg = _aggregate_shared_consensus(model, final_latents) if isinstance(u_final, list) else u_final
            cons_res = _freeze_consensus(model, final_latents, u_final, u_agg, obs_masks=None)
        cond_inf = FlowConditionalInference(len(input_dims), k)
        cond_inf.fit(final_latents)

    tag = "flow_simr_tier2_orthseed_sb035" if use_tier2_orthseed else "flow_simr_base_sb035"
    res = {
        "model": model.cpu(),
        "model_type": tag,
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
    if use_tier2_orthseed and k > 1:
        from pilot_flow_pca_varimax import apply_tier1_post_flow_varimax
        res = apply_tier1_post_flow_varimax(res, data_matrices, k=k, eval_view_indices=eval_view_indices, mode="structure")
        res["model_type"] = tag
    return res


def run_stage0_followup_checks() -> Dict[str, Any]:
    torch.manual_seed(13)
    # 1. ActNormElementwiseRQSplineFlow exact bijectivity, autograd log-det, PCF==0, and variance standardization on SD=9 inputs
    fl = ActNormElementwiseRQSplineFlow(dim=3, num_layers=2, num_bins=8)
    fl.train()
    x_big = torch.randn(24, 3) * torch.tensor([[9.0, 6.5, 4.0]]) + torch.tensor([[12.0, -5.0, 3.0]])
    z0, ld0 = fl(x_big)
    z0_sd = float(z0.std(dim=0, unbiased=False).mean().item())
    assert 0.90 < z0_sd < 1.10, f"Expected unit-SD ActNorm output on first batch, got {z0_sd}"

    fl.eval()
    for p in fl.layers.parameters():
        p.data.add_(0.35 * torch.randn_like(p))
    z1, ld1 = fl(x_big)
    x_rec, ld_inv = fl.inverse_and_log_det(z1)
    inv_err = float((x_rec - x_big).abs().max().item())
    jac0 = torch.autograd.functional.jacobian(lambda inp: fl(inp.unsqueeze(0))[0].squeeze(0), x_big[0])
    ld_exact0 = float(torch.linalg.slogdet(jac0)[1].item())
    ld_diff0 = abs(float(ld1[0].item()) - ld_exact0)
    pcf_val = float(exact_pcf_penalty_bpd(fl, x_big).item())
    assert inv_err < 1e-4 and ld_diff0 < 1e-4 and abs(pcf_val) < 1e-5

    # 2. GaussianPCA + Tier 1 Varimax exact log_prob & reconstruction invariance
    toy = [torch.randn(32, 8), torch.randn(32, 6)]
    r_gp = fit_pilot_flow_simr(toy, 2, variant="gpca", iterations=3, batch_size=16, random_state=5)
    r_gp_vm = apply_gpca_tier1_varimax(r_gp, toy, k=2)
    z_probe = torch.randn(10, 8)
    lp_before = r_gp["model"].gpca_bases[0].log_prob(z_probe)
    lp_after = r_gp_vm["model"].gpca_bases[0].log_prob(z_probe)
    lp_diff = float((lp_before - lp_after).abs().max().item())
    rec_diff = float(max((a - b).abs().max().item() for a, b in zip(r_gp["reconstructions"], r_gp_vm["reconstructions"])))
    assert lp_diff < 1e-4 and rec_diff < 1e-4, f"lp_diff={lp_diff}, rec_diff={rec_diff}"

    return {
        "actnorm_spline_init_sd": z0_sd,
        "actnorm_spline_inv_err": inv_err,
        "actnorm_spline_logdet_err": ld_diff0,
        "actnorm_spline_pcf_bpd": pcf_val,
        "gpca_varimax_logprob_diff": lp_diff,
        "gpca_varimax_recon_diff": rec_diff,
        "all_passed": True,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--n-adni-seeds", type=int, default=3)
    ap.add_argument("--max-subjects", type=int, default=0)
    ap.add_argument("--n-perms", type=int, default=10000)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--out-dir", type=str, default=str(REPO / "reports" / "pilot_flow_pca_varimax"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)

    utc_start = datetime.datetime.now(datetime.timezone.utc).isoformat()
    t0_wall = time.perf_counter()

    iters = 2 if args.smoke else int(args.iters)
    n_seeds = 2 if args.smoke else int(args.n_seeds)
    n_adni_seeds = 2 if args.smoke else int(args.n_adni_seeds)
    max_subjects = 32 if args.smoke else int(args.max_subjects)
    n_perms = 2 if args.smoke else int(args.n_perms)
    rf_trees = 5 if args.smoke else 60
    suffix = "_smoke" if args.smoke else ""

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stage0 = run_stage0_followup_checks()
    rows: List[Dict[str, Any]] = []

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
    ]

    for task_name, builder, run_b in tasks:
        for s_off in range(n_seeds):
            seed = 42 + s_off
            case = builder(seed)
            k = int(case["shared_k"])
            tr_mats, te_mats, y_tr, y_te, u_te = _prepare_standard_split(case, seed, max_subjects=max_subjects)
            sp_list = [0.5 if m.shape[1] > 15 else None for m in tr_mats]

            for m_name, fn in [
                ("flow_v_base", lambda: fit_pilot_flow_v(tr_mats, k, variant="base", iterations=iters, sparseness=sp_list, random_state=seed)),
                ("flow_v_pcf_05", lambda: fit_pilot_flow_v(tr_mats, k, variant="pcf_05", iterations=iters, sparseness=sp_list, random_state=seed)),
                ("flow_v_spline1d", lambda: fit_pilot_flow_v(tr_mats, k, variant="spline1d", iterations=iters, sparseness=sp_list, random_state=seed)),
                ("flow_v_actnorm_spline1d", lambda: fit_actnorm_spline_flow_v(tr_mats, k, iterations=iters, sparseness=sp_list, random_state=seed)),
            ]:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ts = time.perf_counter()
                    res = fn()
                    sec = time.perf_counter() - ts
                met = score_fit_on_split(
                    res, tr_mats, te_mats, y_tr, y_te, k=k,
                    true_u_test=u_te, true_v=case.get("true_v"), track="A", rf_trees=rf_trees,
                )
                rows.append({"Dataset": task_name, "Track": "A_flow_v", "Model": m_name, "Seed": seed, "k": k, "fit_seconds": sec, **met})

            if run_b and k >= 2:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ts = time.perf_counter()
                    r_sb = fit_bounded_flow_simr(tr_mats, k, use_tier2_orthseed=False, scale_bound=0.35, iterations=iters, random_state=seed)
                    sec_sb = time.perf_counter() - ts

                    ts = time.perf_counter()
                    r_t2_sb = fit_bounded_flow_simr(tr_mats, k, use_tier2_orthseed=True, scale_bound=0.35, iterations=iters, random_state=seed)
                    sec_t2 = time.perf_counter() - ts

                    ts = time.perf_counter()
                    r_gp = fit_pilot_flow_simr(tr_mats, k, variant="gpca", iterations=iters, random_state=seed)
                    sec_gp = time.perf_counter() - ts
                    r_gp_vm = apply_gpca_tier1_varimax(r_gp, tr_mats, k=k)

                for m_name, r_obj, s_val in [
                    ("flow_simr_base_sb035", r_sb, sec_sb),
                    ("flow_simr_tier2_orthseed_sb035", r_t2_sb, sec_t2),
                    ("flow_simr_gpca", r_gp, sec_gp),
                    ("flow_simr_gpca_varimax", r_gp_vm, sec_gp),
                ]:
                    met = score_fit_on_split(
                        r_obj, tr_mats, te_mats, y_tr, y_te, k=k,
                        true_u_test=u_te, true_v=case.get("true_v"), track="B", rf_trees=rf_trees,
                    )
                    rows.append({"Dataset": task_name, "Track": "B_flow_simr", "Model": m_name, "Seed": seed, "k": k, "fit_seconds": s_val, **met})
        print(f"[done] {task_name}", flush=True)

    # ADNI-MCI Protocol B
    adni_iters = min(90, iters) if iters > 10 else iters
    for s_off in range(n_adni_seeds):
        seed = 42 + s_off
        d_adni = protocol_b(seed=seed, row_center=False)
        x_tr_np, x_te_np = d_adni["x_tr"], d_adni["x_te"]
        y_tr_multi, y_te_multi = d_adni["y_tr"], d_adni["y_te"]
        pc1_tr, pc1_te = d_adni["pc1_tr"], d_adni["pc1_te"]
        if max_subjects > 0:
            ntr = max(20, int(0.7 * max_subjects))
            nte = max(10, max_subjects - ntr)
            x_tr_np = [m[:ntr, :min(m.shape[1], 24)] for m in x_tr_np]
            x_te_np = [m[:nte, :min(m.shape[1], 24)] for m in x_te_np]
            y_tr_multi, y_te_multi = y_tr_multi[:ntr], y_te_multi[:nte]
            pc1_tr, pc1_te = pc1_tr[:ntr], pc1_te[:nte]

        tr_4 = [torch.tensor(m).float() for m in x_tr_np] + [torch.tensor(y_tr_multi).float()]
        te_4 = [torch.tensor(m).float() for m in x_te_np] + [torch.tensor(y_te_multi).float()]
        sp_adni = [0.5, 0.5, 0.5, None]
        eval_mri = (0, 1, 2)

        for m_name, fn in [
            ("flow_v_base", lambda: fit_pilot_flow_v(tr_4, 3, variant="base", iterations=adni_iters, sparseness=sp_adni, positivity="positive", topology="graph", path_graph=ADNI_PATH_GRAPH, random_state=seed)),
            ("flow_v_pcf_05", lambda: fit_pilot_flow_v(tr_4, 3, variant="pcf_05", iterations=adni_iters, sparseness=sp_adni, positivity="positive", topology="graph", path_graph=ADNI_PATH_GRAPH, random_state=seed)),
            ("flow_v_spline1d", lambda: fit_pilot_flow_v(tr_4, 3, variant="spline1d", iterations=adni_iters, sparseness=sp_adni, positivity="positive", topology="graph", path_graph=ADNI_PATH_GRAPH, random_state=seed)),
            ("flow_v_actnorm_spline1d", lambda: fit_actnorm_spline_flow_v(tr_4, 3, iterations=adni_iters, sparseness=sp_adni, positivity="positive", topology="graph", path_graph=ADNI_PATH_GRAPH, random_state=seed)),
        ]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ts = time.perf_counter()
                res = fn()
                sec = time.perf_counter() - ts
            met = score_fit_on_split(
                res, tr_4, te_4, pc1_tr, pc1_te, k=3, eval_view_indices=eval_mri,
                y_multi_train=y_tr_multi, y_multi_test=y_te_multi, positivity="positive", track="A", rf_trees=rf_trees,
            )
            rows.append({"Dataset": "ADNI_MCI_ProtocolB_k3", "Track": "A_flow_v", "Model": m_name, "Seed": seed, "k": 3, "fit_seconds": sec, **met})

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ts = time.perf_counter()
            r_sb = fit_bounded_flow_simr(tr_4, 3, use_tier2_orthseed=False, scale_bound=0.35, iterations=adni_iters, topology="graph", path_graph=ADNI_PATH_GRAPH, eval_view_indices=eval_mri, random_state=seed)
            sec_sb = time.perf_counter() - ts
            ts = time.perf_counter()
            r_t2 = fit_bounded_flow_simr(tr_4, 3, use_tier2_orthseed=True, scale_bound=0.35, iterations=adni_iters, topology="graph", path_graph=ADNI_PATH_GRAPH, eval_view_indices=eval_mri, random_state=seed)
            sec_t2 = time.perf_counter() - ts
            ts = time.perf_counter()
            r_gp = fit_pilot_flow_simr(tr_4, 3, variant="gpca", iterations=adni_iters, topology="graph", path_graph=ADNI_PATH_GRAPH, eval_view_indices=eval_mri, random_state=seed)
            sec_gp = time.perf_counter() - ts
            r_gp_vm = apply_gpca_tier1_varimax(r_gp, tr_4, k=3, eval_view_indices=eval_mri)

        for m_name, r_obj, s_val in [
            ("flow_simr_base_sb035", r_sb, sec_sb),
            ("flow_simr_tier2_orthseed_sb035", r_t2, sec_t2),
            ("flow_simr_gpca", r_gp, sec_gp),
            ("flow_simr_gpca_varimax", r_gp_vm, sec_gp),
        ]:
            met = score_fit_on_split(
                r_obj, tr_4, te_4, pc1_tr, pc1_te, k=3, eval_view_indices=eval_mri,
                y_multi_train=y_tr_multi, y_multi_test=y_te_multi, track="B", rf_trees=rf_trees,
            )
            rows.append({"Dataset": "ADNI_MCI_ProtocolB_k3", "Track": "B_flow_simr", "Model": m_name, "Seed": seed, "k": 3, "fit_seconds": s_val, **met})
        print(f"[done] ADNI_MCI_ProtocolB_k3 seed={seed}", flush=True)

    raw_df = pd.DataFrame(rows)
    mcols = [
        "oos_r2", "xv_oos_r2", "axis_oos_r2", "best_single_axis_r2", "mean11_r2",
        "coh_L", "coh_S_unrot", "coh_J_flow", "coord_spearman_rho", "min_coord_spearman_rho",
        "hoyer_S", "support_recovery_S", "cmc_latent_u", "cond_impute_r2", "fit_seconds",
    ]
    sum_df = raw_df.groupby(["Dataset", "Track", "Model"], sort=False)[mcols].mean().reset_index()

    # Paired contrasts
    c_specs = [
        ("A_flow_v", "flow_v_actnorm_spline1d", "flow_v_base", "oos_r2"),
        ("A_flow_v", "flow_v_actnorm_spline1d", "flow_v_spline1d", "oos_r2"),
        ("A_flow_v", "flow_v_actnorm_spline1d", "flow_v_pcf_05", "oos_r2"),
        ("A_flow_v", "flow_v_actnorm_spline1d", "flow_v_base", "coord_spearman_rho"),
        ("B_flow_simr", "flow_simr_tier2_orthseed_sb035", "flow_simr_base_sb035", "oos_r2"),
        ("B_flow_simr", "flow_simr_tier2_orthseed_sb035", "flow_simr_base_sb035", "axis_oos_r2"),
        ("B_flow_simr", "flow_simr_tier2_orthseed_sb035", "flow_simr_base_sb035", "support_recovery_S"),
        ("B_flow_simr", "flow_simr_gpca_varimax", "flow_simr_gpca", "axis_oos_r2"),
        ("B_flow_simr", "flow_simr_gpca_varimax", "flow_simr_gpca", "best_single_axis_r2"),
        ("B_flow_simr", "flow_simr_gpca_varimax", "flow_simr_gpca", "hoyer_S"),
    ]
    c_rows = []
    for trk, ma, mb, met_c in c_specs:
        sub = raw_df[raw_df["Track"] == trk]
        for ds, grp in sub.groupby("Dataset"):
            piv = grp.pivot_table(index="Seed", columns="Model", values=met_c, aggfunc="mean")
            if ma not in piv.columns or mb not in piv.columns:
                continue
            pair = piv[[ma, mb]].dropna()
            if len(pair) == 0:
                continue
            diffs = (pair[ma] - pair[mb]).to_numpy(dtype=float)
            nz = diffs[np.abs(diffs) > 1e-12]
            pw = float(wilcoxon(diffs).pvalue) if len(nz) >= 3 else 1.0
            pp = float(exact_sign_flip_permutation(diffs, n_mc=max(2, n_perms)))
            c_rows.append({
                "Dataset": ds, "Track": trk, "Metric": met_c, "Candidate": ma, "Reference": mb,
                "N_Seeds": int(len(pair)), "Mean_Candidate": float(pair[ma].mean()),
                "Mean_Reference": float(pair[mb].mean()), "Mean_Diff": float(diffs.mean()),
                "Wins": int(np.sum(diffs > 1e-9)), "Losses": int(np.sum(diffs < -1e-9)),
                "P_SignFlip": pp, "P_Wilcoxon": pw,
            })
    con_df = pd.DataFrame(c_rows)

    raw_p = out_dir / f"followup_raw{suffix}.csv"
    sum_p = out_dir / f"followup_summary{suffix}.csv"
    con_p = out_dir / f"followup_contrasts{suffix}.csv"
    raw_df.to_csv(raw_p, index=False)
    sum_df.to_csv(sum_p, index=False)
    con_df.to_csv(con_p, index=False)

    # Diagnostic Figure 4: Follow-up comparison (ActNorm+Spline1D on Track A & Scale-Bounded / GPCA-Varimax on Track B)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.6), dpi=(100 if suffix else 180))
    sub_a = raw_df[raw_df["Track"] == "A_flow_v"]
    ds_a = [d for d in ["Synth_Poly_k3", "Synth_LocalWarp_k3", "Synth_Skew_k1", "Diabetes_k2", "ADNI_MCI_ProtocolB_k3"]
            if d in sub_a["Dataset"].unique()]
    mods_a = ["flow_v_base", "flow_v_pcf_05", "flow_v_spline1d", "flow_v_actnorm_spline1d"]
    cols_a = ["#5e81ac", "#d08770", "#88c0d0", "#2a9d8f"]
    xa = np.arange(len(ds_a))
    wa = 0.19
    for idx_m, m_nm in enumerate(mods_a):
        offs = (idx_m - 1.5) * wa
        ms = [float(sub_a[(sub_a.Dataset == d) & (sub_a.Model == m_nm)]["oos_r2"].mean()) for d in ds_a]
        ss = [float(sub_a[(sub_a.Dataset == d) & (sub_a.Model == m_nm)]["oos_r2"].std(ddof=0)) for d in ds_a]
        axes[0].bar(xa + offs, ms, width=wa * 0.92, yerr=ss, capsize=2, label=m_nm, color=cols_a[idx_m], edgecolor="white")
    axes[0].set_xticks(xa)
    axes[0].set_xticklabels([d.replace("_ProtocolB_k3", "").replace("Synth_", "S:") for d in ds_a], rotation=18, ha="right")
    axes[0].set_title("Track A Follow-Up: ActNorm1d + Monotone Spline OOS R² (↑)", fontweight="bold")
    axes[0].legend(fontsize=8.2)
    axes[0].grid(axis="y", alpha=0.25, linestyle="--")

    sub_b = raw_df[raw_df["Track"] == "B_flow_simr"]
    ds_b = [d for d in ["Synth_Poly_k3", "Synth_LocalWarp_k3", "Diabetes_k2", "ADNI_MCI_ProtocolB_k3"]
            if d in sub_b["Dataset"].unique()]
    mods_b = ["flow_simr_base_sb035", "flow_simr_tier2_orthseed_sb035", "flow_simr_gpca", "flow_simr_gpca_varimax"]
    cols_b = ["#5e81ac", "#e76f51", "#b56576", "#2a9d8f"]
    xb = np.arange(len(ds_b))
    wb = 0.19
    for idx_m, m_nm in enumerate(mods_b):
        offs = (idx_m - 1.5) * wb
        ms = [float(sub_b[(sub_b.Dataset == d) & (sub_b.Model == m_nm)]["axis_oos_r2"].mean()) for d in ds_b]
        ss = [float(sub_b[(sub_b.Dataset == d) & (sub_b.Model == m_nm)]["axis_oos_r2"].std(ddof=0)) for d in ds_b]
        axes[1].bar(xb + offs, ms, width=wb * 0.92, yerr=ss, capsize=2, label=m_nm.replace("flow_simr_", ""), color=cols_b[idx_m], edgecolor="white")
    axes[1].set_xticks(xb)
    axes[1].set_xticklabels([d.replace("_ProtocolB_k3", "").replace("Synth_", "S:") for d in ds_b], rotation=18, ha="right")
    axes[1].set_title("Track B Follow-Up: Scale-Bounded Tier 2 & GPCA+Varimax Axis RF R² (↑)", fontweight="bold")
    axes[1].legend(fontsize=8.2)
    axes[1].grid(axis="y", alpha=0.25, linestyle="--")

    fig.tight_layout()
    fig_p = out_dir / f"fig_followup_fixes{suffix}.png"
    fig.savefig(fig_p, bbox_inches="tight")
    plt.close(fig)

    wall_s = time.perf_counter() - t0_wall
    prov = {
        "smoke": bool(args.smoke),
        "cli": [sys.executable, *sys.argv],
        "utc_start": utc_start,
        "utc_end": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "wall_seconds": round(wall_s, 3),
        "host": platform.node(),
        "pysimlr_git_sha": _git_sha(REPO),
        "contract_checks": stage0,
        "outputs": {
            "raw_csv": str(raw_p),
            "summary_csv": str(sum_p),
            "contrasts_csv": str(con_p),
            "figure": str(fig_p),
        },
    }
    prov_p = out_dir / f"provenance_followup{suffix}.json"
    prov_p.write_text(json.dumps(prov, indent=2), encoding="utf-8")
    print(f"[Done] Follow-up finished in {wall_s:.2f}s -> {sum_p.name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
