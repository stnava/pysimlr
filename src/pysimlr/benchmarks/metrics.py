
import torch
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from typing import Dict, Any, List, Optional

from pysimlr import adjusted_rvcoef
from pysimlr.utils import invariant_orthogonality_defect
from pysimlr.interpretability import (
    analyze_first_layer_alignment,
    attribute_prediction_to_features,
    attribute_shared_to_first_layer,
)


def latent_recovery_score(u_pred: torch.Tensor, u_true: torch.Tensor) -> float:
    """Calculate Adjusted RV coefficient between predicted and true latents."""
    return adjusted_rvcoef(u_pred, u_true)


def in_sample_latent_linear_fit_r2(u_pred: torch.Tensor, y_true: np.ndarray) -> float:
    """
    Deprecated: Use heldout_outcome_r2_score for true benchmarks.
    Calculate R2 score for outcome prediction using predicted latents, fitting on the same data.
    """
    u_np = u_pred.detach().cpu().numpy()
    if len(y_true.shape) == 1:
        y_true = y_true.reshape(-1, 1)
    reg = LinearRegression().fit(u_np, y_true)
    return r2_score(y_true, reg.predict(u_np))


def outcome_r2_score(u_pred: torch.Tensor, y_true: np.ndarray) -> float:
    """Alias for backward compatibility."""
    return in_sample_latent_linear_fit_r2(u_pred, y_true)


def heldout_outcome_r2_score(
    u_train: torch.Tensor,
    y_train: np.ndarray,
    u_test: torch.Tensor,
    y_test: np.ndarray,
) -> float:
    """
    Calculate R2 score for outcome prediction on held-out data.
    Fits a downstream model on train latents and scores on test latents.
    """
    u_train_np = u_train.detach().cpu().numpy()
    u_test_np = u_test.detach().cpu().numpy()

    if len(y_train.shape) == 1:
        y_train = y_train.reshape(-1, 1)
    if len(y_test.shape) == 1:
        y_test = y_test.reshape(-1, 1)

    reg = LinearRegression().fit(u_train_np, y_train)
    y_pred = reg.predict(u_test_np)
    return r2_score(y_test, y_pred)


def heldout_outcome_mse(
    u_train: torch.Tensor,
    y_train: np.ndarray,
    u_test: torch.Tensor,
    y_test: np.ndarray,
) -> float:
    """Calculate MSE for outcome prediction on held-out data."""
    u_train_np = u_train.detach().cpu().numpy()
    u_test_np = u_test.detach().cpu().numpy()

    if len(y_train.shape) == 1:
        y_train = y_train.reshape(-1, 1)
    if len(y_test.shape) == 1:
        y_test = y_test.reshape(-1, 1)

    reg = LinearRegression().fit(u_train_np, y_train)
    y_pred = reg.predict(u_test_np)
    return mean_squared_error(y_test, y_pred)


def reconstruction_mse(data: List[torch.Tensor], recons: List[torch.Tensor]) -> float:
    """Calculate mean normalized reconstruction error across modalities."""
    errors = []
    for x, r in zip(data, recons):
        err = torch.norm(x - r, p="fro").item() / (torch.norm(x, p="fro").item() + 1e-10)
        errors.append(err)
    return float(np.mean(errors))


def reconstruction_mse_summary(data: List[torch.Tensor], recons: List[torch.Tensor]) -> Dict[str, float]:
    """Return mean and standard deviation of normalized reconstruction error."""
    errors = []
    for x, r in zip(data, recons):
        err = torch.norm(x - r, p="fro").item() / (torch.norm(x, p="fro").item() + 1e-10)
        errors.append(err)
    return {
        "recon_error": float(np.mean(errors)),
        "recon_error_std": float(np.std(errors)),
    }


def latent_variance_diagnostics(u: torch.Tensor) -> Dict[str, float]:
    """Check for latent collapse and scaling issues."""
    u_std = torch.std(u, dim=0)

    n = u.shape[0]
    u_c = u - u.mean(dim=0, keepdim=True)
    cov = (u_c.T @ u_c) / max(1, (n - 1))
    mask = ~torch.eye(u.shape[1], device=u.device).bool()
    off_diag_norm = torch.norm(cov[mask]).item() if mask.any() else 0.0

    u_norms = torch.linalg.norm(u, dim=1)
    return {
        "u_std_mean": float(torch.mean(u_std).item()),
        "u_std_min": float(torch.min(u_std).item()),
        "u_off_diag_cov": float(off_diag_norm),
        "u_norm_sd": float(torch.std(u_norms).item()),
        "collapsed_dims": int(torch.sum(u_std < 1e-3).item()),
    }


def shared_private_diagnostics(
    shared_latents: List[torch.Tensor],
    private_latents: List[torch.Tensor],
) -> Dict[str, float]:
    """Calculate diagnostics specifically for models with shared/private branches."""
    diagnostics: Dict[str, float] = {}

    for i, (s, p) in enumerate(zip(shared_latents, private_latents)):
        s_var = torch.mean(torch.var(s, dim=0)).item()
        p_var = torch.mean(torch.var(p, dim=0)).item()

        s_c = s - s.mean(dim=0, keepdim=True)
        p_c = p - p.mean(dim=0, keepdim=True)
        cross_cov = torch.abs((s_c.T @ p_c) / max(1, (s.shape[0] - 1))).mean().item()

        diagnostics[f"mod{i}_shared_var"] = s_var
        diagnostics[f"mod{i}_private_var"] = p_var
        diagnostics[f"mod{i}_cross_cov"] = cross_cov

    return diagnostics


def calculate_v_orthogonality(v_mats: List[torch.Tensor]) -> float:
    """Calculate mean invariant orthogonality defect across all modality V matrices."""
    defects = [invariant_orthogonality_defect(v).item() for v in v_mats]
    return float(np.mean(defects))


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().item()
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))


def first_layer_sparsity_metrics(first_layer: Dict[str, Any]) -> Dict[str, float]:
    """Summarize first-layer sparsity and orthogonality across modalities."""
    summaries = first_layer.get("sparsity_summary", [])
    orthogonality = first_layer.get("orthogonality_defect", [])
    component_density: List[float] = []
    component_l0: List[float] = []

    for summary in summaries:
        component_density.extend([float(x) for x in summary.get("component_density", [])])
        component_l0.extend([float(x) for x in summary.get("component_l0", [])])

    return {
        "first_layer_density_mean": _safe_mean(component_density),
        "first_layer_density_sd": float(np.std(component_density)) if component_density else 0.0,
        "first_layer_l0_mean": _safe_mean(component_l0),
        "first_layer_l0_sd": float(np.std(component_l0)) if component_l0 else 0.0,
        "first_layer_orthogonality_mean": _safe_mean([float(x) for x in orthogonality]),
    }


def alignment_metrics_from_report(report: Dict[str, Any]) -> Dict[str, float]:
    """Extract compact benchmark metrics from a PR3 alignment report."""
    modalities = report.get("modalities", [])
    global_r2 = [_safe_float(m.get("global_r2")) for m in modalities]
    mean_abs_corr = []
    feature_concentration = []

    for modality in modalities:
        corr = modality.get("component_correlation")
        if isinstance(corr, torch.Tensor):
            mean_abs_corr.append(float(torch.abs(corr).mean().item()))
        feat = modality.get("feature_importance")
        if isinstance(feat, torch.Tensor) and feat.numel() > 0:
            feat = torch.abs(feat).flatten()
            feature_concentration.append(float(torch.max(feat).item() / (torch.sum(feat).item() + 1e-8)))

    return {
        "first_layer_alignment_r2_mean": _safe_mean(global_r2),
        "first_layer_alignment_corr_mean": _safe_mean(mean_abs_corr),
        "first_layer_feature_concentration_mean": _safe_mean(feature_concentration),
    }


def shared_attribution_metrics_from_report(report: Dict[str, Any]) -> Dict[str, float]:
    """Extract compact benchmark metrics from shared-to-first-layer attribution."""
    combined = report.get("combined", {})
    per_modality = report.get("per_modality", [])

    component_importance = combined.get("component_importance")
    if isinstance(component_importance, torch.Tensor) and component_importance.numel() > 0:
        component_importance = torch.abs(component_importance).flatten()
        component_concentration = float(torch.max(component_importance).item() / (torch.sum(component_importance).item() + 1e-8))
    else:
        component_concentration = 0.0

    modality_r2 = [_safe_float(m.get("global_r2")) for m in per_modality]
    return {
        "shared_to_first_layer_r2_mean": _safe_mean(modality_r2),
        "shared_component_concentration": component_concentration,
    }


def prediction_preservation_metrics_from_report(report: Dict[str, Any]) -> Dict[str, float]:
    """Quantify how much predictive signal is preserved in the first-layer view."""
    per_modality = report.get("per_modality", [])
    modality_r2 = [_safe_float(m.get("global_r2")) for m in per_modality]
    baseline = report.get("shared_latent_baseline")
    baseline_r2 = _safe_float(None if baseline is None else baseline.get("global_r2"))

    if baseline_r2 <= 0:
        preservation = 0.0
    else:
        preservation = min(1.0, max(0.0, _safe_mean(modality_r2) / baseline_r2))

    return {
        "first_layer_prediction_r2_mean": _safe_mean(modality_r2),
        "shared_latent_prediction_r2": baseline_r2,
        "first_layer_prediction_preservation": preservation,
    }


def calculate_all_metrics(
    u_pred: torch.Tensor,
    u_true: torch.Tensor,
    y_true: np.ndarray,
    data: List[torch.Tensor],
    recons: List[torch.Tensor],
    shared_latents: Optional[List[torch.Tensor]] = None,
    private_latents: Optional[List[torch.Tensor]] = None,
    v_mats: Optional[List[torch.Tensor]] = None,
    u_train: Optional[torch.Tensor] = None,
    y_train: Optional[np.ndarray] = None,
    first_layer: Optional[Dict[str, Any]] = None,
    interpretability: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Utility to calculate a standard suite of metrics.

    When first-layer and interpretability payloads are available, this also
    returns PR4-style interpretability-preservation metrics.
    """
    recon_res = reconstruction_mse_summary(data, recons)
    res: Dict[str, Any] = {
        "recovery": latent_recovery_score(u_pred, u_true),
        "recon_error": recon_res["recon_error"],
        "recon_error_std": recon_res["recon_error_std"],
    }

    if u_train is not None and y_train is not None:
        res["test_r2"] = heldout_outcome_r2_score(u_train, y_train, u_pred, y_true)
        res["test_mse"] = heldout_outcome_mse(u_train, y_train, u_pred, y_true)
    else:
        res["in_sample_r2"] = in_sample_latent_linear_fit_r2(u_pred, y_true)
        res["test_r2"] = res["in_sample_r2"]

    res.update(latent_variance_diagnostics(u_pred))

    if shared_latents is not None and private_latents is not None:
        res.update(shared_private_diagnostics(shared_latents, private_latents))

    if v_mats is not None:
        res["orthogonality_defect"] = calculate_v_orthogonality(v_mats)

    if first_layer is not None:
        res.update(first_layer_sparsity_metrics(first_layer))

    if interpretability is not None:
        alignment = interpretability.get("deep_layer_alignment")
        if alignment is not None:
            res.update(alignment_metrics_from_report(alignment))
        shared = interpretability.get("shared_to_first_layer")
        if shared is not None:
            res.update(shared_attribution_metrics_from_report(shared))

    if first_layer is not None and interpretability is None:
        temp_model = {"u": u_pred, "first_layer": first_layer}
        try:
            res.update(alignment_metrics_from_report(analyze_first_layer_alignment(temp_model)))
            res.update(shared_attribution_metrics_from_report(attribute_shared_to_first_layer(temp_model)))
        except Exception:
            pass

    if first_layer is not None:
        temp_model = {"u": u_pred, "first_layer": first_layer}
        try:
            y_tensor = torch.as_tensor(y_true).float()
            pred_report = attribute_prediction_to_features(temp_model, y_tensor)
            res.update(prediction_preservation_metrics_from_report(pred_report))
        except Exception:
            res.update({
                "first_layer_prediction_r2_mean": 0.0,
                "shared_latent_prediction_r2": 0.0,
                "first_layer_prediction_preservation": 0.0,
            })

    return res
