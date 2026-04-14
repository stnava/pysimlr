import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
from ..utils import procrustes_r2, adjusted_rvcoef

def latent_recovery_score(u_pred: torch.Tensor, u_true: torch.Tensor) -> float:
    """
    Measure the recovery of the true latent space after Procrustes alignment.
    """
    return procrustes_r2(u_true, u_pred)

def in_sample_latent_linear_fit_r2(u_pred: torch.Tensor, y_true: np.ndarray) -> float:
    """
    Measure how well the learned latents can linearly predict an outcome (in-sample).
    """
    from sklearn.linear_model import LinearRegression
    u_np = u_pred.detach().cpu().numpy()
    model = LinearRegression().fit(u_np, y_true)
    return float(model.score(u_np, y_true))

def outcome_r2_score(u_pred: torch.Tensor, y_true: np.ndarray) -> float:
    """
    Compute the R-squared score for an external outcome predicted from the latents.
    """
    return in_sample_latent_linear_fit_r2(u_pred, y_true)

def heldout_outcome_r2_score(
    u_train: torch.Tensor, y_train: np.ndarray,
    u_test: torch.Tensor, y_test: np.ndarray
) -> float:
    """
    Cross-validated R-squared score for outcome prediction on held-out data.
    """
    from sklearn.linear_model import LinearRegression
    model = LinearRegression().fit(u_train.detach().cpu().numpy(), y_train)
    return float(model.score(u_test.detach().cpu().numpy(), y_test))

def heldout_outcome_mse(
    u_train: torch.Tensor, y_train: np.ndarray,
    u_test: torch.Tensor, y_test: np.ndarray
) -> float:
    """
    Mean Squared Error for outcome prediction on held-out data.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    model = LinearRegression().fit(u_train.detach().cpu().numpy(), y_train)
    preds = model.predict(u_test.detach().cpu().numpy())
    return float(mean_squared_error(y_test, preds))

def reconstruction_mse(data: List[torch.Tensor], recons: List[torch.Tensor]) -> float:
    """
    Compute the average Mean Squared Error (MSE) across all modalities.
    """
    mses = [torch.mean((d - r)**2).item() for d, r in zip(data, recons)]
    return float(np.mean(mses))

def reconstruction_mse_summary(data: List[torch.Tensor], recons: List[torch.Tensor]) -> Dict[str, float]:
    """
    Compute Reconstruction MSE for each modality independently.
    """
    return {f"modality_{i}_mse": torch.mean((d - r)**2).item() for i, (d, r) in enumerate(zip(data, recons))}

def latent_variance_diagnostics(u: torch.Tensor) -> Dict[str, float]:
    """
    Analyze the variance distribution of the latent space to check for collapse.
    """
    stds = torch.std(u, dim=0)
    collapsed = torch.sum(stds < 1e-4).item()
    res = {
        "latent_mean_std": torch.mean(stds).item(),
        "latent_min_std": torch.min(stds).item(),
        "latent_max_std": torch.max(stds).item(),
        "latent_std_ratio": (torch.min(stds) / (torch.max(stds) + 1e-8)).item(),
        "collapsed_dims": float(collapsed)
    }
    res["u_std_mean"] = res["latent_mean_std"]
    return res

def shared_private_diagnostics(
    shared_latents: List[torch.Tensor],
    private_latents: List[torch.Tensor]
) -> Dict[str, float]:
    """
    Compute orthogonality between shared and private latent components.
    """
    overlaps = [adjusted_rvcoef(s, p) for s, p in zip(shared_latents, private_latents)]
    res = {
        "mean_shared_private_overlap": float(np.mean(overlaps)),
        "max_shared_private_overlap": float(np.max(overlaps))
    }
    for i, (s, p) in enumerate(zip(shared_latents, private_latents)):
        res[f"mod{i}_cross_cov"] = float(adjusted_rvcoef(s, p))
        res[f"mod{i}_shared_var"] = float(torch.var(s).item())
    return res

def calculate_v_orthogonality(v_mats: List[torch.Tensor]) -> float:
    """
    Compute the average orthogonality defect across all basis matrices V.
    """
    from ..utils import invariant_orthogonality_defect
    defects = [invariant_orthogonality_defect(v).item() for v in v_mats]
    return float(np.mean(defects))

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if isinstance(value, torch.Tensor):
            return float(value.item())
        return float(value)
    except:
        return default

def _safe_mean(values: List[float]) -> float:
    if not values: return 0.0
    return float(np.mean([_safe_float(v) for v in values]))

def first_layer_sparsity_metrics(first_layer: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract sparsity and density metrics for the interpretable first layer.
    """
    if not first_layer: return {}
    
    # Try different potential schemas
    densities = []
    l0s = []
    
    if "sparsity_summary" in first_layer:
        for mod in first_layer["sparsity_summary"]:
            if "component_density" in mod:
                densities.append(np.mean(mod["component_density"]))
            if "component_l0" in mod:
                l0s.append(np.mean(mod["component_l0"]))
    elif "modalities" in first_layer:
        for mod in first_layer["modalities"]:
            if "summary" in mod:
                densities.append(mod["summary"].get("density"))
                l0s.append(mod["summary"].get("l0"))

    orth_defect = first_layer.get("orthogonality_defect")
    
    res = {}
    if densities:
        res["first_layer_density_mean"] = _safe_mean(densities)
    if l0s:
        res["first_layer_l0_mean"] = _safe_mean(l0s)
    if orth_defect is not None:
        res["first_layer_orthogonality_mean"] = _safe_mean(orth_defect if isinstance(orth_defect, list) else [orth_defect])
        
    return res

def alignment_metrics_from_report(report: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract alignment metrics from report.
    """
    if not report: return {}
    
    # Try different schemas
    r2s = []
    corrs = []
    
    if "modalities" in report:
        for mod in report["modalities"]:
            r2s.append(mod.get("global_r2"))
            if "component_correlation" in mod:
                corrs.append(torch.mean(torch.abs(mod["component_correlation"])).item())
    elif "modality_alignments" in report:
        # Another potential schema
        for mod in report["modality_alignments"]:
            r2s.append(mod.get("rv_to_first_layer"))

    res = {}
    if r2s:
        res["first_layer_alignment_r2_mean"] = _safe_mean(r2s)
    if corrs:
        res["first_layer_alignment_corr_mean"] = _safe_mean(corrs)
    return res

def shared_attribution_metrics_from_report(report: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract metrics on how well shared consensus is attributed to the first layer.
    """
    if not report: return {}
    
    r2s = []
    concentration = 0.0
    
    if "per_modality" in report:
        for mod in report["per_modality"]:
            r2s.append(mod.get("global_r2"))
    
    if "combined" in report and "component_importance" in report["combined"]:
        imp = report["combined"]["component_importance"]
        concentration = (torch.max(imp) / (torch.sum(imp) + 1e-8)).item()

    res = {}
    if r2s:
        res["shared_to_first_layer_r2_mean"] = _safe_mean(r2s)
    res["shared_component_concentration"] = float(concentration)
    return res

def prediction_preservation_metrics_from_report(report: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract metrics on how well the first layer preserves predictive power.
    """
    if not report: return {}
    
    r2s = []
    preservation = 0.0
    
    if "per_modality" in report:
        for mod in report["per_modality"]:
            r2s.append(mod.get("global_r2"))
            
    if "shared_latent_baseline" in report:
        base_r2 = _safe_float(report["shared_latent_baseline"].get("global_r2"))
        if r2s:
            mean_r2 = _safe_mean(r2s)
            preservation = mean_r2 / (base_r2 + 1e-8)

    res = {}
    if r2s:
        res["first_layer_prediction_r2_mean"] = _safe_mean(r2s)
    res["first_layer_prediction_preservation"] = float(preservation)
    return res

def calculate_all_metrics(
    u_pred: torch.Tensor,
    u_true: Optional[torch.Tensor] = None,
    y_true: Optional[np.ndarray] = None,
    data: Optional[List[torch.Tensor]] = None,
    reconstructions: Optional[List[torch.Tensor]] = None,
    **kwargs
) -> Dict[str, float]:
    """
    Master function to compute a comprehensive suite of SiMLR performance metrics.
    """
    metrics = {}
    
    # 1. Latent Recovery
    if u_true is not None:
        val = latent_recovery_score(u_pred, u_true)
        metrics["latent_recovery"] = val
        metrics["recovery"] = val
        
    # 2. Prediction
    if y_true is not None:
        val = outcome_r2_score(u_pred, y_true)
        metrics["outcome_r2"] = val
        metrics["test_r2"] = val
        
    # 3. Reconstruction
    if data is not None and reconstructions is not None:
        val = reconstruction_mse(data, reconstructions)
        metrics["reconstruction_mse"] = val
        metrics["recon_error"] = val
        
    # 4. Latent Diagnostics
    metrics.update(latent_variance_diagnostics(u_pred))
    
    # 5. Shared/Private
    shared_l = kwargs.get("shared_latents")
    private_l = kwargs.get("private_latents")
    if shared_l is not None and private_l is not None:
        metrics.update(shared_private_diagnostics(shared_l, private_l))
        
    # 6. Orthogonality
    v_mats = kwargs.get("v_mats")
    if v_mats is not None:
        metrics["orthogonality_defect"] = calculate_v_orthogonality(v_mats)
        
    # 7. Interpretability
    first_layer = kwargs.get("first_layer")
    if first_layer is not None:
        metrics.update(first_layer_sparsity_metrics(first_layer))
        
    report = kwargs.get("interpretability")
    if report is not None:
        # Interpretability reports often have sub-reports
        if "deep_layer_alignment" in report:
            metrics.update(alignment_metrics_from_report(report["deep_layer_alignment"]))
        else:
            metrics.update(alignment_metrics_from_report(report))
            
        if "shared_to_first_layer" in report:
            metrics.update(shared_attribution_metrics_from_report(report["shared_to_first_layer"]))
        else:
            metrics.update(shared_attribution_metrics_from_report(report))
            
        if "prediction_attribution" in report:
            metrics.update(prediction_preservation_metrics_from_report(report["prediction_attribution"]))
        else:
            metrics.update(prediction_preservation_metrics_from_report(report))
        
    # Outcome prediction on held-out data (if u_train/y_train provided)
    u_train = kwargs.get("u_train")
    y_train = kwargs.get("y_train")
    if u_train is not None and y_train is not None and y_true is not None:
        metrics["heldout_outcome_r2"] = heldout_outcome_r2_score(u_train, y_train, u_pred, y_true)
        metrics["heldout_outcome_mse"] = heldout_outcome_mse(u_train, y_train, u_pred, y_true)
        metrics["test_mse"] = metrics["heldout_outcome_mse"]

    return metrics
