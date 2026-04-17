import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
from ..utils import procrustes_r2, adjusted_rvcoef

def latent_recovery_score(u_pred: torch.Tensor, u_true: torch.Tensor) -> float:
    return procrustes_r2(u_true, u_pred)

def outcome_r2_score(u_pred: torch.Tensor, y_true: np.ndarray) -> float:
    from sklearn.linear_model import LinearRegression
    u_np = u_pred.detach().cpu().numpy()
    model = LinearRegression().fit(u_np, y_true)
    return float(model.score(u_np, y_true))

def cross_val_metrics(
    u_train: torch.Tensor, y_train: np.ndarray,
    u_test: torch.Tensor, y_test: np.ndarray,
    is_classification: bool = False
) -> Dict[str, float]:
    u_train_np = u_train.detach().cpu().numpy()
    u_test_np = u_test.detach().cpu().numpy()
    if is_classification:
        from sklearn.linear_model import LogisticRegression
        y_train_int = y_train.astype(int).ravel()
        y_test_int = y_test.astype(int).ravel()
        model = LogisticRegression(max_iter=1000).fit(u_train_np, y_train_int)
        train_perf = float(model.score(u_train_np, y_train_int))
        test_perf = float(model.score(u_test_np, y_test_int))
    else:
        from sklearn.linear_model import LinearRegression
        model = LinearRegression().fit(u_train_np, y_train)
        train_perf = float(model.score(u_train_np, y_train))
        test_perf = float(model.score(u_test_np, y_test))
    return {"train": train_perf, "test": test_perf, "gap": train_perf - test_perf}

def reconstruction_mse(data: List[torch.Tensor], recons: List[torch.Tensor]) -> float:
    mses = [torch.mean((d - r)**2).item() for d, r in zip(data, recons)]
    return float(np.mean(mses))

def latent_variance_diagnostics(u: torch.Tensor) -> Dict[str, float]:
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
    overlaps = [adjusted_rvcoef(s, p) for s, p in zip(shared_latents, private_latents)]
    res = {"mean_shared_private_overlap": float(np.mean(overlaps)), "max_shared_private_overlap": float(np.max(overlaps))}
    for i, (s, p) in enumerate(zip(shared_latents, private_latents)):
        res[f"mod{i}_cross_cov"] = float(adjusted_rvcoef(s, p))
        res[f"mod{i}_shared_var"] = float(torch.var(s).item())
    return res

def calculate_v_orthogonality(v_mats: List[torch.Tensor]) -> float:
    from ..utils import invariant_orthogonality_defect
    return float(np.mean([invariant_orthogonality_defect(v).item() for v in v_mats]))

def first_layer_sparsity_metrics(first_layer: Dict[str, Any]) -> Dict[str, float]:
    if not first_layer: return {}
    densities, l0s = [], []
    if "sparsity_summary" in first_layer:
        for mod in first_layer["sparsity_summary"]:
            if "component_density" in mod: densities.append(np.mean(mod["component_density"]))
            if "component_l0" in mod: l0s.append(np.mean(mod["component_l0"]))
    elif "modalities" in first_layer:
        for mod in first_layer["modalities"]:
            if "summary" in mod:
                densities.append(mod["summary"].get("density"))
                l0s.append(mod["summary"].get("l0"))
    res = {}
    if densities: res["first_layer_density_mean"] = float(np.mean(densities))
    if l0s: res["first_layer_l0_mean"] = float(np.mean(l0s))
    if "orthogonality_defect" in first_layer:
        od = first_layer["orthogonality_defect"]
        res["first_layer_orthogonality_mean"] = float(np.mean(od) if isinstance(od, list) else od)
    return res

def alignment_metrics_from_report(report: Dict[str, Any]) -> Dict[str, float]:
    if not report: return {}
    r2s, corrs = [], []
    if "modalities" in report:
        for mod in report["modalities"]:
            r2s.append(mod.get("global_r2"))
            if "component_correlation" in mod:
                c = mod["component_correlation"]
                if isinstance(c, torch.Tensor): c = c.cpu().numpy()
                mask = ~np.eye(c.shape[0], dtype=bool)
                corrs.append(np.mean(np.abs(c[mask])))
    elif "modality_alignments" in report:
        for mod in report["modality_alignments"]: r2s.append(mod.get("rv_to_first_layer"))
    res = {}
    if r2s: res["first_layer_alignment_r2_mean"] = float(np.mean([r for r in r2s if r is not None]))
    if corrs: res["first_layer_alignment_corr_mean"] = float(np.mean(corrs))
    return res

def shared_attribution_metrics_from_report(report: Dict[str, Any]) -> Dict[str, float]:
    if not report: return {}
    r2s, concentration = [], 0.0
    if "per_modality" in report:
        for mod in report["per_modality"]: r2s.append(mod.get("global_r2"))
    if "combined" in report and "component_importance" in report["combined"]:
        imp = report["combined"]["component_importance"]
        concentration = (torch.max(imp) / (torch.sum(imp) + 1e-8)).item()
    res = {}
    if r2s: res["shared_to_first_layer_r2_mean"] = float(np.mean([r for r in r2s if r is not None]))
    res["shared_component_concentration"] = float(concentration)
    return res

def prediction_preservation_metrics_from_report(report: Dict[str, Any]) -> Dict[str, float]:
    if not report: return {}
    r2s, preservation = [], 0.0
    if "per_modality" in report:
        for mod in report["per_modality"]: r2s.append(mod.get("global_r2"))
    if "shared_latent_baseline" in report:
        base_r2 = report["shared_latent_baseline"].get("global_r2", 0.0)
        if r2s: preservation = np.mean(r2s) / (base_r2 + 1e-8)
    res = {}
    if r2s: res["first_layer_prediction_r2_mean"] = float(np.mean([r for r in r2s if r is not None]))
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
    metrics = {}
    if u_true is not None:
        metrics["recovery"] = latent_recovery_score(u_pred, u_true)
        metrics["latent_recovery"] = metrics["recovery"]
    metrics.update(latent_variance_diagnostics(u_pred))
    v_mats = kwargs.get("v_mats")
    if v_mats is not None: metrics["orthogonality_defect"] = calculate_v_orthogonality(v_mats)
    if data is not None and reconstructions is not None:
        metrics["reconstruction_mse"] = reconstruction_mse(data, reconstructions)
        metrics["recon_error"] = metrics["reconstruction_mse"]
    shared_l, private_l = kwargs.get("shared_latents"), kwargs.get("private_latents")
    if shared_l is not None and private_l is not None: metrics.update(shared_private_diagnostics(shared_l, private_l))
    first_layer = kwargs.get("first_layer")
    if first_layer: metrics.update(first_layer_sparsity_metrics(first_layer))
    report = kwargs.get("interpretability")
    if report:
        metrics.update(alignment_metrics_from_report(report.get("deep_layer_alignment") or report))
        metrics.update(shared_attribution_metrics_from_report(report.get("shared_to_first_layer") or report))
        metrics.update(prediction_preservation_metrics_from_report(report.get("prediction_attribution") or report))
    
    u_train, y_train = kwargs.get("u_train"), kwargs.get("y_train")
    if y_true is not None:
        if u_train is not None and y_train is not None:
            y_train_np = y_train.detach().cpu().numpy() if isinstance(y_train, torch.Tensor) else y_train
            y_true_np = y_true.detach().cpu().numpy() if isinstance(y_true, torch.Tensor) else y_true
            unique_y = np.unique(y_train_np); is_classification = len(unique_y) < 10 and np.all(y_train_np % 1 == 0)
            deep_res = cross_val_metrics(u_train, y_train_np, u_pred, y_true_np, is_classification)
            metrics["test_r2"], metrics["train_r2"], metrics["gen_gap"] = deep_res["test"], deep_res["train"], deep_res["gap"]
            if is_classification: metrics["test_accuracy"], metrics["train_accuracy"] = deep_res["test"], deep_res["train"]
            fl_scores_train, fl_scores_test = kwargs.get("first_layer_scores_train"), kwargs.get("first_layer_scores_test")
            if fl_scores_train is not None and fl_scores_test is not None:
                u_lin_train, u_lin_test = torch.cat(fl_scores_train, dim=1), torch.cat(fl_scores_test, dim=1)
                u_lin_train = (u_lin_train - u_lin_train.mean(0)) / (u_lin_train.std(0) + 1e-6)
                u_lin_test = (u_lin_test - u_lin_test.mean(0)) / (u_lin_test.std(0) + 1e-6)
                lin_res = cross_val_metrics(u_lin_train, y_train_np, u_lin_test, y_true_np, is_classification)
                metrics["first_layer_test_r2"], metrics["first_layer_train_r2"], metrics["first_layer_gen_gap"] = lin_res["test"], lin_res["train"], lin_res["gap"]
                if is_classification: metrics["first_layer_test_accuracy"], metrics["first_layer_train_accuracy"] = lin_res["test"], lin_res["train"]
        else:
            # Fallback for simple R2 if u_train is missing but y_true is present
            metrics["test_r2"] = outcome_r2_score(u_pred, y_true)
            
    return metrics
