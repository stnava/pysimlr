from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch

from .utils import invariant_orthogonality_defect


def _sanitize_tensor(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x.detach().cpu(), nan=0.0, posinf=0.0, neginf=0.0)


def _to_2d_target(y: torch.Tensor) -> torch.Tensor:
    y = _sanitize_tensor(y)
    if y.ndim == 1:
        y = y.unsqueeze(1)
    if y.ndim != 2:
        raise ValueError("target must be a 1D or 2D tensor")
    return y


def _fit_linear_map(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    l2: float = 1e-6,
) -> Dict[str, Any]:
    """Fit a ridge-stabilized linear map source -> target."""
    x = _sanitize_tensor(source).to(torch.float32)
    y = _to_2d_target(target).to(torch.float32)
    if x.shape[0] != y.shape[0]:
        raise ValueError("source and target must have the same number of rows")

    x_mean = x.mean(dim=0, keepdim=True)
    y_mean = y.mean(dim=0, keepdim=True)
    xc = x - x_mean
    yc = y - y_mean

    n_features = xc.shape[1]
    
    # Use lstsq for better numerical stability with potentially singular matrices
    # xtx + reg can still be unstable if l2 is too small or data scale is huge.
    # lstsq handles the pseudo-inverse internally.
    try:
        # We use the regularized normal equations approach but with safety
        xtx = xc.T @ xc
        reg = l2 * torch.eye(n_features, dtype=xc.dtype)
        # Add a tiny bit more to the diagonal if it's really bad, or use lstsq
        # But for Ridge, solving (X'X + L*I)B = X'Y is standard.
        # If solve fails, we fallback to lstsq on the augmented problem or just lstsq.
        beta = torch.linalg.solve(xtx + reg, xc.T @ yc)
    except Exception:
        # Fallback to least squares solver which is more robust to singularity
        # We solve the original problem with a tiny bit of regularization implicit in lstsq
        # or we can explicitly augment for Ridge.
        # For simplicity and robustness:
        beta = torch.linalg.lstsq(xc, yc, rcond=1e-6).solution
        
    intercept = y_mean - x_mean @ beta

    y_hat = x @ beta + intercept
    ss_res = ((y - y_hat) ** 2).sum(dim=0)
    ss_tot = ((y - y_mean) ** 2).sum(dim=0)
    r2 = 1.0 - ss_res / torch.clamp(ss_tot, min=1e-8)
    global_r2 = 1.0 - ss_res.sum() / torch.clamp(ss_tot.sum(), min=1e-8)

    return {
        "coefficients": beta,
        "intercept": intercept.squeeze(0),
        "prediction": y_hat,
        "r2_per_target": torch.nan_to_num(r2, nan=0.0),
        "global_r2": float(torch.nan_to_num(global_r2, nan=0.0).item()),
    }


def _column_correlation_matrix(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    a = _sanitize_tensor(a).to(torch.float32)
    b = _sanitize_tensor(b).to(torch.float32)
    if a.shape[0] != b.shape[0]:
        raise ValueError("a and b must have the same number of rows")

    ac = a - a.mean(dim=0, keepdim=True)
    bc = b - b.mean(dim=0, keepdim=True)
    a_scale = torch.sqrt(torch.clamp((ac ** 2).sum(dim=0, keepdim=True), min=eps))
    b_scale = torch.sqrt(torch.clamp((bc ** 2).sum(dim=0, keepdim=True), min=eps))
    return (ac.T @ bc) / (a_scale.T @ b_scale)


def summarize_basis_matrix(
    v: torch.Tensor,
    *,
    feature_names: Optional[Sequence[str]] = None,
    top_k: int = 10,
) -> Dict[str, Any]:
    """Summarize an interpretable first-layer basis matrix."""
    v_cpu = _sanitize_tensor(v)
    n_features, n_components = v_cpu.shape
    abs_v = torch.abs(v_cpu)
    support = abs_v > 0
    l0_counts = support.sum(dim=0)
    density = l0_counts.to(torch.float32) / max(1, n_features)
    l1_norm = abs_v.sum(dim=0)
    l2_norm = torch.linalg.norm(v_cpu, dim=0)
    orthogonality_defect = float(invariant_orthogonality_defect(v_cpu).item())

    resolved_names = list(feature_names) if feature_names is not None else [f"feature_{i}" for i in range(n_features)]
    if len(resolved_names) != n_features:
        raise ValueError("feature_names must match the number of rows in v")

    top_features = []
    top_k = max(1, min(top_k, n_features))
    for component_idx in range(n_components):
        values = abs_v[:, component_idx]
        indices = torch.topk(values, k=top_k).indices.tolist()
        top_features.append([
            {
                "feature_index": int(idx),
                "feature_name": resolved_names[idx],
                "loading": float(v_cpu[idx, component_idx].item()),
                "abs_loading": float(values[idx].item()),
            }
            for idx in indices
        ])

    return {
        "n_features": int(n_features),
        "n_components": int(n_components),
        "orthogonality_defect": orthogonality_defect,
        "component_l0": [int(x) for x in l0_counts.tolist()],
        "component_density": [float(x) for x in density.tolist()],
        "component_l1": [float(x) for x in l1_norm.tolist()],
        "component_l2": [float(x) for x in l2_norm.tolist()],
        "top_features": top_features,
    }


def build_first_layer_contract(
    v_list: Sequence[torch.Tensor],
    score_list: Sequence[torch.Tensor],
    *,
    feature_names: Optional[Sequence[Optional[Sequence[str]]]] = None,
    top_k: int = 10,
) -> Dict[str, Any]:
    """Create a serializable first-layer interpretability contract."""
    if len(v_list) != len(score_list):
        raise ValueError("v_list and score_list must have the same length")

    if feature_names is None:
        feature_names = [None] * len(v_list)
    if len(feature_names) != len(v_list):
        raise ValueError("feature_names must align with the number of modalities")

    v_cpu = [_sanitize_tensor(v) for v in v_list]
    scores_cpu = [_sanitize_tensor(scores) for scores in score_list]
    summaries = [
        summarize_basis_matrix(v, feature_names=names, top_k=top_k)
        for v, names in zip(v_cpu, feature_names)
    ]
    return {
        "v": v_cpu,
        "scores": scores_cpu,
        "orthogonality_defect": [float(s["orthogonality_defect"]) for s in summaries],
        "sparsity_summary": summaries,
    }


def extract_first_layer_factors(
    model_res: Dict[str, Any],
    feature_names: Optional[Sequence[Optional[Sequence[str]]]] = None,
    top_k: int = 10,
) -> Dict[str, Any]:
    """Return the canonical first-layer contract from a fitted deep result."""
    first_layer = model_res.get("first_layer")
    if first_layer is None:
        v_list = model_res.get("v")
        score_list = model_res.get("first_layer_scores") or model_res.get("latents")
        if v_list is None or score_list is None:
            raise KeyError("model_res does not contain first-layer factors")
        first_layer = build_first_layer_contract(v_list, score_list, feature_names=feature_names, top_k=top_k)
    elif feature_names is not None:
        first_layer = build_first_layer_contract(
            first_layer["v"],
            first_layer["scores"],
            feature_names=feature_names,
            top_k=top_k,
        )
    return first_layer


def _feature_importance_from_basis(
    v: torch.Tensor,
    component_importance: torch.Tensor,
) -> torch.Tensor:
    v_cpu = _sanitize_tensor(v).to(torch.float32)
    comp = _sanitize_tensor(component_importance).reshape(-1).to(torch.float32)
    if v_cpu.shape[1] != comp.shape[0]:
        raise ValueError("component_importance must match the number of basis columns")
    return torch.abs(v_cpu) @ comp


def analyze_first_layer_alignment(
    model_res: Dict[str, Any],
    *,
    l2: float = 1e-6,
) -> Dict[str, Any]:
    """Quantify how post-first-layer deep latents relate to the first basis scores."""
    first_layer = extract_first_layer_factors(model_res)
    z0_list = first_layer["scores"]
    z1_list = model_res.get("latents")
    if z1_list is None:
        raise KeyError("model_res does not contain deep latents")
    if len(z0_list) != len(z1_list):
        raise ValueError("first-layer scores and deep latents must have the same number of modalities")

    per_modality: List[Dict[str, Any]] = []
    global_r2_values: List[float] = []
    for modality_index, (z0, z1, v) in enumerate(zip(z0_list, z1_list, first_layer["v"])):
        fit = _fit_linear_map(z0, z1, l2=l2)
        component_importance = torch.mean(torch.abs(fit["coefficients"]), dim=1)
        feature_importance = _feature_importance_from_basis(v, component_importance)
        corr = _column_correlation_matrix(z0, z1)
        per_modality.append({
            "modality_index": int(modality_index),
            "global_r2": float(fit["global_r2"]),
            "r2_per_target": fit["r2_per_target"],
            "coefficients": fit["coefficients"],
            "component_correlation": corr,
            "component_importance": component_importance,
            "feature_importance": feature_importance,
        })
        global_r2_values.append(float(fit["global_r2"]))

    return {
        "modalities": per_modality,
        "mean_global_r2": float(sum(global_r2_values) / max(1, len(global_r2_values))),
    }


def attribute_shared_to_first_layer(
    model_res: Dict[str, Any],
    *,
    l2: float = 1e-6,
) -> Dict[str, Any]:
    """Attribute shared latent variation back to first-layer components and features."""
    first_layer = extract_first_layer_factors(model_res)
    z0_list = first_layer["scores"]
    v_list = first_layer["v"]
    u = model_res.get("u")
    if u is None:
        raise KeyError("model_res does not contain shared latent u")

    per_modality: List[Dict[str, Any]] = []
    global_r2_values: List[float] = []
    for modality_index, (z0, v) in enumerate(zip(z0_list, v_list)):
        fit = _fit_linear_map(z0, u, l2=l2)
        component_importance = torch.mean(torch.abs(fit["coefficients"]), dim=1)
        feature_importance = _feature_importance_from_basis(v, component_importance)
        per_modality.append({
            "modality_index": int(modality_index),
            "global_r2": float(fit["global_r2"]),
            "r2_per_shared_dimension": fit["r2_per_target"],
            "coefficients": fit["coefficients"],
            "component_importance": component_importance,
            "feature_importance": feature_importance,
        })
        global_r2_values.append(float(fit["global_r2"]))

    z0_concat = torch.cat([_sanitize_tensor(z) for z in z0_list], dim=1)
    combined_fit = _fit_linear_map(z0_concat, u, l2=l2)
    combined_component_importance = torch.mean(torch.abs(combined_fit["coefficients"]), dim=1)

    component_slices = []
    combined_feature_importance = []
    start = 0
    for modality_index, v in enumerate(v_list):
        width = v.shape[1]
        stop = start + width
        component_slice = combined_component_importance[start:stop]
        component_slices.append({
            "modality_index": int(modality_index),
            "start": int(start),
            "stop": int(stop),
            "component_importance": component_slice,
        })
        combined_feature_importance.append(_feature_importance_from_basis(v, component_slice))
        start = stop

    return {
        "per_modality": per_modality,
        "mean_global_r2": float(sum(global_r2_values) / max(1, len(global_r2_values))),
        "combined": {
            "global_r2": float(combined_fit["global_r2"]),
            "r2_per_shared_dimension": combined_fit["r2_per_target"],
            "coefficients": combined_fit["coefficients"],
            "component_importance": combined_component_importance,
            "component_slices": component_slices,
            "feature_importance": combined_feature_importance,
        },
    }


def attribute_prediction_to_features(
    model_res: Dict[str, Any],
    target: torch.Tensor,
    *,
    l2: float = 1e-6,
) -> Dict[str, Any]:
    """Attribute an external prediction target back to first-layer components and features."""
    first_layer = extract_first_layer_factors(model_res)
    z0_list = first_layer["scores"]
    v_list = first_layer["v"]
    y = _to_2d_target(target)

    per_modality: List[Dict[str, Any]] = []
    for modality_index, (z0, v) in enumerate(zip(z0_list, v_list)):
        fit = _fit_linear_map(z0, y, l2=l2)
        component_importance = torch.mean(torch.abs(fit["coefficients"]), dim=1)
        feature_importance = _feature_importance_from_basis(v, component_importance)
        per_modality.append({
            "modality_index": int(modality_index),
            "global_r2": float(fit["global_r2"]),
            "r2_per_target": fit["r2_per_target"],
            "coefficients": fit["coefficients"],
            "component_importance": component_importance,
            "feature_importance": feature_importance,
        })

    z0_concat = torch.cat([_sanitize_tensor(z) for z in z0_list], dim=1)
    combined_fit = _fit_linear_map(z0_concat, y, l2=l2)
    combined_component_importance = torch.mean(torch.abs(combined_fit["coefficients"]), dim=1)

    combined_feature_importance = []
    component_slices = []
    start = 0
    for modality_index, v in enumerate(v_list):
        width = v.shape[1]
        stop = start + width
        component_slice = combined_component_importance[start:stop]
        combined_feature_importance.append(_feature_importance_from_basis(v, component_slice))
        component_slices.append({
            "modality_index": int(modality_index),
            "start": int(start),
            "stop": int(stop),
            "component_importance": component_slice,
        })
        start = stop

    u_fit = _fit_linear_map(model_res["u"], y, l2=l2) if "u" in model_res else None

    return {
        "per_modality": per_modality,
        "combined": {
            "global_r2": float(combined_fit["global_r2"]),
            "r2_per_target": combined_fit["r2_per_target"],
            "coefficients": combined_fit["coefficients"],
            "component_importance": combined_component_importance,
            "component_slices": component_slices,
            "feature_importance": combined_feature_importance,
        },
        "shared_latent_baseline": None if u_fit is None else {
            "global_r2": float(u_fit["global_r2"]),
            "r2_per_target": u_fit["r2_per_target"],
            "coefficients": u_fit["coefficients"],
        },
    }


def build_interpretability_report(
    model_res: Dict[str, Any],
    *,
    l2: float = 1e-6,
) -> Dict[str, Any]:
    """Build the default interpretability payload."""
    return {
        "shared_to_first_layer": attribute_shared_to_first_layer(model_res, l2=l2),
        "deep_layer_alignment": analyze_first_layer_alignment(model_res, l2=l2),
    }
