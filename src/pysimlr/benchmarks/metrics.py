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
    r"""
    Score a latent representation with a linear model.

    .. warning::

       This measures the **subspace**, not the basis. For any invertible ``M``,
       ``span(XVM) = span(XV)``, so the fitted values of ordinary least
       squares, ridge and logistic regression are unchanged when ``V`` is
       replaced by ``VM`` -- the coefficients absorb ``M^{-1}``. The score is
       therefore *exactly* invariant to reparametrising the basis and cannot
       distinguish two bases that span the same subspace.

       That matters because non-negativity and sparsity are properties of the
       *axes*. A tie here is not evidence about a basis. Measured on ADNI
       cortical thickness, replacing ``V`` by ``VM`` for random invertible
       ``M`` moved cross-validated linear R-squared by 0.0000 while moving
       random-forest R-squared by 0.057.

       Use this where the subspace is the question, and
       :func:`axis_sensitive_cross_val_metrics` where the axes are. In pysimlr's
       own clinical benchmark the five models sit within 0.002 of each other on
       this metric (Friedman p = 0.61), which is the tie the identity predicts
       rather than a finding about their bases.

    Returns
    -------
    Dict[str, float]
        ``train``, ``test`` and ``gap`` (train minus test).
    """
    u_train_np = u_train.detach().cpu().numpy()
    u_test_np = u_test.detach().cpu().numpy()
    if is_classification:
        from sklearn.linear_model import LogisticRegression
        y_train_int = y_train.astype(int).ravel()
        y_test_int = y_test.astype(int).ravel()
        from sklearn.metrics import balanced_accuracy_score
        model = LogisticRegression(max_iter=1000).fit(u_train_np, y_train_int)
        train_perf = float(model.score(u_train_np, y_train_int))
        test_perf = float(model.score(u_test_np, y_test_int))
        # Plain accuracy is uninformative on an imbalanced cohort: on PPMI
        # (202 vs 819, majority rate 0.802) every model scored exactly 0.800,
        # i.e. every one of them predicted the majority class and the table
        # read as a nine-way tie rather than as nine failures. Balanced
        # accuracy is chance-corrected at 1/n_classes whatever the prior.
        extra = {
            "balanced_train": float(balanced_accuracy_score(
                y_train_int, model.predict(u_train_np))),
            "balanced_test": float(balanced_accuracy_score(
                y_test_int, model.predict(u_test_np))),
            "majority_rate": float(np.bincount(y_test_int).max() / len(y_test_int)),
        }
    else:
        from sklearn.linear_model import LinearRegression
        model = LinearRegression().fit(u_train_np, y_train)
        train_perf = float(model.score(u_train_np, y_train))
        test_perf = float(model.score(u_test_np, y_test))
    out = {"train": train_perf, "test": test_perf, "gap": train_perf - test_perf}
    if is_classification:
        out.update(extra)
    return out

def axis_sensitive_cross_val_metrics(
    u_train: torch.Tensor, y_train: np.ndarray,
    u_test: torch.Tensor, y_test: np.ndarray,
    is_classification: bool = False,
    n_estimators: int = 200,
    random_state: int = 0,
) -> Dict[str, float]:
    """
    Score a latent representation with a model that can see the axes.

    A tree splits on single coordinates, so its fit depends on how the
    subspace is parametrised and not only on which subspace it is. That makes
    it the companion to :func:`cross_val_metrics`, which is exactly invariant
    to reparametrisation and so cannot speak to a basis at all.

    Sensitivity is not preference: which parametrisation a forest favours is
    data dependent, and the point of reporting both is that a difference here
    with no difference there localises the effect to the axes.

    Parameters
    ----------
    u_train, u_test : torch.Tensor
        Latent scores for the training and test split.
    y_train, y_test : np.ndarray
        Outcomes.
    is_classification : bool, default=False
        Select a classifier rather than a regressor.
    n_estimators : int, default=200
        Trees in the forest.
    random_state : int, default=0
        Fixed so the metric is reproducible; the forest is an instrument here,
        not a model being tuned.

    Returns
    -------
    Dict[str, float]
        ``train``, ``test`` and ``gap`` (train minus test).
    """
    u_train_np = u_train.detach().cpu().numpy()
    u_test_np = u_test.detach().cpu().numpy()
    if is_classification:
        from sklearn.ensemble import RandomForestClassifier
        y_train_int = y_train.astype(int).ravel()
        y_test_int = y_test.astype(int).ravel()
        model = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state,
        ).fit(u_train_np, y_train_int)
        train_perf = float(model.score(u_train_np, y_train_int))
        test_perf = float(model.score(u_test_np, y_test_int))
    else:
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(
            n_estimators=n_estimators, random_state=random_state,
        ).fit(u_train_np, np.asarray(y_train).ravel())
        train_perf = float(model.score(u_train_np, np.asarray(y_train).ravel()))
        test_perf = float(model.score(u_test_np, np.asarray(y_test).ravel()))
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
    from ..utils import orthogonality_defect
    return float(np.mean([orthogonality_defect(v).item() for v in v_mats]))

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
            # Prefer the caller's declaration over guessing. The heuristic
            # below reads `len(unique) < 10`, which silently classes a
            # *10-class* problem as regression: on UCI mfeat (10 balanced digit
            # classes) every accuracy metric came back missing and the whole
            # cohort reported NaN. A dataset that knows what it is should say
            # so rather than be inferred from its own label cardinality.
            declared = kwargs.get("is_classification")
            unique_y = np.unique(y_train_np)
            if declared is None:
                is_classification = (len(unique_y) < 10
                                     and np.all(y_train_np % 1 == 0))
            else:
                is_classification = bool(declared)
            deep_res = cross_val_metrics(u_train, y_train_np, u_pred, y_true_np, is_classification)
            metrics["test_r2"], metrics["train_r2"], metrics["gen_gap"] = deep_res["test"], deep_res["train"], deep_res["gap"]
            for _k in ("balanced_train", "balanced_test", "majority_rate"):
                if _k in deep_res:
                    metrics[_k] = deep_res[_k]
            if is_classification: metrics["test_accuracy"], metrics["train_accuracy"] = deep_res["test"], deep_res["train"]
            fl_scores_train, fl_scores_test = kwargs.get("first_layer_scores_train"), kwargs.get("first_layer_scores_test")
            if fl_scores_train is not None and fl_scores_test is not None:
                u_lin_train, u_lin_test = torch.cat(fl_scores_train, dim=1), torch.cat(fl_scores_test, dim=1)
                u_lin_train = (u_lin_train - u_lin_train.mean(0)) / (u_lin_train.std(0) + 1e-6)
                u_lin_test = (u_lin_test - u_lin_test.mean(0)) / (u_lin_test.std(0) + 1e-6)
                lin_res = cross_val_metrics(u_lin_train, y_train_np, u_lin_test, y_true_np, is_classification)
                metrics["first_layer_test_r2"], metrics["first_layer_train_r2"], metrics["first_layer_gen_gap"] = lin_res["test"], lin_res["train"], lin_res["gap"]
                if is_classification: metrics["first_layer_test_accuracy"], metrics["first_layer_train_accuracy"] = lin_res["test"], lin_res["train"]
                # The linear score above is invariant to reparametrising the
                # basis, so it says nothing about the axes that non-negativity
                # and sparsity actually change. Report an axis-sensitive
                # companion beside it; a gap between the two localises an
                # effect to the parametrisation.
                axis_res = axis_sensitive_cross_val_metrics(u_lin_train, y_train_np, u_lin_test, y_true_np, is_classification)
                metrics["first_layer_axis_test_r2"], metrics["first_layer_axis_train_r2"], metrics["first_layer_axis_gen_gap"] = axis_res["test"], axis_res["train"], axis_res["gap"]
                if is_classification: metrics["first_layer_axis_test_accuracy"], metrics["first_layer_axis_train_accuracy"] = axis_res["test"], axis_res["train"]
        else:
            # Fallback for simple R2 if u_train is missing but y_true is present
            metrics["test_r2"] = outcome_r2_score(u_pred, y_true)
            
    return metrics


def support_recovery_score(v_est: List[torch.Tensor],
                           v_true: List[torch.Tensor],
                           tol: float = 1e-6,
                           max_truth_density: float = 0.9) -> float:
    r"""
    Agreement between the *supports* of an estimated and a true basis.

    Every other basis metric in this module is blind to the axes.
    ``Feature_Recovery_V`` uses :func:`~pysimlr.utils.procrustes_r2`, which
    aligns ``v_est`` to ``v_true`` including rotation, so it scores
    :math:`\mathrm{span}(V)` and is unchanged by any orthogonal
    reparametrisation -- the same invariance that makes
    :func:`cross_val_metrics` unable to speak about a basis. Sparsity and
    disjoint support are properties of *which features load on which
    component*, which no rotation-invariant score can see.

    This measures that directly: columns of ``v_est`` are matched one-to-one
    to columns of ``v_true`` by the assignment maximising support overlap
    (columns are exchangeable, so a permutation must not be penalised), and
    the score is the mean Jaccard index of the matched supports.

    Parameters
    ----------
    v_est : List[torch.Tensor]
        Estimated per-view bases, each ``(features, k)``.
    v_true : List[torch.Tensor]
        Ground-truth per-view bases, same shapes. Views whose truth is
        identically zero (real datasets, where no basis is known) are skipped.
    tol : float, default=1e-6
        Magnitude below which an entry counts as zero.
    max_truth_density : float, default=0.9
        Views whose true basis is denser than this are skipped: a fully dense
        truth has no support structure, so "did you recover the support" is
        not a question about the estimate. Without this guard the score is
        maximised by guessing dense -- PCA scored exactly 1.000 on every
        ``torch.randn`` regime in the suite and thereby *led* the support
        ranking, while scoring 0.333 on the only regimes where support is
        defined.

    Returns
    -------
    float
        Mean Jaccard index in ``[0, 1]``, or ``nan`` when no view has a truth
        that is both known and sparse. ``nan`` rather than ``0.0`` so that a
        regime where the question is ill-posed drops out of an average
        instead of counting as a total failure to recover.
        A dense estimate against a sparse truth scores the density of the
        truth, not 1.0.
    """
    from scipy.optimize import linear_sum_assignment

    if not v_est or not v_true:
        return float("nan")

    scores = []
    for ve, vt in zip(v_est, v_true):
        if vt is None:
            continue
        vt_t = torch.as_tensor(vt).float()
        ve_t = torch.as_tensor(ve).float()
        if bool((vt_t == 0).all()) or vt_t.shape != ve_t.shape:
            continue

        s_est = (ve_t.abs() > tol).numpy()
        s_true = (vt_t.abs() > tol).numpy()
        if s_true.mean() > max_truth_density:
            continue
        k = s_true.shape[1]

        # Jaccard between every estimated column and every true column.
        jac = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                inter = np.logical_and(s_est[:, i], s_true[:, j]).sum()
                union = np.logical_or(s_est[:, i], s_true[:, j]).sum()
                jac[i, j] = inter / union if union else 0.0

        rows, cols = linear_sum_assignment(-jac)
        scores.append(float(jac[rows, cols].mean()))

    return float(np.mean(scores)) if scores else float("nan")
