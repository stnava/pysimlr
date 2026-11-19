"""Bagged estimators for any registered method: ``bagged_fit`` and ``bagged_predict``.

`pysimlr.loading_inference` refits a method on row subsamples and summarises the refits
(feature selection frequency, calibrated out-of-bag latent stability, out-of-bag scores).
`bagged_fit` turns the same refits into an estimator with two modes:

* ``aggregate="select"`` (sparse): keep the features with ``selection_frequency >= threshold``
  and refit the method once on them.
* ``aggregate="average"`` (smooth ensemble): keep the subsample fits; scores are the mean of their aligned
  scores.

Subsampling is without replacement (as in `loading_inference`): a duplicated row appears twice in a refit, which
inflates the agreement between views that the model is fitted to maximise.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

AGGREGATES = ("select", "average")

BETWEEN_RUN_NOTE = (
    "Agreement between independent runs: call bagged_fit twice with the same data and random_state but different "
    "draw_random_state (the row draws), then correlate the two runs' scores (result['scores'], or "
    "bagged_predict on a common set of rows) per component; |r| near 1 means the estimator does not depend on "
    "which subsamples were drawn. Components are matched up to sign (and rotation when k > 1)."
)


def _flat(v: Any) -> Any:
    """A JSON-friendly summary of an argument for ``provenance``."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)) and all(x is None or isinstance(x, (bool, int, float, str)) for x in v):
        return list(v)
    if isinstance(v, np.ndarray) and v.ndim == 1 and v.size <= 1000:
        return v.tolist()
    return f"<{type(v).__name__}>"


# ----------------------------------------------------------------------------- feature masks
def mask_views(result: Dict[str, Any], views: Any) -> Any:
    """Subset the columns of ``views`` by ``result["feature_mask"]`` (a no-op when the result has none).

    A view with all the original features (``len(mask)`` columns) is subset to the kept columns; a view that
    already has only the kept columns (``mask.sum()``) is passed through; any other width raises a
    ``ValueError`` naming the view. Accepts what `pysimlr.predict` accepts: a list (``None`` for a missing
    view), a dict by view name, or one matrix. Entries beyond the masked views (views added by
    `pysimlr.extend`) are left alone.
    """
    masks = result.get("feature_mask")
    if masks is None:
        return views
    from .api import as_views
    from .extension import view_names as _names

    def one(x: Any, m: int) -> Any:
        if x is None or m >= len(masks):
            return x
        t = as_views([x])[0]
        mask = np.asarray(masks[m], dtype=bool)
        if t.shape[1] == mask.size:
            return t[:, torch.as_tensor(mask)]
        if t.shape[1] == int(mask.sum()):
            return t
        raise ValueError(
            f"view {m} has {t.shape[1]} columns; this bagged select-mode fit expects either all {mask.size} "
            f"original features (subset by result['feature_mask']) or the {int(mask.sum())} kept features.")

    if isinstance(views, Mapping):
        names = list(_names(result))
        return {k_: one(x, names.index(k_) if k_ in names else len(masks)) for k_, x in views.items()}
    if isinstance(views, (list, tuple)):
        return [one(x, m) for m, x in enumerate(views)]
    return [one(views, 0)]


# ----------------------------------------------------------------------------- ensemble
def _is_joint_fit(model: Dict[str, Any]) -> bool:
    return bool(model.get("basis_is_joint"))


def _member_scores(model: Dict[str, Any], rot: np.ndarray, view_list: List[torch.Tensor]):
    """One member's aligned consensus scores ``(n, k)`` and per-view scores (list of ``(n, k)``)."""
    from .api import predict
    from .sem import _block_scores
    pred = predict(model, view_list)
    u = pred["u"].detach().double().numpy() @ rot
    if _is_joint_fit(model):
        lat = [_block_scores(model, x, m, rot) for m, x in enumerate(view_list)]
    else:
        lat = [z.detach().double().numpy() @ rot for z in pred["latents"]]
    return u, lat


def _ensemble_scores(models: Sequence[Dict[str, Any]], rots: np.ndarray, view_list: List[torch.Tensor]):
    """Mean over members of the aligned consensus scores and of the aligned per-view scores."""
    u_sum, lat_sum = None, None
    for model, rot in zip(models, rots):
        u, lat = _member_scores(model, np.asarray(rot, dtype=float), view_list)
        u_sum = u if u_sum is None else u_sum + u
        lat_sum = lat if lat_sum is None else [a + b for a, b in zip(lat_sum, lat)]
    b = len(models)
    return u_sum / b, [a / b for a in lat_sum]


def bagged_predict(result: Dict[str, Any], views: Any) -> Dict[str, Any]:
    """Predict with a `bagged_fit` result (either mode), on views with all the original features.

    Select mode: the result is an ordinary fit on the kept features; this is `pysimlr.predict` (which subsets the
    columns by ``result["feature_mask"]``). Average mode: the mean over the subsample fits of each fit's
    `pysimlr.predict` scores rotated into the reference frame, returned as ``u`` (``(n, k)`` tensor, the same frame
    as ``result["scores"]``) and ``latents`` (per-view mean aligned scores; no per-view sign correction and no
    standardisation), plus ``n_fits``. `pysimlr.predict` calls this function for an average-mode result; it returns
    no reconstructions, ``r2`` or ``errors`` (a mean over fits has no single decoder). A missing view
    (``None`` / a dict of a subset of the views) is not supported in average mode.
    """
    from .api import predict
    if "aggregate" not in result:
        raise ValueError("bagged_predict needs a result from pysimlr.bagged_fit (it carries 'aggregate').")
    if result["aggregate"] == "select":
        return predict(result, views)
    from .api import as_views
    if isinstance(views, (list, tuple)) and any(v is None for v in views):
        raise NotImplementedError("average-mode bagged_predict needs every view; a missing view is not supported.")
    view_list = as_views(views)
    if len(view_list) != len(result["v"]):
        raise ValueError(f"views has {len(view_list)} entries; the fit has {len(result['v'])}.")
    u, lat = _ensemble_scores(result["ensemble_models"], result["ensemble_rotations"], view_list)
    u_t = torch.as_tensor(u).float()
    return {"u": u_t, "latents": [torch.as_tensor(z).float() for z in lat], "n_fits": len(result["ensemble_models"])}


# ----------------------------------------------------------------------------- averaged weights
def _affine_preprocessing(model: Dict[str, Any], m: int, p: int):
    """``(scale, offset)`` with ``preprocess(x) = x * scale + offset`` per feature, checked on random rows.

    Returns ``None`` if the fit's preprocessing for view ``m`` is not a per-feature affine map (for example a
    confound projection)."""
    from .utils import preprocess_data
    scale_list = list(model.get("scale_list") or ["none"])
    provs = model.get("provenance_list") or []
    if len(provs) != len(model["v"]) or model.get("confound_sink") is not None or model.get("missing_rows") == "mask":
        return None
    pv = provs[m]

    def pre(x):
        return preprocess_data(x.float(), scale_list, provenance=pv).double().numpy()
    a = pre(torch.zeros(1, p))[0]
    d = pre(torch.ones(1, p))[0] - a
    rng = np.random.default_rng(0)
    x = torch.as_tensor(rng.standard_normal((6, p)))
    tol = 1e-4 * (1.0 + np.abs(a) + np.abs(d))
    if not np.all(np.abs(pre(x) - (x.double().numpy() * d + a)) <= tol * 10.0):
        return None
    return d, a


def _averaged_weights(models: Sequence[Dict[str, Any]], rots: np.ndarray):
    """Mean aligned encoder weights, or ``(None, None, None, reason)`` when the encoder is not linear.

    Member ``b`` scores view ``m`` as ``pre_b(X_m) @ E_bm`` with ``E_bm = V_bm @ rot_b / s_bm`` (``s_bm`` is the
    fit's ``latent_scale`` for per-view encoders, 1 for a joint basis). Returns ``averaged_weights[m]`` (the mean of
    ``E_bm``, in each member's own standardised feature space), ``weights_raw[m]`` and ``intercepts[m]`` (the
    same mean with each member's affine preprocessing folded in, so that ``X_m @ weights_raw[m] + intercepts[m]`` is
    exactly the mean aligned per-view score of raw ``X_m``), and the reason when unavailable.
    """
    first = models[0]
    if first.get("family") not in ("linear", "baseline"):
        return None, None, None, (f"{first.get('method')} is a {first.get('family')} model: its encoder is not "
                                  "linear in the features, so only scores are averaged")
    n_views = len(first["v"])
    e_sum = [0.0] * n_views
    w_sum = [0.0] * n_views
    c_sum = [0.0] * n_views
    for model, rot in zip(models, rots):
        rot = np.asarray(rot, dtype=float)
        joint = _is_joint_fit(model)
        scales = [1.0] * n_views if joint else (model.get("latent_scale") or [1.0] * n_views)
        for m in range(n_views):
            v = model["v"][m].detach().double().numpy()
            e = v @ rot / float(scales[m])
            aff = _affine_preprocessing(model, m, v.shape[0])
            if aff is None:
                return None, None, None, ("the preprocessing is not a per-feature affine map (confounds, missing_rows "
                                          "or a non-affine scale), so only scores are averaged")
            e_sum[m] = e_sum[m] + e
            w_sum[m] = w_sum[m] + aff[0][:, None] * e
            c_sum[m] = c_sum[m] + aff[1] @ e
    b = len(models)
    return [e / b for e in e_sum], [w / b for w in w_sum], [c / b for c in c_sum], None


def _effective_features(w: np.ndarray) -> float:
    """``(sum |w|)^2 / sum(w^2)`` of the first component's weights: ``p`` for equal weights, 1 for one feature."""
    a = np.abs(np.asarray(w, dtype=float)[:, 0])
    den = float((a ** 2).sum())
    return float(a.sum() ** 2 / den) if den > 0 else float("nan")


# ----------------------------------------------------------------------------- bagged_fit
def bagged_fit(method: str, views: Any, k: int, *, aggregate: str = "select", n_subsamples: int = 50,
               fraction: float = 0.8, threshold: float = 0.6, strata: Any = None, random_state: int = 0,
               draw_random_state: Optional[int] = None, n_jobs: int = 1, groups: Any = None,
               expected_false: Optional[float] = None, n_permutations: int = 0,
               baseline: bool = False, n_baseline: int = 3,
               tune_learning_rate: str = "per_refit", **params: Any) -> Dict[str, Any]:
    r"""Bagged estimator for any registered method (`pysimlr.method_names()`), in one of two modes.
    The default is ``aggregate="select"``; ``"average"`` is the opt-in alternative and any other value raises
    a ``ValueError`` listing the two valid modes.

    ``n_subsamples`` refits on random ``fraction`` of the rows are drawn WITHOUT replacement (as in
    `pysimlr.loading_inference`: duplicated rows would inflate cross-view agreement) and the method is refitted on
    each (``random_state`` seeds every fit, ``draw_random_state`` the row draws; default ``random_state``).

    ``aggregate="select"`` (sparse). The per-view ``selection_frequency`` of `loading_inference` (each refit selects
    the ``q_m`` features of largest communality; this is a feature-selection frequency, not a stability number) is
    computed; features with frequency ``>= threshold`` are kept (``feature_mask``), and ``method`` is fitted ONCE
    on the kept features of each view with the same ``params``. A view that keeps no feature raises a ``ValueError``
    naming the view, its maximum frequency and the threshold (no fallback). The returned dict is that final
    `pysimlr.fit` result plus the keys below. Prediction on full-width views: `pysimlr.predict` and
    `pysimlr.sem.view_scores` subset the columns by ``result["feature_mask"]``; views that already have only the
    kept columns are also accepted; any other width raises a ``ValueError``.

    ``aggregate="average"`` (smooth ensemble). The subsample fits are kept (``result["ensemble_models"]``). The score
    of any rows is the mean over fits of ``predict(fit_b, rows)["u"] @ R_b``, where ``R_b`` is the fit's alignment to
    the reference fit. `pysimlr.predict` (or `bagged_predict`) returns this ensemble prediction for new rows.

    Returns (both modes), in addition to the fit result: ``aggregate``, ``method``, ``k``, ``n_subsamples``,
    ``fraction``, ``threshold``, ``selection_frequency`` (per view, array; not a stability metric),
    ``feature_mask`` (per view boolean array; all True in average mode), ``effective_features`` (per view),
    ``scores`` (``(n, k)`` training-row scores), ``oob_scores`` (``(n, k)``), ``latent_reliability`` (+
    ``_baseline``, ``_gain`` when ``baseline=True``), ``latent_reliability_averaged``,
    ``latent_reliability_multivariate``, ``cross_view_coupling``, ``feature_agreement``,
    ``feature_agreement_by_view``, ``n_oob_per_row``, ``n_refits_used``, ``alignment``, ``interpretation``,
    ``oos_r2``, ``diagnostics_scope`` (``"pre_selection_subsamples"`` in select mode, ``"ensemble_subsamples"`` in
    average mode), ``refit_learning_rates``, ``converged_fraction``, ``expected_false_selections``,
    ``expected_false_selections_reason``, ``selection_q``, ``between_run_note`` and ``provenance``. Under
    ``aggregate="select"``, all returned stability diagnostics describe the pre-selection all-feature subsample
    refits rather than nested select-then-fit reliability of ``result["v"]`` (stated in ``interpretation`` and
    ``provenance``). Results are wrapped so reading keys removed in 0.41.0 raises an informative ``KeyError``.
    """
    from .api import as_views, fit, resolve_method
    from .latent_stability import _RemovedKeysDict, _build_stability_interpretation
    from .loadings import _procrustes, _validate_tune_learning_rate, loading_inference
    import pysimlr
    if aggregate not in AGGREGATES:
        raise ValueError(f"aggregate must be one of {AGGREGATES}; got {aggregate!r}.")
    resolve_method(method)
    _validate_tune_learning_rate(tune_learning_rate, method)
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be in (0, 1); got {fraction}.")
    if not 0.5 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0.5, 1]; got {threshold}.")
    if baseline and int(n_baseline) < 1:
        raise ValueError(f"n_baseline must be >= 1 when baseline=True; got {n_baseline}.")
    view_list = as_views(views)
    n_views = len(view_list)
    n = int(view_list[0].shape[0])
    widths = [int(x.shape[1]) for x in view_list]
    names = list(views.keys()) if isinstance(views, Mapping) else [f"view{i}" for i in range(n_views)]
    if expected_false is None:
        expected_false = float(max(1.0, np.median(widths) / (16.0 * (2.0 * threshold - 1.0))))
    stability_scope = (
        "pre_selection_all_feature_refits (not nested select-then-fit reliability of result['v'])"
        if aggregate == "select" else "ensemble_subsamples"
    )
    provenance = {
        "method": method, "k": int(k), "aggregate": aggregate, "n_subsamples": int(n_subsamples),
        "fraction": float(fraction), "threshold": float(threshold), "strata": _flat(strata),
        "random_state": random_state, "draw_random_state": draw_random_state, "n_jobs": n_jobs,
        "groups": _flat(groups), "expected_false": float(expected_false),
        "n_permutations": int(n_permutations), "baseline": bool(baseline),
        "n_baseline": int(n_baseline) if (baseline and n_views >= 2) else 0,
        "tune_learning_rate": tune_learning_rate,
        "stability_scope": stability_scope,
        "params": {kk: _flat(vv) for kk, vv in params.items()}, "n_rows": n, "view_widths": widths,
        "subsample": "without replacement", "pysimlr_version": pysimlr.__version__,
    }
    want = ("latents",) if aggregate == "select" else ("latents", "models")
    inf = loading_inference(
        method, views, k, n_subsamples=n_subsamples, fraction=fraction, threshold=threshold,
        expected_false=expected_false, n_permutations=n_permutations, subsample="fraction",
        strata=strata, return_refits=want, random_state=random_state,
        draw_random_state=draw_random_state, n_jobs=n_jobs, groups=groups,
        tune_learning_rate=tune_learning_rate, baseline=baseline, n_baseline=n_baseline, **params)
    refits = inf["refits"]
    if int(refits["structure"].shape[0]) == 0:
        raise RuntimeError("no subsample refit was kept; cannot aggregate.")
    freq = [np.asarray(f, dtype=float) for f in inf["selection_frequency"]]

    # Out-of-bag scores: each row averaged over the refits that did not contain it.
    lat = np.asarray(refits["latents"], dtype=float)                       # (B, n, k)
    lat = np.where(refits["in_bag"][:, :, None], np.nan, lat)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        oob = np.nanmean(lat, axis=0)                                        # (n, k)

    has_base = bool(baseline and n_views >= 2)
    interp = _build_stability_interpretation(inf, baseline=has_base, select_mode=(aggregate == "select"))
    common = {
        "aggregate": aggregate, "method": method, "k": int(k), "n_subsamples": int(n_subsamples),
        "fraction": float(fraction), "threshold": float(threshold),
        "selection_frequency": freq,
        "diagnostics_scope": ("pre_selection_subsamples" if aggregate == "select" else "ensemble_subsamples"),
        "latent_reliability": inf["latent_reliability"],
        "latent_reliability_baseline": inf["latent_reliability_baseline"],
        "latent_reliability_gain": inf["latent_reliability_gain"],
        "latent_reliability_averaged": inf["latent_reliability_averaged"],
        "latent_reliability_averaged_baseline": inf["latent_reliability_averaged_baseline"],
        "latent_reliability_averaged_gain": inf["latent_reliability_averaged_gain"],
        "latent_reliability_multivariate": inf["latent_reliability_multivariate"],
        "latent_reliability_multivariate_baseline": inf["latent_reliability_multivariate_baseline"],
        "latent_reliability_multivariate_gain": inf["latent_reliability_multivariate_gain"],
        "cross_view_coupling": inf["cross_view_coupling"],
        "cross_view_coupling_baseline": inf["cross_view_coupling_baseline"],
        "cross_view_coupling_gain": inf["cross_view_coupling_gain"],
        "cross_view_coupling_n": inf["cross_view_coupling_n"],
        "cross_view_coupling_splits": inf["cross_view_coupling_splits"],
        "feature_agreement": inf["feature_agreement"],
        "feature_agreement_baseline": inf["feature_agreement_baseline"],
        "feature_agreement_gain": inf["feature_agreement_gain"],
        "feature_agreement_by_view": inf["feature_agreement_by_view"],
        "feature_agreement_by_view_baseline": inf["feature_agreement_by_view_baseline"],
        "feature_agreement_by_view_gain": inf["feature_agreement_by_view_gain"],
        "n_oob_per_row": inf["n_oob_per_row"],
        "n_refits_used": inf["n_refits_used"],
        "alignment": inf["alignment"],
        "interpretation": interp,
        "oos_r2": (list(inf["oos_r2"]) if inf["oos_r2"] is not None else None),
        "refit_learning_rates": inf["refit_learning_rates"],
        "converged_fraction": float(inf["converged_fraction"]),
        "expected_false_selections": list(inf["expected_false_selections"]),
        "expected_false_selections_reason": inf["expected_false_selections_reason"],
        "selection_q": list(inf["q"]), "between_run_note": BETWEEN_RUN_NOTE, "provenance": provenance,
    }

    if aggregate == "select":
        mask = [f >= threshold for f in freq]
        empty = [i for i, mk in enumerate(mask) if not mk.any()]
        if empty:
            detail = "; ".join(f"view {names[i]!r} (index {i}): maximum selection frequency {freq[i].max():.3f}"
                               for i in empty)
            raise ValueError(f"bagged_fit(aggregate='select') kept no feature in {detail}; threshold is "
                             f"{threshold}. Lower the threshold, raise expected_false (selection_q was "
                             f"{list(inf['q'])}) or use aggregate='average'.")
        sub = [x[:, torch.as_tensor(mk)] for x, mk in zip(view_list, mask)]
        fit_views = dict(zip(names, sub)) if isinstance(views, Mapping) else sub
        select_params = dict(params)
        if isinstance(select_params.get("per_view_params"), Mapping):
            sub_pvp = {}
            for kv, ov in select_params["per_view_params"].items():
                ov_copy = dict(ov)
                if ov_copy.get("nonneg_mask") is not None:
                    if isinstance(kv, (int, np.integer)) and not isinstance(kv, bool) and 0 <= int(kv) < len(mask):
                        v_idx = int(kv)
                    elif isinstance(kv, str) and kv in names:
                        v_idx = names.index(kv)
                    elif isinstance(kv, str) and kv.startswith("view") and kv[4:].isdigit() and int(kv[4:]) < len(mask):
                        v_idx = int(kv[4:])
                    else:
                        v_idx = None
                    if v_idx is not None:
                        nm_arr = np.asarray(ov_copy["nonneg_mask"], dtype=bool)
                        if nm_arr.shape == mask[v_idx].shape:
                            ov_copy["nonneg_mask"] = nm_arr[mask[v_idx]]
                sub_pvp[kv] = ov_copy
            select_params["per_view_params"] = sub_pvp
        res = fit(method, fit_views, k, random_state=random_state, **select_params)
        rot = res["loading_rotation"].detach().double().numpy()
        scores = res["u"].detach().double().numpy() @ rot
        # Rotate the subsample fits' consensus scores into the final fit's frame (no sign or order is shared).
        oob_al = np.full_like(oob, np.nan)
        ok = np.isfinite(oob).all(axis=1)
        if int(ok.sum()) >= max(3, int(k)):
            q = _procrustes(oob[ok] - oob[ok].mean(0), scores[ok] - scores[ok].mean(0))
            oob_al = oob @ q
        if res.get("family") in ("linear", "baseline") and "v" in res:
            eff = [_effective_features(v.detach().double().numpy()) for v in res["v"]]
        else:
            eff = [int(mk.sum()) for mk in mask]
        res.update(common)
        res.update({"feature_mask": mask, "effective_features": eff, "scores": scores, "oob_scores": oob_al})
        return _RemovedKeysDict(res, carry_table=True)

    # average
    ref = dict(inf["result"])
    models = refits["models"]
    rots = np.asarray(refits["alignment"]["rotation"], dtype=float)
    u_mean, _ = _ensemble_scores(models, rots, view_list)
    ew, raw_w, icpt, why = _averaged_weights(models, rots)
    averaged = ew is not None
    if averaged:
        eff = [_effective_features(w) for w in ew]
    else:
        eff = [None] * len(view_list)
    ref.update(common)
    ref.update({
        "feature_mask": [np.ones(p, dtype=bool) for p in widths], "effective_features": eff,
        "scores": u_mean, "oob_scores": oob, "weights_averaged": bool(averaged),
        "weights_note": ("per-view encoders averaged; see bagged_fit" if averaged else why),
        "ensemble_models": models, "ensemble_rotations": rots, "n_fits": len(models),
    })
    if averaged:
        ref.update({"averaged_weights": ew, "weights_raw": raw_w, "intercepts": icpt})
    return _RemovedKeysDict(ref, carry_table=True)


__all__ = ["bagged_fit", "bagged_predict", "mask_views", "AGGREGATES"]
