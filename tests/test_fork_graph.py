"""Tests for `pysimlr.fork_graph` (causal Fork topology constructor)."""

from __future__ import annotations

import numpy as np
import pytest

import pysimlr


def test_fork_graph_validation_and_conventions() -> None:
    """`fork_graph` validates distinct non-empty nodes and respects fit vs SEM edge orientation."""
    with pytest.raises(ValueError, match="at least one child"):
        pysimlr.fork_graph("Root", [])
    with pytest.raises(ValueError, match="distinct"):
        pysimlr.fork_graph("Root", ["Root", "ChildA"])
    with pytest.raises(ValueError, match="distinct"):
        pysimlr.fork_graph(0, [1, 1])

    fg_exog = pysimlr.fork_graph("Blast_Root", ["T1", "DTI", "ASL"], exogenous_root=True, for_sem=False)
    assert fg_exog == {
        "Blast_Root": [],
        "T1": ["Blast_Root"],
        "DTI": ["Blast_Root"],
        "ASL": ["Blast_Root"],
    }

    fg_bidir = pysimlr.fork_graph("Blast_Root", ["T1", "DTI"], exogenous_root=False, for_sem=False)
    assert fg_bidir == {
        "Blast_Root": ["T1", "DTI"],
        "T1": ["Blast_Root"],
        "DTI": ["Blast_Root"],
    }

    sem_g = pysimlr.fork_graph("Blast_Root", ["T1", "DTI", "ASL"], for_sem=True)
    assert sem_g == {
        "Blast_Root": ["T1", "DTI", "ASL"],
        "T1": [],
        "DTI": [],
        "ASL": [],
    }


def test_exogenous_fork_root_is_invariant_to_child_perturbation_and_missing_rows() -> None:
    """Under `exogenous_root=True`, `V_root`'s sweep update has zero dependence on child prongs."""
    rng = np.random.default_rng(42)
    n = 64
    latent = np.abs(rng.normal(size=(n, 1)))
    x_root = np.clip(latent @ np.array([[1.2, 0.9, 1.5, 0.7]]) + 0.15 * np.abs(rng.normal(size=(n, 4))), 0.0, None)
    x_c1 = latent @ np.array([[0.8, 0.6, 0.9, 0.5, 0.7]]) + 0.35 * rng.normal(size=(n, 5))
    x_c2 = latent @ np.array([[0.7, 0.9, 0.5, 0.8, 0.6]]) + 0.35 * rng.normal(size=(n, 5))
    x_c2_partial = x_c2.copy()
    x_c2_partial[40:] = np.nan

    perm_obs = rng.permutation(40)
    x_c2_perm = x_c2_partial.copy()
    x_c2_perm[:40] = x_c2_partial[perm_obs]

    fg = pysimlr.fork_graph("Root", ["C1", "C2"], exogenous_root=True, for_sem=False)
    # Single-sweep update of V_root is bit-for-bit independent of child permutations/missingness
    step_a = pysimlr.fit(
        "simlr",
        {"Root": x_root, "C1": x_c1, "C2": x_c2_partial},
        k=1,
        topology="graph",
        path_graph=fg,
        positivity="positive",
        energy_type="nc_abs",
        dynamic_weights=False,
        missing_rows="mask",
        iterations=1,
        learning_rate=0.05,
        random_state=7,
    )
    step_b = pysimlr.fit(
        "simlr",
        {"Root": x_root, "C1": x_c1, "C2": x_c2_perm},
        k=1,
        topology="graph",
        path_graph=fg,
        positivity="positive",
        energy_type="nc_abs",
        dynamic_weights=False,
        missing_rows="mask",
        iterations=1,
        learning_rate=0.05,
        random_state=7,
    )
    np.testing.assert_allclose(
        step_a["v"][0].detach().cpu().numpy(),
        step_b["v"][0].detach().cpu().numpy(),
        atol=1e-7,
    )

    # Across multi-sweep fits (`best_iteration` selection), cosine alignment stays > 0.999999
    res_a = pysimlr.fit(
        "simlr",
        {"Root": x_root, "C1": x_c1, "C2": x_c2_partial},
        k=1,
        topology="graph",
        path_graph=fg,
        positivity="positive",
        energy_type="nc_abs",
        dynamic_weights=False,
        missing_rows="mask",
        iterations=25,
        learning_rate=0.01,
        random_state=7,
    )
    res_b = pysimlr.fit(
        "simlr",
        {"Root": x_root, "C1": x_c1, "C2": x_c2_perm},
        k=1,
        topology="graph",
        path_graph=fg,
        positivity="positive",
        energy_type="nc_abs",
        dynamic_weights=False,
        missing_rows="mask",
        iterations=25,
        learning_rate=0.01,
        random_state=7,
    )
    va = res_a["v"][0].detach().cpu().double().numpy().ravel()
    vb = res_b["v"][0].detach().cpu().double().numpy().ravel()
    cos = float(abs(np.dot(va, vb)) / (np.linalg.norm(va) * np.linalg.norm(vb)))
    assert cos > 0.999999


def test_fork_graph_sem_directed_d_separation() -> None:
    """`fork_graph(..., for_sem=True)` with `path_model(..., directed=True, conditioning='parents')` conditions inter-child pairs on root."""
    rng = np.random.default_rng(123)
    n = 90
    z_r = rng.normal(size=(n, 1))
    x_r = z_r @ np.array([[1.0, 0.8, 0.9]]) + 0.1 * rng.normal(size=(n, 3))
    x_1 = 0.85 * z_r @ np.array([[1.0, 0.7, 0.5, 0.6]]) + 0.5 * rng.normal(size=(n, 4))
    x_2 = 0.85 * z_r @ np.array([[0.9, 0.6, 0.8, 0.5]]) + 0.5 * rng.normal(size=(n, 4))

    views_tr = {"Root": x_r[:50], "C1": x_1[:50], "C2": x_2[:50]}
    views_te = {"Root": x_r[50:], "C1": x_1[50:], "C2": x_2[50:]}

    fg_fit = pysimlr.fork_graph("Root", ["C1", "C2"], exogenous_root=True, for_sem=False)
    fg_sem = pysimlr.fork_graph("Root", ["C1", "C2"], for_sem=True)
    fit_res = pysimlr.fit(
        "simlr",
        views_tr,
        k=1,
        topology="graph",
        path_graph=fg_fit,
        dynamic_weights=False,
        iterations=20,
        random_state=11,
    )
    pm_pair = pysimlr.path_model(
        fit_res, views_te, rows="heldout", graph=fg_sem, directed=True, conditioning="pairwise", n_permutations=200, n_bootstrap=0
    )
    pm_par = pysimlr.path_model(
        fit_res, views_te, rows="heldout", graph=fg_sem, directed=True, conditioning="parents", n_permutations=200, n_bootstrap=0
    )
    e_pair = pm_pair["edges"].set_index(["a", "b"])
    e_par = pm_par["edges"].set_index(["a", "b"])
    assert bool(e_par.loc[("Root", "C1"), "present"]) is True
    assert bool(e_par.loc[("C1", "C2"), "present"]) is False
    # Conditioning on common parent `Root` substantially attenuates spurious `C1 -> C2` coupling
    assert float(e_par.loc[("C1", "C2"), "pillai"]) < float(e_pair.loc[("C1", "C2"), "pillai"])


def test_predict_exclude_views_prevents_root_leakage() -> None:
    """`pysimlr.predict(..., exclude_views=...)` computes subset consensus strictly without excluded root/target views."""
    rng = np.random.default_rng(99)
    n = 70
    z = rng.normal(size=(n, 1))
    x_root = z @ np.array([[1.1, 0.9, 1.0]]) + 0.1 * rng.normal(size=(n, 3))
    x_c1 = 0.85 * z @ np.array([[1.0, 0.8, 0.6, 0.7]]) + 0.35 * rng.normal(size=(n, 4))
    x_c2 = 0.85 * z @ np.array([[0.9, 0.7, 0.8, 0.5]]) + 0.35 * rng.normal(size=(n, 4))
    x_ext = 0.80 * z @ np.array([[0.8, 0.9, 0.7]]) + 0.40 * rng.normal(size=(n, 3))

    fg = pysimlr.fork_graph("Root", ["C1", "C2"], exogenous_root=True)
    tr_map = {"Root": x_root[:45], "C1": x_c1[:45], "C2": x_c2[:45]}
    fit_res = pysimlr.fit(
        "simlr",
        tr_map,
        k=1,
        topology="graph",
        path_graph=fg,
        dynamic_weights=False,
        iterations=20,
        random_state=3,
    )
    ext_res = pysimlr.extend(fit_res, tr_map, {"C_ext": x_ext[:45]}, how="anchored", iterations=15, random_state=3, warn=False)

    te_clean = {"Root": x_root[45:], "C1": x_c1[45:], "C2": x_c2[45:], "C_ext": x_ext[45:]}
    te_corrupt_root = {"Root": x_root[45:] * -1e6 + 999.0, "C1": x_c1[45:], "C2": x_c2[45:], "C_ext": x_ext[45:]}

    pred_clean = pysimlr.predict(ext_res, te_clean, exclude_views=["Root"])
    pred_corrupt = pysimlr.predict(ext_res, te_corrupt_root, exclude_views=["Root"])
    assert pred_clean["views_used"] == ["C1", "C2", "C_ext"]
    assert "Root" not in pred_clean["latents"]
    np.testing.assert_allclose(pred_clean["u"].numpy(), pred_corrupt["u"].numpy(), atol=1e-12)

    # Positional list input with integer index exclusion matches None-masked list prediction bit-for-bit
    pred_list_excl = pysimlr.predict(fit_res, [x_root[45:] * 1e5, x_c1[45:], x_c2[45:]], exclude_views=[0])
    pred_list_none = pysimlr.predict(fit_res, [None, x_c1[45:], x_c2[45:]])
    np.testing.assert_allclose(pred_list_excl["u"].numpy(), pred_list_none["u"].numpy(), atol=1e-12)

    # Empty exclude_views=() preserves standard full-view list prediction contract
    pred_full = pysimlr.predict(fit_res, [x_root[45:], x_c1[45:], x_c2[45:]], exclude_views=())
    assert isinstance(pred_full["latents"], list) and len(pred_full["latents"]) == 3

    with pytest.raises(TypeError, match="sequence"):
        pysimlr.predict(fit_res, te_clean, exclude_views="Root")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown view"):
        pysimlr.predict(fit_res, te_clean, exclude_views=["NonExistentView"])
    with pytest.raises(ValueError, match="no views given"):
        pysimlr.predict(fit_res, [x_root[45:], x_c1[45:], x_c2[45:]], exclude_views=[0, 1, 2])

