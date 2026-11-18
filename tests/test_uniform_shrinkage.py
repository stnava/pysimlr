"""Mathematical contract and integration tests for `pysimlr.uniform_shrinkage`."""
from __future__ import annotations

import numpy as np
import pytest
import torch

import pysimlr


def test_complete_graph_resolvent_identity_and_idempotence() -> None:
    """Verifies (I + lambda (I - 11^T/P))^(-1) == (1 - alpha)I + alpha 11^T/P and image idempotence."""
    p = 24
    lam = 2.5
    alpha_expected = lam / (1.0 + lam)
    op = pysimlr.UniformShrinkageOperator(p, lambda_val=lam, positive_part=False)
    assert np.isclose(op.alpha, alpha_expected)

    L_dense = np.eye(p) - np.full((p, p), 1.0 / p)
    resolvent_dense = np.linalg.inv(np.eye(p) + lam * L_dense)

    rng = np.random.default_rng(7)
    v = rng.normal(size=(p, 3))
    out_np = op @ v
    assert np.allclose(out_np, resolvent_dense @ v, atol=1e-12)

    # Idempotent on its own image (prevents double-smoothing inside simlr)
    out_twice = op @ out_np
    assert np.allclose(out_twice, out_np, atol=1e-12)

    # Torch 1-D and 2-D compatibility
    vt = torch.tensor(v[:, 0], dtype=torch.float32)
    ot = op @ vt
    assert isinstance(ot, torch.Tensor)
    assert ot.dtype == torch.float32
    assert np.allclose(ot.numpy(), (resolvent_dense @ v[:, 0]).astype(np.float32), atol=1e-6)
    assert torch.allclose(op @ ot, ot)


def test_positive_part_prevents_mixed_sign_cancellation_and_block_groups() -> None:
    """Verifies positive_part protects zero-sum iterates and block groups shrink within blocks only."""
    # Mixed-sign vector summing to zero would collapse to 0 at alpha=1 without positive_part
    v_mixed = np.array([2.0, -2.0, 1.0, -1.0], dtype=np.float64)
    op_pos = pysimlr.UniformShrinkageOperator(4, alpha=1.0, positive_part=True)
    res_pos = op_pos @ v_mixed
    assert np.allclose(res_pos, np.full(4, 0.75))

    # Block-wise groups (group 0: indices 0,1; group 1: indices 2,3; ungrouped -1: index 4)
    op_grp = pysimlr.UniformShrinkageOperator(5, alpha=1.0, groups=[0, 0, 1, 1, -1], positive_part=False)
    v_grp = np.array([1.0, 3.0, 10.0, 20.0, 99.0], dtype=np.float64)
    assert np.allclose(op_grp @ v_grp, np.array([2.0, 2.0, 15.0, 15.0, 99.0]))


def test_auto_uniform_alpha_asymptotics() -> None:
    """Verifies dimension-adaptive shrinkage scales monotonically with p / (n + p)."""
    a_small_n = pysimlr.auto_uniform_alpha(n_obs=50, n_features=100)
    a_equal = pysimlr.auto_uniform_alpha(n_obs=100, n_features=100)
    a_large_n = pysimlr.auto_uniform_alpha(n_obs=1000, n_features=100)
    assert a_small_n > a_equal > a_large_n
    assert np.isclose(a_equal, 0.5)
    assert np.isclose(a_large_n, 100.0 / 1100.0)


def test_end_to_end_simlr_uniform_limit_matches_plain_mean_and_improves_small_n_cv() -> None:
    """Verifies exact convergence to unweighted plain mean at the shrinkage limit and small-n OOF gain."""
    rng = np.random.default_rng(123)
    n_tr, n_te = 45, 80
    widths = (60, 80, 50)
    z_lat_tr = rng.normal(size=(n_tr, 1))
    z_lat_te = rng.normal(size=(n_te, 1))

    # Planted directional multi-view structure (all features oriented positively with z_lat, focal subset stronger)
    views_tr, views_te = [], []
    for p_m in widths:
        load = np.full((1, p_m), 0.18)
        load[0, : p_m // 5] = 0.55
        x_tr = z_lat_tr @ load + rng.normal(scale=1.0, size=(n_tr, p_m))
        x_te = z_lat_te @ load + rng.normal(scale=1.0, size=(n_te, p_m))
        mu = x_tr.mean(axis=0, keepdims=True)
        sd = x_tr.std(axis=0, keepdims=True)
        views_tr.append((x_tr - mu) / sd)
        views_te.append((x_te - mu) / sd)

    # 1. Exact plain-mean identity at alpha=1.0 (or post_shrink=1.0)
    kw_lim = pysimlr.uniform_shrinkage_params(views_tr, alpha=1.0, clip=(0.0, 1.0), sparseness=None)
    fit_lim = pysimlr.fit("simlr", views_tr, 1, random_state=0, iterations=25, **kw_lim)
    proj_lim = pysimlr.project_uniform_shrunk(fit_lim, views_te)
    plain_mean_te = np.mean([np.mean(v, axis=1) for v in views_te], axis=0)
    assert np.allclose(proj_lim["u"][:, 0], plain_mean_te, atol=1e-5)

    # 2. Small-n (n_tr=45 < p=80): uniform-shrunk SiMLR beats unshrunk SiMLR on held-out latent recovery
    kw_auto = pysimlr.uniform_shrinkage_params(views_tr, alpha="auto", sparseness=0.3)
    fit_shrunk = pysimlr.fit("simlr", views_tr, 1, random_state=0, iterations=60, **kw_auto)
    u_shrunk = pysimlr.project_uniform_shrunk(fit_shrunk, views_te)["u"][:, 0]

    fit_unshrunk = pysimlr.fit(
        "simlr", views_tr, 1, scale_list=["none"], energy_type="recon_r2",
        mixing_algorithm="svd", positivity="conical", sparseness=0.5,
        random_state=0, iterations=60,
    )
    u_unshrunk = pysimlr.predict(fit_unshrunk, views_te)["u"].detach().cpu().numpy()[:, 0]

    r_shrunk = abs(float(np.corrcoef(u_shrunk, z_lat_te[:, 0])[0, 1]))
    r_unshrunk = abs(float(np.corrcoef(u_unshrunk, z_lat_te[:, 0])[0, 1]))
    assert r_shrunk > r_unshrunk


def test_mixed_sign_planted_signal_shows_why_default_is_off() -> None:
    """Demonstrates why uniform shrinkage must NOT be default ON when feature signs are arbitrary."""
    rng = np.random.default_rng(99)
    n = 120
    z = rng.normal(size=(n, 1))
    views = []
    for p_m in (40, 40):
        # Half +0.6, half -0.6 (unoriented raw features whose mean cancels identically)
        signs = np.array([1.0] * 20 + [-1.0] * 20)[None, :]
        x = 0.6 * (z @ signs) + rng.normal(scale=0.8, size=(n, p_m))
        views.append((x - x.mean(0)) / x.std(0))

    fit_default = pysimlr.fit("simlr", views, 1, positivity="either", sparseness=None, random_state=0, iterations=60)
    u_def = fit_default["u"].detach().cpu().numpy()[:, 0]
    r_def = abs(float(np.corrcoef(u_def, z[:, 0])[0, 1]))

    kw_shrunk = pysimlr.uniform_shrinkage_params(views, alpha=0.85, sparseness=None)
    fit_shr = pysimlr.fit("simlr", views, 1, random_state=0, iterations=60, **kw_shrunk)
    u_shr = pysimlr.project_uniform_shrunk(fit_shr, views)["u"][:, 0]
    r_shr = abs(float(np.corrcoef(u_shr, z[:, 0])[0, 1]))

    assert r_def > 0.85
    assert r_def > r_shr + 0.15
