"""Mathematical contract and integration tests for `pysimlr.uniform_shrinkage` (v0.43.1)."""
from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

import pysimlr
from pysimlr import sparsification as sp_mod


def _planted(seed, n_tr, widths, n_te=300, background=0.10, focal=0.20, noise=1.0):
    """One latent z; every feature loads positively; the first fifth of each view loads more."""
    rng = np.random.default_rng(seed)
    z_tr, z_te = rng.normal(size=n_tr), rng.normal(size=n_te)
    v_tr, v_te = [], []
    for p in widths:
        load = np.full(p, background)
        load[: p // 5] = focal
        a = np.outer(z_tr, load) + noise * rng.normal(size=(n_tr, p))
        b = np.outer(z_te, load) + noise * rng.normal(size=(n_te, p))
        mu, sd = a.mean(0), a.std(0)
        v_tr.append((a - mu) / sd)
        v_te.append((b - mu) / sd)
    return v_tr, v_te, z_te


def _planted_k2(seed, n_tr, widths, n_te=200, noise=0.8):
    """Two orthogonal non-negative disjoint latent modes (25% + 25% active features per view)."""
    rng = np.random.default_rng(seed)
    z_tr = rng.normal(size=(n_tr, 2))
    z_te = rng.normal(size=(n_te, 2))
    for Z in (z_tr, z_te):
        Z -= Z.mean(axis=0, keepdims=True)
        q_mat, _ = np.linalg.qr(Z)
        Z[:] = q_mat * np.sqrt(Z.shape[0])
    v_tr, v_te = [], []
    for p in widths:
        m = max(2, round(0.25 * p))
        W = np.zeros((p, 2), dtype=np.float64)
        W[:m, 0] = 0.65
        W[m : 2 * m, 1] = 0.55
        a = z_tr @ W.T + noise * rng.normal(size=(n_tr, p))
        b = z_te @ W.T + noise * rng.normal(size=(n_te, p))
        mu, sd = a.mean(0), np.where(a.std(0) > 1e-12, a.std(0), 1.0)
        v_tr.append((a - mu) / sd)
        v_te.append((b - mu) / sd)
    return v_tr, v_te, z_te


def _score(fit, views_te, z_te):
    u = pysimlr.project_uniform_shrunk(fit, views_te, post_shrink=0.0)["u"][:, 0]
    return abs(float(np.corrcoef(u, z_te)[0, 1]))


def _arm(v_tr, v_te, z_te, seed, alpha, sparseness, iters=60):
    kw = pysimlr.uniform_shrinkage_params(v_tr, alpha=alpha, sparseness=sparseness)
    fit = pysimlr.fit("simlr", v_tr, 1, random_state=seed, iterations=iters, **kw)
    return _score(fit, v_te, z_te)


def test_resolvent_matches_dense_inverse_and_is_stateless() -> None:
    """S = (I + lam L)^-1 exactly; S(Sv) follows the two-step formula (S is not idempotent)."""
    p, lam = 24, 2.5
    alpha = lam / (1.0 + lam)
    op = pysimlr.UniformShrinkageOperator(p, lambda_val=lam, positive_part=False)
    assert np.isclose(op.alpha, alpha)

    pi = np.full((p, p), 1.0 / p)
    dense = np.linalg.inv(np.eye(p) + lam * (np.eye(p) - pi))
    v = np.random.default_rng(7).normal(size=(p, 3))
    once = op @ v
    assert np.allclose(once, dense @ v, atol=1e-12)

    twice = op @ once
    expected = (1 - alpha) ** 2 * v + (1 - (1 - alpha) ** 2) * (pi @ v)
    assert np.allclose(twice, expected, atol=1e-12)
    assert np.allclose(twice, dense @ dense @ v, atol=1e-12)
    assert not np.allclose(twice, once, atol=1e-3)

    assert np.array_equal(op @ v, once)
    assert np.array_equal(op @ once, twice)

    vt = torch.tensor(v[:, 0], dtype=torch.float32)
    ot = op @ vt
    assert isinstance(ot, torch.Tensor) and ot.dtype == torch.float32
    assert np.allclose(ot.numpy(), (dense @ v[:, 0]).astype(np.float32), atol=1e-6)
    assert not torch.allclose(op @ ot, ot, atol=1e-3)


def test_alpha_and_lambda_are_the_same_knob() -> None:
    a = pysimlr.UniformShrinkageOperator(10, alpha=0.3)
    b = pysimlr.UniformShrinkageOperator(10, lambda_val=0.3 / 0.7)
    v = np.random.default_rng(0).uniform(size=(10, 2))
    assert np.allclose(a @ v, b @ v)


def test_positive_part_prevents_mixed_sign_cancellation_and_block_groups() -> None:
    v_mixed = np.array([2.0, -2.0, 1.0, -1.0])
    assert np.allclose(pysimlr.UniformShrinkageOperator(4, alpha=1.0, positive_part=True) @ v_mixed, np.full(4, 0.75))
    assert np.allclose(pysimlr.UniformShrinkageOperator(3, alpha=1.0) @ np.array([-3.0, -3.0, 0.0]), np.full(3, 2.0))

    op_grp = pysimlr.UniformShrinkageOperator(5, alpha=1.0, groups=[0, 0, 1, 1, -1], positive_part=False)
    v_grp = np.array([1.0, 3.0, 10.0, 20.0, 99.0])
    assert np.allclose(op_grp @ v_grp, [2.0, 2.0, 15.0, 15.0, 99.0])


def test_prop1_order_stat_soft_cancels_alpha_while_hard_default_preserves_shrinkage() -> None:
    """Prop 1: order-statistic soft-thresholding cancels alpha identically; hard default preserves it."""
    rng = np.random.default_rng(11)
    p = 60
    v = rng.lognormal(0.0, 0.6, size=(p, 1))
    v /= np.linalg.norm(v)
    ref_soft = None
    for a in (0.0, 0.3, 0.7, 0.95):
        vt = torch.as_tensor(pysimlr.UniformShrinkageOperator(p, alpha=a) @ v, dtype=torch.float64)
        u_soft = sp_mod._apply_quantile_sparseness(vt, 0.3, nonneg=True, alg="soft")
        u_soft = (u_soft / u_soft.norm(dim=0, keepdim=True)).numpy()[:, 0]
        if ref_soft is None:
            ref_soft = u_soft
        else:
            assert np.max(np.abs(u_soft - ref_soft)) < 1e-14

    # Default alg="hard" is non-subtractive and responds monotonically to alpha
    v0 = torch.as_tensor(pysimlr.UniformShrinkageOperator(p, alpha=0.0) @ v, dtype=torch.float64)
    v5 = torch.as_tensor(pysimlr.UniformShrinkageOperator(p, alpha=0.5) @ v, dtype=torch.float64)
    u_h0 = sp_mod._apply_quantile_sparseness(v0, 0.3, nonneg=True)
    u_h5 = sp_mod._apply_quantile_sparseness(v5, 0.3, nonneg=True)
    u_h0 = (u_h0 / u_h0.norm(dim=0, keepdim=True)).numpy()[:, 0]
    u_h5 = (u_h5 / u_h5.norm(dim=0, keepdim=True)).numpy()[:, 0]
    assert np.max(np.abs(u_h5 - u_h0)) > 1e-2


def test_prop2_radial_convexify_preserves_full_positive_support_and_direction() -> None:
    """Prop 2: radial _convexify preserves every positive entry and exact cosine 1.0 even when 1^T v > 1."""
    rng = np.random.default_rng(17)
    u = np.abs(rng.standard_normal((80, 2))) + 0.05
    u /= np.linalg.norm(u, axis=0, keepdims=True)
    u_t = torch.as_tensor(u, dtype=torch.float64)
    assert float(u_t.sum(dim=0).min()) > 1.0

    w = sp_mod._convexify(u_t)
    assert bool((w > 0.0).all())
    assert torch.allclose(w.sum(dim=0), torch.ones(2, dtype=torch.float64), atol=1e-12)
    # Direction is preserved identically (cosine == 1.0)
    w_unit = w / w.norm(dim=0, keepdim=True)
    assert torch.allclose(w_unit, u_t, atol=1e-12)

    with pytest.raises(ValueError, match="no positive entry"):
        sp_mod._convexify(torch.tensor([[1.0, 0.0], [2.0, -1.0]]))


def test_prop3_column_support_operator_exact_orthogonality_across_all_alphas() -> None:
    """Prop 3: ColumnSupportShrinkageOperator with disjoint supports satisfies V^T V = I_2 for all alpha in [0, 1]."""
    rng = np.random.default_rng(23)
    p = 80
    s1 = np.arange(0, 30, dtype=np.int64)
    s2 = np.arange(30, 60, dtype=np.int64)
    V0 = np.zeros((p, 2), dtype=np.float64)
    V0[s1, 0] = rng.uniform(0.2, 1.5, size=s1.size)
    V0[s2, 1] = rng.uniform(0.2, 1.5, size=s2.size)
    V0 /= np.linalg.norm(V0, axis=0, keepdims=True)

    for a in (0.0, 0.25, 0.5, 0.85, 1.0):
        op = pysimlr.ColumnSupportShrinkageOperator(p, [s1, s2], [a, a])
        assert op.is_disjoint()
        Vc = op @ V0
        assert np.all(Vc[60:, :] == 0.0)
        assert np.all(Vc[s2, 0] == 0.0) and np.all(Vc[s1, 1] == 0.0)
        Vc /= np.linalg.norm(Vc, axis=0, keepdims=True)
        fro = float(np.linalg.norm(Vc.T @ Vc - np.eye(2), ord="fro"))
        assert fro < 1e-14


def test_effective_support_dof_and_directional_defaults() -> None:
    assert np.isclose(pysimlr.effective_support_dof(50, 50, mode="ric"), 50.0)
    assert np.isclose(pysimlr.effective_support_dof(20, 50, mode="plugin"), 20.0)
    assert pysimlr.effective_support_dof(20, 50, mode="ric") > 20.0
    assert pysimlr.resolve_directional_defaults(1) == ("conical", "nc", "pca")
    assert pysimlr.resolve_directional_defaults(2) == ("convex", "acc", "pca")
    assert pysimlr.resolve_directional_defaults(
        1, positivity="convex", energy_type="recon_r2", mixing_algorithm="svd"
    ) == ("convex", "recon_r2", "svd")
    assert pysimlr.resolve_directional_defaults(2, positivity="conical") == ("conical", "acc", "pca")


def test_input_validation() -> None:
    for bad in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError, match="alpha"):
            pysimlr.UniformShrinkageOperator(5, alpha=bad)
    with pytest.raises(ValueError, match="lambda_val"):
        pysimlr.UniformShrinkageOperator(5, lambda_val=-1.0)
    with pytest.raises(ValueError, match="groups"):
        pysimlr.UniformShrinkageOperator(5, alpha=0.5, groups=[0, 1, 2])
    with pytest.raises(ValueError, match="Dimension mismatch"):
        _ = pysimlr.UniformShrinkageOperator(5, alpha=0.5) @ np.ones(4)
    with pytest.raises(ValueError, match="alpha"):
        pysimlr.create_uniform_smoother(np.ones((10, 4)), alpha="bogus")
    v_tr, _, _ = _planted(0, 20, (10,))
    with pytest.raises(ValueError, match="groups"):
        pysimlr.uniform_shrinkage_params(v_tr, groups=[[0, 1]])
    with pytest.raises(ValueError, match="positivity"):
        pysimlr.uniform_shrinkage_params(v_tr, positivity="either")
    with pytest.raises(ValueError, match="mode"):
        pysimlr.effective_support_dof(5, 10, mode="invalid")


def test_auto_uniform_alpha_asymptotics() -> None:
    a_small_n = pysimlr.auto_uniform_alpha(n_obs=50, n_features=100)
    a_equal = pysimlr.auto_uniform_alpha(n_obs=100, n_features=100)
    a_large_n = pysimlr.auto_uniform_alpha(n_obs=1000, n_features=100)
    assert a_small_n > a_equal > a_large_n
    assert np.isclose(a_equal, 0.5)
    assert np.isclose(a_large_n, 100.0 / 1100.0)


def test_auto_alpha_counts_rows_with_any_finite_feature() -> None:
    rng = np.random.default_rng(3)
    clean = rng.normal(size=(60, 40))
    holes = clean.copy()
    holes[rng.random(holes.shape) < 0.05] = np.nan
    assert not np.isfinite(holes).all(axis=1).all()
    a_clean = pysimlr.create_uniform_smoother(clean).alpha
    a_holes = pysimlr.create_uniform_smoother(holes).alpha
    assert a_holes == a_clean
    holes[0, :] = np.nan
    assert pysimlr.create_uniform_smoother(holes).alpha > a_clean


def test_resolved_defaults_are_echoed_in_provenance() -> None:
    v_tr, _, _ = _planted(0, 30, (20, 25))
    kw = pysimlr.uniform_shrinkage_params(v_tr)
    prov = kw.provenance
    assert prov["positivity"] == "conical" and prov["positivity_requested"] == "auto"
    assert prov["sparseness"] is None
    assert prov["sparseness_type"] == "hard"
    assert prov["scale_list"] == ["none"] and prov["energy_type"] == "nc" and prov["mixing_algorithm"] == "pca"
    assert prov["alpha_mode"] == "ric"
    assert prov["nsa_w"] == kw["nsa_w"] == 0.0 and prov["nsa_w_requested"] == "auto"
    assert prov["n_obs"] == [30, 30] and prov["n_features"] == [20, 25]
    assert np.allclose(prov["alpha"], [20 / 50, 25 / 55])
    assert "provenance" not in kw
    kw_cx = pysimlr.uniform_shrinkage_params(v_tr, positivity="convex")
    assert kw_cx.provenance["positivity"] == "convex" and kw_cx["positivity"] == "convex"
    supps = [[np.arange(0, 10), np.arange(10, 20)], [np.arange(0, 12), np.arange(12, 25)]]
    kw_k2 = pysimlr.uniform_shrinkage_params(v_tr, k=2, supports=supps)
    assert kw_k2.provenance["positivity"] == "convex" and kw_k2["positivity"] == "convex"
    kw2 = pysimlr.uniform_shrinkage_params(v_tr, nsa_w=0.5, energy_type="reconorm", sparseness=0.3)
    assert kw2["nsa_w"] == 0.5 and kw2["energy_type"] == "reconorm" and kw2.provenance["energy_type"] == "reconorm"
    assert pysimlr.uniform_shrinkage_params(v_tr, sparseness=0.3)["nsa_w"] == 0.2


def test_k_greater_than_one_supports_disjoint_and_guards_global() -> None:
    v_tr, _, _ = _planted(0, 40, (30, 30))
    with pytest.raises(ValueError, match="fit_uniform_shrunk"):
        pysimlr.uniform_shrinkage_params(v_tr, k=2)
    with pytest.raises(ValueError, match="alpha"):
        pysimlr.uniform_shrinkage_params(v_tr, k=2, alpha=0.5, groups=[[0, 1] * 15] * 2)
    # Disjoint supports allow arbitrary alpha in [0, 1] at k > 1
    supps = [[np.arange(0, 15), np.arange(15, 30)]] * 2
    kw_d = pysimlr.uniform_shrinkage_params(v_tr, k=2, alpha=0.8, supports=supps)
    assert kw_d["energy_type"] == "acc" and kw_d["mixing_algorithm"] == "pca"
    assert all(isinstance(op, pysimlr.ColumnSupportShrinkageOperator) for op in kw_d["smoothing_matrices"])
    assert kw_d["nsa_w"] == 0.0
    # Overlapping supports at k > 1 must raise
    with pytest.raises(ValueError, match="disjoint"):
        pysimlr.uniform_shrinkage_params(v_tr, k=2, alpha=0.5, supports=[[np.arange(0, 20), np.arange(10, 30)]] * 2)


def test_fit_uniform_shrunk_k1_dense_sparse_and_k2_disjoint() -> None:
    """Verify fit_uniform_shrunk + support-restricted project_uniform_shrunk for k=1 and k=2."""
    # k=1 dense
    v_tr, v_te, z_te = _planted(10, 50, (40, 50, 30), n_te=100, background=0.20, focal=0.40)
    res_d = pysimlr.fit_uniform_shrunk(v_tr, k=1, random_state=0, iterations=25)
    prov_d = res_d["uniform_shrinkage_provenance"]
    assert prov_d["stage_mode"] == "single_stage_dense"
    assert prov_d["positivity"] == "conical" and prov_d["positivity_requested"] == "auto"
    assert prov_d["resolved_energy_type"] == "nc" and prov_d["resolved_mixing_algorithm"] == "pca"
    proj_d = pysimlr.project_uniform_shrunk(res_d, v_te, train_views=v_tr)
    assert _score(res_d, v_te, z_te) > 0.70 and np.isfinite(proj_d["u"]).all()

    # k=1 sparse two-stage (exact zero preservation in both fit['v'] and project_uniform_shrunk['weights']!)
    res_s = pysimlr.fit_uniform_shrunk(v_tr, k=1, sparseness=0.4, random_state=0, iterations=20)
    prov_s = res_s["uniform_shrinkage_provenance"]
    assert prov_s["stage_mode"] == "two_stage_support_restricted"
    assert prov_s["positivity"] == "conical" and prov_s["positivity_requested"] == "auto"
    proj_s = pysimlr.project_uniform_shrunk(res_s, v_te, train_views=v_tr, post_shrink="auto")
    for m_i, (v_t, w_m, supps_m) in enumerate(zip(res_s["v"], proj_s["weights"], prov_s["supports"])):
        v_np = v_t.detach().cpu().double().numpy()
        s_idx = np.asarray(supps_m[0], dtype=np.int64)
        off_mask = np.ones(v_np.shape[0], dtype=bool)
        off_mask[s_idx] = False
        assert np.all(v_np[off_mask, 0] == 0.0), f"view {m_i} fit leaked outside support"
        assert np.all(w_m[off_mask, 0] == 0.0), f"view {m_i} projection leaked outside support"
        assert np.isclose(w_m[:, 0].sum(), 1.0)

    # k=2 disjoint two-stage (default positivity="auto" -> "convex", plus explicit "conical" & "convex")
    v2_tr, v2_te, _z2_te = _planted_k2(11, 60, (48, 56, 40), n_te=120)
    for pos_req, pos_exp in (("auto", "convex"), ("conical", "conical"), ("convex", "convex")):
        res_k2 = pysimlr.fit_uniform_shrunk(
            v2_tr, k=2, sparseness=0.3, positivity=pos_req, random_state=0, iterations=25
        )
        prov_k2 = res_k2["uniform_shrinkage_provenance"]
        assert prov_k2["stage_mode"] == "two_stage_support_restricted"
        assert prov_k2["positivity"] == pos_exp and prov_k2["positivity_requested"] == pos_req
        assert prov_k2["resolved_energy_type"] == "acc" and prov_k2["resolved_mixing_algorithm"] == "pca"
        proj_k2 = pysimlr.project_uniform_shrunk(res_k2, v2_te, train_views=v2_tr, post_shrink="auto")
        u_te = proj_k2["u"]
        assert u_te.shape == (120, 2) and np.isfinite(u_te).all()
        assert abs(float(np.corrcoef(u_te[:, 0], u_te[:, 1])[0, 1])) < 0.35
        for v_t, w_m in zip(res_k2["v"], proj_k2["weights"]):
            V = v_t.detach().cpu().double().numpy()
            Vn = V / np.linalg.norm(V, axis=0, keepdims=True)
            assert float(np.linalg.norm(Vn.T @ Vn - np.eye(2), ord="fro")) < 1e-6
            assert float(abs(np.dot(w_m[:, 0], w_m[:, 1]))) == 0.0
            assert np.allclose(w_m.sum(axis=0), 1.0)


def test_signed_nc_warning_suppressed_for_k1_and_emitted_for_k2() -> None:
    from pysimlr.similarity import _reset_nc_positive_warning

    v_tr, _, _ = _planted(12, 30, (20, 20))
    _reset_nc_positive_warning()
    with warnings.catch_warnings(record=True) as w_k1:
        warnings.simplefilter("always")
        pysimlr.fit_uniform_shrunk(v_tr, k=1, random_state=0, iterations=5, learning_rate=0.05)
    assert not [w for w in w_k1 if issubclass(w.category, UserWarning) and "nc_abs" in str(w.message)]

    _reset_nc_positive_warning()
    with warnings.catch_warnings(record=True) as w_k2:
        warnings.simplefilter("always")
        pysimlr.fit("simlr", v_tr, 2, positivity="positive", energy_type="nc", iterations=4, random_state=0)
    assert [w for w in w_k2 if issubclass(w.category, UserWarning) and "nc_abs" in str(w.message)]


def test_alpha_one_reproduces_plain_mean_exactly() -> None:
    v_tr, v_te, _ = _planted(1, 45, (60, 80, 50), n_te=80)
    kw = pysimlr.uniform_shrinkage_params(v_tr, alpha=1.0, clip=(0.0, 1.0), sparseness=None)
    fit = pysimlr.fit("simlr", v_tr, 1, random_state=0, iterations=25, **kw)
    proj = pysimlr.project_uniform_shrunk(fit, v_te, post_shrink=0.0)
    plain = np.mean([v.mean(axis=1) for v in v_te], axis=0)
    assert np.allclose(proj["u"][:, 0], plain, atol=1e-5)


def test_alpha_zero_is_close_to_but_not_the_plain_fit() -> None:
    v_tr, _, _ = _planted(2, 150, (30, 30))
    kw = pysimlr.uniform_shrinkage_params(
        v_tr, alpha=0.0, sparseness=None, positivity="conical", energy_type="recon_r2", mixing_algorithm="svd"
    )
    fit0 = pysimlr.fit("simlr", v_tr, 1, random_state=0, iterations=60, **kw)
    plain = pysimlr.fit(
        "simlr", v_tr, 1, scale_list=["none"], energy_type="recon_r2", mixing_algorithm="svd",
        positivity="conical", sparseness=None, random_state=0, iterations=60,
    )
    for a, b in zip(fit0["v"], plain["v"]):
        a, b = a.detach().numpy().ravel(), b.detach().numpy().ravel()
        cos = abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
        assert cos >= 0.95


def test_shrinkage_gain_small_n_and_no_loss_large_n() -> None:
    gains_small, gains_large = [], []
    for seed in range(10):
        v_tr, v_te, z = _planted(seed, 40, (100, 120, 100))
        gains_small.append(_arm(v_tr, v_te, z, seed, "auto", None) - _arm(v_tr, v_te, z, seed, [None] * 3, None))
        v_tr, v_te, z = _planted(seed, 600, (20, 25, 15))
        gains_large.append(_arm(v_tr, v_te, z, seed, "auto", None) - _arm(v_tr, v_te, z, seed, [None] * 3, None))
    assert np.mean(gains_small) >= 0.03
    assert np.mean(np.array(gains_small) > 0) >= 0.8
    assert np.mean(gains_large) >= -0.01


def test_constant_columns_and_single_view() -> None:
    v_tr, v_te, _ = _planted(4, 40, (30, 30))
    for v in (v_tr[0], v_te[0]):
        v[:, 3] = 0.0
    fit = pysimlr.fit_uniform_shrunk(v_tr, k=1, random_state=0, iterations=20)
    proj = pysimlr.project_uniform_shrunk(fit, v_te)
    assert np.isfinite(proj["u"]).all()
    assert all(np.isfinite(w).all() and np.isclose(w.sum(), 1.0) for w in proj["weights"])

    op = pysimlr.UniformShrinkageOperator(6, alpha=0.5)
    assert np.isfinite(op @ np.zeros((6, 2))).all()

    kw1 = pysimlr.uniform_shrinkage_params(v_tr[:1], alpha=0.3, sparseness=None)
    assert len(kw1["smoothing_matrices"]) == 1
    proj1 = pysimlr.project_uniform_shrunk({"v": [np.full((30, 1), 1 / 30.0)]}, v_te[:1], post_shrink=0.0)
    assert np.allclose(proj1["u"][:, 0], v_te[0].mean(axis=1))


def test_deterministic_for_fixed_random_state() -> None:
    v_tr, v_te, _ = _planted(5, 40, (50, 60))
    outs = []
    for _ in range(2):
        fit = pysimlr.fit_uniform_shrunk(v_tr, k=1, sparseness=0.3, random_state=3, iterations=20)
        outs.append(pysimlr.project_uniform_shrunk(fit, v_te, train_views=v_tr)["u"])
    assert np.array_equal(outs[0], outs[1])


def test_mixed_sign_planted_signal_shows_why_default_is_off() -> None:
    rng = np.random.default_rng(99)
    n = 120
    z = rng.normal(size=(n, 1))
    views = []
    for p_m in (40, 40):
        signs = np.array([1.0] * 20 + [-1.0] * 20)[None, :]
        x = 0.6 * (z @ signs) + rng.normal(scale=0.8, size=(n, p_m))
        views.append((x - x.mean(0)) / x.std(0))

    fit_default = pysimlr.fit("simlr", views, 1, positivity="either", sparseness=None, random_state=0, iterations=60)
    r_def = abs(float(np.corrcoef(fit_default["u"].detach().cpu().numpy()[:, 0], z[:, 0])[0, 1]))
    fit_shr = pysimlr.fit_uniform_shrunk(views, k=1, alpha=0.85, random_state=0, iterations=60)
    r_shr = abs(float(np.corrcoef(pysimlr.project_uniform_shrunk(fit_shr, views)["u"][:, 0], z[:, 0])[0, 1]))
    assert r_def > 0.85
    assert r_def > r_shr + 0.15


def test_flipped_sign_view_gives_negated_score() -> None:
    v_tr, v_te, _ = _planted(6, 60, (20, 20))
    fit = pysimlr.fit_uniform_shrunk(v_tr, k=1, alpha=0.3, random_state=0, iterations=20)
    base = pysimlr.project_uniform_shrunk(fit, v_te, post_shrink=0.0)["view_scores"]
    flipped = pysimlr.project_uniform_shrunk(fit, [-v_te[0], v_te[1]], post_shrink=0.0)["view_scores"]
    assert np.allclose(flipped[:, 0], -base[:, 0])
    assert np.allclose(flipped[:, 1], base[:, 1])
