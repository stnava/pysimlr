"""Mathematical contract and integration tests for `pysimlr.uniform_shrinkage`."""
from __future__ import annotations

import numpy as np
import pytest
import torch

import pysimlr


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

    # no hidden state: same input -> same output, in any call order
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
    # all-negative column is flipped to the positive orthant
    assert np.allclose(pysimlr.UniformShrinkageOperator(3, alpha=1.0) @ np.array([-3.0, -3.0, 0.0]), np.full(3, 2.0))

    op_grp = pysimlr.UniformShrinkageOperator(5, alpha=1.0, groups=[0, 0, 1, 1, -1], positive_part=False)
    v_grp = np.array([1.0, 3.0, 10.0, 20.0, 99.0])
    assert np.allclose(op_grp @ v_grp, [2.0, 2.0, 15.0, 15.0, 99.0])


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
        pysimlr.uniform_shrinkage_params(v_tr, groups=[[0, 1]])  # wrong length for a 10-feature view
    with pytest.raises(ValueError, match="positivity"):
        pysimlr.uniform_shrinkage_params(v_tr, positivity="either")


def test_auto_uniform_alpha_asymptotics() -> None:
    a_small_n = pysimlr.auto_uniform_alpha(n_obs=50, n_features=100)
    a_equal = pysimlr.auto_uniform_alpha(n_obs=100, n_features=100)
    a_large_n = pysimlr.auto_uniform_alpha(n_obs=1000, n_features=100)
    assert a_small_n > a_equal > a_large_n
    assert np.isclose(a_equal, 0.5)
    assert np.isclose(a_large_n, 100.0 / 1100.0)


def test_auto_alpha_counts_rows_with_any_finite_feature() -> None:
    """Scattered NaNs must not inflate alpha: alpha equals that of the same data with NaNs imputed."""
    rng = np.random.default_rng(3)
    clean = rng.normal(size=(60, 40))
    holes = clean.copy()
    holes[rng.random(holes.shape) < 0.05] = np.nan
    assert not np.isfinite(holes).all(axis=1).all()  # many rows have at least one NaN
    a_clean = pysimlr.create_uniform_smoother(clean).alpha
    a_holes = pysimlr.create_uniform_smoother(holes).alpha
    assert a_holes == a_clean
    # a fully missing row is not an observation
    holes[0, :] = np.nan
    assert pysimlr.create_uniform_smoother(holes).alpha > a_clean


def test_resolved_defaults_are_echoed_in_provenance() -> None:
    v_tr, _, _ = _planted(0, 30, (20, 25))
    kw = pysimlr.uniform_shrinkage_params(v_tr)
    prov = kw.provenance
    assert prov["scale_list"] == ["none"] and prov["energy_type"] == "recon_r2" and prov["mixing_algorithm"] == "svd"
    assert prov["nsa_w"] == kw["nsa_w"] == 0.2 and prov["nsa_w_requested"] == "auto"
    assert prov["n_obs"] == [30, 30] and prov["n_features"] == [20, 25]
    assert np.allclose(prov["alpha"], [20 / 50, 25 / 55])
    assert "provenance" not in kw  # only fit kwargs travel through **kw
    kw2 = pysimlr.uniform_shrinkage_params(v_tr, nsa_w=0.5, energy_type="reconorm", sparseness=None)
    assert kw2["nsa_w"] == 0.5 and kw2["energy_type"] == "reconorm" and kw2.provenance["energy_type"] == "reconorm"
    assert pysimlr.uniform_shrinkage_params(v_tr, sparseness=None)["nsa_w"] == 0.0


def test_k_greater_than_one_requires_groups_and_small_alpha() -> None:
    v_tr, _, _ = _planted(0, 40, (30, 30))
    with pytest.raises(ValueError, match="k=2"):
        pysimlr.uniform_shrinkage_params(v_tr, k=2)  # no groups
    with pytest.raises(ValueError, match="alpha"):
        pysimlr.uniform_shrinkage_params(v_tr, k=2, alpha=0.5, groups=[[0, 1] * 15] * 2)  # groups, alpha too large
    kw = pysimlr.uniform_shrinkage_params(v_tr, k=2, alpha=0.1, groups=[[0, 1] * 15] * 2)
    assert all(v.shape[1] == 2 for v in kw["initial_v"])
    assert pysimlr.uniform_shrinkage_params(v_tr, k=1).provenance["k"] == 1


def test_alpha_one_reproduces_plain_mean_exactly() -> None:
    """Identity by construction: alpha=1 makes the fitted direction uniform, so the score is the plain mean."""
    v_tr, v_te, _ = _planted(1, 45, (60, 80, 50), n_te=80)
    kw = pysimlr.uniform_shrinkage_params(v_tr, alpha=1.0, clip=(0.0, 1.0), sparseness=None)
    fit = pysimlr.fit("simlr", v_tr, 1, random_state=0, iterations=25, **kw)
    proj = pysimlr.project_uniform_shrunk(fit, v_te, post_shrink=0.0)
    plain = np.mean([v.mean(axis=1) for v in v_te], axis=0)
    assert np.allclose(proj["u"][:, 0], plain, atol=1e-5)


def test_alpha_zero_is_close_to_but_not_the_plain_fit() -> None:
    """alpha=0 is the identity smoother but not bit-identical to a plain `simlr` fit
    (nsa_w, initial_v and positive_part differ); fitted weights stay highly aligned (cosine >= 0.95)."""
    v_tr, _, _ = _planted(2, 150, (30, 30))
    kw = pysimlr.uniform_shrinkage_params(v_tr, alpha=0.0, sparseness=None)
    fit0 = pysimlr.fit("simlr", v_tr, 1, random_state=0, iterations=60, **kw)
    plain = pysimlr.fit("simlr", v_tr, 1, scale_list=["none"], energy_type="recon_r2", mixing_algorithm="svd",
                        positivity="conical", sparseness=None, random_state=0, iterations=60)
    for a, b in zip(fit0["v"], plain["v"]):
        a, b = a.detach().numpy().ravel(), b.detach().numpy().ravel()
        cos = abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
        assert cos >= 0.95


def test_shrinkage_gain_small_n_and_no_loss_large_n() -> None:
    """Paired by seed, equal sparseness (None), same scoring. n<p: mean gain >= 0.03 (observed ~0.05,
    every seed better). n>>p: mean change >= -0.01."""
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
    v_tr, v_te, z = _planted(4, 40, (30, 30))
    for v in (v_tr[0], v_te[0]):
        v[:, 3] = 0.0  # a constant column after standardization
    kw = pysimlr.uniform_shrinkage_params(v_tr, alpha="auto", sparseness=None)
    fit = pysimlr.fit("simlr", v_tr, 1, random_state=0, iterations=20, **kw)
    proj = pysimlr.project_uniform_shrunk(fit, v_te)
    assert np.isfinite(proj["u"]).all()
    assert all(np.isfinite(w).all() and np.isclose(w.sum(), 1.0) for w in proj["weights"])

    # constant column inside the operator: no NaN
    op = pysimlr.UniformShrinkageOperator(6, alpha=0.5)
    assert np.isfinite(op @ np.zeros((6, 2))).all()

    # single view: the projector runs and u equals the view score
    kw1 = pysimlr.uniform_shrinkage_params(v_tr[:1], alpha=0.3, sparseness=None)
    assert len(kw1["smoothing_matrices"]) == 1
    proj1 = pysimlr.project_uniform_shrunk(
        {"v": [np.full((30, 1), 1 / 30.0)]}, v_te[:1], post_shrink=0.0)
    assert np.allclose(proj1["u"][:, 0], v_te[0].mean(axis=1))


def test_deterministic_for_fixed_random_state() -> None:
    v_tr, v_te, _ = _planted(5, 40, (50, 60))
    outs = []
    for _ in range(2):
        kw = pysimlr.uniform_shrinkage_params(v_tr, alpha="auto", sparseness=0.3)
        fit = pysimlr.fit("simlr", v_tr, 1, random_state=3, iterations=30, **kw)
        outs.append(pysimlr.project_uniform_shrunk(fit, v_te)["u"])
    assert np.array_equal(outs[0], outs[1])


def test_mixed_sign_planted_signal_shows_why_default_is_off() -> None:
    """Unoriented features (half +0.6, half -0.6) cancel in the mean: uniform shrinkage loses the signal."""
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
    kw = pysimlr.uniform_shrinkage_params(views, alpha=0.85, sparseness=None)
    fit_shr = pysimlr.fit("simlr", views, 1, random_state=0, iterations=60, **kw)
    r_shr = abs(float(np.corrcoef(pysimlr.project_uniform_shrunk(fit_shr, views)["u"][:, 0], z[:, 0])[0, 1]))
    assert r_def > 0.85
    assert r_def > r_shr + 0.15


def test_flipped_sign_view_gives_negated_score() -> None:
    """Orientation is a user contract: flipping a view's sign negates its score; nothing checks it."""
    v_tr, v_te, _ = _planted(6, 60, (20, 20))
    kw = pysimlr.uniform_shrinkage_params(v_tr, alpha=0.3, sparseness=None)
    fit = pysimlr.fit("simlr", v_tr, 1, random_state=0, iterations=20, **kw)
    base = pysimlr.project_uniform_shrunk(fit, v_te, post_shrink=0.0)["view_scores"]
    flipped = pysimlr.project_uniform_shrunk(fit, [-v_te[0], v_te[1]], post_shrink=0.0)["view_scores"]
    assert np.allclose(flipped[:, 0], -base[:, 0])
    assert np.allclose(flipped[:, 1], base[:, 1])
