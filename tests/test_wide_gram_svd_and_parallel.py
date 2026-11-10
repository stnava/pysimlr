"""Contract and regression tests for:
1. Wide-matrix (p >= 4n) float64 Gram eigh fast-path in `ba_svd` / `safe_pca` / `whiten_matrix`.
2. `simlr_perm` `n_jobs` multi-process execution and `SeedSequence` reproducibility.
3. Single-batch `(candidate, k, subsample)` task flattening in `select_model`.
"""
import importlib
import warnings

import numpy as np
import pytest
import torch

import pysimlr
from pysimlr import simlr_perm
from pysimlr.svd import ba_svd, safe_pca, top_right_singular_vectors, whiten_matrix
from pysimlr.utils import safe_svd

loadings_mod = importlib.import_module("pysimlr.loadings")


# ---------------------------------------------------------------------------
# 1. Wide-matrix float64 Gram SVD / PCA / whitening (p >= 4n)
# ---------------------------------------------------------------------------
def test_wide_ba_svd_matches_full_svd_subspace_and_reconstruction():
    g = torch.Generator().manual_seed(0)
    z = torch.randn(40, 5, generator=g)
    x = z @ torch.randn(5, 320, generator=g) + 0.3 * torch.randn(40, 320, generator=g)
    assert x.shape[1] >= 4 * x.shape[0]

    u, s, v = ba_svd(x, nu=5, nv=5)
    u_ref, s_ref, vh_ref = safe_svd(x, full_matrices=False)
    v_ref = vh_ref[:5].t()

    assert u.shape == (40, 5)
    assert s.shape == (5,)
    assert v.shape == (320, 5)
    assert torch.allclose(s, s_ref[:5], rtol=1e-5, atol=1e-5)
    assert torch.allclose(u.t() @ u, torch.eye(5), atol=1e-5)
    assert torch.allclose(v.t() @ v, torch.eye(5), atol=1e-5)
    assert float(torch.linalg.svdvals(v.double().t() @ v_ref.double()).min()) > 1.0 - 1e-5
    assert float(torch.linalg.svdvals(u.double().t() @ u_ref[:, :5].double()).min()) > 1.0 - 1e-5
    # Sign consistency between U and V: U @ diag(S) @ V.T matches rank-5 truncated SVD of X
    recon_gram = (u.double() * s.double().unsqueeze(0)) @ v.double().t()
    recon_ref = (u_ref[:, :5].double() * s_ref[:5].double().unsqueeze(0)) @ v_ref.double().t()
    assert torch.allclose(recon_gram, recon_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("nu,nv", [(0, 4), (4, 0), (0, 0), (3, 5), (5, 2)])
def test_wide_ba_svd_asymmetric_nu_nv_and_top_right(nu, nv):
    g = torch.Generator().manual_seed(1)
    x = torch.randn(25, 160, generator=g)
    u, s, v = ba_svd(x, nu=nu, nv=nv)
    assert u.shape == (25, nu)
    assert v.shape == (160, nv)
    assert s.shape == (min(nu, nv),)
    if nu > 0:
        assert torch.allclose(u.t() @ u, torch.eye(nu), atol=1e-5)
    if nv > 0:
        assert torch.allclose(v.t() @ v, torch.eye(nv), atol=1e-5)
    vr = top_right_singular_vectors(x, 4)
    _, _, v4 = ba_svd(x, nu=0, nv=4)
    assert torch.equal(vr, v4)


def test_wide_ba_svd_rank_deficient_zero_and_single_row_cases():
    # Rank-0 all-zeros wide matrix
    x_zero = torch.zeros(12, 80)
    u0, s0, v0 = ba_svd(x_zero, nu=4, nv=4)
    assert torch.equal(s0, torch.zeros(4))
    assert torch.allclose(u0.t() @ u0, torch.eye(4), atol=1e-6)
    assert torch.allclose(v0.t() @ v0, torch.eye(4), atol=1e-6)

    # Rank-3 matrix of shape (16, 100) with nu=6, nv=6 requested
    g = torch.Generator().manual_seed(2)
    a = torch.randn(16, 3, generator=g)
    b = torch.randn(3, 100, generator=g)
    x_low = a @ b
    u, s, v = ba_svd(x_low, nu=6, nv=6)
    assert u.shape == (16, 6) and v.shape == (100, 6) and s.shape == (6,)
    assert (s[:3] > 1.0).all()
    assert torch.allclose(s[3:], torch.zeros(3), atol=1e-5)
    assert torch.allclose(u.t() @ u, torch.eye(6), atol=1e-5)
    assert torch.allclose(v.t() @ v, torch.eye(6), atol=1e-5)
    assert torch.allclose((u * s.unsqueeze(0)) @ v.t(), x_low, atol=1e-4)

    # Single-sample N=1 wide matrix
    x_row = torch.randn(1, 50, generator=g)
    ur, sr, vr = ba_svd(x_row, nu=1, nv=1)
    assert ur.shape == (1, 1) and sr.shape == (1,) and vr.shape == (50, 1)
    assert torch.allclose((ur * sr) @ vr.t(), x_row, atol=1e-5)

    # Full-width whitening on wide matrix (centered rank is n - 1 = 29)
    x_wh = torch.randn(30, 200, generator=g)
    wh = whiten_matrix(x_wh)
    assert wh["rank"] == 29
    w = wh["whitened_matrix"]
    assert torch.equal(w[:, 29], torch.zeros(30))
    wc = w[:, :29] - w[:, :29].mean(0, keepdim=True)
    cov = wc.t() @ wc / 29.0
    assert torch.allclose(cov, torch.eye(29), atol=1e-4)


def test_wide_ba_svd_ill_conditioned_and_eigh_error_fall_back_accurately(monkeypatch):
    # Construct wide matrix with singular values below the 1e-4 * s_0 Gram guard
    g = torch.Generator().manual_seed(3)
    u_true, _ = torch.linalg.qr(torch.randn(10, 4, dtype=torch.float64, generator=g))
    v_true, _ = torch.linalg.qr(torch.randn(80, 4, dtype=torch.float64, generator=g))
    spec = torch.tensor([100.0, 10.0, 1e-3, 2e-4], dtype=torch.float64)
    x = ((u_true * spec) @ v_true.t()).float()
    u, s, v = ba_svd(x, nu=4, nv=4)
    u_ref, s_ref, vh_ref = safe_svd(x, full_matrices=False)
    assert torch.equal(u, u_ref[:, :4])
    assert torch.equal(s, s_ref[:4])
    assert torch.equal(v, vh_ref.t()[:, :4])
    assert float(v[:, 3].dot(v_true[:, 3].float()).abs()) > 0.99

    # If torch.linalg.eigh raises RuntimeError, ba_svd falls back cleanly to safe_svd
    def broken_eigh(_g):
        raise RuntimeError("synthetic eigh failure")

    monkeypatch.setattr(torch.linalg, "eigh", broken_eigh)
    x_well = torch.randn(12, 72, generator=g)
    uw, sw, vw = ba_svd(x_well, nu=3, nv=3)
    uw_ref, sw_ref, vhw_ref = safe_svd(x_well, full_matrices=False)
    assert torch.equal(uw, uw_ref[:, :3])
    assert torch.equal(sw, sw_ref[:3])
    assert torch.equal(vw, vhw_ref.t()[:, :3])


# ---------------------------------------------------------------------------
# 2. simlr_perm: n_jobs + SeedSequence reproducibility and RNG isolation
# ---------------------------------------------------------------------------
def test_simlr_perm_reproducible_and_independent_of_n_jobs():
    rng = np.random.default_rng(5)
    z = rng.standard_normal((45, 2)).astype(np.float32)
    views = [
        z @ rng.standard_normal((2, 12)).astype(np.float32) + 0.3 * rng.standard_normal((45, 12)).astype(np.float32),
        z @ rng.standard_normal((2, 10)).astype(np.float32) + 0.3 * rng.standard_normal((45, 10)).astype(np.float32),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.manual_seed(99)
        expect_next = torch.randperm(8).tolist()

        torch.manual_seed(99)
        r1 = simlr_perm(views, k=2, n_perms=4, iterations=5, learning_rate=0.05, random_state=17, n_jobs=1)
        got_next = torch.randperm(8).tolist()
        assert got_next == expect_next, "seeded simlr_perm disturbed caller's global torch RNG"

        r1_repeat = simlr_perm(views, k=2, n_perms=4, iterations=5, learning_rate=0.05, random_state=17, n_jobs=1)
        r2_diff_seed = simlr_perm(views, k=2, n_perms=4, iterations=5, learning_rate=0.05, random_state=18, n_jobs=1)

    st1 = r1["stats"]["sim_0_1"]
    st1_rep = r1_repeat["stats"]["sim_0_1"]
    st_diff = r2_diff_seed["stats"]["sim_0_1"]
    assert st1["observed"] == st1_rep["observed"]
    assert st1["null_distribution"] == st1_rep["null_distribution"]
    assert st1["p_value"] == st1_rep["p_value"]
    assert st1["null_distribution"] != st_diff["null_distribution"]

    with pytest.raises(ValueError, match="n_jobs"):
        simlr_perm(views, k=2, n_perms=2, n_jobs=0)
    with pytest.raises(ValueError, match="n_jobs"):
        simlr_perm(views, k=2, n_perms=2, n_jobs=-2)


def test_simlr_perm_does_not_depend_on_n_jobs():
    rng = np.random.default_rng(8)
    views = [
        rng.standard_normal((40, 10)).astype(np.float32),
        rng.standard_normal((40, 8)).astype(np.float32),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = simlr_perm(views, k=2, n_perms=4, iterations=5, learning_rate=0.05, random_state=21, n_jobs=1)
        b = simlr_perm(views, k=2, n_perms=4, iterations=5, learning_rate=0.05, random_state=21, n_jobs=2)
    assert torch.equal(a["simlr_result"]["u"], b["simlr_result"]["u"])
    assert a["stats"] == b["stats"]


# ---------------------------------------------------------------------------
# 3. select_model: single-batch _run_fits flattening + exact parity
# ---------------------------------------------------------------------------
def test_select_model_batches_all_cells_into_single_run_fits_call(monkeypatch):
    rng = np.random.default_rng(19)
    z = rng.standard_normal((60, 2))
    views = [
        torch.tensor(z @ rng.standard_normal((2, p)) + 0.4 * rng.standard_normal((60, p))).float()
        for p in (14, 12, 10)
    ]
    candidates = [
        {"label": "pca", "method": "pca", "params": {}},
        {
            "label": "simlr_lin",
            "method": "simlr",
            "params": {"positivity": "linear", "iterations": 6, "learning_rate": 0.05},
        },
    ]
    real_run_fits = loadings_mod._run_fits
    batch_sizes = []

    def spy_run_fits(tasks, n_jobs, executor=None):
        batch_sizes.append(len(tasks))
        return real_run_fits(tasks, n_jobs, executor=executor)

    monkeypatch.setattr(loadings_mod, "_run_fits", spy_run_fits)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = pysimlr.select_model(
            views, candidates, [1, 2], n_subsamples=3, fraction=0.7, exclude_unconverged=False, random_state=0, n_jobs=1
        )

    # 2 candidates * 2 ks = 4 cells, each with 1 ref fit + 3 subsample refits = 16 tasks in ONE call
    assert batch_sizes == [16], f"expected single flattened batch of 16 tasks, got {batch_sizes}"

    # Verify every cell matches standalone select_rank bit-for-bit
    tbl = out["table"]
    for cand in candidates:
        for k in (1, 2):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sr = pysimlr.select_rank(
                    cand["method"],
                    views,
                    [k],
                    n_subsamples=3,
                    fraction=0.7,
                    exclude_unconverged=False,
                    random_state=0,
                    **cand["params"],
                )["table"][0]
            row = tbl[(tbl["label"] == cand["label"]) & (tbl["k"] == k)].iloc[0]
            for col in (
                "component_stability",
                "subspace_stability",
                "oos_r2",
                "oos_r2_se",
                "oos_coupling",
                "converged_fraction",
                "n_effective_refits",
                "min_effective_rank",
                "basis_degenerate",
            ):
                assert row[col] == sr[col], (cand["label"], k, col, row[col], sr[col])


def test_select_model_does_not_depend_on_n_jobs():
    rng = np.random.default_rng(31)
    z = rng.standard_normal((60, 2))
    views = [
        torch.tensor(z @ rng.standard_normal((2, p)) + 0.4 * rng.standard_normal((60, p))).float()
        for p in (14, 12, 10)
    ]
    candidates = [
        {"label": "pca", "method": "pca", "params": {}},
        {
            "label": "simlr_lin",
            "method": "simlr",
            "params": {"positivity": "linear", "iterations": 6, "learning_rate": 0.05},
        },
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = pysimlr.select_model(views, candidates, [1, 2], n_subsamples=3, fraction=0.7, random_state=0, n_jobs=1)
        b = pysimlr.select_model(views, candidates, [1, 2], n_subsamples=3, fraction=0.7, random_state=0, n_jobs=2)
    def _eq_nan(x, y):
        if isinstance(x, float) and isinstance(y, float) and np.isnan(x) and np.isnan(y):
            return True
        if isinstance(x, list) and isinstance(y, list):
            return len(x) == len(y) and all(_eq_nan(xi, yi) for xi, yi in zip(x, y))
        if isinstance(x, dict) and isinstance(y, dict):
            return x.keys() == y.keys() and all(_eq_nan(x[k], y[k]) for k in x)
        return x == y

    assert a["signal_detected"] == b["signal_detected"]
    assert _eq_nan(a["selected"], b["selected"])
    assert _eq_nan(a["table"].to_dict(orient="list"), b["table"].to_dict(orient="list"))
