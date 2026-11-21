"""Tests for monotone RQS splines, PCF Jacobian regularization, GaussianPCA base,
and shared post-flow Varimax gauge rotation in `pysimlr.flows` (`D63`)."""

import math
import warnings

import numpy as np
import pytest
import torch

import pysimlr
from pysimlr import predict_deep
from pysimlr.flows import (
    FLOW_BASE_DISTRIBUTIONS,
    FLOW_TYPES,
    ActNormElementwiseRQSplineFlow,
    ElementwiseRQSplineLayer,
    FlowSiMRModel,
    FlowSiMRVModel,
    GaussianPCABase,
    NormalizingFlow,
    PositiveActNorm1d,
    exact_pcf_penalty_bpd,
    flow_simr,
    flow_simr_v,
    resolve_flow_type,
)
from pysimlr.loadings import structure_coefficients


def _kaiser_varimax_score(structure_list) -> float:
    mat = torch.cat([s.detach().double() for s in structure_list], dim=0)
    row_norms = mat.norm(dim=1, keepdim=True).clamp_min(1e-12)
    h = mat / row_norms
    u2 = h.pow(2)
    return float((u2.pow(2).mean(dim=0) - u2.mean(dim=0).pow(2)).sum().item())


def _offdiag_jacobian_coherence(flow_module, x: torch.Tensor) -> float:
    xr = x.detach().float().requires_grad_(True)
    z, _ = flow_module.forward_and_log_det(xr)
    d = int(z.shape[1])
    rows = []
    for j in range(d):
        gj = torch.autograd.grad(z[:, j].sum(), xr, retain_graph=True)[0]
        rows.append(gj / gj.norm(dim=1, keepdim=True).clamp_min(1e-12))
    j_unit = torch.stack(rows, dim=1)  # (N, d, d)
    gram = torch.bmm(j_unit, j_unit.transpose(1, 2))
    eye = torch.eye(d, device=gram.device, dtype=gram.dtype).unsqueeze(0)
    return float(((gram - eye).pow(2).sum(dim=(1, 2)) / (d * (d - 1))).mean().item())


# ---------------------------------------------------------------------------
# WP-Flow-1: Monotone RQS Spline + PositiveActNorm1d + flow_type="auto" at k=1
# ---------------------------------------------------------------------------


def test_resolve_flow_type_validation_and_auto_dispatch():
    assert FLOW_TYPES == ("auto", "realnvp", "spline1d")
    assert resolve_flow_type("auto", 1) == "spline1d"
    assert resolve_flow_type("auto", 2) == "realnvp"
    assert resolve_flow_type("auto", 5) == "realnvp"
    assert resolve_flow_type("realnvp", 1) == "realnvp"
    assert resolve_flow_type("spline1d", 3) == "spline1d"
    with pytest.raises(ValueError, match="flow_type"):
        resolve_flow_type("invalid", 2)
    with pytest.raises(ValueError, match="k must be >= 1"):
        resolve_flow_type("auto", 0)


@pytest.mark.parametrize("dim", [1, 3])
def test_actnorm_rqspline_bijection_logdet_monotonicity_and_zero_pcf(dim: int):
    gen = torch.Generator().manual_seed(17 + dim)
    x = torch.randn(32, dim, generator=gen) * 2.5 + 0.7  # includes interior and tail points

    flow = ActNormElementwiseRQSplineFlow(dim=dim, num_layers=2, num_bins=8, tail_bound=3.5, scale_bound=1.5)
    flow.train()
    assert flow.use_fallback is False
    # Trigger PositiveActNorm1d initialization, then perturb spline parameters away from identity
    _ = flow.forward_and_log_det(x)
    assert bool(flow.actnorm.initialized.item()) is True
    with torch.no_grad():
        for idx, layer in enumerate(flow.layers):
            layer.unnorm_widths.add_(0.4 * torch.randn_like(layer.unnorm_widths))
            layer.unnorm_heights.add_(-0.35 * torch.randn_like(layer.unnorm_heights))
            layer.unnorm_derivs.add_(0.25 * (idx + 1))
            layer.raw_log_scale.fill_(0.15)
            layer.shift.fill_(-0.1)
    flow.eval()

    z, ld = flow.forward_and_log_det(x)
    x_rec, ld_inv = flow.inverse_and_log_det(z)
    assert torch.allclose(x_rec, x, atol=1e-4, rtol=1e-4)
    assert torch.allclose(ld + ld_inv, torch.zeros_like(ld), atol=1e-4)

    # Exact autograd Jacobian check on batch subset
    for i in range(6):
        xi = x[i : i + 1].detach().clone().requires_grad_(True)
        jac = torch.autograd.functional.jacobian(lambda inp: flow.forward_and_log_det(inp)[0].squeeze(0), xi.squeeze(0))
        sign, logabsdet = torch.linalg.slogdet(jac)
        assert float(sign.item()) == pytest.approx(1.0)
        assert float(logabsdet.item()) == pytest.approx(float(ld[i].item()), abs=2e-4)
        # Strictly positive diagonal, strictly zero off-diagonal
        diag = torch.diagonal(jac)
        assert bool((diag > 0).all())
        if dim > 1:
            off = jac - torch.diag_embed(diag)
            assert float(off.abs().max().item()) < 1e-6

    # Monotonicity along each coordinate (Spearman == 1 -> strictly increasing grid)
    grid = torch.linspace(-5.0, 5.0, 64).unsqueeze(1).repeat(1, dim)
    z_grid, _ = flow.forward_and_log_det(grid)
    diffs = z_grid[1:] - z_grid[:-1]
    assert bool((diffs > 0).all())
    # Genuine curvature (non-zero second difference) even at dim == 1
    second_diff = (z_grid[2:] - 2.0 * z_grid[1:-1] + z_grid[:-2]).abs().max().item()
    assert second_diff > 0.02

    # PCF penalty is identically 0.0 for diagonal spline flows
    pcf_val = exact_pcf_penalty_bpd(flow, x)
    assert float(pcf_val.item()) == 0.0

    # State dict round-trip preserves PositiveActNorm1d initialized buffer
    clone = ActNormElementwiseRQSplineFlow(dim=dim, num_layers=2, num_bins=8, tail_bound=3.5, scale_bound=1.5)
    clone.load_state_dict(flow.state_dict())
    clone.eval()
    assert bool(clone.actnorm.initialized.item()) is True
    z_clone, ld_clone = clone.forward_and_log_det(x)
    assert torch.allclose(z_clone, z, atol=1e-6)
    assert torch.allclose(ld_clone, ld, atol=1e-6)


def test_flow_v_auto_uses_spline1d_at_k1_and_preserves_exact_realnvp_at_k_ge_2():
    rng = np.random.default_rng(101)
    n_tr, n_te = 120, 60
    s_all = rng.standard_normal(n_tr + n_te)
    # Strongly skewed monotone rank-1 shared warp on view 0, cubic monotone warp on view 1
    w0 = np.exp(0.85 * s_all) - np.exp(0.85 ** 2 / 2.0)
    w1 = s_all + 0.35 * (s_all ** 3)
    a0 = rng.standard_normal((1, 8))
    a1 = rng.standard_normal((1, 8))
    x0 = torch.tensor(w0[:, None] @ a0 + 0.15 * rng.standard_normal((n_tr + n_te, 8)), dtype=torch.float32)
    x1 = torch.tensor(w1[:, None] @ a1 + 0.15 * rng.standard_normal((n_tr + n_te, 8)), dtype=torch.float32)
    tr_views = [x0[:n_tr], x1[:n_tr]]
    te_views = [x0[n_tr:], x1[n_tr:]]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_auto_k1 = pysimlr.fit("flow_v", tr_views, k=1, iterations=60, batch_size=32, random_state=7)
        res_rnvp_k1 = pysimlr.fit(
            "flow_v", tr_views, k=1, iterations=60, batch_size=32, random_state=7, flow_type="realnvp"
        )
    assert res_auto_k1["flow_type"] == "spline1d"
    assert res_auto_k1["flow_type_requested"] == "auto"
    assert isinstance(res_auto_k1["model"].flows[0], ActNormElementwiseRQSplineFlow)
    assert res_rnvp_k1["flow_type"] == "realnvp"
    assert isinstance(res_rnvp_k1["model"].flows[0], NormalizingFlow)

    pred_sp = pysimlr.predict(res_auto_k1, te_views)
    pred_rn = pysimlr.predict(res_rnvp_k1, te_views)
    mean_r2_sp = float(np.mean(pred_sp["r2"]))
    mean_r2_rn = float(np.mean(pred_rn["r2"]))
    # At k=1, RealNVP is strictly affine while spline1d captures the monotone warp
    assert mean_r2_sp > mean_r2_rn + 0.03, (
        f"Expected spline1d k=1 R2 ({mean_r2_sp:.4f}) to beat affine RealNVP k=1 ({mean_r2_rn:.4f})"
    )

    # At k=2, flow_type='auto' must be 100% bit-identical to flow_type='realnvp'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_auto_k2 = pysimlr.fit("flow_v", tr_views, k=2, iterations=8, batch_size=32, random_state=5)
        res_rnvp_k2 = pysimlr.fit(
            "flow_v", tr_views, k=2, iterations=8, batch_size=32, random_state=5, flow_type="realnvp"
        )
        res_sp_k2 = pysimlr.fit(
            "flow_v", tr_views, k=2, iterations=8, batch_size=32, random_state=5, flow_type="spline1d"
        )
    assert res_auto_k2["flow_type"] == "realnvp"
    assert torch.equal(res_auto_k2["u"], res_rnvp_k2["u"])
    for va, vb in zip(res_auto_k2["v"], res_rnvp_k2["v"]):
        assert torch.equal(va, vb)
    assert not torch.equal(res_auto_k2["u"], res_sp_k2["u"])


# ---------------------------------------------------------------------------
# WP-Flow-2: Opt-in Principal Component Flow penalty (`pcf_alpha`) on flow_v
# ---------------------------------------------------------------------------


def test_pcf_alpha_validation_and_jacobian_orthogonality_on_flow_v():
    rng = np.random.default_rng(23)
    n = 64
    z = rng.standard_normal((n, 3))
    views = [
        torch.tensor(z @ rng.standard_normal((3, 10)) + 0.2 * rng.standard_normal((n, 10)), dtype=torch.float32),
        torch.tensor(z @ rng.standard_normal((3, 9)) + 0.2 * rng.standard_normal((n, 9)), dtype=torch.float32),
    ]

    for bad in (-0.1, float("nan"), float("inf"), True, "0.1"):
        with pytest.raises(ValueError, match="pcf_alpha"):
            flow_simr_v(views, k=3, iterations=2, pcf_alpha=bad)  # type: ignore[arg-type]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res0 = flow_simr_v(views, k=3, iterations=30, batch_size=32, warmup_epochs=5, pcf_alpha=0.0, random_state=9)
        res1 = flow_simr_v(views, k=3, iterations=30, batch_size=32, warmup_epochs=5, pcf_alpha=0.25, random_state=9)

    assert res0["pcf_alpha"] == 0.0
    assert res1["pcf_alpha"] == pytest.approx(0.25)
    assert not torch.equal(res0["u"], res1["u"])
    assert not torch.equal(res0["v"][0], res1["v"][0])

    # Stationary post-warmup loss invariant (`best_loss == min(loss_history[warmup:])`)
    w = res1["convergence"]["warmup_epochs"]
    assert res1["convergence"]["best_loss"] == pytest.approx(min(res1["loss_history"][w:]), rel=1e-6)

    # Encoder Jacobian off-diagonal coherence on standardized projections decreases under pcf_alpha > 0
    xs0, _ = pysimlr.api._preprocess_views(views, res0["scale_list"], res0["provenance_list"])
    coh0 = np.mean([
        _offdiag_jacobian_coherence(res0["model"].flows[m], xs0[m] @ res0["v"][m])
        for m in range(len(views))
    ])
    coh1 = np.mean([
        _offdiag_jacobian_coherence(res1["model"].flows[m], xs0[m] @ res1["v"][m])
        for m in range(len(views))
    ])
    assert coh1 < coh0, f"Expected pcf_alpha=0.25 to lower Jacobian coherence ({coh1:.5f} vs {coh0:.5f})"

    # Composes cleanly with missing_rows='mask'
    masked = [v.clone() for v in views]
    masked[0][:8] = float("nan")
    masked[1][10:16] = float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_mask = pysimlr.fit(
            "flow_v", masked, k=2, iterations=12, batch_size=24,
            missing_rows="mask", pcf_alpha=0.1, random_state=4,
        )
    assert bool(torch.isfinite(res_mask["u"]).all())
    assert bool(torch.isnan(res_mask["latents"][0][:8]).all())


# ---------------------------------------------------------------------------
# WP-Flow-3: flow_simr scale_bound=0.5 + GaussianPCABase + shared O(k) Varimax
# ---------------------------------------------------------------------------


def test_gaussian_pca_base_exact_multivariate_normal_parity_including_k_gt_d():
    assert FLOW_BASE_DISTRIBUTIONS == ("diag", "gaussian_pca")
    for dim, k in ((8, 3), (4, 4), (4, 7)):
        torch.manual_seed(30 + dim + k)
        gpca = GaussianPCABase(dim=dim, latent_dim=k, sigma=0.2)
        with torch.no_grad():
            gpca.loc.copy_(0.3 * torch.randn(1, dim))
            gpca.W.copy_(0.4 * torch.randn(k, dim))
        z = torch.randn(20, dim)
        lp = gpca.log_prob(z)

        # Reference full D x D MultivariateNormal in float64
        loc64 = gpca.loc.double().squeeze(0)
        w64 = gpca.W.double()
        s2_64 = gpca._clamped_sigma2().double()
        cov64 = w64.t() @ w64 + s2_64 * torch.eye(dim, dtype=torch.float64)
        mvn = torch.distributions.MultivariateNormal(loc=loc64, covariance_matrix=cov64)
        lp_ref = mvn.log_prob(z.double()).float()
        assert torch.allclose(lp, lp_ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("base_distribution", ["diag", "gaussian_pca"])
def test_flow_simr_varimax_preserves_exact_nll_reconstruction_and_subspace_while_localizing(
    base_distribution: str,
):
    rng = np.random.default_rng(77)
    n_tr, n_te, k = 80, 32, 3
    s_lat = rng.standard_normal((n_tr + n_te, k))
    # Rotate mixing matrices by a dense orthogonal mix so raw coordinates entangle factors
    q_mix, _ = np.linalg.qr(np.array([[1.0, 1.0, 0.8], [-0.9, 1.0, 0.5], [0.4, -0.8, 1.0]]))
    s_ent = s_lat @ q_mix
    # Disjoint block footprints in each view so Varimax has a distinct simple-structure gauge
    b0 = np.zeros((k, 12))
    b1 = np.zeros((k, 12))
    for j in range(k):
        b0[j, 4 * j : 4 * (j + 1)] = [1.4, -1.1, 0.9, 1.2]
        b1[j, 4 * j : 4 * (j + 1)] = [1.3, 1.0, -1.2, 0.8]
    x0 = torch.tensor(s_ent @ b0 + 0.15 * rng.standard_normal((n_tr + n_te, 12)), dtype=torch.float32)
    x1 = torch.tensor(s_ent @ b1 + 0.15 * rng.standard_normal((n_tr + n_te, 12)), dtype=torch.float32)
    tr_views, te_views = [x0[:n_tr], x1[:n_tr]], [x0[n_tr:], x1[n_tr:]]

    # Default scale_bound is now 0.5 on FlowSiMRModel and flow_simr
    import inspect
    assert inspect.signature(flow_simr).parameters["scale_bound"].default == 0.5
    assert inspect.signature(FlowSiMRModel.__init__).parameters["scale_bound"].default == 0.5

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_plain = flow_simr(
            tr_views, k=k, iterations=25, batch_size=32, warmup_epochs=4,
            base_distribution=base_distribution, varimax=False, random_state=12,
        )
        res_vmax = flow_simr(
            tr_views, k=k, iterations=25, batch_size=32, warmup_epochs=4,
            base_distribution=base_distribution, varimax=True, random_state=12,
        )

    assert res_plain["base_distribution"] == base_distribution
    assert res_plain["varimax"] is False and res_plain["varimax_rotation"] is None
    assert res_vmax["varimax"] is True
    r_orth = res_vmax["varimax_rotation"]
    assert r_orth is not None and r_orth.shape == (k, k)
    assert torch.allclose(r_orth.t() @ r_orth, torch.eye(k), atol=1e-5)
    assert all(f.use_fallback is False for f in res_vmax["model"].flows)

    # Individual latent coordinates change (non-trivial gauge rotation), while u_vmax == u_plain @ R
    assert not torch.allclose(res_plain["u"], res_vmax["u"], atol=1e-2)
    assert torch.allclose(res_plain["u"] @ r_orth, res_vmax["u"], atol=1e-4)

    # Exact reconstruction invariance both in-sample and out-of-sample (`predict_deep`)
    for r_a, r_b in zip(res_plain["reconstructions"], res_vmax["reconstructions"]):
        assert torch.allclose(r_a, r_b, atol=1e-4, rtol=1e-4)
    pred_plain = predict_deep(te_views, res_plain, device="cpu")
    pred_vmax = predict_deep(te_views, res_vmax, device="cpu")
    assert torch.allclose(pred_plain["u"] @ r_orth, pred_vmax["u"], atol=1e-4)
    for r_a, r_b in zip(pred_plain["reconstructions"], pred_vmax["reconstructions"]):
        assert torch.allclose(r_a, r_b, atol=1e-4, rtol=1e-4)

    # Exact flow log-likelihood invariance on held-out views
    xs_te, _ = pysimlr.api._preprocess_views(te_views, res_plain["scale_list"], res_plain["provenance_list"])
    with torch.no_grad():
        for m in range(2):
            za, lda = res_plain["model"].flows[m].forward_and_log_det(xs_te[m])
            zb, ldb = res_vmax["model"].flows[m].forward_and_log_det(xs_te[m])
            if base_distribution == "gaussian_pca":
                lpa = res_plain["model"].gpca_bases[m].log_prob(za) + lda
                lpb = res_vmax["model"].gpca_bases[m].log_prob(zb) + ldb
            else:
                lpa = -0.5 * za.pow(2).sum(dim=1) + lda
                lpb = -0.5 * zb.pow(2).sum(dim=1) + ldb
            assert torch.allclose(lpa, lpb, atol=1e-4, rtol=1e-4)

    # Conditional cross-view synthesis is invariant under the shared O(k) conjugation
    z1_plain = pred_plain["latents"][0]
    z1_vmax = pred_vmax["latents"][0]
    z2_cond_plain = res_plain["cond_inference"].predict_conditional(0, 1, z1_plain)
    z2_cond_vmax = res_vmax["cond_inference"].predict_conditional(0, 1, z1_vmax)
    x2_cond_plain = res_plain["model"].decoders[1](z2_cond_plain)
    x2_cond_vmax = res_vmax["model"].decoders[1](z2_cond_vmax)
    assert torch.allclose(x2_cond_plain, x2_cond_vmax, atol=1e-4, rtol=1e-4)

    # Stacked training structure Varimax simplicity increases or equals plain
    xs_tr, _ = pysimlr.api._preprocess_views(tr_views, res_plain["scale_list"], res_plain["provenance_list"])
    vm_plain = _kaiser_varimax_score(structure_coefficients(xs_tr, res_plain["u"]))
    vm_rot = _kaiser_varimax_score(structure_coefficients(xs_tr, res_vmax["u"]))
    assert vm_rot >= vm_plain - 1e-6


def test_flow_simr_argument_validation_and_masked_rows():
    rng = np.random.default_rng(5)
    views = [
        torch.tensor(rng.standard_normal((48, 8)), dtype=torch.float32),
        torch.tensor(rng.standard_normal((48, 6)), dtype=torch.float32),
    ]
    with pytest.raises(ValueError, match="base_distribution"):
        flow_simr(views, k=2, iterations=2, base_distribution="pca")
    with pytest.raises(ValueError, match="varimax on flow_simr must be a bool"):
        flow_simr(views, k=2, iterations=2, varimax=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="requires k <= min"):
        flow_simr(views, k=10, iterations=2, base_distribution="diag", varimax=True)

    # Masked rows + GaussianPCA + Varimax works end-to-end
    masked = [v.clone() for v in views]
    masked[0][:6] = float("nan")
    masked[1][8:13] = float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = flow_simr(
            masked, k=2, iterations=8, batch_size=24,
            missing_rows="mask", base_distribution="gaussian_pca", varimax=True, random_state=3,
        )
    assert bool(torch.isfinite(res["u"]).all())
    assert bool(torch.isnan(res["latents"][0][:6]).all())
    assert bool(torch.isfinite(res["latents"][0][6:]).all())
    pred = predict_deep(masked, res, device="cpu")
    assert bool(torch.isfinite(pred["u"]).all())


def test_predict_conditional_view_across_flow_v_and_flow_simr():
    from pysimlr.utils import invert_preprocessing

    rng = np.random.default_rng(31)
    n_tr, n_te = 72, 24
    s = rng.standard_normal((n_tr + n_te, 2))
    # Shifted/scaled raw input units to verify leak-free provenance + return_input_units
    x0 = torch.tensor(
        (s @ rng.standard_normal((2, 6)) + 0.15 * rng.standard_normal((n_tr + n_te, 6))) * 3.5 + 12.0,
        dtype=torch.float32,
    )
    x1 = torch.tensor(
        (s @ rng.standard_normal((2, 5)) + 0.15 * rng.standard_normal((n_tr + n_te, 5))) * 0.4 - 7.5,
        dtype=torch.float32,
    )
    tr_views, te_views = [x0[:n_tr], x1[:n_tr]], [x0[n_tr:], x1[n_tr:]]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_k1 = pysimlr.fit("flow_v", tr_views, k=1, iterations=12, batch_size=24, random_state=2)
        res_k2 = pysimlr.fit("flow_v", tr_views, k=2, iterations=12, batch_size=24, random_state=2)
        res_fs0 = flow_simr(
            tr_views, k=2, iterations=12, batch_size=24,
            base_distribution="gaussian_pca", varimax=False, random_state=5,
        )
        res_fs1 = flow_simr(
            tr_views, k=2, iterations=12, batch_size=24,
            base_distribution="gaussian_pca", varimax=True, random_state=5,
        )

    for fit_res, k_exp in ((res_k1, 1), (res_k2, 2), (res_fs0, 2)):
        out_std = pysimlr.predict_conditional_view(fit_res, te_views[0], observed_idx=0, target_idx=1)
        out_raw = pysimlr.predict_conditional_view(
            fit_res, te_views[0], observed_idx=0, target_idx=1, return_input_units=True
        )
        assert out_std["observed_latent"].shape == (n_te, k_exp)
        assert out_std["target_latent"].shape == (n_te, k_exp)
        assert out_std["target_standardized"].shape == (n_te, 5)
        assert torch.equal(out_std["target_standardized"], out_std["target_reconstruction"])
        expected_raw = invert_preprocessing(
            out_std["target_standardized"], fit_res["scale_list"], fit_res["provenance_list"][1]
        )
        assert torch.allclose(out_raw["target_reconstruction"], expected_raw, atol=1e-5)

    # Exact conditional-synthesis invariance under shared O(k) Varimax gauge rotation
    c0 = pysimlr.predict_conditional_view(res_fs0, te_views[0], observed_idx=0, target_idx=1, return_input_units=True)
    c1 = pysimlr.predict_conditional_view(res_fs1, te_views[0], observed_idx=0, target_idx=1, return_input_units=True)
    assert torch.allclose(c0["target_reconstruction"], c1["target_reconstruction"], atol=1e-4, rtol=1e-4)

    # Masked unobserved test rows propagate NaN cleanly without corrupting observed rows
    masked_tr = [v.clone() for v in tr_views]
    masked_tr[0][:5] = float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_m = pysimlr.fit("flow_v", masked_tr, k=2, iterations=8, batch_size=24, missing_rows="mask", random_state=3)
    te0_masked = te_views[0].clone()
    te0_masked[2:5] = float("nan")
    cm = pysimlr.predict_conditional_view(res_m, te0_masked, observed_idx=0, target_idx=1, return_input_units=True)
    assert bool(torch.isnan(cm["target_reconstruction"][2:5]).all())
    assert bool(torch.isfinite(cm["target_reconstruction"][5:]).all())

    # Validation checks
    with pytest.raises(ValueError, match="observed_idx"):
        pysimlr.predict_conditional_view(res_k2, te_views[0], observed_idx=3, target_idx=1)
    with pytest.raises(ValueError, match="columns"):
        pysimlr.predict_conditional_view(res_k2, te_views[1], observed_idx=0, target_idx=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lin_res = pysimlr.fit("simlr", tr_views, k=2, iterations=3, random_state=0)
    with pytest.raises(ValueError, match="fitted flow result"):
        pysimlr.predict_conditional_view(lin_res, te_views[0], observed_idx=0, target_idx=1)
