"""Tests for Phase 1 & Phase 2 SiMLR optimizer variants, feasible/spectral starts,
warm-started ``initial_v`` / ``sparseness_schedule``, provenance, and 0.40.0
bit-identical default preservation.
"""
import sys
import warnings

import numpy as np
import pytest
import torch

import pysimlr
from pysimlr import simlr
from pysimlr.optimizers import create_optimizer
from pysimlr.simlr import initial_basis_for_view, initialize_simlr


def _planted_rank1_views(seed=42, n=150, widths=(40, 35, 30)):
    rng = np.random.default_rng(seed)
    z_true = rng.standard_normal((n, 1))
    views = []
    for p in widths:
        z_priv = rng.standard_normal((n, 3)) * np.array([2.2, 1.6, 1.2])
        w_priv = rng.standard_normal((p, 3)) / np.sqrt(p)
        w_sh = np.zeros((p, 1))
        supp = rng.choice(p, size=max(3, round(0.2 * p)), replace=False)
        w_sh[supp, 0] = rng.choice([-1.0, 1.0], size=len(supp)) * 1.1
        views.append(z_true @ w_sh.T + z_priv @ w_priv.T + 0.5 * rng.standard_normal((n, p)))
    return z_true, views


# ---------------------------------------------------------------------------
# 1. Golden regression: defaults stay bit-identical to 0.40.0 legacy aliases
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "label,kw,k",
    [
        ("either", {"positivity": "either"}, 2),
        ("linear", {"positivity": "linear"}, 1),
        ("positive", {"positivity": "positive"}, 2),
        ("convex", {"positivity": "convex"}, 1),
        (
            "mixed",
            {
                "per_view_params": {
                    0: {"positivity": "linear"},
                    1: {"positivity": "conical"},
                    2: {"positivity": "convex"},
                }
            },
            2,
        ),
    ],
)
def test_default_fit_is_bit_identical_to_legacy_aliases(label, kw, k):
    rng = np.random.default_rng(7)
    views = [rng.standard_normal((60, p)) for p in (24, 18, 15)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_def = pysimlr.fit(
            "simlr", views, k, sparseness=0.8,
            iterations=10, random_state=7, **kw,
        )
        res_leg = pysimlr.fit(
            "simlr", views, k, sparseness=0.8,
            iterations=10, random_state=7,
            optimizer_type="hybrid_adam_legacy",
            initialization_type="pca_legacy",
            **kw,
        )
    assert torch.equal(res_def["u"], res_leg["u"])
    for vd, vl in zip(res_def["v"], res_leg["v"]):
        assert torch.equal(vd, vl)


# ---------------------------------------------------------------------------
# 2. First-step relative step size across HybridAdam variants
# ---------------------------------------------------------------------------
def test_hybrid_adam_step0_relative_displacement():
    torch.manual_seed(0)
    p = 100
    v0 = torch.randn(p, 1)
    v0 = v0 / v0.norm()
    g0 = torch.randn(p, 1) * 0.05

    # Legacy omits bias correction -> ||v1 - v0|| = 3.1623 * lr * sqrt(p)
    opt_leg = create_optimizer("hybrid_adam_legacy", [v0.clone()], learning_rate=0.01)
    v1_leg = opt_leg.step(0, v0.clone(), g0)
    assert float((v1_leg - v0).norm()) == pytest.approx(0.31622776, rel=1e-2)

    # Bias-corrected -> ||v1 - v0|| = 1.0 * lr * sqrt(p) = 0.10
    opt_bc = create_optimizer("hybrid_adam_bc", [v0.clone()], learning_rate=0.01)
    v1_bc = opt_bc.step(0, v0.clone(), g0)
    assert float((v1_bc - v0).norm()) == pytest.approx(0.10, rel=1e-2)

    # Normalized -> O(lr * ||v0||) <= 0.02
    opt_norm = create_optimizer("hybrid_adam_norm", [v0.clone()], learning_rate=0.01)
    v1_norm = opt_norm.step(0, v0.clone(), g0)
    assert float((v1_norm - v0).norm()) <= 0.02

    # Trust-region cap at lr=1.0 -> ||v1 - v0|| <= first_step_cap * max(||v0||, 1.0)
    opt_trust = create_optimizer(
        "hybrid_adam_trust", [v0.clone()], learning_rate=1.0, first_step_cap=1.0
    )
    v1_trust = opt_trust.step(0, v0.clone(), g0)
    assert float((v1_trust - v0).norm()) <= 1.0 + 1e-6


@pytest.mark.parametrize(
    "label,pos_kw",
    [
        ("linear", {"positivity": "linear"}),
        ("conical", {"positivity": "conical"}),
        ("convex", {"positivity": "convex"}),
        (
            "mixed",
            {
                "per_view_params": {
                    0: {"positivity": "linear"},
                    1: {"positivity": "conical"},
                    2: {"positivity": "convex"},
                }
            },
        ),
    ],
)
def test_simlr_first_step_sizes_under_all_constraints(label, pos_kw):
    _, views = _planted_rank1_views(seed=11, n=80, widths=(36, 48, 30))
    simlr_mod = sys.modules["pysimlr.simlr"]
    orig_create = simlr_mod.create_optimizer
    orig_sp = simlr_mod.simlr_sparseness
    max_pre = {}
    max_post = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for opt_name in ("hybrid_adam_legacy", "hybrid_adam_bc", "hybrid_adam_trust", "hybrid_adam_norm"):
            step0_io = {}
            proj_map = {}

            def wrapped_sp(v, *args, **kwargs):
                out = orig_sp(v, *args, **kwargs)
                mi = kwargs.get("modality_index")
                if mi is not None:
                    proj_map[(mi, len(proj_map))] = (v.detach().clone(), out.detach().clone())
                return out

            def wrapped_create(ot, vmats, **kw):
                opt = orig_create(ot, vmats, **kw)
                orig_step = opt.step

                def spy_step(i, vc, dg, fe=None):
                    vn = orig_step(i, vc, dg, fe)
                    if i not in step0_io:
                        step0_io[i] = (vc.detach().clone(), vn.detach().clone())
                    return vn

                opt.step = spy_step
                return opt

            simlr_mod.create_optimizer = wrapped_create
            simlr_mod.simlr_sparseness = wrapped_sp
            try:
                simlr(
                    views, k=1, iterations=1, sparseness=0.8,
                    initialization_type="pca_feasible", optimizer_type=opt_name,
                    learning_rate=1.0, random_state=42, **pos_kw,
                )
            finally:
                simlr_mod.create_optimizer = orig_create
                simlr_mod.simlr_sparseness = orig_sp
            pre_rels = []
            post_rels = []
            for i in range(len(views)):
                vc, vn = step0_io[i]
                denom = float(vc.norm().clamp_min(1e-8).item())
                pre_rels.append(float((vn - vc).norm().item()) / denom)
                p_vn = None
                for (mi, _), (pin, pout) in reversed(list(proj_map.items())):
                    if mi == i and torch.equal(pin, vn):
                        p_vn = pout
                        break
                assert p_vn is not None
                post_rels.append(float((p_vn - vc).norm().item()) / denom)
            max_pre[opt_name] = max(pre_rels)
            max_post[opt_name] = max(post_rels)

    assert max_pre["hybrid_adam_legacy"] > 10.0
    assert max_pre["hybrid_adam_bc"] < max_pre["hybrid_adam_legacy"]
    assert max_pre["hybrid_adam_trust"] <= 1.0 + 1e-5
    assert max_pre["hybrid_adam_norm"] <= 1.0 + 1e-5
    assert max_post["hybrid_adam_trust"] <= 1.0 + 1e-5
    assert max_post["hybrid_adam_norm"] <= 1.0 + 1e-5


# ---------------------------------------------------------------------------
# 3. Feasible starts (B3), hetero/mask alignment, & spectral/C2/C3 checks
# ---------------------------------------------------------------------------
def test_pca_feasible_returns_sparse_unit_norm_basis_while_legacy_stays_dense():
    torch.manual_seed(0)
    x = torch.randn(90, 100)
    v_feas = initial_basis_for_view(
        x, 1, positivity="either", initialization_type="pca_feasible", sparseness=0.8
    )
    zero_frac = float((v_feas == 0).float().mean())
    assert zero_frac >= 0.75
    assert float(v_feas.norm()) == pytest.approx(1.0, abs=1e-5)

    v_leg = initial_basis_for_view(
        x, 1, positivity="either", initialization_type="pca", sparseness=0.8
    )
    assert float((v_leg == 0).float().mean()) < 0.1


@pytest.mark.parametrize(
    "init_t",
    ["pca_feasible", "spectral", "gcca", "ica", "random", "perturbed_pca", "varimax"],
)
@pytest.mark.parametrize(
    "constraint",
    ["linear", "conical", "convex", "mixed"],
)
@pytest.mark.parametrize("sp", [0.5, 0.8])
def test_b3_start_feasibility_across_constraints_and_sparseness(init_t, constraint, sp):
    from pysimlr.sparsification import simlr_sparseness
    from pysimlr.utils import orthogonality_defect

    rng = np.random.default_rng(19)
    views = [torch.tensor(rng.standard_normal((60, p))).float() for p in (24, 20, 18)]
    pos_arg = "linear" if constraint != "mixed" else "linear"
    pos_list = (
        ["linear", "conical", "convex"]
        if constraint == "mixed"
        else [constraint] * len(views)
    )
    for k in (1, 2):
        gen = torch.Generator().manual_seed(0)
        v_mats = initialize_simlr(
            views, k, positivity=pos_arg, pos_list=pos_list,
            initialization_type=init_t, generator=gen, sparseness=sp,
        )
        assert len(v_mats) == len(views)
        for m, vm in enumerate(v_mats):
            p_m = pos_list[m]
            assert torch.isfinite(vm).all()
            if p_m in ("conical", "convex"):
                assert (vm >= -1e-6).all()
            if init_t not in ("pca_feasible", "spectral"):
                # Legacy v0.40.0 initializers (gcca, ica, random, perturbed_pca, varimax)
                # preserve bit-exact v0.40.0 starting bases (pinned by
                # tests/test_legacy_starts_golden.py) rather than enforcing Part B3
                # unit-column feasibility; only pca_feasible (plus varimax=True below)
                # and spectral enforce Part B3 unit-column feasibility across constraints.
                continue
            p_hard = simlr_sparseness(
                vm, constraint_type="none", positivity=p_m, sparseness=sp,
                modality_index=m, unit_columns=(p_m != "convex"),
            )
            assert torch.allclose(p_hard, vm, atol=1e-5), (
                f"{init_t} / {constraint} / sp={sp} / k={k} view {m} not feasible: "
                f"diff={float((p_hard - vm).norm())}"
            )
            if k == 1:
                p_ortho = simlr_sparseness(
                    vm, constraint_type="ortho", positivity=p_m, sparseness=sp,
                    modality_index=m, unit_columns=(p_m != "convex"),
                )
                assert torch.allclose(p_ortho, vm, atol=1e-5)
            else:
                assert float(orthogonality_defect(vm)) < 0.5
            if p_m == "convex":
                assert torch.allclose(vm.sum(dim=0), torch.ones(k, dtype=vm.dtype), atol=1e-5)
            else:
                assert torch.allclose(vm.norm(dim=0), torch.ones(k, dtype=vm.dtype), atol=1e-5)

    # pca_feasible with varimax=True provides the feasible sparse counterpart of varimax
    v_vmx_feas = initialize_simlr(
        views, 2, positivity=pos_arg, pos_list=pos_list,
        initialization_type="pca_feasible", varimax=True, sparseness=sp,
    )
    for m, vm in enumerate(v_vmx_feas):
        p_m = pos_list[m]
        p_hard = simlr_sparseness(
            vm, constraint_type="none", positivity=p_m, sparseness=sp,
            modality_index=m, unit_columns=(p_m != "convex"),
        )
        assert torch.allclose(p_hard, vm, atol=1e-5)


def test_k1_sign_alignment_for_pca_feasible_and_spectral():
    rng = np.random.default_rng(5)
    z = rng.standard_normal((100, 1))
    w0 = rng.standard_normal((25, 1))
    w1 = rng.standard_normal((25, 1))
    x0 = torch.tensor(z @ w0.T + 0.1 * rng.standard_normal((100, 25))).float()
    x1 = torch.tensor(z @ w1.T + 0.1 * rng.standard_normal((100, 25))).float()

    for init_t in ("pca_feasible", "spectral"):
        v_mats = initialize_simlr(
            [x0, x1], k=1, initialization_type=init_t, positivity="either", sparseness=0.8
        )
        p0 = (x0 @ v_mats[0]).numpy()[:, 0]
        p1 = (x1 @ v_mats[1]).numpy()[:, 0]
        assert np.corrcoef(p0, p1)[0, 1] > 0.0

    # Non-negative views must never be negated by k=1 alignment
    x0_pos = torch.tensor(np.abs(z) @ np.abs(w0.T) + 0.05 * np.abs(rng.standard_normal((100, 25)))).float()
    x1_pos = torch.tensor(np.abs(z) @ np.abs(w1.T) + 0.05 * np.abs(rng.standard_normal((100, 25)))).float()
    v_pos = initialize_simlr(
        [x0_pos, x1_pos], k=1, initialization_type="pca_feasible", positivity="positive", sparseness=0.8
    )
    for v in v_pos:
        assert (v >= 0).all()


def test_hetero_and_missing_rows_mask_alignment_without_changing_legacy():
    from pysimlr.simlr import _prepare_fit_inputs

    rng = np.random.default_rng(123)
    n = 80
    z = rng.standard_normal((n, 2))
    w0 = np.zeros((24, 2))
    w0[:8, 0] = 0.5 + rng.random(8)
    w0[8:16, 1] = 0.5 + rng.random(8)
    w1 = np.zeros((20, 2))
    w1[:8, 1] = 0.5 + rng.random(8)
    w1[8:16, 0] = 0.5 + rng.random(8)
    x0 = z @ w0.T + 0.25 * rng.standard_normal((n, 24))
    x1 = z @ w1.T + 0.25 * rng.standard_normal((n, 20))
    x1_mask = x1.copy()
    x1_mask[:12, :] = np.nan

    # Legacy pca with align_initial_views=None matches align_initial_views=False under mask
    _, _, v_leg_none, _ = _prepare_fit_inputs(
        [x0, x1_mask], 2, ["centerAndScale", "np"], "linear", "pca", 0, None, 0.5, None,
        missing_rows="mask", pos_list=["linear", "linear"],
    )
    _, _, v_leg_off, _ = _prepare_fit_inputs(
        [x0, x1_mask], 2, ["centerAndScale", "np"], "linear", "pca", 0, None, 0.5, None,
        missing_rows="mask", pos_list=["linear", "linear"], align_initial_views=False,
    )
    for a, b in zip(v_leg_none, v_leg_off):
        assert torch.equal(a, b)

    # pca_feasible, spectral, and align_initial_views=True align on mutually observed rows
    for pos in ("linear", "conical", "convex"):
        if pos != "linear":
            zp = np.abs(z)
            m0 = zp @ w0.T + 0.25 * np.abs(rng.standard_normal((n, 24)))
            m1 = zp @ w1.T + 0.25 * np.abs(rng.standard_normal((n, 20)))
            m1[:12, :] = np.nan
            mats = [m0, m1]
        else:
            mats = [x0, x1_mask]
        for init_t, aiv in [("pca_feasible", None), ("spectral", None), ("varimax", True)]:
            tm, _, v0, om = _prepare_fit_inputs(
                mats, 2, ["centerAndScale", "np"], pos, init_t, 0, None, 0.5, None,
                missing_rows="mask", pos_list=[pos, pos], align_initial_views=aiv,
            )
            shared = om[0] & om[1]
            p0 = (tm[0][shared] @ v0[0]).numpy()
            p1 = (tm[1][shared] @ v0[1]).numpy()
            assert min(np.corrcoef(p0[:, j], p1[:, j])[0, 1] for j in range(2)) > 0.7

    # Sparse mutual overlap (< max(k + 1, 3)) leaves view 1 unpermuted cleanly
    x0_disjoint = x0.copy()
    x1_disjoint = x1.copy()
    x0_disjoint[40:, :] = np.nan
    x1_disjoint[:38, :] = np.nan  # 2 shared rows (38, 39) < max(2 + 1, 3) = 3
    _, _, v_small_overlap, _ = _prepare_fit_inputs(
        [x0_disjoint, x1_disjoint], 2, ["centerAndScale", "np"], "linear", "pca_feasible",
        0, None, 0.5, None, missing_rows="mask", pos_list=["linear", "linear"],
    )
    _, _, v_no_align, _ = _prepare_fit_inputs(
        [x0_disjoint, x1_disjoint], 2, ["centerAndScale", "np"], "linear", "pca_feasible",
        0, None, 0.5, None, missing_rows="mask", pos_list=["linear", "linear"], align_initial_views=False,
    )
    assert torch.equal(v_small_overlap[1], v_no_align[1])


def test_spectral_and_gcca_preserve_advantage_and_subsample_cosine_stability():
    z_true, views = _planted_rank1_views(seed=42, n=150, widths=(40, 35, 30))
    rng = np.random.default_rng(99)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ws = []
        for _ in range(6):
            idx = rng.choice(len(z_true), size=int(0.8 * len(z_true)), replace=False)
            sub = [v[idx] for v in views]
            res = simlr(
                sub, k=1, positivity="linear", sparseness=0.8,
                initialization_type="gcca", optimizer_type="hybrid_adam_norm",
                iterations=40, random_state=42,
            )
            r = abs(float(np.corrcoef(z_true[idx, 0], res["u"].numpy()[:, 0])[0, 1]))
            assert r > 0.95
            w = np.concatenate([vm.numpy().ravel() for vm in res["v"]])
            w /= np.linalg.norm(w) + 1e-12
            ws.append(w)
    pair_cos = [
        abs(float(ws[i] @ ws[j]))
        for i in range(len(ws))
        for j in range(i + 1, len(ws))
    ]
    assert np.median(pair_cos) > 0.80
    assert min(pair_cos) > 0.80


@pytest.mark.parametrize("constraint", ["linear", "conical", "convex", "mixed"])
def test_c2_spectral_small_step_respects_per_view_constraints(constraint):
    _, views = _planted_rank1_views(seed=21, n=70, widths=(24, 20, 18))
    kw = (
        {
            "per_view_params": {
                0: {"positivity": "linear", "sparseness": 0.5},
                1: {"positivity": "conical", "sparseness": 0.8},
                2: {"positivity": "convex", "sparseness": 0.5},
            }
        }
        if constraint == "mixed"
        else {"positivity": constraint, "sparseness": 0.8}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = pysimlr.fit(
            "simlr", views, k=2, iterations=10,
            initialization_type="spectral", optimizer_type="hybrid_adam_trust",
            random_state=0, **kw,
        )
    assert res["parameters"]["initialization_variant"] == "spectral"
    assert res["parameters"]["optimizer_variant"] == "hybrid_adam_trust"
    if constraint == "conical":
        for vm in res["v"]:
            assert (vm >= -1e-6).all()
    elif constraint == "convex":
        for vm in res["v"]:
            assert (vm >= -1e-6).all()
            assert torch.allclose(vm.sum(dim=0), torch.ones(2, dtype=vm.dtype), atol=1e-5)
    elif constraint == "mixed":
        assert (res["v"][1] >= -1e-6).all()
        assert float((res["v"][1] == 0).float().mean()) >= 0.75
        assert (res["v"][2] >= -1e-6).all()
        assert torch.allclose(res["v"][2].sum(dim=0), torch.ones(2, dtype=res["v"][2].dtype), atol=1e-5)


def test_c3_soft_sparseness_continuity_and_all_constraints():
    from pysimlr.sparsification import simlr_sparseness

    v_a = torch.tensor([[1.0, 0.99, 0.1, 0.05, 0.01]]).T
    v_b = torch.tensor([[0.99, 1.0, 0.1, 0.05, 0.01]]).T
    for pos in ("linear", "conical", "convex"):
        sa = simlr_sparseness(v_a, constraint_type="ortho", positivity=pos, sparseness=0.8,
                              sparseness_type="soft", unit_columns=(pos != "convex"))
        sb = simlr_sparseness(v_b, constraint_type="ortho", positivity=pos, sparseness=0.8,
                              sparseness_type="soft", unit_columns=(pos != "convex"))
        ha = simlr_sparseness(v_a, constraint_type="ortho", positivity=pos, sparseness=0.8,
                              sparseness_type="hard", unit_columns=(pos != "convex"))
        hb = simlr_sparseness(v_b, constraint_type="ortho", positivity=pos, sparseness=0.8,
                              sparseness_type="hard", unit_columns=(pos != "convex"))
        cos_soft = float((sa * sb).sum() / (sa.norm() * sb.norm()))
        cos_hard = float((ha * hb).sum() / (ha.norm() * hb.norm()))
        assert cos_soft > 0.99 and cos_hard < 0.1
        if pos == "convex":
            assert torch.allclose(sa.sum(dim=0), torch.ones(1), atol=1e-5)

    _, views = _planted_rank1_views(seed=33, n=65, widths=(20, 18, 16))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for kw in (
            {"positivity": "linear"},
            {"positivity": "conical"},
            {"positivity": "convex"},
            {"per_view_params": {0: {"positivity": "linear"}, 1: {"positivity": "conical"}, 2: {"positivity": "convex"}}},
        ):
            res = pysimlr.fit(
                "simlr", views, k=1, iterations=8, sparseness=0.8,
                sparseness_type="soft", random_state=0, **kw,
            )
            assert res["parameters"]["sparseness_type"] == "soft"
            if kw.get("positivity") == "convex":
                for vm in res["v"]:
                    assert (vm >= -1e-6).all()
                    assert torch.allclose(vm.sum(dim=0), torch.ones(1, dtype=vm.dtype), atol=1e-5)
            elif "per_view_params" in kw:
                assert (res["v"][1] >= -1e-6).all()
                assert (res["v"][2] >= -1e-6).all()
                assert torch.allclose(res["v"][2].sum(dim=0), torch.ones(1, dtype=res["v"][2].dtype), atol=1e-5)

    with pytest.raises(ValueError, match="sparseness_type"):
        simlr(views, k=1, iterations=2, sparseness_type="invalid")


# ---------------------------------------------------------------------------
# 4. initial_v and sparseness_schedule (Phase 2D)
# ---------------------------------------------------------------------------
def test_initial_v_warm_start_and_validation():
    _, views = _planted_rank1_views(seed=3, n=60, widths=(20, 15, 12))
    custom = [torch.ones(v.shape[1], 1) * (i + 1) for i, v in enumerate(views)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res0 = simlr(views, k=1, iterations=0, initial_v=custom, learning_rate=0.05)
    for got, exp in zip(res0["v"], custom):
        assert torch.allclose(got.float(), exp.float())
    assert res0["initialization_type"] == "initial_v"

    with pytest.raises(ValueError, match="entries"):
        simlr(views, k=1, iterations=1, initial_v=custom[:2], learning_rate=0.05)
    with pytest.raises(ValueError, match="2-D"):
        simlr(views, k=1, iterations=1, initial_v=[torch.ones(20), custom[1], custom[2]], learning_rate=0.05)
    with pytest.raises(ValueError, match="shape"):
        simlr(views, k=1, iterations=1, initial_v=[torch.ones(20, 2), custom[1], custom[2]], learning_rate=0.05)
    bad = custom[0].clone()
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        simlr(views, k=1, iterations=1, initial_v=[bad, custom[1], custom[2]], learning_rate=0.05)


def test_sparseness_schedule_stages_and_validation():
    _, views = _planted_rank1_views(seed=8, n=70, widths=(25, 20, 18))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_int = simlr(
            views, k=1, iterations=10, positivity="linear", sparseness=0.8,
            sparseness_schedule=3, optimizer_type="hybrid_adam_trust", random_state=0,
        )
        res_seq = simlr(
            views, k=1, iterations=10, positivity="linear", sparseness=0.8,
            sparseness_schedule=[0.0, 0.4, 0.8], optimizer_type="hybrid_adam_trust", random_state=0,
        )
    for res in (res_int, res_seq):
        assert len(res["sparseness_schedule"]) == 3
        for v in res["v"]:
            assert float((v == 0).float().mean()) >= 0.75

    for bad_sched in (1, 0, True, [], [-0.1, 0.5], [0.5, 1.5], [0.5, float("nan")]):
        with pytest.raises(ValueError):
            simlr(views, k=1, iterations=2, sparseness=0.8, sparseness_schedule=bad_sched)
    with pytest.raises(ValueError, match="active target"):
        simlr(views, k=1, iterations=2, sparseness=None, sparseness_schedule=3)


# ---------------------------------------------------------------------------
# 5. Provenance & strict rejection on unsupported methods
# ---------------------------------------------------------------------------
def test_api_fit_provenance_and_no_silent_fallback():
    _, views = _planted_rank1_views(seed=2, n=60, widths=(18, 16, 14))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_auto = pysimlr.fit(
            "simlr", views, k=1, iterations=5,
            optimizer_type="hybrid_adam_norm", initialization_type="pca_feasible",
            random_state=0,
        )
        r_fixed = pysimlr.fit(
            "simlr", views, k=1, iterations=5, learning_rate=0.05,
            initial_v=[torch.ones(v.shape[1], 1) for v in views],
            sparseness_schedule=[0.0, 0.5], sparseness_type="soft",
            random_state=0,
        )
    assert isinstance(r_auto["learning_rate_used"], float)
    assert np.isfinite(r_auto["learning_rate_used"])
    p_auto = r_auto["parameters"]
    assert p_auto["optimizer_variant"] == "hybrid_adam_norm"
    assert p_auto["initialization_variant"] == "pca_feasible"
    assert p_auto["learning_rate_rule"] == "auto"
    assert p_auto["learning_rate_used"] == r_auto["learning_rate_used"]
    assert p_auto["sparseness_schedule"] is None
    assert p_auto["sparseness_type"] == "hard"

    p_fixed = r_fixed["parameters"]
    assert p_fixed["initialization_variant"] == "initial_v"
    assert p_fixed["learning_rate_rule"] == "fixed"
    assert p_fixed["learning_rate_used"] == pytest.approx(0.05)
    assert p_fixed["sparseness_schedule"] == [None, 0.5]
    assert p_fixed["sparseness_type"] == "soft"

    with pytest.raises(ValueError, match="pca"):
        pysimlr.fit("pca", views, k=1, initial_v=[torch.ones(v.shape[1], 1) for v in views])
    with pytest.raises(ValueError, match="lend"):
        pysimlr.fit("lend", views, k=1, sparseness_schedule=[0.0, 0.5])
