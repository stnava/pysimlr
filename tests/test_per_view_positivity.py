"""Tests for per-view positivity, view-name keys, mixed positivity reporting, and nonneg_mask."""
from __future__ import annotations

import numpy as np
import pytest
import torch

import pysimlr
from pysimlr.positivity import resolve_positivity
from pysimlr.sparsification import apply_positivity, simlr_sparseness


def _make_three_views(n: int = 80, seed: int = 11):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 2))
    w0 = np.abs(rng.standard_normal((2, 16)))
    # View 1 has strong two-sided (positive and negative) contrast loadings
    w1 = rng.standard_normal((2, 18))
    w1[0, :9] = np.abs(w1[0, :9]) + 1.0
    w1[0, 9:] = -np.abs(w1[0, 9:]) - 1.0
    w2 = np.abs(rng.standard_normal((2, 14)))
    x0 = z @ w0 + 0.35 * rng.standard_normal((n, 16))
    x1 = z @ w1 + 0.35 * rng.standard_normal((n, 18))
    x2 = z @ w2 + 0.35 * rng.standard_normal((n, 14))
    return [x0, x1, x2]


# ---------------------------------------------------------------------------
# Item 1: resolve_positivity pointer hint & list/mapping positivity on simlr
# ---------------------------------------------------------------------------
def test_resolve_positivity_type_error_points_to_per_view_params():
    with pytest.raises(TypeError, match="per_view_params"):
        resolve_positivity(["convex", "either"])


@pytest.mark.parametrize("method", ["lend", "flow_v"])
def test_deep_and_flow_refuse_list_positivity_with_clear_error(method):
    xs = _make_three_views(40)
    with pytest.raises(ValueError, match="per_view_params"):
        pysimlr.fit(method, xs, k=1, positivity=["convex", "either", "convex"], iterations=2, random_state=0)


@pytest.mark.parametrize("pos_val", ["convex", "positive", "either"])
@pytest.mark.parametrize("init_type", ["pca", "random", "gcca"])
def test_identical_positivity_list_reproduces_scalar_string_fit_bit_for_bit(pos_val, init_type):
    xs = _make_three_views(60)
    r_str = pysimlr.fit(
        "simlr", xs, k=2, positivity=pos_val, energy_type="recon_r2",
        initialization_type=init_type, iterations=20, random_state=7,
    )
    r_lst = pysimlr.fit(
        "simlr", xs, k=2, positivity=[pos_val, pos_val, pos_val], energy_type="recon_r2",
        initialization_type=init_type, iterations=20, random_state=7,
    )
    assert torch.equal(r_str["u"], r_lst["u"])
    for va, vb in zip(r_str["v"], r_lst["v"]):
        assert torch.equal(va, vb)
    assert r_lst["parameters"]["positivity_resolved"] == pos_val
    assert r_lst["parameters"]["positivity_by_view"] == [pos_val, pos_val, pos_val]


def test_mixed_positivity_list_and_mapping_match_per_view_params():
    xs = _make_three_views(70)
    named = {"vol": xs[0], "rel_thk": xs[1], "dti": xs[2]}

    r_pvp = pysimlr.fit(
        "simlr", named, k=1, positivity="convex", energy_type="recon_r2",
        per_view_params={"rel_thk": {"positivity": "either"}},
        iterations=25, random_state=3,
    )
    r_list = pysimlr.fit(
        "simlr", named, k=1, positivity=["convex", "either", "convex"],
        energy_type="recon_r2", iterations=25, random_state=3,
    )
    r_map = pysimlr.fit(
        "simlr", named, k=1,
        positivity={"vol": "convex", "rel_thk": "linear", "dti": "convex"},
        energy_type="recon_r2", iterations=25, random_state=3,
    )

    for r in (r_pvp, r_list, r_map):
        assert torch.all(r["v"][0] >= -1e-7)
        assert torch.allclose(r["v"][0].double().sum(0), torch.ones(1, dtype=torch.float64), atol=1e-5)
        assert bool((r["v"][1] < -1e-3).any()), "signed view should carry negative weights"
        assert torch.all(r["v"][2] >= -1e-7)
        assert torch.allclose(r["v"][2].double().sum(0), torch.ones(1, dtype=torch.float64), atol=1e-5)

    for va, vb, vc in zip(r_pvp["v"], r_list["v"], r_map["v"]):
        assert torch.equal(va, vb)
        assert torch.equal(va, vc)


def test_positivity_list_or_map_conflicts_with_per_view_params_raise():
    xs = _make_three_views(50)
    with pytest.raises(ValueError, match="both positivity and per_view_params"):
        pysimlr.fit(
            "simlr", xs, k=1,
            positivity=["convex", "either", "convex"],
            per_view_params={1: {"positivity": "either"}},
            iterations=5, random_state=0,
        )
    # Non-conflicting per_view_params keys (e.g. sparseness) compose cleanly
    res = pysimlr.fit(
        "simlr", xs, k=1,
        positivity=["convex", "either", "convex"],
        per_view_params={1: {"sparseness": 0.7}},
        iterations=10, random_state=0,
    )
    assert res["parameters"]["positivity_by_view"] == ["convex", "either", "convex"]


def test_positivity_list_wrong_length_raises():
    xs = _make_three_views(40)
    with pytest.raises(ValueError, match="positivity has 2 entries for 3 views"):
        pysimlr.fit("simlr", xs, k=1, positivity=["convex", "either"], iterations=3, random_state=0)


# ---------------------------------------------------------------------------
# Item 2: View keys by name in both pysimlr.fit and pysimlr.simlr
# ---------------------------------------------------------------------------
def test_view_keys_by_name_in_fit_and_native_simlr_and_duplicate_errors():
    xs = _make_three_views(50)
    named = {"thickness": xs[0], "perfusion": xs[1], "dti": xs[2]}

    r_fit = pysimlr.fit(
        "simlr", named, k=1, positivity="convex", energy_type="recon_r2",
        per_view_params={"perfusion": {"positivity": "either", "sparseness": 0.6}},
        iterations=12, random_state=5,
    )
    r_nat = pysimlr.simlr(
        named, k=1, positivity="convex", energy_type="recon_r2",
        per_view_params={"perfusion": {"positivity": "either", "sparseness": 0.6}},
        iterations=12, random_state=5,
    )
    assert r_fit["view_names"] == ["thickness", "perfusion", "dti"]
    assert r_nat["view_names"] == ["thickness", "perfusion", "dti"]
    for va, vb in zip(r_fit["v"], r_nat["v"]):
        assert torch.equal(va, vb)

    # Unknown view name raises ValueError naming the valid view names
    with pytest.raises(ValueError, match="unknown view key 'unknown_mod'"):
        pysimlr.fit("simlr", named, k=1, per_view_params={"unknown_mod": {"positivity": "either"}})
    with pytest.raises(ValueError, match="unknown view key 'unknown_mod'"):
        pysimlr.simlr(named, k=1, per_view_params={"unknown_mod": {"positivity": "either"}})

    # Duplicate view specification (index + name or index + 'viewN') raises ValueError
    with pytest.raises(ValueError, match="gives view 0 twice"):
        pysimlr.fit(
            "simlr", named, k=1,
            per_view_params={0: {"sparseness": 0.5}, "thickness": {"positivity": "either"}},
        )
    with pytest.raises(ValueError, match="gives view 1 twice"):
        pysimlr.simlr(
            named, k=1,
            per_view_params={"view1": {"sparseness": 0.5}, "perfusion": {"positivity": "either"}},
        )


# ---------------------------------------------------------------------------
# Item 3: positivity_by_view, positivity_resolved='mixed', loadings & extend
# ---------------------------------------------------------------------------
def test_mixed_positivity_reporting_and_downstream_readers():
    xs = _make_three_views(70)
    res = pysimlr.fit(
        "simlr", xs, k=2, positivity="convex", energy_type="recon_r2",
        per_view_params={1: {"positivity": "either"}},
        iterations=20, random_state=11,
    )
    assert res["parameters"]["positivity_by_view"] == ["convex", "either", "convex"]
    assert res["parameters"]["positivity_resolved"] == "mixed"

    # Loadings: view 0 & 2 are non-negative OLEE hard assignments with margins;
    # view 1 ('either') keeps signed SVD polar loadings and None margin!
    assert torch.all(res["loadings"][0] >= 0.0)
    assert torch.all(res["loadings"][2] >= 0.0)
    assert bool((res["loadings"][1] < -1e-4).any()), "signed view OLEE loadings must not be clamped to >= 0"
    assert res["loading_margin"] is not None
    assert res["loading_margin"][0] is not None
    assert res["loading_margin"][1] is None
    assert res["loading_margin"][2] is not None

    # Baseline methods without positivity still carry positivity_by_view=[None, ...]
    r_pca = pysimlr.fit("pca", xs, k=2)
    assert r_pca["parameters"]["positivity_resolved"] is None
    assert r_pca["parameters"]["positivity_by_view"] == [None, None, None]

    # extend(how='anchored') refuses implicit positivity on a 'mixed' fit unless explicit positivity= is given
    new_x = np.random.default_rng(0).standard_normal((70, 10))
    with pytest.raises(ValueError, match="mixed per-view positivity"):
        pysimlr.extend(res, xs, {"extra": new_x}, iterations=5)
    ext = pysimlr.extend(res, xs, {"extra": new_x}, positivity="either", iterations=5)
    assert ext["extensions"]["extra"]["positivity"] == "either"


# ---------------------------------------------------------------------------
# Item 5: Per-feature nonneg_mask inside a view
# ---------------------------------------------------------------------------
def test_apply_positivity_and_simlr_sparseness_with_nonneg_mask():
    rng = torch.Generator().manual_seed(42)
    v = torch.randn(12, 2, generator=rng)
    # Force masked rows negative on column 0 to verify column sign resolution + clamping
    mask = np.array([False] * 10 + [True, True])
    v[10:, 0] = torch.tensor([-2.0, -1.5])
    v[:5, 0] = torch.tensor([0.8, -0.9, 1.1, -1.2, 0.5])

    vp = apply_positivity(v, "positive", nonneg_mask=mask)
    assert torch.all(vp[10:] >= 0.0)
    assert bool((vp[:10] < 0.0).any()), "unmasked features must remain signed"

    vs = simlr_sparseness(
        v, constraint_type="ortho", positivity="positive",
        sparseness=0.5, constraint_weight=0.5, energy_type="recon_r2",
        nonneg_mask=mask,
    )
    assert torch.all(vs[10:] >= 0.0)
    assert bool((vs[:10] < 0.0).any())

    # All-True mask matches unmasked call bit-for-bit
    all_true = np.ones(12, dtype=bool)
    vs_plain = simlr_sparseness(
        v, constraint_type="ortho", positivity="positive",
        sparseness=0.5, constraint_weight=0.5, energy_type="recon_r2",
    )
    vs_all_true = simlr_sparseness(
        v, constraint_type="ortho", positivity="positive",
        sparseness=0.5, constraint_weight=0.5, energy_type="recon_r2",
        nonneg_mask=all_true,
    )
    assert torch.equal(vs_plain, vs_all_true)


@pytest.mark.parametrize("init_type", ["pca", "random", "gcca"])
def test_nonneg_mask_in_fit_masked_nonneg_unmasked_signed_and_all_true_exact(init_type):
    rng = np.random.default_rng(19)
    n = 75
    u_lat = rng.standard_normal((n, 1))
    # View 0: 20 relative contrast features (half +, half -) + 2 positive absolute features
    rel_w = np.concatenate([np.full(10, 1.2), np.full(10, -1.2)])
    abs_w = np.array([1.5, 1.3])
    x0 = u_lat @ np.concatenate([rel_w, abs_w])[None, :] + 0.3 * rng.standard_normal((n, 22))
    x0[:, :20] -= x0[:, :20].mean(axis=1, keepdims=True)  # row-centered relative block
    x1 = u_lat @ np.abs(rng.standard_normal((1, 15))) + 0.3 * rng.standard_normal((n, 15))

    mask = np.array([False] * 20 + [True, True])
    res_masked = pysimlr.fit(
        "simlr", [x0, x1], k=1, positivity="positive", energy_type="recon_r2",
        sparseness=0.5, initialization_type=init_type, iterations=25, random_state=9,
        per_view_params={0: {"positivity": "positive", "nonneg_mask": mask}},
    )
    v0 = res_masked["v"][0]
    assert torch.all(v0[20:] >= 0.0), "masked features must stay non-negative"
    assert bool((v0[:20] < -1e-4).any()), "unmasked relative features must allow negative weights"
    assert torch.all(res_masked["v"][1] >= 0.0)
    # Loadings for partially masked view 0 must not clamp negative correlations to 0
    assert bool((res_masked["loadings"][0][:20] < -1e-3).any())

    # All-True mask reproduces plain fit bit-for-bit
    res_plain = pysimlr.fit(
        "simlr", [x0, x1], k=1, positivity="positive", energy_type="recon_r2",
        sparseness=0.5, initialization_type=init_type, iterations=25, random_state=9,
    )
    res_all_true = pysimlr.fit(
        "simlr", [x0, x1], k=1, positivity="positive", energy_type="recon_r2",
        sparseness=0.5, initialization_type=init_type, iterations=25, random_state=9,
        per_view_params={0: {"positivity": "positive", "nonneg_mask": np.ones(22, dtype=bool)}},
    )
    assert torch.equal(res_plain["u"], res_all_true["u"])
    for va, vb in zip(res_plain["v"], res_all_true["v"]):
        assert torch.equal(va, vb)


def test_nonneg_mask_convex_and_validation_errors():
    xs = _make_three_views(45)
    partial = np.array([False] * 15 + [True])

    # Partial mask with convex raises ValueError explaining unit-sum simplex gauge
    with pytest.raises(ValueError, match="simplex"):
        pysimlr.fit(
            "simlr", xs, k=1, positivity="convex", energy_type="recon_r2",
            per_view_params={0: {"nonneg_mask": partial}}, iterations=5, random_state=0,
        )

    # All-True mask with convex is accepted and matches plain convex fit bit-for-bit
    r_plain = pysimlr.fit("simlr", xs, k=1, positivity="convex", energy_type="recon_r2", iterations=10, random_state=2)
    r_true = pysimlr.fit(
        "simlr", xs, k=1, positivity="convex", energy_type="recon_r2", iterations=10, random_state=2,
        per_view_params={0: {"nonneg_mask": np.ones(16, dtype=bool)}},
    )
    for va, vb in zip(r_plain["v"], r_true["v"]):
        assert torch.equal(va, vb)

    # Wrong length raises ValueError
    with pytest.raises(ValueError, match="expected 16 features"):
        pysimlr.fit(
            "simlr", xs, k=1, positivity="positive",
            per_view_params={0: {"nonneg_mask": np.ones(10, dtype=bool)}},
        )

    # nonneg_mask with positivity='either' raises ValueError
    with pytest.raises(ValueError, match="non-negative positivity"):
        pysimlr.fit(
            "simlr", xs, k=1, positivity="either",
            per_view_params={0: {"nonneg_mask": partial}},
        )

    # Non-boolean integer mask raises TypeError
    with pytest.raises(TypeError, match="boolean"):
        pysimlr.fit(
            "simlr", xs, k=1, positivity="positive",
            per_view_params={0: {"nonneg_mask": [0, 1] * 8}},
        )


def test_bagged_fit_select_and_average_with_per_view_positivity_and_nonneg_mask():
    xs = _make_three_views(60)
    # Ensure the last (masked) feature in view 1 carries strong shared signal so select-mode keeps it
    xs[1] = xs[1].copy()
    xs[1][:, -1] = xs[0][:, 0] * 2.0
    named = {"vol": xs[0], "rel_thk": xs[1], "dti": xs[2]}
    mask = np.array([False] * 16 + [True, True])

    bag_sel = pysimlr.bagged_fit(
        "simlr", named, k=1, aggregate="select", n_subsamples=6, fraction=0.8, threshold=0.55,
        positivity="positive", energy_type="recon_r2", iterations=10, random_state=0,
        per_view_params={"rel_thk": {"positivity": "positive", "nonneg_mask": mask}},
    )
    kept_mask = mask[bag_sel["feature_mask"][1]]
    expected_p1 = "positive" if kept_mask.any() else "either"
    assert bag_sel["parameters"]["positivity_by_view"] == ["positive", expected_p1, "positive"]
    if kept_mask.any():
        assert torch.all(bag_sel["v"][1][torch.as_tensor(kept_mask)] >= 0.0)

    bag_avg = pysimlr.bagged_fit(
        "simlr", named, k=1, aggregate="average", n_subsamples=5, fraction=0.8,
        positivity=["convex", "either", "convex"], scale_list=["centerAndScale"],
        energy_type="recon_r2", iterations=10, random_state=0,
    )
    assert bag_avg["weights_averaged"] is True
    assert len(bag_avg["effective_features"]) == 3
    assert all(np.isfinite(ef) for ef in bag_avg["effective_features"])
