"""Tests for the matched energy pair (`recon_r2` vs `align`) and cross-view `system_matrices` prior."""
import warnings

import numpy as np
import pytest
import torch

import pysimlr
from pysimlr.similarity import _reset_linear_align_warning


def _system_views(seed=11, n=72, k=2):
    """Two views with a shared 2-system structure plus ambiguous noise features so
    cross-view system coupling pulls system-level loadings together."""
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n, k))
    # View 0: 8 features; system 0 -> cols 0..1, system 1 -> cols 2..3; cols 4..7 carry competing local signal
    v0 = np.zeros((8, k))
    v0[0:2, 0] = 0.8
    v0[2:4, 1] = 0.8
    local0 = rng.standard_normal((n, 2))
    v0_loc = np.zeros((8, 2))
    v0_loc[4:6, 0] = 1.1
    v0_loc[6:8, 1] = 1.1
    x0 = u @ v0.T + local0 @ v0_loc.T + 0.25 * rng.standard_normal((n, 8))

    # View 1: 10 features; system 0 -> cols 0..2, system 1 -> cols 3..5; cols 6..9 carry independent local signal
    v1 = np.zeros((10, k))
    v1[0:3, 0] = 0.75
    v1[3:6, 1] = 0.75
    local1 = rng.standard_normal((n, 2))
    v1_loc = np.zeros((10, 2))
    v1_loc[6:8, 0] = 1.1
    v1_loc[8:10, 1] = 1.1
    x1 = u @ v1.T + local1 @ v1_loc.T + 0.25 * rng.standard_normal((n, 10))

    p0 = np.zeros((2, 8), dtype=np.float32)
    p0[0, 0:2] = 1.0
    p0[1, 2:4] = 1.0

    p1 = np.zeros((2, 10), dtype=np.float32)
    p1[0, 0:3] = 1.0
    p1[1, 3:6] = 1.0

    return [torch.tensor(x0).float(), torch.tensor(x1).float()], [torch.tensor(p0), torch.tensor(p1)]


# --------------------------------------------------------------------------
# Part 1: matched energy contract (`recon_r2` on linear, `align` on deep/flow)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("method", ("lend", "ned", "ned_shared_private", "flow_v"))
def test_deep_and_flow_refuse_recon_r2_and_name_align_counterpart(method):
    views, _ = _system_views(n=36)
    with pytest.raises(ValueError, match="linear counterpart of 'align' is 'recon_r2'"):
        pysimlr.fit(method, views, k=2, energy_type="recon_r2", iterations=2)


def test_linear_align_warns_once_and_deep_align_does_not_warn():
    views, _ = _system_views(n=36)
    _reset_linear_align_warning()
    try:
        with pytest.warns(UserWarning, match="matched linear objective is 'recon_r2'"):
            pysimlr.fit("simlr", views, k=2, energy_type="align", iterations=2, random_state=0)

        # Second call stays quiet (warn-once gate).
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            pysimlr.fit("simlr", views, k=2, energy_type="align", iterations=2, random_state=0)
        align_warns = [w for w in caught if "matched linear objective" in str(w.message)]
        assert not align_warns

        # Deep/flow with energy_type='align' never emits the linear-only warning.
        _reset_linear_align_warning()
        with warnings.catch_warnings(record=True) as deep_caught:
            warnings.simplefilter("always", UserWarning)
            pysimlr.fit("lend", views, k=2, energy_type="align", iterations=2, batch_size=18, random_state=0)
        assert not [w for w in deep_caught if "matched linear objective" in str(w.message)]
    finally:
        _reset_linear_align_warning()


def test_matched_energy_helper_and_describe_method_cover_registry():
    expected = {
        "simlr": "recon_r2",
        "lend": "align",
        "ned": "align",
        "ned_shared_private": "align",
        "flow_v": "align",
        "pca": None,
        "nsa_pipeline": None,
    }
    assert set(expected) == set(pysimlr.method_names())
    for name, want in expected.items():
        assert pysimlr.matched_energy(name) == want
        desc = pysimlr.describe_method(name)
        assert desc["matched_energy"] == want
        if want is not None:
            assert "recon_r2" in desc["description"] and "align" in desc["description"]


# --------------------------------------------------------------------------
# Part 3: Addendum B (`system_matrices`, `system_lambda`, `system_penalty`)
# --------------------------------------------------------------------------
def test_system_prior_reduces_cross_view_system_discrepancy_and_respects_row_order():
    """`system_matrices` couples row `s` across views: enabling it reduces
    `||P_0 V_0 - P_1 V_1||_F^2`, and permuting the rows of `P_1` changes the
    solution (unlike within-view `domain_matrices`, which is row-permutation invariant)."""
    views, (p0, p1) = _system_views()
    p0_norm = p0 / p0.norm(dim=1, keepdim=True)
    p1_norm = p1 / p1.norm(dim=1, keepdim=True)

    base = pysimlr.fit("simlr", views, k=2, iterations=25, positivity="positive", random_state=0)
    coupled = pysimlr.fit(
        "simlr", views, k=2, iterations=25, positivity="positive",
        system_matrices=[p0, p1], system_lambda=5.0, system_penalty="l2", random_state=0,
    )
    swapped = pysimlr.fit(
        "simlr", views, k=2, iterations=25, positivity="positive",
        system_matrices=[p0, p1.flip(0)], system_lambda=5.0, system_penalty="l2", random_state=0,
    )

    disc_base = float(((p0_norm @ base["v"][0]) - (p1_norm @ base["v"][1])).pow(2).sum())
    disc_coupled = float(((p0_norm @ coupled["v"][0]) - (p1_norm @ coupled["v"][1])).pow(2).sum())
    assert disc_coupled < disc_base, (
        f"system prior did not reduce cross-view system discrepancy ({disc_coupled:.5f} vs {disc_base:.5f})"
    )
    # Row correspondence across views is live: flipping P_1's system index changes the fitted loadings.
    diff_swapped = float((coupled["v"][0] - swapped["v"][0]).norm() + (coupled["v"][1] - swapped["v"][1]).norm())
    assert diff_swapped > 1e-3

    # Per-view positivity constraint still holds under system coupling.
    for v_mat in coupled["v"]:
        assert float(v_mat.min()) >= -1e-7


@pytest.mark.parametrize("method", ("simlr", "lend", "ned", "ned_shared_private", "flow_v"))
@pytest.mark.parametrize("penalty", ("l2", "group_l2"))
def test_system_prior_runs_across_all_five_methods_and_supports_dict_and_missing_systems(method, penalty):
    views, (p0, p1) = _system_views(n=48)
    # Add a 3rd view that does not participate (None) and a 3rd system row absent from view 1 (all zeros)
    rng = np.random.default_rng(21)
    v2 = torch.tensor(rng.standard_normal((48, 6))).float()
    tri_views = {"mod_a": views[0], "mod_b": views[1], "mod_c": v2}

    p0_ext = torch.zeros((3, 8), dtype=torch.float32)
    p0_ext[:2] = p0
    p0_ext[2, 4:6] = 1.0  # system 2 present only in mod_a

    p1_ext = torch.zeros((3, 10), dtype=torch.float32)
    p1_ext[:2] = p1  # row 2 all-zero -> mod_b lacks system 2

    kw = dict(
        system_matrices={"mod_a": p0_ext, "mod_b": p1_ext},
        system_lambda=1.5,
        system_penalty=penalty,
        positivity="positive",
        sparseness=0.6,
        iterations=4,
        random_state=0,
    )
    if method != "simlr":
        kw["batch_size"] = 24
    res = pysimlr.fit(method, tri_views, k=2, **kw)
    assert len(res["v"]) == 3
    for v_mat in res["v"]:
        assert bool(torch.isfinite(v_mat).all())
        assert float(v_mat.min()) >= -1e-6


def test_system_prior_validation_errors():
    views, (p0, p1) = _system_views(n=32)

    # system_lambda without system_matrices raises
    with pytest.raises(ValueError, match="without system_matrices"):
        pysimlr.fit("simlr", views, k=2, system_lambda=1.0, iterations=2)

    # Negative system_lambda raises
    with pytest.raises(ValueError, match="non-negative"):
        pysimlr.fit("simlr", views, k=2, system_matrices=[p0, p1], system_lambda=-0.5, iterations=2)

    # Mismatched row count S across views raises
    with pytest.raises(ValueError, match="common system row count"):
        pysimlr.fit("simlr", views, k=2, system_matrices=[p0, p1[:1]], iterations=2)

    # Wrong feature width p_i raises
    with pytest.raises(ValueError, match="features"):
        pysimlr.fit("simlr", views, k=2, system_matrices=[p0[:, :5], p1], iterations=2)

    # Unknown system_penalty raises
    with pytest.raises(ValueError, match="system_penalty"):
        pysimlr.fit("simlr", views, k=2, system_matrices=[p0, p1], system_penalty="l1", iterations=2)

    # l2 with zero shared active system rows raises clearly
    disjoint_p0 = torch.zeros_like(p0)
    disjoint_p0[0, 0:2] = 1.0
    disjoint_p1 = torch.zeros_like(p1)
    disjoint_p1[1, 3:6] = 1.0
    with pytest.raises(ValueError, match="at least two views"):
        pysimlr.fit("simlr", views, k=2, system_matrices=[disjoint_p0, disjoint_p1],
                    system_penalty="l2", iterations=2)
