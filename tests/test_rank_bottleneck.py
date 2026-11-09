"""Tests for `pysimlr.bottleneck_rank` and its use in `select_rank`."""
import numpy as np
import pytest

import pysimlr
from pysimlr.benchmarks.synthetic_cases import build_path_graph_case

GRAPH = [(0, 1), (1, 2)]


def _case(n=400, k=3, p=(20, 25, 6), noise=0.5, cov=0.0, seed=0):
    return build_path_graph_case(GRAPH, n, k, p, noise, covariate_effect=cov, random_state=seed)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_cross_view_bottleneck_recovers_planted_rank(seed):
    """Cross-view rank finds the planted k = 3 in the 6-column view on every seed; within-view
    parallel analysis misses the weakly footprinted third latent on most seeds."""
    case = _case(seed=seed)
    b = pysimlr.bottleneck_rank(case["views"], random_state=seed)
    assert b["k_max"] == 3 and b["parameters"]["mode"] == "cross"
    assert (b["per_view"]["cross_rank"].to_numpy() == 3).all()
    assert (b["per_view"]["within_rank"].to_numpy()[:2] >= 3).all()
    assert b["per_view"]["within_rank"].to_numpy()[2] <= 3
    # the fourth cross-view singular value of the 6-column view is far under its threshold
    assert b["singular_values"][2][3] < b["thresholds"][2][3]
    w = pysimlr.bottleneck_rank(case["views"], mode="within", random_state=seed)
    assert w["k_max"] == w["per_view"]["within_rank"].min() and w["k_max"] <= 3


def test_private_structure_does_not_count_in_cross_mode():
    """A view with a strong private factor: within-view rank counts it, cross-view rank does not."""
    case = _case(n=400, k=2, p=(20, 25, 12), seed=7)
    views = [v.numpy().copy() for v in case["views"]]
    rng = np.random.default_rng(7)
    private = rng.standard_normal((400, 1))
    views[2][:, :6] += 2.0 * private                     # one private factor on half of view 2's columns
    b = pysimlr.bottleneck_rank(views)
    assert b["per_view"]["within_rank"].to_numpy()[2] >= 3
    assert b["per_view"]["cross_rank"].to_numpy()[2] == 2 and b["k_max"] == 2


def test_pure_noise_has_little_rank():
    rng = np.random.default_rng(1)
    views = [rng.standard_normal((300, p)) for p in (10, 12, 8)]
    assert pysimlr.bottleneck_rank(views)["k_max"] <= 1


def test_confound_leak_is_removed():
    case = _case(n=300, k=2, p=(15, 15, 15), cov=1.5)
    without = pysimlr.bottleneck_rank(case["views"])
    with_c = pysimlr.bottleneck_rank(case["views"], confounds=case["C"])
    assert with_c["parameters"]["confounds"] is True
    # the leaked covariates are shared across views, so cross-view rank counts them until removed
    assert without["k_max"] > with_c["k_max"] == 2
    assert (with_c["per_view"]["cross_rank"].to_numpy() <= without["per_view"]["cross_rank"].to_numpy()).all()


def test_determinism_and_tensor_numpy_equivalence():
    case = _case(n=200)
    a = pysimlr.bottleneck_rank(case["views"], random_state=3)
    b = pysimlr.bottleneck_rank(case["views"], random_state=3)
    c = pysimlr.bottleneck_rank([v.numpy() for v in case["views"]], random_state=3)
    assert a["k_max"] == b["k_max"] == c["k_max"]
    for key in ("thresholds", "eigen_thresholds", "singular_values"):
        for x, y, z in zip(a[key], b[key], c[key]):
            assert np.array_equal(x, y) and np.array_equal(x, z)
    with pytest.raises(ValueError, match="mode"):
        pysimlr.bottleneck_rank(case["views"], mode="between")


def test_errors():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((50, 5))
    with pytest.raises(ValueError):
        pysimlr.bottleneck_rank([x])
    with pytest.raises(ValueError):
        pysimlr.bottleneck_rank([x, x[:40]])
    bad = x.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        pysimlr.bottleneck_rank([x, bad])


def test_select_rank_auto_and_warning():
    case = _case(n=150)
    b = pysimlr.bottleneck_rank(case["views"])
    sel = pysimlr.select_rank("simlr", case["views"], "auto", n_subsamples=2, iterations=3,
                              random_state=0)
    assert sel["bottleneck"]["k_max"] == b["k_max"]
    assert [r["k"] for r in sel["table"]] == list(range(1, b["k_max"] + 1))
    with pytest.warns(RuntimeWarning, match="bottleneck"):
        sel2 = pysimlr.select_rank("simlr", case["views"], [2, 9], n_subsamples=2,
                                   iterations=3, random_state=0)
    assert set(sel2) == set(sel)


def test_bottleneck_rank_per_view_confounds():
    """Per-view confound lists and dicts (as `fit` takes them) residualise each view on its own block."""
    case = build_path_graph_case([(0, 1), (1, 2)], 300, 2, (15, 15, 15), 0.5, covariate_effect=1.5, random_state=5)
    views, C = case["views"], case["C"]
    shared = pysimlr.bottleneck_rank(views, confounds=C)
    as_list = pysimlr.bottleneck_rank(views, confounds=[C, C, C])
    as_dict = pysimlr.bottleneck_rank(views, confounds={0: C, 1: C, 2: C})
    assert shared["k_max"] == as_list["k_max"] == as_dict["k_max"]
    assert np.allclose(shared["singular_values"][1], as_list["singular_values"][1])
    partial = pysimlr.bottleneck_rank(views, confounds=[C, None, None])
    none = pysimlr.bottleneck_rank(views)
    assert np.allclose(partial["eigenvalues"][1], none["eigenvalues"][1])      # view 1 untouched (within)
    assert np.allclose(partial["eigenvalues"][0], shared["eigenvalues"][0])    # view 0 residualised
    with pytest.raises(ValueError):
        pysimlr.bottleneck_rank(views, confounds=C[:-1])
