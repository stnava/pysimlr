"""Tests for `pysimlr.bottleneck_rank` and its use in `select_rank`."""
import numpy as np
import pytest

import pysimlr
from pysimlr.benchmarks.synthetic_cases import build_path_graph_case

GRAPH = [(0, 1), (1, 2)]


def _case(n=400, k=3, p=(20, 25, 6), noise=0.5, cov=0.0, seed=0):
    return build_path_graph_case(GRAPH, n, k, p, noise, covariate_effect=cov, random_state=seed)


def test_bottleneck_is_smallest_view():
    case = _case()
    b = pysimlr.bottleneck_rank(case["views"])
    # the third planted latent loads weakly on the 6-column view (eigenvalue 0.5-1.4 vs null ~1.06),
    # so the bound is 2 or 3 depending on seed; it never exceeds the 6 columns
    assert 2 <= b["k_max"] <= 6
    assert b["bottleneck_view"] == 2
    rr = b["per_view"]["reliable_rank"].to_numpy()
    assert rr[0] >= 3 and rr[1] >= 3


def test_pure_noise_has_little_rank():
    rng = np.random.default_rng(1)
    views = [rng.standard_normal((300, p)) for p in (10, 12, 8)]
    assert pysimlr.bottleneck_rank(views)["k_max"] <= 1


def test_confound_leak_is_removed():
    case = _case(n=300, k=2, p=(15, 15, 15), cov=1.5)
    without = pysimlr.bottleneck_rank(case["views"])
    with_c = pysimlr.bottleneck_rank(case["views"], confounds=case["C"])
    assert with_c["parameters"]["confounds"] is True
    assert (with_c["per_view"]["reliable_rank"].to_numpy()
            <= without["per_view"]["reliable_rank"].to_numpy()).all()
    assert with_c["k_max"] >= 2


def test_determinism_and_tensor_numpy_equivalence():
    case = _case(n=200)
    a = pysimlr.bottleneck_rank(case["views"], random_state=3)
    b = pysimlr.bottleneck_rank(case["views"], random_state=3)
    c = pysimlr.bottleneck_rank([v.numpy() for v in case["views"]], random_state=3)
    assert a["k_max"] == b["k_max"] == c["k_max"]
    for x, y, z in zip(a["thresholds"], b["thresholds"], c["thresholds"]):
        assert np.array_equal(x, y) and np.array_equal(x, z)


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
    assert np.allclose(shared["eigenvalues"][1], as_list["eigenvalues"][1])
    partial = pysimlr.bottleneck_rank(views, confounds=[C, None, None])
    none = pysimlr.bottleneck_rank(views)
    assert np.allclose(partial["eigenvalues"][1], none["eigenvalues"][1])      # view 1 untouched
    assert np.allclose(partial["eigenvalues"][0], shared["eigenvalues"][0])    # view 0 residualised
    with pytest.raises(ValueError):
        pysimlr.bottleneck_rank(views, confounds=C[:-1])
