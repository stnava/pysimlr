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
    b = pysimlr.bottleneck_rank(case["views"], bound="max")
    sel = pysimlr.select_rank("simlr", case["views"], "auto", n_subsamples=2, iterations=3,
                              learning_rate=0.05, random_state=0)
    assert sel["bottleneck"]["k_max"] == b["k_max"]
    assert sel["bottleneck"]["parameters"]["bound"] == "max"
    assert [r["k"] for r in sel["table"]] == list(range(1, b["k_max"] + 1))
    with pytest.warns(RuntimeWarning, match="bottleneck"):
        sel2 = pysimlr.select_rank("simlr", case["views"], [2, 9], n_subsamples=2,
                                   iterations=3, learning_rate=0.05, random_state=0)
    assert set(sel2) == set(sel)


def test_bottleneck_rank_bounds_and_auto_zero_message():
    case = _case(n=300, k=3, p=(20, 25, 6), noise=0.45, seed=0)
    rng = np.random.default_rng(100)
    views = [v.numpy() for v in case["views"]] + [rng.standard_normal((300, 20))]
    b_min = pysimlr.bottleneck_rank(views, bound="min", n_permutations=60, random_state=0)
    b_max = pysimlr.bottleneck_rank(views, bound="max", n_permutations=60, random_state=0)
    b_stk = pysimlr.bottleneck_rank(views, bound="stacked", n_permutations=60, random_state=0)
    assert b_min["k_max"] == 0 and b_min["bottleneck_view"] == 3
    assert b_max["k_max"] == 3 and b_max["bottleneck_view"] in (0, 1, 2)
    assert 3 <= b_stk["k_max"] <= 4 and b_stk["bottleneck_view"] is None
    for b in (b_min, b_max, b_stk):
        assert b["k_max_min"] == 0 and b["k_max_max"] == 3 and 3 <= b["k_max_stacked"] <= 4
    with pytest.raises(ValueError, match="bound"):
        pysimlr.bottleneck_rank(views, bound="median")

    with pytest.raises(ValueError, match=r"bound='min'.*sigma1/tau95"):
        pysimlr.select_rank("pca", views, "auto", bound="min", random_state=0)

    rng_null = np.random.default_rng(42)
    noise_views = [rng_null.standard_normal((120, p)) for p in (10, 12, 8)]
    with pytest.raises(ValueError, match=r"bound='max'.*sigma1/tau95"):
        pysimlr.select_rank("pca", noise_views, "auto", random_state=0)


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


def test_bottleneck_rank_borderline_flag():
    rng = np.random.default_rng(31)
    n = 120
    z = rng.standard_normal((n, 2))
    v_strong0 = z @ rng.standard_normal((2, 12)) + 0.4 * rng.standard_normal((n, 12))
    v_strong1 = z @ rng.standard_normal((2, 12)) + 0.4 * rng.standard_normal((n, 12))
    # Pick a pure-noise draw with ratio < 0.85
    v_noise = None
    for s in range(30):
        cand = np.random.default_rng(500 + s).standard_normal((n, 12))
        b0 = pysimlr.bottleneck_rank([v_strong0, v_strong1, cand], n_permutations=60, random_state=0)
        pv0 = b0["per_view"].iloc[2]
        if pv0["first_singular_value"] / pv0["threshold_first"] < 0.84:
            v_noise = cand
            break
    assert v_noise is not None

    # Calibrate view 3 to ratio = 1.0 +- 0.015 via bisection on signal amplitude
    base_noise = rng.standard_normal((n, 12))
    sig_dir = z[:, [0]] @ rng.standard_normal((1, 12))
    lo, hi = 0.0, 1.0
    v_border = base_noise
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        cand = base_noise + mid * sig_dir
        bm = pysimlr.bottleneck_rank([v_strong0, v_strong1, v_noise, cand],
                                     n_permutations=60, random_state=0)
        pvm = bm["per_view"].iloc[3]
        r_mid = pvm["first_singular_value"] / pvm["threshold_first"]
        if abs(r_mid - 1.0) <= 0.005:
            v_border = cand
            break
        if r_mid < 1.0:
            lo = mid
        else:
            hi = mid
        v_border = cand

    b = pysimlr.bottleneck_rank([v_strong0, v_strong1, v_noise, v_border],
                                n_permutations=60, random_state=0)
    pv = b["per_view"]
    assert {"sigma1_over_tau95", "tau95_rel_se", "borderline"} <= set(pv.columns)
    assert pv.loc[0, "sigma1_over_tau95"] > 1.25 and bool(pv.loc[0, "borderline"]) is False
    assert pv.loc[2, "sigma1_over_tau95"] < 0.85 and bool(pv.loc[2, "borderline"]) is False
    assert abs(pv.loc[3, "sigma1_over_tau95"] - 1.0) <= 0.015
    assert bool(pv.loc[3, "borderline"]) is True

