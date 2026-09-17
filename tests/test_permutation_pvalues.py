"""Calibration tests for the permutation p-values.

`simlr_perm` previously reported `scipy.stats.ttest_1samp(null, observed,
alternative='less')`, which tests whether the *mean* of the null lies below the
observed value rather than where the observation falls in the null's upper
tail. On data with no shared structure that rejected at alpha=0.05 in 42% of
runs (measured, 24 replicates) and produced p-values as small as 2e-11.
`paths.permutation_test` used a plain mean(null >= obs), which can return an
unattainable p of exactly 0.
"""
import warnings

import numpy as np
import pytest
import torch

from pysimlr import simlr_perm
from pysimlr.paths import permutation_test


# ---------------------------------------------------------------- basic shape

def test_simlr_perm_reports_permutation_fields():
    torch.manual_seed(0)
    res = simlr_perm(
        [torch.randn(40, 8), torch.randn(40, 6)], k=2, n_perms=5, iterations=4
    )
    st = res["stats"]["sim_0_1"]
    for key in ("observed", "p_value", "n_permutations", "n_exceeding",
                "null_mean", "null_sd", "z_score", "null_distribution"):
        assert key in st, f"missing {key}"
    assert st["n_permutations"] == 5
    assert len(st["null_distribution"]) == 5
    assert "t_stat" not in st, "the misleading t-statistic should be gone"


@pytest.mark.parametrize("n_perms", [1, 5, 19])
def test_p_value_is_in_range_and_respects_the_add_one_floor(n_perms):
    torch.manual_seed(1)
    res = simlr_perm(
        [torch.randn(40, 8), torch.randn(40, 6)], k=2,
        n_perms=n_perms, iterations=4,
    )
    st = res["stats"]["sim_0_1"]
    p = st["p_value"]
    assert 0.0 < p <= 1.0, p
    # a finite permutation set cannot attain p = 0; the floor is 1/(1+n)
    assert p >= 1.0 / (1.0 + n_perms) - 1e-12
    # and the value must equal the add-one formula exactly
    assert p == pytest.approx((1.0 + st["n_exceeding"]) / (1.0 + n_perms))


def test_zero_permutations_yields_nan_not_a_fabricated_p_value():
    torch.manual_seed(2)
    res = simlr_perm(
        [torch.randn(30, 6), torch.randn(30, 5)], k=2, n_perms=0, iterations=4
    )
    st = res["stats"]["sim_0_1"]
    assert np.isnan(st["p_value"])
    assert st["n_permutations"] == 0


# -------------------------------------------------------------- calibration

def test_null_p_values_are_not_concentrated_at_the_floor():
    """Fast calibration screen: under a true null the p-values must spread out.

    The thorough version is `test_null_p_values_are_calibrated_at_the_nominal_level`
    (marked slow). This cheap variant would still catch the old t-test, which
    put 42% of runs at or below 0.05 and produced values as small as 2e-11.
    """
    warnings.filterwarnings("ignore")
    n_perms = 9
    floor = 1.0 / (1.0 + n_perms)
    ps = []
    for rep in range(6):
        torch.manual_seed(1000 + rep)
        x1 = torch.randn(30, 5)
        x2 = torch.randn(30, 4)        # independent => null is true
        st = simlr_perm([x1, x2], k=2, n_perms=n_perms, iterations=3)
        ps.append(st["stats"]["sim_0_1"]["p_value"])
    ps = np.asarray(ps)

    assert (ps >= floor - 1e-12).all(), f"p below the add-one floor: {ps}"
    # Under a true null the expected share at the floor is `floor` itself, so
    # with 8 replicates at most a couple should land there. Counting p <= floor
    # is the rejection count; do NOT use a threshold above the floor, since
    # every floor-hit would then be miscounted as a rejection.
    n_at_floor = int(np.sum(ps <= floor + 1e-12))
    assert n_at_floor <= 3, (
        f"{n_at_floor}/6 runs at the floor p={floor} under a true null "
        f"({np.sort(ps)}); either the p-values are anti-conservative or the "
        f"permutation null has collapsed (see tests/test_rng_isolation.py)"
    )
    assert ps.max() > 0.3, f"no large p-values under the null: {np.sort(ps)}"
    # A degenerate null gives an identical p every replicate.
    assert np.unique(ps).size > 1, f"every replicate returned p={ps[0]}"


@pytest.mark.slow
def test_null_p_values_are_calibrated_at_the_nominal_level():
    """With no shared structure, the type-I error at alpha must be about alpha.

    This is the property the t-test violated most severely.
    """
    warnings.filterwarnings("ignore")
    torch.manual_seed(3)
    n_perms = 19          # floor p = 1/20 = 0.05
    n_reps = 24
    ps = []
    for _ in range(n_reps):
        x1 = torch.randn(40, 8)
        x2 = torch.randn(40, 6)   # independent of x1 => the null is true
        st = simlr_perm([x1, x2], k=2, n_perms=n_perms, iterations=5)
        ps.append(st["stats"]["sim_0_1"]["p_value"])
    ps = np.asarray(ps)

    # Binomial(24, 0.05) has mean 1.2; allow up to 4 rejections before calling
    # this miscalibrated. The old t-test produced ~10 of 24 here.
    n_reject = int(np.sum(ps <= 0.05))
    assert n_reject <= 4, (
        f"{n_reject}/{n_reps} false positives at alpha=0.05 -- p-values are "
        f"anti-conservative. Observed p-values: {np.sort(ps)}"
    )
    # p-values must actually spread out rather than piling up at the floor
    assert ps.max() > 0.5, f"no large p-values under the null: {np.sort(ps)}"
    assert ps.mean() > 0.25, f"mean p={ps.mean():.3f} is far below uniform"


@pytest.mark.slow
def test_strong_shared_signal_is_detected():
    """Sanity check the other direction: real shared structure must reject."""
    warnings.filterwarnings("ignore")
    torch.manual_seed(4)
    z = torch.randn(80, 3)
    x1 = z @ torch.randn(3, 10) + 0.05 * torch.randn(80, 10)
    x2 = z @ torch.randn(3, 8) + 0.05 * torch.randn(80, 8)
    st = simlr_perm([x1, x2], k=3, n_perms=19, iterations=15)["stats"]["sim_0_1"]
    assert st["p_value"] <= 0.10, (
        f"failed to detect a near-noiseless shared latent: p={st['p_value']}, "
        f"observed={st['observed']:.4f}, null_mean={st['null_mean']:.4f}"
    )
    assert st["observed"] > st["null_mean"]


# ------------------------------------------------------- paths.permutation_test

def test_paths_permutation_test_never_returns_zero():
    torch.manual_seed(5)
    z = torch.randn(40, 2)
    res = permutation_test(
        [z @ torch.randn(2, 6), z @ torch.randn(2, 5)],
        k=2, n_permutations=5, iterations=4,
    )
    assert res["p_value"] > 0.0, "p == 0 is not attainable from finite permutations"
    assert res["p_value"] >= 1.0 / 6.0 - 1e-12
    assert res["n_permutations"] == 5
    assert res["p_value"] == pytest.approx((1.0 + res["n_exceeding"]) / 6.0)


def test_paths_permutation_test_zero_permutations_is_nan():
    torch.manual_seed(6)
    res = permutation_test(
        [torch.randn(20, 5), torch.randn(20, 4)], k=2, n_permutations=0, iterations=3
    )
    assert np.isnan(res["p_value"])
    assert res["null_similarities"] == []
