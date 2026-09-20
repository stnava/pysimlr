import pytest
import numpy as np
import torch
from pysimlr.benchmarks.metrics import support_recovery_score


def _disjoint(p=9, k=3):
    v = torch.zeros(p, k)
    step = p // k
    for j in range(k):
        v[j * step:(j + 1) * step, j] = 1.0
    return v


def test_support_recovery_is_permutation_invariant():
    v = _disjoint()
    assert support_recovery_score([v], [v]) == 1.0
    assert support_recovery_score([v[:, [2, 0, 1]]], [v]) == 1.0


def test_dense_estimate_scores_the_truth_density_not_one():
    v = _disjoint()
    assert support_recovery_score([torch.ones(9, 3)], [v]) == pytest.approx(1 / 3)


def test_dense_truth_is_undefined_not_a_free_win():
    """Guessing dense must not win where there is no support to recover."""
    dense = torch.ones(9, 3)
    assert np.isnan(support_recovery_score([dense], [dense]))
    assert np.isnan(support_recovery_score([_disjoint()], [dense]))


def test_absent_truth_is_nan_not_zero():
    assert np.isnan(support_recovery_score([_disjoint()], [torch.zeros(9, 3)]))
    assert np.isnan(support_recovery_score([], []))


def test_partial_support_is_graded():
    v = _disjoint()
    damaged = v.clone()
    damaged[0, 0] = 0.0
    s = support_recovery_score([damaged], [v])
    assert 0.5 < s < 1.0
