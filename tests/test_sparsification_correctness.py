"""Numerical-property tests for the indicator/segmentation sparsifiers.

`indicator_opt_both_ways` used to pass the *masked values* returned by
`optimize_indicator_matrix` back in as if they were a 0/1 indicator, which
squared every retained entry and discarded its sign. `rank_based_matrix_segmentation`
dropped its `positivity`/`transpose`/`quantile` arguments on the non-basic path,
and treated "positive" and "negative" identically on the basic path.
"""
import numpy as np
import pytest
import torch

from pysimlr.sparsification import (
    _assignment_indicator,
    indicator_opt_both_ways,
    optimize_indicator_matrix,
    rank_based_matrix_segmentation,
)


def test_retained_values_are_not_squared():
    m = torch.tensor([[2.0, -1.0], [0.5, 3.0]])
    out = indicator_opt_both_ways(m)
    retained = out[out != 0]
    assert sorted(retained.tolist()) == [2.0, 3.0], (
        f"expected the original values 2 and 3, got {retained.tolist()} "
        "(4 and 9 would mean entries were squared)"
    )


def test_negative_matrix_retains_negative_values():
    m = torch.tensor([[-2.0, 1.0], [-0.5, -3.0]])
    out = indicator_opt_both_ways(m)
    retained = out[out != 0]
    assert (retained < 0).all(), (
        f"a predominantly negative matrix must keep negative entries, got {retained.tolist()}"
    )


def test_output_is_a_subset_of_the_input_entries():
    torch.manual_seed(0)
    m = torch.randn(8, 5)
    out = indicator_opt_both_ways(m)
    nz = out != 0
    assert torch.allclose(out[nz], m[nz]), "retained entries were altered"


def test_assignment_is_one_per_row_and_column():
    torch.manual_seed(1)
    for shape in [(6, 4), (4, 6), (5, 5), (1, 3), (3, 1)]:
        ind = _assignment_indicator(torch.randn(*shape))
        assert set(torch.unique(ind).tolist()) <= {0.0, 1.0}
        assert (ind.sum(dim=0) <= 1).all(), shape
        assert (ind.sum(dim=1) <= 1).all(), shape
        assert ind.sum() == min(shape), shape


def test_assignment_is_optimal_not_merely_greedy():
    """A matrix where the greedy column sweep is provably suboptimal."""
    # greedy on columns takes (row0,col0)=10, forcing col1 onto row1 => 10+1=11
    # the optimum is (row1,col0)=9 and (row0,col1)=8 => 17
    m = torch.tensor([[10.0, 8.0], [9.0, 1.0]])
    ind = _assignment_indicator(m)
    assert torch.sum(m * ind).item() == pytest.approx(17.0), (
        f"assignment objective {torch.sum(m * ind).item()} is not optimal"
    )


def test_optimize_indicator_matrix_returns_masked_values():
    m = torch.tensor([[2.0, -1.0], [0.5, 3.0]])
    out = optimize_indicator_matrix(m, preprocess=False)
    assert torch.equal(out, torch.tensor([[2.0, 0.0], [0.0, 3.0]]))


@pytest.mark.parametrize("transpose", [False, True])
def test_segmentation_respects_positivity(transpose):
    torch.manual_seed(2)
    v = torch.randn(10, 6)
    pos = rank_based_matrix_segmentation(
        v, 0.5, basic=True, positivity="positive", transpose=transpose
    )
    neg = rank_based_matrix_segmentation(
        v, 0.5, basic=True, positivity="negative", transpose=transpose
    )
    assert (pos >= 0).all(), "positivity='positive' returned negative entries"
    assert (neg <= 0).all(), "positivity='negative' returned positive entries"
    assert not torch.equal(pos, neg), "positive and negative gave identical output"


@pytest.mark.parametrize("transpose", [False, True])
def test_non_basic_segmentation_respects_positivity_and_transpose(transpose):
    torch.manual_seed(3)
    v = torch.randn(9, 4)
    pos = rank_based_matrix_segmentation(
        v, 0.5, basic=False, positivity="positive", transpose=transpose
    )
    assert pos.shape == v.shape
    assert (pos >= 0).all(), "non-basic path ignored positivity='positive'"
    neg = rank_based_matrix_segmentation(
        v, 0.5, basic=False, positivity="negative", transpose=transpose
    )
    assert (neg <= 0).all(), "non-basic path ignored positivity='negative'"


def test_segmentation_quantile_controls_retained_count():
    torch.manual_seed(4)
    v = torch.randn(6, 10)
    counts = []
    for q in [0.0, 0.5, 0.9]:
        seg = rank_based_matrix_segmentation(
            v, q, basic=True, positivity="either", transpose=False
        )
        counts.append(int((seg != 0).sum()))
    assert counts[0] > counts[1] > counts[2], f"quantile had no effect: {counts}"


def test_segmentation_quantile_one_zeroes_everything():
    v = torch.randn(5, 8)
    seg = rank_based_matrix_segmentation(v, 1.0, basic=True, positivity="either")
    assert int((seg != 0).sum()) == 0


def test_all_zero_input_is_preserved():
    z = torch.zeros(4, 3)
    assert torch.equal(indicator_opt_both_ways(z), z)
    assert torch.equal(
        rank_based_matrix_segmentation(z, 0.5, basic=True), z
    )
