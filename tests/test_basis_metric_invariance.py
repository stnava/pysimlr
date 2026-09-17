"""
A linear score on projected scores cannot evidence anything about the basis.

For any invertible ``M``, ``span(XVM) = span(XV)``, so ordinary least squares,
ridge and logistic regression fit identical values when ``V`` is replaced by
``VM``: the coefficients absorb ``M^{-1}``. `Strictly Linear Accuracy` --
`first_layer_test_r2` / `first_layer_test_accuracy` -- is such a score, so it
measures the subspace and is blind to the axes.

That is the part that matters, because non-negativity and sparsity are
properties of the axes. In pysimlr's own clinical benchmark the five models sit
within 0.002 of each other on this metric (Friedman p = 0.61), which is the tie
the identity predicts rather than a finding about their bases.

These tests pin the invariance so the metric cannot quietly be read as evidence
about a basis, and pin that the axis-sensitive companion is not invariant.
"""

import numpy as np
import pytest
import torch

from pysimlr.benchmarks.metrics import (axis_sensitive_cross_val_metrics,
                                        cross_val_metrics)


def _case(seed=0, n=200, k=4, classification=False):
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(n, k, generator=g)
    signal = 1.5 * u[:, 0] - u[:, 2] + 0.3 * torch.randn(n, generator=g)
    y = (signal > 0).long().numpy() if classification else signal.numpy()
    split = int(0.75 * n)
    return u[:split], y[:split], u[split:], y[split:]


def _invertible(k, seed):
    g = torch.Generator().manual_seed(seed)
    while True:
        m = torch.randn(k, k, generator=g)
        if abs(float(torch.linalg.det(m))) > 0.1:
            return m


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_regression_score_is_exactly_invariant(seed):
    """Ordinary least squares is unpenalised, so the invariance is exact."""
    u_tr, y_tr, u_te, y_te = _case(seed=seed)
    m = _invertible(u_tr.shape[1], seed + 100)

    base = cross_val_metrics(u_tr, y_tr, u_te, y_te, False)
    reparam = cross_val_metrics(u_tr @ m, y_tr, u_te @ m, y_te, False)

    assert reparam["test"] == pytest.approx(base["test"], abs=1e-9), (
        "OLS moved under V -> VM; it should be exactly invariant")
    assert reparam["train"] == pytest.approx(base["train"], abs=1e-9)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_classification_score_is_only_approximately_invariant(seed):
    """
    The invariance is exact only without a penalty.

    ``scikit-learn``'s ``LogisticRegression`` regularises by default (C=1), and
    an L2 penalty is defined in the coordinates, so it is *not* invariant under
    ``V -> VM``. Measured drift is around 0.02 on this fixture, against
    0.000e+00 for OLS and 0.000e+00 for the same model at C=1e6.

    That is worth pinning rather than glossing: on a classification benchmark
    the metric carries a little axis sensitivity, and it comes from the
    regularizer rather than from anything about the basis -- so it is an
    artifact, not a signal, and it can exceed the differences between methods
    that the metric is being used to compare.
    """
    u_tr, y_tr, u_te, y_te = _case(seed=seed, classification=True)
    m = _invertible(u_tr.shape[1], seed + 100)

    base = cross_val_metrics(u_tr, y_tr, u_te, y_te, True)
    reparam = cross_val_metrics(u_tr @ m, y_tr, u_te @ m, y_te, True)

    drift = abs(reparam["test"] - base["test"])
    assert drift < 0.15, (
        f"penalised logistic regression drifted by {drift:.3f} under a "
        "reparametrisation, which is too much to call approximately invariant")


def test_removing_the_penalty_restores_exact_invariance_for_classification():
    """Localises the drift above to the regularizer rather than the link."""
    from sklearn.linear_model import LogisticRegression

    u_tr, y_tr, u_te, y_te = _case(seed=1, classification=True)
    m = _invertible(u_tr.shape[1], 101)

    def score(train, test):
        model = LogisticRegression(max_iter=5000, C=1e6).fit(
            train.numpy(), y_tr.astype(int).ravel())
        return float(model.score(test.numpy(), y_te.astype(int).ravel()))

    assert score(u_tr @ m, u_te @ m) == pytest.approx(score(u_tr, u_te), abs=1e-9)


def test_a_permutation_and_a_column_rescaling_leave_the_regression_score_unchanged():
    """Two reparametrisations that change how a basis *reads* -- the order of
    the components and their scales -- and leave the linear score untouched."""
    u_tr, y_tr, u_te, y_te = _case()
    k = u_tr.shape[1]

    base = cross_val_metrics(u_tr, y_tr, u_te, y_te, False)

    perm = torch.randperm(k, generator=torch.Generator().manual_seed(3))
    permuted = cross_val_metrics(u_tr[:, perm], y_tr, u_te[:, perm], y_te, False)
    assert permuted["test"] == pytest.approx(base["test"], abs=1e-9)

    # Rescaling by up to 100x costs float32 precision -- the invariance is
    # exact in exact arithmetic, and the residual here is roundoff (~1e-7), not
    # sensitivity to the basis.
    scales = torch.tensor([1.0, 10.0, 0.1, 100.0][:k])
    rescaled = cross_val_metrics(u_tr * scales, y_tr, u_te * scales, y_te, False)
    assert rescaled["test"] == pytest.approx(base["test"], abs=1e-6)
    rescaled64 = cross_val_metrics((u_tr * scales).double(), y_tr,
                                   (u_te * scales).double(), y_te, False)
    base64 = cross_val_metrics(u_tr.double(), y_tr, u_te.double(), y_te, False)
    assert rescaled64["test"] == pytest.approx(base64["test"], abs=1e-9), (
        "in float64 the rescaling invariance should be exact")


def test_the_axis_sensitive_score_is_not_invariant():
    """A tree splits on single coordinates, so its fit depends on the axes.
    Without this, reporting it alongside the linear score would be pointless."""
    u_tr, y_tr, u_te, y_te = _case()
    m = _invertible(u_tr.shape[1], 7)

    base = axis_sensitive_cross_val_metrics(u_tr, y_tr, u_te, y_te)
    reparam = axis_sensitive_cross_val_metrics(u_tr @ m, y_tr, u_te @ m, y_te)

    assert abs(reparam["test"] - base["test"]) > 1e-3, (
        "the axis-sensitive metric did not respond to V -> VM, so it is not "
        f"doing its job: {base['test']:.6f} vs {reparam['test']:.6f}")


def test_the_axis_sensitive_score_is_reproducible():
    """It fixes the forest's seed, so repeated evaluation of the same scores
    agrees exactly -- otherwise a difference between two bases could not be
    distinguished from the estimator's own noise."""
    u_tr, y_tr, u_te, y_te = _case()
    a = axis_sensitive_cross_val_metrics(u_tr, y_tr, u_te, y_te)
    b = axis_sensitive_cross_val_metrics(u_tr, y_tr, u_te, y_te)
    assert a == b


def test_both_metrics_are_reported_together():
    """The invariant score must never appear without its axis-sensitive
    companion, or a tie on it will be read as a statement about the basis."""
    from pysimlr.benchmarks.metrics import calculate_all_metrics

    g = torch.Generator().manual_seed(0)
    n, k = 120, 3
    u = torch.randn(n, k, generator=g)
    y = (1.5 * u[:, 0] - u[:, 1]).numpy()
    split = 90
    scores_train = [u[:split, :k]]
    scores_test = [u[split:, :k]]

    metrics = calculate_all_metrics(
        u_pred=u[split:], y_true=y[split:],
        u_train=u[:split], y_train=y[:split],
        first_layer_scores_train=scores_train,
        first_layer_scores_test=scores_test,
    )
    assert "first_layer_test_r2" in metrics
    assert "first_layer_axis_test_r2" in metrics, (
        "the axis-sensitive companion is missing from the metric payload")


if __name__ == "__main__":
    pytest.main([__file__])
