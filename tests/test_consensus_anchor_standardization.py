"""Regression tests for anchored-prediction consistency in compute_shared_consensus.

The anchor branch used to `return` before the shared centering/standardization
block, so an anchored prediction produced a U on a different scale than the
training-time U for the SVD/PCA/ICA mixing algorithms.
"""
import numpy as np
import pytest
import torch

from pysimlr.consensus import compute_shared_consensus

VOLATILE = ["svd", "pca", "ica"]
STABLE = ["avg", "newton"]


def _projections(seed=0, n=60, k=3):
    g = torch.Generator().manual_seed(seed)
    # deliberately off-centre and differently scaled, so a missing
    # standardization step shows up as a mean/std mismatch
    return [
        torch.randn(n, k, generator=g) * 5.0 + 2.0,
        torch.randn(n, k, generator=g) * 3.0 - 1.0,
    ]


@pytest.mark.parametrize("alg", VOLATILE)
def test_anchored_prediction_matches_training_on_identical_data(alg):
    projs = _projections()
    u_train, anchor = compute_shared_consensus(
        projs, mixing_algorithm=alg, k=3, training=True
    )
    u_pred = compute_shared_consensus(
        projs, mixing_algorithm=alg, k=3, training=False, anchor=anchor
    )
    assert anchor is not None
    assert torch.allclose(u_train, u_pred, atol=1e-4), (
        f"{alg}: anchored prediction diverges from training U on identical input"
    )


@pytest.mark.parametrize("alg", VOLATILE)
def test_anchored_prediction_is_standardized(alg):
    projs = _projections()
    _, anchor = compute_shared_consensus(
        projs, mixing_algorithm=alg, k=3, training=True
    )
    u_pred = compute_shared_consensus(
        projs, mixing_algorithm=alg, k=3, training=False, anchor=anchor
    )
    assert torch.allclose(u_pred.mean(0), torch.zeros(3), atol=1e-4)
    assert torch.allclose(u_pred.std(0), torch.ones(3), atol=1e-4)


@pytest.mark.parametrize("alg", STABLE)
def test_non_volatile_mixing_unaffected_by_anchor(alg):
    projs = _projections()
    u_train, anchor = compute_shared_consensus(
        projs, mixing_algorithm=alg, k=3, training=True
    )
    u_pred = compute_shared_consensus(
        projs, mixing_algorithm=alg, k=3, training=False, anchor=anchor
    )
    assert anchor is None
    assert torch.allclose(u_train, u_pred, atol=1e-6)


@pytest.mark.xfail(
    strict=True,
    reason="Known remaining defect: the anchor pins the rotation, but the "
    "centering/scaling moments are still recomputed from each batch instead "
    "of being stored at training time, so anchored prediction remains "
    "batch-composition dependent. Remove this mark once train-time "
    "center/scale are persisted alongside the anchor.",
)
@pytest.mark.parametrize("alg", VOLATILE)
def test_anchor_holds_coordinates_fixed_across_disjoint_batches(alg):
    """The point of the anchor: two disjoint sample sets must land in the
    same coordinate frame, rather than each getting a fresh SVD rotation."""
    projs = _projections(n=120)
    _, anchor = compute_shared_consensus(
        projs, mixing_algorithm=alg, k=3, training=True
    )
    first = [p[:60] for p in projs]
    second = [p[60:] for p in projs]
    kw = dict(mixing_algorithm=alg, k=3, training=False, anchor=anchor)
    u_a = compute_shared_consensus(first, **kw)
    u_b = compute_shared_consensus(second, **kw)
    # same frame => the same projection applied to both halves, so the
    # concatenation must reproduce a single anchored pass, column for column
    u_all = compute_shared_consensus(projs, **kw)
    for j in range(3):
        c = np.corrcoef(
            torch.cat([u_a[:, j], u_b[:, j]]).numpy(), u_all[:, j].numpy()
        )[0, 1]
        assert c > 0.99, f"{alg}: column {j} frame drifted between batches (r={c:.3f})"


def test_single_sample_prediction_does_not_crash():
    projs = _projections()
    _, anchor = compute_shared_consensus(
        projs, mixing_algorithm="svd", k=3, training=True
    )
    one = [p[:1] for p in projs]
    u = compute_shared_consensus(
        one, mixing_algorithm="svd", k=3, training=False, anchor=anchor
    )
    assert u.shape == (1, 3)
    assert torch.isfinite(u).all()


# --- anchor sign alignment (prevents EMA cancellation) -----------------------

def test_align_anchor_columns_signs():
    from pysimlr.consensus import align_anchor_columns

    a = torch.randn(6, 3)
    ref = a * torch.tensor([[1.0, -1.0, 1.0]])
    assert align_anchor_columns(a, ref).tolist() == [[1.0, -1.0, 1.0]]
    # a degenerate (zero-initialised) reference must not flip anything
    assert align_anchor_columns(a, torch.zeros_like(a)).tolist() == [[1.0, 1.0, 1.0]]
    assert align_anchor_columns(a, None).tolist() == [[1.0, 1.0, 1.0]]
    # shape mismatch degrades gracefully rather than raising
    assert align_anchor_columns(a, torch.randn(4, 2)).tolist() == [[1.0, 1.0, 1.0]]


def test_align_anchor_columns_is_idempotent():
    from pysimlr.consensus import align_anchor_columns

    a = torch.randn(8, 4)
    ref = torch.randn(8, 4)
    signs = align_anchor_columns(a, ref)
    assert align_anchor_columns(a * signs, ref).tolist() == [[1.0, 1.0, 1.0, 1.0]]


@pytest.mark.parametrize("alg", VOLATILE)
def test_training_anchor_is_sign_stable_across_steps(alg):
    """Successive training calls must return a consistently oriented basis,
    otherwise an EMA over them cancels toward zero."""
    projs = _projections(n=80)
    k = 3
    ema = torch.zeros(projs[0].shape[1] * len(projs), k)
    for _ in range(25):
        _, new_anchor = compute_shared_consensus(
            projs, mixing_algorithm=alg, k=k, training=True, anchor=ema
        )
        ema = 0.9 * ema + 0.1 * new_anchor
    col_norms = ema.norm(dim=0)
    # each column of an SVD/ICA basis has unit norm; an EMA of consistently
    # oriented unit columns must approach 1, not cancel toward 0
    assert (col_norms > 0.9).all(), f"{alg}: anchor columns cancelled: {col_norms.tolist()}"


def test_anchor_ema_cancels_without_alignment():
    """Guard the premise: without alignment, sign-flipped bases do cancel.
    If this ever stops holding, the alignment above has become unnecessary."""
    basis = torch.linalg.qr(torch.randn(6, 3))[0]
    ema = torch.zeros_like(basis)
    for step in range(25):
        flipped = basis * (1.0 if step % 2 == 0 else -1.0)
        ema = 0.9 * ema + 0.1 * flipped
    assert ema.norm() < 0.5 * basis.norm()


# --- orthogonalize (was accepted and silently ignored) ----------------------

def _max_offdiag_cov(u):
    uc = u - u.mean(dim=0)
    g = uc.t() @ uc / (u.shape[0] - 1)
    return float((g - torch.diag(torch.diag(g))).abs().max())


@pytest.mark.parametrize("alg", ["svd", "pca", "avg", "newton", "ica"])
def test_orthogonalize_decorrelates_the_consensus(alg):
    torch.manual_seed(0)
    projs = [
        torch.randn(80, 3) @ torch.randn(3, 3),
        torch.randn(80, 3) @ torch.randn(3, 3),
    ]
    u = compute_shared_consensus(
        projs, mixing_algorithm=alg, k=3, orthogonalize=True
    )
    assert _max_offdiag_cov(u) < 1e-4, (
        f"{alg}: orthogonalize=True left off-diagonal covariance "
        f"{_max_offdiag_cov(u):.2e}"
    )


def test_orthogonalize_actually_changes_the_result():
    """Guards against the parameter silently reverting to a no-op."""
    torch.manual_seed(1)
    projs = [torch.randn(60, 3) @ torch.randn(3, 3) for _ in range(2)]
    off = compute_shared_consensus(projs, mixing_algorithm="avg", k=3, orthogonalize=False)
    on = compute_shared_consensus(projs, mixing_algorithm="avg", k=3, orthogonalize=True)
    assert not torch.allclose(off, on, atol=1e-4)
    assert _max_offdiag_cov(off) > 10 * _max_offdiag_cov(on)


def test_orthogonalize_is_skipped_rather_than_crashing_when_underdetermined():
    projs = [torch.randn(2, 4), torch.randn(2, 4)]
    u = compute_shared_consensus(projs, mixing_algorithm="avg", k=4, orthogonalize=True)
    assert u.shape == (2, 4)
    assert torch.isfinite(u).all()


# --- topology return type must not change between train and predict --------

@pytest.mark.parametrize("topology", ["star", "loo", "graph"])
def test_topology_return_type_is_stable_across_train_and_predict(topology):
    torch.manual_seed(2)
    projs = [torch.randn(50, 3) for _ in range(3)]
    kw = dict(mixing_algorithm="svd", k=3, topology=topology)
    if topology == "graph":
        kw["path_graph"] = {0: [1], 1: [0, 2], 2: [1]}
    u_train, anchor = compute_shared_consensus(projs, training=True, **kw)
    u_pred = compute_shared_consensus(projs, training=False, anchor=anchor, **kw)
    assert type(u_train) is type(u_pred), (
        f"{topology}: training returned {type(u_train).__name__} but prediction "
        f"returned {type(u_pred).__name__} -- prediction used a different estimator"
    )
    if isinstance(u_train, list):
        assert len(u_train) == len(u_pred) == len(projs)
