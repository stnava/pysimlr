"""Correctness tests for graph-Laplacian construction and smoothing operators.

Covers four defects:
  * `Tensor.is_sparse` is False for the CSR layout, so `return_torch=True`
    output could not be fed back into `SparseGraphResolvent` or
    `create_graph_laplacian` -- it raised "can't convert SparseCsr layout
    tensor to numpy".
  * Auto-detection silently misread integer voxel coordinates as mesh faces,
    uint8 masks as coordinates, and small square adjacency matrices as
    coordinates.
  * `lambda_val` was not scale-free: inverse-distance weights carry the
    coordinate units, so the same geometry in mm vs um gave 0.41 vs 0.998
    shrinkage.
  * Grid construction used a pure-Python triple loop.
"""
import warnings

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from pysimlr.sparse import (
    SparseGraphResolvent,
    create_graph_laplacian,
    create_grid_graph_laplacian,
    create_smoothing_operator,
)

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------- Laplacian algebra

@pytest.mark.parametrize("normalized", [False, True])
def test_laplacian_rows_sum_to_zero(normalized):
    np.random.seed(0)
    L = create_graph_laplacian(
        np.random.rand(40, 3).astype(np.float32),
        source_type="coordinates", k=5, normalized=normalized,
    )
    if not normalized:
        assert np.allclose(np.array(L.sum(axis=1)).ravel(), 0.0, atol=1e-5)
    # a Laplacian is symmetric positive semi-definite either way
    assert abs(L - L.T).max() < 1e-6
    evals = np.linalg.eigvalsh(L.toarray())
    assert evals.min() > -1e-5, f"not PSD: min eigenvalue {evals.min():.3e}"


def test_scale_degree_normalizes_mean_degree_to_one():
    np.random.seed(1)
    coords = np.random.rand(60, 3).astype(np.float32)
    raw = create_graph_laplacian(coords, source_type="coordinates", k=6)
    scaled = create_graph_laplacian(coords, source_type="coordinates", k=6,
                                    scale_degree=True)
    d_raw, d_scaled = raw.diagonal(), scaled.diagonal()
    # off by default, so the raw Laplacian keeps its natural (unit-dependent) scale
    assert d_raw[d_raw > 0].mean() > 1.5
    assert d_scaled[d_scaled > 0].mean() == pytest.approx(1.0, abs=1e-4)


def test_raw_laplacian_is_the_textbook_one():
    """A 3-node path graph must give [[1,-1,0],[-1,2,-1],[0,-1,1]] unscaled."""
    a = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    L = create_graph_laplacian(a, source_type="adjacency")
    assert np.allclose(L.toarray(), [[1, -1, 0], [-1, 2, -1], [0, -1, 1]])


def test_smoothing_factories_scale_degree_by_default():
    np.random.seed(11)
    coords = np.random.rand(50, 3).astype(np.float32)
    S = create_smoothing_operator(coords, lambda_val=0.05, k=6)
    deg = S._L_sp.diagonal()
    assert deg[deg > 0].mean() == pytest.approx(1.0, abs=1e-4)


# -------------------------------------------------------------- scale freedom

@pytest.mark.parametrize("scale", [0.001, 1.0, 1000.0])
def test_lambda_val_is_invariant_to_coordinate_units(scale):
    np.random.seed(2)
    coords = np.random.rand(200, 3).astype(np.float32)
    v = np.random.randn(200, 1).astype(np.float32)
    ref = create_smoothing_operator(coords, lambda_val=0.05, k=6) @ v
    got = create_smoothing_operator(coords * scale, lambda_val=0.05, k=6) @ v
    assert np.allclose(got, ref, atol=1e-4), (
        f"smoothing changed when coordinates were scaled by {scale}"
    )


def test_smoothing_strength_increases_monotonically_with_lambda():
    np.random.seed(3)
    coords = np.random.rand(150, 3).astype(np.float32)
    v = np.random.randn(150, 1).astype(np.float32)
    norms = [
        float(np.linalg.norm(create_smoothing_operator(coords, lambda_val=lam, k=6) @ v))
        for lam in (0.0, 0.05, 0.5, 5.0)
    ]
    assert norms[0] == pytest.approx(float(np.linalg.norm(v)), rel=1e-5), (
        "lambda_val=0 must be the identity"
    )
    assert norms[0] > norms[1] > norms[2] > norms[3]


def test_lambda_val_comparable_across_weighting_schemes():
    np.random.seed(4)
    coords = np.random.rand(200, 3).astype(np.float32)
    v = np.random.randn(200, 1).astype(np.float32)
    inv = np.linalg.norm(create_smoothing_operator(coords, lambda_val=0.5, k=6) @ v)
    gau = np.linalg.norm(
        create_smoothing_operator(coords, lambda_val=0.5, k=6, sigma=0.2) @ v
    )
    assert abs(inv - gau) / inv < 0.15, (
        f"inverse-distance and Gaussian weights disagree by "
        f"{abs(inv - gau) / inv:.1%} at the same lambda_val"
    )


def test_negative_lambda_is_rejected():
    with pytest.raises(ValueError):
        create_smoothing_operator(np.random.rand(10, 3).astype(np.float32), lambda_val=-1.0)


# ------------------------------------------------------- torch layout support

@pytest.mark.parametrize("layout", ["csr", "coo", "dense"])
def test_torch_laplacian_round_trips_through_the_resolvent(layout):
    np.random.seed(5)
    coords = np.random.rand(25, 3).astype(np.float32)
    L = create_graph_laplacian(coords, source_type="coordinates", k=4, return_torch=True)
    assert L.layout == torch.sparse_csr
    if layout == "coo":
        L = L.to_sparse_coo()
    elif layout == "dense":
        L = L.to_dense()

    S = SparseGraphResolvent(L, lambda_val=0.05)
    out = S @ torch.randn(25, 2)
    assert out.shape == (25, 2)
    assert torch.isfinite(out).all()

    again = create_graph_laplacian(L, source_type="adjacency")
    assert again.shape == (25, 25)


def test_grid_laplacian_torch_output_is_reusable():
    L = create_grid_graph_laplacian(np.ones((6, 6), dtype=bool), return_torch=True)
    S = SparseGraphResolvent(L, lambda_val=0.1)
    assert (S @ torch.randn(36, 1)).shape == (36, 1)


# --------------------------------------------------------- source_type safety

def test_integer_3col_input_is_refused_as_ambiguous():
    coords = np.array([[10, 20, 30], [11, 20, 30], [12, 20, 30]], dtype=np.int64)
    with pytest.raises(ValueError, match="Ambiguous"):
        create_graph_laplacian(coords)
    # explicit intent works, and gives one node per row
    assert create_graph_laplacian(coords, source_type="coordinates").shape == (3, 3)
    # so does casting to float
    assert create_graph_laplacian(coords.astype(np.float32)).shape == (3, 3)


def test_integer_2col_input_is_refused_as_ambiguous():
    with pytest.raises(ValueError, match="Ambiguous"):
        create_graph_laplacian(np.array([[0, 1], [1, 2]], dtype=np.int64))


@pytest.mark.parametrize("dtype", [bool, np.uint8, np.int32])
@pytest.mark.parametrize("ndim", [2, 3])
def test_non_bool_masks_are_detected(dtype, ndim):
    np.random.seed(6)
    shape = (12, 12) if ndim == 2 else (7, 7, 7)
    mask = (np.random.rand(*shape) > 0.3).astype(dtype)
    n_fg = int(mask.astype(bool).sum())
    L = create_graph_laplacian(mask)
    assert L.shape == (n_fg, n_fg), (
        f"{ndim}D {np.dtype(dtype).name} mask was not read as a mask"
    )


def test_small_square_adjacency_is_detected():
    a = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    L = create_graph_laplacian(a)
    assert L.shape == (3, 3)
    # node 1 is connected to both others, nodes 0 and 2 to one each
    deg = L.diagonal()
    assert deg[1] == pytest.approx(2.0 * deg[0], rel=1e-5)


def test_unknown_source_type_is_rejected():
    with pytest.raises(ValueError, match="Unknown source_type"):
        create_graph_laplacian(np.random.rand(5, 3), source_type="nonsense")


# ------------------------------------------------------------- grid structure

@pytest.mark.parametrize("shape,conn", [((12, 10), 4), ((12, 10), 8),
                                        ((6, 7, 5), 6), ((6, 7, 5), 26)])
def test_grid_laplacian_matches_explicit_neighbour_enumeration(shape, conn):
    """Validates the vectorized shift construction against a direct walk."""
    np.random.seed(7)
    mask = np.random.rand(*shape) > 0.35
    got = create_grid_graph_laplacian(mask, connectivity=conn)

    P = int(mask.sum())
    idx = np.full(shape, -1, dtype=np.int64)
    idx[mask] = np.arange(P)
    if len(shape) == 2:
        deltas = ([(-1, 0), (1, 0), (0, -1), (0, 1)] if conn == 4 else
                  [(-1, 0), (1, 0), (0, -1), (0, 1),
                   (-1, -1), (-1, 1), (1, -1), (1, 1)])
    else:
        deltas = [d for d in np.ndindex(3, 3, 3)]
        deltas = [tuple(a - 1 for a in d) for d in deltas]
        deltas = [d for d in deltas if any(d)]
        if conn == 6:
            deltas = [d for d in deltas if sum(a * a for a in d) == 1]
    rows, cols, ws = [], [], []
    for pos in np.ndindex(*shape):
        i1 = idx[pos]
        if i1 < 0:
            continue
        for d in deltas:
            q = tuple(a + b for a, b in zip(pos, d))
            if all(0 <= a < n for a, n in zip(q, shape)) and idx[q] >= 0:
                rows.append(i1); cols.append(idx[q])
                ws.append(1.0 / np.sqrt(sum(a * a for a in d)))
    W = sp.coo_matrix((ws, (rows, cols)), shape=(P, P), dtype=np.float64).tocsr()
    expected = (sp.diags(np.array(W.sum(axis=1)).ravel()) - W).tocsr()

    diff = (got - expected)
    assert (abs(diff).max() if diff.nnz else 0.0) < 1e-5


def test_empty_mask_is_rejected():
    with pytest.raises(ValueError, match="0 foreground"):
        create_grid_graph_laplacian(np.zeros((5, 5), dtype=bool))


def test_single_voxel_mask_is_an_isolated_node():
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True
    L = create_grid_graph_laplacian(mask)
    assert L.shape == (1, 1)
    assert L.toarray()[0, 0] == pytest.approx(0.0)
    # and the resolvent of an isolated node is the identity
    S = SparseGraphResolvent(L, lambda_val=0.5)
    v = torch.randn(1, 1)
    assert torch.allclose(S @ v, v, atol=1e-6)


def test_disconnected_components_do_not_mix():
    mask = np.zeros((3, 9), dtype=bool)
    mask[1, 0:2] = True          # component A
    mask[1, 7:9] = True          # component B, far away
    L = create_grid_graph_laplacian(mask, connectivity=4)
    S = SparseGraphResolvent(L, lambda_val=1.0)
    v = np.array([[1.0], [1.0], [0.0], [0.0]], dtype=np.float32)
    out = S @ v
    assert out[2, 0] == pytest.approx(0.0, abs=1e-6)
    assert out[3, 0] == pytest.approx(0.0, abs=1e-6)


def test_resolvent_preserves_constants_within_a_component():
    """(I + lambda L) 1 = 1 because L annihilates constants, so S 1 = 1."""
    L = create_grid_graph_laplacian(np.ones((7, 7), dtype=bool))
    S = SparseGraphResolvent(L, lambda_val=2.0)
    ones = torch.ones(49, 1)
    assert torch.allclose(S @ ones, ones, atol=1e-4)


def test_resolvent_dimension_mismatch_is_reported():
    L = create_grid_graph_laplacian(np.ones((4, 4), dtype=bool))
    S = SparseGraphResolvent(L, lambda_val=0.1)
    with pytest.raises(ValueError, match="Dimension mismatch"):
        S @ torch.randn(5, 1)
