import pytest
import numpy as np
import scipy.sparse as sp
import torch

from pysimlr import (
    SparseGraphResolvent,
    create_grid_graph_laplacian,
    create_graph_laplacian,
    create_laplacian_resolvent_operator,
    create_spatial_smoothing_operator,
    create_smoothing_operator,
    simlr
)


def test_create_grid_graph_laplacian_2d():
    H, W = 10, 10
    yy, xx = np.ogrid[:H, :W]
    mask = (xx - 4.5)**2 + (yy - 4.5)**2 <= 4.0**2
    P = int(np.sum(mask))

    L_8 = create_grid_graph_laplacian(mask, connectivity=8)
    assert L_8.shape == (P, P)
    assert sp.issparse(L_8)
    diff = L_8 - L_8.T
    assert np.allclose(diff.data, 0, atol=1e-6)
    row_sums = np.array(L_8.sum(axis=1)).flatten()
    assert np.allclose(row_sums, 0, atol=1e-5)

    L_4 = create_grid_graph_laplacian(mask, connectivity=4)
    assert L_4.shape == (P, P)
    diff4 = L_4 - L_4.T
    assert np.allclose(diff4.data, 0, atol=1e-6)

    L_torch = create_grid_graph_laplacian(mask, connectivity=8, return_torch=True)
    assert isinstance(L_torch, torch.Tensor)
    assert L_torch.is_sparse_csr
    assert L_torch.shape == (P, P)


def test_create_grid_graph_laplacian_3d():
    mask_3d = np.ones((6, 6, 6), dtype=bool)
    P = 6 * 6 * 6

    L_26 = create_grid_graph_laplacian(mask_3d, connectivity=26)
    assert L_26.shape == (P, P)
    assert np.allclose((L_26 - L_26.T).data, 0, atol=1e-6)

    L_6 = create_grid_graph_laplacian(mask_3d, connectivity=6)
    assert L_6.shape == (P, P)
    assert np.allclose((L_6 - L_6.T).data, 0, atol=1e-6)


def test_sparse_graph_resolvent_numerical_accuracy():
    mask = np.ones((8, 8), dtype=bool)
    P = 64
    lam = 0.10

    L_sp = create_grid_graph_laplacian(mask, connectivity=8)
    resolvent = SparseGraphResolvent(L_sp, lambda_val=lam)

    I_dense = np.eye(P, dtype=np.float32)
    A_dense = I_dense + lam * L_sp.toarray().astype(np.float32)
    S_exact = np.linalg.inv(A_dense)

    # 1. 1D NumPy
    v_np = np.random.randn(P).astype(np.float32)
    smoothed_op = resolvent @ v_np
    smoothed_exact = S_exact @ v_np
    assert np.allclose(smoothed_op, smoothed_exact, atol=1e-4)

    # 2. 2D Torch tensor (P x 3)
    v_torch = torch.randn(P, 3, dtype=torch.float32)
    smoothed_torch = resolvent @ v_torch
    assert isinstance(smoothed_torch, torch.Tensor)
    assert smoothed_torch.shape == (P, 3)
    assert np.allclose(smoothed_torch.numpy(), S_exact @ v_torch.numpy(), atol=1e-4)

    # 3. to_dense()
    S_op_dense = resolvent.to_dense().numpy()
    assert np.allclose(S_op_dense, S_exact, atol=1e-4)

    # 4. to_sparse_tensor()
    S_sparse_t = resolvent.to_sparse_tensor(threshold=1e-3)
    assert isinstance(S_sparse_t, torch.Tensor)
    assert S_sparse_t.is_sparse_csr


def test_create_graph_laplacian_general_structures():
    # 1. Coordinates (P x 3)
    P = 80
    coords = np.random.randn(P, 3).astype(np.float32)
    L_coords = create_graph_laplacian(coords, k=6)
    assert L_coords.shape == (P, P)
    assert np.allclose((L_coords - L_coords.T).data, 0, atol=1e-6)

    # Test with Gaussian sigma
    L_gauss = create_graph_laplacian(coords, k=6, sigma=1.0)
    assert L_gauss.shape == (P, P)

    # Test normalized Laplacian
    L_norm = create_graph_laplacian(coords, k=6, normalized=True)
    assert L_norm.shape == (P, P)
    assert np.allclose((L_norm - L_norm.T).data, 0, atol=1e-6)

    # 2. Triangular mesh faces (F x 3)
    faces = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [0, 2, 4]], dtype=np.int64)
    L_mesh = create_graph_laplacian(faces)
    assert L_mesh.shape == (5, 5)
    assert np.allclose((L_mesh - L_mesh.T).data, 0, atol=1e-6)

    # 3. Edge list (E x 2)
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int64)
    L_edges = create_graph_laplacian(edges)
    assert L_edges.shape == (4, 4)
    assert np.allclose((L_edges - L_edges.T).data, 0, atol=1e-6)

    # 4. Adjacency matrix
    adj = sp.csr_matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
    L_adj = create_graph_laplacian(adj)
    assert L_adj.shape == (3, 3)
    assert np.allclose(L_adj.toarray(), [[1, -1, 0], [-1, 2, -1], [0, -1, 1]])


def test_create_smoothing_operator_universal():
    # 1. From coordinates
    coords = np.random.randn(50, 3).astype(np.float32)
    S_coords = create_smoothing_operator(coords, lambda_val=0.05, k=5)
    assert isinstance(S_coords, SparseGraphResolvent)
    assert S_coords.shape == (50, 50)

    # Test multiplication
    v = torch.randn(50, 2)
    v_smooth = S_coords @ v
    assert v_smooth.shape == (50, 2)

    # 2. From mesh faces
    faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    S_mesh = create_smoothing_operator(faces, lambda_val=0.1)
    assert S_mesh.shape == (4, 4)

    # 3. From boolean mask
    mask = np.ones((6, 6), dtype=bool)
    S_mask = create_smoothing_operator(mask, lambda_val=0.05)
    assert S_mask.shape == (36, 36)


def test_simlr_integration_with_universal_operator():
    N = 25
    P = 40
    coords = np.random.randn(P, 3).astype(np.float32)
    S_op = create_smoothing_operator(coords, lambda_val=0.05, k=6)

    torch.manual_seed(42)
    X = torch.randn(N, P)
    Y = torch.randn(N, 1)

    res = simlr(
        data_matrices=[X, Y],
        k=1,
        iterations=10,
        smoothing_matrices=[S_op, None],
        sparseness_quantile=0.10,
        optimizer_type='lars',
        verbose=False
    )

    assert 'u' in res
    assert 'v' in res
    assert res['u'].shape == (N, 1)
    assert res['v'][0].shape == (P, 1)
    assert not torch.isnan(res['u']).any()
    assert not torch.isnan(res['v'][0]).any()
