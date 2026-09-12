import pytest
import numpy as np
import scipy.sparse as sp
import torch

from pysimlr import (
    SparseGraphResolvent,
    create_grid_graph_laplacian,
    create_laplacian_resolvent_operator,
    create_spatial_smoothing_operator,
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


def test_create_spatial_smoothing_operator():
    mask = np.ones((10, 10), dtype=bool)
    S_op = create_spatial_smoothing_operator(mask, lambda_val=0.05, connectivity=8)
    assert isinstance(S_op, SparseGraphResolvent)
    assert S_op.P == 100
    assert S_op.shape == (100, 100)


def test_simlr_integration_with_sparse_resolvent():
    N = 30
    P = 64
    mask = np.ones((8, 8), dtype=bool)

    S_op = create_spatial_smoothing_operator(mask, lambda_val=0.05)

    torch.manual_seed(42)
    X = torch.randn(N, P)
    Y = torch.randn(N, 1)

    res = simlr(
        data_matrices=[X, Y],
        k=1,
        iterations=10,
        smoothing_matrices=[S_op, None],
        sparseness_quantile=0.15,
        optimizer_type='lars',
        verbose=False
    )

    assert 'u' in res
    assert 'v' in res
    assert res['u'].shape == (N, 1)
    assert res['v'][0].shape == (P, 1)
    assert not torch.isnan(res['u']).any()
    assert not torch.isnan(res['v'][0]).any()
