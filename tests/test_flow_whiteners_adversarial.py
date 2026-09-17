import pytest
import torch
import numpy as np
import pandas as pd

from antstorch.lamnr_flows import lamnr_flows_whitener, apply_lamnr_flows_whitener
from pysimlr import FlowWhitener, flow_whiten_matrix


class TestFlowWhitenersAdversarial:
    """
    Adversarial stress test harness for FlowWhitener and lamnr_flows_whitener.
    Targeting multi-view unequal dimensions, rank deficits, reconstruction accuracy,
    alignment loss types, and single-sample (N=1) prediction.
    """

    @pytest.fixture
    def multiview_unequal_data(self):
        """Generate multi-view data with unequal dimensions (D1=5, D2=15, D3=30)."""
        np.random.seed(42)
        N = 50
        D1, D2, D3 = 5, 15, 30
        df1 = pd.DataFrame(np.random.randn(N, D1), columns=[f"v1_{i}" for i in range(D1)])
        df2 = pd.DataFrame(np.random.randn(N, D2), columns=[f"v2_{i}" for i in range(D2)])
        df3 = pd.DataFrame(np.random.randn(N, D3), columns=[f"v3_{i}" for i in range(D3)])
        return [df1, df2, df3]

    @pytest.fixture
    def rank_deficient_data(self):
        """Generate multi-view data with extreme rank deficiency, collinearity, and zero-variance features."""
        np.random.seed(101)
        N = 60
        D1, D2, D3 = 5, 15, 30
        z_latent = np.random.randn(N, 2)

        # View 1: rank 2 + small noise
        v1 = z_latent @ np.random.randn(2, D1) + 1e-4 * np.random.randn(N, D1)

        # View 2: exactly collinear columns
        v2_base = z_latent @ np.random.randn(2, 2)
        v2 = np.zeros((N, D2))
        v2[:, :2] = v2_base
        for j in range(2, D2):
            v2[:, j] = v2_base[:, 0] * (j + 0.5)

        # View 3: rank 1 + constant column
        v3 = np.zeros((N, D3))
        v3[:, 0] = z_latent[:, 0]
        for j in range(1, D3 - 1):
            v3[:, j] = z_latent[:, 0] * 2.0
        v3[:, -1] = 0.0  # constant zero-variance column

        df1 = pd.DataFrame(v1, columns=[f"v1_{i}" for i in range(D1)])
        df2 = pd.DataFrame(v2, columns=[f"v2_{i}" for i in range(D2)])
        df3 = pd.DataFrame(v3, columns=[f"v3_{i}" for i in range(D3)])
        return [df1, df2, df3]

    @pytest.mark.parametrize("loss_type", ["barlow_twins_align", "info_nce", "vicreg"])
    def test_multiview_unequal_dimensions_and_losses(self, multiview_unequal_data, loss_type):
        """
        Verify FlowWhitener handles multi-view data with unequal dimensions (D1=5, D2=15, D3=30)
        across alignment loss formulations (barlow_twins_align, info_nce, vicreg).
        """
        dfs = multiview_unequal_data
        pca_dim = 4
        fw = FlowWhitener(
            K=2,
            max_iter=10,
            batch_size=25,
            base_distribution="GaussianPCA",
            pca_latent_dimension=pca_dim,
            penalty_type=loss_type,
            output_space="whitened"
        )
        res = fw.fit_transform(dfs)

        assert isinstance(res, list)
        assert len(res) == 3
        for r in res:
            assert isinstance(r, pd.DataFrame)
            assert r.shape == (50, pca_dim)
            assert np.all(np.isfinite(r.values))
            assert not np.isnan(r.values).any()

    def test_rank_deficit_and_extreme_collinearity(self, rank_deficient_data):
        """
        Verify FlowWhitener numerical stability under rank deficits, collinearity,
        and zero-variance features.
        """
        dfs = rank_deficient_data
        fw = FlowWhitener(
            K=2,
            max_iter=10,
            batch_size=20,
            base_distribution="GaussianPCA",
            pca_latent_dimension=3,
            penalty_type="barlow_twins_align"
        )
        res = fw.fit_transform(dfs)

        assert len(res) == 3
        for r in res:
            assert r.shape == (60, 3)
            assert np.all(np.isfinite(r.values))
            assert not np.isnan(r.values).any()

    def test_forward_and_inverse_reconstruction_accuracy_latent_z(self, multiview_unequal_data):
        """
        Verify exact bijective invertibility (< 1e-5) in latent space z for multi-view
        unequal dimensions across torch.Tensor and pd.DataFrame representations.
        """
        dfs = multiview_unequal_data
        fw = FlowWhitener(
            K=2,
            max_iter=10,
            batch_size=25,
            base_distribution="GaussianPCA",
            pca_latent_dimension=4,
            output_space="z"
        )
        fw.fit(dfs)

        # 1. Test DataFrame round-trip
        z_dfs = fw.transform(dfs, output_space="z")
        rec_dfs = fw.inverse_transform(z_dfs, input_space="z")
        for orig, rec in zip(dfs, rec_dfs):
            assert rec.shape == orig.shape
            max_diff = np.max(np.abs(rec.values - orig.values))
            assert max_diff < 1e-5, f"DataFrame z-space reconstruction error too high: {max_diff}"

        # 2. Test Tensor round-trip
        tensors = [torch.tensor(df.values, dtype=torch.float32) for df in dfs]
        z_tensors = fw.transform(tensors, output_space="z")
        rec_tensors = fw.inverse_transform(z_tensors, input_space="z")
        for orig_t, rec_t in zip(tensors, rec_tensors):
            assert isinstance(rec_t, torch.Tensor)
            assert rec_t.shape == orig_t.shape
            max_diff = (rec_t - orig_t).abs().max().item()
            assert max_diff < 1e-5, f"Tensor z-space reconstruction error too high: {max_diff}"

    @pytest.mark.xfail(
        strict=False,
        reason="ANTsTorch apply_lamnr_flows_whitener.py:171 bug: uses L.T instead of L in _z_from_whitened_full"
    )
    def test_whitened_full_reconstruction_upstream_defect(self):
        """
        Empirical demonstration of upstream ANTsTorch defect in _z_from_whitened_full.
        Due to multiplying by L.T instead of L, inverse reconstruction in whitened_full
        space exhibits an error of ~1.5 rather than machine precision (< 1e-5).
        """
        np.random.seed(42)
        df = pd.DataFrame(np.random.randn(25, 4), columns=[f"x_{i}" for i in range(4)])
        out = lamnr_flows_whitener(
            [df],
            K=2,
            max_iter=5,
            batch_size=15,
            val_batch_size=15,
            base_distribution="GaussianPCA",
            pca_latent_dimension=3,
            early_stop_enabled=False
        )
        w_full = apply_lamnr_flows_whitener(out, [df], direction="forward", output_space="whitened_full")
        rec_full = apply_lamnr_flows_whitener(out, w_full, direction="inverse", input_space="whitened_full")
        max_diff = np.max(np.abs(rec_full[0].values - df.values))
        assert max_diff < 1e-4, f"whitened_full reconstruction difference too large: {max_diff}"

    def test_single_sample_inference_n1(self, multiview_unequal_data):
        """
        Verify single sample prediction (N=1) across FlowWhitener and apply_lamnr_flows_whitener.
        Ensures batch slicing, ActNorm evaluation mode, and output shaping work seamlessly for N=1.
        """
        dfs = multiview_unequal_data
        fw = FlowWhitener(
            K=2,
            max_iter=10,
            batch_size=25,
            base_distribution="GaussianPCA",
            pca_latent_dimension=4,
            output_space="whitened"
        )
        fw.fit(dfs)

        # 1. Single sample DataFrame (1, D)
        single_dfs = [df.iloc[[0]] for df in dfs]
        out_single = fw.transform(single_dfs)
        assert len(out_single) == 3
        for o in out_single:
            assert isinstance(o, pd.DataFrame)
            assert o.shape == (1, 4)
            assert np.all(np.isfinite(o.values))

        # 2. Single sample inverse in z-space
        z_single = fw.transform(single_dfs, output_space="z")
        rec_single = fw.inverse_transform(z_single, input_space="z")
        for orig_s, rec_s in zip(single_dfs, rec_single):
            assert rec_s.shape == orig_s.shape
            max_diff = np.max(np.abs(rec_s.values - orig_s.values))
            assert max_diff < 1e-5, f"N=1 reconstruction error too high: {max_diff}"

        # 3. Single sample Tensor (1, D)
        single_tensors = [torch.tensor(df.iloc[[0]].values, dtype=torch.float32) for df in dfs]
        out_t = fw.transform(single_tensors)
        for ot in out_t:
            assert isinstance(ot, torch.Tensor)
            assert ot.shape == (1, 4)

        # 4. Direct apply_lamnr_flows_whitener with N=1
        applied_n1 = apply_lamnr_flows_whitener(
            fw.trainer_output,
            single_dfs,
            direction="forward",
            output_space="whitened"
        )
        assert len(applied_n1) == 3
        for a in applied_n1:
            assert a.shape == (1, 4)

    def test_1d_input_contract(self):
        """
        Verify that 1D array/tensor without sample dimension (D,) is rejected,
        while 2D single sample (1, D) is accepted.
        """
        fw = FlowWhitener(K=2, max_iter=5, batch_size=10, pca_latent_dimension=2)
        x_2d = torch.randn(15, 4)
        fw.fit(x_2d)

        # 1D tensor should raise an exception (IndexError or ValueError)
        with pytest.raises((IndexError, ValueError)):
            fw.transform(torch.randn(4))

        # 2D single-row tensor (1, 4) should succeed
        out = fw.transform(torch.randn(1, 4))
        assert out.shape == (1, 2)

    def test_multiview_diaggaussian_unequal_dims_guard(self, multiview_unequal_data):
        """
        Verify that DiagGaussian base distribution with unequal multi-view dimensions
        safely raises RuntimeError rather than generating silent matrix corruption.
        """
        dfs = multiview_unequal_data
        fw = FlowWhitener(
            K=2,
            max_iter=5,
            batch_size=20,
            base_distribution="DiagGaussian"
        )
        with pytest.raises(RuntimeError, match="DiagGaussian cross-view penalty requires equal dims"):
            fw.fit(dfs)

    def test_flow_whiten_matrix_dictionary_contract_adversarial(self, multiview_unequal_data):
        """
        Verify flow_whiten_matrix dictionary contract under multi-view stress.
        """
        dfs = multiview_unequal_data
        res = flow_whiten_matrix(
            dfs,
            K=2,
            max_iter=10,
            batch_size=25,
            base_distribution="GaussianPCA",
            pca_latent_dimension=4,
            output_space="whitened"
        )
        assert isinstance(res, dict)
        assert "whitened_views" in res
        assert "whitened_dfs" in res
        assert "models" in res
        assert "trainer_output" in res
        assert "metrics" in res
        assert "whitener" in res
        # Multi-view input should not define whitened_matrix (only single matrix does)
        assert "whitened_matrix" not in res
        assert len(res["whitened_views"]) == 3
        assert len(res["whitened_dfs"]) == 3

    @pytest.mark.slow
    def test_unfitted_state_and_mismatched_view_count_errors(self, multiview_unequal_data):
        """
        Verify that FlowWhitener fast-fails with descriptive exceptions when transform/inverse_transform
        are invoked prematurely or with mismatched view counts.
        """
        dfs = multiview_unequal_data
        fw = FlowWhitener()

        # Unfitted call
        with pytest.raises(RuntimeError, match="must be fitted before"):
            fw.transform(dfs)

        with pytest.raises(RuntimeError, match="must be fitted before"):
            fw.inverse_transform(dfs)

        # Fit on 3 views
        fw.fit(dfs)

        # Transform on 2 views (mismatch)
        with pytest.raises(ValueError, match="Number of models .* must match number of views"):
            fw.transform(dfs[:2])
