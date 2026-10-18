import pytest
import torch
import numpy as np
import scipy.sparse as sp

from pysimlr.optimizers import (
    ArmijoGradient,
    HybridAdam,
    NSALBFGSB,
    backtracking_linesearch,
    create_optimizer,
)
from pysimlr.simlr import simlr, predict_shared_latent
from pysimlr.sparsification import GAUGE_FREE_ENERGIES, energy_needs_unit_columns
from pysimlr.svd import safe_pca, multiscale_svd
from pysimlr.sparse import create_graph_laplacian, create_grid_graph_laplacian
from pysimlr.nsa_backend import backend_report


class TestOptimizerDescentAndMomentumReset:
    """F1.4: Line search descent condition & momentum reset."""

    def test_backtracking_linesearch_rejects_uphill_direction(self):
        """Line search must return 0.0 when slope_term >= 0."""
        v = torch.randn(10, 2)
        ascent_grad = torch.ones_like(v)
        # Search direction pointing in same direction as ascent gradient (uphill)
        uphill_direction = torch.ones_like(v)

        def dummy_energy(x):
            return float(torch.sum(x ** 2))

        step_size = backtracking_linesearch(
            v_current=v,
            descent_direction=uphill_direction,
            ascent_gradient=ascent_grad,
            energy_function=dummy_energy,
            initial_step_size=1.0,
        )
        assert step_size == 0.0, "backtracking_linesearch accepted an uphill search direction"

    def test_armijo_gradient_resets_uphill_momentum(self):
        """ArmijoGradient must detect uphill momentum and reset buffer to fresh descent gradient."""
        v = torch.zeros(6, 2)
        opt = ArmijoGradient("armijo_gradient", [v], learning_rate=0.1)

        # Step 1: positive descent gradient creates positive momentum
        g1 = torch.ones(6, 2)
        opt.step(0, v, g1)
        assert torch.all(opt.state[0]["momentum"] > 0)

        # Step 2: strong negative descent gradient makes stale positive momentum point uphill
        # < -descent_gradient, search_direction > = < -(-10), +momentum > > 0 (uphill!)
        g2 = -10.0 * torch.ones(6, 2)
        v_next = opt.step(0, v, g2)

        # Buffer should have been reset to g2 (cloned)
        assert torch.all(opt.state[0]["momentum"] < 0)
        # Iterate must have moved in direction of g2 (negative)
        assert torch.all(v_next < v)

    def test_hybrid_adam_resets_uphill_momentum(self):
        """HybridAdam must reset state['m'] when momentum points uphill with respect to descent gradient."""
        v = torch.zeros(6, 2)
        opt = HybridAdam("hybrid_adam", [v], learning_rate=0.1)

        # Step 1: establish positive momentum
        g1 = torch.ones(6, 2)
        opt.step(0, v, g1)
        assert torch.all(opt.state[0]["m"] > 0)

        # Step 2: reverse gradient sharply
        g2 = -10.0 * torch.ones(6, 2)
        v_next = opt.step(0, v, g2)

        # state['m'] should be reset and point negative
        assert torch.all(opt.state[0]["m"] < 0)
        assert torch.all(v_next < v)

    def test_nsalbfgsb_positivity_derivation(self):
        """F1.6: NSALBFGSB derives self.nonneg from positivity parameter."""
        v = [torch.randn(10, 2)]
        try:
            opt_either = NSALBFGSB("nsa_lbfgsb", v, positivity="either")
            assert opt_either.nonneg is False, "positivity='either' should set nonneg=False"

            opt_pos = NSALBFGSB("nsa_lbfgsb", v, positivity="positive")
            assert opt_pos.nonneg is True, "positivity='positive' should set nonneg=True"

            opt_nonneg = NSALBFGSB("nsa_lbfgsb", v, positivity="nonnegative")
            assert opt_nonneg.nonneg is True, "positivity='nonnegative' should set nonneg=True"

            opt_hard = NSALBFGSB("nsa_lbfgsb", v, positivity="hard")
            assert opt_hard.nonneg is True, "positivity='hard' should set nonneg=True"
        except ImportError:
            pytest.skip("nsa_flow backend not available for NSALBFGSB instantiation")


class TestSimlrCalibrationAndEnergyTracking:
    """F1.2 & F1.3: Weight calibration, initial energy tracking, and degenerate exclusion."""

    def test_simlr_calibrates_weights_at_init(self):
        """Normalizing and penalty weights must be calibrated before iteration 0."""
        torch.manual_seed(42)
        x1 = torch.randn(40, 15)
        x2 = torch.randn(40, 10)

        res = simlr([x1, x2], k=2, iterations=5, energy_type="acc")
        # Normalizing weights should not remain uncalibrated [1.0, 1.0]
        norm_w = res["normalizing_weights"]
        assert len(norm_w) == 2
        assert not (norm_w[0] == 1.0 and norm_w[1] == 1.0), "Weights were not calibrated at init"

        # Initial energy should be recorded
        assert "best_energy" in res
        assert "energy_start" in res
        assert np.isfinite(res["best_energy"])
        assert res["best_iteration"] is not None
        assert res["best_iteration"] >= 0

    def test_simlr_nonneg_u_consistency(self):
        """F1.5: nonneg_u=True produces non-negative final u and predict_shared_latent."""
        torch.manual_seed(42)
        x1 = torch.randn(50, 12)
        x2 = torch.randn(50, 8)

        res = simlr([x1, x2], k=2, iterations=5, nonneg_u=True)
        assert res.get("nonneg_u") is True

        u = res["u"]
        if isinstance(u, list):
            for u_i in u:
                assert torch.all(u_i >= -1e-6), "Final u violates nonneg_u=True"
        else:
            assert torch.all(u >= -1e-6), "Final u violates nonneg_u=True"

        # Out-of-sample prediction must also be non-negative
        x1_new = torch.randn(15, 12)
        x2_new = torch.randn(15, 8)
        u_pred = predict_shared_latent([x1_new, x2_new], res)
        if isinstance(u_pred, list):
            for u_p in u_pred:
                assert torch.all(u_p >= -1e-6), "Predicted u violates nonneg_u=True"
        else:
            assert torch.all(u_pred >= -1e-6), "Predicted u violates nonneg_u=True"


class TestSVDUtilitiesCorrectness:
    """F1.8: safe_pca 1D support and multiscale_svd safe row slicing."""

    def test_safe_pca_supports_1d_input(self):
        """safe_pca must accept 1D tensors (N,) without IndexError."""
        x_1d = torch.randn(40)
        res = safe_pca(x_1d, nc=1)
        assert res["u"].shape == (40, 1)
        assert res["v"].shape == (1, 1)
        assert res["s"].shape == (1,)

        # Constant 1D input
        x_const = torch.ones(30)
        res_const = safe_pca(x_const, nc=1)
        assert res_const["u"].shape == (30, 1)
        assert res_const["v"].shape == (1, 1)
        assert res_const["s"].shape == (1,)

    def test_multiscale_svd_safe_row_slicing(self):
        """multiscale_svd must handle cases where available components < actual_nev without crash."""
        x = torch.randn(20, 2)  # p=2, max possible rank is 2
        # Request nev=5 on p=2 data
        res = multiscale_svd(x, r=[1.0, 2.0], locn=5, nev=5, knn=3)
        evals = res["evals_vs_scale"]
        # actual_nev will be min(nev, max_rank) = 2
        assert evals.shape == (2, 2)
        assert not torch.all(torch.isnan(evals))


class TestGraphLaplacianMaskForwarding:
    """F1.9: Graph Laplacian mask forwarding of scale_degree and normalized."""

    def test_mask_forwards_scale_degree_and_normalized(self):
        """create_graph_laplacian with source_type='mask' forwards scale_degree and normalized."""
        mask = np.ones((6, 6), dtype=bool)

        # Standard unscaled, unnormalized
        L_std = create_graph_laplacian(mask, source_type="mask", scale_degree=False, normalized=False)
        assert isinstance(L_std, sp.csr_matrix)
        assert L_std.shape == (36, 36)

        # Scaled degree
        L_scaled = create_graph_laplacian(mask, source_type="mask", scale_degree=True, normalized=False)
        assert isinstance(L_scaled, sp.csr_matrix)
        assert L_scaled.shape == (36, 36)

        # Normalized Laplacian: diagonal should be 1.0 for connected nodes
        L_norm = create_graph_laplacian(mask, source_type="mask", scale_degree=False, normalized=True)
        assert isinstance(L_norm, sp.csr_matrix)
        diag = L_norm.diagonal()
        assert np.allclose(diag, 1.0, atol=1e-5), "Normalized Laplacian diagonal is not 1.0"

    def test_create_grid_graph_laplacian_normalized(self):
        """create_grid_graph_laplacian directly supports normalized=True."""
        mask = np.ones((5, 5), dtype=bool)
        L_norm = create_grid_graph_laplacian(mask, normalized=True)
        diag = L_norm.diagonal()
        assert np.allclose(diag, 1.0, atol=1e-5)


class TestGaugeFreeEnergiesAndBackendReport:
    """F1.7 and F1.10: GAUGE_FREE_ENERGIES and backend_report default optimizer."""

    def test_recon_energies_in_gauge_free_set(self):
        """recon_r2 and recon must be in GAUGE_FREE_ENERGIES."""
        assert "recon_r2" in GAUGE_FREE_ENERGIES
        assert "recon" in GAUGE_FREE_ENERGIES
        assert "regression" in GAUGE_FREE_ENERGIES

        assert energy_needs_unit_columns("recon_r2") is False
        assert energy_needs_unit_columns("recon") is False
        assert energy_needs_unit_columns("regression") is False
        assert energy_needs_unit_columns("acc") is True

    def test_backend_report_default_optimizer_lbfgsb(self):
        """backend_report(extended=True) reports 'lbfgsb' for nsa_flow >= 3.0."""
        rep = backend_report(extended=True)
        if rep.get("available") and rep.get("version"):
            v_tuple = tuple(int(x) for x in rep["version"].split(".")[:2] if x.isdigit())
            if v_tuple >= (3, 0):
                assert rep["default_optimizer"] == "lbfgsb"
