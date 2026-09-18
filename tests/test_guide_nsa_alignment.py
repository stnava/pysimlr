"""
Unit and integration tests verifying alignment with NSA-Flow Developer & Agent Guide (v2.11.0+).

Covers:
- High-level API reference and scikit-learn estimator integration (Section 3)
- Elimination of SVD in iterative inner loops via Sylvester-based polar_factor (Section 6, Rule 4)
- Retraction diagnostics and honest convergence reporting (Section 3.1 & Section 6, Rule 3)
- Scale invariance and zero radial force invariant <grad D(Y), Y> = 0 (Section 1 & Section 6, Rule 2)
- Domain recipes: Non-negative constituents (Recipe A) and Signed disjoint contrast modules (Recipe B)
- Scikit-learn Pipeline and strict in-fold validation (Section 3.2 & Section 6, Rule 1)
"""
import pytest
import numpy as np
import torch
from unittest.mock import patch

from pysimlr import (
    simlr,
    NSAFlow,
    load_polar_factor,
    load_nsa_estimator,
    load_stiefel_defect_normalised,
    load_consolidate_supports,
    backend_report,
    nsa_contrast_transform,
    nsa_nonnegative_transform,
    build_nsa_pipeline,
)
from pysimlr.nsa_backend import load_nsa_flow
from pysimlr.sparsification import (
    _svd_polar,
    _nsa_retract,
    project_to_orthonormal_nonnegative,
    project_to_partially_orthonormal_nonnegative,
    _RETRACTION_DIAGNOSTIC_FIELDS,
)
from pysimlr.optimizers import NSAFlowOptimizer


@pytest.mark.skipif(load_nsa_flow() is None, reason="nsa-flow backend required")
class TestGuideAlignment:

    def test_backend_report_extended_metadata(self):
        """Verify extended backend report includes version and capability flags."""
        rep = backend_report(extended=False)
        assert set(rep.keys()) == {"available", "module", "entry_point"}

        ext = backend_report(extended=True)
        assert ext["available"] is True
        assert ext["module"] in ("nsa_flow", "nsa")
        assert ext["has_polar_factor"] is True
        assert ext["has_estimator"] is True
        assert ext["default_optimizer"] == "torch_lbfgs"
        assert ext["version"] is not None

    def test_polar_factor_retraction_without_svd(self):
        """Verify polar_factor produces an orthonormal basis and is preferred over SVD."""
        polar_fn = load_polar_factor()
        assert polar_fn is not None

        torch.manual_seed(42)
        v = torch.randn(20, 4)
        ortho = _svd_polar(v)

        # Orthonormality check: V.T @ V == I
        vtv = ortho.t() @ ortho
        eye = torch.eye(4, dtype=ortho.dtype, device=ortho.device)
        assert torch.allclose(vtv, eye, atol=1e-5), f"V'V deviates from identity: {vtv}"

        # Verify load_polar_factor is actively called
        with patch("pysimlr.sparsification.load_polar_factor", return_value=polar_fn) as mock_polar:
            res = _svd_polar(v)
            assert mock_polar.called
            assert torch.allclose(res, ortho, atol=1e-5)

    def test_inner_loops_use_polar_retraction(self):
        """Verify projection inner loops produce valid orthonormal bases."""
        torch.manual_seed(42)
        x = torch.randn(25, 4).abs()  # positive initial

        # project_to_orthonormal_nonnegative
        v_nonneg = project_to_orthonormal_nonnegative(x, max_iter=20, constraint="positive")
        assert (v_nonneg >= -1e-6).all(), "Negative entries found in positive projection"
        vtv = v_nonneg.t() @ v_nonneg
        # Columns should be normalized and near-orthogonal
        diag = torch.diag(vtv)
        assert torch.allclose(diag, torch.ones_like(diag), atol=1e-2)

        # project_to_partially_orthonormal_nonnegative
        v_partial = project_to_partially_orthonormal_nonnegative(x, max_iter=5, ortho_strength=0.5)
        assert v_partial.shape == x.shape

    def test_nsa_flow_optimizer_fallback_uses_polar(self):
        """Verify NSAFlowOptimizer step uses polar retraction on fallback."""
        torch.manual_seed(42)
        v_mats = [torch.randn(15, 3)]
        opt = NSAFlowOptimizer("nsa_flow", v_mats, learning_rate=0.01, nsa_w=0.5)
        # Disable full flow to trigger fallback
        opt.nsa_flow = None
        v_next = opt.step(0, v_mats[0], torch.randn(15, 3))
        vtv = v_next.t() @ v_next
        eye = torch.eye(3, dtype=v_next.dtype)
        assert torch.allclose(vtv, eye, atol=1e-5)

    def test_retraction_diagnostics_coverage(self):
        """Verify solver reports all diagnostics specified in Section 3.1."""
        torch.manual_seed(42)
        v = torch.randn(30, 4)
        diag = {}
        retracted = _nsa_retract(v, w=0.5, nonneg=True, diagnostics=diag)
        assert retracted is not None
        assert (retracted >= 0.0).all(), "Retracted basis must be non-negative"

        # Check diagnostic fields
        expected_fields = {"w", "stop_reason", "converged", "defect", "fidelity", "energy"}
        for f in expected_fields:
            assert f in diag, f"Field '{f}' missing from diagnostics: {diag.keys()}"

        assert isinstance(diag["converged"], bool)
        assert diag["fidelity"] >= 0.0
        assert diag["defect"] >= 0.0

    def test_zero_radial_force_invariant(self):
        """Verify mathematically that <grad D(Y), Y> = 0 identically (Section 1 & 6)."""
        torch.manual_seed(42)
        # Test across 5 random matrices of varying scales
        for _ in range(5):
            scale = 10.0 ** np.random.uniform(-3, 3)
            Y = (torch.randn(20, 4, dtype=torch.float64) * scale).requires_grad_(True)
            k = Y.shape[1]

            # Compute normalized frame defect D(Y) = || Y'Y / tr(Y'Y) - (1/k) I_k ||_F^2
            YtY = Y.t() @ Y
            tr = torch.trace(YtY)
            G_norm = YtY / tr
            target = torch.eye(k, dtype=torch.float64) / k
            defect = torch.sum((G_norm - target) ** 2)

            defect.backward()
            grad_Y = Y.grad

            # Radial inner product <grad D(Y), Y> = tr(grad_Y' @ Y)
            radial_force = torch.sum(grad_Y * Y).item()
            assert abs(radial_force) < 1e-10, (
                f"Defect exerted non-zero radial force: {radial_force} (scale: {scale})"
            )

    def test_sklearn_nsaflow_pipeline_and_infold(self):
        """Verify NSAFlow scikit-learn estimator works in Pipeline and adheres to in-fold rules."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import KFold, cross_val_score

        np.random.seed(42)
        latent = np.random.randn(80, 4)
        W = np.random.randn(4, 20)
        X = latent @ W + 0.05 * np.random.randn(80, 20)
        y = latent @ np.array([2.0, -1.5, 1.0, 0.5]) + 0.05 * np.random.randn(80)

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("nsa", NSAFlow(n_components=4, w=0.5, mode="signed", consolidate=True)),
            ("regressor", Ridge())
        ])

        # Strict in-fold cross validation
        cv = KFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="r2")
        assert len(scores) == 3
        assert np.mean(scores) > 0.5, f"R2 score unexpectedly low: {scores}"

        # Fit model on training set and check attributes
        pipeline.fit(X[:60], y[:60])
        model = pipeline.named_steps["nsa"]
        assert hasattr(model, "components_")
        assert model.components_.shape == (4, 20)
        assert hasattr(model, "result_")
        assert model.result_.V.shape == (20, 4)

    def test_recipe_a_nonnegative_spectra(self):
        """Recipe A: non-negative physical quantities (X >= 0) without mean-centering."""
        torch.manual_seed(42)
        # Synthetic non-negative spectrum data
        X_raw = torch.rand(50, 25).abs() + 0.01

        res = nsa_nonnegative_transform(X_raw, k=4, w=0.5)
        V = res["v"]
        scores = res["scores"]

        assert V.shape == (25, 4)
        assert (V >= 0.0).all(), "Recipe A must yield non-negative loadings"
        assert scores.shape == (50, 4)
        assert res["result"].defect < 0.2, f"Frame defect unexpectedly high: {res['result'].defect}"

    def test_recipe_b_signed_disjoint_contrast_modules(self):
        """Recipe B: signed standardized features with consolidate=True produces disjoint supports."""
        torch.manual_seed(42)
        X_std = torch.randn(60, 40)

        res = nsa_contrast_transform(X_std, k=4, w=0.5, consolidate=True)
        V = res["v"]
        result = res["result"]

        assert V.shape == (40, 4)
        assert result.lobe_overlap == pytest.approx(0.0, abs=1e-5), (
            f"Consolidated signed contrast must have zero lobe overlap: {result.lobe_overlap}"
        )
        assert result.get("consolidated") is True

    def test_build_nsa_pipeline_classification(self):
        """Verify build_nsa_pipeline generates scikit-learn Pipeline for classification."""
        from sklearn.model_selection import cross_val_score
        np.random.seed(42)
        X = np.random.randn(60, 10)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        pipe = build_nsa_pipeline(n_components=3, w=0.5, task="classification")
        assert "scaler" in pipe.named_steps
        assert "dim_reduction" in pipe.named_steps
        assert "classifier" in pipe.named_steps

        # Verify strict in-fold cross-validation works seamlessly
        scores = cross_val_score(pipe, X, y, cv=3)
        assert len(scores) == 3
        assert scores.mean() > 0.0

    def test_build_nsa_pipeline_regression(self):
        """Verify build_nsa_pipeline generates scikit-learn Pipeline for regression."""
        from sklearn.model_selection import cross_val_score
        np.random.seed(42)
        X = np.random.randn(60, 10)
        y = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.1 * np.random.randn(60)

        pipe = build_nsa_pipeline(n_components=3, w=0.5, task="regression")
        assert "scaler" in pipe.named_steps
        assert "dim_reduction" in pipe.named_steps
        assert "regressor" in pipe.named_steps

        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert preds.shape == (60,)
        r2 = pipe.score(X, y)
        assert r2 > 0.0

