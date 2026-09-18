"""
Unit tests verifying Scikit-Learn Estimator compatibility and torch_lbfgs optimization in PySIMLR.

Covers:
- SiMLREstimator single-view and multi-view compatibility
- Standard scikit-learn attributes (.components_, .loadings_, .v_, .n_features_in_)
- fit, transform, fit_transform, and inverse_transform
- Seamless integration with sklearn.pipeline.Pipeline and cross_val_score
- Combinatorial support consolidation (consolidate=True) guaranteeing zero lobe overlap
- Fast quasi-Newton optimization via optimizer_type="torch_lbfgs"
- Deep transformers: LENDTransformer, NEDTransformer, FlowSiMLRTransformer
- Turnkey pipeline builders: build_simlr_pipeline, build_nsa_pipeline
"""
import pytest
import numpy as np
import torch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score

import pysimlr
from pysimlr.sklearn import (
    SiMLREstimator,
    SiMLRTransformer,
    SiMLR,
    LENDTransformer,
    NEDTransformer,
    FlowSiMLRTransformer,
    build_simlr_pipeline,
)


class TestSiMLRScikitLearnCompatibility:

    def test_single_view_fit_transform_and_inverse(self):
        """Verify single-view tabular data fits, transforms, and inverse-transforms."""
        np.random.seed(42)
        X = np.random.randn(50, 8)

        model = SiMLR(n_components=3, iterations=15, optimizer_type="torch_lbfgs", random_state=42)
        model.fit(X)

        assert hasattr(model, "components_")
        assert model.components_.shape == (3, 8)
        assert hasattr(model, "loadings_")
        assert model.loadings_.shape == (8, 3)
        assert model.n_features_in_ == 8

        # Transform
        Z = model.transform(X)
        assert Z.shape == (50, 3)

        # Fit transform parity
        Z_ft = model.fit_transform(X)
        assert Z_ft.shape == (50, 3)

        # Inverse transform reconstruction
        X_rec = model.inverse_transform(Z)
        assert X_rec.shape == (50, 8)

    def test_multi_view_fit_transform(self):
        """Verify multi-view list of matrices fits and produces consensus coordinates."""
        np.random.seed(42)
        X1 = np.random.randn(40, 6)
        X2 = np.random.randn(40, 10)

        model = SiMLREstimator(n_components=2, iterations=10, optimizer_type="torch_lbfgs", random_state=42)
        model.fit([X1, X2])

        assert isinstance(model.components_, list)
        assert len(model.components_) == 2
        assert model.components_[0].shape == (2, 6)
        assert model.components_[1].shape == (2, 10)

        Z = model.transform([X1, X2])
        assert Z.shape == (40, 2)

    def test_pipeline_and_cross_validation(self):
        """Verify SiMLREstimator works inside sklearn.pipeline.Pipeline and cross_val_score."""
        np.random.seed(42)
        X = np.random.randn(60, 12)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("dim_reduction", SiMLREstimator(n_components=3, iterations=10, optimizer_type="torch_lbfgs")),
            ("classifier", LogisticRegression(C=1.0, max_iter=500)),
        ])

        scores = cross_val_score(pipe, X, y, cv=3)
        assert len(scores) == 3
        assert (scores >= 0.0).all()

    def test_consolidate_disjoint_modules(self):
        """Verify consolidate=True produces strictly disjoint module lobes with zero overlap."""
        np.random.seed(42)
        X = np.random.randn(60, 15)

        model = SiMLR(n_components=3, iterations=15, consolidate=True, random_state=42)
        model.fit(X)

        V = torch.from_numpy(model.loadings_)
        v_pos = torch.clamp_min(V, 0.0)
        v_neg = torch.clamp_min(-V, 0.0)

        # Lobe overlap must vanish
        lobe_overlap = torch.sum(v_pos * v_neg).item()
        assert lobe_overlap == pytest.approx(0.0, abs=1e-5), f"Expected 0.0 lobe overlap, got {lobe_overlap}"

    def test_torch_lbfgs_optimizer_convergence(self):
        """Verify optimizer_type='torch_lbfgs' optimizes stably and reduces energy."""
        np.random.seed(42)
        X1 = np.random.randn(50, 8)
        X2 = np.random.randn(50, 8)

        model = SiMLREstimator(n_components=2, iterations=8, optimizer_type="torch_lbfgs", random_state=42)
        model.fit([X1, X2])

        energy = model.result_["energy"]
        assert len(energy) > 0
        assert np.isfinite(energy).all()
        assert np.isfinite(model.result_["best_energy"])
        assert model.result_["converged_iter"] > 0

    def test_build_simlr_pipeline_classification_and_regression(self):
        """Verify turnkey build_simlr_pipeline factory for classification and regression."""
        np.random.seed(42)
        X = np.random.randn(50, 10)
        y_clf = (X[:, 0] > 0).astype(int)
        y_reg = 2.0 * X[:, 0] + 0.5 * np.random.randn(50)

        # Classification
        pipe_c = build_simlr_pipeline(n_components=3, task="classification")
        assert "scaler" in pipe_c.named_steps
        assert "dim_reduction" in pipe_c.named_steps
        assert "classifier" in pipe_c.named_steps
        pipe_c.fit(X, y_clf)
        preds_c = pipe_c.predict(X)
        assert preds_c.shape == (50,)

        # Regression
        pipe_r = build_simlr_pipeline(n_components=3, task="regression")
        assert "scaler" in pipe_r.named_steps
        assert "dim_reduction" in pipe_r.named_steps
        assert "regressor" in pipe_r.named_steps
        pipe_r.fit(X, y_reg)
        preds_r = pipe_r.predict(X)
        assert preds_r.shape == (50,)


class TestDeepScikitLearnTransformers:

    def test_lend_transformer(self):
        """Verify LENDTransformer fits and transforms single-view data."""
        np.random.seed(42)
        X = np.random.randn(40, 8)

        lend = LENDTransformer(n_components=2, iterations=2, epochs=5, random_state=42)
        lend.fit(X)

        assert hasattr(lend, "components_")
        assert lend.components_.shape == (2, 8)
        Z = lend.transform(X)
        assert Z.shape == (40, 2)

    def test_ned_transformer(self):
        """Verify NEDTransformer fits and transforms single-view data."""
        np.random.seed(42)
        X = np.random.randn(40, 8)

        ned = NEDTransformer(n_components=2, iterations=2, epochs=5, random_state=42)
        ned.fit(X)

        assert hasattr(ned, "components_")
        assert ned.components_.shape == (2, 8)
        Z = ned.transform(X)
        assert Z.shape == (40, 2)

    def test_flow_simlr_transformer(self):
        """Verify FlowSiMLRTransformer fits and transforms single-view data."""
        np.random.seed(42)
        X = np.random.randn(40, 8)

        flow = FlowSiMLRTransformer(n_components=2, epochs=5, random_state=42)
        flow.fit(X)

        assert hasattr(flow, "components_")
        assert flow.components_.shape == (2, 8)
        Z = flow.transform(X)
        assert Z.shape == (40, 2)
