"""Unit tests for loadings_support_auc metric."""
import sys
import os
import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from optimizer_benchmark_synthetic import loadings_support_auc


def test_perfect_support_recovery_single_view():
    """When |v| has high magnitude on planted support and low elsewhere, AUROC is 1.0."""
    p, k = 20, 2
    v = np.zeros((p, k))
    # Planted support: factor 0 has features 0..4, factor 1 has features 5..9
    sup = [{0, 1, 2, 3, 4}, {5, 6, 7, 8, 9}]
    # Perfect alignment
    v[0:5, 0] = 1.0
    v[5:10, 1] = 1.0
    auc = loadings_support_auc(v, sup)
    assert auc == pytest.approx(1.0, abs=1e-6)


def test_permuted_components_hungarian_matching():
    """Hungarian matching matches permuted components to planted factors."""
    p, k = 30, 3
    v = np.zeros((p, k))
    sup = [{0, 1, 2}, {3, 4, 5}, {6, 7, 8}]
    # Permute components: component 0 -> factor 2, component 1 -> factor 0, component 2 -> factor 1
    v[6:9, 0] = 2.5
    v[0:3, 1] = 1.8
    v[3:6, 2] = 3.1
    # Add noise below support magnitude
    v += 0.01 * np.random.default_rng(0).random(v.shape)
    auc = loadings_support_auc(v, sup)
    assert auc == pytest.approx(1.0, abs=1e-4)


def test_multi_view_loadings_support_auc():
    """Verify mean AUROC across multiple views and components."""
    # View 1: p1=15, k=2; View 2: p2=20, k=2
    v1 = np.zeros((15, 2))
    v2 = np.zeros((20, 2))
    sup1 = [{0, 1, 2}, {3, 4, 5}]
    sup2 = [{0, 1, 2, 3}, {4, 5, 6, 7}]

    v1[0:3, 0] = 1.0
    v1[3:6, 1] = 1.0
    v2[0:4, 0] = 1.0
    v2[4:8, 1] = 1.0

    auc = loadings_support_auc([v1, v2], [sup1, sup2])
    assert auc == pytest.approx(1.0, abs=1e-6)


def test_signed_loadings_magnitude():
    """AUROC evaluates |loadings|, so negative values on true support are credited."""
    p, k = 20, 2
    v = np.zeros((p, k))
    sup = [{0, 1, 2}, {3, 4, 5}]
    v[0:3, 0] = -2.0  # negative loadings
    v[3:6, 1] = 2.0   # positive loadings
    auc = loadings_support_auc(v, sup)
    assert auc == pytest.approx(1.0, abs=1e-6)


def test_random_loadings_near_chance():
    """Pure random noise loadings give mean AUROC near 0.5."""
    rng = np.random.default_rng(42)
    p, k = 100, 2
    v = rng.standard_normal((p, k))
    sup = [set(range(0, 20)), set(range(20, 40))]
    auc = loadings_support_auc(v, sup)
    assert 0.40 <= auc <= 0.65
