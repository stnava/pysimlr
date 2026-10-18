import numpy as np
import pytest
from pysimlr import MultiViewPCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

def test_multiview_pca_multi_modal():
    np.random.seed(42)
    n = 100
    p1, p2, p3 = 50, 40, 30
    X1 = np.random.randn(n, p1)
    X2 = np.random.randn(n, p2)
    X3 = np.random.randn(n, p3)
    y = np.random.randn(n)

    # 1. Uniform k across views
    k = 8
    mupca = MultiViewPCA(n_components=k, scale=True, concat=True, random_state=42)
    Z = mupca.fit_transform([X1, X2, X3])

    assert Z.shape == (n, 3 * k)
    assert len(mupca.loadings_) == 3
    for m, p in zip([0, 1, 2], [p1, p2, p3]):
        assert mupca.loadings_[m].shape == (p, k)
        # Check column orthonormality
        v = mupca.loadings_[m]
        gram = v.T @ v
        np.testing.assert_allclose(gram, np.eye(k), atol=1e-5)

    # 2. Variable k per view
    k_list = [5, 4, 3]
    mupca_var = MultiViewPCA(n_components=k_list, scale=True, concat=True, random_state=42)
    Z_var = mupca_var.fit_transform([X1, X2, X3])
    assert Z_var.shape == (n, sum(k_list))

    # 3. List output when concat=False
    mupca_list = MultiViewPCA(n_components=k, scale=True, concat=False)
    Z_list = mupca_list.fit_transform([X1, X2, X3])
    assert isinstance(Z_list, list)
    assert len(Z_list) == 3
    for z_view in Z_list:
        assert z_view.shape == (n, k)

    # 4. Out-of-sample transform
    X1_test = np.random.randn(20, p1)
    X2_test = np.random.randn(20, p2)
    X3_test = np.random.randn(20, p3)
    Z_test = mupca.transform([X1_test, X2_test, X3_test])
    assert Z_test.shape == (20, 3 * k)

def test_multiview_pca_single_view():
    np.random.seed(42)
    n, p = 80, 25
    X = np.random.randn(n, p)
    mupca = MultiViewPCA(n_components=6)
    Z = mupca.fit_transform(X)
    assert Z.shape == (n, 6)
    assert mupca.loadings_[0].shape == (p, 6)
