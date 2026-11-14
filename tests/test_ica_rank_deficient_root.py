"""D55: ICA mixing on a rank-deficient neighbourhood (a one-column root view with k > 1).

A one-column view projects to an n x k matrix of rank 1. Under ``topology="graph"`` every
feature view of a target-rooted graph has that projection as its whole neighbourhood, and
a k-component FastICA on it divided by ~0 eigenvalues and raised
``ValueError: slice 0 has a NaN entry`` (about 60-70% of fits, any target). Synthetic data only.
"""
import warnings

import numpy as np
import pytest
import torch

import pysimlr
from pysimlr.consensus import compute_shared_consensus
from pysimlr.reliable_attribution import target_root_fit_params


def _views(seed, n=200, widths=(24, 30, 18, 36, 20), nan_rows=True):
    r = np.random.default_rng(seed)
    z = r.normal(size=(n, 3))
    views = [z @ r.normal(size=(3, w)) + 0.7 * r.normal(size=(n, w)) for w in widths]
    if nan_rows:
        for v in views:
            v[r.choice(n, 8, replace=False)] = np.nan
    return views, z


def _root_fit(views, target, seed, k=4, iterations=3, energy="recon_r2"):
    kw = target_root_fit_params(len(views), sparseness=0.8)
    kw["iterations"] = iterations
    for d in kw["per_view_params"].values():
        d["energy_type"] = energy
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pysimlr.fit("simlr", views + [target[:, None]], k=k, random_state=seed,
                           mixing_algorithm="ica", **kw)


def test_shuffled_target_root_fit_succeeds_over_50_seeds():
    ok = 0
    for seed in range(50):
        views, z = _views(seed)
        t = np.random.default_rng(1000 + seed).permutation(z[:, 0])
        t = (t - t.mean()) / t.std()
        res = _root_fit(views, t, seed)
        assert np.isfinite(np.asarray(res["u"])).all()
        ok += 1
    assert ok == 50


@pytest.mark.parametrize("width", [1, 2, 3])
def test_ica_neighbourhood_narrower_than_k_has_zero_extra_columns(width):
    k = 4
    r = np.random.default_rng(0)
    p = torch.tensor(r.normal(size=(100, width)) @ r.normal(size=(width, k)), dtype=torch.float32)
    u, anchor = compute_shared_consensus([p], mixing_algorithm="ica", k=k, training=True, topology="star")
    u = u.numpy()
    assert u.shape == (100, k) and anchor.shape == (k, k)
    assert np.isfinite(u).all() and np.isfinite(anchor.numpy()).all()
    # defined behaviour: `rank` standardized ICA columns, the other k - rank columns zero
    assert (np.abs(u[:, :width]).max(axis=0) > 0).all()
    assert np.abs(u[:, width:]).max() == 0.0
    assert np.abs(anchor.numpy()[:, width:]).max() == 0.0


def test_ica_full_rank_neighbourhood_keeps_k_standardized_columns():
    r = np.random.default_rng(1)
    p = torch.tensor(r.normal(size=(150, 12)), dtype=torch.float32)
    u, anchor = compute_shared_consensus([p], mixing_algorithm="ica", k=4, topology="star", training=True)
    u = u.numpy()
    assert np.allclose(u.std(axis=0, ddof=1), 1.0, atol=1e-4)
    assert (np.abs(anchor.numpy()).max(axis=0) > 0).all()


def test_graph_consensus_shares_identical_neighbourhoods():
    """Views with the same neighbourhood get equal, independent (not aliased) consensus tensors."""
    r = np.random.default_rng(2)
    k = 3
    projs = [torch.tensor(r.normal(size=(80, k)), dtype=torch.float32) for _ in range(4)]
    graph = {0: [3], 1: [3], 2: [3], 3: [0, 1, 2]}
    u = compute_shared_consensus(projs, mixing_algorithm="ica", k=k, topology="graph", path_graph=graph)
    assert torch.equal(u[0], u[1]) and torch.equal(u[1], u[2])
    assert u[0].data_ptr() != u[1].data_ptr()
    single = compute_shared_consensus([projs[3]], mixing_algorithm="ica", k=k, topology="star")
    # neighbourhood [3] is normalized inside the call, as the single-view star call is
    assert torch.allclose(u[0], single, atol=1e-5)


@pytest.mark.parametrize("mixing", ["svd", "pca", "avg", "newton"])
def test_other_mixings_finite_on_one_column_root(mixing):
    views, z = _views(0)
    t = np.random.default_rng(5).permutation(z[:, 0])
    kw = target_root_fit_params(len(views), sparseness=0.8)
    kw["iterations"] = 3
    for d in kw["per_view_params"].values():
        d["energy_type"] = "recon_r2"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = pysimlr.fit("simlr", views + [t[:, None]], k=4, random_state=0,
                              mixing_algorithm=mixing, **kw)
        except ValueError as err:  # a clear, early, named error is an accepted outcome
            assert "NaN" not in str(err)
            return
    assert np.isfinite(np.asarray(res["u"])).all()
