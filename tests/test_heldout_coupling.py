"""`pysimlr.heldout_coupling`: cross-view agreement on held-out rows beside the training value."""
import warnings

import numpy as np
import pytest
import torch

import pysimlr


def _fit(xs, n_tr, k=2, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pysimlr.fit("simlr", [x[:n_tr] for x in xs], k, random_state=0, **kw)


def _signal(n=1000, widths=(12, 9, 7), seed=1):
    g = np.random.default_rng(seed)
    z = g.standard_normal((n, 2))
    return [torch.tensor(z @ g.standard_normal((2, p)) + g.standard_normal((n, p))).float() for p in widths]


def _noise(n=450, widths=(120, 120, 120), seed=3):
    g = np.random.default_rng(seed)
    return [torch.tensor(g.standard_normal((n, p))).float() for p in widths]


def test_noise_views_with_p_ge_n_couple_only_in_training():
    xs = _noise()
    res = _fit(xs, 50, iterations=60)
    h = pysimlr.heldout_coupling(res, [x[50:] for x in xs])
    assert h["views"] == [0, 1, 2] and h["n_train"] == 50 and h["n_heldout"] == 400
    assert min(h["cross_view_train"]) > 0.7          # measured 0.88-0.91
    assert max(h["cross_view_heldout"]) < 0.2        # measured 0.04-0.05
    assert min(h["cross_view_gap"]) > 0.5


def test_genuine_shared_signal_holds_out():
    xs = _signal()
    res = _fit(xs, 600)
    h = pysimlr.heldout_coupling(res, [x[600:] for x in xs])
    assert min(h["cross_view_heldout"]) > 0.7        # measured 0.86-0.88
    assert max(abs(g) for g in h["cross_view_gap"]) < 0.1


def test_statistics_match_a_direct_computation():
    xs = _signal()
    res = _fit(xs, 600)
    te = [x[600:] for x in xs]
    h = pysimlr.heldout_coupling(res, te)
    pred = pysimlr.predict(res, te)
    z = [np.asarray(l.detach(), float) for l in pred["latents"]]
    u = np.asarray(pred["u"].detach(), float)
    for m in range(3):
        want = [abs(np.corrcoef(z[m][:, j], u[:, j])[0, 1]) for j in range(2)]
        assert np.allclose(h["coupling_heldout"][m], want, atol=1e-10)
    pairs = [(0, 1), (0, 2), (1, 2)]
    want = [np.mean([abs(np.corrcoef(z[a][:, j], z[b][:, j])[0, 1]) for a, b in pairs]) for j in range(2)]
    assert np.allclose(h["cross_view_heldout"], want, atol=1e-10)
    assert np.allclose(h["coupling_train"], res["coupling_train"], atol=1e-10)


def test_missing_view_uses_the_available_views_in_both_splits():
    xs = _signal()
    res = _fit(xs, 600)
    h = pysimlr.heldout_coupling(res, [xs[0][600:], None, xs[2][600:]])
    assert h["views"] == [0, 2] and len(h["coupling_heldout"]) == 2
    ztr = [np.asarray(res["latents"][m].detach(), float) for m in (0, 2)]
    want = [abs(np.corrcoef(ztr[0][:, j], ztr[1][:, j])[0, 1]) for j in range(2)]
    assert np.allclose(h["cross_view_train"], want, atol=1e-10)


def test_errors():
    xs = _signal(n=200)
    res = _fit(xs, 150)
    with pytest.raises(ValueError, match="entries"):
        pysimlr.heldout_coupling(res, [x[150:] for x in xs[:2]])
    with pytest.raises(ValueError, match="no held-out view"):
        pysimlr.heldout_coupling(res, [None, None, None])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pca = pysimlr.fit("pca", [x[:150] for x in xs], 2, random_state=0)
    with pytest.raises(ValueError, match="joint basis"):
        pysimlr.heldout_coupling(pca, [x[150:] for x in xs])


def test_wide_view_warning_points_to_the_helper():
    xs = _noise(n=100, widths=(10, 80))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pysimlr.fit("simlr", [x[:60] for x in xs], 2, random_state=0, iterations=40)
    msg = [str(x.message) for x in w if issubclass(x.category, pysimlr.WideViewWarning)]
    assert msg and "pysimlr.heldout_coupling" in msg[0]
