"""Training coupling of a view with at least as many features as samples is not evidence of shared
signal: `simlr` reports it (`coupling_train`, `view_width_ratio`, `wide_views`) and warns."""
import warnings

import numpy as np
import torch

import pysimlr


def _fit(xs, **kw):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = pysimlr.fit("simlr", xs, 2, random_state=0, iterations=60, **kw)
    return res, [x for x in w if issubclass(x.category, pysimlr.WideViewWarning)]


def _views(widths, n, seed=0, shared=True):
    g = np.random.default_rng(seed)
    z = g.standard_normal((n, 2))
    return [torch.tensor((z @ g.standard_normal((2, p)) if shared else 0) + g.standard_normal((n, p))).float()
            for p in widths]


def test_narrow_views_do_not_warn():
    res, w = _fit(_views((10, 8), 60))
    assert w == [] and res["wide_views"] == []
    assert np.allclose(res["view_width_ratio"], [10 / 60, 8 / 60])


def test_wide_view_warns_and_is_named():
    res, w = _fit(_views((10, 80), 60))
    assert res["wide_views"] == [1]
    assert len(w) == 1 and "view 1 (p=80" in str(w[0].message) and "n=60" in str(w[0].message)


def test_coupling_train_is_the_score_consensus_correlation():
    res, _ = _fit(_views((10, 8), 60))
    u = res["u"].detach().double().numpy()
    for m, z in enumerate(res["latents"]):
        z = z.detach().double().numpy()
        want = [abs(np.corrcoef(z[:, j], u[:, j])[0, 1]) for j in range(2)]
        assert np.allclose(res["coupling_train"][m], want, atol=1e-10)


def test_wide_noise_views_couple_in_training_but_not_held_out():
    """The defect this guards: independent noise views look coupled in training when p >= n."""
    n_tr, n_te, p = 50, 400, 120
    xs = _views((p, p, p), n_tr + n_te, seed=3, shared=False)          # no shared signal at all
    tr = [x[:n_tr] for x in xs]
    te = [x[n_tr:] for x in xs]
    res, w = _fit(tr)
    assert res["wide_views"] == [0, 1, 2] and len(w) == 1
    lat_tr = [z.detach().double().numpy() for z in res["latents"]]
    lat_te = [z.detach().double().numpy() for z in pysimlr.predict(res, te)["latents"]]

    def cross(lat):
        return np.mean([abs(np.corrcoef(lat[a][:, j], lat[b][:, j])[0, 1])
                        for a in range(3) for b in range(a + 1, 3) for j in range(2)])
    assert cross(lat_tr) > 0.7
    assert cross(lat_te) < 0.2


def test_warning_class_is_separate_from_convergence():
    assert not issubclass(pysimlr.WideViewWarning, pysimlr.ConvergenceWarning)
    assert issubclass(pysimlr.WideViewWarning, UserWarning)
