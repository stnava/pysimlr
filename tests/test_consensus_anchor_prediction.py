"""
Out-of-sample prediction must reuse the consensus map, not re-derive it.

`predict_shared_latent` projects new data through V and then computes the
consensus. For ``svd``, ``pca`` and ``ica`` the derived basis is defined only
up to rotation, column sign and ordering, so deriving it afresh on a held-out
sample places those samples in different axes than the training latent. A model
fitted on the training U is then evaluated against coordinates that do not
correspond to its coefficients.

Measured on Diabetes with a 70/30 split, a linear regression fitted on the
training U scored -0.67 out of sample before this was fixed and +0.28 after,
against +0.40 through the fixed map ``X @ V``. That negative figure, not any
property of the fit, is what produced SiMLR's negative benchmark accuracy and
8 of its 9 degenerate cells.

The deep models never had this problem: they carry a `consensus_anchor` buffer
and pass it at eval. These tests pin the same guarantee for the linear path.
"""

import numpy as np
import pytest
import torch

from pysimlr import simlr
from pysimlr.simlr import predict_shared_latent

VOLATILE = ["svd", "pca", "ica"]
FIXED = ["newton", "avg"]


def _case(seed=0, n=200, p=(8, 6), k=2):
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(n, k, generator=g)
    mats = [u @ torch.randn(k, pi, generator=g) + 0.3 * torch.randn(n, pi, generator=g)
            for pi in p]
    return mats, u


def _fit(mats, alg, k=2, **kw):
    return simlr(mats, k=k, iterations=15, mixing_algorithm=alg, verbose=False, **kw)


@pytest.mark.parametrize("alg", VOLATILE)
def test_an_anchor_is_captured_for_the_volatile_algorithms(alg):
    mats, _ = _case()
    res = _fit(mats, alg)
    anchor = res["consensus_anchor"]
    assert anchor is not None, f"{alg} derives an arbitrary basis and needs an anchor"
    # concatenated projections -> latent: (modalities * k, k)
    assert anchor.shape == (len(mats) * 2, 2)
    assert torch.isfinite(anchor).all()


@pytest.mark.parametrize("alg", FIXED)
def test_no_anchor_is_captured_where_none_is_needed(alg):
    """``newton`` and ``avg`` are fixed functions of their input, so there is no
    arbitrary basis to pin and nothing to store."""
    mats, _ = _case()
    assert _fit(mats, alg)["consensus_anchor"] is None


@pytest.mark.parametrize("alg", VOLATILE + FIXED)
def test_prediction_reproduces_the_training_latent_on_the_training_data(alg):
    """The sharpest form of the guarantee: handed back its own training data,
    prediction must return the same latent the fit reported."""
    mats, _ = _case()
    res = _fit(mats, alg)
    u_fit = res["u"]
    u_fit = u_fit[0] if isinstance(u_fit, list) else u_fit
    u_pred = predict_shared_latent(mats, res)
    u_pred = u_pred[0] if isinstance(u_pred, list) else u_pred

    rel = float(torch.norm(u_fit - u_pred) / (torch.norm(u_fit) + 1e-12))
    assert rel < 1e-5, f"{alg}: prediction re-derived the latent (rel diff {rel:.4f})"


@pytest.mark.parametrize("alg", VOLATILE)
def test_held_out_latent_lands_in_the_training_axes(alg):
    """
    The out-of-sample property that matters. A regression fitted on the
    training latent must transfer to held-out samples; without the anchor it
    scored negative R-squared because the axes did not correspond.
    """
    from sklearn.linear_model import LinearRegression

    mats, u_true = _case(n=300)
    y = (1.5 * u_true[:, 0] - u_true[:, 1]).numpy()
    ntr = 210
    tr = [m[:ntr] for m in mats]
    te = [m[ntr:] for m in mats]

    res = _fit(tr, alg)
    u_fit = res["u"]
    u_fit = u_fit[0] if isinstance(u_fit, list) else u_fit
    u_te = predict_shared_latent(te, res)
    u_te = u_te[0] if isinstance(u_te, list) else u_te

    model = LinearRegression().fit(u_fit.numpy(), y[:ntr])
    r2 = model.score(u_te.numpy(), y[ntr:])
    assert r2 > 0.0, (
        f"{alg}: held-out R-squared {r2:.4f} -- the test latent is not in the "
        "training axes")


@pytest.mark.parametrize("alg", VOLATILE)
def test_the_anchor_is_what_makes_it_work(alg):
    """Localises the guarantee to the anchor: drop it and the transfer breaks
    again, so the test above is not passing for some incidental reason."""
    from sklearn.linear_model import LinearRegression

    mats, u_true = _case(n=300)
    y = (1.5 * u_true[:, 0] - u_true[:, 1]).numpy()
    ntr = 210
    tr, te = [m[:ntr] for m in mats], [m[ntr:] for m in mats]

    res = _fit(tr, alg)
    u_fit = res["u"]
    u_fit = u_fit[0] if isinstance(u_fit, list) else u_fit
    model = LinearRegression().fit(u_fit.numpy(), y[:ntr])

    with_anchor = predict_shared_latent(te, res)
    with_anchor = with_anchor[0] if isinstance(with_anchor, list) else with_anchor

    stripped = dict(res)
    stripped["consensus_anchor"] = None
    without = predict_shared_latent(te, stripped)
    without = without[0] if isinstance(without, list) else without

    r2_with = model.score(with_anchor.numpy(), y[ntr:])
    r2_without = model.score(without.numpy(), y[ntr:])
    assert r2_with > r2_without, (
        f"{alg}: anchored {r2_with:.4f} did not beat unanchored {r2_without:.4f}")


def test_a_result_without_the_key_still_predicts():
    """Backwards compatibility: a result dict from an older fit has no
    `consensus_anchor`, and must still predict rather than raising."""
    mats, _ = _case()
    res = _fit(mats, "svd")
    legacy = {key: value for key, value in res.items() if key != "consensus_anchor"}
    u = predict_shared_latent(mats, legacy)
    u = u[0] if isinstance(u, list) else u
    assert u.shape == (mats[0].shape[0], 2)
    assert torch.isfinite(u).all()


if __name__ == "__main__":
    pytest.main([__file__])
