"""`random_state` is one seed for a whole fit, on every entry point.

Contract:
  * the same seed gives a bit-identical result (also on input with a
    near-constant column, where `_standardize_deep` draws jitter noise);
  * two unseeded calls differ, and each reports the seed it drew in
    ``result["random_state"]``, which reproduces it;
  * the caller's global RNG stream is left exactly where it was, apart from
    the one draw an unseeded call makes to pick its seed.

The deep and flow trainers used to draw dropout masks, weight initialisation
and DataLoader shuffles from the global stream, so they were reproducible only
by seeding the process, and seeding the process was the only way to make them
reproducible.
"""
import numpy as np
import pytest
import torch

from pysimlr.deep import lend_simr, ned_simr, ned_simr_shared_private
from pysimlr.flows import flow_simr, flow_simr_v

STOCHASTIC = [lend_simr, ned_simr, ned_simr_shared_private, flow_simr_v, flow_simr]


def _views(near_constant=False):
    rng = np.random.default_rng(0)
    u = rng.standard_normal((40, 2))
    x1 = u @ rng.standard_normal((2, 7)) + 0.1 * rng.standard_normal((40, 7))
    x2 = u @ rng.standard_normal((2, 5)) + 0.1 * rng.standard_normal((40, 5))
    if near_constant:
        x1[:, 0] = 1.0
    return [torch.tensor(x1).float(), torch.tensor(x2).float()]


def _fit(fn, views, **kw):
    return fn(views, k=2, iterations=3, batch_size=16, verbose=False, **kw)


@pytest.mark.parametrize("near_constant", [False, True])
@pytest.mark.parametrize("fn", STOCHASTIC, ids=lambda f: f.__name__)
def test_the_same_seed_reproduces_the_fit(fn, near_constant):
    views = _views(near_constant)
    a = _fit(fn, views, random_state=7)
    torch.manual_seed(12345)          # a different global state must not matter
    b = _fit(fn, views, random_state=7)
    assert a["random_state"] == b["random_state"] == 7
    assert torch.equal(a["u"], b["u"])
    for va, vb in zip(a.get("v") or [], b.get("v") or []):
        assert torch.equal(va, vb)


@pytest.mark.parametrize("fn", STOCHASTIC, ids=lambda f: f.__name__)
def test_unseeded_fits_differ_and_report_a_reproducing_seed(fn):
    views = _views()
    torch.manual_seed(0)
    a = _fit(fn, views)
    b = _fit(fn, views)
    assert a["random_state"] != b["random_state"]
    assert not torch.equal(a["u"], b["u"])
    c = _fit(fn, views, random_state=a["random_state"])
    assert torch.equal(a["u"], c["u"])


@pytest.mark.parametrize("fn", STOCHASTIC, ids=lambda f: f.__name__)
def test_a_seeded_fit_leaves_the_global_stream_untouched(fn):
    views = _views()
    torch.manual_seed(99)
    expected = torch.randn(5)
    torch.manual_seed(99)
    _fit(fn, views, random_state=3)
    assert torch.equal(torch.randn(5), expected)
