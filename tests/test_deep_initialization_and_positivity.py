"""Two related fixes to the deep entry points (`lend_simr`, `ned_simr`,
`ned_simr_shared_private`, `flow_simr_v`):

1. `initialization_type` reaches their first-layer encoders. Before this,
   `initialize_v`/`initialize_weights` hardcoded PCA regardless of what a
   caller believed they had selected -- exactly the "silently discarded
   argument" defect class `CORRECTNESS_AUDIT.md` documents elsewhere.
2. `positivity` now defaults to `"either"`, not `"positive"`. The old default
   silently forced every first-layer encoder non-negative even on
   standardized (signed, zero-mean) real data. `flow_simr` has no such
   parameter at all and should not gain a decorative one -- it has no linear
   encoder for `positivity` to constrain.
"""
import numpy as np
import pytest
import torch

from pysimlr.deep import lend_simr, ned_simr, ned_simr_shared_private
from pysimlr.flows import flow_simr, flow_simr_v
from pysimlr.simlr import INITIALIZATION_TYPES


def _views(n=40, dims=(8, 6, 7), seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(n, p, generator=g) for p in dims]


DEEP_METHODS = [
    ("lend_simr", lambda views, **kw: lend_simr(views, k=3, epochs=3, warmup_epochs=0, verbose=False, **kw)),
    ("ned_simr", lambda views, **kw: ned_simr(views, k=3, epochs=3, warmup_epochs=0, verbose=False, **kw)),
    ("flow_simr_v", lambda views, **kw: flow_simr_v(views, k=3, epochs=3, warmup_epochs=0, device="cpu", verbose=False, **kw)),
]


@pytest.mark.parametrize("name,fn", DEEP_METHODS)
def test_positivity_defaults_to_either(name, fn):
    """Calling with no `positivity` at all must not silently rectify the
    encoder -- that only happens if the default actually changed."""
    views = _views()
    out = fn(views)
    v0 = out["v"][0]
    assert (v0 < 0).any(), (
        f"{name}'s default produced an all-non-negative basis; the "
        f"positivity default may have reverted to 'positive'"
    )


def test_flow_simr_has_no_positivity_parameter():
    """flow_simr has no linear encoder, so there is nothing for `positivity`
    to constrain -- it should refuse the argument rather than silently
    accepting and ignoring it."""
    views = _views()
    with pytest.raises(TypeError):
        flow_simr(views, k=3, epochs=2, warmup_epochs=0, positivity="either")


@pytest.mark.parametrize("name,fn", DEEP_METHODS)
@pytest.mark.parametrize("t", INITIALIZATION_TYPES)
def test_initialization_type_is_dispatchable(name, fn, t):
    views = _views()
    kwargs = {"initialization_type": t}
    if t == "random":
        kwargs["init_seed"] = 1
    if t == "domain":
        kwargs["domain_matrices"] = [torch.randn(2, v.shape[1]) for v in views]
    out = fn(views, **kwargs)
    assert torch.isfinite(out["u"]).all()
    for v in out["v"]:
        assert torch.isfinite(v).all()


@pytest.mark.parametrize("name,fn", DEEP_METHODS)
def test_initialization_type_actually_changes_the_start(name, fn):
    """The regression this guards against: `initialize_v` hardcoding 'pca'
    regardless of what was requested, so every `initialization_type` produced
    an identical fit."""
    views = _views()
    torch.manual_seed(0)
    out_pca = fn(views, initialization_type="pca")
    torch.manual_seed(0)
    out_gcca = fn(views, initialization_type="gcca")
    diffs = [float((a - b).norm()) for a, b in zip(out_pca["v"], out_gcca["v"])]
    assert any(d > 1e-6 for d in diffs), (
        f"{name}: 'pca' and 'gcca' produced identical bases -- "
        f"initialization_type is not reaching the encoder"
    )


def test_ned_simr_shared_private_accepts_initialization_type():
    views = _views()
    out = ned_simr_shared_private(views, k=2, private_k=1, epochs=3, warmup_epochs=0,
                                  verbose=False, initialization_type="gcca")
    assert torch.isfinite(out["u"]).all()


@pytest.mark.parametrize("name,fn", DEEP_METHODS)
def test_random_initialization_type_needs_a_seed_or_is_still_reproducible_without_one(name, fn):
    """Unlike `simlr()`/`initialize_simlr`, these entry points don't require
    `init_seed` up front for 'random' -- they fall back to an unseeded
    `torch.Generator()`, matching `simlr()`'s own fallback. This just pins
    that it runs rather than raising.
    """
    views = _views()
    out = fn(views, initialization_type="random")
    assert torch.isfinite(out["u"]).all()
