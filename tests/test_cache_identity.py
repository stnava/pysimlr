"""The retraction and smoothing caches must not serve one object's results to
another. They were keyed on `id()`, which Python reuses once an object is
freed; per-test mock backends then hit a previous mock's entries and were never
called (an intermittent failure of
test_backend_is_invoked_for_every_soft_ortho_constraint)."""
from unittest.mock import MagicMock, patch

import torch

from pysimlr import sparsification as sp


def _backend():
    m = MagicMock()
    m.side_effect = lambda Y, **kw: {"Y": Y.clone()}
    return m


def test_a_new_backend_at_a_reused_address_is_still_called():
    v = torch.rand(10, 3, generator=torch.Generator().manual_seed(0)) + 0.1
    first, second = _backend(), _backend()
    with patch.object(sp, "load_nsa_flow", return_value=first):
        sp._nsa_retract(v, w=0.5, nonneg=True)
    # Force the address collision: rewrite the first entry's key to the second
    # backend's id, as a recycled address would.
    for key in list(sp._NSA_RETRACT_CACHE):
        if key[0] == id(first):
            sp._NSA_RETRACT_CACHE[(id(second),) + key[1:]] = sp._NSA_RETRACT_CACHE.pop(key)
    with patch.object(sp, "load_nsa_flow", return_value=second):
        sp._nsa_retract(v, w=0.5, nonneg=True)
    assert second.called


def test_smoothing_cache_does_not_reuse_another_operators_result():
    v = torch.rand(4, 2, generator=torch.Generator().manual_seed(1))
    a, b = torch.eye(4), 2.0 * torch.eye(4)
    out_a = sp._apply_smoothing(v, a)
    for key in list(sp._SMOOTHING_CACHE):
        if key[0] == id(a):
            sp._SMOOTHING_CACHE[(id(b),) + key[1:]] = sp._SMOOTHING_CACHE.pop(key)
    assert torch.allclose(sp._apply_smoothing(v, b), 2.0 * v)
    assert torch.allclose(out_a, v)
