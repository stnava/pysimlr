"""View order and the sign convention of the averaging mixings in the deep and flow models.

`lend`, `ned`, `ned_shared_private`, `flow_v` and `flow` all average their views' latents through
`consensus.compute_shared_consensus` (via `deep._ModalityWeightingMixin._training_consensus` and
`deep._aggregate_shared_consensus`), so they used the same order-dependent greedy sign rule as
`simlr`. They now take `sign_alignment` (default "spectral") and hand it to that function.

Order dependence has two sources that these tests keep apart:

* the sign rule (fixed by "spectral"), isolated by giving each view the same weights whatever its
  position: in the forward tests the second model is a permuted copy of the first; in the fit tests a
  subclass seeds each view's decoder/flow weights from that view's width. Dropout is off and the batch is the
  whole sample, so no other random draw depends on position.
* the optimizer: the nonlinear `ned` and `ned_shared_private` fits are chaotic (a 1e-7 relative jitter of the
  data, same order, moves the latent by up to ~1 after 30 epochs), so no multi-epoch order test is made for
  them; their sign step is tested on the forward pass. The weight initialization drawn in view order is order
  dependent by design and is not tested.

Synthetic only.
"""
import inspect
import warnings

import numpy as np
import pytest
import torch

import pysimlr
from pysimlr import deep, flows
from pysimlr.utils import align_column_signs
from test_view_order_sign_alignment import _noise_views

ORDER_A = [f"V{i}" for i in range(5)] + ["Root"]
ORDER_B = ["V0", "V1", "V2", "V4", "V3", "Root"]

ENTRY_POINTS = [deep.lend_simr, deep.ned_simr, deep.ned_simr_shared_private, flows.flow_simr, flows.flow_simr_v]


@pytest.mark.parametrize("fn", ENTRY_POINTS, ids=lambda f: f.__name__)
def test_every_entry_point_defaults_to_spectral(fn):
    assert inspect.signature(fn).parameters["sign_alignment"].default == "spectral"


def test_default_of_simlr_consensus_and_utility_is_spectral():
    assert inspect.signature(pysimlr.simlr).parameters["sign_alignment"].default == "spectral"
    from pysimlr.consensus import compute_shared_consensus
    assert inspect.signature(compute_shared_consensus).parameters["sign_alignment"].default == "spectral"
    assert inspect.signature(align_column_signs).parameters["method"].default == "spectral"


def test_unknown_sign_alignment_is_rejected_by_deep_models():
    with pytest.raises(ValueError, match="sign_alignment"):
        deep.LENDSiMRModel([5, 6, 7], 1, sign_alignment="nope")


def test_spectral_signs_carry_no_gradient_of_their_own():
    """The signs are constants: the gradient of the aligned sum w.r.t. each view is exactly its sign."""
    g = torch.Generator().manual_seed(0)
    mats = [torch.randn(30, 2, generator=g, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    out = align_column_signs(mats, method="spectral")
    torch.stack(out).sum().backward()
    for m, o in zip(mats, out):
        ratio = (o.detach() / m.detach())[0]                      # +-1 per column
        assert torch.equal(m.grad, ratio.expand_as(m.grad))


def _copy_view_weights(src, dst, perm):
    """dst view j gets the weights of src view perm[j] (every top-level ModuleList of one entry per view)."""
    n = len(perm)
    for name, mod in src.named_children():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) == n:
            for j in range(n):
                src_mod = mod[perm[j]]
                src_mod = src_mod.head if isinstance(src_mod, _Centered) else src_mod
                getattr(dst, name)[j].load_state_dict(src_mod.state_dict())


FORWARD_MODELS = [
    ("lend", deep.LENDSiMRModel, (), dict(dropout=0.0)),
    ("ned", deep.NEDSiMRModel, (), dict(dropout=0.0)),
    ("ned_shared_private", deep.NEDSharedPrivateSiMRModel, (1,), dict(dropout=0.0)),
    ("flow_v", flows.FlowSiMRVModel, (), {}),
    ("flow", flows.FlowSiMRModel, (), {}),
]


class _Centered(torch.nn.Module):
    """Test-only: remove the batch mean of a head's output. Untrained nonlinear heads emit latents whose means
    dominate their inner products (so greedy and spectral coincide); centred latents are the frustrated case the
    sign rule exists for (they are what the linear first layers produce)."""

    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, z):
        h = self.head(z)
        return h - h.mean(dim=0, keepdim=True)


def _forward_u(cls, extra, kw, views, order, sign_alignment, seed, perm_from=None):
    dims = [views[n].shape[1] for n in order]
    torch.manual_seed(seed)
    model = cls(dims, 1, *extra, topology="star", sign_alignment=sign_alignment, **kw)
    if perm_from is not None:
        _copy_view_weights(perm_from, model, [ORDER_A.index(n) for n in order])
    for name in ("nonlinear_heads", "shared_heads"):
        if hasattr(model, name):
            heads = getattr(model, name)
            for j in range(len(heads)):
                heads[j] = _Centered(heads[j])
    model.train()
    x = [torch.tensor(views[n], dtype=torch.float32) for n in order]
    with torch.no_grad():
        out = model(x)
    return model, out[2]


@pytest.mark.parametrize("name,cls,extra,kw", FORWARD_MODELS, ids=[m[0] for m in FORWARD_MODELS])
def test_forward_consensus_is_order_free_only_with_spectral(name, cls, extra, kw):
    """Same per-view weights, two view orders, untrained model: the consensus on the same data.

    Greedy flips different views for different orders on some of the seeds (the latents of untrained or
    weakly coupled views are not sign consistent); spectral agrees to float rounding on all of them.
    """
    worst = {"greedy": 0.0, "spectral": 0.0}
    for seed in range(8):
        views = _noise_views(100 + seed)
        for sa in worst:
            base, u1 = _forward_u(cls, extra, kw, views, ORDER_A, sa, seed)
            _, u2 = _forward_u(cls, extra, kw, views, ORDER_B, sa, seed, perm_from=base)
            d = min(float((u1 - u2).abs().max()), float((u1 + u2).abs().max()))
            worst[sa] = max(worst[sa], d)
    assert worst["greedy"] > 0.1, worst        # structural, not rounding
    assert worst["spectral"] < 1e-4, worst


def _fixed_view_init(cls):
    """Subclass that seeds every per-view Linear layer (except the first-layer encoders) from the view's width."""
    class Fixed(cls):
        def __init__(self, input_dims, *a, **kw):
            super().__init__(input_dims, *a, **kw)
            n = len(input_dims)
            for _, mod in self.named_children():
                if isinstance(mod, torch.nn.ModuleList) and len(mod) == n:
                    for i, sub in enumerate(mod):
                        g = torch.Generator().manual_seed(1000 + int(input_dims[i]))
                        for m in sub.modules():
                            if isinstance(m, torch.nn.Linear):
                                b = 1.0 / np.sqrt(m.in_features)
                                with torch.no_grad():
                                    m.weight.copy_((torch.rand(m.weight.shape, generator=g) * 2 - 1) * b)
                                    m.bias.copy_((torch.rand(m.bias.shape, generator=g) * 2 - 1) * b)
    return Fixed


# (name, entry point, module, model class name, fit kwargs, tolerance after 1 epoch, tolerance after 10 epochs)
FIT_MODELS = [
    ("lend", deep.lend_simr, deep, "LENDSiMRModel", dict(dropout=0.0, batch_size=174), 1e-5, 1e-4),
    ("flow_v", flows.flow_simr_v, flows, "FlowSiMRVModel", dict(batch_size=174), 5e-4, 5e-3),
    ("flow", flows.flow_simr, flows, "FlowSiMRModel", dict(batch_size=174), 1e-5, 5e-2),
]


def _ud(a, b):
    a, b = (np.asarray(t.detach().double()).ravel() for t in (a, b))
    return min(np.abs(a - b).max(), np.abs(a + b).max())


@pytest.mark.parametrize("name,fn,mod,attr,kw,tol1,tol10", FIT_MODELS, ids=[m[0] for m in FIT_MODELS])
def test_fit_is_order_free_with_default_and_not_with_greedy(name, fn, mod, attr, kw, tol1, tol10, monkeypatch):
    """Per-view weights fixed independently of position; two view orders; consensus and per-view latents
    (undoing the permutation) compared after 1 epoch and after 10. Default (spectral) within the stated
    tolerance; the same fits with greedy alignment differ by > 0.1 on some seed (so the test is not vacuous)."""
    monkeypatch.setattr(mod, attr, _fixed_view_init(getattr(mod, attr)))
    idx = [ORDER_B.index(n) for n in ORDER_A]
    for iterations, tol in ((1, tol1), (10, tol10)):
        worst = {"greedy": 0.0, "default": 0.0}
        for seed in (3, 4, 6):
            views = _noise_views(seed)
            for label, extra in (("greedy", {"sign_alignment": "greedy"}), ("default", {})):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r1, r2 = (fn([views[n] for n in order], k=1, iterations=iterations, random_state=3, **kw, **extra)
                              for order in (ORDER_A, ORDER_B))
                worst[label] = max(worst[label], _ud(r1["u"], r2["u"]),
                                   *[_ud(a, r2["latents"][j]) for a, j in zip(r1["latents"], idx)])
        assert worst["greedy"] > 0.1, (name, iterations, worst)
        assert worst["default"] < tol, (name, iterations, worst)
