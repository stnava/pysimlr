"""View order must not change a fit through the sign convention of the averaging mixings.

Synthetic only. At initialization the views' latents can be nearly orthogonal, so the signs that
`align_column_signs` assigns are decided by the order in which the views are visited (greedy running
sum). `sign_alignment="spectral"` is permutation-equivariant.
"""
import itertools
import warnings

import numpy as np
import pytest
import torch

import pysimlr
from pysimlr.utils import align_column_signs


def _frustrated_triangle(n=400, seed=0):
    """Three unit columns with cosines (a,b)=0.1, (b,c)=0.5, (a,c)=-0.4: no consistent sign assignment."""
    gram = np.array([[1.0, 0.1, -0.4], [0.1, 1.0, 0.5], [-0.4, 0.5, 1.0]])
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((n, 3)))   # orthonormal columns
    x = q @ np.linalg.cholesky(gram).T                                          # x.T @ x == gram
    return [torch.from_numpy(x[:, [i]].copy()) for i in range(3)]


def _aligned_mean_up_to_sign(mats, order, method):
    out = align_column_signs([mats[i] for i in order], method=method)
    u = torch.stack(out).mean(0)
    return u


def _same_up_to_global_sign(u1, u2, tol=1e-12):
    return min(float((u1 - u2).abs().max()), float((u1 + u2).abs().max())) < tol


def test_greedy_alignment_depends_on_order_on_a_frustrated_triple():
    mats = _frustrated_triangle()
    ref = _aligned_mean_up_to_sign(mats, (0, 1, 2), "greedy")
    others = [_aligned_mean_up_to_sign(mats, p, "greedy") for p in itertools.permutations(range(3))]
    assert not all(_same_up_to_global_sign(ref, o) for o in others), "expected the historical greedy rule to be order dependent"


def test_spectral_alignment_is_permutation_equivariant():
    mats = _frustrated_triangle()
    ref = _aligned_mean_up_to_sign(mats, (0, 1, 2), "spectral")
    for p in itertools.permutations(range(3)):
        assert _same_up_to_global_sign(ref, _aligned_mean_up_to_sign(mats, p, "spectral"))


def test_spectral_equals_greedy_when_views_agree():
    g = torch.Generator().manual_seed(1)
    z = torch.randn(100, 2, generator=g, dtype=torch.float64)
    mats = [z + 0.2 * torch.randn(100, 2, generator=g, dtype=torch.float64) for _ in range(4)]
    mats[2] = -mats[2]
    gr, sp = align_column_signs(mats, method="greedy"), align_column_signs(mats, method="spectral")
    assert all(torch.equal(a, b) for a, b in zip(gr, sp))


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="sign_alignment"):
        align_column_signs([torch.ones(4, 1), torch.ones(4, 1)], method="nope")


def _noise_views(seed, n=174, widths=(65, 54, 72, 57, 196)):
    """Pure-noise brain views (no shared signal) and a root view; the draw order is part of the fixture."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    names = [f"V{i}" for i in range(len(widths))]
    views = {}
    for nm, p in zip(names, widths):
        rng.standard_normal(p), rng.random(p)
        views[nm] = rng.standard_normal((n, p))
    views["Root"] = (z + 0.7 * rng.standard_normal(n))[:, None]
    return views


def _fit(views, order, iterations, sign_alignment):
    pvp = {i: ({"positivity": "conical", "sparseness": None} if nm == "Root" else {"positivity": "conical", "sparseness": 0.8})
           for i, nm in enumerate(order)}
    g = pysimlr.fork_graph("Root", [n for n in order if n != "Root"], exogenous_root=False)
    pg = {order.index(a): [order.index(b) for b in bs] for a, bs in g.items()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pysimlr.fit("simlr", [views[n] for n in order], k=1, per_view_params=pvp, random_state=7,
                           iterations=iterations, topology="graph", path_graph=pg, dynamic_weights=False,
                           sign_alignment=sign_alignment)


def _udiff(r1, r2):
    u1, u2 = (np.asarray(r["u"].detach().double()) for r in (r1, r2))
    return min(np.abs(u1 - u2).max(), np.abs(u1 + u2).max())


def test_fit_after_one_sweep_is_order_free_only_with_spectral_alignment():
    """Seeds 3 and 4 of this synthetic family flip different views under greedy alignment for the two orders."""
    o1 = [f"V{i}" for i in range(5)] + ["Root"]
    o2 = ["V0", "V1", "V2", "V4", "V3", "Root"]
    worst_greedy = worst_spectral = 0.0
    for seed in (3, 4):
        views = _noise_views(seed)
        worst_greedy = max(worst_greedy, _udiff(_fit(views, o1, 1, "greedy"), _fit(views, o2, 1, "greedy")))
        worst_spectral = max(worst_spectral, _udiff(_fit(views, o1, 1, "spectral"), _fit(views, o2, 1, "spectral")))
    assert worst_greedy > 0.1, worst_greedy          # structural, not rounding (measured ~3.6)
    assert worst_spectral < 1e-4, worst_spectral     # float32 rounding only
