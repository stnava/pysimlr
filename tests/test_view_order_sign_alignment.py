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


def _fit(views, order, iterations, sign_alignment="default"):
    """`iterations=None` uses the default number of iterations; `sign_alignment="default"` leaves the argument out."""
    pvp = {i: ({"positivity": "conical", "sparseness": None} if nm == "Root" else {"positivity": "conical", "sparseness": 0.8})
           for i, nm in enumerate(order)}
    g = pysimlr.fork_graph("Root", [n for n in order if n != "Root"], exogenous_root=False)
    pg = {order.index(a): [order.index(b) for b in bs] for a, bs in g.items()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kw = {} if iterations is None else {"iterations": iterations}
        if sign_alignment != "default":
            kw["sign_alignment"] = sign_alignment
        return pysimlr.fit("simlr", [views[n] for n in order], k=1, per_view_params=pvp, random_state=7,
                           topology="graph", path_graph=pg, dynamic_weights=False, **kw)


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


# ---------------------------------------------------------------------------------------------------------
# Default settings: several view orders, consensus AND per-view latents, first sweep and full default fit
# ---------------------------------------------------------------------------------------------------------
ORDERS = {
    "A": [f"V{i}" for i in range(5)] + ["Root"],
    "swap_V3_V4": ["V0", "V1", "V2", "V4", "V3", "Root"],
    "reversed": ["Root", "V4", "V3", "V2", "V1", "V0"],
    "root_first_rotated": ["Root", "V2", "V3", "V4", "V0", "V1"],
}


def _worst_difference(views, iterations, sign_alignment="default"):
    """Max over orders and seeds of |u| and per-view latent differences to order A, permutation undone."""
    base = _fit(views, ORDERS["A"], iterations, sign_alignment)
    worst = 0.0
    for name, order in ORDERS.items():
        if name == "A":
            continue
        r = _fit(views, order, iterations, sign_alignment)
        worst = max(worst, _udiff(base, r))
        for i, nm in enumerate(ORDERS["A"]):
            worst = max(worst, _udiff({"u": base["latents"][i]}, {"u": r["latents"][order.index(nm)]}))
    return worst


def test_default_is_the_order_free_rule():
    assert _fit(_noise_views(3), ORDERS["A"], 1)["sign_alignment"] == "spectral"


def test_default_fit_is_order_free_first_sweep_and_full_fit_but_greedy_is_not():
    """Five weakly coupled views plus a root, k=1, four view orders (reversed and rotated ones included).

    Default settings: consensus and every view's latent agree across orders to float32 rounding at the first
    sweep (< 1e-4) and, on this well-conditioned fixture, after the full default fit (< 1e-3; measured
    ~5e-7). The same comparison with sign_alignment="greedy" differs by > 0.1 at the first sweep on some seed,
    so the assertion is not vacuous. Seeds 3 and 4 flip different views under greedy for different orders.
    """
    worst = {"greedy_1": 0.0, "default_1": 0.0, "default_full": 0.0}
    for seed in (3, 4, 6):
        views = _noise_views(seed)
        worst["greedy_1"] = max(worst["greedy_1"], _worst_difference(views, 1, "greedy"))
        worst["default_1"] = max(worst["default_1"], _worst_difference(views, 1))
    for seed in (3, 4):
        worst["default_full"] = max(worst["default_full"], _worst_difference(_noise_views(seed), None))
    assert worst["greedy_1"] > 0.1, worst
    assert worst["default_1"] < 1e-4, worst
    assert worst["default_full"] < 1e-3, worst


def test_two_views_spectral_matches_greedy_up_to_a_global_column_sign():
    """With two views both rules give each column the same relative sign; only the pair's global sign may differ
    (greedy keeps the first view's sign, spectral fixes it symmetrically). Identical when the views agree."""
    g = torch.Generator().manual_seed(5)
    for _ in range(50):
        x = torch.randn(40, 4, generator=g, dtype=torch.float64)
        flip = torch.randint(0, 2, (1, 4), generator=g).double() * 2 - 1
        y = 0.5 * x * flip + torch.randn(40, 4, generator=g, dtype=torch.float64)
        gr, sp = align_column_signs([x, y], method="greedy"), align_column_signs([x, y], method="spectral")
        assert torch.equal(torch.sign((gr[0] * gr[1]).sum(0)), torch.sign((sp[0] * sp[1]).sum(0)))
        s = torch.sign((gr[0] * sp[0]).sum(0))
        assert all(torch.equal(a, b * s) for a, b in zip(gr, sp))
    agree = [x, x * 0.8 + 0.1 * torch.randn(40, 4, generator=g, dtype=torch.float64)]
    assert all(torch.equal(a, b) for a, b in zip(align_column_signs(agree, method="greedy"),
                                                 align_column_signs(agree, method="spectral")))
