"""MAI: the max_cc metric, the degrees-of-freedom correction, the permutation floor, and
gating on the floor (consensus.loo_mai_calibrated, mai_gated_target)."""
import warnings

import numpy as np
import pytest
import torch

from pysimlr.consensus import (MAI_MIN_SHARE, compute_shared_consensus, loo_mai_calibrated,
                               mai_dof_adjust, mai_ema, mai_gated_target, mai_score, view_dofs)


def _g(s):
    return torch.Generator().manual_seed(s)


def test_max_cc_is_the_first_canonical_correlation_and_rank_adaptive():
    n = 200
    Z = torch.randn(n, 3, generator=_g(0), dtype=torch.float64)
    U = Z @ torch.randn(3, 3, generator=_g(1), dtype=torch.float64)       # same span
    assert mai_score(Z, U, "max_cc") == pytest.approx(1.0, abs=1e-8)
    # one shared direction among three: max_cc sees it, the aggregate metrics dilute it
    V = torch.cat([Z[:, :1], torch.randn(n, 2, generator=_g(2), dtype=torch.float64)], 1)
    assert mai_score(Z, V, "max_cc") > 0.99
    assert mai_score(Z / Z.norm(), V / V.norm(), "cca") < 0.6
    # rank-1 latent (a narrow view): scored on its one real direction
    R1 = Z[:, :1] @ torch.ones(1, 3, dtype=torch.float64)
    assert mai_score(R1, U, "max_cc") == pytest.approx(1.0, abs=1e-8)


def test_dof_adjustment():
    assert mai_dof_adjust(0.5, "procrustes_r2", 100, None) == 0.5
    assert np.isnan(mai_dof_adjust(0.99, "max_cc", 100, 99))      # can fit anything: not assessable
    r2 = mai_dof_adjust(0.5, "procrustes_r2", 101, 10)
    assert r2 == pytest.approx(1 - 0.5 * 100 / 90)
    r = mai_dof_adjust(0.8, "max_cc", 101, 10)
    assert r == pytest.approx(np.sqrt(1 - (1 - 0.64) * 100 / 90))
    v = torch.zeros(10, 3); v[:4, 0] = 1; v[2:6, 1] = 1; v[8, 2] = 1
    assert view_dofs([v], 50, "procrustes_r2") == [pytest.approx(3.0)]
    assert view_dofs([v], 50, "max_cc") == [7.0]                      # union of supports


def test_floors_are_none_without_permutations_and_above_zero_raw():
    lat = [torch.randn(60, 3, generator=_g(s)) for s in range(3)]
    s, f = loo_mai_calibrated(lat, "max_cc", n_perm=0)
    assert f is None and len(s) == 3
    s, f = loo_mai_calibrated(lat, "max_cc", n_perm=30)
    assert all(fi > 0.1 for fi in f)     # sigma_1 of random 60 x 3 pairs is far from 0


def test_gate_favours_views_above_their_floor_and_never_goes_binary():
    M = 4; lo, hi = MAI_MIN_SHARE / M, 1 - MAI_MIN_SHARE * (M - 1) / M
    mai = torch.tensor([0.9, 0.85, 0.3, 0.05]); floor = torch.tensor([0.1, 0.1, 0.1, 0.2])
    w = mai_gated_target(mai, 15.0, floor)
    assert float(w.sum()) == pytest.approx(1.0)
    assert float(w[3]) == pytest.approx(lo) and w[0] > w[2] > w[3]       # below floor: the minimum
    # one view barely above a floor of 0 (ADNI-MCI: 0.006) cannot take all the weight
    w = mai_gated_target(torch.tensor([0.0, 0.0, 0.0, 0.006]), 15.0, torch.zeros(4))
    assert float(w.max()) == pytest.approx(hi) and float(w.min()) == pytest.approx(lo)
    eq = mai_gated_target(torch.full((4,), 0.8), 15.0, torch.full((4,), 0.1))
    assert torch.allclose(eq, torch.full((4,), 0.25))                 # all coupled: uniform
    with pytest.warns(RuntimeWarning, match="null floor"):
        w = mai_gated_target(torch.full((3,), 0.1), 15.0, torch.full((3,), 0.2))
    assert torch.allclose(w, torch.full((3,), 1 / 3))


def test_not_assessable_views_keep_their_uniform_share():
    nan = float("nan")
    w = mai_gated_target(torch.tensor([nan, nan, 0.8, 0.0]), 15.0, torch.tensor([nan, nan, 0.1, 0.1]))
    assert float(w[0]) == pytest.approx(0.25) and float(w[1]) == pytest.approx(0.25)
    assert w[2] > w[3] > 0
    assert torch.allclose(mai_gated_target(torch.full((3,), nan), 15.0, torch.full((3,), nan)),
                          torch.full((3,), 1 / 3))
    prev = torch.tensor([0.5, nan, 0.5]); new = torch.tensor([nan, 0.4, 0.3])
    out = mai_ema(prev, new, 0.8, first=False)
    assert torch.isnan(out[0]) and float(out[1]) == pytest.approx(0.4)
    assert float(out[2]) == pytest.approx(0.8 * 0.5 + 0.2 * 0.3)


def test_dof_correction_uses_the_rows_the_basis_was_fitted_on():
    """Deep models score MAI on a 64-row batch; correcting with n=64 made any view
    with ~64 non-zeros per column not assessable (ADNI-MCI: dof 80, n_fit 758)."""
    g = torch.Generator().manual_seed(0)
    z = torch.randn(64, 3, generator=g)
    lat = [z + 0.3 * torch.randn(64, 3, generator=g) for _ in range(3)]
    s_batch, _ = loo_mai_calibrated(lat, "procrustes_r2", dofs=[80, 80, 30], n_perm=0)
    s_fit, _ = loo_mai_calibrated(lat, "procrustes_r2", dofs=[80, 80, 30], n_perm=0, n_fit=758)
    assert np.isnan(s_batch[0]) and s_batch[2] < s_fit[2]
    assert all(0.5 < x <= 1 for x in s_fit)


@pytest.mark.parametrize("mixing", ["svd", "pca", "ica", "newton", "avg"])
@pytest.mark.parametrize("topology", ["star", "loo", "graph"])
def test_consensus_is_finite_when_a_neighbourhood_has_zero_weight(mixing, topology):
    """ADNI-MCI's bipartite graph: weights [0, 0, 0, 1] left view 3's neighbours
    all at 0, and svd/pca/ica returned NaN (159 of 1,025 sweep cells failed)."""
    g = torch.Generator().manual_seed(1)
    z = torch.randn(120, 3, generator=g)
    projs = [z @ torch.randn(3, 3, generator=g) + 0.2 * torch.randn(120, 3, generator=g) for _ in range(4)]
    pg = {0: [3], 1: [3], 2: [3], 3: [0, 1, 2]} if topology == "graph" else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = compute_shared_consensus(projs, mixing_algorithm=mixing, k=3, training=True,
                                       topology=topology, path_graph=pg,
                                       modality_weights=torch.tensor([0.0, 0.0, 0.0, 1.0]))
    u = out[0]
    for t in (u if isinstance(u, list) else [u]):
        assert torch.isfinite(t).all()


def test_simlr_ranks_a_wide_noise_view_last_and_keeps_sub_coupled_and_narrow_views():
    """The planted design of scripts/mai_calibration_check.py, one seed. Before: the p > n
    noise view got 0.30 and the sub-coupled and narrow views ~0."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from mai_calibration_check import views
    import pysimlr
    z, vs = views(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = pysimlr.fit("simlr", vs, 3, dynamic_weights=True, positivity="positive", random_state=0)
    w = np.asarray(r["modality_weights"], float)
    assert w.argmin() == 4 and w[4] < 1 / 5 and min(w[2], w[3]) > w[4]
    assert w.min() >= MAI_MIN_SHARE / 5 - 1e-6


def test_cca_is_adjusted_on_its_correlation_scale():
    """cca of unit-norm latents is mean(s) <= 1/k; adjusting it as a correlation in
    [0, 1] zeroed every view (all deep fits went uniform)."""
    assert mai_dof_adjust(0.3, "cca", 48, 4, k=3) == pytest.approx(
        np.sqrt(1 - (1 - 0.81) * 47 / 43) / 3)
