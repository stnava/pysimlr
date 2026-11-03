"""MAI: the max_cc metric, the degrees-of-freedom correction, the permutation floor, and
gating on the floor (consensus.loo_mai_calibrated, mai_gated_target)."""
import warnings

import numpy as np
import pytest
import torch

from pysimlr.consensus import (loo_mai_calibrated, mai_dof_adjust, mai_gated_target, mai_score,
                               view_dofs)


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
    assert mai_dof_adjust(0.99, "max_cc", 100, 99) == 0.0          # can fit anything
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


def test_gate_keeps_views_above_their_floor_and_only_those():
    mai = torch.tensor([0.9, 0.85, 0.3, 0.05]); floor = torch.tensor([0.1, 0.1, 0.1, 0.2])
    w = mai_gated_target(mai, 15.0, floor)
    assert w[3] == 0 and w[2] > 0.05 and float(w.sum()) == pytest.approx(1.0)
    eq = mai_gated_target(torch.full((4,), 0.8), 15.0, torch.full((4,), 0.1))
    assert torch.allclose(eq, torch.full((4,), 0.25))                 # all coupled: uniform
    with pytest.warns(RuntimeWarning, match="null floor"):
        w = mai_gated_target(torch.full((3,), 0.1), 15.0, torch.full((3,), 0.2))
    assert torch.allclose(w, torch.full((3,), 1 / 3))


def test_simlr_suppresses_a_wide_noise_view_and_keeps_sub_coupled_and_narrow_views():
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
    assert w[4] < 0.05 and w[2] > 0.05 and w[3] > 0.05


def test_cca_is_adjusted_on_its_correlation_scale():
    """cca of unit-norm latents is mean(s) <= 1/k; adjusting it as a correlation in
    [0, 1] zeroed every view (all deep fits went uniform)."""
    assert mai_dof_adjust(0.3, "cca", 48, 4, k=3) == pytest.approx(
        np.sqrt(1 - (1 - 0.81) * 47 / 43) / 3)
