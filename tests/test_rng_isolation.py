"""Fitting must not disturb the caller's global RNG state.

The optional NSA-Flow backend takes a ``seed`` argument (default 42) and seeds
the *process-wide* generator with it. `simlr_sparseness` calls it once per view
per iteration, so before this was guarded every `torch` random draw made after
a `simlr()` call was reset to the same fixed state: `torch.randperm` returned
an identical permutation every time.

That silently collapsed every permutation null distribution to a single
repeated value -- all 19 replicates identical -- which pins the p-value at
either the floor or 1.0 depending only on which side the observed statistic
falls. `simlr_perm`, `paths.permutation_test` and `estimate_rank` all draw
their nulls this way, as does any user code doing train/test splits or
bootstrapping after a fit. Nothing failed; the numbers were just wrong.
"""
import warnings

import numpy as np
import pytest
import torch

from pysimlr import simlr, simlr_perm
from pysimlr.nsa_backend import backend_report
from pysimlr.sparsification import simlr_sparseness


def _successive_permutations(make_call, n=3):
    torch.manual_seed(123)
    out = []
    for _ in range(n):
        make_call()
        out.append(tuple(torch.randperm(8).tolist()))
    return out


@pytest.mark.parametrize("constraint_type", ["ortho", "Stiefel", "none"])
def test_simlr_sparseness_does_not_reseed_the_global_rng(constraint_type):
    perms = _successive_permutations(
        lambda: simlr_sparseness(
            torch.randn(20, 3), constraint_type=constraint_type,
            constraint_weight=0.1,
        )
    )
    assert len(set(perms)) == len(perms), (
        f"{constraint_type}: successive calls left the RNG in the same state, "
        f"so every later random draw repeats: {perms}"
    )


def test_simlr_does_not_reseed_the_global_rng():
    x = [torch.randn(40, 8), torch.randn(40, 6)]
    perms = _successive_permutations(lambda: simlr(x, k=2, iterations=3))
    assert len(set(perms)) == len(perms), f"RNG state frozen by simlr(): {perms}"


def test_rng_state_is_exactly_preserved_across_a_fit():
    """Stronger: the stream should continue as if the fit had not happened."""
    x = [torch.randn(30, 6), torch.randn(30, 5)]

    torch.manual_seed(7)
    expected = torch.randperm(10).tolist()

    torch.manual_seed(7)
    simlr(x, k=2, iterations=3)
    got = torch.randperm(10).tolist()

    assert got == expected, (
        "a fit consumed or reset global RNG state; downstream draws are no "
        "longer reproducible from the caller's seed"
    )


def test_permutation_null_is_not_degenerate():
    """The direct consequence: the null must actually vary."""
    warnings.filterwarnings("ignore")
    torch.manual_seed(0)
    st = simlr_perm(
        [torch.randn(50, 8), torch.randn(50, 6)], k=2, n_perms=9, iterations=5
    )["stats"]["sim_0_1"]
    null = np.asarray(st["null_distribution"])
    assert null.size == 9
    assert np.unique(np.round(null, 10)).size > 1, (
        f"all {null.size} permutation replicates produced the same statistic "
        f"({null[0]:.6f}); the permutations were identical"
    )
    assert null.std() > 1e-9


def test_backend_report_is_available_for_provenance():
    rep = backend_report()
    assert set(rep) == {"available", "module", "entry_point"}
