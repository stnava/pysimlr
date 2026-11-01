"""Explicit generator parameters must reach the data.

With view-private background each planted generator replaced the caller's
`p_list` (and `shared_k`, `noise_scale`) with a hard-coded regime default,
because those are named parameters and `kwargs.get` could never see them.
Regime defaults now apply only to parameters left unset.
"""
from pysimlr.benchmarks import synthetic_cases as sc


def _widths(case):
    return [int(x.shape[1]) for x in case["data"]]


def test_explicit_p_list_is_honoured_with_private_background():
    for fn in (sc.build_linear_footprint_case, sc.build_nonlinear_shared_case,
               sc.build_shared_plus_private_case, sc.build_nonnegative_parts_case):
        case = fn(n_samples=40, seed=0, p_list=[11, 7], private_scale=0.5)
        assert _widths(case) == [11, 7], fn.__name__


def test_regime_defaults_still_apply_when_unset():
    assert _widths(sc.build_nonlinear_shared_case(n_samples=40, private_scale=0.5)) == [120, 80, 50]
    assert _widths(sc.build_nonlinear_shared_case(n_samples=40)) == [100, 80, 60]
    assert _widths(sc.build_shared_plus_private_case(n_samples=40, private_scale=0.5)) == [160, 60, 30]
    assert _widths(sc.build_nonnegative_parts_case(n_samples=40, private_scale=0.5)) == [120, 45, 30]
