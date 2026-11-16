"""Tests for `pysimlr.multistart_fit` (synthetic data only)."""
import warnings

import numpy as np
import pytest

import pysimlr
from pysimlr import multistart_fit

warnings.filterwarnings("ignore", message=".*did not converge.*")


def toy(n=150, s2=0.9, seed=0):
    """Two shared signals of nearly equal strength (features 0-4 and 5-9): k=1 has more than one local solution."""
    rng = np.random.default_rng(seed)
    za, zb = rng.standard_normal((n, 1)), rng.standard_normal((n, 1))
    out = []
    for _ in range(3):
        a = np.zeros((1, 10))
        a[0, :5] = 1
        b = np.zeros((1, 10))
        b[0, 5:] = s2
        out.append(za @ a + zb @ b + 0.5 * rng.standard_normal((n, 10)))
    return out


@pytest.fixture(scope="module")
def ms():
    return multistart_fit("simlr", toy(), 1, n_starts=8, iterations=200)


def test_two_solution_toy_returns_the_best(ms):
    e = [s["energy"] for s in ms["starts"]]
    # Criterion: the starts disagree (at least two distinct solutions, spread of energies > 1e-3), the default
    # start is not the best, and the returned fit is the lowest-energy start.
    assert ms["n_distinct_solutions"] >= 2
    assert max(e) - min(e) > 1e-3
    assert ms["starts"][0]["label"] == "default" and e[0] > min(e) + 1e-3
    assert ms["chosen_index"] == int(np.argmin(e)) and ms["chosen"] == ms["starts"][ms["chosen_index"]]["label"]
    assert ms["best_energy"] == min(e)
    assert ms["energy_gap"] is not None and ms["energy_gap"] <= 0.0
    assert ms["starts"][ms["chosen_index"]]["score_correlation_with_chosen"] == pytest.approx(1.0)
    assert any(s["score_correlation_with_chosen"] < 0.5 for s in ms["starts"])
    assert len(ms["starts"]) == 9 and ms["starts"][0]["seed"] is None
    seeds = [s["seed"] for s in ms["starts"][1:]]
    assert len(set(seeds)) == 8
    for s in ms["starts"]:
        assert set(s) == {"label", "seed", "energy", "converged", "n_steps", "score_correlation_with_chosen"}
    assert ms["u"].shape == (150, 1) and ms["method"] == "simlr"


def test_n_starts_zero_equals_plain_fit():
    v = toy()
    plain = pysimlr.fit("simlr", v, 1, iterations=30)
    one = multistart_fit("simlr", v, 1, n_starts=0, iterations=30)
    assert np.array_equal(one["u"].numpy(), plain["u"].numpy())
    assert one["best_energy"] == plain["best_energy"]
    assert one["n_distinct_solutions"] == 1 and one["energy_gap"] is None and one["chosen"] == "default"
    assert len(one["starts"]) == 1


def test_without_default_and_nothing_to_run():
    r = multistart_fit("simlr", toy(), 1, n_starts=2, include_default=False, iterations=20)
    assert [s["label"] for s in r["starts"]] == ["random_0", "random_1"]
    with pytest.raises(ValueError, match="nothing to run"):
        multistart_fit("simlr", toy(), 1, n_starts=0, include_default=False)
    with pytest.raises(ValueError, match="n_starts"):
        multistart_fit("simlr", toy(), 1, n_starts=-1)


def test_determinism_and_seed_dependence():
    v = toy()
    a = multistart_fit("simlr", v, 1, n_starts=3, iterations=30, random_state=5)
    b = multistart_fit("simlr", v, 1, n_starts=3, iterations=30, random_state=5)
    c = multistart_fit("simlr", v, 1, n_starts=3, iterations=30, random_state=6)
    assert [s["energy"] for s in a["starts"]] == [s["energy"] for s in b["starts"]]
    assert np.array_equal(a["u"].numpy(), b["u"].numpy())
    assert [s["seed"] for s in a["starts"]] != [s["seed"] for s in c["starts"]]


def test_n_jobs_two_equals_one():
    v = toy()
    a = multistart_fit("simlr", v, 1, n_starts=3, iterations=30, n_jobs=1)
    b = multistart_fit("simlr", v, 1, n_starts=3, iterations=30, n_jobs=2)
    assert a["chosen"] == b["chosen"] and a["n_distinct_solutions"] == b["n_distinct_solutions"]
    assert np.allclose([s["energy"] for s in a["starts"]], [s["energy"] for s in b["starts"]], atol=1e-6)
    assert np.allclose(a["u"].numpy(), b["u"].numpy(), atol=1e-4)


def test_mapping_views_and_k2():
    v = {f"m{i}": x for i, x in enumerate(toy())}
    r = multistart_fit("simlr", v, 2, n_starts=1, iterations=20)
    assert r["u"].shape == (150, 2) and r["view_names"] == ["m0", "m1", "m2"]


@pytest.mark.parametrize("method", ["pca", "nsa_pipeline"])
def test_methods_without_a_varying_start_raise(method):
    with pytest.raises(NotImplementedError, match=f"'{method}'.*initialization_type"):
        multistart_fit(method, toy(), 1, n_starts=2)


def test_method_without_comparable_energy_raises():
    with pytest.raises(NotImplementedError, match="'lend'.*best_energy"):
        multistart_fit("lend", toy(n=40), 1, n_starts=1, iterations=2)


def test_reserved_and_unknown_arguments():
    with pytest.raises(ValueError, match="sets"):
        multistart_fit("simlr", toy(), 1, n_starts=1, initialization_type="pca")
    with pytest.raises(ValueError, match="unknown method"):
        multistart_fit("nope", toy(), 1)
