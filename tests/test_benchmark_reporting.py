"""What every benchmark model must report, and must not crash on.

Two defects this pins, both found by running the benchmark rather than the
suite:

* `pca` and `nsa_pipeline` build their result dict inline in
  `run_single_experiment` rather than in a model function, so they reported
  NaN for the rank fields while the other five models reported real numbers.
  The rank is the one quantity that reveals a basis returned with fewer usable
  components than the caller asked for -- predictive accuracy does not, having
  been measured flat across a w range over which the effective rank fell from
  3.00 to 2.27 -- so a model that omits it omits the only warning.
* `ned_simr_shared_private` failed 6 of 80 fits with "Input X contains NaN" at
  the scoring step. It no longer reproduces (0 of 80 over the same w grid and
  seeds) after the encoder stopped consolidating its basis on every access and
  after the backend's non-negative path became the certified prox, but nothing
  in the suite would have caught it and nothing would catch its return.
"""
import numpy as np
import pytest
import torch

pytest.importorskip("sklearn")

from pysimlr.benchmarks.runner import run_single_experiment

K = 2
RANK_FIELDS = ("effective_rank", "numerical_rank",
               "condition_number", "max_column_overlap")
JOINT_FIELDS = tuple("joint_" + f for f in RANK_FIELDS)
#: Models that fit ONE basis on the concatenated views. The runner slices it
#: for reporting, but a slice of a column-orthonormal matrix is not
#: orthonormal, so a per-view rank for these is a statement about how unevenly
#: a view contributes to the shared components -- not about lost components.
JOINT_BASIS_MODELS = {"pca", "nsa_pipeline"}

ALL_BENCHMARK_MODELS = ["pca", "nsa_pipeline", "linear", "lend", "ned",
                        "shared_private", "flow_v"]


def _case(seed=0, n=90, dims=(8, 6)):
    rng = np.random.default_rng(seed)
    u = np.abs(rng.standard_normal((n, K)))
    mats, true_v = [], []
    for p in dims:
        v = np.zeros((p, K))
        blk = p // K
        for j in range(K):
            v[j * blk:(j + 1) * blk, j] = 0.5 + rng.random(blk)
        true_v.append(v)
        mats.append(torch.tensor(u @ v.T + 0.1 * rng.standard_normal((n, p))).float())
    y = torch.tensor(u[:, 0] + 0.1 * rng.standard_normal(n)).float()
    return {"data": mats, "true_u": torch.tensor(u).float(), "true_v": true_v,
            "outcome": y, "shared_k": K, "is_classification": False}


@pytest.mark.parametrize("model_type", ALL_BENCHMARK_MODELS)
def test_every_benchmark_model_reports_a_joint_rank(model_type):
    """One number comparable across both model families.

    The per-view figure is not that number. Reported for `pca` it read as
    0.48-0.70 of k against ~0.99 for SiMLR, which looked like the
    unconstrained methods returning fewer components than asked for. They do
    not: PCA's actual basis has singular values [1, 1, 1], effective rank
    exactly k and condition number 1.000. The apparent deficit was a slice of
    an orthonormal matrix being measured as though it were a basis.
    """
    out = run_single_experiment(model_type, _case(), seed=0, iterations=6,
                                epochs=4, nsa_w=0.5, positivity="positive",
                                mixing_algorithm="avg")
    res = out["result"]
    for field in JOINT_FIELDS:
        assert field in res, f"{model_type} does not report {field}"
    e = res["joint_effective_rank"][0]
    assert 0.0 < e <= K + 1e-6, f"{model_type}: joint effective rank {e} outside (0, {K}]"
    assert res["joint_numerical_rank"][0] == K, (
        f"{model_type} returned a genuinely rank-deficient joint basis "
        f"({res['joint_numerical_rank'][0]} of {K})")


@pytest.mark.parametrize("model_type", ALL_BENCHMARK_MODELS)
def test_a_per_view_rank_is_reported_only_where_it_means_something(model_type):
    """Withheld for joint-basis models rather than reported and misread."""
    out = run_single_experiment(model_type, _case(), seed=0, iterations=6,
                                epochs=4, nsa_w=0.5, positivity="positive",
                                mixing_algorithm="avg")
    res = out["result"]
    if model_type in JOINT_BASIS_MODELS:
        assert res.get("basis_is_joint") is True
        for field in RANK_FIELDS:
            assert field not in res, (
                f"{model_type} has no per-view basis, so reporting {field} "
                f"per view invites comparing it against models that do")
    else:
        assert res.get("basis_is_joint") is False
        for field in RANK_FIELDS:
            assert field in res and len(res[field]) == len(res["v"])


def test_pca_is_not_rank_deficient():
    """Pinned as a property, since this is what the misreading claimed.

    `components_` has orthonormal rows, so the fitted basis has orthonormal
    columns: singular values all 1, participation ratio exactly k.
    """
    out = run_single_experiment("pca", _case(), seed=0, iterations=6)
    res = out["result"]
    assert res["joint_effective_rank"][0] == pytest.approx(float(K), abs=1e-4)
    assert res["joint_condition_number"][0] == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("nsa_w", [0.05, 0.5, 0.99])
def test_shared_private_is_scorable_across_the_constraint_range(nsa_w):
    """The NaN regression: a basis with a dead column reaches the scorer as
    NaN features, and the failure surfaces from sklearn rather than from the
    model, which is why it was hard to place."""
    out = run_single_experiment("shared_private", _case(), seed=0, iterations=6,
                                epochs=6, nsa_w=nsa_w, positivity="positive",
                                mixing_algorithm="avg", consolidate=False)
    m = out["metrics"]
    assert np.isfinite(m.get("test_r2", np.nan)), (
        f"shared_private produced an unscorable fit at nsa_w={nsa_w}")
    for v in out["result"]["v"]:
        assert torch.isfinite(v).all(), "non-finite entries in the returned basis"
        assert (v.abs().sum(dim=0) > 0).all(), "a column came back dead"
