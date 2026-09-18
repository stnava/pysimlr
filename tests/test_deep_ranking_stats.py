"""
Unit tests for deep statistical ranking and inference engine.
"""

import numpy as np
import pandas as pd
import pytest

from pysimlr.benchmarks.deep_ranking import (
    friedman_test,
    compute_nemenyi_cd,
    exact_sign_flip_permutation,
    pairwise_wilcoxon_holm,
    compute_bayesian_win_rates,
    compute_pareto_frontier,
)


def test_friedman_test_known():
    # 4 datasets, 3 models
    # Model 0 consistently best, Model 1 middle, Model 2 worst
    perf = np.array([
        [0.9, 0.8, 0.7],
        [0.85, 0.75, 0.65],
        [0.95, 0.80, 0.70],
        [0.88, 0.78, 0.68],
    ])
    res = friedman_test(perf, higher_is_better=True)
    assert np.allclose(res["mean_ranks"], [1.0, 2.0, 3.0])
    assert res["chi2"] > 0
    assert res["p_value_chi2"] < 0.05
    assert res["p_value_f"] < 0.05


def test_nemenyi_cd_k7():
    cd_10 = compute_nemenyi_cd(n_models=7, n_datasets=10, alpha=0.05)
    # CD = 2.9483 * sqrt(56 / 60) ~= 2.9483 * 0.96609 ~= 2.848
    assert 2.7 < cd_10 < 3.0

    cd_50 = compute_nemenyi_cd(n_models=7, n_datasets=50, alpha=0.05)
    assert cd_50 < cd_10


def test_exact_sign_flip_permutation():
    # For N=5, all positive differences: smallest attainable p-value is 2 / 2^5 = 0.0625
    diffs_5 = [1.0, 2.0, 1.5, 0.8, 1.2]
    p_perm_5 = exact_sign_flip_permutation(diffs_5)
    assert np.isclose(p_perm_5, 2.0 / 32.0, atol=1e-5)

    # For N=10, all positive differences: 2 / 2^10 = 2 / 1024 ~= 0.001953
    diffs_10 = [1.0] * 10
    p_perm_10 = exact_sign_flip_permutation(diffs_10)
    assert np.isclose(p_perm_10, 2.0 / 1024.0, atol=1e-5)

    # Zero difference returns 1.0
    assert np.isclose(exact_sign_flip_permutation([0.0, 0.0]), 1.0)


def test_pairwise_wilcoxon_holm_and_win_rates():
    # Create synthetic benchmark table
    rows = []
    for d in ["Dataset1", "Dataset2"]:
        for s in range(5):
            rows.append({"Dataset": d, "Seed": s, "Model": "ModelA", "Acc": 0.85 + 0.01 * s})
            rows.append({"Dataset": d, "Seed": s, "Model": "ModelB", "Acc": 0.75 + 0.01 * s})
            rows.append({"Dataset": d, "Seed": s, "Model": "ModelC", "Acc": 0.65 + 0.01 * s})
    df = pd.DataFrame(rows)

    res = pairwise_wilcoxon_holm(df, metric_col="Acc")
    assert len(res) == 3  # (A,B), (A,C), (B,C)
    assert "P_Wilcoxon_Holm" in res.columns
    assert "P_Permutation" in res.columns
    assert np.all(res["P_Wilcoxon_Holm"] >= res["P_Wilcoxon_Raw"])

    win_matrix = compute_bayesian_win_rates(df, metric_col="Acc")
    assert win_matrix.shape == (3, 3)
    assert np.isclose(win_matrix.loc["ModelA", "ModelB"], 1.0)
    assert np.isclose(win_matrix.loc["ModelB", "ModelA"], 0.0)
    assert np.isclose(win_matrix.loc["ModelA", "ModelA"], 0.5)


def test_compute_pareto_frontier():
    # Model 1: High acc (0.9), slow (100s)
    # Model 2: Med acc (0.8), fast (10s)
    # Model 3: Low acc (0.7), slow (120s) -> strictly dominated by Model 1 and Model 2
    df = pd.DataFrame([
        {"Model": "M1", "Acc": 0.9, "Time": 100.0},
        {"Model": "M2", "Acc": 0.8, "Time": 10.0},
        {"Model": "M3", "Acc": 0.7, "Time": 120.0},
    ])
    pareto = compute_pareto_frontier(
        df,
        objectives=["Acc", "Time"],
        maximize=[True, False],
        model_col="Model"
    )
    m1_opt = pareto.loc[pareto["Model"] == "M1", "is_pareto_optimal"].values[0]
    m2_opt = pareto.loc[pareto["Model"] == "M2", "is_pareto_optimal"].values[0]
    m3_opt = pareto.loc[pareto["Model"] == "M3", "is_pareto_optimal"].values[0]

    assert m1_opt is True or m1_opt == 1
    assert m2_opt is True or m2_opt == 1
    assert m3_opt is False or m3_opt == 0
