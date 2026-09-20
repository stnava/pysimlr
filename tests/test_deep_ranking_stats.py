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


# --- Regression tests for the ranking-block construction -------------------
# These pin the three defects that made PCA look like the best method:
# pseudo-replicated seeds inflating N, floor regimes entering the ranks, and
# a ranking metric that cannot see the axes it is supposed to be judging.

def _toy_results():
    """Three datasets x three models x four seeds, with known pathologies."""
    import pandas as pd
    rows = []
    for seed in range(4):
        # 'Fixed' ignores the seed entirely -- the pseudo-replication case.
        rows += [{"Dataset": "Fixed", "Seed": seed, "Model": m, "Score": s}
                 for m, s in (("A", 0.80), ("B", 0.70), ("C", 0.60))]
        # 'Varying' is a genuine replicate.
        rows += [{"Dataset": "Varying", "Seed": seed, "Model": m, "Score": s + 0.01 * seed}
                 for m, s in (("A", 0.50), ("B", 0.60), ("C", 0.70))]
        # 'Floor' is a regime where nothing works.
        rows += [{"Dataset": "Floor", "Seed": seed, "Model": m, "Score": s}
                 for m, s in (("A", 0.05), ("B", 0.08), ("C", 0.03))]
    return pd.DataFrame(rows)


def test_ranking_blocks_are_datasets_not_dataset_seed_pairs():
    """N must be the number of datasets; seeds are replicates, not blocks."""
    from pysimlr.benchmarks.deep_ranking import build_ranking_blocks
    df = _toy_results()
    blocks, info = build_ranking_blocks(df, metric_col="Score")
    # 3 datasets x 4 seeds = 12 would be the inflated count.
    assert info["n_datasets"] == 2, "Floor regime should be dropped, leaving 2"
    assert len(blocks) == info["n_datasets"]
    assert info["n_seeds"] == 4


def test_ranking_blocks_flag_seed_invariant_datasets():
    """A dataset with zero seed-to-seed variance is not really replicated."""
    from pysimlr.benchmarks.deep_ranking import build_ranking_blocks
    blocks, info = build_ranking_blocks(_toy_results(), metric_col="Score")
    assert "Fixed" in info["degenerate_datasets"]
    assert "Varying" not in info["degenerate_datasets"]


def test_ranking_blocks_drop_floor_regimes():
    from pysimlr.benchmarks.deep_ranking import build_ranking_blocks
    df = _toy_results()
    blocks, info = build_ranking_blocks(df, metric_col="Score", floor_threshold=0.2)
    assert info["dropped_floor"] == ["Floor"]
    assert "Floor" not in blocks.index
    kept, _ = build_ranking_blocks(df, metric_col="Score", floor_threshold=None)
    assert "Floor" in kept.index


def test_inflating_n_shrinks_critical_difference():
    """The defect this guards: N=70 gave CD=1.25, N=7 gives CD=3.97."""
    from pysimlr.benchmarks.deep_ranking import compute_nemenyi_cd
    cd_honest = compute_nemenyi_cd(n_models=8, n_datasets=7)
    cd_inflated = compute_nemenyi_cd(n_models=8, n_datasets=70)
    assert cd_honest > 3.0 * cd_inflated * 0.9  # ~sqrt(10) apart
    assert cd_inflated < 1.5 < cd_honest


def test_undefined_metric_regimes_are_separated_from_floor_regimes():
    """`support_recovery_score` is nan where the truth is dense, not zero."""
    import pandas as pd
    from pysimlr.benchmarks.deep_ranking import build_ranking_blocks
    rows = []
    for seed in range(3):
        for m, s in (("A", 0.9), ("B", 0.8)):
            rows.append({"Dataset": "Sparse", "Seed": seed, "Model": m, "Score": s})
            rows.append({"Dataset": "Dense", "Seed": seed, "Model": m, "Score": float("nan")})
            rows.append({"Dataset": "Floor", "Seed": seed, "Model": m, "Score": 0.01})
    blocks, info = build_ranking_blocks(pd.DataFrame(rows), metric_col="Score")
    assert info["dropped_undefined"] == ["Dense"]
    assert info["dropped_floor"] == ["Floor"]
    assert list(blocks.index) == ["Sparse"]
    assert not blocks.isna().any().any(), "Friedman needs complete blocks"


def test_duplicate_blocks_are_detected():
    """Two regimes identical under a metric are one block, not two."""
    import pandas as pd
    from pysimlr.benchmarks.deep_ranking import build_ranking_blocks
    rows = []
    for seed in range(3):
        for m, s in (("A", 0.9), ("B", 0.5)):
            # Same basis, different outcome -> identical basis metric.
            rows.append({"Dataset": "Parts", "Seed": seed, "Model": m, "Score": s})
            rows.append({"Dataset": "PartsWeak", "Seed": seed, "Model": m, "Score": s})
            rows.append({"Dataset": "Other", "Seed": seed, "Model": m, "Score": s / 2})
    _, info = build_ranking_blocks(pd.DataFrame(rows), metric_col="Score")
    assert info["duplicate_blocks"] == [["Parts", "PartsWeak"]]
