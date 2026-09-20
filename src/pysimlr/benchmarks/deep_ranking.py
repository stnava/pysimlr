"""
Rigorous statistical ranking and inference engine for multi-view benchmarks.

Implements standard non-parametric hypothesis testing (Friedman test, Nemenyi
critical differences), exact distribution-free sign-flip permutation tests,
pairwise Wilcoxon signed-rank tests with Holm-Bonferroni correction, Bayesian
win-rate matrices, and multi-objective Pareto frontier identification following
Demšar (2006) and statistical best practices.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats


def build_ranking_blocks(
    df: pd.DataFrame,
    metric_col: str,
    model_col: str = "Model",
    dataset_col: str = "Dataset",
    seed_col: str = "Seed",
    floor_threshold: Optional[float] = 0.2,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    r"""
    Build the paired blocks for a Friedman/Nemenyi analysis, one per dataset.

    Demšar's :math:`N` is the number of **datasets**, not the number of fitted
    models. Pivoting on ``(Dataset, Seed)`` instead treats every seed as an
    independent block, which inflates :math:`N` tenfold and shrinks the
    critical difference by :math:`\sqrt{10}`. On this suite that was the
    difference between ``CD = 1.25`` (almost every pair "separated") and
    ``CD = 3.97`` (almost none), and it reversed the reported 2nd/3rd place.

    The inflation was not merely optimistic, it was partly vacuous: ``Heart``
    and ``Diabetes`` are loaded from fixed files and, under the old prefix
    split, produced byte-identical rows for all ten seeds. Twenty of the
    seventy "independent" blocks were exact duplicates.

    Seeds are averaged within a dataset, which is what they are: replicates of
    one task, useful for reducing the noise in that task's estimate and not for
    multiplying the number of tasks.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format results, one row per (dataset, model, seed).
    metric_col : str
        Metric to rank on.
    model_col, dataset_col, seed_col : str
        Column names.
    floor_threshold : float, optional, default=0.2
        Drop datasets where *no* model exceeds this score. Such a regime
        contributes a block of near-ties driven by noise, which adds rank
        variance without adding evidence -- on this suite the ``Sine`` regime
        left every model between 0.06 and 0.11. Pass ``None`` to keep all
        datasets.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, object]]
        ``(blocks, info)``. ``blocks`` is datasets x models, seeds averaged.
        ``info`` records ``n_datasets``, ``n_seeds``, ``dropped_floor`` and
        ``degenerate_datasets`` (those with no seed-to-seed variance in any
        model, which indicates a split or loader that ignores the seed).
    """
    all_datasets = list(pd.unique(df[dataset_col].dropna()))
    per_seed = df.pivot_table(
        index=[dataset_col, seed_col], columns=model_col, values=metric_col, aggfunc="mean"
    )

    # A dataset whose every model has zero variance across seeds is not being
    # replicated at all; surface it rather than silently averaging duplicates.
    degenerate = []
    for ds, grp in per_seed.groupby(level=0):
        if len(grp) > 1 and float(np.nanmax(grp.std(axis=0, ddof=0).values)) < 1e-12:
            degenerate.append(ds)

    blocks = per_seed.groupby(level=0).mean()

    # A metric can be undefined on a regime rather than merely low --
    # `support_recovery_score` is nan wherever the true basis is dense, because
    # there is no support to recover. Those datasets leave the ranking as
    # "not applicable", which is a different statement from "everything failed
    # here", and they must go before the floor rule so the two are not
    # conflated in the report.
    # `pivot_table` drops a group whose values are entirely NaN, so an
    # undefined regime disappears from `blocks` rather than showing up as a
    # NaN row. Recover it by difference against the input, or it would vanish
    # from the report without ever being mentioned.
    undefined = [d for d in all_datasets if d not in blocks.index]
    undefined += list(blocks.index[blocks.isna().all(axis=1)])
    blocks = blocks.drop(index=[d for d in undefined if d in blocks.index])

    # A dataset scored for some models but not others cannot be ranked: the
    # Friedman test needs a complete block.
    partial = list(blocks.index[blocks.isna().any(axis=1)])
    blocks = blocks.drop(index=partial)

    # Two datasets can be distinct tasks yet identical under a given metric.
    # `NonnegParts` and `NonnegPartsWeak` deliberately share a basis and differ
    # only in which latent carries the outcome, so they are two blocks for a
    # predictive metric and one block for any basis metric. Counting them twice
    # would re-introduce, on a smaller scale, exactly the pseudo-replication
    # that averaging the seeds removed.
    duplicate_blocks: List[List[str]] = []
    seen: List[str] = []
    for ds in blocks.index:
        for other in seen:
            if np.allclose(blocks.loc[ds].values, blocks.loc[other].values,
                           rtol=0, atol=1e-9, equal_nan=True):
                duplicate_blocks.append([other, ds])
                break
        else:
            seen.append(ds)

    dropped: List[str] = []
    if floor_threshold is not None and len(blocks):
        keep = blocks.max(axis=1) > floor_threshold
        dropped = list(blocks.index[~keep])
        if keep.any():
            blocks = blocks[keep]

    info = {
        "n_datasets": int(len(blocks)),
        "n_seeds": int(df[seed_col].nunique()),
        "dropped_floor": dropped,
        "dropped_undefined": undefined,
        "dropped_incomplete": partial,
        "degenerate_datasets": degenerate,
        "duplicate_blocks": duplicate_blocks,
        "floor_threshold": floor_threshold,
    }
    return blocks, info


def friedman_test(perf_matrix: np.ndarray, higher_is_better: bool = True) -> Dict[str, float]:
    """
    Perform the Friedman test and Iman-Davenport extension across models.

    Parameters
    ----------
    perf_matrix : np.ndarray
        Array of shape (N_datasets_or_tasks, K_models) containing performance scores.
    higher_is_better : bool, default=True
        If True, higher scores get better (lower numerical) ranks (rank 1 = best).
        If False, lower scores get better ranks.

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - 'chi2': Friedman chi-square statistic
        - 'p_value_chi2': Asymptotic p-value for chi-square test
        - 'f_stat': Iman-Davenport F statistic
        - 'p_value_f': Asymptotic p-value for F test
        - 'mean_ranks': Array of mean ranks for each model (1-indexed, lower is better)
    """
    mat = np.asarray(perf_matrix, dtype=float)
    if mat.ndim != 2:
        raise ValueError(f"perf_matrix must be 2D, got shape {mat.shape}")
    n, k = mat.shape
    if n < 2 or k < 2:
        raise ValueError(f"Need at least 2 datasets and 2 models, got ({n}, {k})")

    # Rank each row across models. If higher_is_better, negate values so smallest is best.
    ranks = np.zeros_like(mat)
    for i in range(n):
        row = -mat[i] if higher_is_better else mat[i]
        ranks[i] = stats.rankdata(row)

    mean_ranks = np.mean(ranks, axis=0)

    # Friedman chi2 statistic
    # chi2 = [12*N / (k*(k+1))] * sum(R_j^2) - 3*N*(k+1)
    ss_ranks = np.sum(mean_ranks ** 2)
    chi2_stat = (12.0 * n / (k * (k + 1))) * ss_ranks - 3.0 * n * (k + 1)
    p_chi2 = float(stats.chi2.sf(chi2_stat, df=k - 1))

    # Iman-Davenport F statistic: F_F = (N - 1) * chi2 / (N * (k - 1) - chi2)
    denom = n * (k - 1) - chi2_stat
    if denom <= 0:
        f_stat = np.inf
        p_f = 0.0
    else:
        f_stat = float((n - 1) * chi2_stat / denom)
        p_f = float(stats.f.sf(f_stat, dfn=k - 1, dfd=(k - 1) * (n - 1)))

    return {
        "chi2": float(chi2_stat),
        "p_value_chi2": p_chi2,
        "f_stat": float(f_stat),
        "p_value_f": p_f,
        "mean_ranks": mean_ranks,
    }


def compute_nemenyi_cd(n_models: int, n_datasets: int, alpha: float = 0.05) -> float:
    """
    Compute the Critical Difference (CD) for the two-tailed Nemenyi post-hoc test.

    CD = q_alpha * sqrt(k * (k + 1) / (6 * N))

    Parameters
    ----------
    n_models : int
        Number of models (k >= 2).
    n_datasets : int
        Number of independent datasets / replicates (N >= 1).
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    float
        Critical difference distance in mean rank units.
    """
    if n_models < 2 or n_datasets < 1:
        raise ValueError("n_models must be >= 2 and n_datasets >= 1")

    # studentized_range critical value divided by sqrt(2)
    try:
        q_val = float(stats.studentized_range.ppf(1.0 - alpha, n_models, np.inf) / np.sqrt(2.0))
    except Exception:
        # Fallback approximation for studentized range
        q_val = 2.948 if n_models == 7 else 2.569

    cd = q_val * np.sqrt((n_models * (n_models + 1)) / (6.0 * n_datasets))
    return float(cd)


def exact_sign_flip_permutation(diffs: Sequence[float], n_mc: int = 200_000) -> float:
    """
    Two-sided p-value from sign-flip permutation of paired differences.

    Parameters
    ----------
    diffs : Sequence[float]
        Array of paired differences across independent replicates.
    n_mc : int, default=200000
        Number of Monte-Carlo draws if enumeration is too large (> 20).

    Returns
    -------
    float
        Exact permutation p-value bounded below by 2 / 2**N.
    """
    arr = np.asarray(diffs, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n == 0:
        return float("nan")
    observed = abs(float(arr.mean()))
    if observed < 1e-15:
        return 1.0

    if n <= 20:
        hits = sum(
            1 for signs in itertools.product((-1.0, 1.0), repeat=n)
            if abs(float(np.mean(np.asarray(signs) * arr))) >= observed - 1e-14
        )
        return float(hits / (2 ** n))

    rng = np.random.default_rng(42)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_mc, n))
    means = np.abs((signs * arr).mean(axis=1))
    hits = int(np.sum(means >= observed - 1e-14))
    return float((hits + 1) / (n_mc + 1))


def pairwise_wilcoxon_holm(
    df: pd.DataFrame,
    metric_col: str,
    model_col: str = "Model",
    replicate_cols: Sequence[str] = ("Dataset", "Seed"),
    higher_is_better: bool = True
) -> pd.DataFrame:
    """
    Compute pairwise Wilcoxon signed-rank tests with Holm-Bonferroni correction
    and exact sign-flip permutation tests across all pairs of models.

    Parameters
    ----------
    df : pd.DataFrame
        Results dataframe.
    metric_col : str
        Target metric to compare.
    model_col : str, default='Model'
        Column identifying model architecture.
    replicate_cols : Sequence[str]
        Columns defining independent paired blocks (e.g. ['Dataset', 'Seed']).
    higher_is_better : bool, default=True
        Whether higher values represent superior performance.

    Returns
    -------
    pd.DataFrame
        Table of pairwise comparisons with columns:
        ['Model_A', 'Model_B', 'Mean_Diff', 'Mean_A', 'Mean_B',
         'P_Wilcoxon_Raw', 'P_Wilcoxon_Holm', 'P_Permutation', 'Significant']
    """
    models = sorted(df[model_col].dropna().unique())
    n_models = len(models)
    records = []

    # Pivot to align paired evaluations
    pivoted = df.pivot_table(
        index=list(replicate_cols),
        columns=model_col,
        values=metric_col,
        aggfunc="mean"
    )

    pairs = list(itertools.combinations(models, 2))
    for m_a, m_b in pairs:
        if m_a not in pivoted.columns or m_b not in pivoted.columns:
            continue
        sub = pivoted[[m_a, m_b]].dropna()
        if len(sub) == 0:
            continue

        vals_a = sub[m_a].values
        vals_b = sub[m_b].values
        diffs = (vals_a - vals_b) if higher_is_better else (vals_b - vals_a)
        mean_diff = float(np.mean(diffs))

        # Wilcoxon signed-rank test
        try:
            non_zero_diffs = diffs[diffs != 0]
            if len(non_zero_diffs) >= 3:
                w_stat, p_wilc = stats.wilcoxon(diffs, alternative="two-sided")
                p_wilc = float(p_wilc)
            else:
                p_wilc = 1.0
        except Exception:
            p_wilc = 1.0

        p_perm = exact_sign_flip_permutation(diffs)

        records.append({
            "Model_A": m_a,
            "Model_B": m_b,
            "Mean_A": float(np.mean(vals_a)),
            "Mean_B": float(np.mean(vals_b)),
            "Mean_Diff": mean_diff,
            "P_Wilcoxon_Raw": p_wilc,
            "P_Permutation": p_perm,
            "N_Pairs": len(sub),
        })

    if not records:
        return pd.DataFrame()

    res_df = pd.DataFrame(records)

    # Apply Holm-Bonferroni correction to Wilcoxon p-values
    p_vals = res_df["P_Wilcoxon_Raw"].values
    m = len(p_vals)
    order = np.argsort(p_vals)
    adj_p = np.zeros(m)
    for rank_idx, idx in enumerate(order):
        adj_p[idx] = p_vals[idx] * (m - rank_idx)
    # Enforce monotonicity: p_{adj}[i] = max(p_{adj}[i], p_{adj}[i-1]) in sorted order
    sorted_adj = adj_p[order]
    for i in range(1, m):
        sorted_adj[i] = max(sorted_adj[i], sorted_adj[i - 1])
    adj_p[order] = np.clip(sorted_adj, 0.0, 1.0)

    res_df["P_Wilcoxon_Holm"] = adj_p
    res_df["Significant"] = res_df["P_Wilcoxon_Holm"] < 0.05
    return res_df


def compute_bayesian_win_rates(
    df: pd.DataFrame,
    metric_col: str,
    model_col: str = "Model",
    replicate_cols: Sequence[str] = ("Dataset", "Seed"),
    higher_is_better: bool = True
) -> pd.DataFrame:
    """
    Compute empirical win-rate matrix P(Model_i > Model_j) across all paired tasks.

    Parameters
    ----------
    df : pd.DataFrame
        Benchmark results.
    metric_col : str
        Target metric.
    model_col : str, default='Model'
        Model column.
    replicate_cols : Sequence[str]
        Replication indices.
    higher_is_better : bool, default=True
        Whether higher values are better.

    Returns
    -------
    pd.DataFrame
        Square matrix where cell (row, col) contains P(row beats col).
    """
    models = sorted(df[model_col].dropna().unique())
    pivoted = df.pivot_table(
        index=list(replicate_cols),
        columns=model_col,
        values=metric_col,
        aggfunc="mean"
    )

    n_models = len(models)
    win_matrix = np.zeros((n_models, n_models))

    for i, m_a in enumerate(models):
        for j, m_b in enumerate(models):
            if i == j:
                win_matrix[i, j] = 0.5
                continue
            if m_a not in pivoted.columns or m_b not in pivoted.columns:
                continue
            sub = pivoted[[m_a, m_b]].dropna()
            if len(sub) == 0:
                continue
            diffs = (sub[m_a] - sub[m_b]) if higher_is_better else (sub[m_b] - sub[m_a])
            wins = np.sum(diffs > 1e-12) + 0.5 * np.sum(np.abs(diffs) <= 1e-12)
            win_matrix[i, j] = wins / len(sub)

    return pd.DataFrame(win_matrix, index=models, columns=models).round(4)


def compute_pareto_frontier(
    df: pd.DataFrame,
    objectives: Sequence[str],
    maximize: Sequence[bool],
    model_col: str = "Model"
) -> pd.DataFrame:
    """
    Identify non-dominated (Pareto optimal) models across multiple objectives.

    Parameters
    ----------
    df : pd.DataFrame
        Aggregated summary dataframe with one row per model.
    objectives : Sequence[str]
        Names of the metric columns to optimize.
    maximize : Sequence[bool]
        Boolean list specifying whether each objective is maximized (True) or minimized (False).
    model_col : str, default='Model'
        Column name for model.

    Returns
    -------
    pd.DataFrame
        DataFrame with an added boolean column 'is_pareto_optimal'.
    """
    summary = df[[model_col] + list(objectives)].copy().drop_duplicates(subset=[model_col])
    pts = summary[list(objectives)].values.copy()

    # Convert all objectives to maximization
    for j, max_flag in enumerate(maximize):
        if not max_flag:
            pts[:, j] = -pts[:, j]

    n_points = pts.shape[0]
    is_pareto = np.ones(n_points, dtype=bool)

    for i in range(n_points):
        for j in range(n_points):
            if i == j:
                continue
            # Point j dominates point i if all elements in j >= i and at least one is strictly >
            if np.all(pts[j] >= pts[i]) and np.any(pts[j] > pts[i]):
                is_pareto[i] = False
                break

    summary["is_pareto_optimal"] = is_pareto
    return summary


def rank_effect_size(
    df: pd.DataFrame,
    metric_col: str,
    model_col: str = "Model",
    dataset_col: str = "Dataset",
    seed_col: str = "Seed",
) -> Dict[str, object]:
    r"""
    Compare the between-model spread of a metric to its seed-to-seed noise.

    A Friedman test asks whether an ordering is *consistent*, not whether it is
    *large*. With eight models and eight datasets, differences far below the
    replicate noise still reach ``p < 0.05`` when they happen to fall the same
    way -- the test is behaving correctly and the ranking is still meaningless.

    This is not hypothetical. Ranking this suite on ``Axis_Sensitive_Metric``
    gave a clean-looking ordering at ``p = 0.016``, while the spread across all
    eight models was at or below the seed noise in every regime (median ratio
    1.23, and 0.10 on ``NonnegParts``, where the eight models sat inside
    0.0035 of one another against a seed standard deviation of 0.034). Any
    metric used for a headline ranking should be required to clear this
    diagnostic first.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format results.
    metric_col : str
        Metric to characterise.
    model_col, dataset_col, seed_col : str
        Column names.

    Returns
    -------
    Dict[str, object]
        ``per_dataset`` (a frame of spread, noise and their ratio),
        ``median_ratio``, ``max_ratio`` and ``resolvable`` -- ``False`` when
        the median ratio is below 2, i.e. the typical gap between the best and
        worst model is smaller than twice the noise on a single measurement.
    """
    means = df.pivot_table(index=dataset_col, columns=model_col, values=metric_col)
    sds = df.groupby([dataset_col, model_col])[metric_col].std()

    records = []
    for ds in means.index:
        row = means.loc[ds]
        if row.isna().all():
            continue
        spread = float(row.max() - row.min())
        noise = float(sds.loc[ds].mean())
        if not np.isfinite(spread) or not np.isfinite(noise) or noise <= 0:
            continue
        records.append({"Dataset": ds, "spread": spread, "noise": noise,
                        "ratio": spread / noise})

    if not records:
        return {"per_dataset": pd.DataFrame(), "median_ratio": float("nan"),
                "max_ratio": float("nan"), "resolvable": False}

    per_dataset = pd.DataFrame(records).set_index("Dataset")
    median_ratio = float(per_dataset["ratio"].median())
    return {
        "per_dataset": per_dataset,
        "median_ratio": median_ratio,
        "max_ratio": float(per_dataset["ratio"].max()),
        "resolvable": bool(median_ratio >= 2.0),
    }
