r"""Shared statistics helpers: multiplicity, Bayes factors, permutation and pooling.

One home for the small inferential routines that had been re-implemented,
privately and slightly differently, in ``antsxbrainmapping``
(``discovery_transport``, ``multiview_simlr``, ``triangulation_variance``,
``cognitive_networks``, ``scripts/run_four_hypothesis_triangulation.py``),
``antsxbrainmappingprivate`` and pysimlr itself (``screening._bh_qvalues``,
``benchmarks.deep_ranking.holm_adjust``). Every function takes and returns
numpy objects, is deterministic given its ``seed`` and raises on inputs it
cannot interpret instead of returning a neutral value.

Contents
--------
``bh_fdr``, ``holm``
    Benjamini-Hochberg step-up q-values and Holm-Bonferroni step-down
    adjusted p-values.
``jzs_bf10_correlation``, ``jzs_bf10_correlation_grid``
    Jeffreys-Zellner-Siow Bayes factor for a Pearson correlation (Wetzels &
    Wagenmakers, 2012, Eq. 13). Two implementations are kept because they do
    not agree: see the note under `jzs_bf10_correlation_grid`.
``jzs_bf10_one_sample_t``
    JZS Bayes factor for a one-sample t statistic (Rouder et al., 2009).
``permutation_correlations``
    Vectorised Pearson correlations of every column of ``X`` against ``y``
    (or, ``paired=True``, against the matching column of ``Y``) with
    parametric, permutation, BH and max-|r| FWER p-values, bootstrap CIs and
    BF10.
``freedman_lane_partial_p``
    Partial correlation and its Freedman-Lane permutation p-value.
``pooled_correlation_fisher_z``, ``stouffer_z``
    Fixed-effect meta-analytic pooling of correlations (Fisher z, inverse
    variance weights ``n - 3``) and of directional p-values (Stouffer).
``icc_3``
    Shrout-Fleiss ICC(3,1) and ICC(3,k) for an ``(n, k)`` table.
``adaptive_local_linear_curve``, ``modality_dose_recruitment``
    0-mass-anchored, covariate-adjusted adaptive-``N_eff`` Gaussian local
    linear dose-response curve with studentised maximum-modulus (``sup-t``)
    simultaneous bootstrap bands and multi-view stage-recruitment matrices.

scipy is imported inside the functions that need it so that ``import
pysimlr`` (which imports ``screening``, which imports `bh_fdr`) stays free
of scipy; ``tests/test_package_import.py`` enforces that.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

__all__ = [
    "adaptive_local_linear_curve",
    "bh_fdr",
    "freedman_lane_partial_p",
    "holm",
    "icc_3",
    "jzs_bf10_correlation",
    "jzs_bf10_correlation_grid",
    "jzs_bf10_one_sample_t",
    "modality_dose_recruitment",
    "permutation_correlations",
    "pooled_correlation_fisher_z",
    "stouffer_z",
]

#: Default Cauchy scale of the one-sample JZS prior on effect size (``sqrt(2)/2``,
#: the "medium" scale of Morey & Rouder's ``BayesFactor``).
ONE_SAMPLE_RSCALE_DEFAULT = math.sqrt(2.0) / 2.0


# --------------------------------------------------------------------------- helpers
def _as_1d_float(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}")
    return arr


def _check_p(p: np.ndarray, name: str = "p") -> None:
    if not np.all(np.isfinite(p)):
        raise ValueError(f"{name} contains non-finite values")
    if np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError(f"{name} must lie in [0, 1]")


# ----------------------------------------------------------------------- multiplicity
def bh_fdr(p: Sequence[float] | np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg step-up q-values.

    ``q_(i) = min_{j >= i} min(1, p_(j) m / j)`` over the sorted p-values,
    returned in the input order. Monotone in ``p`` and capped at 1.

    Parameters
    ----------
    p : array-like, shape (m,)
        Raw p-values in ``[0, 1]``. An empty input returns an empty array.

    Returns
    -------
    numpy.ndarray, shape (m,)

    Raises
    ------
    ValueError
        If ``p`` is not one-dimensional, contains non-finite values or values
        outside ``[0, 1]``.

    Examples
    --------
    >>> bh_fdr([0.01, 0.04, 0.03, 0.5]).round(4)
    array([0.04  , 0.0533, 0.0533, 0.5   ])
    """
    p = _as_1d_float(p, "p")
    m = p.size
    if m == 0:
        return p.copy()
    _check_p(p)
    order = np.argsort(p, kind="stable")
    ranked = p[order] * m / np.arange(1, m + 1, dtype=np.float64)
    q_sorted = np.minimum(1.0, np.minimum.accumulate(ranked[::-1])[::-1])
    out = np.empty(m, dtype=np.float64)
    out[order] = q_sorted
    return out


def holm(p: Sequence[float] | np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values.

    ``p~_(i) = max_{j <= i} min(1, p_(j) (m - j + 1))`` over the sorted
    p-values, returned in the input order. Controls the family-wise error
    rate without any dependence assumption.

    Parameters
    ----------
    p : array-like, shape (m,)
        Raw p-values in ``[0, 1]``. An empty input returns an empty array.

    Returns
    -------
    numpy.ndarray, shape (m,)

    Raises
    ------
    ValueError
        As for `bh_fdr`.

    Examples
    --------
    >>> holm([0.01, 0.04, 0.03, 0.5])
    array([0.04, 0.09, 0.09, 0.5 ])
    """
    p = _as_1d_float(p, "p")
    m = p.size
    if m == 0:
        return p.copy()
    _check_p(p)
    order = np.argsort(p, kind="stable")
    stepped = p[order] * np.arange(m, 0, -1, dtype=np.float64)
    adj_sorted = np.minimum(1.0, np.maximum.accumulate(stepped))
    out = np.empty(m, dtype=np.float64)
    out[order] = adj_sorted
    return out


# ----------------------------------------------------------------------- Bayes factors
def _log_prefactor_zs(n: int, rscale: float) -> float:
    """``log`` of the Zellner-Siow normalising constant ``sqrt(n rscale^2 / 2) / Gamma(1/2)``."""
    return 0.5 * math.log(0.5 * n * rscale * rscale) - math.lgamma(0.5)


def _jzs_bf10_correlation_scalar(r: float, n: int, rscale: float) -> float:
    from scipy import integrate

    r2 = r * r
    log_prefactor = _log_prefactor_zs(n, rscale)
    half_n_r2 = 0.5 * n * rscale * rscale

    def _integrand(g: float) -> float:
        if g <= 0.0:
            return 0.0
        log_val = (
            log_prefactor
            + 0.5 * (n - 2.0) * math.log1p(g)
            - 0.5 * (n - 1.0) * math.log1p((1.0 - r2) * g)
            - 1.5 * math.log(g)
            - half_n_r2 / g
        )
        if log_val < -700.0:
            return 0.0
        return math.exp(min(log_val, 700.0))

    integral, _ = integrate.quad(_integrand, 0.0, np.inf, limit=200, epsabs=1e-10, epsrel=1e-8)
    return float(max(integral, 1e-12))


def jzs_bf10_correlation(
    r: float | Sequence[float] | np.ndarray,
    n: int,
    *,
    rscale: float = 1.0,
) -> float | np.ndarray:
    r"""JZS Bayes factor ``BF10`` for a Pearson correlation by adaptive quadrature.

    Wetzels & Wagenmakers (2012, *Psychon. Bull. Rev.* 19:1057, Eq. 13),
    the Zellner-Siow regression Bayes factor for a single standardised
    predictor, evaluated by log-stabilised ``scipy.integrate.quad`` over
    ``g in (0, inf)``:

    .. math::

        BF_{10} = \frac{\sqrt{n s^2 / 2}}{\Gamma(1/2)} \int_0^\infty
        (1 + g)^{(n-2)/2} \, \bigl(1 + (1 - r^2) g\bigr)^{-(n-1)/2}
        \, g^{-3/2} e^{-n s^2 / (2 g)} \, dg

    with ``s = rscale`` the Cauchy scale of the prior on the standardised
    slope (``g ~ InvGamma(1/2, n s^2 / 2)``). ``rscale=1`` is the JZS
    default and the value behind every antsxbrainmapping ledger.

    Agrees with a 2,000,001-point log-grid reference to a maximum relative
    error of 7e-11 over ``r in [-0.95, 0.95]``, ``n in {10, 30, 52, 146}``.
    ``r`` is clipped to ``|r| <= 0.999999`` before integration, as in the
    source.

    Parameters
    ----------
    r : float or array-like
        Pearson correlation(s), each finite and in ``[-1, 1]``.
    n : int
        Sample size, at least 4.
    rscale : float, optional
        Prior scale, positive. Default 1.0.

    Returns
    -------
    float or numpy.ndarray
        ``BF10`` (evidence for a non-zero correlation), floored at 1e-12.
        Scalar in, scalar out; array in, array of the same shape out.
        ``BF01 = 1 / BF10``.

    Raises
    ------
    ValueError
        If ``n < 4``, ``rscale <= 0`` or any ``r`` is non-finite or
        outside ``[-1, 1]``.

    Provenance
    ----------
    Port of ``antsxbrainmapping.discovery_transport.jzs_bayes_factor_correlation``,
    which produced the ``bf10`` / ``bf01_null`` columns of the
    ``four_study_transport/results/*_hypothesis_ledger.csv`` ledgers and the
    ``bf10_*`` columns of ``triangulation_variance`` (through
    ``_univariate_perm_boot_stats``).
    """
    n = int(n)
    if n < 4:
        raise ValueError(f"n must be at least 4 for the correlation Bayes factor, got {n}")
    if not (rscale > 0.0):
        raise ValueError(f"rscale must be positive, got {rscale}")
    r_arr = np.asarray(r, dtype=np.float64)
    if not np.all(np.isfinite(r_arr)):
        raise ValueError("r contains non-finite values")
    if np.any(np.abs(r_arr) > 1.0):
        raise ValueError("r must lie in [-1, 1]")
    r_clip = np.clip(r_arr, -0.999999, 0.999999)
    flat = r_clip.reshape(-1)
    rscale_f = float(rscale)
    if flat.size <= 4:
        out = np.array([_jzs_bf10_correlation_scalar(float(rv), n, rscale_f) for rv in flat], dtype=np.float64)
    else:
        uniq_r2, inv = np.unique(flat * flat, return_inverse=True)
        if uniq_r2.size <= 4:
            uniq_out = np.array(
                [_jzs_bf10_correlation_scalar(math.sqrt(float(r2v)), n, rscale_f) for r2v in uniq_r2],
                dtype=np.float64,
            )
        else:
            from scipy import integrate

            omr2 = 1.0 - uniq_r2
            log_pref = _log_prefactor_zs(n, rscale_f)
            half_n_r2 = 0.5 * n * rscale_f * rscale_f
            c1 = 0.5 * (n - 2.0)
            c2 = -0.5 * (n - 1.0)
            g_probe = np.logspace(-4.0, 8.0, 49)
            lp = (
                log_pref
                + c1 * np.log1p(g_probe)[None, :]
                + c2 * np.log1p(omr2[:, None] * g_probe[None, :])
                - 1.5 * np.log(g_probe)[None, :]
                - half_n_r2 / g_probe[None, :]
            )
            shift = np.clip(np.max(lp, axis=1), -600.0, 650.0)

            def _vec_integrand(g: float) -> np.ndarray:
                gf = float(g)
                if gf <= 0.0:
                    return np.zeros_like(uniq_r2)
                lv = log_pref + c1 * math.log1p(gf) + c2 * np.log1p(omr2 * gf) - 1.5 * math.log(gf) - half_n_r2 / gf
                return np.where(lv < -700.0, 0.0, np.exp(np.minimum(lv, 700.0) - shift))

            val, _ = integrate.quad_vec(
                _vec_integrand, 0.0, np.inf, epsabs=1e-12, epsrel=1e-11, norm="max", limit=200
            )
            with np.errstate(over="ignore"):
                uniq_out = np.maximum(val * np.exp(shift), 1e-12)
        out = uniq_out[inv]
    if r_arr.ndim == 0:
        return float(out[0])
    return out.reshape(r_arr.shape)


def jzs_bf10_correlation_grid(r: float | Sequence[float] | np.ndarray, n: int) -> float | np.ndarray:
    """JZS correlation ``BF10`` on a fixed 400-point ``log g`` grid, ``log BF10`` clipped to ``[-50, 50]``.

    Exact port of ``antsxbrainmapping.multiview_simlr._jzs_bayes_factor_bf10``
    (``rscale`` fixed at 1). Kept as a distinct function because it does
    **not** reproduce `jzs_bf10_correlation`: the equivalence sweep over
    ``r in [-0.95, 0.95]`` (step 0.05) and ``n in {10, 30, 52, 146}`` found
    no cell within 1e-6 relative. Two defects relative to the quadrature:

    * the 400-point rectangle rule on ``u = log g in [-10, 12]`` carries a
      relative error of 4e-5 (median, n=10) to 8e-4 (max, n=146 at |r|=0.7);
    * ``log BF10`` is clipped to ``[-50, 50]``, so every ``BF10`` above
      ``e^50 = 5.2e21`` is reported as 5.2e21 (n=52 from |r|>=0.95, n=146
      from |r|>=0.75), a relative error approaching 1.

    Use it only to reproduce the ``bf10`` entries of outputs that were made
    with ``multiview_simlr._vectorized_permutation_correlations`` /
    ``_vectorized_paired_permutation_correlations``. New work should call
    `jzs_bf10_correlation`.

    Parameters
    ----------
    r : float or array-like
        Pearson correlation(s), finite. Clipped to ``|r| <= 0.9999`` as in
        the source.
    n : int
        Sample size, at least 4.

    Returns
    -------
    float or numpy.ndarray
        Same shape as ``r``.

    Raises
    ------
    ValueError
        If ``n < 4`` or ``r`` contains non-finite values.
    """
    n = int(n)
    if n < 4:
        raise ValueError(f"n must be at least 4 for the correlation Bayes factor, got {n}")
    r_in = np.asarray(r, dtype=np.float64)
    if not np.all(np.isfinite(r_in)):
        raise ValueError("r contains non-finite values")
    r_arr = np.clip(r_in, -0.9999, 0.9999)
    u_grid = np.linspace(-10.0, 12.0, 400, dtype=np.float64)
    du = float(u_grid[1] - u_grid[0])
    g = np.exp(u_grid)
    one_minus_r2 = np.maximum(1e-8, 1.0 - r_arr.reshape(-1, 1) ** 2)
    log_int = (
        0.5 * float(n - 2) * np.log1p(g)[None, :]
        - 0.5 * float(n - 1) * np.log1p(one_minus_r2 * g[None, :])
        - 0.5 * u_grid[None, :]
        - float(n) / (2.0 * g[None, :])
    )
    max_log = np.max(log_int, axis=1, keepdims=True)
    integral = np.sum(np.exp(log_int - max_log), axis=1) * du
    log_bf10 = 0.5 * np.log(float(n) / (2.0 * np.pi)) + max_log.reshape(-1) + np.log(np.maximum(1e-300, integral))
    bf10 = np.exp(np.clip(log_bf10, -50.0, 50.0))
    if r_in.ndim == 0:
        return float(bf10[0])
    return bf10.reshape(r_in.shape)


def jzs_bf10_one_sample_t(t: float, n: int, *, rscale: float = ONE_SAMPLE_RSCALE_DEFAULT) -> float:
    r"""JZS Bayes factor ``BF10`` for a one-sample (or paired) t statistic.

    Rouder, Speckman, Sun, Morey & Iverson (2009, *Psychon. Bull. Rev.*
    16:225, Eq. 1) with a Cauchy(0, ``rscale``) prior on the standardised
    effect, i.e. ``g ~ InvGamma(1/2, rscale^2 / 2)``:

    .. math::

        BF_{10} = \frac{\int_0^\infty (1 + n g s^2)^{-1/2}
        \bigl(1 + \tfrac{t^2}{(1 + n g s^2)\,\nu}\bigr)^{-(\nu+1)/2}
        (2\pi)^{-1/2} g^{-3/2} e^{-1/(2g)} \, dg}
        {(1 + t^2/\nu)^{-(\nu+1)/2}}, \qquad \nu = n - 1.

    Evaluated with the substitution ``g = u^-2`` (prior weight
    ``sqrt(2/pi) exp(-u^2/2) du`` on ``u in (0, inf)``) by log-stabilised
    ``scipy.integrate.quad``.

    Parameters
    ----------
    t : float
        One-sample t statistic, finite.
    n : int
        Number of observations (pairs), at least 2.
    rscale : float, optional
        Cauchy prior scale; default ``sqrt(2)/2`` ("medium").

    Returns
    -------
    float
        ``BF10``, clipped to ``[1e-12, 1e15]`` as in the source.

    Raises
    ------
    ValueError
        If ``n < 2``, ``rscale <= 0`` or ``t`` is not finite.

    Provenance
    ----------
    ``antsxbrainmapping.triangulation_variance._one_sample_jzs_bf10`` evaluates
    the same integral by the trapezoid rule on 1,200 points of
    ``u in [1e-4, 12]``; it produced the ``within_bf10`` columns of the
    triangulation ledgers. Against this quadrature the grid carries a maximum
    relative error of 1.7e-4 over ``t in [-8, 8]``, ``n in {5, 10, 30, 61, 146}``.
    """
    from scipy import integrate

    n = int(n)
    if n < 2:
        raise ValueError(f"n must be at least 2 for the one-sample Bayes factor, got {n}")
    if not (rscale > 0.0):
        raise ValueError(f"rscale must be positive, got {rscale}")
    if not np.isfinite(t):
        raise ValueError(f"t must be finite, got {t}")
    nu = float(n - 1)
    t2 = float(t) * float(t)
    log_denom = -0.5 * (nu + 1.0) * math.log1p(t2 / nu)
    n_r2 = float(n) * rscale * rscale
    log_c = 0.5 * math.log(2.0 / math.pi)

    def _integrand(u: float) -> float:
        if u <= 0.0:
            return 0.0
        one_plus = 1.0 + n_r2 / (u * u)
        log_val = (
            log_c
            - 0.5 * math.log(one_plus)
            - 0.5 * (nu + 1.0) * math.log1p(t2 / (one_plus * nu))
            - 0.5 * u * u
            - log_denom
        )
        if log_val < -700.0:
            return 0.0
        return math.exp(min(log_val, 700.0))

    integral, _ = integrate.quad(_integrand, 0.0, np.inf, limit=200, epsabs=1e-12, epsrel=1e-8)
    return float(np.clip(integral, 1e-12, 1e15))


# -------------------------------------------------------------- permutation inference
def _standardise_columns(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column z-scores (ddof=1) and the mask of columns with finite, non-degenerate variance."""
    x_c = x - np.mean(x, axis=0, keepdims=True)
    x_sd = np.std(x_c, axis=0, ddof=1, keepdims=True)
    valid = (x_sd.reshape(-1) > 1e-10) & np.all(np.isfinite(x), axis=0)
    x_z = np.where(valid.reshape(1, -1), x_c / np.where(x_sd > 1e-10, x_sd, 1.0), 0.0)
    return x_z, valid


def permutation_correlations(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_perms: int,
    n_boot: int = 0,
    seed: int,
    paired: bool = False,
    groups: Any = None,
) -> dict[str, np.ndarray]:
    """Columnwise Pearson correlations with permutation, FDR, FWER, bootstrap and Bayes-factor inference.

    ``paired=False``: ``r[j] = corr(X[:, j], y)`` for a single outcome
    ``y`` of shape ``(n,)``. ``paired=True``: ``r[j] = corr(X[:, j], y[:, j])``
    for homologous columns of ``y`` of shape ``(n, p)``. In both cases the
    permutation null permutes the rows of (standardised) ``y`` jointly across
    columns, so the max-|r| FWER p-value is a Westfall-Young step-free bound.

    Parameters
    ----------
    X : numpy.ndarray, shape (n, p)
    y : numpy.ndarray, shape (n,) or (n, p)
    n_perms : int
        Number of row permutations, at least 1. Two-sided p-values use the
        add-one estimator ``(1 + #{|r_null| >= |r|}) / (1 + n_perms)``.
    n_boot : int, optional
        Subject-bootstrap replicates for the 95 percent percentile CI and the
        probability of direction. ``0`` (default) skips the bootstrap and
        returns NaN for ``ci95_low``, ``ci95_high`` and ``prob_direction``.
    seed : int
        Seed of the ``numpy.random.default_rng`` stream. Permutations are
        drawn first, then bootstrap indices, so results are reproducible and
        match the antsxbrainmapping sources for the same seed.
    paired : bool, optional
        See above.
    groups : sequence of length n, optional
        Group labels (site, cohort, scanner). Default ``None`` reproduces the ungrouped result
        exactly. With groups, ``X`` and ``y`` are centered within group first, so ``r`` is the
        pooled within-group correlation and a group mean difference cannot produce an association;
        permutations move a row only within its group (the null keeps the group structure; an
        unrestricted permutation of data that differ in group means is not a valid null for
        within-group claims), bootstrap replicates resample within group, and ``p_param`` and
        ``bf10`` use the effective sample size ``n - G + 1`` (``G`` groups; ``n - G - 1`` degrees of
        freedom). Groups with fewer than 2 rows carry no within-group information. Requires ``n - G >= 3``.

    Returns
    -------
    dict[str, numpy.ndarray]
        Each value has shape ``(p,)``:

        ``r``
            Pearson r, clipped to ``|r| <= 1 - 1e-7``.
        ``p_param``
            Two-sided Student-t p-value on ``n - 2`` degrees of freedom.
        ``p_perm``
            Two-sided permutation p-value.
        ``q_fdr``
            `bh_fdr` of ``p_perm`` across the ``p`` columns.
        ``p_fwer``
            Max-|r| permutation family-wise p-value.
        ``ci95_low``, ``ci95_high``
            Bootstrap percentile 2.5 / 97.5 of r (NaN when ``n_boot=0``).
        ``prob_direction``
            Larger of the bootstrap fractions ``r > 0`` and ``r < 0`` (NaN
            when ``n_boot=0``).
        ``bf10``
            `jzs_bf10_correlation` (quadrature, ``rscale=1``) at ``r``.
        ``valid``
            Boolean mask of columns with finite entries and non-degenerate
            variance (in ``X`` and, if paired, in ``y``). Degenerate columns
            are kept in the output, annotated ``valid=False`` and reported as
            ``r=0``, ``p=1``, ``bf10=1`` (CI 0, probability of direction 0.5)
            exactly as the source did, so that ``q_fdr`` over all ``p``
            columns is reproducible.

    Raises
    ------
    ValueError
        If ``n < 4``, shapes disagree, ``n_perms < 1``, ``n_boot < 0``, or
        ``y`` is non-finite or (unpaired) has zero variance.

    Provenance
    ----------
    ``antsxbrainmapping.multiview_simlr._vectorized_permutation_correlations``
    (unpaired) and ``_vectorized_paired_permutation_correlations`` (paired).
    Every column except ``bf10`` reproduces them to floating-point round-off
    for the same ``seed``; their ``bf10`` came from `jzs_bf10_correlation_grid`.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be (n, p), got shape {X.shape}")
    n_samples, n_features = X.shape
    if n_samples < 4:
        raise ValueError(f"at least 4 rows are needed, got {n_samples}")
    g_codes = None
    n_eff = n_samples
    if groups is not None:
        from .utils import center_within_groups, group_codes, within_group_permutation
        g_codes = group_codes(groups, n_samples)
        n_groups = int(g_codes.max()) + 1
        n_eff = n_samples - n_groups + 1
        if n_eff < 4:
            raise ValueError(f"{n_groups} groups for {n_samples} rows leave {n_eff - 1} within-group degrees of freedom; need at least 3.")
    n_perms = int(n_perms)
    n_boot = int(n_boot)
    if n_perms < 1:
        raise ValueError(f"n_perms must be at least 1, got {n_perms}")
    if n_boot < 0:
        raise ValueError(f"n_boot must be non-negative, got {n_boot}")
    if paired:
        if y.shape != X.shape:
            raise ValueError(f"paired=True needs y of shape {X.shape}, got {y.shape}")
    else:
        if y.shape != (n_samples,):
            raise ValueError(f"y must have shape ({n_samples},), got {y.shape}")
        if not np.all(np.isfinite(y)):
            raise ValueError("y contains non-finite values")
        if np.std(y, ddof=1) <= 1e-10:
            raise ValueError("y has zero variance")

    if g_codes is not None:
        if not (np.all(np.isfinite(X)) and np.all(np.isfinite(y))):
            raise ValueError("groups requires finite X and y.")
        X = center_within_groups(X, g_codes)
        y = center_within_groups(y, g_codes)
        if not paired and np.std(y, ddof=1) <= 1e-10:
            raise ValueError("y has zero variance within groups")

    from scipy import stats as _sps

    x_z, valid = _standardise_columns(X)
    inv_dof = 1.0 / float(n_samples - 1)
    if paired:
        y_z, valid_y = _standardise_columns(y)
        valid = valid & valid_y
        x_z = np.where(valid.reshape(1, -1), x_z, 0.0)
        y_z = np.where(valid.reshape(1, -1), y_z, 0.0)
        r_obs = np.sum(x_z * y_z, axis=0) * inv_dof
    else:
        y_c = y - np.mean(y)
        y_z = y_c / np.std(y_c, ddof=1)
        r_obs = (x_z.T @ y_z) * inv_dof
    r_obs = np.clip(r_obs, -0.9999999, 0.9999999)
    df = max(1, n_eff - 2)
    t_obs = r_obs * np.sqrt(df / np.maximum(1e-12, 1.0 - r_obs**2))
    p_param = np.where(valid, 2.0 * _sps.t.sf(np.abs(t_obs), df=df), 1.0)

    rng = np.random.default_rng(int(seed))
    chunk = 64
    if paired:
        perm_all = np.empty((n_perms, n_samples), dtype=np.intp)
        for b in range(n_perms):
            perm_all[b] = rng.permutation(n_samples) if g_codes is None else within_group_permutation(rng, g_codes)
        r_null = np.empty((n_features, n_perms), dtype=np.float64)
        for b0 in range(0, n_perms, chunk):
            b1 = min(b0 + chunk, n_perms)
            r_null[:, b0:b1] = np.abs(np.einsum("np,bnp->pb", x_z, y_z[perm_all[b0:b1]]) * inv_dof)
    else:
        y_perms = np.empty((n_samples, n_perms), dtype=np.float64)
        for b in range(n_perms):
            y_perms[:, b] = y_z[rng.permutation(n_samples) if g_codes is None
                                else within_group_permutation(rng, g_codes)]
        r_null = np.abs((x_z.T @ y_perms) * inv_dof)
    abs_obs = np.abs(r_obs)[:, None]
    n_exceed = np.sum(r_null >= abs_obs, axis=1)
    p_perm = np.where(valid, (1.0 + n_exceed) / (1.0 + float(n_perms)), 1.0)
    q_fdr = bh_fdr(p_perm)
    max_r_null = np.max(r_null, axis=0)
    n_exceed_fwer = np.sum(max_r_null[None, :] >= abs_obs, axis=1)
    p_fwer = np.where(valid, (1.0 + n_exceed_fwer) / (1.0 + float(n_perms)), 1.0)

    if n_boot > 0:
        if g_codes is None:
            boot_idx = rng.integers(0, n_samples, size=(n_boot, n_samples))
        else:
            boot_idx = np.empty((n_boot, n_samples), dtype=np.intp)
            for g in range(int(g_codes.max()) + 1):
                rows_g = np.flatnonzero(g_codes == g)
                boot_idx[:, rows_g] = rows_g[rng.integers(0, rows_g.shape[0], size=(n_boot, rows_g.shape[0]))]
        r_boot = np.empty((n_features, n_boot), dtype=np.float64)
        if paired:
            for b0 in range(0, n_boot, chunk):
                b1 = min(b0 + chunk, n_boot)
                idx_c = boot_idx[b0:b1]
                xb = X[idx_c]
                xb_c = xb - np.mean(xb, axis=1, keepdims=True)
                yb = y[idx_c]
                yb_c = yb - np.mean(yb, axis=1, keepdims=True)
                denom = np.sqrt(np.sum(xb_c * xb_c, axis=1) * np.sum(yb_c * yb_c, axis=1)).T
                num = np.sum(xb_c * yb_c, axis=1).T
                r_boot[:, b0:b1] = np.where(denom > 1e-12, num / np.where(denom > 1e-12, denom, 1.0), 0.0)
        else:
            yb_all = y[boot_idx]
            yb_c_all = yb_all - np.mean(yb_all, axis=1, keepdims=True)
            ss_y_all = np.sum(yb_c_all * yb_c_all, axis=1)
            for b0 in range(0, n_boot, chunk):
                b1 = min(b0 + chunk, n_boot)
                xb = X[boot_idx[b0:b1]]
                xb_c = xb - np.mean(xb, axis=1, keepdims=True)
                denom = np.sqrt(np.sum(xb_c * xb_c, axis=1).T * ss_y_all[None, b0:b1])
                num = np.einsum("bnp,bn->pb", xb_c, yb_c_all[b0:b1])
                r_boot[:, b0:b1] = np.where(denom > 1e-12, num / np.where(denom > 1e-12, denom, 1.0), 0.0)
        pct_lo, pct_hi = np.percentile(r_boot, [2.5, 97.5], axis=1)
        ci95_low = np.where(valid, pct_lo, 0.0)
        ci95_high = np.where(valid, pct_hi, 0.0)
        pos_frac = np.mean(r_boot > 0.0, axis=1)
        neg_frac = np.mean(r_boot < 0.0, axis=1)
        prob_dir = np.where(valid, np.maximum(pos_frac, neg_frac), 0.5)
    else:
        ci95_low = np.full(n_features, np.nan)
        ci95_high = np.full(n_features, np.nan)
        prob_dir = np.full(n_features, np.nan)

    bf10 = np.ones(n_features, dtype=np.float64)
    if np.any(valid):
        bf10[valid] = jzs_bf10_correlation(r_obs[valid], n_eff)

    return {
        "r": r_obs,
        "p_param": p_param,
        "p_perm": p_perm,
        "q_fdr": q_fdr,
        "p_fwer": p_fwer,
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
        "prob_direction": prob_dir,
        "bf10": bf10,
        "valid": valid,
    }


def freedman_lane_partial_p(
    y: np.ndarray,
    x: np.ndarray,
    covariates: np.ndarray,
    *,
    n_perms: int,
    seed: int,
) -> tuple[float, float]:
    """Partial correlation ``r(x, y | covariates)`` and its Freedman-Lane permutation p-value.

    Both ``x`` and ``y`` are residualised on ``[1, covariates]`` by least
    squares; the partial correlation is the Pearson correlation of the
    residuals. Under Freedman & Lane (1983) the residuals of ``y`` are
    permuted and re-projected off the covariate span before correlating with
    the (already orthogonal) ``x`` residuals; the two-sided p-value is the
    add-one estimator over ``n_perms`` permutations.

    Parameters
    ----------
    y : numpy.ndarray, shape (n,)
    x : numpy.ndarray, shape (n,)
    covariates : numpy.ndarray, shape (n,) or (n, c)
        Nuisance regressors; an intercept is added.
    n_perms : int
        At least 1.
    seed : int

    Returns
    -------
    (partial_r, p_two_sided) : tuple[float, float]

    Raises
    ------
    ValueError
        If ``n < 4``, shapes disagree, any input is non-finite, ``x`` or
        ``y`` has zero variance before or after residualisation.

    Provenance
    ----------
    ``antsxbrainmapping.triangulation_variance._freedman_lane_partial_perm_p``
    (argument order there: ``x_target, z_nuis, y``; it returned ``(0, 1)``
    for ``n < 6`` or degenerate inputs, which raise here).
    """
    y = _as_1d_float(y, "y")
    x = _as_1d_float(x, "x")
    z = np.asarray(covariates, dtype=np.float64)
    n = y.size
    if n < 4:
        raise ValueError(f"at least 4 observations are needed, got {n}")
    if x.shape != (n,):
        raise ValueError(f"x must have shape ({n},), got {x.shape}")
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    if z.ndim != 2 or z.shape[0] != n:
        raise ValueError(f"covariates must have {n} rows, got shape {z.shape}")
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(x)) and np.all(np.isfinite(z))):
        raise ValueError("inputs contain non-finite values")
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        raise ValueError("x and y must have non-zero variance")
    n_perms = int(n_perms)
    if n_perms < 1:
        raise ValueError(f"n_perms must be at least 1, got {n_perms}")

    Z = np.column_stack([np.ones(n, dtype=np.float64), z])
    Z_pinv = np.linalg.pinv(Z)
    r_y = y - Z @ (Z_pinv @ y)
    r_x = x - Z @ (Z_pinv @ x)
    sx = float(np.std(r_x))
    sy = float(np.std(r_y))
    if sx <= 1e-12 or sy <= 1e-12:
        raise ValueError("x or y is fully explained by the covariates; the partial correlation is undefined")
    rx_z = (r_x - float(np.mean(r_x))) / sx
    ry_z = (r_y - float(np.mean(r_y))) / sy
    part_r = float(np.clip(np.mean(rx_z * ry_z), -1.0, 1.0))

    Q_z, _, _ = np.linalg.svd(Z, full_matrices=False)
    rng = np.random.default_rng(int(seed))
    perm_idx = np.array([rng.permutation(n) for _ in range(n_perms)], dtype=np.int64)
    ry_perm = r_y[perm_idx]
    ry_perp = ry_perm - (ry_perm @ Q_z) @ Q_z.T
    norms = np.linalg.norm(ry_perp, axis=1)
    rx_centred = r_x - float(np.mean(r_x))
    rx_unit = rx_centred / max(float(np.linalg.norm(rx_centred)), 1e-12)
    r_null = np.where(norms > 1e-12, (ry_perp @ rx_unit) / np.maximum(norms, 1e-12), 0.0)
    p_val = float((np.sum(np.abs(r_null) >= abs(part_r) - 1e-12) + 1) / (n_perms + 1))
    return part_r, p_val


# ------------------------------------------------------------------- meta-analytic pooling
def pooled_correlation_fisher_z(
    rs: Sequence[float] | np.ndarray,
    ns: Sequence[float] | np.ndarray,
    signs: Sequence[float] | np.ndarray | None = None,
) -> tuple[float, float]:
    """Fixed-effect pooled correlation by inverse-variance weighted Fisher z.

    ``z_i = atanh(clip(s_i r_i, -0.999, 0.999))`` with weights
    ``w_i = max(n_i - 3, 1)``; ``z_bar = sum(w z) / sum(w)``,
    ``se = 1 / sqrt(sum(w))``; the pooled r is ``tanh(z_bar)`` and the
    two-sided p-value is ``2 Phi(-|z_bar / se|)``.

    Parameters
    ----------
    rs : array-like, shape (k,)
        Per-study correlations.
    ns : array-like, shape (k,)
        Per-study sample sizes.
    signs : array-like, shape (k,), optional
        Expected direction per study; each ``r_i`` is multiplied by ``-1``
        where ``signs[i] < 0`` and ``+1`` otherwise, so that the pooled r is
        positive when the studies agree with their hypotheses. ``None``
        leaves the correlations as given.

    Returns
    -------
    (pooled_r, p_two_sided) : tuple[float, float]

    Raises
    ------
    ValueError
        On empty, mismatched or non-finite inputs.

    Provenance
    ----------
    The inline ``_pool_r`` closure of
    ``antsxbrainmapping/scripts/run_four_hypothesis_triangulation.py``
    (meta-analytic pooling block), reproduced term for term.
    """
    from scipy import stats as _sps

    rs = _as_1d_float(rs, "rs")
    ns = _as_1d_float(ns, "ns")
    if rs.size == 0:
        raise ValueError("rs is empty")
    if ns.shape != rs.shape:
        raise ValueError(f"ns must match rs in shape, got {ns.shape} vs {rs.shape}")
    if not (np.all(np.isfinite(rs)) and np.all(np.isfinite(ns))):
        raise ValueError("rs and ns must be finite")
    if signs is None:
        sgn = np.ones_like(rs)
    else:
        signs_arr = _as_1d_float(signs, "signs")
        if signs_arr.shape != rs.shape:
            raise ValueError(f"signs must match rs in shape, got {signs_arr.shape} vs {rs.shape}")
        sgn = np.where(signs_arr < 0.0, -1.0, 1.0)
    w = np.maximum(ns - 3.0, 1.0)
    z = np.arctanh(np.clip(sgn * rs, -0.999, 0.999))
    z_bar = float(np.sum(w * z) / np.sum(w))
    se_z = float(1.0 / np.sqrt(np.sum(w)))
    p_two = float(2.0 * _sps.norm.sf(abs(z_bar / se_z)))
    return float(np.tanh(z_bar)), p_two


def stouffer_z(
    p_directional: Sequence[float] | np.ndarray,
    weights: Sequence[float] | np.ndarray | None = None,
) -> tuple[float, float]:
    """Stouffer's weighted Z for one-sided (directional) p-values.

    ``z_i = Phi^-1(1 - p_i)`` with ``p_i`` clipped to ``[1e-15, 1 - 1e-15]``;
    ``Z = sum(sqrt(w_i) z_i) / sqrt(sum(w_i))``; ``p = 1 - Phi(Z)``.

    Parameters
    ----------
    p_directional : array-like, shape (k,)
        One-sided p-values, each testing the same direction.
    weights : array-like, shape (k,), optional
        Per-study weights ``w_i``, conventionally the sample sizes, so that
        each study enters with ``sqrt(n_i)`` (the optimal weighting for equal
        per-subject effect sizes). ``None`` gives equal weights,
        ``Z = sum(z_i) / sqrt(k)``.

    Returns
    -------
    (z, p_one_sided) : tuple[float, float]

    Raises
    ------
    ValueError
        On empty or mismatched inputs, p-values outside ``[0, 1]``, or
        non-positive / non-finite weights.

    Provenance
    ----------
    The within-person pooling block of
    ``antsxbrainmapping/scripts/run_four_hypothesis_triangulation.py``
    (``z_stouffer`` / ``p_stouffer`` with ``weights = within_n_subjects``).
    """
    from scipy import stats as _sps

    p = _as_1d_float(p_directional, "p_directional")
    if p.size == 0:
        raise ValueError("p_directional is empty")
    _check_p(p, "p_directional")
    if weights is None:
        w = np.ones_like(p)
    else:
        w = _as_1d_float(weights, "weights")
        if w.shape != p.shape:
            raise ValueError(f"weights must match p_directional in shape, got {w.shape} vs {p.shape}")
        if not np.all(np.isfinite(w)) or np.any(w <= 0.0):
            raise ValueError("weights must be finite and positive")
    z_ind = _sps.norm.isf(np.clip(p, 1e-15, 1.0 - 1e-15))
    z = float(np.sum(np.sqrt(w) * z_ind) / np.sqrt(np.sum(w)))
    return z, float(_sps.norm.sf(z))


# ------------------------------------------------------------------------- reliability
def icc_3(Y: np.ndarray) -> tuple[float, float]:
    """Shrout-Fleiss ICC(3,1) and ICC(3,k) of an ``(n, k)`` subjects-by-raters table.

    Two-way mixed, consistency: ``ICC(3,1) = (MS_B - MS_E) / (MS_B + (k-1) MS_E)``
    and ``ICC(3,k) = (MS_B - MS_E) / MS_B`` with ``MS_B`` the between-subject
    mean square (``k`` times the row-mean variance, ``n - 1`` df) and ``MS_E``
    the residual mean square after removing row and column means
    (``(n-1)(k-1)`` df). Rows (subjects) with any non-finite entry are dropped
    before the analysis.

    Parameters
    ----------
    Y : numpy.ndarray, shape (n, k)
        One row per subject, one column per rater / session / day.

    Returns
    -------
    (icc_3_1, icc_3_k) : tuple[float, float]
        NaN for either coefficient whose denominator is below 1e-12 (all
        subjects identical).

    Raises
    ------
    ValueError
        If ``Y`` is not two-dimensional, has fewer than 2 columns or fewer
        than 5 complete rows.

    Provenance
    ----------
    ``_icc_3_k_across_days`` in
    ``antsxbrainmapping/scripts/run_four_hypothesis_triangulation.py``
    (which returned NaN for ``n < 5`` or ``k < 2`` and 0.0 for a degenerate
    denominator).
    """
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim != 2:
        raise ValueError(f"Y must be (n, k), got shape {Y.shape}")
    complete = np.all(np.isfinite(Y), axis=1)
    Yc = Y[complete]
    n, k = Yc.shape
    if k < 2:
        raise ValueError(f"at least 2 columns are needed, got {k}")
    if n < 5:
        raise ValueError(f"at least 5 complete rows are needed, got {n}")
    grand = float(np.mean(Yc))
    row_m = np.mean(Yc, axis=1, keepdims=True)
    col_m = np.mean(Yc, axis=0, keepdims=True)
    ss_between = float(k * np.sum((row_m - grand) ** 2))
    ss_resid = float(np.sum((Yc - row_m - col_m + grand) ** 2))
    ms_b = ss_between / (n - 1)
    ms_e = ss_resid / ((n - 1) * (k - 1))
    denom_1 = ms_b + (k - 1) * ms_e
    icc_3_1 = float((ms_b - ms_e) / denom_1) if denom_1 > 1e-12 else float("nan")
    icc_3_k = float((ms_b - ms_e) / ms_b) if ms_b > 1e-12 else float("nan")
    return icc_3_1, icc_3_k


# --------------------------------------------------- continuous dose-response recruitment
def _normal_two_sided_quantile(ci_level: float) -> float:
    """Two-sided standard normal critical value ``z`` with ``P(|Z| <= z) = ci_level``."""
    if not (0.0 < ci_level < 1.0):
        raise ValueError(f"ci_level must lie strictly in (0, 1), got {ci_level}")
    lo, hi = 0.0, 10.0
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        if math.erf(mid * inv_sqrt2) < ci_level:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _as_optional_2d_covariates(
    covariates: Sequence[float] | np.ndarray | None,
    n_rows: int,
    name: str = "covariates",
) -> np.ndarray | None:
    if covariates is None:
        return None
    arr = np.asarray(covariates, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    elif arr.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D, got shape {arr.shape}")
    if arr.shape[0] != n_rows:
        raise ValueError(f"{name} row count {arr.shape[0]} != dose length {n_rows}")
    if arr.shape[1] == 0:
        return None
    return arr


def _partial_out_mean_preserving(
    y: np.ndarray,
    cov: np.ndarray | None,
) -> np.ndarray:
    """Partial out covariates on complete rows while preserving ``mean(y)``."""
    if cov is None:
        return y.copy()
    valid = np.isfinite(y) & np.all(np.isfinite(cov), axis=1)
    n_valid = int(np.sum(valid))
    if n_valid < 10:
        return y.copy()
    cov_v = cov[valid]
    col_sd = np.std(cov_v, axis=0)
    keep = col_sd > 1e-10
    if not np.any(keep):
        return y.copy()
    cov_centered = cov_v[:, keep] - np.mean(cov_v[:, keep], axis=0, keepdims=True)
    if n_valid <= cov_centered.shape[1] + 4:
        return y.copy()
    design = np.column_stack([np.ones(n_valid, dtype=np.float64), cov_centered])
    beta, *_ = np.linalg.lstsq(design, y[valid], rcond=None)
    out = np.full_like(y, np.nan, dtype=np.float64)
    out[valid] = y[valid] - cov_centered @ beta[1:]
    return out


def _solve_adaptive_gaussian_bandwidths(
    x_active: np.ndarray,
    grid: np.ndarray,
    *,
    target_neff: float,
    min_sigma: float,
    max_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve Gaussian bandwidth ``sigma(x_0)`` per grid coordinate for Kish ``N_eff``."""
    n_act = int(x_active.size)
    g_len = int(grid.size)
    eff_target = min(float(target_neff), max(12.0, 0.60 * n_act))

    dists = np.abs(grid[:, None] - x_active[None, :])
    lo = np.full(g_len, float(min_sigma), dtype=np.float64)
    hi = np.full(g_len, float(max_sigma), dtype=np.float64)

    for _ in range(36):
        mid = 0.5 * (lo + hi)
        u = dists / mid[:, None]
        raw_w = np.exp(-0.5 * u * u)
        sw = np.sum(raw_w, axis=1)
        sw2 = np.sum(raw_w * raw_w, axis=1)
        neff = np.where(sw2 > 1e-12, (sw * sw) / np.maximum(sw2, 1e-12), 0.0)
        need_wider = neff < eff_target
        lo = np.where(need_wider, mid, lo)
        hi = np.where(need_wider, hi, mid)

    sigma_grid = 0.5 * (lo + hi)
    u_fin = dists / sigma_grid[:, None]
    raw_w = np.exp(-0.5 * u_fin * u_fin)
    sw = np.sum(raw_w, axis=1)
    sw2 = np.sum(raw_w * raw_w, axis=1)
    neff_grid = np.where(sw2 > 1e-12, (sw * sw) / np.maximum(sw2, 1e-12), 0.0)
    return sigma_grid, raw_w, neff_grid


def _eval_anchored_local_linear(
    x_pos: np.ndarray,
    y_pos: np.ndarray,
    grid: np.ndarray,
    raw_w_pos: np.ndarray,
    sigma_grid: np.ndarray,
    *,
    has_zero_mass: bool,
    x_anchor: float,
    mu0: float,
    zero_anchor_mass: float,
    ridge: float = 2e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate weighted local linear fit ``y ~ b0 + b1 * (x - x_0)`` with 0-mass anchor."""
    if has_zero_mass and zero_anchor_mass > 0.0:
        w0 = (zero_anchor_mass * np.exp(-0.5 * ((grid - x_anchor) / sigma_grid) ** 2))[:, None]
        x_all = np.concatenate([[x_anchor], x_pos])
        y_all = np.concatenate([[mu0], y_pos])
        w_raw = np.hstack([w0, raw_w_pos])
    else:
        x_all = x_pos
        y_all = y_pos
        w_raw = raw_w_pos

    w_norm = w_raw / np.maximum(np.sum(w_raw, axis=1, keepdims=True), 1e-12)
    dx = x_all[None, :] - grid[:, None]
    s0 = np.sum(w_norm, axis=1)
    s1 = np.sum(w_norm * dx, axis=1)
    s2 = np.sum(w_norm * dx * dx, axis=1) + ridge
    t0 = np.sum(w_norm * y_all[None, :], axis=1)
    t1 = np.sum(w_norm * dx * y_all[None, :], axis=1)
    denom = np.maximum(s0 * s2 - s1 * s1, 1e-8)
    b0 = (s2 * t0 - s1 * t1) / denom
    b1 = (s0 * t1 - s1 * t0) / denom
    return b0, b1


def adaptive_local_linear_curve(
    dose: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    *,
    covariates: Sequence[float] | np.ndarray | None = None,
    grid: Sequence[float] | np.ndarray | None = None,
    n_grid: int = 120,
    log_transform: bool = True,
    target_neff: float = 32.0,
    min_sigma: float = 0.28,
    max_sigma: float = 3.0,
    zero_mass_min: int = 10,
    zero_anchor_cap: float = 8.0,
    min_local_count: int = 5,
    support_window: float = 0.60,
    n_boot: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
    allow_insufficient: bool = False,
) -> dict[str, Any]:
    """Fit a 0-mass-anchored, covariate-adjusted adaptive-``N_eff`` local linear curve.

    Estimates continuous outcome displacement ``Delta(x_0) = mu(x_0) - mu_0`` and
    local derivative ``d mu / d x`` across an evaluation grid using a Gaussian
    local linear kernel whose bandwidth ``sigma(x_0)`` adapts at every coordinate
    to maintain Kish effective sample size ``(sum w_i)^2 / sum w_i^2 = target_neff``
    within ``[min_sigma, max_sigma]``.

    When at least ``zero_mass_min`` observations lie at ``dose == 0``, those
    baseline observations are consolidated into a boundary anchor at ``x = 0``
    (weight capped at ``zero_anchor_cap``) so zero-inflation neither collapses
    positive-dose bandwidths nor disconnects the curve from the unexposed
    reference mean ``mu_0``. Grid coordinates outside the observed support are
    masked to ``NaN`` (``in_support == False``) rather than extrapolated.
    Uncertainty is quantified by subject-level bootstrap (refitting covariate
    partial residuals inside every replicate) yielding both pointwise intervals
    and studentised maximum-modulus (``sup-t``) simultaneous confidence bands.

    Parameters
    ----------
    dose : 1-D array-like of shape (n,)
        Continuous exposure, disease-duration or burden measurements. ``NaN``
        entries are excluded pairwise with ``y`` and ``covariates``. Must be
        non-negative on finite entries when ``log_transform=True``.
    y : 1-D array-like of shape (n,)
        Outcome or out-of-fold view score per observation.
    covariates : 1-D or 2-D array-like of shape (n,) or (n, q), optional
        Co-exposure or nuisance covariates partialled out via mean-preserving
        OLS both on the full sample and inside every bootstrap replicate.
    grid : 1-D array-like, optional
        Explicit strictly increasing evaluation coordinates in the regression
        domain (``log10(dose + 1)`` when ``log_transform=True``, raw ``dose``
        otherwise). Built automatically with ``n_grid`` points when ``None``.
    n_grid : int, default 120
        Number of grid coordinates when ``grid`` is ``None`` (must be >= 8).
    log_transform : bool, default True
        Map ``dose`` to ``x = log10(dose + 1)`` and invert onset coordinates via
        ``10**x - 1``.
    target_neff : float, default 32.0
        Target Kish effective sample size ``N_eff`` per grid point (>= 4.0).
    min_sigma, max_sigma : float, default (0.28, 3.0)
        Lower and upper bounds on Gaussian kernel standard deviation ``sigma(x_0)``
        in regression-domain units.
    zero_mass_min : int, default 10
        Minimum count of ``dose <= 0`` observations required to activate
        explicit 0-mass boundary anchoring.
    zero_anchor_cap : float, default 8.0
        Maximum effective mass assigned to the consolidated ``x = 0`` anchor in
        positive-tail local linear fits.
    min_local_count : int, default 5
        Minimum number of observed data points within ``+-support_window`` of
        ``x_0`` for ``x_0`` to be marked ``in_support``.
    support_window : float, default 0.60
        Half-window (in regression-domain units) used to verify local observation
        density.
    n_boot : int, default 1000
        Number of subject-level bootstrap resamples (>= 0).
    ci_level : float, default 0.95
        Confidence level in ``(0, 1)`` for pointwise and ``sup-t`` bands.
    seed : int, default 42
        Seed for a local ``numpy.random.default_rng`` stream.
    allow_insufficient : bool, default False
        When ``True``, return ``{"valid": False, ...}`` with ``NaN`` curves if
        too few finite observations exist instead of raising ``ValueError``.

    Returns
    -------
    dict[str, Any]
        Curve dictionary containing ``grid``, ``grid_dose``, ``in_support``,
        ``delta``, ``se``, ``pw_lo``, ``pw_hi``, ``sim_lo``, ``sim_hi``,
        ``slope``, ``slope_se``, ``sigma``, ``neff``, ``onset_pointwise``,
        ``onset_simultaneous``, ``c_simultaneous`` and sample diagnostics.
    """
    dose_arr = _as_1d_float(dose, "dose")
    y_arr = _as_1d_float(y, "y")
    if dose_arr.size != y_arr.size:
        raise ValueError(f"dose and y must have identical length, got {dose_arr.size} vs {y_arr.size}")
    if n_grid < 8:
        raise ValueError(f"n_grid must be >= 8, got {n_grid}")
    if target_neff < 4.0:
        raise ValueError(f"target_neff must be >= 4.0, got {target_neff}")
    if min_sigma <= 0.0 or max_sigma < min_sigma:
        raise ValueError(f"require 0 < min_sigma <= max_sigma, got ({min_sigma}, {max_sigma})")
    if n_boot < 0:
        raise ValueError(f"n_boot must be >= 0, got {n_boot}")
    z_pw = _normal_two_sided_quantile(ci_level)
    cov_mat = _as_optional_2d_covariates(covariates, dose_arr.size)

    finite_dose = dose_arr[np.isfinite(dose_arr)]
    if log_transform and finite_dose.size > 0 and np.any(finite_dose < 0.0):
        raise ValueError("dose must be non-negative on all finite entries when log_transform=True")

    if grid is not None:
        grid_arr = _as_1d_float(grid, "grid")
        if grid_arr.size < 4 or not np.all(np.isfinite(grid_arr)) or np.any(np.diff(grid_arr) <= 0.0):
            raise ValueError("grid must be finite, strictly increasing, and have length >= 4")
    else:
        grid_arr = None

    def _to_dose_scale(g_vals: np.ndarray | float) -> np.ndarray | float:
        if log_transform:
            return np.power(10.0, g_vals) - 1.0
        return g_vals

    def _invalid_payload(n_obs_val: int, g_vec: np.ndarray) -> dict[str, Any]:
        nan_vec = np.full(g_vec.size, np.nan, dtype=np.float64)
        return {
            "valid": False,
            "n_obs": int(n_obs_val),
            "n_zero": 0,
            "has_zero_mass": False,
            "n_pos": 0,
            "log_transform": bool(log_transform),
            "target_neff": float(target_neff),
            "ci_level": float(ci_level),
            "mu0": float("nan"),
            "se0": float("nan"),
            "z_pointwise": float(z_pw),
            "c_simultaneous": float("nan"),
            "median_neff": float("nan"),
            "median_sigma": float("nan"),
            "median_span_2sigma": float("nan"),
            "onset_pointwise": float("nan"),
            "onset_simultaneous": float("nan"),
            "onset_pointwise_grid": float("nan"),
            "onset_simultaneous_grid": float("nan"),
            "peak_delta": float("nan"),
            "peak_slope": float("nan"),
            "peak_slope_dose": float("nan"),
            "grid": g_vec.copy(),
            "grid_dose": np.asarray(_to_dose_scale(g_vec), dtype=np.float64),
            "in_support": np.zeros(g_vec.size, dtype=bool),
            "delta": nan_vec.copy(),
            "se": nan_vec.copy(),
            "pw_lo": nan_vec.copy(),
            "pw_hi": nan_vec.copy(),
            "sim_lo": nan_vec.copy(),
            "sim_hi": nan_vec.copy(),
            "slope": nan_vec.copy(),
            "slope_se": nan_vec.copy(),
            "sigma": nan_vec.copy(),
            "neff": nan_vec.copy(),
            "scatter_x": np.empty(0, dtype=np.float64),
            "scatter_y": np.empty(0, dtype=np.float64),
        }

    adj_y = _partial_out_mean_preserving(y_arr, cov_mat)
    valid = np.isfinite(dose_arr) & np.isfinite(adj_y)
    if cov_mat is not None:
        valid = valid & np.all(np.isfinite(cov_mat), axis=1)

    n_obs = int(np.sum(valid))
    fallback_grid = grid_arr if grid_arr is not None else np.linspace(0.3, 4.0, n_grid, dtype=np.float64)
    if n_obs < 18:
        if allow_insufficient:
            return _invalid_payload(n_obs, fallback_grid)
        raise ValueError(f"at least 18 complete observations are required, got {n_obs}")

    rc = dose_arr[valid]
    yv = adj_y[valid]
    raw_orig = y_arr[valid]
    cov_v = cov_mat[valid] if cov_mat is not None else None
    xv = np.log10(rc + 1.0) if log_transform else rc.copy()

    zero_mask = rc <= 0.0
    n_zeros = int(np.sum(zero_mask))
    has_zero_mass = zero_mass_min > 0 and n_zeros >= zero_mass_min

    if has_zero_mass:
        mu0 = float(np.mean(yv[zero_mask]))
        se0 = float(np.std(yv[zero_mask], ddof=1) / math.sqrt(n_zeros)) if n_zeros > 1 else 0.0
        x_pos = xv[~zero_mask]
        y_pos = yv[~zero_mask]
        zero_anchor_mass = float(min(float(n_zeros), float(zero_anchor_cap)))
        x_anchor = 0.0
        x_ref = 0.0
    else:
        x_pos = xv
        y_pos = yv
        zero_anchor_mass = 0.0
        x_anchor = float(np.min(xv))
        x_ref = float(np.percentile(xv, 15.0))
        ref_sub = xv <= np.percentile(xv, 25.0)
        n_ref = max(1, int(np.sum(ref_sub)))
        mu0 = float(np.mean(yv[ref_sub]))
        se0 = float(np.std(yv[ref_sub], ddof=1) / math.sqrt(n_ref)) if n_ref > 1 else 0.0

    if int(x_pos.size) < 12:
        if allow_insufficient:
            return _invalid_payload(n_obs, fallback_grid)
        raise ValueError(f"at least 12 positive/active dose observations are required, got {x_pos.size}")

    if grid_arr is None:
        g_lo = float(min(0.28, float(np.min(x_pos)))) if (has_zero_mass and log_transform) else float(np.percentile(x_pos, 1.5))
        g_hi = float(np.percentile(x_pos, 98.5))
        if g_hi <= g_lo + 1e-6:
            g_lo = float(np.min(xv))
            g_hi = float(np.max(xv))
        if g_hi <= g_lo + 1e-6:
            if allow_insufficient:
                return _invalid_payload(n_obs, fallback_grid)
            raise ValueError("dose has zero spread across observations")
        grid_arr = np.linspace(g_lo, g_hi, n_grid, dtype=np.float64)

    domain_span = float(grid_arr[-1] - grid_arr[0])
    tol_lo = 0.015 * domain_span if not log_transform else 0.05
    tol_hi = 0.010 * domain_span if not log_transform else 0.03
    x_min_obs = float(np.percentile(x_pos, 2.0)) if not has_zero_mass else float(grid_arr[0])
    x_max_obs = float(np.percentile(x_pos, 97.5))
    local_cnt = np.sum(np.abs(grid_arr[:, None] - xv[None, :]) <= float(support_window), axis=1)

    in_support = (
        (grid_arr >= max(float(grid_arr[0]), x_min_obs - tol_lo))
        & (grid_arr <= x_max_obs + tol_hi)
        & (local_cnt >= int(min_local_count))
    )
    if int(np.sum(in_support)) < 6:
        if allow_insufficient:
            return _invalid_payload(n_obs, grid_arr)
        raise ValueError(
            f"fewer than 6 grid coordinates have sufficient local support ({int(np.sum(in_support))})"
        )

    grid_sub = grid_arr[in_support]
    sigma_sub, raw_w_pos, neff_sub = _solve_adaptive_gaussian_bandwidths(
        x_pos,
        grid_sub,
        target_neff=target_neff,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
    )
    b0_sub, b1_sub = _eval_anchored_local_linear(
        x_pos,
        y_pos,
        grid_sub,
        raw_w_pos,
        sigma_sub,
        has_zero_mass=has_zero_mass,
        x_anchor=x_anchor,
        mu0=mu0,
        zero_anchor_mass=zero_anchor_mass,
    )
    if not has_zero_mass:
        ref_idx = int(np.argmin(np.abs(grid_sub - x_ref)))
        mu0_eff = float(b0_sub[ref_idx])
    else:
        mu0_eff = mu0
    delta_sub = b0_sub - mu0_eff

    rng = np.random.default_rng(seed)
    boot_deltas = np.full((n_boot, grid_sub.size), np.nan, dtype=np.float64)
    boot_slopes = np.full((n_boot, grid_sub.size), np.nan, dtype=np.float64)
    valid_boots = 0

    for _ in range(n_boot):
        idx = rng.integers(0, n_obs, size=n_obs)
        rc_b = rc[idx]
        raw_y_b = raw_orig[idx]
        cov_b = cov_v[idx] if cov_v is not None else None
        y_b = _partial_out_mean_preserving(raw_y_b, cov_b)
        xv_b = np.log10(rc_b + 1.0) if log_transform else rc_b.copy()

        if has_zero_mass:
            zm_b = rc_b <= 0.0
            n_zb = int(np.sum(zm_b))
            if n_zb < 3 or int(np.sum(~zm_b)) < 10:
                continue
            mu0_b = float(np.mean(y_b[zm_b]))
            xp_b = xv_b[~zm_b]
            yp_b = y_b[~zm_b]
            zam_b = float(min(float(n_zb), float(zero_anchor_cap)))
        else:
            xp_b = xv_b
            yp_b = y_b
            mu0_b = 0.0
            zam_b = 0.0

        dists_b = np.abs(grid_sub[:, None] - xp_b[None, :])
        rw_b = np.exp(-0.5 * (dists_b / sigma_sub[:, None]) ** 2)
        b0_b, b1_b = _eval_anchored_local_linear(
            xp_b,
            yp_b,
            grid_sub,
            rw_b,
            sigma_sub,
            has_zero_mass=has_zero_mass,
            x_anchor=x_anchor,
            mu0=mu0_b,
            zero_anchor_mass=zam_b,
        )
        if not has_zero_mass:
            ref_idx = int(np.argmin(np.abs(grid_sub - x_ref)))
            mu0_b_eff = float(b0_b[ref_idx])
        else:
            mu0_b_eff = mu0_b

        boot_deltas[valid_boots] = b0_b - mu0_b_eff
        boot_slopes[valid_boots] = b1_b
        valid_boots += 1

    if valid_boots >= 2:
        boot_deltas = boot_deltas[:valid_boots]
        boot_slopes = boot_slopes[:valid_boots]
        se_sub = np.std(boot_deltas, axis=0, ddof=1)
        se_slope_sub = np.std(boot_slopes, axis=0, ddof=1)
        y_sd_floor = max(1e-6, 0.045 * float(np.std(yv, ddof=1)))
        se_floor = np.maximum(se_sub, y_sd_floor)
        sup_t_vals = np.max(np.abs(boot_deltas - delta_sub[None, :]) / se_floor[None, :], axis=1)
        c_sim = float(np.percentile(sup_t_vals, 100.0 * ci_level)) if valid_boots >= 20 else float(1.25 * z_pw)
    else:
        se_sub = np.full(grid_sub.size, np.nan, dtype=np.float64)
        se_slope_sub = np.full(grid_sub.size, np.nan, dtype=np.float64)
        c_sim = float("nan")

    pw_lo_sub = delta_sub - z_pw * se_sub
    pw_hi_sub = delta_sub + z_pw * se_sub
    sim_lo_sub = delta_sub - c_sim * se_sub
    sim_hi_sub = delta_sub + c_sim * se_sub

    min_run = max(2, min(5, grid_sub.size // 20))

    def _first_sustained_crossing(lb_arr: np.ndarray) -> tuple[float, float]:
        if not np.any(np.isfinite(lb_arr)):
            return float("nan"), float("nan")
        above = np.isfinite(lb_arr) & (lb_arr > 0.0)
        for idx_g in range(above.size):
            if above[idx_g] and np.all(above[idx_g : min(above.size, idx_g + min_run)]):
                g_hit = float(grid_sub[idx_g])
                return float(_to_dose_scale(g_hit)), g_hit
        return float("nan"), float("nan")

    onset_pw, onset_pw_g = _first_sustained_crossing(pw_lo_sub)
    onset_sim, onset_sim_g = _first_sustained_crossing(sim_lo_sub)

    def _pad(arr_sub: np.ndarray) -> np.ndarray:
        full = np.full(grid_arr.size, np.nan, dtype=np.float64)
        full[in_support] = arr_sub
        return full

    peak_slope_idx = int(np.argmax(b1_sub))
    med_sig = float(np.median(sigma_sub))
    return {
        "valid": True,
        "n_obs": n_obs,
        "n_zero": n_zeros,
        "has_zero_mass": bool(has_zero_mass),
        "n_pos": int(x_pos.size),
        "log_transform": bool(log_transform),
        "target_neff": float(target_neff),
        "ci_level": float(ci_level),
        "mu0": float(mu0_eff),
        "se0": float(se0),
        "z_pointwise": float(z_pw),
        "c_simultaneous": float(c_sim),
        "median_neff": float(np.median(neff_sub)),
        "median_sigma": med_sig,
        "median_span_2sigma": float(2.0 * med_sig),
        "onset_pointwise": onset_pw,
        "onset_simultaneous": onset_sim,
        "onset_pointwise_grid": onset_pw_g,
        "onset_simultaneous_grid": onset_sim_g,
        "peak_delta": float(np.max(delta_sub)),
        "peak_slope": float(b1_sub[peak_slope_idx]),
        "peak_slope_dose": float(_to_dose_scale(float(grid_sub[peak_slope_idx]))),
        "grid": grid_arr.copy(),
        "grid_dose": np.asarray(_to_dose_scale(grid_arr), dtype=np.float64),
        "in_support": in_support.copy(),
        "delta": _pad(delta_sub),
        "se": _pad(se_sub),
        "pw_lo": _pad(pw_lo_sub),
        "pw_hi": _pad(pw_hi_sub),
        "sim_lo": _pad(sim_lo_sub),
        "sim_hi": _pad(sim_hi_sub),
        "slope": _pad(b1_sub),
        "slope_se": _pad(se_slope_sub),
        "sigma": _pad(sigma_sub),
        "neff": _pad(neff_sub),
        "scatter_x": xv.copy(),
        "scatter_y": (yv - mu0_eff).copy(),
    }


def modality_dose_recruitment(
    scores: Mapping[str, Sequence[float] | np.ndarray] | np.ndarray,
    doses: Mapping[str, Sequence[float] | np.ndarray] | np.ndarray,
    *,
    covariates: Mapping[str, Sequence[float] | np.ndarray | None] | Sequence[float] | np.ndarray | None = None,
    grids: Mapping[str, Sequence[float] | np.ndarray] | Sequence[float] | np.ndarray | None = None,
    view_names: Sequence[str] | None = None,
    dose_names: Sequence[str] | None = None,
    standardize_scores: bool = False,
    target_neff: float = 32.0,
    min_sigma: float = 0.28,
    max_sigma: float = 3.0,
    zero_mass_min: int = 10,
    zero_anchor_cap: float = 8.0,
    min_local_count: int = 5,
    support_window: float = 0.60,
    n_grid: int = 120,
    log_transform: bool = True,
    n_boot: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate continuous stage-dependent modality recruitment across exposure axes.

    Runs :func:`adaptive_local_linear_curve` across every ``(dose_name, view_name)``
    pair while preserving view-specific missingness (unscanned subjects carry
    ``NaN`` in ``scores`` and have their uncovered dose regions masked via
    ``in_support == False``) and optional dose-specific co-exposure adjustment.

    Parameters
    ----------
    scores : Mapping[str, array-like] or 2-D ndarray of shape (n, M)
        Per-view outcome or out-of-fold scores.
    doses : Mapping[str, array-like] or 1-D / 2-D ndarray of shape (n,) or (n, D)
        One or more continuous exposure, disease-duration or burden axes.
    covariates : Mapping[str, array-like or None] or array-like, optional
        Either a single covariate array applied to all ``doses`` or a mapping
        keyed by ``dose_name`` supplying regime-specific co-exposures.
    grids : Mapping[str, array-like] or 1-D array-like, optional
        Optional shared 1-D evaluation grid or per-dose grid mapping.
    view_names, dose_names : Sequence[str], optional
        Explicit names (or ordering subset when inputs are mappings).
    standardize_scores : bool, default False
        If ``True``, standardise each view score to unit variance (``ddof=1``)
        across its finite observations prior to curve estimation.
    target_neff, min_sigma, max_sigma, zero_mass_min, zero_anchor_cap, min_local_count, support_window, n_grid, log_transform, n_boot, ci_level, seed
        Forwarded to :func:`adaptive_local_linear_curve`.

    Returns
    -------
    dict[str, Any]
        Result object of ``kind="modality_dose_recruitment"`` accepted directly by
        :func:`pysimlr.plot_dose_recruitment`, containing ``view_names``,
        ``dose_names``, ``curves`` (``curves[dose_name][view_name]``) and
        ``summary_rows``.
    """
    if isinstance(scores, Mapping):
        if len(scores) == 0:
            raise ValueError("scores mapping must not be empty")
        v_names = tuple(view_names) if view_names is not None else tuple(scores.keys())
        score_map = {str(k): _as_1d_float(scores[k], f"scores[{k!r}]") for k in v_names}
    else:
        s_arr = np.asarray(scores, dtype=np.float64)
        if s_arr.ndim == 1:
            s_arr = s_arr[:, None]
        elif s_arr.ndim != 2 or s_arr.shape[1] == 0:
            raise ValueError(f"scores array must have shape (n, M) with M >= 1, got {s_arr.shape}")
        if view_names is not None:
            if len(view_names) != s_arr.shape[1]:
                raise ValueError(f"len(view_names)={len(view_names)} != scores.shape[1]={s_arr.shape[1]}")
            v_names = tuple(str(v) for v in view_names)
        else:
            v_names = tuple(f"view_{i}" for i in range(s_arr.shape[1]))
        score_map = {v_names[i]: s_arr[:, i] for i in range(s_arr.shape[1])}

    n_rows = next(iter(score_map.values())).size
    for vn, vec in score_map.items():
        if vec.size != n_rows:
            raise ValueError(f"scores[{vn!r}] length {vec.size} != {n_rows}")
        if standardize_scores:
            fin = np.isfinite(vec)
            if int(np.sum(fin)) >= 5:
                sd = float(np.std(vec[fin], ddof=1))
                if sd > 1e-10:
                    mu = float(np.mean(vec[fin]))
                    score_map[vn] = np.where(fin, (vec - mu) / sd, np.nan)

    if isinstance(doses, Mapping):
        if len(doses) == 0:
            raise ValueError("doses mapping must not be empty")
        d_names = tuple(dose_names) if dose_names is not None else tuple(doses.keys())
        dose_map = {str(k): _as_1d_float(doses[k], f"doses[{k!r}]") for k in d_names}
    else:
        d_arr = np.asarray(doses, dtype=np.float64)
        if d_arr.ndim == 1:
            d_arr = d_arr[:, None]
            default_dnames = ("dose",)
        elif d_arr.ndim == 2 and d_arr.shape[1] >= 1:
            default_dnames = tuple(f"dose_{i}" for i in range(d_arr.shape[1]))
        else:
            raise ValueError(f"doses array must be 1-D or 2-D, got {d_arr.shape}")
        if dose_names is not None:
            if len(dose_names) != d_arr.shape[1]:
                raise ValueError(f"len(dose_names)={len(dose_names)} != doses.shape[1]={d_arr.shape[1]}")
            d_names = tuple(str(d) for d in dose_names)
        else:
            d_names = default_dnames
        dose_map = {d_names[i]: d_arr[:, i] for i in range(d_arr.shape[1])}

    for dn, dvec in dose_map.items():
        if dvec.size != n_rows:
            raise ValueError(f"doses[{dn!r}] length {dvec.size} != {n_rows}")

    curves: dict[str, dict[str, dict[str, Any]]] = {}
    summary_rows: list[dict[str, Any]] = []

    any_score_finite = np.zeros(n_rows, dtype=bool)
    for vec in score_map.values():
        any_score_finite |= np.isfinite(vec)

    for dn in d_names:
        curves[dn] = {}
        cov_dn = covariates.get(dn) if isinstance(covariates, Mapping) else covariates
        grid_dn = grids.get(dn) if isinstance(grids, Mapping) else grids
        if grid_dn is None:
            d_raw = dose_map[dn]
            d_valid = np.isfinite(d_raw) & any_score_finite & ((d_raw >= 0.0) if log_transform else True)
            if cov_dn is not None:
                c_tmp = _as_optional_2d_covariates(cov_dn, n_rows)
                if c_tmp is not None:
                    d_valid &= np.all(np.isfinite(c_tmp), axis=1)
            if int(np.sum(d_valid)) >= 4:
                rc_all = d_raw[d_valid]
                xv_all = np.log10(rc_all + 1.0) if log_transform else rc_all
                zm_all = rc_all <= 0.0
                hz_all = zero_mass_min > 0 and int(np.sum(zm_all)) >= zero_mass_min
                xp_all = xv_all[~zm_all] if hz_all else xv_all
                if xp_all.size >= 2:
                    g_lo = (
                        float(min(0.28, float(np.min(xp_all))))
                        if (hz_all and log_transform)
                        else float(np.percentile(xp_all, 1.5))
                    )
                    g_hi = float(np.percentile(xp_all, 98.5))
                    if g_hi <= g_lo + 1e-6:
                        g_lo, g_hi = float(np.min(xv_all)), float(np.max(xv_all))
                    if g_hi > g_lo + 1e-6:
                        grid_dn = np.linspace(g_lo, g_hi, int(n_grid), dtype=np.float64)
        for vn in v_names:
            res = adaptive_local_linear_curve(
                dose_map[dn],
                score_map[vn],
                covariates=cov_dn,
                grid=grid_dn,
                n_grid=n_grid,
                log_transform=log_transform,
                target_neff=target_neff,
                min_sigma=min_sigma,
                max_sigma=max_sigma,
                zero_mass_min=zero_mass_min,
                zero_anchor_cap=zero_anchor_cap,
                min_local_count=min_local_count,
                support_window=support_window,
                n_boot=n_boot,
                ci_level=ci_level,
                seed=seed,
                allow_insufficient=True,
            )
            curves[dn][vn] = res
            summary_rows.append(
                {
                    "dose": dn,
                    "view": vn,
                    "valid": bool(res["valid"]),
                    "n_obs": int(res["n_obs"]),
                    "n_zero": int(res["n_zero"]),
                    "median_neff": float(res["median_neff"]),
                    "median_sigma": float(res["median_sigma"]),
                    "median_span_2sigma": float(res["median_span_2sigma"]),
                    "c_simultaneous": float(res["c_simultaneous"]),
                    "onset_pointwise": float(res["onset_pointwise"]),
                    "onset_simultaneous": float(res["onset_simultaneous"]),
                    "peak_delta": float(res["peak_delta"]),
                    "peak_slope": float(res["peak_slope"]),
                    "peak_slope_dose": float(res["peak_slope_dose"]),
                }
            )

    return {
        "kind": "modality_dose_recruitment",
        "view_names": v_names,
        "dose_names": d_names,
        "log_transform": bool(log_transform),
        "target_neff": float(target_neff),
        "ci_level": float(ci_level),
        "curves": curves,
        "summary_rows": summary_rows,
    }

