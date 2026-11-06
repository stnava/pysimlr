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

scipy is imported inside the functions that need it so that ``import
pysimlr`` (which imports ``screening``, which imports `bh_fdr`) stays free
of scipy; ``tests/test_package_import.py`` enforces that.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

__all__ = [
    "bh_fdr",
    "freedman_lane_partial_p",
    "holm",
    "icc_3",
    "jzs_bf10_correlation",
    "jzs_bf10_correlation_grid",
    "jzs_bf10_one_sample_t",
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
    out = np.array([_jzs_bf10_correlation_scalar(float(rv), n, float(rscale)) for rv in flat])
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
    df = max(1, n_samples - 2)
    t_obs = r_obs * np.sqrt(df / np.maximum(1e-12, 1.0 - r_obs**2))
    p_param = np.where(valid, 2.0 * _sps.t.sf(np.abs(t_obs), df=df), 1.0)

    rng = np.random.default_rng(int(seed))
    if paired:
        r_null = np.empty((n_features, n_perms), dtype=np.float64)
        for b in range(n_perms):
            perm_idx = rng.permutation(n_samples)
            r_null[:, b] = np.abs(np.sum(x_z * y_z[perm_idx, :], axis=0) * inv_dof)
    else:
        y_perms = np.empty((n_samples, n_perms), dtype=np.float64)
        for b in range(n_perms):
            y_perms[:, b] = y_z[rng.permutation(n_samples)]
        r_null = np.abs((x_z.T @ y_perms) * inv_dof)
    abs_obs = np.abs(r_obs)[:, None]
    n_exceed = np.sum(r_null >= abs_obs, axis=1)
    p_perm = np.where(valid, (1.0 + n_exceed) / (1.0 + float(n_perms)), 1.0)
    q_fdr = bh_fdr(p_perm)
    max_r_null = np.max(r_null, axis=0)
    n_exceed_fwer = np.sum(max_r_null[None, :] >= abs_obs, axis=1)
    p_fwer = np.where(valid, (1.0 + n_exceed_fwer) / (1.0 + float(n_perms)), 1.0)

    if n_boot > 0:
        boot_idx = rng.integers(0, n_samples, size=(n_boot, n_samples))
        r_boot = np.empty((n_features, n_boot), dtype=np.float64)
        for b in range(n_boot):
            idx_b = boot_idx[b]
            xb = X[idx_b]
            xb_c = xb - np.mean(xb, axis=0, keepdims=True)
            if paired:
                yb = y[idx_b]
                yb_c = yb - np.mean(yb, axis=0, keepdims=True)
                denom = np.sqrt(np.sum(xb_c**2, axis=0) * np.sum(yb_c**2, axis=0))
                num = np.sum(xb_c * yb_c, axis=0)
            else:
                yb = y[idx_b]
                yb_c = yb - np.mean(yb)
                denom = np.sqrt(np.sum(xb_c**2, axis=0) * np.sum(yb_c**2))
                num = xb_c.T @ yb_c
            r_boot[:, b] = np.where(denom > 1e-12, num / np.where(denom > 1e-12, denom, 1.0), 0.0)
        ci95_low = np.where(valid, np.percentile(r_boot, 2.5, axis=1), 0.0)
        ci95_high = np.where(valid, np.percentile(r_boot, 97.5, axis=1), 0.0)
        pos_frac = np.mean(r_boot > 0.0, axis=1)
        neg_frac = np.mean(r_boot < 0.0, axis=1)
        prob_dir = np.where(valid, np.maximum(pos_frac, neg_frac), 0.5)
    else:
        ci95_low = np.full(n_features, np.nan)
        ci95_high = np.full(n_features, np.nan)
        prob_dir = np.full(n_features, np.nan)

    bf10 = np.ones(n_features, dtype=np.float64)
    if np.any(valid):
        bf10[valid] = jzs_bf10_correlation(r_obs[valid], n_samples)

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
