# TODO (pinned by the package author, 17 November 2026): a "data safety" diagnostic for views

Status: implemented (`src/pysimlr/safety.py`, `prepare_views(..., safety=...)`, `fit(..., safety=...)`, `tests/test_data_safety.py`). Standard methods only; use existing packages (numpy, scipy, scikit-learn). No new estimator.

## Why

On a real five-view study (n = 186, k = 1) one participant had a failed structural processing run. In one view his row
sat on average 3.5 standard deviations from everyone across all 100 features (the most extreme row), and his component
score was 4.75 SD. The fit bent that view's weights around him: correcting his data alone moved the component (r 0.88
between the two fits) and the study's main coefficient (0.34 to 0.31). Nothing in `prepare_views` or `fit` flagged it.
After the correction, a check of the same data still showed a heavy tail in another view (a few rows with robust z up to
13 and leverage 28 times the average). Separately, 13 of 186 leave-one-out refits landed on a neighbouring solution
(r 0.92 to 0.95 with the full fit), which is a fit-sensitivity signal, not an outlier.

## What to add

`pysimlr.data_safety(views, *, fit_rows=None, groups=None, n_pc=5, robust_z=5.0, leverage_ratio=3.0,
md_quantile=0.999, influence=False, method="simlr", k=1, **fit_params) -> dict`, to be called on the PREPARED views
(after `prepare_views`), and optionally called by `prepare_views` / `fit` with a warning.

Per view, on the rows that have it (aggregates and row indices; the caller decides what to print):

1. Row scale: mean absolute z across features, with a robust z over rows (median and 1.4826 x MAD). Flag above
   `robust_z`. This is the check that would have caught the case above.
2. Leverage: hat values on the `n_pc` leading principal components (average `n_pc / n`). Flag above `leverage_ratio`
   times the average.
3. Multivariate distance: robust squared Mahalanobis distance on those components
   (`sklearn.covariance.MinCovDet`), flagged against the chi-square `md_quantile`.
4. Column checks: constant or near-constant columns, columns with a single extreme value (maximum robust z per
   column), share missing, duplicated columns.
5. Group structure (if `groups` given): counts of flags by group, and whether flagged rows concentrate in one group
   (a cluster of flags in one group usually means an acquisition difference, not outliers).
6. Optional influence (`influence=True`, costs n fits): leave-one-row-out refit; report each row's effect on the
   consensus score (correlation with the full fit) and the number of refits that land on a different solution.
   Reuse `loading_inference` / `bagged_fit` machinery where possible.

Return: `rules`, per-view arrays (`mean_abs_z`, `robust_z`, `leverage_ratio`, `md2`) and boolean flags, `flagged_rows`
(indices, by rule and by view), `n_flagged_by_group`, column findings, `warnings` (plain sentences), and, with
`influence`, `loo_score_correlation` and `n_other_solution`.

Behaviour: never drop or change data. `prepare_views(..., safety="warn" | "raise" | "off")` (default "warn") emits one
`RuntimeWarning` per view with a gross row outlier and records the report under `result["data_safety"]`.

## Tests (synthetic)

- A planted gross row outlier in one view is flagged by rule 1 in that view only; clean data gives no flags.
- A planted high-leverage, small-scale row is flagged by rule 2 or 3 and not by rule 1.
- Rows without a view are ignored for that view; group counts add up; determinism.
- `influence=True` on a two-solution toy reports the rows whose removal changes the solution.

## Reference

A study-level version of rules 1 to 3 exists in downstream study code
and can be generalized.
