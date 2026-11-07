# WP3 Report: Path Model Visualization and Brain Integration

Executor: Agent B (WP3 Visualization Worker), 2026-11-07.
Specifications: `docs/plans/2026-11-06_path_model.md` Section 9; `docs/plans/2026-11-06_path_model_handoff.md` Section 4.
Evidence Directory: `docs/audit/evidence/path_model/wp3/`.

## 1. What Was Built

Moved reference path-model visualization functions from `scripts/path_model_demo.py` into `src/pysimlr/plotting.py` and implemented key enhancements:
- `plot_path_model`: Circular DAG layout; edge width proportional to Pillai (0.6..6 pt) or per-component `|r_j|`; solid for `p_maxt < alpha`, dashed for hypothesised non-significant; `_WARN` for absent significant edges; Pillai prefix `V = ` with padded halos (`pad=0.32`); moderation indicator (`^` marker) when `p_holm < alpha`; status line on rejected graph fit (`pm["fit"]["rejected"]`); clean p-value formatting (`_format_p`) avoiding raw strings (e.g. `p < 0.001` or `p < 0.0002` at add-one floor).
- In `component` mode (`component=j`): signed colours (`_BLUE[6]` positive, `_RED[4]` negative); non-significant hypothesised edges rendered in dashed `_INK2` to strictly match legend.
- `plot_edge_matrix`: Per-component heatmap (rows: edges "a - b", cols: components) on diverging `_cmap("div")`; hatched cells for `q_bh >= alpha`; right-hand column with edge Pillai on `_cmap("seq")`. Supports `what="r"` and `what="beta"`.
- `plot_path_comparison`: Forest plot of held-out log-likelihood deltas vs top graph with paired subject bootstrap 95% CIs. Fraction-passing panel vertically dodges present-pass and absent-pass markers (`±0.12`) when equal to eliminate occlusion. Expanded `xlim` eliminates right-side annotation gutter spillover.
- `plot_edge_null`: Small multiples of Freedman-Lane permutation null histograms (`_BLUE[2]`) with vertical observed lines (`_RED[4]`); observed text offset cleanly to avoid crossing the line; explicit y-axis label ("Permutation count").
- `plot_edge_features`: Horizontal lollipop plots based on `pm["feature_contributions"]` (`structure_a * beta_ab`) in the style of `plot_top_features`; signed diverging stem/marker colors (`_BLUE[6]` positive, `_RED[4]` negative); clear endpoint view labels in titles and axis labels.
- `plot_edge_brain_map`: Connects `pm["feature_contributions"]` to `antsxviz` (DKT cortical surfaces / volumetric atlases via `_require_antsxviz()` and `brain_rows()`) using symmetric diverging colormaps.
- Exports & Integration: Registered all 6 functions in `src/pysimlr/__init__.py` (`__all__`, `_LAZY_ATTRS`), `src/pysimlr/plotting.py`, and `docs/usage_notes.md`. Updated `scripts/path_model_demo.py` to import from `pysimlr.plotting` and deleted redundant copies.
- Test Suite: Comprehensive test suite in `tests/test_plotting_sem.py` (10 tests: Figure return types, single-panel `ax` honoring, edge counts matching significant edges, component sign colors, moderation markers, graph-fit rejection status, marker dodging, edge features, and antsxviz integration).

## 2. Plan Section 9 Review Checklist (Figure by Figure)

### `plot_path_model` (`reference_path_model.png`, `reference_path_model_component0.png`)
- **Labels legible at 200 dpi**: Yes; view nodes (8.5 pt bold white), outcome square (8.5 pt bold white), consensus hollow circle; edge labels 8 pt ink with round surface halo (`pad=0.32`).
- **No overlapping edge labels**: Offset perpendicular to edge (`ox, oy = -dy/nrm*0.22, dx/nrm*0.22`) with halo cleanly covering underlying strokes.
- **Package palette only**: View categories (`VIEW_COLORS`), primary ink (`_INK`, `#0b0b0b`), secondary (`_INK2`, `#52514e`), warning (`_WARN`, `#fab219`), signed blue (`_BLUE[6]`) / red (`_RED[4]`).
- **Legend explains all encodings**: Dedicated handles for significant edge, hypothesised non-significant, not hypothesised significant (`_WARN`), component positive/negative r, and moderated edge (`^`).
- **Rows, null, and sweeps stated**: Subtitle explicitly states rows, n, conditioning, null, B sweeps, edge weights metric, and omnibus p.
- **Typography & Defaults**: Standard `_RC` Helvetica/sans-serif hierarchy; zero default Matplotlib blue.

### `plot_edge_matrix` (`reference_edge_matrix.png`)
- **Labels legible at 200 dpi**: Yes; edge pair row labels and component column labels ("c1", "c2", "c3") with tabular typography.
- **Package palette only**: Diverging `_cmap("div")` (blue-neutral-red) for correlations/slopes; sequential `_cmap("seq")` (blue ramp) for Pillai; muted ink (`_MUTED`) for FDR hatching.
- **Legend / Subtitle**: Subtitle declares FDR threshold (`BH q >= alpha`) and edge Pillai right-hand column.

### `plot_path_comparison` (`reference_comparison.png`)
- **Labels legible at 200 dpi**: Graph names with edge counts, delta log-likelihood ticks, replicate percentages (`best in 93% of replicates`).
- **No overlapping markers**: When present-pass and absent-pass fractions are equal (e.g. 1.0 vs 1.0 on chain), markers are dodged vertically by `±0.12`, rendering both distinctly visible.
- **No gutter spillover**: Right axis boundary expanded (`xlim = [min_ci - 0.06*span, max_ci + 0.45*span]`) so replicate text is contained cleanly within panel boundaries.
- **Package palette only**: Top model and present pass in `_BLUE[6]`; absent pass in `_RED[4]`; non-top models in `_INK2`.

### `plot_edge_null` (`reference_edge_null.png`)
- **Labels legible at 200 dpi**: Edge titles, observed Pillai and add-one p-value, x-axis label.
- **No text line collision**: Observed text label dynamically placed cleanly to the left of the observed vertical red line when `x_obs` is in the upper range.
- **Axes labeled**: Explicit "Permutation count" y-axis label and "Pillai under the null" x-axis label.
- **Package palette only**: Permutation histogram in `_BLUE[2]`, observed line in `_RED[4]`.

### `plot_edge_features` (`reference_edge_features.png`)
- **Labels legible at 200 dpi**: Feature names, component headers, contribution axis ticks.
- **Signed diverging colors**: Feature stems and dot markers colored by sign (`_BLUE[6]` positive, `_RED[4]` negative).
- **Endpoint labeling**: Panel titles and x-axis labels explicitly name source and target nodes (e.g. `view0 -> view1`).

### `plot_edge_brain_map`
- **antsxviz surface rendering**: Renders DKT cortical regions on fsaverage surface mesh or volumetric scenes; symmetric diverging colormap (`_cmap("div")`).

## 3. Verification Commands and Verbatim Results

- `python -m pytest -q tests/test_plotting_sem.py`
  `10 passed, 1 warning in 12.62s`
- `python -m antsxqc.lint src/pysimlr`
  `0 finding(s)`
- `python -m pytest -q tests/test_implementation_standards.py`
  `7 passed in 6.13s`
- `python -m pytest -q tests/test_plotting.py`
  `46 passed, 4 warnings in 33.54s`
- `python -m pytest -q -p no:cacheprovider`
  `2008 passed, 1 skipped, 2 xfailed, 937 warnings in 152.10s (0:02:32)`
- `python scripts/path_model_demo.py --out docs/audit/evidence/path_model/wp3`
  `wrote docs/audit/evidence/path_model/wp3/reference_report.html` (all 5 figures and HTML report generated).
