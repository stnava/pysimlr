# pysimlr: score-level path model (edge-wise inference, omnibus tests, topology comparison, prediction)

Repo: `/Users/stnava/data/repos/pysimlr` (v0.22.2, `main`; untracked at hand-off: `src/pysimlr/sem.py`, `scripts/path_model_demo.py`, `docs/audit/evidence/path_model/`, this plan).
Plan author: Claude Fable 5.1 (supervisor). Executor: the implementing agent. Owner: stnava.
Date: 2026-11-06. Target release: v0.23.0 (minor: new public API, one behaviour change in
`permutation_test`, one function retired).

This plan is self-contained. Every decision an executor would otherwise have to ask about is
written down as a rule. Do not ask the owner; if a rule is missing, apply the closest rule here,
record the choice in the work-package report, and continue.

## 0. Working protocol (binding)

1. Read `docs/usage_notes.md`, `docs/api_contract.md`, `docs/tracking.md` (D3, D6, D30, D37
   entries) and the module docstrings of `relevance.py`, `confounds.py`, `loadings.py` before
   writing code. The style, nulls and vocabulary of this plan come from those files.
2. Evidence first. Every number in a report is produced by a script under `scripts/` or a test
   under `tests/`, and the command that produced it is written next to it. No number from memory.
3. Tests define done. A work package is complete when its listed tests pass under
   `pytest -n auto` (fast suite) and, where marked, `pytest -m "" -k <name>` (slow suite), and
   `python -m antsxqc.lint src/pysimlr` reports 0 findings.
4. No silent fallbacks. No `except Exception: pass`, no substituted values. A quantity that
   cannot be computed is `None` with a recorded reason, or raises `ValueError` with the reason.
5. No global RNG seeding in library code; `np.random.default_rng(random_state)` only.
6. No mutable default arguments. No new broad exception handlers. (Enforced by
   `tests/test_implementation_standards.py`; run it.)
7. Do not commit, tag or push. Leave the tree with the work staged by file in the WP report
   (`git add -N` is fine). The supervisor commits after review. Do not touch `paper/`.
8. An implausible result is a defect to find, not a finding to explain. If a p-value sits at the
   floor for every edge on held-out rows, if a null edge rejects at 30%, or if a held-out
   statistic exceeds its training value, stop that WP and write the investigation in the report.
9. Every figure is rendered to PNG, opened, and reviewed against the checklist in section 9
   before it is called done. Default-looking matplotlib output is rejected.
10. Report format after each WP (one Markdown file per WP under
    `docs/audit/evidence/path_model/WP<n>_report.md`): what was built (files, functions), the
    exact commands run, their verbatim summary lines (pass counts, lint count), every deviation
    from this plan with the reason, open questions for the supervisor. Keep it under 150 lines.

## 1. Context and what exists

Every `pysimlr.fit` result, for every family (linear `simlr`, deep `lend`/`ned`/`nedpp`, flow
`flow`/`flow_v`, joint baselines `pca`/`nsa_pipeline`/`gcca`), gives through `pysimlr.predict`
the same objects on any rows: per-view scores (`latents`, one `(n, k)` block per view), the
consensus `u`, reconstructions and R². Scores are in each view's preprocessed space, confounds
are already removed by the sink (`confounds.py`), and the same affine map applies to training and
new rows. The loading rotation `result["loading_rotation"]` identifies the components, and
`loadings._oob_metrics` shows the canonical way to put scores into that frame and sign-align them
to `u`.

That is the measurement model of a structural equation model (SEM). Its inference exists:
`loading_inference` (CIs, stability selection), `feature_relevance` (per-feature permutation
cutoff with fixed-model and refit nulls), `component_attribution`.

What is missing is the structural model: a graph over views (plus covariates, group factors and
outcomes) whose edges are estimated, tested, compared and used for prediction. Today:

| need | what exists | gap |
|---|---|---|
| omnibus "any shared structure" | `paths.permutation_test` (`paths.py:98`): mean view-to-consensus adjusted RV, every view permuted, add-one p | calls `simlr` directly, so deep/flow/joint methods have no test; one scalar, no edges |
| per-view coupling test | `screening.screen_wide_views` Westfall-Young `p_view_mean`/`p_view_max` | pre-fit screen of features, not a test of a fitted model; per node, not per edge |
| per-edge statistic | `utils.heldout_coupling` computes mean over pairs of per-component abs correlation | averaged away; no null; raises for joint-basis fits |
| topology comparison | `structural.fit_structural_models`: fits each `path_graph`, reports mean of the last five training losses | in-sample, graph-conditional loss (denser graphs have more terms), deep/flow only, no held-out, no uncertainty |
| path diagram | `diagrams.plot_path_model`: networkx spring layout of the adjacency list | draws the hypothesis, not the fitted model; networkx optional dependency |
| refit null plumbing | `relevance._null_refit_task`, `_run`, `confounds.permute_view_confounds` | reusable as is |
| out-of-bag hook | `loadings._oob_metrics` evaluated on every `loading_inference` subsample refit | reusable; its score extraction is duplicated in `heldout_coupling` |

Decisions taken in planning (supervisor, 2026-11-06), following the owner's standing rules
(backward compatibility is not a constraint, D3; duplicate code is eliminated, D6; new options
ship for all families as opt-in, not pilot-gated; pilot on real data before claiming a result):

- D-1 One new module `src/pysimlr/sem.py` holds the structural model. Public names:
  `pysimlr.path_model`, `pysimlr.compare_path_models`, `pysimlr.predict_path`,
  `pysimlr.view_scores`. Plotting in `plotting.py`: `plot_path_model`, `plot_edge_matrix`,
  `plot_path_comparison`, `plot_edge_null`.
- D-2 `paths.permutation_test` is reimplemented on top of `path_model`'s omnibus and routed
  through `pysimlr.fit` (signature gains `method="simlr"`); same return keys plus the new table.
- D-3 `structural.fit_structural_models` is retired. `compare_path_models` replaces it;
  `create_path_graph` stays. `diagrams.plot_path_model` is removed in favour of the plotting one.
  `tests/test_structural_models.py` is rewritten for the new function.
- D-4 Score extraction becomes one helper `sem.view_scores`, used by `_oob_metrics`,
  `heldout_coupling` and the new code. Joint-basis fits get per-view scores by projecting each
  view's block of `v` (as `_oob_metrics` does today) and the result records
  `scores_source="joint_block"`; nothing raises for them any more.
- D-5 Primary edge statistic is rotation-invariant (Pillai trace of partial canonical
  correlations). Per-component statistics are secondary and conditioned on the reference
  rotation, exactly as `loading_inference` conditions its per-component claims.
- D-6 Training-row inference uses the refit null by default; held-out rows use the fixed-model
  null. The function refuses to report fixed-model p-values on training rows for wide views
  (`result["wide_views"]` non-empty) unless `allow_train_fixed=True`; it warns otherwise.
- D-7 Graphs are undirected unless the caller passes `directed=True`. `pa(x)` is the neighbour
  set for undirected graphs and the parent set for directed ones.
- D-8 Version bump is minor (0.23.0), done by the supervisor with the `release` skill.

## 1b. Reference implementation (supervisor, 2026-11-06)

`src/pysimlr/sem.py` and `scripts/path_model_demo.py` exist as a REFERENCE for the numerical core:
`view_scores`, edge statistics, the Freedman-Lane joint sweep, multiplicity, omnibus, graph fit,
`compare_path_models` (4.1 only) and `predict_path`. The demo writes
`docs/audit/evidence/path_model/reference_report.html` and four reviewed PNGs. Rules for the executor:

- Keep the semantics and the function signatures. Refactor freely (chunking, helpers, names of
  private functions). Every change must keep the smoke results reproducible:
  `python scripts/path_model_demo.py` on the planted chain gives present edges at the add-one floor,
  absent edges `p_perm > 0.05` given parents, `rotation invariance` of `pillai` to 1e-10, the
  Freedman-Lane adjusted p-values on the covariate-leak case above 0.2, and the chain graph
  passing its fit test.
- The four plot functions in the demo script are reference drawings: move them into `plotting.py`
  under the plan's names, keep their encodings (solid/dashed, warning colour for an unhypothesised
  significant edge, hatched non-significant cells, paired deltas with CI and "best in x%").
- Not in the reference, to be added per the plan: refit null (3.2), bootstrap CIs and
  `ci_refit_*` (3.6), group moderation (section 2), training-topology comparison (4.2), feature
  contributions (5), `permutation_test` re-routing, removals (D-3), `__init__` exports, docs.

Findings from the reference run that change the spec:

- F1 **Parsimony rule for `best`.** On the planted chain with the sink, held-out log-likelihood
  put the complete graph ahead of the true chain by 4.3 (CI -14.5 to 5.6, best in 80% of
  replicates) while BIC preferred the chain (7196 vs 7338) and the chain passed its fit test.
  Held-out likelihood cannot penalise a weakly supported extra edge. Rule: `best` is the graph
  with the fewest edges whose paired delta CI against the top held-out log-likelihood includes 0;
  report the raw top as `best_by_loglik` beside it. Test 10 asserts `best` is the chain.
- F2 **Absent edges are attenuated, not removed, by conditioning on estimated scores.** The
  chain's 0-2 edge given view1 had `p_perm` 0.027 and 0.072 on two draws: scores are noisy
  estimates of the latent, so partialling out the middle node leaves residual association
  (measurement error in the conditioning set). The graph-fit test therefore errs toward
  rejecting absent edges, more so for small k and noisy views. Document this in the usage note
  ("what the test cannot say"); set test 4's tolerance to 8 of 10 seeds at n = 500 and
  `noise = 0.3`; and report `absent_pass` with its unadjusted p's so the reader sees borderline edges.
- F3 **Confound leakage must go through the sink.** With covariates leaking into every view and
  no sink, `simlr` found the shared leak as its components (view2-outcome Pillai 0.14, outcome
  R² 0.14); with `fit(..., confounds=C)` the same run gives 0.65 and 0.78. The ADNI pilot passes
  its confounds to `fit` (as it already does) and again as `covariates=` to `path_model`.
- F4 Undirected node regressions count each edge in two regressions; BIC's parameter count
  double-counts. Rule: for undirected graphs, `n_params` counts each edge once (`k * k`) plus one
  nuisance block per node; the predictive log-likelihood is unchanged.
- F5 Per-component `r` for an edge into a one-column outcome correlates each component of the
  view with that column (reference `_comparable` / `_broadcast_cols`); edges between two
  multi-column nodes of different k report `pillai` only.

## 2. The model

Notation. M views, k components, n rows. After `view_scores`, `Z_m` is `(n, k)`: the view's
scores rotated by `loading_rotation`, sign-aligned to `u` on the rows the model was fitted on,
then column-standardised on the evaluation rows. `N` is the nuisance design: an intercept, the
covariates `C` (`(n, q)`, standardised), and the group dummies `G` (one-hot minus one level).
`Y` is an optional outcome block `(n, r)`, standardised.

Nodes: the M views; optionally the outcome `Y` (node name `"outcome"`). The consensus `u` is
not a node (`include_consensus=False`; `True` adds it as a node named `"consensus"` for
diagnostics only, never in graph comparison).

Edges `(a, b)`: present edges are the graph's; absent edges are the complement over node pairs.
Each edge is tested; present edges are expected to pass, absent edges are expected to fail.

Conditioning set for an edge, by `conditioning`:
- `"pairwise"` (default): `S = N`.
- `"parents"`: `S = N ∪ {Z_c : c ∈ (pa(a) ∪ pa(b)) \ {a, b}}`. This is the graph-fit test for
  absent edges and the unique-contribution test for present ones.

Residualise on S by least squares (ridge with `lambda = 1e-8 * trace/p`, recorded): `Ra = Z_a -
P_S Z_a`, `Rb = Z_b - P_S Z_b`.

Statistics per edge:
- `pillai`: with `rho_i` the canonical correlations of `(Ra, Rb)`, `pillai = mean(rho_i^2)` in
  `[0, 1]`. Rotation-invariant. Primary.
- `rv`: `utils.adjusted_rvcoef(Ra, Rb)`. Reported, not tested.
- per component j: `r_j = corr(Ra[:, j], Rb[:, j])` and `beta_j` (slope of `Rb[:, j]` on
  `Ra[:, j]`; with standardised columns `beta_j = r_j` up to residual scale; report both).
- `moderation` (only when `G` is given): the F-statistic of adding `Ra ⊙ G` to the regression of
  `Rb` on `[Ra, G]`, one per component and the sum; tests whether the edge differs across groups.

For the outcome node, `Z_outcome = Y` standardised and the same statistics apply (edges from
views to the outcome are the ANCOVA: outcome on view scores given covariates and group).

## 3. Inference

### 3.1 Fixed-model null (Freedman and Lane 1983)

For every permutation sweep s = 1..B, draw one permutation per node (`rng.permutation(n)`),
shared by all of that node's columns so within-view dependence is preserved. For an edge
`(a, b)` under conditioning S: `Rb_perm = P_S Z_b + pi_b(Rb)` (fitted part kept, residual
permuted), recompute the statistics against `Ra`. All edges are recomputed in the same sweep,
so the sweep's maximum over edges is a draw from the joint null and max-T is valid.

Implementation rule: vectorise over sweeps in chunks (`_CHUNK = 500` sweeps): residuals are
`(n, k)`, so `pillai` for a chunk is a batch of `k x k` SVDs; use `torch.linalg.svdvals` on
`(chunk, k, k)`. Default `n_permutations = 10_000` on held-out rows. The cost must be under 10 s
for M = 4, k = 5, n = 300, B = 10 000 on one core (test asserts under 30 s).

p-values are add-one: `(1 + #{null >= obs}) / (1 + B)` (floor `1/(B+1)`), as everywhere in
the package. Two-sided for signed per-component `r_j` (use `|r_j|`), one-sided for `pillai`,
`moderation`.

### 3.2 Refit null

As `relevance._null_refit_task`: for edge `(a, b)` permute node b's training rows together
with its confound rows (`confounds.permute_view_confounds`), refit with `pysimlr.fit` using the
result's recorded `parameters["used"]`, recompute scores on the same rows and the edge statistic.
Default `n_refits = 200`, parallel through `relevance._run`. A refit null is per edge (one
permuted node), so max-T across edges is not available from it; report `p_refit` per edge
with Holm across edges. Group permutations by node b so one refit serves every edge into b.

### 3.3 Multiplicity

Per family of hypotheses, report `p_perm` (unadjusted), `p_maxt` (single-step max-T over the
family from the joint sweeps), `p_holm`, `q_bh`. Families: `edges` (pillai over all tested
edges), `edge_components` (|r_j| over edges x components), `moderation`, `outcome_edges`
(edges into the outcome node, tested as their own family because that is the ANCOVA question).

### 3.4 Omnibus

From the same sweeps: `omnibus["max_pillai"]` (p of max over tested edges) and
`omnibus["mean_pillai"]`. `permutation_test` returns `mean_pillai` as `observed_similarity`
and its null as `null_similarities`, keeping its current keys.

### 3.5 Graph fit

`fit["present_pass"]`: fraction of present edges with `p_holm < alpha` under
`conditioning="parents"`. `fit["absent_pass"]`: fraction of absent edges with `p_perm >= alpha`
under `conditioning="parents"` (no correction: the claim is independence, correction would make
it easier to pass). `fit["max_absent_pillai"]` with its max-T p over absent edges: the
permutation analogue of the SEM chi-square test; the graph is rejected when it is below alpha.

### 3.6 Uncertainty of effect sizes

`bootstrap = 2000` subject bootstrap replicates on the evaluation rows (measurement fixed) give
percentile CIs for `pillai`, `r_j`, `beta_j`. When `loading_inference` results are passed
(`inference=inf`), each subsample refit's out-of-bag rows are scored with the same code and the
spread across refits is reported as `ci_refit_*` (no extra fits: hook `_oob_metrics`).

## 4. Model comparison

### 4.1 Structural comparison (measurement fixed)

`compare_path_models(result, views_train, views_test, graphs={name: graph}, covariates=,
groups=, outcome=, criterion=("heldout_loglik", "bic"), n_bootstrap=2000)`.

For each graph and each node b with parents `pa(b)`: regress `Z_b` (training rows) on
`[N, Z_pa(b)]` with the same tiny ridge, residual variance per column from training; evaluate the
Gaussian log-likelihood of the held-out `Z_b` under those parameters. `heldout_loglik` is the
sum over nodes and columns. `bic` is in-sample, with parameter count `k * (|pa(b)| * k + cols(N))`
per node. A node with no parents is modelled by its nuisance regression only, so every graph
is scored on the same quantity. Paired subject bootstrap over held-out rows of the difference
in `heldout_loglik` between each graph and the best one gives CI and the fraction of replicates
in which the best graph stays best. Also report each graph's section 3.5 fit test.

Output: `comparison` table (graph, n_edges, heldout_loglik, bic, present_pass, absent_pass,
max_absent_p, delta_vs_best, ci_low, ci_high, best_fraction), `best` (by `heldout_loglik`),
`edges` per graph (the `path_model` tables).

### 4.2 Training-topology comparison (refits)

Same function with `method=` and `k=` instead of `result`: for each graph, `fit(method, train,
k, topology="graph", path_graph=graph, **params)` (a graph with no edges uses `topology="star"`
and is reported as such), then: (i) the structural comparison of 4.1 on that fit's scores under
its own graph and under the complete graph, (ii) graph-agnostic held-out criteria:
`oos_coupling` (mean held-out leave-one-view-out coupling, `_oob_metrics`), `oos_r2` (adjusted
held-out reconstruction R²), and, with an outcome, held-out `outcome_r2` from a ridge on the
consensus. `n_splits` repeated 70/30 splits (default 5, seeds `random_state + s`), paired over
splits. The training loss is not reported; the plan's reason is in section 1.

## 5. Prediction

`predict_path(pm, views, outcome=True)` where `pm` is a `path_model` result and `views` is a
list with `None` for missing views, or a dict by name. Steps: `view_scores` for the present
views; for each missing node in an order that uses only present or already-predicted neighbours
(iterate to a fixed point; a node with no available neighbour gets the nuisance-only prediction
and is flagged in `predicted_from`), predict its scores by the ridge from `[N, Z_pa]` fitted on
the training scores (stored in `pm["edge_models"]`); the outcome by its own node regression.
Returns `scores` (per node), `outcome_pred`, `predicted_from` (node -> list of nodes used), and
`r2_per_node` when the true views are also passed (`views_true=`). Rule: with all views present,
`scores` must equal `view_scores(result, views)` exactly (test).

Feature-level read-out (`pm["feature_contributions"]`): for each present edge `(a, b)` and
component j, `structure_a[:, j] * beta_abj` as a `(p_a,)` vector; exported as a long table
(edge, component, feature, contribution) and consumed by `plot_top_features`-style bars
(`plot_edge_features`, WP5, optional if time is short; say so in the report).

## 6. API

```python
pm = pysimlr.path_model(
    result, views, *, rows="heldout",            # the rows `views` are: "heldout" or "train"
    graph=None,                                    # None = complete graph; else {node: [neighbours]} or edge list
    directed=False, view_names=None,
    covariates=None, groups=None, outcome=None,    # arrays aligned with `views` rows
    conditioning="pairwise",                       # or "parents"
    include_consensus=False,
    null="fixed",                                  # "fixed" | "refit" | "both"; rows="train" defaults to "refit"
    n_permutations=10_000, n_refits=200, n_bootstrap=2000,
    train_views=None, train_covariates=None, train_groups=None,  # needed for "refit" and for predict_path
    inference=None,                                # a loading_inference result for ci_refit_*
    alpha=0.05, random_state=0, n_jobs=1, allow_train_fixed=False,
)
```

`pm` keys: `nodes`, `graph`, `directed`, `conditioning`, `rows`, `n`, `edges` (DataFrame:
a, b, present, pillai, rv, p_perm, p_maxt, p_holm, q_bh, p_refit, ci_low, ci_high, ci_refit_low,
ci_refit_high), `edge_components` (DataFrame: a, b, component, r, beta, p_perm, p_maxt, q_bh,
ci_low, ci_high), `moderation` (DataFrame or None), `omnibus` (dict), `fit` (dict, section 3.5),
`edge_models` (for `predict_path`), `feature_contributions`, `null` (per edge: the null
`pillai` draws, kept for `plot_edge_null`; cap at 2000 stored draws per edge), `scores_source`,
`parameters` (everything requested), `warnings` (list of strings also emitted as warnings).

`pysimlr.view_scores(result, views, *, train_views=None) -> dict(scores: list[np.ndarray],
consensus: np.ndarray, signs, source)`.

`pysimlr.compare_path_models(...)` as in section 4; `pysimlr.predict_path(...)` as in section 5.

`permutation_test(data_matrices, k, n_permutations=100, method="simlr", verbose=False,
**kwargs)`: fits with `pysimlr.fit`, runs `path_model(rows="train", null="refit", ...)` with
`n_refits=n_permutations` on the complete graph, returns the old keys from `omnibus` plus
`edges`.

Register everything in `__init__.py` (`__all__` and the lazy-module map), `api_contract.md`
(a "Structural model" section), and a `usage_notes.md` section "Edges between views: the
score-level path model" after "New samples". Deprecate nothing: removed names are removed.

## 7. Synthetic generator and tests

`benchmarks/synthetic_cases.build_path_graph_case(graph, n_samples, k, p_per_view, noise,
covariate_effect=0.0, group_effect=0.0, random_state)`: latent chain generation. For an
undirected graph with a spanning tree, draw the root's latent `L_root ~ N(0, I_k)`; each
neighbour `L_b = alpha * L_a + sqrt(1 - alpha^2) * E_b` with `alpha = 0.7`, so a chain a-b-c has
`L_a ⟂ L_c | L_b` exactly. Views `X_m = L_m B_m + noise * E`, `B_m` from
`_sparse_signed_basis`. `covariate_effect` adds `C beta` to every view (so unadjusted edges are
inflated and Freedman-Lane must remove it). `group_effect` scales `alpha` by group on one named
edge (so `moderation` has a planted positive). Returns `views, latents, graph, C, G, outcome`
(outcome = `L_last[:, 0] + noise`).

Tests (`tests/test_sem.py`, fast unless marked):

1. `test_view_scores_matches_oob_metrics_frame`: the scores `view_scores` returns equal what
   `_oob_metrics` builds (refactor keeps bitwise agreement).
2. `test_pillai_is_rotation_invariant`: rotate `Z_b` by a random orthogonal `k x k`; `pillai`
   unchanged to 1e-10; `r_j` changes.
3. `test_freedman_lane_removes_covariate_inflation`: `covariate_effect=1.0`, graph empty;
   unadjusted pairwise `pillai` is large; with `covariates=C` every edge `p_perm > 0.05` in at
   least 9 of 10 seeds at n = 200.
4. `test_absent_edges_pass_given_parents_and_fail_pairwise`: chain 0-1-2; edge (0, 2):
   `conditioning="pairwise"` rejects, `"parents"` does not, 9 of 10 seeds.
5. `test_null_calibration_fixed_model` (slow): empty graph, 200 datasets, B = 500, n = 120, M =
   3, k = 2; fraction of `p_perm < 0.05` per edge within `[0.03, 0.07]`, and `p_maxt` family
   rate within `[0.03, 0.07]`.
6. `test_null_calibration_refit` (slow): 100 datasets, `n_refits = 100`, `simlr` with
   `iterations = 5`; rate within `[0.02, 0.09]`.
7. `test_every_family_runs`: for method in `("simlr", "lend", "ned", "flow_v", "pca")` with
   tiny budgets: `path_model` returns the full key set, every p in `(0, 1]`, `scores_source`
   is `"joint_block"` for `pca` only.
8. `test_moderation_detects_planted_group_effect`: `group_effect=0.5` on edge (0, 1), n = 300;
   `moderation` p < 0.05 on that edge and > 0.05 on an unmoderated edge, 8 of 10 seeds.
9. `test_outcome_edges_are_the_ancova`: with `outcome`, the outcome node's edge r_j agrees with
   `statsmodels`-free OLS partial correlation computed inline to 1e-8.
10. `test_compare_path_models_selects_planted_graph`: chain vs star vs complete vs empty, n =
    500; best by `heldout_loglik` is the chain in at least 8 of 10 seeds, and `best_fraction`
    for the chain exceeds 0.8 in those seeds.
11. `test_compare_training_topologies_runs_and_is_paired`: `method="simlr"`, two graphs,
    `n_splits=2`; output shapes; the two graphs are evaluated on identical held-out rows per split.
12. `test_predict_path_full_views_equals_view_scores`; `test_predict_path_missing_view_is_
    predicted_from_neighbours` (r2 of the predicted missing view's scores against its true
    scores exceeds 0.5 at n = 500, noise 0.3).
13. `test_permutation_test_routes_through_fit`: `method="lend"` works; keys unchanged;
    `p_value` has the add-one floor (keep `test_paths_permutation_test_never_returns_zero`).
14. `test_refit_null_permutes_confounds_with_the_view`: monkeypatch `fit` to record its
    `confounds` argument; the permuted node's confound rows are permuted with it.
15. `test_wide_view_training_fixed_null_is_refused`: a `p >= n` view, `rows="train"`,
    `null="fixed"` warns and reports `p_perm` as `None` unless `allow_train_fixed=True`.
16. `test_plotting_sem.py`: each of the four plot functions returns a `Figure`, honours `ax`
    where single-panel, and encodes significance (dashed vs solid edge count equals the table).
17. `test_fixed_null_cost`: M = 4, k = 5, n = 300, B = 10 000 under 30 s.

Add the slow tests to `conftest._LPT_PRIORITY`.

## 8. Pilot on ADNI-MCI (WP4, before the docs are written)

Script `scripts/adni_path_model_pilot.py`, data as `scripts/benchmark_confound_sink.load_adni`
(`data/adni/adni_mci_cognitive_merged.csv`: views `T1Hier_vol`, `T1Hier_thk`, `T1w_mean`;
outcome CogPC1 = first PC of the cognitive battery fitted on training rows; k = 3). Do not use
`scripts/adni_loader.get_adni_case` / `get_adni_cdrsb_case` (T1/DTI/rsfMRI) or any PPMI case:
those caches were processed incorrectly and are off limits (owner rule). Confounds
age, education, head size through the sink; `groups=sex` (a column used as the group factor
must not be among the confounds, rule). Five 70/30 splits (seeds 0-4), method `simlr` with
the package defaults, then the same with `lend`.

Outputs to `docs/audit/evidence/path_model/`: `adni_edges_<method>_seed<s>.csv`,
`adni_comparison.csv` (graphs: complete, star on `T1Hier_thk`, chain vol-thk-mean, empty, each
plus the outcome node), and the four figures for seed 0, PNG, reviewed.

Rules that decide whether the pilot passed (write the outcome in the WP4 report):
- P1 The `T1Hier_vol`-`T1Hier_thk` edge has held-out `p_maxt < 0.05` in at least 4 of 5 splits.
- P2 The outcome node has at least one view with `p_holm < 0.05` in the `outcome_edges` family
  in at least 4 of 5 splits.
- P3 The fixed-model held-out `pillai` is below the training `pillai` for every edge (if not,
  rule 8 of section 0 applies).
- P4 The complete graph is not the best by `heldout_loglik` in every split (if it is, report
  the paired deltas; this is not a failure, it is the result).
If P1 or P2 fails, the pilot has found something to investigate (score extraction, sign
alignment, the sink) before any documentation is written.

## 9. Visualization (WP3)

All in `plotting.py`, inside `theme()`, using `_view_color`, `_cmap`, `_matrix`, `_header`,
`_clean`, `_dots`; matplotlib only, no networkx. Each function takes `ax=None` where
single-panel and returns the `Figure`.

- `plot_path_model(pm, component=None, alpha=0.05, show="pillai")`: nodes on a circle (views in
  view colours, outcome as a square in `_INK`, consensus hollow if included), edge width
  proportional to `pillai` (range 0.6 to 6 pt), solid when `p_maxt < alpha`, dashed otherwise,
  absent-but-significant edges drawn in `_WARN` (a graph-fit violation is the most important
  thing in the figure), label `pillai` and `p_maxt` at mid-edge with a white halo, a marker on
  moderated edges, subtitle with `rows`, `n`, `conditioning`, `null`, B. `component=j` draws
  `|r_j|` with sign colour (`_BLUE[6]` positive, `_RED[4]` negative). A `status` line reports
  the omnibus p.
- `plot_edge_matrix(pm, what="r")`: edges (rows, labelled "a - b") x components (columns)
  heatmap on the diverging map, hatched cells where `q_bh >= alpha`, a right-hand column with
  the edge's `pillai`; one panel per null when both were run.
- `plot_path_comparison(cmp)`: forest plot of `heldout_loglik` deltas vs the best graph with
  bootstrap CIs, graphs sorted, `best_fraction` as text, a second panel with `present_pass` /
  `absent_pass` as dots.
- `plot_edge_null(pm, edges=None)`: small multiples, one per edge, null histogram (`_BLUE[2]`)
  with the observed statistic as a vertical line and the p-value as text.

Review checklist for every figure (the executor writes the answers in the WP report): every
label legible at 200 dpi; no overlapping edge labels (offset along the edge if needed); colours
from the package palette only; the legend explains solid/dashed and the warning colour; the
figure says what rows and null it is built from; nothing in the default matplotlib blue or in
a default font.

## 10. Work packages, in order

Each WP ends with its report (section 0, item 10) and a `pytest -n auto` run.

- **WP1 scores and statistics** (`sem.py`, starting from the reference in 1b): `view_scores` and its use in `_oob_metrics` and
  `heldout_coupling`; `_edge_stats` (pillai, rv, r_j, beta_j, moderation); the fixed-model null
  with chunked vectorised sweeps; multiplicity; omnibus; graph fit; bootstrap CIs; `path_model`
  on held-out rows. Generator `build_path_graph_case`. Tests 1-5, 7-9, 15, 17.
- **WP2 refit null, comparison, prediction**: refit null via `_null_refit_task`-style tasks;
  `compare_path_models` (4.1 and 4.2); `predict_path`; `permutation_test` re-routed;
  `fit_structural_models` and `diagrams.plot_path_model` removed; `__init__` exports. Tests 6,
  10-14; `tests/test_structural_models.py` rewritten.
- **WP3 visualization**: the four plot functions; `tests/test_plotting_sem.py`; rendered PNGs
  on the synthetic chain case in `docs/audit/evidence/path_model/synthetic_*.png`, reviewed.
- **WP4 pilot**: section 8. Report the rule outcomes P1-P4 and the figures.
- **WP5 documentation**: `usage_notes.md` section (what the test can and cannot say, in the
  register of the `feature_relevance` docstring), `api_contract.md`, module docstring,
  `docs/tracking.md` entry "D40 (score-level path model ...)" in the house format, optional
  `plot_edge_features`. The supervisor then releases v0.23.0.

Estimated size: `sem.py` 900-1100 lines (reference is 520), plotting 350 lines, tests 600 lines, generator 80
lines, script 150 lines.

## 11. What this plan does not do

- No changes to the consensus optimiser or to how `topology="graph"` trains (the structural
  model is post-fit). Training-topology selection is done by refitting, section 4.2.
- No indirect (mediation) effects. They are a natural next step on `edge_models` for directed
  graphs and are listed as an open item in the D40 tracking entry.
- No paper changes, no NHANES run (listed as follow-up after the ADNI pilot).
- No claim of exact family-wise control under partial nulls: single-step max-T from joint
  sweeps is reported as such in the docs (subset pivotality is assumed), with Holm beside it.

## 12. Supervisor review checklist (per WP)

- [ ] Report present, commands and verbatim pass/lint lines included.
- [ ] No new broad handlers, global seeding, or mutable defaults (`test_implementation_standards`).
- [ ] Duplicated score extraction gone (`grep -n "loading_rotation" src/pysimlr/{loadings,utils,sem}.py`).
- [ ] p-values add-one, never 0; floor documented.
- [ ] Calibration tests pass under `pytest -m ""`.
- [ ] Figures opened and the section 9 checklist answered.
- [ ] Pilot rules P1-P4 reported with numbers from the CSVs.
- [ ] Docs describe what the test cannot say, not only what it can.
