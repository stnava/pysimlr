# Score-level path model: next steps after v0.24.0

Written 2026-11-07 by the supervising agent after the takeover audit (`/tmp/rev_aud_nv.md`).
State: v0.23.0 (path model, stats core, viz), v0.23.1 (pilot audit fixes), v0.24.0
(`path_model_from_scores`, training-row p-value policy, worker-process note) are released and
pushed. Tracking entries D41-D43. Each item below has a rule for "done" so it can be handed to an
agent without further supervision. Order is by value to the owner's stated goal: an interpretable
multivariate SEM / ANCOVA that generalises across all model families.

## 1. Mediation on `edge_models` (open since D41)

The node regressions in `pm["edge_models"]` already give direct effects. Add
`sem.mediation(pm, a, m, b)`: indirect effect a -> m -> b as the product of the ridge slopes
(per component, and the Pillai-style block version through the fitted node models), direct effect
of a on b given m, and the proportion mediated. Uncertainty from the subject bootstrap already run
for the edges (reuse `n_bootstrap` draws; no new fits) and a Freedman-Lane permutation of m's
residual for the indirect effect's p-value (add-one).
Done when: a test on the planted chain (`build_path_graph_case([(0, 1), (1, 2)])`) recovers an
indirect view0 -> view1 -> view2 effect with p at the floor and a direct view0 -> view2 effect
with p > 0.05; a star-shaped case gives the reverse; `plot_path_model` can annotate the indirect
path; usage notes state the linear-structural assumption.

## 2. Plan rule P3 needs a scope decision

P3 ("held-out Pillai below training for every edge") fails on ADNI only on view-outcome edges
(14/30), never on view-view edges (30/30). The measurement model never sees the outcome, so
training and held-out outcome Pillai are both out-of-sample estimates of a weak effect and
P3 has no reason to hold there. Decide and record in the plan: restrict P3 to edges whose both
ends were fitted (then the pilot passes 4/4), or keep it and report it as failed. Do not change
the rule in the script without the plan saying so (that was defect D2).
Done when: the plan file, the pilot script and `WP4_report.md` state the same rule and the same
outcome.

## 3. NHANES path model run (open since D41)

The NHANES cardiometabolic case is the package's public separation example. Run the path model
on it with the same discipline as the ADNI pilot: confounds to both `fit(confounds=)` and
`path_model(covariates=)`, held-out rows, plan-style rules written before the run, CSV numbers
in the report, figures opened. No injected nuisance, no semi-synthetic variants.
Done when: `scripts/nhanes_path_model_pilot.py` reproduces its CSVs live, a WP report with
verbatim test lines exists, and the tracking entry records the pre-registered rules and their
outcomes.

## 4. Scores-only entry points beyond `path_model`

`path_model_from_scores` and `predict_path` work on given scores; `compare_path_models` in
structural mode still needs a fit. Add `compare_path_models_from_scores(scores_train,
scores_test, graphs, ...)` through the same node-model code, and give `plot_edge_features` /
`plot_edge_brain_map` a clear error on a scores-only model (no `feature_contributions`).
Done when: a test shows the comparison on `view_scores` output equals the fit-based one, and
the two plot functions raise a `ValueError` naming the missing measurement model.

## 5. Attenuation of absent edges (finding F2)

Conditioning on estimated scores leaves residual association on an absent edge (planted chain:
absent-edge p 0.03-0.07 on some draws). Two options, both cheap to prototype on the planted
chain: (a) report the absent-edge test alongside a disattenuated partial Pillai using the
subsample-refit reliability from `loading_inference` (Croon-style correction of the score
covariance); (b) report power and the expected residual association under the fitted
reliabilities so the reader knows how much of an absent edge's signal is attenuation.
Done when: on the planted chain across 100 seeds the absent-edge rejection rate under (a) is
within [0.02, 0.09] at alpha 0.05 while the present-edge rate stays at 1.0; the usage notes
section "what the tests cannot say" is updated with the measured numbers.

## 6. Guidance for deep encoders in topology comparison

On ADNI, LEND selected the complete graph in 4/5 splits where simlr selected the star. A deep
encoder trained on all views couples every pair, so the structural comparison with the
measurement model fixed favours denser graphs. The training-topology refit mode (`method=`,
`k=`, `n_splits`) is the right comparison for deep models. Write this in usage notes with the
ADNI numbers already in `adni_comparison_refits.csv`, and make `compare_path_models` warn when
a deep method is compared in fixed mode only.
Done when: the warning has a test and the usage notes carry the ADNI example.

## 7. Consolidate the remaining score path in `relevance.py`

`relevance.py` lines 83-102 rotate `predict` latents by `loading_rotation` and sign-align them
against `u` on the given rows, a second extractor outside `sem.view_scores`. Route it through
`view_scores(result, views, sign_against=u, _pred=pred)` as `loadings._oob_metrics` does.
Done when: the relevance tests pass unchanged and `grep -n "loading_rotation" src/pysimlr/relevance.py`
shows no projection code.

## 8. `n_jobs` ergonomics

`n_jobs > 1` from a notebook fails with `BrokenProcessPool` (documented in v0.24.0). Detect the
condition up front: when `__main__` has no `__file__` (interactive) and `n_jobs > 1`, raise a
`ValueError` that says to use a script or `n_jobs=1`, instead of the opaque pool error.
Done when: a test monkeypatches `sys.modules["__main__"]` and asserts the message.

## 9. Paper chapter (open since D41)

Write the methods chapter from the module docstring and the plan: model, statistics, nulls,
multiplicity, comparison, prediction, what the tests cannot say, with the planted-chain and ADNI
figures. Use `brian-avants-writing` conventions; numbers only from CSVs in
`docs/audit/evidence/path_model/`.
Done when: every number in the chapter is traceable to a tracked CSV and the figures are the
committed PNGs.

## Not planned

Longitudinal or mixed-effects structural models, latent-variable SEM with free loadings, and
nonlinear structural models. The structural model is linear on scores by design (plan section 11).
