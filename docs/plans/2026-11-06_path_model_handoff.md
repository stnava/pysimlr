# Handoff: finish the score-level path model (WP3-WP5, verification, release)

Date: 2026-11-06. Supervisor so far: Claude Fable 5.1. Executor so far: one Opus 5.5 agent
(WP1 accepted; WP2 code-complete, stopped before its report). This document is written for an
agent team that has not seen this conversation. The full spec remains
`docs/plans/2026-11-06_path_model.md` (sections 0-12; 1b and F1-F5 are binding findings).
Read that first, then this file, then `docs/audit/evidence/path_model/WP1_report.md`.

## 1. State of the repository (verified by the supervisor at hand-off)

Branch `main`, HEAD `d0d35bd`. Nothing pushed. Two interim commits carry this work:

| commit | content | verification at commit time |
|---|---|---|
| `80616ec` | WP1: `src/pysimlr/sem.py` core, `tests/test_sem.py`, generator `build_path_graph_case`, `scripts/path_model_demo.py`, plan, evidence, WP1 report | fast suite 1993 passed; slow `-k test_sem` 21 passed; lint 0; WP1 report reviewed and accepted |
| `d0d35bd` | WP2: refit null, `compare_path_models` refit mode (4.2), `predict_path`, `predict_views` fix for joint-basis fits with a missing view, `feature_maps` on `view_scores`, `permutation_test` via `fit`, removals of `fit_structural_models` and `diagrams.plot_path_model`, exports | fast suite 1998 passed, 1 skipped, 2 xfailed; lint 0; standards 7 passed. **Slow tests not run. No WP2 report.** |

`285f238` (another session) added `pysimlr.stats`; `sem.py` uses its `holm` and `bh_fdr`. That
session took tracking entry D40; this work's entry is **D41**.

Public API now: `pysimlr.path_model`, `pysimlr.compare_path_models`, `pysimlr.predict_path`,
`pysimlr.view_scores`, `pysimlr.permutation_test(method=...)`, `pysimlr.create_path_graph`.
Reference figures and report: `docs/audit/evidence/path_model/reference_*` and `wp1/`.

## 2. Team and roles

Four agents, one lead. The lead does no implementation.

- **Lead (reviewer).** Assigns the packages below, reviews every report against section 6, runs
  the verification commands itself (never trusts a report's numbers without re-running at
  least the fast suite and lint), commits accepted packages one at a time with the house
  commit style (conventional prefix, body with verification lines, `Co-Authored-By` line from
  the session's attribution reminder), and runs the release at the end with the `release` skill.
- **Agent A: WP2 close-out and verification** (section 3). Must finish before C starts.
- **Agent B: WP3 visualization** (section 4). Can start immediately; depends only on `d0d35bd`.
- **Agent C: WP4 ADNI-MCI pilot** (section 5). Starts after A reports; uses B's plots if they
  have landed, otherwise the demo script's reference drawings.
- **Agent D: WP5 documentation** (section 5). Starts after C's pilot rules are in.

Rules for every agent (plan section 0 is binding; the essentials):
no commits, tags or pushes by agents (the lead commits); nothing in `paper/`; no new broad
excepts, global RNG seeding or mutable defaults (`tests/test_implementation_standards.py`);
add-one p-values; an implausible number is a defect to find; every figure rendered and
reviewed; a report per package under `docs/audit/evidence/path_model/WP<n>_report.md`, under
150 lines, with the exact commands and verbatim summary lines; deviations recorded with reasons;
no questions to the owner, apply the closest rule and record the choice.

## 3. Agent A: close WP2 (verification and report)

The code is in `d0d35bd`. Do not redesign it. Tasks:

1. Run and record: `python -m pytest -q -p no:cacheprovider` (fast),
   `python -m pytest -q -m "" -p no:cacheprovider tests/test_sem.py` (includes the slow
   `test_null_calibration_fixed_model` and `test_null_calibration_refit`),
   `python -m pytest -q tests/test_implementation_standards.py`,
   `python -m antsxqc.lint src/pysimlr`, `python scripts/path_model_demo.py --out
   docs/audit/evidence/path_model/wp2`.
2. Calibration rule: the refit-null per-edge rejection rate at alpha 0.05 must lie in
   [0.02, 0.09] (plan test 6). If it does not, that is a defect in `_refit_null` or in the
   permutation of confound rows; find it, do not widen the band.
3. Confirm by reading `sem._refit_null` that the permuted node's confound rows are permuted
   with it (`confounds.permute_view_confounds`) and that `test_refit_null_permutes_confounds_with_the_view`
   actually asserts on the `confounds` argument that reaches `fit` (not merely that fit was called).
4. Confirm `permutation_test` keeps its old keys (`observed_similarity`, `null_similarities`,
   `p_value`, `n_permutations`, `n_exceeding`) and the add-one floor
   (`test_paths_permutation_test_never_returns_zero` still exists and passes).
5. Grep for leftovers: `grep -rn "fit_structural_models\|holm_adjust\|_bh_qvalues\|_view_scores" src tests docs/usage_notes.md docs/api_contract.md README.md`.
   Any reference outside a docstring that says the function was removed is a defect.
6. Write `WP2_report.md` in the WP1 format: what WP2 built (read the `d0d35bd` diff), the
   command lines and their verbatim summary lines, deviations from plan sections 3.2, 4.2, 5,
   6 with reasons, open items. Include the WP3 carry-over: in `plot_path_comparison` the
   present-pass and absent-pass markers overlap when equal; dodge them vertically.

Acceptance: both slow calibration tests pass with their rates reported; fast suite and lint
green; report present.

## 4. Agent B: WP3 visualization

Spec: plan section 9. The reference drawings are the four functions in
`scripts/path_model_demo.py` (`plot_path_model`, `plot_edge_matrix`, `plot_path_comparison`,
`plot_edge_null`), already reviewed twice; move them into `src/pysimlr/plotting.py` under those
names, keep their encodings, and fix: (i) the overlapping markers in the comparison panel,
(ii) `plot_path_model(component=j)` draws per-component `|r_j|` with sign colour (`_BLUE[6]`
positive, `_RED[4]` negative), (iii) a `status` line on the path diagram when the graph-fit
test rejects (`pm["fit"]["rejected"]`), (iv) moderated edges marked when `pm["moderation"]`
has `p_holm < alpha`. Add `plot_edge_features(pm, edge, top_n)` from
`pm["feature_contributions"]` in the style of `plot_top_features`.

Then: `tests/test_plotting_sem.py` (each function returns a `Figure`, honours `ax` where
single-panel, the number of solid edges drawn equals the number of significant edges in the
table); register the names in `__init__.py` `__all__` and `_LAZY_ATTRS` (module `plotting`),
and in `docs/usage_notes.md`'s plotting table; make the demo script import them from
`pysimlr.plotting` and delete its local copies. Render the four figures plus
`plot_edge_features` on the planted chain to `docs/audit/evidence/path_model/wp3/` and answer
the plan's section 9 review checklist in the report, figure by figure.

Acceptance: figures reviewed against the checklist (no overlapping labels, palette only,
legend explains every encoding, subtitle states rows, null and B); `pytest -n auto` green;
lint 0; `WP3_report.md`.

## 5. Agents C and D: WP4 pilot, then WP5 documentation

**WP4 (Agent C)**, plan section 8, unchanged except as noted: data through
`scripts/benchmark_confound_sink.load_adni` (`data/adni/adni_mci_cognitive_merged.csv`), views
`T1Hier_vol`, `T1Hier_thk`, `T1w_mean`, outcome CogPC1 fitted on training rows, k = 3,
confounds age, education, head size passed to **both** `fit(confounds=)` and
`path_model(covariates=)`; `groups=sex` (never also a confound). Five 70/30 splits, methods
`simlr` then `lend`. Never use `scripts/adni_loader` multimodal cases or PPMI. Graphs:
complete, star on `T1Hier_thk`, chain vol-thk-mean, empty, each plus the outcome node; run
`compare_path_models` in both modes (fixed measurement, and refits with `n_splits=5`).
Outputs to `docs/audit/evidence/path_model/adni/`. Report rules P1-P4 with the numbers from
the CSVs. If P1 or P2 fails, investigate score extraction, sign alignment and the sink before
writing anything else; write what was found.

**WP5 (Agent D)**, plan section 10: `docs/usage_notes.md` section "Edges between views: the
score-level path model" after "New samples", in the register of the `feature_relevance`
docstring, including what the tests cannot say (F2 attenuation; max-T under partial nulls;
training-row coupling); `docs/api_contract.md` "Structural model" section listing every
result key of `path_model`, `compare_path_models`, `predict_path`; the `sem.py` module
docstring updated from "reference implementation" to the shipped description; the
`docs/tracking.md` entry **D41** in the house format (status, what landed, measured numbers
from WP1-WP4 reports, decisions, open items: mediation on `edge_models`, NHANES run, paper);
README one-paragraph pointer. `WP5_report.md`.

## 6. Lead: review checklist per package (from plan section 12, extended)

- [ ] Report present with verbatim pytest and lint lines; lead re-ran the fast suite and lint.
- [ ] Standards test passes; no new broad handlers, global seeding or mutable defaults.
- [ ] No second score extractor: `grep -n "loading_rotation" src/pysimlr/{loadings,utils,extension}.py` shows no projection code outside `sem.view_scores` calls.
- [ ] p-values add-one; calibration tests ran under `-m ""` with rates recorded.
- [ ] Figures opened by the lead and checked against section 9.
- [ ] Pilot rules P1-P4 reported with CSV numbers; failures investigated, not explained.
- [ ] Docs say what the tests cannot say.
- [ ] One commit per accepted package; message body carries the verification lines.

## 7. Release (lead, after WP5)

Version bump is minor, 0.22.2 -> 0.23.0 (new public API; `fit_structural_models` removed;
`permutation_test` now goes through `fit`). Use the `release` skill: it runs the pre-release
checks, bumps `pyproject.toml`, updates tracking, commits, tags and pushes. Before invoking it:
full suite `pytest -m ""` green, lint 0, `python scripts/path_model_demo.py` reproduces the
plan 1b smoke results, and the ADNI pilot CSVs and figures are in the tree.

## 8. Known open items carried forward (do not solve in this hand-off)

- Mediation / indirect effects on `edge_models` for directed graphs.
- NHANES run of the path model (after ADNI).
- Paper chapter on the path model (owner's call).
- `loading_inference` always stores `oob_scores` (about 4 MB at n = 1000, M = 4, k = 5, 100
  subsamples); revisit only if a user hits memory limits.
