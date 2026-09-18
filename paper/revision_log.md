# Revision Log: Systematic Paper Improvement

## Round 1: Systematic Gap Resolution (Completed)
- [x] Task 1: Conceptual Clarity (Introduction) - @viz-educator (Fixed Post-hoc vs Mechanical comparison)
- [x] Task 2: Reproducibility (Methods - NEDPP Schedule) - @dl-optimizer (Fixed N=10 and Newton-Schulz consensus explanation)
- [x] Task 3: Statistical Grounding (Experiments - Random Baseline) - @statistician (Improved confidence intervals with N=10 runs)
- [x] Task 4: Clinical Depth (Experiments - Expert Interpretation) - @xai-scientist (Added Local Patient Audit waterfall plot)
- [x] Task 5: Figure Simplification (Architecture) - @viz-educator (Integrated orthogonality generalization analysis)

## Round 2: Final Review & Refinement (Completed)
- [x] Cross-review of implemented changes by @mediator and @xai-scientist.
- [x] Final narrative alignment and figure polish.
- [x] Build successful with N=10 v10_final results.

## Round 3: Statistical correctness review (code-driven)

Prompted by a correctness audit of `pysimlr` (see `CORRECTNESS_AUDIT.md`) and a
re-derivation of the benchmark statistics (see `STATISTICAL_REANALYSIS.md`).

Analysis code:
- [x] `scripts/calc_stats.py`: `LEND` was grouped as `Linear`. LEND is a deep
      architecture with a linear *encoder*, and the manuscript text at
      `03_experiments.qmd:280` already described it as Deep — so the published
      numbers were computed with a grouping that contradicted the stated
      comparison. LEND is now Deep.
- [x] `scripts/calc_stats.py`: `Flow-SiMLR-V` was absent from `model_map`, so
      `pandas.Series.map` returned NaN and all 160 of its rows were dropped
      from every Deep-vs-Linear test silently. Added, and the script now raises
      on any unmapped model rather than discarding it.
- [x] Added `scripts/calc_stats_corrected.py`: seed-level paired tests plus
      exact sign-flip permutation p-values, replacing the pooled two-sample
      t-test over all 200 rows per cell. Each cell holds only 5 independent
      replicates (seeds); the rest are repeated measures across losses and
      consensus algorithms on the same splits, so the pooled test understated
      the standard error.

Manuscript:
- [x] `03_experiments.qmd:280`: replaced `p = 8e-11` / `p = 3e-4` with
      seed-level effect sizes, 95% CIs and seed-consistency counts. Stated the
      0.0625 permutation floor for a 5-seed design.
- [x] `03_experiments.qmd:280`: added the Polynomial-regime negative result
      (3--4/5 seeds, p = 0.166 and 0.123, CIs spanning zero). The regimes that
      do separate are Linear, Sine and Private/Newton-ICA.
- [x] `03_experiments.qmd:280`: added the CMC qualification — deep is *worse*
      than linear in the Linear regime on 0/5 seeds.
- [x] `03_experiments.qmd:280`: added the Diabetes consensus-conditionality
      qualification (present under SVD/PCA, absent under Newton/ICA).
- [x] `03_experiments.qmd:420` and two table captions: corrected statistics and
      clarified that "linear" denotes the baseline *model*, whereas LEND is a
      deep architecture with a linear encoder.
- [x] `04_discussion.qmd:40`: replaced the Welch t-test p-value with the
      seed-level effect size and CI; added the consensus conditionality and a
      paragraph on the limits of a 5-seed design.

Not done (needs author decision):
- [ ] Raising the seed count. This is the only way to obtain p-values below
      0.0625; crossing more losses or consensus algorithms adds no replication.
- [ ] Re-running the benchmarks against the corrected library. Every number
      above comes from the existing `results_cache/*_v21.csv`, produced before
      the correctness fixes. Several of those fixes change numerical results
      (the deep early-stopping fix in particular, which had been truncating
      training to roughly a fifth of the requested epochs).
