# How to re-run the full real-data benchmark evaluation

This is a handoff doc for reproducing (or extending) the multi-pass real-data
benchmark sweep that re-established pysimlr's method/initialization
performance after a round of internal changes (new initialization
strategies, `positivity="either"`, `safe_svd` LAPACK fallback, convergence
budget fixes). Follow it literally; do not re-derive the approach from
scratch.

## 1. What you are evaluating

Six method families, each optionally paired with an initialization strategy:

- `PCA` (raw, concatenate-views-then-PCA) -- no method, a baseline
- `GCCA` (raw, joint MAXVAR consensus) -- no method, a baseline
- `SiMLR` -- linear, has an initializable basis
- `LEND-SiMR` -- deep, has an initializable basis
- `NED-SiMR` -- deep, has an initializable basis
- `Flow-SiMR` -- normalizing flow, **no** linear encoder -- cannot be
  initialized, method-comparison only
- `Flow-SiMR-V` -- normalizing flow with a Stiefel linear encoder -- has an
  initializable basis

Initialization strategies (`INITIALIZATION_TYPES` in `src/pysimlr/simlr.py`):
`pca`, `random`, `gcca`, `nndsvd`, `perturbed_pca`, `domain`, `best_of_n`.
**Exclude `best_of_n` from benchmark sweeps** (`BENCHMARK_INITIALIZATION_TYPES`
in `scripts/real_data_init_comparison.py`) -- its own scoring proxy makes it
uninteresting as a "strategy" to benchmark (it's a wrapper that tries several
of the others and keeps the best by that proxy, so it can only ever look
average-to-good, never bad, which tells you nothing).

Naming convention -- **use this exactly, it was a hard-won fix for real
ambiguity**: `Method[init]` when a method's initialization matters (e.g.
`NED[gcca]`), `<strategy>-raw` when the "method" IS the initializer used
standalone (e.g. `gcca-raw` = joint MAXVAR consensus with no downstream
learning). Never print a bare `"GCCA"` or `"PCA"` label without saying
whether it means the raw baseline or an initializer for something else --
the two readings differ dramatically in score and utility, and conflating
them was the source of an actual back-and-forth confusion in the original
run.

## 2. Datasets that are "established as useful"

Use exactly this list -- it survived investigation and everything else has
been tried and explicitly rejected. Do not re-add rejected datasets without
new evidence.

| Dataset | Loader | Task | N | Views |
|---|---|---|---|---|
| Heart Disease | `load_heart` | classification | 297 | [7, 6] |
| Diabetes | `load_diabetes` | regression | 442 | [5, 5] |
| mfeat | `load_mfeat` | classification | 2000 | [76, 64, 6, 47] |
| ADNI-DX | `load_adni` | classification | 685 | [300, 145, 36] |
| ADNI-CDRSB | `get_adni_cdrsb_case` (`adni_loader.py`) | regression w/ covariates | 691 | [300, 145, 36] |
| TCGA-KIRC | `load_kirc` | classification | 467 | [300, 300, 232] |

Loaders for the first three and KIRC live in
`scripts/real_data_init_comparison.py`; ADNI loaders live in
`scripts/adni_loader.py`.

**Rejected -- do not use without a new reason:**
- TCGA-BRCA: every method sat at/below majority-class baseline; cannot
  currently distinguish initialization or method quality from noise.
- ADNI tau/amyloid cohorts (PPMI-only N=241, or the merged N=47 biomarker
  cache): too small to draw any conclusion from; N=47 in particular was
  investigated and confirmed to be a real ceiling of co-acquired
  tau+amy+imaging subjects, not a loader bug.
- ADNI-CDRSB restricted to MCI-only subjects, or MMSE as target: tried,
  no method beat the covariates-only baseline -- not informative.

If you find a new candidate dataset, vet it the same way before adding it to
this table: run the method comparison once, and only keep it if at least one
method beats the trivial baseline (majority class / mean-prediction / R2 of
covariates alone) by a real margin.

## 3. Convergence discipline -- read this before changing any budget

**Do not hand-pick a fixed epoch/iteration count and assume it's enough.**
Every method here has its own `ConvergenceMonitor` (relative-improvement
early stopping, patience=10, returns the best-loss iterate not the last
one). The job of a "budget" constant is only to be a generous cap that the
monitor should trip *before* reaching -- verify this, don't assume it.

Current verified budgets (`scripts/real_data_method_comparison.py`,
`scripts/real_data_init_x_method.py`):
```python
DEEP_EPOCHS = 200        # LEND, NED -- converges well within this on every dataset checked
FLOW_EPOCHS = 400        # Flow-SiMR, Flow-SiMR-V need more -- see below
SIMLR_ITERATIONS = 150
DEEP_WARMUP = None       # resolves to min(20, epochs // 4); a flat warmup can
                         # silently disable the alignment term on short runs
```
`FLOW_EPOCHS=400` exists because Flow-SiMR-V was caught genuinely still
improving at epoch 199/200 on ADNI-DX (`best_epoch=199, max_steps=200`,
i.e. never stopped early) -- raising the cap to 400 let it converge
properly (`stop_reason='converged'`, `best_epoch=285`) and its score moved
from 0.4612 to 0.5000. LEND/NED did not need this.

After every run, call the convergence check on the output and print (don't
silently swallow) any cap-hit:
```python
def _check_converged(label, out):
    conv = out.get("convergence")
    if conv is not None:
        if not conv.get("converged", False):
            print(f"    [!] {label} did not converge within its cap "
                 f"(best_epoch={conv.get('best_epoch')}, max_steps={conv.get('max_steps')})", flush=True)
        return
    if out.get("stop_reason") == "max_iter":
        print(f"    [!] {label} hit max_iter without converging", flush=True)
```

**Known, already-investigated false positive:** `SiMLR[pca]` (and
occasionally `SiMLR+random`) frequently prints `[!] did not converge` even
though the same call reproduced standalone with an identical seed reports
`converged: True`. This is a genuine floating-point/BLAS-threading
knife-edge -- the certificate value lands within ~1e-7 of `tol=1e-6` on
already-near-optimal, well-conditioned real data. It is not a bug. **Do not
re-investigate this every time it recurs** -- it will recur on most
datasets. Only escalate a `[!]` if `best_epoch` is at or near the hard cap
(genuine still-improving non-convergence, like the Flow-SiMR-V case above),
not if it's a SiMLR cell with no `best_epoch`/`max_steps` reported at all.

Also required: `positivity="either"` passed explicitly everywhere (not the
old `"positive"` default), and every raw `torch.linalg.svd` call must go
through `safe_svd` (in `src/pysimlr/utils.py`), which falls back to
`scipy.linalg.svd(..., lapack_driver='gesvd')` when `gesdd` raises
`torch.linalg.LinAlgError` -- this happens reliably (100% of attempts) for
`gcca`/`nndsvd` initialization on real ADNI-scale data after standardization.
If you see `gcca`/`nndsvd` failing on any new real dataset, this is almost
certainly the same LAPACK driver issue -- check `safe_svd` is actually being
called on the path that's failing before treating it as a new bug.

## 4. Downstream evaluation protocol

- Learn a shared `k=3`-dimensional latent `u` per (dataset, method, init)
  combination.
- Score with a **random forest** (`RandomForestClassifier`/
  `RandomForestRegressor`, `n_estimators=300`, fixed `random_state=seed`),
  not a linear/logistic probe -- a nonlinear reader avoids penalizing
  methods that pack curved structure into few dimensions.
- 70/30 train/test split (`train_test_split`, `stratify=y` for
  classification), same `seed` used for the split and every method's
  internal RNG state (`torch.manual_seed(seed)` before deep methods,
  `init_seed=seed` for `random`/`perturbed_pca` init, and a
  `np.random.default_rng(seed)` for constructing `domain_matrices`).
- For ADNI-CDRSB specifically: it's `"regression_with_covariates"` -- the
  latent `u` gets concatenated with clinical covariates before the RF, and
  you must also print a `covariates-only` baseline (covariates alone, no
  latent) so you can tell whether the learned representation adds anything
  over demographics/clinical variables alone.
- For plain regression (Diabetes), do **not** print a "majority baseline" --
  that's a classification concept and is meaningless (and was a real bug
  caught mid-run: printed `acc=0.0301` for a continuous target). Print
  instead: `"(regression target -- R2=0 is the mean-prediction baseline)"`.
- Single run per cell per pass -- this is a spot-check sweep, not a
  many-seeds statistical benchmark. Seed-sensitivity is instead handled by
  running the *whole sweep* at a second seed (see Section 6) and comparing.

## 5. Execution order and script layout

Run **one dataset fully to completion before starting the next.** Do not
run datasets in parallel / interleaved -- merged log output from concurrent
runs is genuinely hard to attribute correctly and caused real misreadings
in the original session.

The master script is `scripts/full_benchmark_pass.py`:
```
python scripts/full_benchmark_pass.py <pass_label> [seed]
```
It runs, in order, for each of the 6 datasets: standalone/method comparison
(PCA-raw, GCCA-raw, then each of SiMLR/LEND/NED/Flow-SiMR/Flow-SiMR-V
initialized with each method's own default, i.e. `Method[pca]`), then the
full initialization x method sweep (6 strategies x {SiMLR, LEND, NED,
Flow-SiMR-V}). It imports its method/loader logic from
`real_data_method_comparison.py` and `real_data_init_x_method.py` rather
than duplicating it -- if you add a method or dataset, add it there, not by
copy-pasting into `full_benchmark_pass.py`.

Run it in the background and tail-filter the log rather than watching raw
output:
```
nohup python scripts/full_benchmark_pass.py pass1 42 > /tmp/benchmark_pass1.out 2>&1 &
tail -f -n +1 /tmp/benchmark_pass1.out | grep -E "^====|^###|acc=|R2=|FAILED|\[!\]|Traceback|Error|^  ---"
```
Wait for the literal line `### <pass_label> complete ###` before treating a
pass as finished and moving to the next.

## 6. The three-pass protocol

1. **Pass 1 (baseline)**: `python scripts/full_benchmark_pass.py pass1 42`.
   Run with whatever budgets/fixes are currently in place. Read the full
   output for anything implausible (bogus baselines, unexpected `FAILED`,
   `[!]` at a genuine hard-cap boundary) and fix root causes before Pass 2 --
   don't carry forward a fix you're not sure about, verify it by direct
   reproduction outside the sweep first.
2. **Pass 2 (same seed, with fixes)**: `python scripts/full_benchmark_pass.py pass2 42`.
   Same seed as Pass 1 -- this isolates the effect of your fixes from
   run-to-run noise, since with a fixed seed the two passes should be
   deterministically comparable cell-by-cell except where a fix changed
   behavior. Confirm every fix actually changed what you expected (e.g. a
   convergence-budget fix should remove a specific `[!]` line and move that
   cell's score) and nothing else moved unexpectedly.
3. **Pass 3 (different seed, consistency check)**:
   `python scripts/full_benchmark_pass.py pass3-seed123 123`.
   This is the seed-robustness check, not a rerun of the same thing --
   its entire purpose is to tell you which "best combo per dataset"
   conclusions from Pass 2 are real signal vs. single-seed noise.

## 7. How to read the final result -- seed-robust vs. seed-sensitive

Compare Pass 2 and Pass 3 cell-by-cell per dataset. Classify each dataset's
"best combo" finding as:
- **Seed-robust** if the same method/init combination (or a small, stable
  set of them) wins at both seeds, even if exact scores differ. Report
  these as reliable conclusions.
- **Seed-sensitive** if the winning combo flips between seeds, especially
  on smaller/weaker-signal datasets. Report the qualitative pattern that
  *does* survive (e.g. "random init is bad everywhere," "GCCA-raw is
  consistently worst on this dataset") rather than the specific
  single-seed "best" cell, and say explicitly that the specific winner
  is not to be trusted.

Do not average Pass 2 and Pass 3 scores together or treat Pass 3 as a
tiebreaker for "the" answer -- the point of Pass 3 is to reveal which
answers don't have one.

Last known result of running this exact protocol (2026-09-21, for
reference/regression-checking only -- re-verify, don't just cite this):
Heart Disease (gcca/domain-init dominance), mfeat (NED[pca] /
NED+perturbed_pca best), and TCGA-KIRC (NED best method family, GCCA-raw
consistently worst) were seed-robust. Diabetes's best init and
ADNI-CDRSB's specific best combo were not seed-robust (though NED+gcca
recurred as a strong ADNI-CDRSB performer at both seeds). ADNI-DX showed
weak, noise-dominated separation at seed 123 versus more visible
separation at seed 42.
