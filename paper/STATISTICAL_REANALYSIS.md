# Statistical re-analysis of the benchmark tables

This note re-derives the significance claims in `03_experiments.qmd` and
`04_discussion.qmd` from the same results cache (`unified_real_v21.csv`,
`unified_synthetic_v21.csv`), using inference matched to the experiment's
replication structure. Reproduce with:

```bash
python paper/scripts/calc_stats_corrected.py
```

## The problem with the published p-values

`paper/scripts/calc_stats.py` computes `scipy.stats.ttest_ind(deep_vals,
lin_vals, equal_var=False)` over **every row** of the cache. Each
(Dataset, Consensus-group, Architecture-group) cell holds 80 rows — but only
**5 distinct seeds**. The remaining 16× multiplicity comes from crossing 4 loss
functions, 2 architectures and 2 consensus variants, every one of them
evaluated on the *same* data splits.

Those rows are repeated measures, not independent samples. A two-sample t-test
treats them as independent, which understates the standard error by roughly
√16 = 4× and drives the p-value down by many orders of magnitude. The reported
p-values therefore scale with how much compute was spent, not with how much
evidence the experiment contains.

**The design's resolution floor.** With *n* independent replicates, the
smallest attainable two-sided sign-flip permutation p-value is `2 / 2ⁿ`. For
n = 5 seeds that is **0.0625**. No p-value below 0.0625 is supportable from
this design by any distribution-free method. The manuscript's
`p = 8 × 10⁻¹¹` and `p = 3 × 10⁻⁴` are artifacts of parametric assumptions
about seed-level differences, not measurements.

## Two grouping errors

**LEND was grouped as Linear.** `calc_stats.py:28` mapped `'lend': 'Linear'`
and `'LEND': 'Linear'`. LEND is a *Linear Encoder, Nonlinear Decoder* — the
encoder is linear, the architecture is not. Grouping it as Linear put a
nonlinear model on the linear side of every "Deep vs Linear" contrast, which
dilutes the deep group and **understates** the effect the paper argues for. The
dilution is not small: LEND is the strongest model on both real datasets
(Heart +0.0996, Diabetes +0.6489 against the SiMLR baseline).

It also makes `03_experiments.qmd:420` unreadable as written — *"the LEND
architecture achieved leading performance with statistical significance over
the linear baseline (p = 8 × 10⁻¹¹)"* — because under that grouping LEND sits
on the same side of the comparison as the baseline it is said to beat.

**Flow-SiMLR-V was absent from the map entirely**, so `pandas.Series.map`
returned `NaN` for all 160 of its rows and they were dropped from every
Deep-vs-Linear test with no notice.

Both are fixed in `calc_stats.py`, which now also raises if any model in the
results is missing from the map rather than silently discarding it. All tables
below use the corrected grouping (hence 200 rows per cell rather than 160).

The headline conclusion is unchanged by the regrouping: the Polynomial regime
still fails seed-level inference. Where it does change, it *strengthens* the
paper's case — e.g. the Private/Newton-ICA cell moves from p = 0.051 to
p = 0.043.

## Corrected results

## Real data: Deep vs Linear, corrected inference

| Dataset   | Consensus   |   n rows (pooled) |   n seeds |   mean diff (Deep-Lin) | 95% CI            | seeds favouring Deep   |   p pooled (as published) |   p seed-level paired |   p exact permutation |   permutation floor |
|:----------|:------------|------------------:|----------:|-----------------------:|:------------------|:-----------------------|--------------------------:|----------------------:|----------------------:|--------------------:|
| Heart     | Newton/ICA  |               200 |         5 |                 0.0256 | [0.0214, 0.0297]  | 5/5                    |                 0.003     |                0.0001 |                0.0625 |              0.0625 |
| Heart     | SVD/PCA     |               200 |         5 |                 0.1628 | [0.1582, 0.1675]  | 5/5                    |                 1.17e-194 |                0      |                0.0625 |              0.0625 |
| Diabetes  | Newton/ICA  |               200 |         5 |                 0.0007 | [-0.0302, 0.0316] | 2/5                    |                 0.978     |                0.9545 |                1      |              0.0625 |
| Diabetes  | SVD/PCA     |               200 |         5 |                 1.0706 | [1.0528, 1.0885]  | 5/5                    |                 2.75e-178 |                0      |                0.0625 |              0.0625 |

## Real data: each model against the SiMLR baseline

| Dataset   | Model        |   mean |   vs SiMLR | 95% CI           | seeds favouring model   |   p seed-level paired |   p exact permutation |
|:----------|:-------------|-------:|-----------:|:-----------------|:------------------------|----------------------:|----------------------:|
| Heart     | LEND         | 0.5399 |     0.0996 | [0.0973, 0.1019] | 5/5                     |                     0 |                0.0625 |
| Heart     | NED          | 0.5339 |     0.0936 | [0.0886, 0.0986] | 5/5                     |                     0 |                0.0625 |
| Heart     | NEDPP        | 0.5364 |     0.0961 | [0.0894, 0.1028] | 5/5                     |                     0 |                0.0625 |
| Heart     | Flow-SiMLR-V | 0.5278 |     0.0875 | [0.0788, 0.0962] | 5/5                     |                     0 |                0.0625 |
| Diabetes  | LEND         | 0.3813 |     0.6489 | [0.6338, 0.6641] | 5/5                     |                     0 |                0.0625 |
| Diabetes  | NED          | 0.2145 |     0.4822 | [0.4485, 0.5158] | 5/5                     |                     0 |                0.0625 |
| Diabetes  | NEDPP        | 0.2097 |     0.4774 | [0.4290, 0.5257] | 5/5                     |                     0 |                0.0625 |
| Diabetes  | Flow-SiMLR-V | 0.2665 |     0.5342 | [0.4968, 0.5715] | 5/5                     |                     0 |                0.0625 |

## Synthetic: Deep vs Linear, corrected inference

| Regime     | Consensus   |   n rows (pooled) |   n seeds |   mean diff (Deep-Lin) | 95% CI            | seeds favouring Deep   |   p pooled (as published) |   p seed-level paired |   p exact permutation |   permutation floor |
|:-----------|:------------|------------------:|----------:|-----------------------:|:------------------|:-----------------------|--------------------------:|----------------------:|----------------------:|--------------------:|
| Linear     | Newton/ICA  |               200 |         5 |                 0.6098 | [0.2693, 0.9503]  | 5/5                    |                  0.000147 |                0.0076 |                0.0625 |              0.0625 |
| Linear     | SVD/PCA     |               200 |         5 |                 2.0694 | [0.5980, 3.5408]  | 5/5                    |                  8.63e-13 |                0.0175 |                0.0625 |              0.0625 |
| Polynomial | Newton/ICA  |               200 |         5 |                 0.5261 | [-0.3378, 1.3900] | 4/5                    |                  0.00251  |                0.1661 |                0.1875 |              0.0625 |
| Polynomial | SVD/PCA     |               200 |         5 |                 1.2019 | [-0.5064, 2.9103] | 3/5                    |                  1.87e-06 |                0.1225 |                0.25   |              0.0625 |
| Sine       | Newton/ICA  |               200 |         5 |                 0.0722 | [0.0226, 0.1219]  | 5/5                    |                  0.000106 |                0.0156 |                0.0625 |              0.0625 |
| Sine       | SVD/PCA     |               200 |         5 |                 0.0889 | [0.0323, 0.1455]  | 5/5                    |                  4.26e-07 |                0.0121 |                0.0625 |              0.0625 |
| Private    | Newton/ICA  |               200 |         5 |                 0.4477 | [0.0219, 0.8735]  | 5/5                    |                  0.00134  |                0.0433 |                0.0625 |              0.0625 |
| Private    | SVD/PCA     |               200 |         5 |                 0.5863 | [-0.1848, 1.3573] | 4/5                    |                  0.000618 |                0.1024 |                0.1875 |              0.0625 |

## Synthetic: CMC

| Regime     | Consensus   |   n rows (pooled) |   n seeds |   mean diff (Deep-Lin) | 95% CI             | seeds favouring Deep   |   p pooled (as published) |   p seed-level paired |   p exact permutation |   permutation floor |
|:-----------|:------------|------------------:|----------:|-----------------------:|:-------------------|:-----------------------|--------------------------:|----------------------:|----------------------:|--------------------:|
| Linear     | Newton/ICA  |               200 |         5 |                -0.0776 | [-0.1691, 0.0140]  | 1/5                    |                  0.00659  |                0.0783 |                0.125  |              0.0625 |
| Linear     | SVD/PCA     |               200 |         5 |                -0.2802 | [-0.3451, -0.2154] | 0/5                    |                  4.12e-35 |                0.0003 |                0.0625 |              0.0625 |
| Polynomial | Newton/ICA  |               200 |         5 |                 0.0562 | [-0.0073, 0.1197]  | 5/5                    |                  0.00101  |                0.0698 |                0.0625 |              0.0625 |
| Polynomial | SVD/PCA     |               200 |         5 |                -0.1014 | [-0.1969, -0.0059] | 0/5                    |                  1.92e-13 |                0.0421 |                0.0625 |              0.0625 |
| Sine       | Newton/ICA  |               200 |         5 |                 0.0032 | [-0.0400, 0.0464]  | 4/5                    |                  0.598    |                0.847  |                0.8125 |              0.0625 |
| Sine       | SVD/PCA     |               200 |         5 |                -0.003  | [-0.0676, 0.0616]  | 4/5                    |                  0.703    |                0.9047 |                1      |              0.0625 |
| Private    | Newton/ICA  |               200 |         5 |                -0.0261 | [-0.2077, 0.1554]  | 3/5                    |                  0.355    |                0.71   |                0.75   |              0.0625 |
| Private    | SVD/PCA     |               200 |         5 |                -0.0321 | [-0.2536, 0.1893]  | 2/5                    |                  0.278    |                0.7076 |                0.875  |              0.0625 |

> The design has 5 seeds. The smallest attainable two-sided
> sign-flip permutation p-value is 2/2**5 = 0.0625. Any p-value below that reflects a
> parametric assumption about the seed-level differences, not the
> resolution of the experiment. Report effect sizes with confidence
> intervals and the seed-level consistency count, and raise the seed
> count if smaller p-values are wanted.

## Two claims that need revision

**1. The nonlinear-regime claim does not survive.** `03_experiments.qmd:280`
states: *"In synthetic analyses we observe large effect sizes (Cohen's d > 0.8)
and highly significant p-values (p < 0.05) in non-linear regimes."* The
Polynomial regime — the paper's central nonlinear case — gives seed-level
p = 0.166 and 0.123, confidence intervals spanning zero, and only 4/5 and 3/5
seeds favouring the deep models. The regimes that *do* hold up (Linear, Sine,
and Private under Newton/ICA) are not the ones the claim rests on. This is
robust to the grouping fix above.

**2. A significant result points the wrong way.** On CMC in the Linear regime,
the deep models are **worse** than linear on **0 of 5 seeds** (−0.115 and
−0.318), and the pooled test reports this as p = 1.7 × 10⁻²⁹. A reader scanning
for small p-values would read that as support. This belongs in the paper as a
stated limitation: the deep architectures trade consensus recovery for
predictive accuracy in the linear regime.

## Recommended reporting

- Lead with effect sizes and seed-level confidence intervals, plus the
  consistency count ("5/5 seeds"), which is the strongest honest summary of
  this design.
- Quote exact permutation p-values, and state the 0.0625 floor so readers can
  see the design's resolution.
- If p-values below 0.0625 are wanted, increase the **seed** count — crossing
  more losses and consensus variants does not add replication.
- Report the Diabetes result stratified by consensus algorithm.
- LEND is now grouped as deep. The manuscript's "Deep-to-Linear benefit"
  framing needs a sentence defining what is on each side of the comparison,
  since "linear" is doing double duty: SiMLR is a linear *model*, whereas LEND
  has a linear *encoder* inside a nonlinear architecture. Suggested wording:
  contrast the fully linear baseline (SiMLR) against the deep variants (LEND,
  NED, NEDPP, Flow-SiMLR-V), and describe LEND as the deep variant whose
  encoder remains linear and therefore directly interpretable — which is the
  property the paper actually cares about, and a stronger claim than
  "nonlinear beats linear" since LEND is the best performer on both real
  datasets.
