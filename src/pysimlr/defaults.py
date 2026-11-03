"""Package-wide defaults, defined once.

Every entry point, model constructor and helper that has an ``energy_type`` or
an NSA-Flow weight takes its default from here, so a method cannot quietly
disagree with the others. Before 0.3.0 the linear path defaulted to
``energy_type="acc"`` and ``nsa_w=0.5`` while the deep and flow paths used
``"regression"`` and ``0.1``, with no recorded reason for the difference.
"""

#: Similarity term of the objective (see `similarity.SIMILARITY`): normalised
#: correlation between each view's projection and the shared consensus. Chosen
#: jointly with `DEFAULT_LINEAR_OPTIMIZER` and newton mixing on
#: scripts/default_selection_benchmark.py (4 energies x 5 optimizers x 4 mixings
#: x both positivities x 2 generators x 5 seeds) and
#: scripts/default_followup_checks.py (docs/audit/evidence/default_*.csv):
#: "nc"+hybrid_adam+newton recovers the planted latent at mean R^2 0.969 (worst
#: cell 0.915) against 0.65 (worst 0.20) for the previous "acc" default, passes
#: the narrow-view (0.938/0.985), NED-collapse (0/5) and NED-budget checks, and
#: scores ADNI-MCI held-out 0.167/0.162 against 0.155/0.153. It needs the
#: collapse penalty (`DEFAULT_COLLAPSE_WEIGHT`): "nc" alone does not see a
#: collapsed latent.
DEFAULT_ENERGY_TYPE = "nc"

#: NSA-Flow trade-off weight w in (0, 1) between fidelity to the unconstrained
#: basis and orthogonality. 0.5 sits inside the stable conditioning plateau
#: (README, "nsa_flow Trade-off Weight").
DEFAULT_NSA_W = 0.5

#: Largest `nsa_w` any entry point accepts. At w = 1 the NSA-Flow fidelity
#: term drops out, the scale of the result is unconstrained and the answer is
#: arbitrary; it is known to be unstable and must not be run (owner rule). The
#: backend clamps the weight it receives further (`sparsification.NSA_MAX_W`).
MAX_NSA_W = 0.99


def check_nsa_w(nsa_w) -> float:
    """`nsa_w` as a float in ``[0, MAX_NSA_W]``, else `ValueError`."""
    w = float(nsa_w)
    if not 0.0 <= w <= MAX_NSA_W:
        raise ValueError(f"nsa_w must be in [0, {MAX_NSA_W}]; got {w}. nsa_w = 1 is known to be "
                         f"unstable (the NSA-Flow fidelity term vanishes).")
    return w

#: Consensus mixing. "newton" (orthogonal polar factor by Newton iteration) was
#: the best mixing for every top optimizer under "nc" in
#: scripts/default_selection_benchmark.py (hybrid_adam: 0.969 mean latent R^2
#: with 0 NED collapses; "avg" 0.971 with 1 collapse in 5 seeds;
#: docs/audit/evidence/default_selection_grid.csv, default_followup_checks.csv).
DEFAULT_MIXING_ALGORITHM = "newton"

#: Consensus each view is trained against. "loo": view m is scored against the
#: consensus of the other views only, so its own private variance cannot raise
#: its own objective. Under "star" the consensus includes the view itself; on a
#: generator whose per-view leading directions are private, `simlr` iteration
#: then does not improve on its PCA start (8-seed mean gain in least-squares latent R^2
#: -0.002), against +0.217 under "loo" (scripts/experiment_iteration_vs_init.py,
#: docs/audit/evidence). The reported shared latent and the prediction map are
#: the all-view consensus under either topology.
DEFAULT_TOPOLOGY = "loo"

#: Optimizer for the linear `simlr` path. Chosen on
#: scripts/optimizer_benchmark_synthetic.py (objectives acc/nc x mixing
#: newton/svd/avg x positivity positive/either, 3 seeds each; optimality gap at
#: 25 iterations against a 200-iteration reference): hybrid_adam had the best
#: mean rank (1.9 of 6) and was in the top three in 11 of 12 settings, against
#: mean rank 4.8 for the previous default armijo_gradient. No optimizer changed
#: the held-out ADNI-MCI score beyond seed noise. The deep and flow paths use
#: their own torch optimizers.
DEFAULT_LINEAR_OPTIMIZER = "hybrid_adam"

#: Weight on the dimensional-collapse penalty (`utils.correlation_penalty` of
#: each view's scores), the same on the linear and deep paths. Without it, "nc"
#: + newton mixing returned bases whose two score columns correlated -0.98
#: (linear, narrow views: outcome R^2 0.30) and NED latents correlated +-0.97 on
#: 2 of 3 seeds, while the orthogonalised consensus showed nothing.
DEFAULT_COLLAPSE_WEIGHT = 1.0

#: Correlation below which `utils.correlation_penalty` charges nothing; above
#: it the cost rises quadratically to 1 per fully collinear column pair. 0.3
#: from the comparison recorded in that function's docstring. It is a trade-off,
#: settable per fit as ``collapse_threshold``: it protects narrow views (at 0.7
#: the narrow fixture's score correlation rose to 0.89 and outcome R^2 fell
#: 0.10), but it also caps the score correlation that correlated factors can
#: keep under ``mixing_algorithm="avg"`` at about the threshold (converged fits:
#: max 0.31 at 0.3, 0.51 at 0.5, 0.69 at 0.7;
#: docs/audit/evidence/collapse_threshold_selection.csv).
DEFAULT_COLLAPSE_THRESHOLD = 0.3


def check_collapse_threshold(collapse_threshold) -> float:
    """`collapse_threshold` as a float in ``[0, 1)``, else `ValueError`."""
    t = float(collapse_threshold)
    if not 0.0 <= t < 1.0:
        raise ValueError(f"collapse_threshold must be in [0, 1); got {t}.")
    return t


#: The relative tolerance of the energy/loss plateau rule, for every method.
#: Linear: at 1e-6 the rule almost never fired (0/20 default fits converged
#: within 100 sweeps, 14/20 within 300, the rest creeping monotonically by
#: 1e-6 to 1e-5 per sweep); fits stopped at 1e-4 match the 300-sweep fits to
#: < 0.001 latent R^2 with support F1, frame defect and overlap unchanged
#: (scripts/linear_tol_selection.py). Deep: the per-epoch loss moves by ~1e-3
#: from minibatch noise, so 1e-6 and 1e-4 made identical decisions on all 36
#: lend / ned / ned_shared_private fits measured (scripts/deep_tol_selection.py).
DEFAULT_TOL = 1e-4

#: `simlr`'s sweep budget. At `DEFAULT_TOL` the measured fits stop after ~73;
#: the rest is headroom for harder problems.
DEFAULT_SIMLR_ITERATIONS = 300

#: Epoch budget of every deep and flow fit (lend, ned, ned_shared_private,
#: flow_simr, flow_simr_v); the plateau rule (`DEFAULT_TOL`) stops most fits
#: well before it. At 150 the 20 warmup epochs left 130 steps, and on the
#: R13 grid (6 synthetic regimes x 3 seeds) ned_shared_private converged on
#: 11/18 and lend on 16/18. At 300: 18/18 and 17/18 (ned 17 -> 18; flow_v
#: 18/18 at both), with predictive and latent metrics unchanged within noise
#: (scripts/deep_budget_check.py, docs/audit/evidence/deep_budget_check.csv).
DEFAULT_DEEP_ITERATIONS = 300
