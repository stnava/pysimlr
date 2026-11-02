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
#: from the comparison recorded in that function's docstring.
DEFAULT_COLLAPSE_THRESHOLD = 0.3


#: `simlr` stopping. At tol = 1e-6 the plateau rule almost never fired: 0/20
#: default fits converged within 100 sweeps and 14/20 within 300, every
#: non-converged trace still creeping monotonically by 1e-6 to 1e-5 per sweep.
#: Fits stopped at 1e-4 match the 300-sweep fits to < 0.001 latent R^2 with
#: support F1, frame defect and overlap unchanged
#: (scripts/linear_tol_selection.py). 1e-4 is also the ConvergenceMonitor and
#: flow default. 300 sweeps is headroom for harder problems; at 1e-4 the
#: measured fits stop after ~73.
DEFAULT_SIMLR_TOL = 1e-4
DEFAULT_SIMLR_ITERATIONS = 300
