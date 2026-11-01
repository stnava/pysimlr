"""Package-wide defaults, defined once.

Every entry point, model constructor and helper that has an ``energy_type`` or
an NSA-Flow weight takes its default from here, so a method cannot quietly
disagree with the others. Before 0.3.0 the linear path defaulted to
``energy_type="acc"`` and ``nsa_w=0.5`` while the deep and flow paths used
``"regression"`` and ``0.1``, with no recorded reason for the difference.
"""

#: Similarity term of the objective (see `similarity.SIMILARITY`): the absolute
#: cross-covariance between each view's projection and the shared consensus.
DEFAULT_ENERGY_TYPE = "acc"

#: NSA-Flow trade-off weight w in (0, 1) between fidelity to the unconstrained
#: basis and orthogonality. 0.5 sits inside the stable conditioning plateau
#: (README, "nsa_flow Trade-off Weight").
DEFAULT_NSA_W = 0.5

#: Consensus each view is trained against. "loo": view m is scored against the
#: consensus of the other views only, so its own private variance cannot raise
#: its own objective. Under "star" the consensus includes the view itself; on a
#: generator whose per-view leading directions are private, `simlr` iteration
#: then does not improve on its PCA start (8-seed mean gain in least-squares latent R^2
#: -0.002), against +0.217 under "loo" (scripts/experiment_iteration_vs_init.py,
#: docs/audit/evidence). The reported shared latent and the prediction map are
#: the all-view consensus under either topology.
DEFAULT_TOPOLOGY = "loo"
