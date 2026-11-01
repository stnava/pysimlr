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
