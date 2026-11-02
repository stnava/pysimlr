"""The OLEE contract: the loadings a user reads are orthogonal, and under
positivity each feature belongs to one component.

The contract sits on `result["loadings"]` (see `pysimlr.loadings`), the
identified summary, not on the encoder V, which is weakly identified. Checked
for the default fit on the planted linear case under both sign modes.
"""
import numpy as np
import pytest
import torch

from pysimlr import api
from pysimlr.benchmarks.synthetic_cases import build_case


@pytest.mark.parametrize("positivity", ["either", "positive"])
def test_default_fit_loadings_meet_the_olee_contract(positivity):
    c = build_case("linear", n_samples=300, seed=4, private_scale=0.5)
    views = [torch.as_tensor(np.asarray(x)).float() for x in c["data"]]
    k = np.asarray(c["true_u"]).shape[1]
    res = api.fit("simlr", views, k, positivity=positivity, random_state=4)
    for l in res["loadings"]:
        l = l.double()
        g = l.T @ l
        off = g - torch.diag(torch.diag(g))
        assert float(off.abs().max()) <= 1e-5 * float(torch.diag(g).max()), off
        if positivity == "positive":
            assert (l >= 0).all() and ((l > 0).sum(1) <= 1).all()
