"""The default fit meets the OLEE contract: near-orthogonal, low-overlap loadings.

`nsa_w = 0.5` (NSA-Flow paper) drives both the retraction and the calibrated
orthogonality-defect term of the objective. On the planted linear case under
non-negativity (`scripts/constraint_selection.py`, seed 4, default budget) the
default fit measured frame defect 0.0000 and max |V|-column overlap 0.0039;
the same fit with the defect term removed measured 0.0240 and 0.2200. The
thresholds sit between the two, so this fails if the defect term stops acting.
"""
import numpy as np
import torch

from pysimlr.benchmarks.synthetic_cases import build_case
from pysimlr.simlr import simlr
from pysimlr.utils import orthogonality_defect


def _max_abs_column_overlap(v):
    a = v.detach().abs().double()
    a = a / a.norm(dim=0, keepdim=True).clamp_min(1e-12)
    g = a.T @ a
    return float((g - torch.diag(torch.diag(g))).max())


def test_default_fit_meets_the_olee_contract():
    c = build_case("linear", n_samples=300, seed=4, private_scale=0.5)
    views = [torch.as_tensor(np.asarray(x)).float() for x in c["data"]]
    k = np.asarray(c["true_u"]).shape[1]
    res = simlr(views, k=k, positivity="positive", random_state=4, verbose=False)
    for v in res["v"]:
        assert float(orthogonality_defect(v.float(), normalized=True)) < 0.005
        assert _max_abs_column_overlap(v) < 0.05
