"""
The retraction weight is capped everywhere, and the solver's self-report is kept.

Two gaps this closes. The encoder's `nsa_w` was handed straight to the backend
layer without passing through `_clamp_retraction_weight`, so `lend_simr` could
still be run at the degenerate w=1. And `_nsa_retract` read the solver's result
only to extract `Y`, discarding the stopping rule, the stationarity
certificate, the effective rank and which fidelity was chosen -- exactly the
fields that say whether a retraction can be trusted.
"""

import warnings

import pytest
import torch

from pysimlr import simlr
from pysimlr.nsa_backend import load_nsa_flow
from pysimlr.sparsification import (NSA_MAX_W, NSA_MIN_W, _nsa_retract,
                                    simlr_sparseness)


# ----------------------------------------------------- the cap reaches the layer

@pytest.mark.parametrize("requested,expected", [
    (1.0, NSA_MAX_W), (1.5, NSA_MAX_W), (2.0, NSA_MAX_W),
    (0.0, NSA_MIN_W), (-1.0, NSA_MIN_W),
    (0.5, 0.5), (0.9, 0.9),
])
def test_encoder_nsa_w_is_capped_before_it_reaches_the_backend(requested, expected):
    from pysimlr.deep import LENDNSAEncoder

    encoder = LENDNSAEncoder(20, latent_dim=3, nsa_w=requested,
                             positivity="positive", use_nsa=True)
    assert encoder.nsa_w == pytest.approx(expected)
    assert 0.0 < encoder.nsa_w < 1.0
    if encoder.nsa_linear is not None:
        assert float(encoder.nsa_linear.w) == pytest.approx(expected), (
            "the backend layer was constructed with an unclamped weight")


def test_deep_model_cannot_be_run_at_the_degenerate_weight():
    """`lend_simr(nsa_w=1.0)` must not reach w=1, where the fidelity term drops
    out and every scaled Stiefel matrix is optimal."""
    from pysimlr.deep import lend_simr

    torch.manual_seed(0)
    x = torch.randn(40, 12)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = lend_simr([x], k=2, epochs=2, batch_size=20, warmup_epochs=0,
                        use_nsa=True, nsa_w=1.0, verbose=False)
    model = res["model"]
    encoders = getattr(model, "linear_encoders", getattr(model, "encoders", []))
    assert encoders
    for enc in encoders:
        assert enc.nsa_w <= NSA_MAX_W
        if enc.nsa_linear is not None:
            assert float(enc.nsa_linear.w) <= NSA_MAX_W


# --------------------------------------------------------------- diagnostics

@pytest.mark.skipif(load_nsa_flow() is None, reason="no NSA-Flow backend installed")
def test_nsa_retract_reports_what_the_solver_said():
    torch.manual_seed(0)
    v = torch.rand(25, 3, dtype=torch.float64)

    diagnostics = {}
    out = _nsa_retract(v, w=0.5, nonneg=True, diagnostics=diagnostics)
    assert out is not None

    assert diagnostics["w"] == pytest.approx(0.5)
    # The fields that decide whether the result is trustworthy.
    assert diagnostics["stop_reason"] in {"grad_map", "line_search", "max_iter"}
    assert isinstance(diagnostics["converged"], bool)
    assert diagnostics["grad_map"] >= 0.0
    assert 0.0 < diagnostics["effective_rank"] <= 3.0


@pytest.mark.skipif(load_nsa_flow() is None, reason="no NSA-Flow backend installed")
def test_diagnostics_are_reported_even_when_the_result_is_rejected():
    """A rejected retraction is exactly when the caller most wants to know what
    the solver did, so the report must not be conditional on acceptance."""
    from unittest.mock import patch

    torch.manual_seed(0)
    v = torch.rand(10, 3, dtype=torch.float64)

    def degenerate(target, **kwargs):
        return {"Y": torch.zeros_like(target), "stop_reason": "max_iter",
                "converged": False, "grad_map": 1.0}

    diagnostics = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with patch("pysimlr.sparsification.load_nsa_flow", return_value=degenerate):
            out = _nsa_retract(v, w=0.5, nonneg=True, diagnostics=diagnostics)

    assert out is None, "an all-zero retraction should still be rejected"
    assert diagnostics["stop_reason"] == "max_iter"
    assert diagnostics["converged"] is False


@pytest.mark.skipif(load_nsa_flow() is None, reason="no NSA-Flow backend installed")
def test_simlr_carries_per_modality_retraction_diagnostics():
    torch.manual_seed(0)
    mats = [torch.randn(80, 30), torch.randn(80, 24)]
    res = simlr(mats, k=3, iterations=8, constraint="orthox0.5",
                positivity="positive", verbose=False)

    assert "v_retraction" in res
    assert len(res["v_retraction"]) == len(mats)
    for i, d in enumerate(res["v_retraction"]):
        assert d, f"modality {i} reported no diagnostics"
        assert d["stop_reason"] in {"grad_map", "line_search", "max_iter"}
        assert d["w"] <= NSA_MAX_W
        assert d["fidelity_mode"] in {"anchor", "subspace"}


def test_diagnostics_are_optional_and_a_missing_backend_raises():
    """The out-dict is opt-in, and a missing backend leaves it empty rather
    than raising or inventing values."""
    from unittest.mock import patch

    torch.manual_seed(0)
    v = torch.randn(20, 3)

    # No dict passed: must not raise.
    simlr_sparseness(v, constraint_type="ortho", positivity="positive",
                     constraint_weight=0.5)

    # With the backend gone there is no projection to report on, and no
    # substitute is applied, so the call raises rather than returning a basis
    # produced by a different operator with an empty diagnostics dict.
    diagnostics = {}
    with patch("pysimlr.sparsification.load_nsa_flow", return_value=None):
        with pytest.raises(ImportError, match="requires the NSA-Flow backend"):
            simlr_sparseness(v, constraint_type="ortho", positivity="positive",
                             constraint_weight=0.5,
                             retraction_diagnostics=diagnostics)
    assert diagnostics == {}


if __name__ == "__main__":
    pytest.main([__file__])
