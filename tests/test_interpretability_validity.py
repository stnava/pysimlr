"""The interpretability report's R-squared must not be an artifact of overfitting.

Every "alignment" and "attribution" figure came from `_fit_linear_map`, which
reported only an in-sample R-squared from an effectively unregularized fit
(l2=1e-6). With one free coefficient per predictor per target that approaches 1
by construction whenever the predictor is wide relative to the sample count --
measured 0.92 on pure noise at n=40, p=35 -- so it carried no evidence of a
real relationship. Cross-validated figures are now reported alongside.
"""
import numpy as np
import pytest
import torch

from pysimlr.interpretability import _fit_linear_map, _ridge_solve


def test_in_sample_r2_is_inflated_on_pure_noise_but_cv_is_not():
    torch.manual_seed(0)
    x = torch.randn(40, 35)      # wide: n barely exceeds p
    y = torch.randn(40, 3)       # independent of x
    fit = _fit_linear_map(x, y)
    assert fit["global_r2"] > 0.5, (
        "sanity check of the premise: the in-sample fit should look good here"
    )
    assert fit["global_r2_cv"] < 0.1, (
        f"cross-validated R2 {fit['global_r2_cv']:.3f} should expose that there "
        f"is no real relationship"
    )


def test_cv_and_in_sample_agree_when_the_signal_is_real():
    torch.manual_seed(1)
    x = torch.randn(300, 10)
    y = x[:, :3] @ torch.randn(3, 3) + 0.1 * torch.randn(300, 3)
    fit = _fit_linear_map(x, y)
    assert fit["global_r2"] > 0.95
    assert fit["global_r2_cv"] > 0.95
    assert abs(fit["global_r2"] - fit["global_r2_cv"]) < 0.05


def test_cv_r2_is_reported_per_target():
    torch.manual_seed(2)
    x = torch.randn(200, 6)
    y = torch.stack([x[:, 0], torch.randn(200)], dim=1)   # target 0 real, 1 noise
    fit = _fit_linear_map(x, y)
    cv = fit["r2_per_target_cv"]
    assert cv.shape == (2,)
    assert float(cv[0]) > 0.9, "the recoverable target should score high"
    assert float(cv[1]) < 0.2, "the noise target should not"


def test_cv_is_disabled_gracefully_for_tiny_samples():
    fit = _fit_linear_map(torch.randn(3, 2), torch.randn(3, 1))
    assert np.isnan(fit["global_r2_cv"])
    assert torch.isnan(fit["r2_per_target_cv"]).all()
    assert np.isfinite(fit["global_r2"])


def test_rank_deficient_design_is_solved_not_silently_mangled():
    """torch.linalg.lstsq's default CPU driver assumes full rank and ignores
    rcond, so the old fallback returned unreliable coefficients without error."""
    torch.manual_seed(3)
    x = torch.randn(60, 6)
    x[:, 1] = x[:, 0] * 2.0            # exact collinearity
    y = x[:, :2] @ torch.randn(2, 2)
    fit = _fit_linear_map(x, y)
    assert torch.isfinite(fit["coefficients"]).all()
    assert fit["global_r2"] > 0.99
    assert fit["global_r2_cv"] > 0.9


def test_ridge_solve_matches_the_closed_form():
    torch.manual_seed(4)
    xc = torch.randn(50, 4, dtype=torch.float64)
    yc = torch.randn(50, 2, dtype=torch.float64)
    l2 = 0.3
    got = _ridge_solve(xc, yc, l2)
    want = torch.linalg.inv(xc.T @ xc + l2 * torch.eye(4, dtype=torch.float64)) @ (xc.T @ yc)
    assert torch.allclose(got, want, atol=1e-10)


def test_report_surfaces_cv_alongside_in_sample():
    from pysimlr.deep import ned_simr

    torch.manual_seed(5)
    z = torch.randn(120, 3)
    x = [z @ torch.randn(3, 12) + 0.2 * torch.randn(120, 12),
         z @ torch.randn(3, 9) + 0.2 * torch.randn(120, 9)]
    res = ned_simr(x, k=3, epochs=12, warmup_epochs=3, verbose=False)
    rep = res["interpretability"]
    for m in rep["deep_layer_alignment"]["modalities"]:
        assert "global_r2" in m and "global_r2_cv" in m
        assert "r2_per_target_cv" in m
    for m in rep["shared_to_first_layer"]["per_modality"]:
        assert "global_r2" in m and "global_r2_cv" in m
        assert "r2_per_shared_dimension_cv" in m
