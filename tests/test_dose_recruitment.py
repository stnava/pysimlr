"""Tests for adaptive-N_eff continuous dose-response and multi-view recruitment."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest
from matplotlib.figure import Figure

import pysimlr

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _make_planted_recruitment_dataset(seed: int = 123) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = 260

    # Zero-inflated primary dose: 75 unexposed zeros + 185 positive counts across 1..10^4
    n_zero = 75
    n_pos = n - n_zero
    log_pos = rng.uniform(0.35, 4.10, size=n_pos)
    primary_dose = np.concatenate([np.zeros(n_zero), np.power(10.0, log_pos) - 1.0])
    perm = rng.permutation(n)
    primary_dose = primary_dose[perm]
    x_log = np.log10(primary_dose + 1.0)

    # Co-occurring nuisance exposure correlated with primary_dose AND strongly driving View_Null
    nuisance_log = 0.45 * x_log + rng.normal(0.0, 0.90, size=n)
    secondary_dose = np.maximum(0.0, np.power(10.0, np.clip(nuisance_log + 2.0, 0.0, 5.5)) - 1.0)

    # View_Early turns on rapidly around dose >= 25 (log10(26) ~ 1.41)
    early_signal = 1.35 / (1.0 + np.exp(-4.2 * (x_log - 1.35))) - 1.35 / (1.0 + np.exp(4.2 * 1.35))
    view_early = early_signal + rng.normal(0.0, 0.42, size=n)

    # View_Late turns on later around dose >= 450 (log10(451) ~ 2.65)
    late_signal = 1.45 / (1.0 + np.exp(-4.5 * (x_log - 2.65))) - 1.45 / (1.0 + np.exp(4.5 * 2.65))
    view_late = late_signal + rng.normal(0.0, 0.42, size=n)

    # View_Null has ZERO true effect of primary_dose, only confounding from nuisance_log
    view_null = 0.85 * nuisance_log + rng.normal(0.0, 0.48, size=n)

    return {
        "primary_dose": primary_dose,
        "secondary_dose": secondary_dose,
        "nuisance_log": nuisance_log,
        "view_early": view_early,
        "view_late": view_late,
        "view_null": view_null,
    }


def test_planted_sequential_recruitment_and_coexposure_adjustment() -> None:
    data = _make_planted_recruitment_dataset(seed=2026)
    rec = pysimlr.modality_dose_recruitment(
        scores={
            "Early_Vascular": data["view_early"],
            "Late_Structural": data["view_late"],
            "Confounded_Null": data["view_null"],
        },
        doses={
            "Primary_Exposure": data["primary_dose"],
            "Secondary_Exposure": data["secondary_dose"],
        },
        covariates={
            "Primary_Exposure": data["nuisance_log"],
            "Secondary_Exposure": np.log10(data["primary_dose"] + 1.0),
        },
        standardize_scores=True,
        target_neff=32.0,
        min_sigma=0.22,
        n_grid=90,
        n_boot=220,
        seed=42,
    )

    assert rec["kind"] == "modality_dose_recruitment"
    c_early = rec["curves"]["Primary_Exposure"]["Early_Vascular"]
    c_late = rec["curves"]["Primary_Exposure"]["Late_Structural"]
    c_null = rec["curves"]["Primary_Exposure"]["Confounded_Null"]

    assert c_early["valid"] and c_late["valid"] and c_null["valid"]
    assert c_early["has_zero_mass"] is True
    assert c_early["n_zero"] == 75

    # Sequential onset ordering: Early_Vascular crosses well before Late_Structural
    assert np.isfinite(c_early["onset_pointwise"])
    assert np.isfinite(c_late["onset_pointwise"])
    assert c_early["onset_pointwise"] < c_late["onset_pointwise"]
    assert c_early["onset_pointwise"] < 120.0
    assert c_late["onset_pointwise"] > 120.0

    # Simultaneous sup-t critical value is strictly wider than pointwise z_0.95 = 1.96
    assert c_early["c_simultaneous"] > c_early["z_pointwise"]
    assert np.isfinite(c_early["onset_simultaneous"])
    assert c_early["onset_simultaneous"] >= c_early["onset_pointwise"]

    # Co-exposure adjustment removes spurious primary onset on Confounded_Null
    assert np.isnan(c_null["onset_simultaneous"])
    assert abs(c_null["peak_delta"]) < 0.35


def test_adaptive_neff_accuracy_and_unscanned_support_masking() -> None:
    data = _make_planted_recruitment_dataset(seed=77)
    dose = data["primary_dose"]
    y_full = data["view_early"].copy()

    # Unconstrained min_sigma hits target_neff closely across active interior grid points
    curve = pysimlr.adaptive_local_linear_curve(
        dose,
        y_full,
        min_sigma=0.05,
        max_sigma=3.0,
        target_neff=30.0,
        n_grid=80,
        n_boot=80,
        seed=11,
    )
    neff_sup = curve["neff"][curve["in_support"]]
    assert abs(float(np.median(neff_sup)) - 30.0) < 0.5

    # Mask all observations above dose > 250 as unscanned (NaN): upper grid must NOT extrapolate
    y_partial = np.where(dose <= 250.0, y_full, np.nan)
    explicit_grid = np.linspace(0.30, 4.00, 80)
    curve_masked = pysimlr.adaptive_local_linear_curve(
        dose,
        y_partial,
        grid=explicit_grid,
        target_neff=24.0,
        n_boot=60,
        seed=11,
    )
    high_tail = explicit_grid > 2.60
    assert np.all(~curve_masked["in_support"][high_tail])
    assert np.all(np.isnan(curve_masked["delta"][high_tail]))


def test_determinism_validation_and_plot_dose_recruitment() -> None:
    data = _make_planted_recruitment_dataset(seed=99)
    c1 = pysimlr.adaptive_local_linear_curve(
        data["primary_dose"], data["view_early"], n_grid=50, n_boot=60, seed=19
    )
    c2 = pysimlr.adaptive_local_linear_curve(
        data["primary_dose"], data["view_early"], n_grid=50, n_boot=60, seed=19
    )
    np.testing.assert_allclose(c1["delta"], c2["delta"], equal_nan=True)
    np.testing.assert_allclose(c1["sim_lo"], c2["sim_lo"], equal_nan=True)

    # Validation errors on malformed inputs
    with pytest.raises(ValueError, match="non-negative"):
        pysimlr.adaptive_local_linear_curve(np.array([-2.0] * 30), np.zeros(30), log_transform=True)
    with pytest.raises(ValueError, match="18 complete observations"):
        pysimlr.adaptive_local_linear_curve(np.arange(10, dtype=float), np.zeros(10))

    # Partial view with < 18 observations inside modality_dose_recruitment is marked valid=False cleanly
    sparse_view = np.full_like(data["view_early"], np.nan)
    sparse_view[:8] = 1.0
    rec = pysimlr.modality_dose_recruitment(
        scores={
            "Consensus_Score": data["view_early"],
            "Late_View": data["view_late"],
            "Unscanned_Substudy": sparse_view,
        },
        doses={
            "Primary": data["primary_dose"],
            "Secondary": data["secondary_dose"],
        },
        n_grid=45,
        n_boot=50,
        seed=5,
    )
    assert rec["curves"]["Primary"]["Unscanned_Substudy"]["valid"] is False

    fig = pysimlr.plot_dose_recruitment(
        rec,
        companion_dose={"Secondary": "Primary"},
        dose_units="events",
        title="Synthetic Dose-Recruitment Smoke Verification",
    )
    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 6
    plt.close(fig)
