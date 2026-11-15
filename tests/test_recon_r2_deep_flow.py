import numpy as np
import pytest
import torch

import pysimlr
from pysimlr.flows import flow_simr


def test_recon_r2_accepted_on_linear_first_layer_models_and_refused_without_one():
    """Minimal reproduction & architectural contract for recon_r2 across methods."""
    rng = np.random.default_rng(0)
    z = rng.standard_normal((60, 1))
    views = [
        z @ rng.standard_normal((1, 8)) + 0.2 * rng.standard_normal((60, 8)),
        z @ rng.standard_normal((1, 10)) + 0.2 * rng.standard_normal((60, 10)),
        z @ rng.standard_normal((1, 6)) + 0.2 * rng.standard_normal((60, 6)),
    ]

    for k in (1, 2):
        for method in ("simlr", "lend", "flow_v"):
            res = pysimlr.fit(
                method,
                views,
                k=k,
                energy_type="recon_r2",
                mixing_algorithm="svd",
                iterations=5,
                random_state=0,
            )
            assert res["u"].shape == (60, k)
            assert res["parameters"]["used"]["energy_type"] == "recon_r2"
            assert not res["parameters"]["ignored"]
            assert np.isfinite(res["convergence"]["grad_map"])

            pred = pysimlr.predict(res, views)
            assert pred["u"].shape == (60, k)
            assert bool(torch.isfinite(torch.as_tensor(pred["u"])).all())

    for method in ("ned", "ned_shared_private"):
        with pytest.raises(ValueError, match="no linear first layer"):
            pysimlr.fit(method, views, k=1, energy_type="recon_r2", iterations=2, random_state=0)

    with pytest.raises(ValueError, match="no linear first layer"):
        flow_simr(views, k=1, energy_type="recon_r2", iterations=2, random_state=0)

    # Degenerate unprofiled 'recon' stays refused as a fit objective everywhere
    for method in ("simlr", "lend", "flow_v"):
        with pytest.raises(ValueError, match="recon_r2"):
            pysimlr.fit(method, views, k=1, energy_type="recon", iterations=2, random_state=0)
    with pytest.raises(ValueError, match="no linear first layer"):
        pysimlr.fit("ned", views, k=1, energy_type="recon", iterations=2, random_state=0)


def test_describe_method_documents_recon_r2_and_sim_weight():
    """describe_method('lend') and describe_method('flow_v') explain recon_r2 + sim_weight."""
    for method in ("lend", "flow_v"):
        desc = pysimlr.describe_method(method)["description"]
        assert "recon_r2" in desc
        assert "sim_weight" in desc


def test_lend_recon_r2_matches_simlr_on_planted_signal():
    """With a linear decoder and dominant sim_weight, lend + recon_r2 recovers the same consensus as simlr."""
    rng = np.random.default_rng(42)
    n = 120
    z_true = rng.standard_normal((n, 1))
    views = [
        z_true @ rng.standard_normal((1, 10)) + 0.15 * rng.standard_normal((n, 10)),
        z_true @ rng.standard_normal((1, 12)) + 0.15 * rng.standard_normal((n, 12)),
        z_true @ rng.standard_normal((1, 8)) + 0.15 * rng.standard_normal((n, 8)),
    ]

    res_lin = pysimlr.fit(
        "simlr",
        views,
        k=1,
        energy_type="recon_r2",
        mixing_algorithm="svd",
        sparseness=None,
        iterations=30,
        random_state=0,
    )
    res_lend = pysimlr.fit(
        "lend",
        views,
        k=1,
        energy_type="recon_r2",
        mixing_algorithm="svd",
        hidden_dims=(),
        dropout=0.0,
        sparseness=None,
        sim_weight=5.0,
        warmup_epochs=0,
        learning_rate=1e-2,
        batch_size=n,
        iterations=80,
        random_state=0,
    )

    u_lin = np.asarray(res_lin["u"]).ravel()
    u_lend = np.asarray(res_lend["u"]).ravel()
    corr = abs(float(np.corrcoef(u_lin, u_lend)[0, 1]))
    assert corr > 0.95, f"Expected |r| > 0.95 between lend and simlr on recon_r2, got {corr:.4f}"


def test_recon_r2_minibatch_loo_positivity_no_nsa_and_confounds():
    """Verify remainder batches, topology='loo', positive+sparse STE basis, use_nsa=False, and confounds."""
    rng = np.random.default_rng(7)
    n = 75  # batch_size=64 leaves an 11-row final batch (>= MIN_ROWS_FOR_MOMENT)
    conf = rng.standard_normal((n, 2))
    z = rng.standard_normal((n, 2))
    views = [
        np.abs(z @ rng.uniform(0.5, 1.5, (2, 9))) + conf @ rng.standard_normal((2, 9)) + 0.1 * rng.uniform(size=(n, 9)),
        np.abs(z @ rng.uniform(0.5, 1.5, (2, 11))) + conf @ rng.standard_normal((2, 11)) + 0.1 * rng.uniform(size=(n, 11)),
        np.abs(z @ rng.uniform(0.5, 1.5, (2, 7))) + conf @ rng.standard_normal((2, 7)) + 0.1 * rng.uniform(size=(n, 7)),
    ]

    for method in ("lend", "flow_v"):
        res = pysimlr.fit(
            method,
            views,
            k=2,
            energy_type="recon_r2",
            topology="loo",
            positivity="positive",
            sparseness=0.5,
            use_nsa=False,
            batch_size=64,
            warmup_epochs=0,
            iterations=6,
            confounds=conf,
            random_state=3,
        )
        assert res["parameters"]["used"]["energy_type"] == "recon_r2"
        assert res["parameters"]["used"]["topology"] == "loo"
        assert res["parameters"]["used"]["positivity"] == "positive"
        assert np.isfinite(res["convergence"]["grad_map"])
        for v_m in res["v"]:
            v_arr = v_m.detach().cpu().numpy() if isinstance(v_m, torch.Tensor) else np.asarray(v_m)
            assert float(v_arr.min()) >= -1e-6

        pred = pysimlr.predict(res, views)
        assert pred["u"].shape == (n, 2)
        assert bool(torch.isfinite(torch.as_tensor(pred["u"])).all())
        assert "u_sink" in pred
