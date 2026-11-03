"""Every public function in pysimlr.plotting: returns a Figure (feature_table: a DataFrame),
honours ax where it is single-panel, and plots the quantity it names."""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch

import pysimlr
from pysimlr import plotting


@pytest.fixture(scope="module")
def fits():
    rng = np.random.default_rng(0); n = 90
    z = np.abs(rng.standard_normal((n, 2)))
    views = [torch.tensor(z @ rng.random((2, p)) + 0.1 * rng.standard_normal((n, p))).float() for p in (12, 9)]
    tr, te = [x[:70] for x in views], [x[70:] for x in views]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sig = pysimlr.fit("simlr", tr, 2, learning_rate=0.01, random_state=0)
        pos = pysimlr.fit("simlr", tr, 2, positivity="positive", learning_rate=0.01, random_state=0)
        short = pysimlr.fit("simlr", tr, 2, iterations=1, learning_rate=0.01, random_state=0)
        flow = pysimlr.fit("flow_v", tr, 2, iterations=4, random_state=0)
        pca = pysimlr.fit("pca", tr, 2, random_state=0)
        inf = pysimlr.loading_inference("simlr", tr, 2, n_subsamples=4, n_permutations=2,
                                        learning_rate=0.01, random_state=0)
        rank = pysimlr.select_rank("simlr", tr, [1, 2], n_subsamples=3, learning_rate=0.01, random_state=0)
    bench = pd.DataFrame({"model": ["a", "a", "b", "b"] * 2, "sparseness": [0.3, 0.5] * 4,
                          "recovery": rng.random(8)})
    splits = pd.DataFrame({"method": np.repeat(["ridge", "simlr", "ned"], 6), "split": np.tile(range(6), 3),
                           "score": np.r_[0.10 + 0.01 * rng.random(6), 0.20 + 0.01 * rng.random(6),
                                          0.10 + 0.05 * rng.standard_normal(6)]})
    return dict(tr=tr, te=te, z=z[:70], sig=sig, pos=pos, short=short, flow=flow, pca=pca, inf=inf,
                rank=rank, pred=pysimlr.predict(sig, te), bench=bench, splits=splits)


CALLS = {
    "plot_convergence": lambda f: (f["flow"],),
    "plot_basis_diagnostics": lambda f: (f["sig"],),
    "plot_rank_selection": lambda f: (f["rank"],),
    "plot_latent_coupling": lambda f: (f["sig"],),
    "plot_modality_weights": lambda f: (f["flow"],),
    "plot_flow_warping": lambda f: (f["flow"], f["tr"]),
    "plot_view_correlations": lambda f: (f["tr"],),
    "plot_top_features": lambda f: (f["sig"],),
    "plot_feature_groups": lambda f: (f["sig"], ["a"] * 6 + ["b"] * 6),
    "plot_loadings": lambda f: (f["pos"],),
    "plot_variance_explained": lambda f: (f["sig"], f["tr"]),
    "plot_scores": lambda f: (f["sig"],),
    "plot_latent_recovery": lambda f: (f["sig"], f["z"]),
    "plot_fit_summary": lambda f: (f["sig"], f["tr"]),
    "plot_benchmark": lambda f: (f["bench"], "sparseness", "recovery"),
    "plot_benchmark_uncertainty": lambda f: (f["splits"],),
    "plot_predictions": lambda f: (f["z"][:, 0], f["z"][:, 1]),
}
SINGLE_PANEL = {"plot_convergence", "plot_rank_selection", "plot_view_correlations", "plot_feature_groups",
                "plot_loadings", "plot_scores", "plot_latent_recovery", "plot_benchmark", "plot_predictions"}
FUSED = {"plot_fused_coefficients", "plot_coefficient_paths", "plot_fused_ridge_cv"}   # test_fused_ridge


def test_every_public_function_is_covered():
    assert set(plotting.__all__) == set(CALLS) | FUSED | {"feature_table"}


@pytest.mark.parametrize("name", sorted(CALLS))
def test_plot_returns_a_figure(name, fits):
    fig = getattr(pysimlr, name)(*CALLS[name](fits))
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


@pytest.mark.parametrize("name", sorted(SINGLE_PANEL))
def test_single_panel_plots_draw_into_a_given_axis(name, fits):
    fig, ax = plt.subplots()
    assert getattr(pysimlr, name)(*CALLS[name](fits), ax=ax) is fig
    plt.close(fig)


def test_feature_table_is_the_rotated_structure_with_inference(fits):
    res, inf = fits["sig"], fits["inf"]
    tab = pysimlr.feature_table(res, inf)
    assert len(tab) == sum(x.shape[1] for x in fits["tr"]) * 2
    for m, C in enumerate(res["structure"]):
        for j in range(2):
            d = tab[(tab.view == m) & (tab.component == j + 1)]
            assert np.allclose(d.structure, C[:, j].numpy(), atol=1e-6)
            assert np.array_equal(d.excludes_zero, (d.ci_low > 0) | (d.ci_high < 0))
    assert {"selected", "selection_frequency", "component_stability"} <= set(tab.columns)


def test_latent_coupling_is_corr_of_rotated_view_and_consensus_scores(fits):
    res = fits["sig"]
    R = res["loading_rotation"].double().numpy(); U = res["u"].double().numpy() @ R
    want = np.corrcoef((res["latents"][0].double().numpy() @ R)[:, 0], U[:, 0])[0, 1]
    fig = pysimlr.plot_latent_coupling(res, pred=fits["pred"])
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert f"{want:.2f}" == texts[0]
    assert len(fig.axes) >= 3                                       # training, held-out, view-vs-view
    plt.close(fig)
    fig = pysimlr.plot_latent_coupling(fits["pca"])                  # joint basis: says so
    assert "joint latent" in fig.axes[0].texts[0].get_text()
    plt.close(fig)


def test_convergence_cards_and_flags(fits):
    for key, needle in (("short", "NOT converged"), ("sig", "Converged"), ("pca", "no iteration history")):
        fig = pysimlr.plot_convergence(fits[key])
        ax = fig.axes[0]
        txt = ax.get_title() + " ".join(t.get_text() for t in ax.texts)
        assert needle in txt, (key, txt)
        plt.close(fig)


def test_top_features_mark_selection_and_stability(fits):
    fig = pysimlr.plot_top_features(fits["sig"], inf=fits["inf"], top_n=5)
    assert len(fig.axes) == 2 and "stability" in fig.axes[0].get_title()
    assert any(abs(l.get_xdata()[0]) == pytest.approx(fits["inf"]["null_max_abs"])
               for l in fig.axes[0].get_lines() if l.get_linestyle() == "--")
    plt.close(fig)


def test_flow_warping_of_simlr_is_the_identity(fits):
    fig = pysimlr.plot_flow_warping(fits["sig"], fits["tr"], components=(0,))
    assert "Spearman 1.00" in fig.axes[0].get_title()
    plt.close(fig)


def test_benchmark_uncertainty_flags_a_real_paired_difference(fits):
    fig = pysimlr.plot_benchmark_uncertainty(fits["splits"], baseline="ridge")
    solid = [c for c in fig.axes[1].collections if c.get_linestyle()[0][1] is None]
    assert len(solid) >= 1                                           # simlr - ridge excludes 0
    plt.close(fig)
    with pytest.raises(ValueError, match="baseline"):
        pysimlr.plot_benchmark_uncertainty(fits["splits"], baseline="nope")


def test_feature_groups_counts_need_inference(fits):
    with pytest.raises(ValueError, match="needs inf"):
        pysimlr.plot_feature_groups(fits["sig"], ["a"] * 12, statistic="n_selected")
    for stat in ("n_selected", "n_significant"):
        plt.close(pysimlr.plot_feature_groups(fits["sig"], ["a"] * 6 + ["b"] * 6, inf=fits["inf"], statistic=stat))


def test_rank_selection_plots_every_criterion(fits):
    for crit in pysimlr.RANK_CRITERIA:
        fig = pysimlr.plot_rank_selection(fits["rank"], criterion=crit)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)
    with pytest.raises(ValueError, match="criterion"):
        pysimlr.plot_rank_selection(fits["rank"], criterion="nope")


def test_old_plotting_modules_are_gone():
    for mod in ("pysimlr.viz", "pysimlr.visualization", "pysimlr.benchmarks.plotting"):
        with pytest.raises(ModuleNotFoundError):
            __import__(mod)
