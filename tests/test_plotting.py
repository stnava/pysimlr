"""Every public plot in pysimlr.plotting returns a Figure, draws into a given ax, and plots
the interpretable quantity it names."""
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
    rng = np.random.default_rng(0); n = 80
    z = np.abs(rng.standard_normal((n, 2)))
    views = [torch.tensor(z @ rng.random((2, p)) + 0.1 * rng.standard_normal((n, p))).float()
             for p in (12, 9)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pos = pysimlr.fit("simlr", views, 2, positivity="positive", random_state=0)
        sig = pysimlr.fit("simlr", views, 2, random_state=0)
        short = pysimlr.fit("simlr", views, 2, iterations=1, learning_rate=0.01, random_state=0)
        conv = pysimlr.fit("simlr", views, 2, learning_rate=0.01, random_state=0)
        deep = pysimlr.fit("ned", views, 2, iterations=3, random_state=0)
        inf = pysimlr.loading_inference("simlr", views, 2, n_subsamples=4, random_state=0)
        rank = pysimlr.select_rank("simlr", views, [1, 2], n_subsamples=3, random_state=0)
    bench = pd.DataFrame({"model": ["a", "a", "b", "b"] * 2, "sparsity": [0.3, 0.5] * 4,
                          "recovery": rng.random(8), "test_r2": rng.random(8)})
    return dict(views=views, z=z, pos=pos, sig=sig, short=short, conv=conv, deep=deep, inf=inf, rank=rank,
                bench=bench)


CALLS = {
    "plot_loadings": lambda f: (f["pos"],),
    "plot_structure_intervals": lambda f: (f["inf"],),
    "plot_component_stability": lambda f: (f["rank"],),
    "plot_scores": lambda f: (f["sig"],),
    "plot_view_agreement": lambda f: (f["sig"],),
    "plot_convergence": lambda f: (f["deep"],),
    "plot_view_correlations": lambda f: (f["views"],),
    "plot_latent_recovery": lambda f: (f["sig"], f["z"]),
    "plot_benchmark": lambda f: (f["bench"], "sparsity", "recovery"),
    "plot_predictions": lambda f: (f["z"][:, 0], f["z"][:, 1]),
}


def test_every_public_plot_is_covered():
    fused = {"plot_fused_coefficients", "plot_coefficient_paths", "plot_fused_ridge_cv"}  # test_fused_ridge
    assert set(plotting.__all__) == set(CALLS) | fused


@pytest.mark.parametrize("name", sorted(CALLS))
def test_plot_returns_a_figure_and_draws_into_a_given_axis(name, fits):
    fn = getattr(pysimlr, name)
    fig = fn(*CALLS[name](fits))
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)
    fig2, ax = plt.subplots()
    assert fn(*CALLS[name](fits), ax=ax) is fig2
    plt.close(fig2)


def test_scores_are_the_rotated_scores(fits):
    res = fits["sig"]
    fig = pysimlr.plot_scores(res)
    pts = fig.axes[0].collections[0].get_offsets()
    want = (res["u"].double() @ res["loading_rotation"].double()).numpy()[:, :2]
    assert np.allclose(np.asarray(pts), want, atol=1e-5)
    plt.close(fig)


def test_held_out_scores_are_projected_and_rotated(fits):
    res, views = fits["sig"], fits["views"]
    fig = pysimlr.plot_scores(res, new_views=[v[:10] for v in views])
    held = np.asarray(fig.axes[0].collections[-1].get_offsets())
    want = (pysimlr.predict(res, [v[:10] for v in views])["u"].double()
            @ res["loading_rotation"].double()).numpy()[:, :2]
    assert np.allclose(held, want, atol=1e-5)
    plt.close(fig)


def test_convergence_plot_flags_an_unconverged_fit(fits):
    fig = pysimlr.plot_convergence(fits["short"])
    assert "NOT converged" in fig.axes[0].get_title()
    plt.close(fig)
    assert fits["conv"]["convergence"]["converged"]
    fig = pysimlr.plot_convergence(fits["conv"])
    assert "NOT converged" not in fig.axes[0].get_title()
    plt.close(fig)


def test_benchmark_kinds_and_missing_columns(fits):
    for kind in ("line", "scatter", "bar"):
        plt.close(pysimlr.plot_benchmark(fits["bench"], "sparsity", "test_r2", kind=kind))
    with pytest.raises(KeyError, match="missing columns"):
        pysimlr.plot_benchmark(fits["bench"], "sparsity", "nope")


def test_old_plotting_modules_are_gone():
    for mod in ("pysimlr.viz", "pysimlr.visualization", "pysimlr.benchmarks.plotting"):
        with pytest.raises(ModuleNotFoundError):
            __import__(mod)
