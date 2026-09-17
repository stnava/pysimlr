"""Regression tests for the deep training loop's early-stopping baseline.

The similarity term is switched on at `warmup_epochs`, which discontinuously
raises the objective. `best_loss` used to be carried across that boundary, so
no post-warmup epoch could ever beat it and the patience counter incremented
every epoch -- training always stopped at warmup+patience, giving the
similarity objective almost no optimization at all.
"""
import numpy as np
import torch

from pysimlr.deep import ned_simr


def _coupled_views(n=160, k=3, seed=0):
    torch.manual_seed(seed)
    z = torch.randn(n, k)
    return [
        z @ torch.randn(k, 20) + 0.3 * torch.randn(n, 20),
        z @ torch.randn(k, 15) + 0.3 * torch.randn(n, 15),
    ]


def test_training_continues_past_warmup_plus_patience():
    warmup, patience, epochs = 10, 5, 80
    res = ned_simr(
        _coupled_views(), k=3, epochs=epochs, warmup_epochs=warmup,
        batch_size=64, patience=patience, verbose=False,
    )
    ran = len(res["loss_history"])
    # the old bug pinned this to exactly warmup + patience + 1
    assert ran > warmup + patience + 1, (
        f"training stopped at {ran} epochs, i.e. immediately after warmup "
        f"({warmup}) plus patience ({patience}) -- the early-stopping baseline "
        f"was not reset when the similarity term switched on"
    )
    assert ran <= epochs


def test_similarity_objective_actually_improves_after_warmup():
    warmup = 10
    res = ned_simr(
        _coupled_views(), k=3, epochs=80, warmup_epochs=warmup,
        batch_size=64, patience=5, verbose=False,
    )
    sim = res["sim_history"]
    assert len(sim) > warmup + 1, "no post-warmup epochs were run"
    # the similarity term is only active from `warmup` onward; it must make
    # measurable progress rather than being cut off after a couple of epochs
    assert sim[-1] < sim[warmup], (
        f"similarity loss did not improve after warmup: "
        f"{sim[warmup]:.4f} -> {sim[-1]:.4f}"
    )


def test_converged_iter_is_reported_and_consistent():
    res = ned_simr(
        _coupled_views(), k=3, epochs=40, warmup_epochs=5,
        batch_size=64, patience=5, verbose=False,
    )
    conv = res["converged_iter"]
    assert isinstance(conv, int) and conv > 0
    assert conv == len(res["loss_history"])


def test_full_epoch_budget_used_when_patience_is_large():
    epochs = 25
    res = ned_simr(
        _coupled_views(), k=3, epochs=epochs, warmup_epochs=5,
        batch_size=64, patience=epochs + 1, verbose=False,
    )
    assert len(res["loss_history"]) == epochs
