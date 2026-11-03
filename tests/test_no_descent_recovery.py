"""A sweep in which every view's step is rejected reduces the learning rate and retries,
instead of stopping the fit (simlr.NO_DESCENT_LR_FLOOR)."""
import sys
import warnings

import numpy as np
import torch

import pysimlr


def _toy(seed):
    rng = np.random.default_rng(seed); z = np.abs(rng.standard_normal((80, 2)))
    return [torch.tensor(z @ rng.random((2, p)) + 0.1 * rng.standard_normal((80, p))).float()
            for p in (12, 9)]


def test_default_fit_on_small_data_recovers_instead_of_stopping():
    """Before: 7 of 10 such fits stopped with no_descent after a few sweeps."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", pysimlr.ConvergenceWarning)
        r = pysimlr.fit("simlr", _toy(0), 2, positivity="positive", random_state=0)
    c = r["convergence"]
    assert c["converged"] and c["learning_rate_reductions"] >= 1
    assert c["learning_rate_final"] < r["learning_rate"]


def test_without_the_retry_the_same_fit_stops_with_no_descent(monkeypatch):
    import pysimlr.simlr  # noqa: F401
    monkeypatch.setattr(sys.modules["pysimlr.simlr"], "NO_DESCENT_LR_FLOOR", float("inf"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = pysimlr.fit("simlr", _toy(0), 2, positivity="positive", random_state=0)
    assert r["stop_reason"] == "no_descent" and r["convergence"]["learning_rate_reductions"] == 0


def test_every_fit_reports_its_learning_rate():
    """A fixed-rate fit had no `learning_rate` key; only "auto" fits reported one."""
    import numpy as np
    from pysimlr.simlr import simlr
    g = np.random.default_rng(0)
    z = g.normal(size=(60, 2))
    views = [z @ g.normal(size=(2, p)) + 0.3 * g.normal(size=(60, p)) for p in (12, 10)]
    r = simlr(views, k=2, iterations=5, learning_rate=0.05, random_state=0)
    assert r["learning_rate"] == 0.05
    assert r["convergence"]["learning_rate_final"] <= 0.05
