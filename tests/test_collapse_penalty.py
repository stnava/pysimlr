"""Dimensional collapse is penalised on each view's scores (`utils.correlation_penalty`).

The penalty used to act on the mixed consensus `u`, which `newton` mixing
orthogonalises, so it read ~0 while the scores had collapsed; and "nc" scores a
collapsed latent as well as a full-rank one (-0.9947 vs -0.9945). Measured
before the fix: linear "nc" + hybrid_adam + newton on narrow views gave score
columns correlated -0.98 and outcome R^2 0.30.

NED under "nc" + loo + newton collapsed (latent correlation +-0.97) on seeds
1 and 2; without any collapse penalty it collapses on 6 of 20 (data seed,
start seed) pairs, with this one on 0 of 20.
"""
import warnings

import torch

from pysimlr.utils import correlation_penalty


def test_correlation_penalty_is_zero_for_decorrelated_and_one_for_collapsed():
    g = torch.Generator().manual_seed(0)
    q, _ = torch.linalg.qr(torch.randn(50, 3, generator=g, dtype=torch.float64))
    q = q - q.mean(0)
    assert float(correlation_penalty(q)) < 0.05
    # moderate correlation below the threshold is free
    a = torch.randn(400, 1, generator=g); b = 0.25 * a + torch.randn(400, 1, generator=g)
    assert float(correlation_penalty(torch.cat([a, b], 1), threshold=0.3)) == 0.0
    col = torch.randn(50, 1, generator=g)
    # three collinear columns: 3 pairs, each adding 1
    assert abs(float(correlation_penalty(torch.cat([col, 2 * col, -col], 1))) - 3.0) < 1e-6
    assert abs(float(correlation_penalty(10 * q)) - float(correlation_penalty(q))) < 1e-9


def _narrow():
    torch.manual_seed(0)
    n, k = 300, 2
    z = torch.randn(n, k)
    w = torch.randn(k, 1)
    x = [z @ torch.randn(k, 5) + 0.3 * torch.randn(n, 5) for _ in range(2)]
    return x, (z @ w).squeeze(1)


def _outcome_r2(x, y, v):
    s = torch.cat([m @ b for m, b in zip(x, v)], 1)
    s = (s - s.mean(0)) / (s.std(0) + 1e-8)
    d = torch.cat([s, torch.ones(len(y), 1)], 1)
    beta = torch.linalg.lstsq(d, y.unsqueeze(1)).solution
    return 1 - float(((y - (d @ beta).squeeze(1)) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def test_linear_nc_newton_keeps_both_components_on_narrow_views():
    from pysimlr.simlr import simlr
    x, y = _narrow()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = simlr(x, k=2, iterations=30, positivity="positive", sparseness=0.5, energy_type="nc",
                  optimizer_type="hybrid_adam", mixing_algorithm="newton", random_state=0)
    assert _outcome_r2(x, y, r["v"]) > 0.9
    scores = x[0] @ r["v"][0]
    assert abs(float(torch.corrcoef(scores.T)[0, 1])) < 0.9


def test_ned_nc_newton_does_not_collapse():
    """With the latent correlation penalty: 0 of 20 (data seed, start seed)
    pairs collapse; without any penalty, 6 of 20 (|r| 0.76-0.98)."""
    from pysimlr.deep import ned_simr
    for seed in range(1, 6):
        torch.manual_seed(seed)
        n = 200
        u = torch.randn(n, 2)
        x = [u @ torch.randn(2, 20) + 0.1 * torch.randn(n, 20) for _ in range(2)]
        x = [(a - a.mean(0)) / (a.std(0) + 1e-6) for a in x]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = ned_simr(x, k=2, iterations=100, batch_size=32, warmup_epochs=10, energy_type="nc",
                         mixing_algorithm="newton", random_state=seed)
        assert abs(float(torch.corrcoef(r["u"].T)[0, 1])) < 0.5, seed
