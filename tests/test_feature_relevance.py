"""`pysimlr.feature_relevance`: a hard relevance cutoff (permutation null, held-out replication,
refit stability, tiers) and its figures."""
import types

import numpy as np
import pytest
import torch

import pysimlr
from pysimlr.loadings import _relative_weight_basis, _relative_weight_shares, _zscore
from pysimlr.relevance import _perm_pvalues

FAST = dict(n_permutations=2000, n_permutations_test=2000, n_refits=0, n_jobs=1)


def _design(seed=0, n=500, p0=24, n_drivers=4, null_p=60):
    """View 0: ``n_drivers`` features carry the shared signal s, the rest are noise. Views 1 and 2
    carry s in every feature (they build the target). View 3: pure noise (the null view).
    Returns (train views, test views, s_train, s_test)."""
    g = np.random.default_rng(seed)
    s = g.standard_normal(n)

    def view(p, w):
        return torch.tensor(np.outer(s, w) + g.standard_normal((n, p))).float()
    w0 = np.zeros(p0); w0[:n_drivers] = 1.0
    xs = [view(p0, w0), view(12, g.uniform(0.6, 1.2, 12)), view(10, g.uniform(0.6, 1.2, 10)),
          view(null_p, np.zeros(null_p))]
    tr, te = slice(0, 320), slice(320, n)
    return [x[tr] for x in xs], [x[te] for x in xs], s[tr], s[te]


@pytest.fixture(scope="module")
def planted():
    xtr, xte, s_tr, s_te = _design()
    res = pysimlr.fit("simlr", xtr, k=2, iterations=40, random_state=0)
    return res, xtr, xte, s_tr, s_te


def test_permutation_pvalues_match_a_loop():
    r = np.random.default_rng(4)
    xz = _zscore(r.standard_normal((80, 6)))
    t = _zscore(xz[:, 0] + r.standard_normal(80))
    zo, lam2 = _relative_weight_basis(xz)
    obs = _relative_weight_shares(zo, lam2, t[:, None])[:, 0]
    p, pg = _perm_pvalues(zo, lam2, t, obs, {"g": [0, 1]}, 300, np.random.default_rng(9))
    g = np.random.default_rng(9)
    perms = g.permuted(np.tile(t[:, None], (1, 300)), axis=0)
    null = np.column_stack([_relative_weight_shares(zo, lam2, perms[:, [b]])[:, 0] for b in range(300)])
    assert np.allclose(p, (1 + (null >= obs[:, None]).sum(1)) / 301)
    assert np.isclose(pg["g"], (1 + (null[[0, 1]].sum(0) >= obs[[0, 1]].sum()).sum()) / 301)
    assert np.isclose(obs.sum(), 1.0)


def test_planted_drivers_replicate_and_noise_does_not(planted):
    res, xtr, xte, s_tr, s_te = planted
    rel = pysimlr.feature_relevance(res, xtr, xte, outcome=s_tr, outcome_test=s_te, views=(0,), **FAST)
    f = rel["features"]
    assert list(f.tier[:4]) == [2, 2, 2, 2]
    assert (f.tier[4:] == 2).sum() <= 1
    assert np.isclose(f.share_train.sum(), 1.0) and np.isclose(f.share_test.sum(), 1.0)
    assert (f.r_outcome_test[:4].abs() > 0.3).all()
    assert np.allclose(f.ratio_fair, f.share_train * len(f))
    s = rel["summary"].iloc[0]
    assert s.n_replicated == (f.tier == 2).sum() and np.isclose(s.expected_by_chance, 0.05 * 24)


def test_null_view_gives_about_alpha_false_positives(planted):
    res, xtr, xte, s_tr, _ = planted
    rel = pysimlr.feature_relevance(res, xtr, xte, outcome=s_tr, views=(3,), **FAST)
    s = rel["summary"].iloc[0]
    assert s.n_train <= 9 and s.n_test <= 9          # 60 features, chance 3; P(>= 10) ~ 0.003
    assert s.n_replicated <= 2
    assert rel["features"].r_outcome_test.isna().all()   # no outcome_test


def test_groups_sum_shares(planted):
    res, xtr, xte, s_tr, _ = planted
    groups = {"block": lambda n: "drivers" if int(n.split(".")[1]) < 4 else ("rest" if int(n.split(".")[1]) < 12
                                                                              else None)}
    rel = pysimlr.feature_relevance(res, xtr, xte, outcome=s_tr, views=(0,), groups=groups, **FAST)
    g = rel["groups"].set_index("group")
    f = rel["features"]
    assert set(g.grouping) == {"block"}
    assert np.isclose(g.loc["drivers", "share_train"], f.share_train[:4].sum())
    assert g.loc["drivers", "n"] == 4 and g.loc["rest", "n"] == 8
    assert g.loc["drivers", "tier"] == 2 and g.loc["rest", "tier"] == 0
    assert np.isclose(g.loc["drivers", "ratio_fair"], f.share_train[:4].sum() * 24 / 4)


def test_without_test_views_nothing_replicates(planted):
    res, xtr, _, s_tr, _ = planted
    rel = pysimlr.feature_relevance(res, xtr, outcome=s_tr, views=(0,), **FAST)
    f = rel["features"]
    assert f.share_test.isna().all() and f.p_test.isna().all()
    assert set(f.tier) <= {0, 1} and (f.tier[:4] == 1).all()
    assert rel["n_permutations_test"] == 0


def test_refits_and_null_refits_are_reproducible(planted):
    res, xtr, xte, s_tr, _ = planted
    kw = dict(FAST, n_refits=2, refit_permutations=500, n_null_refits=3)
    a = pysimlr.feature_relevance(res, xtr, xte, outcome=s_tr, views=(0,), **kw)
    b = pysimlr.feature_relevance(res, xtr, xte, outcome=s_tr, views=(0,), **kw)
    fa, fb = a["features"], b["features"]
    assert a["n_refits"] == 2 and a["n_null_refits"] == 3
    assert fa.refit_p05_freq.between(0, 1).all() and np.isfinite(fa.p_null_refit).all()
    assert (fa.p_null_refit >= 0.25 - 1e-12).all()      # (1 + count) / (3 + 1)
    assert fa.equals(fb)
    assert (fa.refit_p05_freq[:4] > 0).any()


def test_missing_view_at_test_time(planted):
    res, xtr, xte, s_tr, _ = planted
    rel = pysimlr.feature_relevance(res, xtr, [xte[0], xte[1], None, xte[3]], outcome=s_tr, **FAST)
    assert set(rel["features"].view) == {0, 1, 3}
    assert (rel["features"].query("view == 0").tier[:4] == 2).all()


def test_errors(planted):
    res, xtr, xte, s_tr, _ = planted
    with pytest.raises(ValueError, match="train_views"):
        pysimlr.feature_relevance(res, xtr[:3], **FAST)
    with pytest.raises(ValueError, match="not available"):
        pysimlr.feature_relevance(res, xtr, [xte[0], None, xte[2], xte[3]], outcome=s_tr, views=(1,), **FAST)
    with pytest.raises(ValueError, match="component= or outcome="):
        pysimlr.feature_relevance(res, xtr, xte, views=(0,), **FAST)
    with pytest.raises(ValueError, match="spaces"):
        pysimlr.feature_relevance(res, xtr, xte, outcome=s_tr, views=(0,), groups={"a b": {}}, **FAST)
    with pytest.raises(ValueError, match="outcome has"):
        pysimlr.feature_relevance(res, xtr, xte, outcome=s_tr[:10], views=(0,), **FAST)


def test_component_given_directly(planted):
    res, xtr, xte, s_tr, _ = planted
    a = pysimlr.feature_relevance(res, xtr, xte, outcome=s_tr, views=(0,), **FAST)
    b = pysimlr.feature_relevance(res, xtr, xte, component=a["component"], views=(0,), **FAST)
    assert b["component"] == a["component"]
    assert np.allclose(a["features"].share_train, b["features"].share_train)


def test_plot_feature_relevance(planted):
    import matplotlib
    matplotlib.use("Agg")
    from pysimlr.plotting import plot_feature_relevance
    res, xtr, xte, s_tr, s_te = planted
    names = [[f"f{i}" for i in range(x.shape[1])] for x in xtr]
    rel = pysimlr.feature_relevance(res, xtr, xte, outcome=s_tr, outcome_test=s_te, views=(0,), feature_names=names,
                                    groups={"half": lambda n: "a" if int(n[1:]) < 12 else "b"}, **FAST)
    fig = plot_feature_relevance(rel, 0, separate=lambda n: n in ("f22", "f23"), label=str.upper)
    assert len(fig.axes) == 3
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert labels[:4] and set(labels) >= {"F0", "F1", "F2", "F3"}
    fig = plot_feature_relevance(rel, 0, show_all=True)
    assert len(fig.axes[0].get_yticklabels()) == 24
    with pytest.raises(ValueError, match="not in this feature_relevance"):
        plot_feature_relevance(rel, 2)
    with pytest.raises(ValueError, match="grouping"):
        plot_feature_relevance(rel, 0, grouping="lobe")
    matplotlib.pyplot.close("all")


def test_tier_rows_paint_regions(monkeypatch):
    """The tier option of plot_brain_map maps features to DKT regions with brain_rows' rules."""
    import pandas as pd

    from pysimlr.plotting import _TIER_UNITS, _tier_rows
    av = types.SimpleNamespace(dkt_id=lambda n: {"left_entorhinal": 1006, "right_entorhinal": 2006}[n])
    f = pd.DataFrame(dict(view=[0] * 4, feature=["left_entorhinaldktregions", "left_entorhinaldktcortex",
                                                 "right_entorhinal", "csftissues"], tier=[0, 2, 1, 2]))
    rows, dropped = _tier_rows({"features": f}, 0, None, av)
    row = rows["relevance tier"]
    assert row["values"] == {"left_entorhinal": 2.0, "right_entorhinal": 1.0}   # dktcortex wins
    assert set(dropped) == {"left_entorhinaldktregions", "csftissues"}
    assert row["units"] == _TIER_UNITS and row["vmax"] == 2.0
    with pytest.raises(ValueError, match="view 1"):
        _tier_rows({"features": f}, 1, None, av)
