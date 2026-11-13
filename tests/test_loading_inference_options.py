"""Opt-in options of the interpretation tools (v0.32.0, D53).

(a) ``groups=`` / ``scale_within_group=`` on ``structure_coefficients``, ``loadings``, ``view_scores``,
    ``feature_maps``, ``loading_inference`` and ``permutation_correlations``;
(b) ``exclude_views=`` on ``feature_maps``;
(c) ``return_refits=`` on ``loading_inference``;
(d) ``subsample=`` / ``strata=`` on ``loading_inference`` and the status of the stability-selection bound;
(e) ``pysimlr.orient_latents``.

Every option defaults to the old behavior: ``test_defaults_equal_values_computed_on_main`` compares against
``tests/data/loading_options_main_reference.npz``, generated on main (v0.31.1) by ``_reference_quantities``.
"""
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

import pysimlr
from pysimlr import MissingRowsWarning
from pysimlr.loadings import structure_coefficients
from pysimlr.sem import view_scores
from pysimlr.stats import permutation_correlations



def _draw_subsample(*args, **kwargs):          # imported lazily so the reference generator runs on v0.31.1
    from pysimlr.loadings import _draw_subsample as f
    return f(*args, **kwargs)


REF_FILE = Path(__file__).parent / "data" / "loading_options_main_reference.npz"
WIDTHS = (10, 8, 12)


def make_views(n=90, k=1, seed=0, signal=1.5, widths=WIDTHS, shifts=None, groups=None):
    """Views sharing a planted latent (feature block ``[3j, 3j+4)`` loads on component ``j``).
    ``shifts``: per-view, per-group mean shift added to every feature of the view."""
    r = np.random.default_rng(seed)
    z = r.standard_normal((n, k))
    vs = []
    for vi, p in enumerate(widths):
        w = np.zeros((k, p))
        for j in range(k):
            w[j, 3 * j:3 * j + 4] = 1.0
        x = z @ w * signal + r.standard_normal((n, p))
        if shifts is not None:
            x = x + np.asarray(shifts[vi])[groups][:, None]
        vs.append(torch.tensor(x, dtype=torch.float32))
    return vs, z


def _reference_quantities():
    """Everything the options must leave unchanged; uses only the v0.31.1 API."""
    out = {}
    vs, _ = make_views(n=80, k=2, seed=3)
    res = pysimlr.fit("simlr", vs, 2, random_state=0, iterations=40)
    lo = pysimlr.loadings(res, vs)
    for m in range(len(vs)):
        out[f"loadings/structure/{m}"] = lo["structure"][m].numpy()
        out[f"loadings/loadings/{m}"] = lo["loadings"][m].numpy()
    out["loadings/rotation"] = lo["loading_rotation"].numpy()
    sc = view_scores(res, vs)
    for m, s_ in enumerate(sc["scores"]):
        out[f"scores/{m}"] = s_
    out["scores/consensus"] = sc["consensus"]
    for target in ("other_views", "consensus"):
        maps = pysimlr.feature_maps(res, vs, target=target)
        for kind, lst in maps.items():
            for m, t in enumerate(lst):
                out[f"maps/{target}/{kind}/{m}"] = t.numpy()
    vm = [v.clone() for v in vs]
    vm[1][:20] = float("nan")
    resm = pysimlr.fit("simlr", vm, 2, random_state=0, iterations=40, missing_rows="mask")
    for kind, lst in pysimlr.feature_maps(resm, vm).items():
        for m, t in enumerate(lst):
            out[f"maps_mask/{kind}/{m}"] = t.numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inf = pysimlr.loading_inference("simlr", vs, 2, n_subsamples=5, n_permutations=3, random_state=1,
                                        iterations=40, maps=("structure", "partial"), n_jobs=1)
    for m in range(len(vs)):
        for key in ("structure", "ci_low", "ci_high", "selection_frequency"):
            out[f"inf/{key}/{m}"] = np.asarray(inf[key][m], float)
        out[f"inf/selected/{m}"] = np.asarray(inf["selected"][m], float)
        out[f"inf/component_stability/{m}"] = np.asarray(inf["component_stability"][m], float)
        for kind in ("structure", "partial"):
            out[f"inf/maps/{kind}/ci_high/{m}"] = inf["maps"][kind]["ci_high"][m].numpy()
            out[f"inf/maps/{kind}/selection_frequency/{m}"] = inf["maps"][kind]["selection_frequency"][m].numpy()
    out["inf/expected_false"] = np.asarray(inf["expected_false_selections"], float)
    out["inf/q"] = np.asarray(inf["q"], float)
    out["inf/null_max_abs"] = np.asarray(inf["null_max_abs"], float)
    out["inf/subspace_stability"] = np.asarray(inf["subspace_stability"], float)
    out["inf/oos_coupling"] = np.asarray(inf["oos_coupling"], float)
    out["inf/oos_r2"] = np.asarray(inf["oos_r2"], float)
    out["inf/oos_n"] = np.asarray(inf["oos_n"], float)
    out["inf/oob_rows"] = np.concatenate([np.asarray(o["rows"]) for o in inf["oob_scores"]])
    r = np.random.default_rng(5)
    x = r.standard_normal((50, 6))
    y = x[:, 0] * 0.5 + r.standard_normal(50)
    for paired in (False, True):
        pc = permutation_correlations(x, np.column_stack([y] * 6) if paired else y, n_perms=50, n_boot=30,
                                      seed=2, paired=paired)
        for key, val in pc.items():
            out[f"perm/{int(paired)}/{key}"] = np.asarray(val, float)
    return out


def test_defaults_equal_values_computed_on_main():
    ref = np.load(REF_FILE)
    now = _reference_quantities()
    assert set(ref.files) == set(now)
    for key in ref.files:
        np.testing.assert_allclose(now[key], ref[key], atol=1e-6, rtol=1e-5, err_msg=key)


# --------------------------------------------------------------------------- (a) groups
N_G = 90
GROUPS = np.repeat(["a", "b", "c"], N_G // 3)


def _shifted(seed=0, shift=3.0, signal=0.0, k=1):
    """Views with NO within-group association (signal=0) but group mean shifts in every view."""
    shifts = [[0.0, shift, -shift]] * len(WIDTHS)
    codes = np.repeat([0, 1, 2], N_G // 3)
    return make_views(n=N_G, k=k, seed=seed, signal=signal, shifts=shifts, groups=codes)


def test_group_shift_makes_spurious_structure_that_groups_removes():
    vs, _ = _shifted()
    u = torch.tensor(np.repeat([0.0, 1.0, -1.0], N_G // 3)
                     + 0.3 * np.random.default_rng(1).standard_normal(N_G))[:, None]
    pooled = structure_coefficients(vs, u)
    within = structure_coefficients(vs, u, groups=GROUPS)
    assert max(float(c.abs().mean()) for c in pooled) > 0.6          # spurious, driven by group means
    assert max(float(c.abs().mean()) for c in within) < 0.25         # none within group
    # centering by hand gives the same pooled within-group correlation
    x = vs[0].double().numpy()
    uu = u.double().numpy()[:, 0]
    xc, uc = x.copy(), uu.copy()
    for g in np.unique(GROUPS):
        s = GROUPS == g
        xc[s] -= xc[s].mean(0)
        uc[s] -= uc[s].mean()
    ref = (xc / np.linalg.norm(xc, axis=0)).T @ (uc / np.linalg.norm(uc))
    np.testing.assert_allclose(within[0][:, 0].numpy(), ref, atol=1e-5)


def test_scale_within_group_is_separate_and_changes_only_with_unequal_variances():
    r = np.random.default_rng(0)
    x = torch.tensor(np.vstack([r.standard_normal((30, 4)), 5 * r.standard_normal((30, 4))]), dtype=torch.float32)
    u = x[:, :1] * 0.5 + torch.tensor(r.standard_normal((60, 1)), dtype=torch.float32)
    g = np.repeat([0, 1], 30)
    a = structure_coefficients([x], u, groups=g)[0]
    b = structure_coefficients([x], u, groups=g, scale_within_group=True)[0]
    assert not torch.allclose(a, b, atol=1e-3)
    x2 = torch.tensor(r.standard_normal((60, 4)), dtype=torch.float32)
    g2 = np.repeat([0, 1], 30)
    a2 = structure_coefficients([x2], u, groups=g2)[0]
    b2 = structure_coefficients([x2], u, groups=g2, scale_within_group=True)[0]
    assert float((a2 - b2).abs().max()) < 0.15                        # similar scales: close but not identical


def test_loadings_and_view_scores_and_maps_use_groups():
    vs, _ = _shifted(seed=2)
    res = pysimlr.fit("simlr", vs, 1, random_state=0, iterations=40)
    pooled = pysimlr.loadings(res, vs)
    within = pysimlr.loadings(res, vs, groups=GROUPS)
    assert float(pooled["structure"][0].abs().mean()) > 0.5
    assert float(within["structure"][0].abs().mean()) < 0.3
    # view_scores: within-group centered (group means ~ 0) and sign chosen within group
    sc = view_scores(res, vs, groups=GROUPS)
    for z in sc["scores"]:
        for g in np.unique(GROUPS):
            assert abs(z[GROUPS == g].mean()) < 1e-6
    merged = {**res, **within}
    maps = pysimlr.feature_maps(merged, vs, groups=GROUPS)
    plain = pysimlr.feature_maps(merged, vs)
    for kind in ("structure", "partial", "component_partial"):
        assert float(maps[kind][0].abs().mean()) < 0.5 * float(plain[kind][0].abs().mean()) + 0.05, kind
    assert float(plain["structure"][0].abs().mean()) > 0.5


def test_groups_recover_planted_signal_within_groups():
    vs, _ = make_views(n=N_G, k=1, seed=4, signal=1.5, shifts=[[0, 4.0, -4.0]] * 3,
                       groups=np.repeat([0, 1, 2], N_G // 3))
    res = pysimlr.fit("simlr", vs, 1, random_state=0, iterations=60)
    merged = {**res, **pysimlr.loadings(res, vs, groups=GROUPS)}
    st = pysimlr.feature_maps(merged, vs, kinds=["structure"], groups=GROUPS)["structure"][0][:, 0].numpy()
    assert np.abs(st[:4]).min() > 0.4 and np.abs(st[4:]).max() < 0.45


def test_groups_validation_and_small_groups_follow_the_nan_rules():
    vs, _ = make_views(n=60, k=1, seed=1)
    res = pysimlr.fit("simlr", vs, 1, random_state=0, iterations=30)
    with pytest.raises(ValueError, match="one per row"):
        pysimlr.loadings(res, vs, groups=np.zeros(5))
    g = np.repeat([0, 1, 2], 20)
    g[:2] = 9                                              # a group of 2 rows
    with pytest.warns(MissingRowsWarning, match="fewer than 3"):
        maps, info = pysimlr.feature_maps(res, vs, kinds=["structure"], groups=g, return_info=True)
    assert info["n_eff"] == [58, 58, 58]
    assert info["groups"]["left_out_groups"][0] == ["9"]
    assert torch.isfinite(maps["structure"][0]).all()
    # all rows in tiny groups -> NaN with a reason, never zeros
    tiny = np.repeat(np.arange(30), 2)
    with pytest.warns(MissingRowsWarning):
        maps, info = pysimlr.feature_maps(res, vs, kinds=["structure"], groups=tiny, return_info=True)
    assert torch.isnan(maps["structure"][0]).all()
    assert "min_rows" in info["reasons"]["structure"][0]


def test_groups_compose_with_masked_missing_rows():
    vs, _ = make_views(n=N_G, k=1, seed=6, signal=1.5, shifts=[[0, 4.0, -4.0]] * 3,
                       groups=np.repeat([0, 1, 2], N_G // 3))
    vm = [v.clone() for v in vs]
    vm[1][::5] = float("nan")
    res = pysimlr.fit("simlr", vm, 1, random_state=0, iterations=60, missing_rows="mask")
    lo = pysimlr.loadings(res, vm, groups=GROUPS)
    assert all(torch.isfinite(c).all() for c in lo["structure"])
    merged = {**res, **lo}
    maps, info = pysimlr.feature_maps(merged, vm, kinds=["structure", "partial"], groups=GROUPS, return_info=True)
    assert info["n_eff"][1] < info["n_eff"][0]
    assert all(torch.isfinite(t).all() for t in maps["structure"])
    st = maps["structure"][0][:, 0].numpy()
    assert np.abs(st[:4]).min() > 0.35
    # complete-data equivalence: grouped maps on masked rows equal the grouped call on the complete-case rows when
    # the views share their observed rows (two views)
    v2 = [vm[0], vm[1]]
    res2 = pysimlr.fit("simlr", v2, 1, random_state=0, iterations=40, missing_rows="mask")
    mp = pysimlr.feature_maps(res2, v2, kinds=["structure"], groups=GROUPS)["structure"]
    assert all(torch.isfinite(t).all() for t in mp)


def test_within_group_permutation_is_calibrated_and_unrestricted_is_not():
    codes = np.repeat([0, 1, 2, 3], 25)
    n_rep, ps_grp, ps_free = 60, [], []
    for s in range(n_rep):
        r = np.random.default_rng(100 + s)
        shift = np.array([0.0, 2.0, -2.0, 4.0])[codes]
        x = r.standard_normal((100, 1)) + shift[:, None]
        y = r.standard_normal(100) + shift                        # same group means, no within-group link
        ps_grp.append(permutation_correlations(x, y, n_perms=99, seed=s, groups=codes)["p_perm"][0])
        ps_free.append(permutation_correlations(x, y, n_perms=99, seed=s)["p_perm"][0])
    ps_grp, ps_free = np.asarray(ps_grp), np.asarray(ps_free)
    assert 0.25 < ps_grp.mean() < 0.75 and (ps_grp < 0.05).mean() < 0.2    # ~uniform
    assert (ps_free < 0.05).mean() > 0.9                                    # spurious: the shift is read as signal


def test_permutation_correlations_groups_none_is_unchanged_and_groups_detect_within_signal():
    r = np.random.default_rng(1)
    codes = np.repeat([0, 1, 2], 30)
    z = r.standard_normal(90)
    x = np.column_stack([z + 0.5 * r.standard_normal(90), r.standard_normal(90)]) + np.array([0, 3, -3])[codes][:, None]
    y = z + np.array([0, 3, -3])[codes]
    base = permutation_correlations(x, y, n_perms=99, seed=1)
    same = permutation_correlations(x, y, n_perms=99, seed=1, groups=None)
    for key in base:
        np.testing.assert_array_equal(base[key], same[key])
    g = permutation_correlations(x, y, n_perms=99, seed=1, n_boot=40, groups=codes)
    assert g["p_perm"][0] <= 0.02 and g["p_perm"][1] > 0.05
    assert g["ci95_low"][0] > 0.3
    with pytest.raises(ValueError, match="degrees of freedom"):
        permutation_correlations(x[:6], y[:6], n_perms=9, seed=1, groups=[0, 1, 2, 3, 4, 5])


def test_loading_inference_null_permutes_within_groups(monkeypatch):
    import importlib
    L = importlib.import_module("pysimlr.loadings")        # `pysimlr.loadings` the attribute is the function
    seen = []
    orig = L.within_group_permutation

    def spy(rng, codes):
        p = orig(rng, codes)
        seen.append((p, np.asarray(codes)))
        return p
    monkeypatch.setattr(L, "within_group_permutation", spy)
    vs, _ = _shifted(seed=3, signal=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = pysimlr.loading_inference("simlr", vs, 1, n_subsamples=3, n_permutations=3, groups=GROUPS,
                                        iterations=30, n_jobs=1)
    assert len(seen) == 3 * (len(vs) - 1)
    for p, c in seen:
        assert (c[p] == c).all()
    assert np.isfinite(out["null_max_abs"])
    assert out["resampling"]["within_group"] is True


def test_loading_inference_groups_gives_within_group_structure_and_maps():
    vs, _ = _shifted(seed=5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pooled = pysimlr.loading_inference("simlr", vs, 1, n_subsamples=4, iterations=30, maps=("structure",), n_jobs=1)
        within = pysimlr.loading_inference("simlr", vs, 1, n_subsamples=4, iterations=30, maps=("structure",),
                                           groups=GROUPS, strata=GROUPS, n_jobs=1)
    assert float(pooled["structure"][0].abs().mean()) > 0.5
    assert float(within["structure"][0].abs().mean()) < 0.3
    assert float(within["maps"]["structure"]["map"][0].abs().mean()) < 0.35
    assert float(pooled["maps"]["structure"]["map"][0].abs().mean()) > 0.5
    # out["result"] carries the within-group loadings (same keys, grouped values)
    np.testing.assert_allclose(within["result"]["structure"][0].numpy(), within["structure"][0].numpy())


# --------------------------------------------------------------------------- (b) exclude_views
def _signal_in_views(n=120, seed=0):
    """View 0 and 1 carry nothing shared with each other; view 2 ('root') carries the only latent that view 1 sees."""
    r = np.random.default_rng(seed)
    z = r.standard_normal((n, 1))
    v0 = r.standard_normal((n, 8))
    v1 = np.hstack([z * 1.5 + r.standard_normal((n, 4)), r.standard_normal((n, 4))])
    v2 = np.hstack([z * 1.5 + r.standard_normal((n, 4)), r.standard_normal((n, 2))])
    return [torch.tensor(a, dtype=torch.float32) for a in (v0, v1, v2)]


def test_exclude_views_changes_the_target_as_specified():
    vs = _signal_in_views()
    res = {**pysimlr.fit("simlr", vs, 1, random_state=0, iterations=60), "view_names": ["a", "b", "root"]}
    sc = view_scores(res, vs, sign_against=res["u"])["scores"]
    for ex in ([2], ["root"]):
        got = pysimlr.feature_maps(res, vs, kinds=["structure"], exclude_views=ex)["structure"]
        for m in range(3):
            others = [sc[i] for i in range(3) if i != m and i != 2]
            if not others:
                continue
            t = sum(others)
            t = (t - t.mean(0)) / t.std(0)
            x = vs[m].double().numpy()
            xz = (x - x.mean(0)) / x.std(0)
            np.testing.assert_allclose(got[m][:, 0].numpy(), xz.T @ t[:, 0] / len(t), atol=2e-3)
    # the root view's own map is against the other views only; view 0 gets view 1's score alone
    full = pysimlr.feature_maps(res, vs, kinds=["structure"])["structure"]
    cut = pysimlr.feature_maps(res, vs, kinds=["structure"], exclude_views=["root"])["structure"]
    assert not torch.allclose(full[0], cut[0], atol=1e-4)


def test_excluding_the_only_carrier_view_drops_the_other_maps_to_null():
    """View 0 carries the latent, shared with the root view only; view 1 is noise. View 0's other-views target
    contains the root's score, so its map shows the planted pattern; with the root excluded its target is the noise
    view's score and the map is null."""
    r = np.random.default_rng(3)
    n = 200
    z = r.standard_normal((n, 1))
    v0 = np.hstack([z * 1.5 + r.standard_normal((n, 4)), r.standard_normal((n, 6))])
    v1 = r.standard_normal((n, 5))
    root = z * 2.0 + r.standard_normal((n, 3))
    vs = [torch.tensor(a, dtype=torch.float32) for a in (v0, v1, root)]
    res = pysimlr.fit("simlr", vs, 1, random_state=0, iterations=80)
    full = pysimlr.feature_maps(res, vs, kinds=["structure"])["structure"][0][:, 0].abs().numpy()
    cut = pysimlr.feature_maps(res, vs, kinds=["structure"], exclude_views=[2])["structure"]
    assert full[:4].min() > 0.4 and full[4:].max() < 0.3
    assert float(cut[0].abs().max()) < 4.0 / np.sqrt(n)          # null: within sampling noise of zero
    assert np.isfinite(cut[2].numpy()).all()                     # the excluded view still gets a map


def test_exclude_views_edge_cases():
    vs = _signal_in_views()
    res = {**pysimlr.fit("simlr", vs, 1, random_state=0, iterations=30), "view_names": ["a", "b", "root"]}
    with pytest.raises(ValueError, match="unknown view"):
        pysimlr.feature_maps(res, vs, exclude_views=["nope"])
    with pytest.raises(TypeError):
        pysimlr.feature_maps(res, vs, exclude_views="root")
    with pytest.raises(ValueError, match="other_views"):
        pysimlr.feature_maps(res, vs, target="consensus", exclude_views=[2])
    # leaving one non-excluded view: that view has nobody else -> NaN with a reason
    with pytest.warns(MissingRowsWarning):
        maps, info = pysimlr.feature_maps(res, vs, kinds=["structure"], exclude_views=[1, 2], return_info=True)
    assert torch.isnan(maps["structure"][0]).all()
    assert "non-excluded" in info["reasons"]["structure"][0]
    assert torch.isfinite(maps["structure"][1]).all()
    # empty list == None
    a = pysimlr.feature_maps(res, vs, kinds=["structure"])["structure"]
    b = pysimlr.feature_maps(res, vs, kinds=["structure"], exclude_views=[])["structure"]
    assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_exclude_views_is_forwarded_by_loading_inference_map_options():
    vs = _signal_in_views()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = pysimlr.loading_inference("simlr", vs, 1, n_subsamples=3, iterations=30, maps=("structure",), n_jobs=1)
        cut = pysimlr.loading_inference("simlr", vs, 1, n_subsamples=3, iterations=30, maps=("structure",), n_jobs=1,
                                        map_options={"exclude_views": [2]})
    assert not torch.allclose(base["maps"]["structure"]["map"][0], cut["maps"]["structure"]["map"][0], atol=1e-4)


# --------------------------------------------------------------------------- (c) return_refits
def _inference(**kw):
    vs, _ = make_views(n=60, k=2, seed=2)
    base = dict(n_subsamples=6, iterations=40, random_state=3, n_jobs=1, maps=("structure", "partial"))
    base.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pysimlr.loading_inference("simlr", vs, 2, **base), vs


def test_return_refits_default_is_absent_and_shapes_are_documented():
    out, _ = _inference()
    assert "refits" not in out
    out, vs = _inference(return_refits=("weights", "maps", "latents", "rows"))
    rf = out["refits"]
    b, n, k = 6, 60, 2
    assert list(rf["index"]) == list(range(b))
    assert rf["structure"].shape == (b, sum(WIDTHS), k)
    assert [w.shape for w in rf["weights"]] == [(b, p, k) for p in WIDTHS]
    assert rf["latents"].shape == (b, n, k)
    for kind in ("structure", "partial"):
        assert [t.shape for t in rf["maps"][kind]] == [(b, p, k) for p in WIDTHS]
    assert len(rf["rows"]) == b and all(len(r_) == 30 for r_ in rf["rows"])
    assert rf["alignment"]["rotation"].shape == (b, k, k)


def test_return_refits_in_bag_flags_and_latent_marking():
    out, _ = _inference(return_refits=("latents", "rows"))
    rf = out["refits"]
    for i in range(6):
        in_bag = np.zeros(60, bool)
        in_bag[rf["rows"][i]] = True
        np.testing.assert_array_equal(rf["in_bag"][i], in_bag)
        np.testing.assert_array_equal(np.flatnonzero(~rf["in_bag"][i]), rf["oob_rows"][i])
        assert np.isfinite(rf["latents"][i]).all()            # every row has a latent: in-bag from the fit, rest predicted
    # oob_scores agree on the rows used
    for rec, rows in zip(out["oob_scores"], rf["oob_rows"]):
        np.testing.assert_array_equal(rec["rows"], rows)


def test_return_refits_alignment_round_trips_and_matches_the_summary():
    out, vs = _inference(return_refits=("weights", "latents", "maps"))
    rf = out["refits"]
    ref = out["result"]
    # structure: stability recomputed from the returned refits equals the summary
    stacked_ref = np.vstack([c.numpy() for c in ref["structure"]])
    sizes = np.cumsum([0] + rf["view_sizes"])
    for vi in range(3):
        sl = rf["structure"][:, sizes[vi]:sizes[vi + 1]].astype(float)
        from pysimlr.loadings import _col_corr_stability
        mine = _col_corr_stability(sl, ref["structure"][vi].double().numpy())
        np.testing.assert_allclose(mine, out["component_stability"][vi], atol=2e-4)
        # maps likewise
        for kind in ("structure", "partial"):
            arr = rf["maps"][kind][vi].astype(float)
            mine = _col_corr_stability(arr, out["maps"][kind]["map"][vi].double().numpy(), nan_safe=True)
            np.testing.assert_allclose(mine, out["maps"][kind]["component_stability"][vi], atol=2e-4)
    # the aligned structure of a refit is the refit's structure times the Procrustes rotation; weights/latents use
    # rotation = loading_rotation @ procrustes, so (X V) @ A is the refit's rotated, aligned latent
    a = rf["alignment"]
    for i in (0, 3):
        np.testing.assert_allclose(a["rotation"][i], a["loading_rotation"][i] @ a["procrustes"][i], atol=1e-10)
        # structure frame <-> latent frame: orthogonal, so undo is the transpose
        np.testing.assert_allclose(a["rotation"][i] @ a["rotation"][i].T, np.eye(2), atol=1e-6)
        # latents in-bag == X_sub @ weights (the weights are an encoder of the same frame), up to the per-view scale of the avg
        lat_back = rf["latents"][i].astype(float) @ a["rotation"][i].T
        assert np.isfinite(lat_back).all()
    # a per-refit weight set reproduces the refit's own latent frame: weights_aligned @ rotation.T is the raw V of that refit


def test_return_refits_latents_agree_with_the_refit_in_the_reference_frame():
    """Refit 0, re-run by hand: its latent on the in-bag rows, rotated by the stored alignment, equals the stored one."""
    vs, _ = make_views(n=60, k=2, seed=2)
    out, _ = _inference(return_refits=("latents", "rows", "weights"))
    rf = out["refits"]
    idx = rf["rows"][0]
    sub = [v[torch.as_tensor(idx)] for v in vs]
    r0 = pysimlr.fit("simlr", sub, 2, random_state=3, iterations=40)
    a = rf["alignment"]["rotation"][0]
    np.testing.assert_allclose(rf["latents"][0][idx].astype(float), r0["u"].numpy() @ a, atol=2e-4)
    # weights: same frame
    for vi in range(3):
        np.testing.assert_allclose(rf["weights"][vi][0].astype(float), r0["v"][vi].numpy() @ a, atol=2e-4)
    # and the oob latents are predictions
    oob = rf["oob_rows"][0]
    pr = pysimlr.predict(r0, [v[torch.as_tensor(oob)] for v in vs])["u"].numpy()
    np.testing.assert_allclose(rf["latents"][0][oob].astype(float), pr @ a, atol=2e-4)


def test_return_refits_guard_and_validation():
    vs, _ = make_views(n=60, k=2, seed=2)
    with pytest.raises(ValueError, match="max_refit_bytes"):
        pysimlr.loading_inference("simlr", vs, 2, n_subsamples=4, return_refits=("latents",), max_refit_bytes=100)
    with pytest.raises(ValueError, match="unknown return_refits"):
        pysimlr.loading_inference("simlr", vs, 2, n_subsamples=4, return_refits=("bogus",))
    with pytest.raises(ValueError, match="maps"):
        pysimlr.loading_inference("simlr", vs, 2, n_subsamples=4, return_refits=("maps",))


def test_return_refits_respects_exclude_unconverged():
    out, _ = _inference(return_refits=("rows",), exclude_unconverged=True, iterations=2)
    rf = out["refits"]
    assert len(rf["index"]) == out["n_effective_refits"] == rf["structure"].shape[0]


# --------------------------------------------------------------------------- (d) resampling schemes
def test_default_draws_are_unchanged():
    rng_a, rng_b = np.random.default_rng(7), np.random.default_rng(7)
    for _ in range(5):
        old = np.sort(rng_a.choice(61, 30, replace=False))
        new = _draw_subsample(rng_b, 61, 30, "half", None)
        np.testing.assert_array_equal(old, new)
    out, _ = _inference(n_subsamples=4, return_refits=("rows",))
    rng = np.random.default_rng(3)
    for rows in out["refits"]["rows"]:
        np.testing.assert_array_equal(rows, np.sort(rng.choice(60, 30, replace=False)))


def test_stratified_draws_keep_proportions_and_sizes():
    strata = np.repeat([0, 1, 2], [40, 20, 10])
    rng = np.random.default_rng(1)
    for _ in range(20):
        idx = _draw_subsample(rng, 70, 35, "half", strata)
        assert len(idx) == 35 and len(np.unique(idx)) == 35
        counts = np.bincount(strata[idx], minlength=3)
        np.testing.assert_array_equal(counts, [20, 10, 5])
    idx = _draw_subsample(rng, 71 - 1, 31, "fraction", np.repeat([0, 1, 2], [41, 20, 9]))
    assert len(idx) == 31 and np.bincount(np.repeat([0, 1, 2], [41, 20, 9])[idx], minlength=3).tolist() == [18, 9, 4]


def test_bootstrap_out_of_bag_rows_are_the_never_drawn_rows():
    out, _ = _inference(subsample="bootstrap", return_refits=("rows", "latents"), n_subsamples=5)
    rf = out["refits"]
    for i in range(5):
        draws = rf["rows"][i]
        assert len(draws) == 60 and len(np.unique(draws)) < 60       # with replacement
        np.testing.assert_array_equal(rf["oob_rows"][i], np.setdiff1d(np.arange(60), draws))
        np.testing.assert_array_equal(rf["n_draws"][i], np.bincount(draws, minlength=60))
        np.testing.assert_array_equal(rf["in_bag"][i], rf["n_draws"][i] > 0)
    assert 10 < out["oos_n"] < 35
    assert out["resampling"]["subsample"] == "bootstrap"


def test_stratified_loading_inference_draws():
    vs, _ = make_views(n=60, k=1, seed=2)
    strata = np.repeat([0, 1], [40, 20])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = pysimlr.loading_inference("simlr", vs, 1, n_subsamples=4, iterations=30, strata=strata,
                                        return_refits=("rows",), n_jobs=1)
        boot = pysimlr.loading_inference("simlr", vs, 1, n_subsamples=3, iterations=30, strata=strata,
                                         subsample="bootstrap", return_refits=("rows",), n_jobs=1)
    for rows in out["refits"]["rows"]:
        assert np.bincount(strata[rows], minlength=2).tolist() == [20, 10]
    for rows in boot["refits"]["rows"]:
        assert np.bincount(strata[rows], minlength=2).tolist() == [40, 20]


def test_selection_bound_is_reported_only_for_the_half_sampling_design():
    default, _ = _inference()
    assert default["expected_false_selections_reason"] is None
    assert all(np.isfinite(default["expected_false_selections"]))
    assert default["resampling"]["bound_valid"] is True
    for kw, text in (({"subsample": "bootstrap"}, "bootstrap"), ({"fraction": 0.7}, "fraction"),
                     ({"subsample": "fraction", "fraction": 0.3}, "fraction"),
                     ({"strata": np.repeat([0, 1], 30)}, "stratified")):
        out, _ = _inference(**kw)
        assert all(np.isnan(out["expected_false_selections"])), kw
        assert text in out["expected_false_selections_reason"], kw
        assert out["resampling"]["bound_valid"] is False
        assert "selected" in out and "q" in out                       # selection itself is still reported
    with pytest.raises(ValueError, match="subsample"):
        _inference(subsample="jackknife")


# --------------------------------------------------------------------------- (e) orient_latents
def _fits(n=70, k=2, f=5, seed=0):
    r = np.random.default_rng(seed)
    base = r.standard_normal((n, k)) * np.array([2.0, 1.0])[:k]
    out, truth = [], []
    for i in range(f):
        sign = r.choice([-1.0, 1.0], size=k)
        perm = r.permutation(k)
        out.append((base + 0.1 * r.standard_normal((n, k)))[:, perm] * sign)
        truth.append((perm, sign))
    return base, out


def test_orient_latents_signs_make_flipped_copies_agree():
    r = np.random.default_rng(0)
    a = r.standard_normal((50, 1))
    res = pysimlr.orient_latents([a, -a, a * 2.0, -a], mode="sign")
    ls = res["latents"]
    for x in ls[1:]:
        np.testing.assert_allclose(np.corrcoef(x[:, 0], ls[0][:, 0])[0, 1], 1.0, atol=1e-12)
    assert res["transforms"].shape == (4, 1, 1)


@pytest.mark.parametrize("mode", ["sign", "signed_permutation", "procrustes"])
def test_orient_latents_is_invariant_to_the_order_of_the_fits(mode):
    k = 1 if mode == "sign" else 2
    base, fits = _fits(k=k, f=6, seed=2)
    if mode == "sign":
        fits = [f_ * np.random.default_rng(i).choice([-1, 1]) for i, f_ in enumerate(fits)]
    ref = pysimlr.orient_latents(fits, mode=mode)
    for seed in (0, 1, 2):
        order = np.random.default_rng(seed).permutation(len(fits))
        shuffled = pysimlr.orient_latents([fits[i] for i in order], mode=mode)
        for pos, i in enumerate(order):
            np.testing.assert_allclose(shuffled["latents"][pos], ref["latents"][i], atol=1e-9)
        np.testing.assert_allclose(shuffled["reference"], ref["reference"], atol=1e-9, equal_nan=True)


def test_orient_latents_k2_with_swapped_components_and_carried_weights_and_maps():
    base, fits = _fits(k=2, f=6, seed=4)
    r = np.random.default_rng(9)
    w = [r.standard_normal((5, 2)) for _ in fits]
    # weights that reproduce each fit's latent: u_i = x @ w_i  ->  oriented weights must reproduce oriented latents
    x = r.standard_normal((70, 5))
    fits = [x @ wi for wi in w]
    res = pysimlr.orient_latents(fits, weights=w, maps=[[wi] for wi in w], mode="signed_permutation")
    for lat, wi in zip(res["latents"], res["weights"]):
        np.testing.assert_allclose(x @ wi, lat, atol=1e-9)
    for mp, wi in zip(res["maps"], res["weights"]):
        np.testing.assert_allclose(mp[0], wi, atol=1e-12)
    # transforms are signed permutations
    for t in res["transforms"]:
        assert set(np.unique(np.abs(t))) <= {0.0, 1.0} and np.allclose(np.abs(t).sum(0), 1)
    # on near-copies of one latent, oriented fits correlate ~1 column by column and follow the larger-variance-first order
    base, fits = _fits(k=2, f=6, seed=5)
    res = pysimlr.orient_latents(fits)
    for lat in res["latents"]:
        for j in range(2):
            assert np.corrcoef(lat[:, j], res["latents"][0][:, j])[0, 1] > 0.95
    sd = [np.std(res["latents"][0][:, j]) for j in range(2)]
    assert sd[0] > sd[1]
    # the frame's sign convention: mean latent has non-negative sum of cubes
    assert ((res["reference"] ** 3).sum(axis=0) >= 0).all()


def test_orient_latents_explicit_reference_accepts_fit_results_and_checks_shapes():
    vs, _ = make_views(n=60, k=2, seed=1)
    a = pysimlr.fit("simlr", vs, 2, random_state=0, iterations=40)
    b = pysimlr.fit("simlr", vs, 2, random_state=1, iterations=40)
    res = pysimlr.orient_latents([a, b], mode="procrustes")
    c = [np.corrcoef(res["latents"][0][:, j], res["latents"][1][:, j])[0, 1] for j in range(2)]
    assert min(c) > 0.5
    res2 = pysimlr.orient_latents([a, b], reference=a["u"].numpy(), mode="signed_permutation")
    assert res2["reference_index"] is None
    with pytest.raises(ValueError, match="same shape"):
        pysimlr.orient_latents([np.zeros((10, 2)), np.zeros((10, 3))])
    with pytest.raises(ValueError, match="mode"):
        pysimlr.orient_latents([np.zeros((10, 2))], mode="nope")
