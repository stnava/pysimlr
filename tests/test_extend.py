"""pysimlr.extend: adding unseen modalities to a fitted model (anchored and reembed),
and pysimlr.predict from any subset of original and extended views."""
import warnings

import numpy as np
import pytest
import torch

import pysimlr

K = 3


@pytest.fixture(scope="module")
def study():
    g = np.random.default_rng(0); n = 360
    z = np.abs(g.normal(size=(n, K)))

    def view(p, noise=0.5):
        W = np.zeros((K, p)); blk = p // K
        for j in range(K):
            W[j, j * blk:(j + 1) * blk] = 0.5 + g.random(blk)
        return (z @ W + noise * g.normal(size=(n, p))).astype(np.float32), W
    A, _ = view(30); B, _ = view(24); C, WC = view(18)
    noise = g.normal(size=(n, 18)).astype(np.float32)
    tr, te = slice(0, 300), slice(300, None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = pysimlr.fit("simlr", [A[tr], B[tr]], K, positivity="positive", random_state=0)
        ext = pysimlr.extend(res, [A[tr], B[tr]], {"C": C[tr]})
    return dict(A=A, B=B, C=C, WC=WC, noise=noise, tr=tr, te=te, res=res, ext=ext, z=z)


def _r(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return np.array([np.corrcoef(a[:, j], b[:, j])[0, 1] for j in range(a.shape[1])])


def test_extension_leaves_the_model_and_its_predictions_unchanged(study):
    s = study; res, ext = s["res"], s["ext"]
    assert "extensions" not in res and len(res["structure"]) == 2          # input not mutated
    assert torch.equal(pysimlr.predict(res, [s["A"][s["te"]], s["B"][s["te"]]])["u"],
                       pysimlr.predict(ext, [s["A"][s["te"]], s["B"][s["te"]]])["u"])
    assert ext["view_names"] == ["view0", "view1"] and list(ext["extensions"]) == ["C"]
    assert len(ext["structure"]) == len(ext["loadings"]) == 3 and ext["structure"][2].shape == (18, K)


def test_new_view_aligns_with_the_frozen_consensus_out_of_sample(study):
    e = study["ext"]["extensions"]["C"]
    assert min(e["holdout"]["coupling"]) >= 0.8 > e["holdout"]["null_bar"]
    full = pysimlr.predict(study["res"], [study["A"][study["te"]], study["B"][study["te"]]])["u"]
    only_c = pysimlr.predict(study["ext"], {"C": study["C"][study["te"]]})
    assert only_c["views_used"] == ["C"]
    assert _r(only_c["u"], full).min() >= 0.7


def test_new_view_loadings_recover_the_planted_support_in_the_original_frame(study):
    L = study["ext"]["loadings"][2].numpy(); planted = study["WC"].T > 0; sel = L > 1e-6
    for j in range(K):
        q = int(np.argmax([(sel[:, j] & planted[:, c]).sum() for c in range(K)]))
        tp = (sel[:, j] & planted[:, q]).sum()
        assert 2 * tp / (sel[:, j].sum() + planted[:, q].sum()) >= 0.8


def test_subset_prediction_adds_information(study):
    s = study; te = s["te"]
    full = pysimlr.predict(s["res"], [s["A"][te], s["B"][te]])["u"]
    a = _r(pysimlr.predict(s["ext"], {"view0": s["A"][te]})["u"], full)
    ac = _r(pysimlr.predict(s["ext"], {"view0": s["A"][te], "C": s["C"][te]})["u"], full)
    assert ac.mean() >= a.mean() - 1e-3
    # missing views on an unextended result, by name or with None
    u1 = pysimlr.predict(s["res"], {"view0": s["A"][te]})["u"]
    u2 = pysimlr.predict(s["res"], [s["A"][te], None])["u"]
    assert torch.allclose(u1, u2) and _r(u1, full).min() >= 0.9


def test_a_view_unrelated_to_the_model_is_flagged(study):
    s = study
    with pytest.warns(RuntimeWarning, match="does not align"):
        en = pysimlr.extend(s["res"], [s["A"][s["tr"]], s["B"][s["tr"]]], {"N": s["noise"][s["tr"]]})
    assert max(en["extensions"]["N"]["holdout"]["coupling"]) < en["extensions"]["N"]["holdout"]["null_bar"]


def test_extension_errors_are_explicit(study):
    s = study
    with pytest.raises(ValueError, match="already exist"):
        pysimlr.extend(s["ext"], [s["A"][s["tr"]], s["B"][s["tr"]]], {"C": s["C"][s["tr"]]})
    with pytest.raises(ValueError, match="rows"):
        pysimlr.extend(s["res"], [s["A"][s["tr"]], s["B"][s["tr"]]], {"D": s["C"][:10]})
    with pytest.raises(ValueError, match="how must be"):
        pysimlr.extend(s["res"], [s["A"][s["tr"]], s["B"][s["tr"]]], {"D": s["C"][s["tr"]]}, how="nope")
    with pytest.raises(ValueError, match="unknown view"):
        pysimlr.predict(s["ext"], {"nope": s["C"][s["te"]]})


@pytest.mark.parametrize("method,kw", [("lend", dict(iterations=6)), ("ned", dict(iterations=6)),
                                       ("flow_v", dict(iterations=4))])
def test_every_family_can_be_extended(study, method, kw):
    s = study; tr, te = s["tr"], s["te"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = pysimlr.fit(method, [s["A"][tr], s["B"][tr]], K, random_state=0, **kw)
        ext = pysimlr.extend(res, [s["A"][tr], s["B"][tr]], {"C": s["C"][tr]}, iterations=40)
    pysimlr.validate_result(ext)
    out = pysimlr.predict(ext, {"C": s["C"][te], "view1": s["B"][te]})
    assert out["u"].shape == (60, K) and torch.isfinite(out["u"]).all()


def test_reembed_keeps_every_antsr_option(study):
    s = study; tr = s["tr"]
    C = s["C"][tr].copy(); C[:, 3] = 1.0                                   # a constant column
    base = dict(result=s["res"], views=[s["A"][tr], s["B"][tr]], new_views={"C": C}, how="reembed")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cat = pysimlr.extend(**base)
        spl = pysimlr.extend(**base, mode="split")
        zero = pysimlr.extend(**base, preprocess="zero")
        fixed = pysimlr.extend(**base, k_new={"C": 2}, joint_k_policy="min", min_k=2)
    assert list(cat["blocks"]) == ["sim", "C"] and list(spl["blocks"]) == ["view0", "view1", "C"]
    assert cat["diagnostics"]["preprocessing"]["C"]["dropped_columns"] == ["C_3"]
    assert zero["diagnostics"]["preprocessing"]["C"]["zeroed_columns"] == ["C_3"]
    assert fixed["k_new_used"] == {"C": 2} and fixed["joint_k"] == 2
    assert cat["reembedded"]["u"].shape[1] == cat["joint_k"]
    assert len(cat["updated_result"]["v"]) == 3 and cat["updated_result"]["view_names"][-1] == "C"
    with pytest.raises(ValueError, match="mode must be"):
        pysimlr.extend(**base, mode="nope")
    with pytest.raises(ValueError, match="cor_threshold"):
        pysimlr.extend(**base, cor_threshold=1.5)


def test_plots_show_extended_views(study):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = study
    fig = pysimlr.plot_view_chord(s["ext"], view_names=["A", "B", "C"])
    assert fig._pysimlr_values["pairwise"].shape == (K, 3, 3)                 # C joins the chord
    plt.close(fig)
    pred = pysimlr.predict(s["ext"], {"view0": s["A"][s["te"]], "view1": s["B"][s["te"]], "C": s["C"][s["te"]]})
    fig = pysimlr.plot_component_radar(s["ext"], pred=pred, view_names=["A", "B", "C"])
    assert fig._pysimlr_values["coupling"].shape == (3, K)
    plt.close(fig)
