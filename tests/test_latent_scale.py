"""Linear SiMLR returns per-view latents at unit average column variance (one divisor per view,
`result["latent_scale"]`), the scale of `u`; internal uses are unchanged by that constant."""
import copy
import warnings

import numpy as np
import torch

import pysimlr


def _fit(seed=0, n=300, widths=(12, 9, 7)):
    g = np.random.default_rng(seed)
    z = g.standard_normal((n, 2))
    xs = [torch.tensor(z @ g.standard_normal((2, p)) + g.standard_normal((n, p))).float() for p in widths]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pysimlr.fit("simlr", xs, 2, random_state=0), xs


def test_latents_have_unit_average_variance_and_match_predict():
    res, xs = _fit()
    assert len(res["latent_scale"]) == 3
    for z in res["latents"]:
        zc = z.double() - z.double().mean(0)
        assert abs(float(zc.pow(2).mean()) - 1.0) < 1e-6          # unit average column variance
    pred = pysimlr.predict(res, xs)
    for a, b in zip(res["latents"], pred["latents"]):
        assert torch.allclose(a, b, atol=1e-5)
    u_sd = pred["u"].double().std(0)
    assert torch.all((u_sd > 0.5) & (u_sd < 2.0))


def test_one_divisor_per_view_keeps_relative_component_size():
    res, xs = _fit()
    raw = [x.double() @ v.double() for x, v in zip(pysimlr.api._preprocess_views(xs, res["scale_list"], res["provenance_list"])[0], res["v"])]
    for r, z, s in zip(raw, res["latents"], res["latent_scale"]):
        assert torch.allclose(z.double() * s, r, atol=1e-6)


def test_subset_prediction_is_unchanged_by_the_scale():
    """`extension.consensus_maps` uses a GCV ridge whose penalty is relative to the data's own scale."""
    res, xs = _fit()
    old = copy.deepcopy(res)
    old["latents"] = [z * s for z, s in zip(res["latents"], res["latent_scale"])]
    old.pop("latent_scale")
    old.pop("_view_maps", None)
    res.pop("_view_maps", None)
    a = pysimlr.predict(res, [xs[0], None, xs[2]])["u"]
    b = pysimlr.predict(old, [xs[0], None, xs[2]])["u"]
    assert torch.allclose(a.double(), b.double(), atol=1e-6)


def test_results_without_latent_scale_keep_the_internal_scale():
    res, xs = _fit()
    old = dict(res)
    old.pop("latent_scale")
    pred = pysimlr.predict(old, xs)
    for z, s, zo in zip(res["latents"], res["latent_scale"], pred["latents"]):
        assert torch.allclose(zo.double(), z.double() * s, atol=1e-6)
