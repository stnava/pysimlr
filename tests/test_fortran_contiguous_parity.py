"""Fortran-contiguous (column-major) vs C-contiguous (row-major) view parity.

`pandas.DataFrame.to_numpy(np.float64, copy=True)` on homogeneous float blocks
returns a Fortran-contiguous array (stride `(8, 8*n)` bytes). Wrapping it via
`torch.as_tensor(m).float()` without `.contiguous()` preserves Fortran strides
`(1, n)`, whereas row indexing (`m[ident]` or `m[torch.randperm(n)]`) returns a
C-contiguous array `(p, 1)`. On Apple Accelerate BLAS in float32, `(1, n)` vs
`(p, 1)` dispatches `SGEMM('T')` vs `SGEMM('N')`, introducing ~1e-7 rounding
differences that can tip `tune_learning_rate`'s discrete probe selection and
shift observed vs identity-permuted statistics.

Every view/matrix ingestion boundary must normalize 2-D inputs to C-contiguous
row-major layout so Fortran-ordered DataFrames, column-major NumPy arrays, and
C-contiguous row slices produce bit-identical results.
"""
import numpy as np
import pandas as pd
import pytest
import torch

import pysimlr
from pysimlr.paths import simlr_path
from pysimlr.relevance import feature_relevance
from pysimlr.simlr import (
    decompose_energy,
    estimate_rank,
    initialize_simlr,
    l1_normalize_features,
    pairwise_matrix_similarity,
    predict_shared_latent,
    predict_simlr,
    simlr,
    simlr_perm,
)
from pysimlr.svd import ba_svd, safe_pca, varimax, whiten_matrix
from pysimlr.utils import preprocess_data


def _make_fortran_and_c_views(seed: int = 0, n: int = 93, widths=(76, 54, 38), k: int = 3):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, k))
    c_views = []
    f_views = []
    df_views = []
    for p in widths:
        raw = np.ascontiguousarray(
            z @ rng.standard_normal((k, p)) + 0.35 * rng.standard_normal((n, p)),
            dtype=np.float64,
        )
        # Homogeneous float64 DataFrame from column dict -> Fortran-contiguous block
        df = pd.DataFrame({f"f{j}": raw[:, j] for j in range(p)})
        arr_from_df = df.to_numpy(np.float64, copy=True)
        arr_f = np.asfortranarray(raw)
        assert arr_f.flags["F_CONTIGUOUS"] and not arr_f.flags["C_CONTIGUOUS"]
        c_views.append(raw)
        f_views.append(arr_f if not arr_from_df.flags["F_CONTIGUOUS"] else arr_from_df)
        df_views.append(df)
    return c_views, f_views, df_views


@pytest.mark.parametrize("k", [1, 3])
def test_simlr_perm_identity_matches_fortran_input_exactly(k: int):
    """Exact reproduction of the `simlr_perm(..., n_perms=0)` vs `m[ident]` guard."""
    c_views, f_views, _ = _make_fortran_and_c_views(seed=17, n=93, widths=(76, 54), k=max(k, 2))
    ident = np.arange(93)
    kw = dict(iterations=12, sparseness=0.5, positivity="positive", random_state=0)

    perm_f = simlr_perm(f_views, k=k, n_perms=0, **kw)
    perm_c = simlr_perm([m[ident] for m in f_views], k=k, n_perms=0, **kw)

    for key in perm_f["stats"]:
        obs_f = perm_f["stats"][key]["observed"]
        obs_c = perm_c["stats"][key]["observed"]
        assert obs_f == obs_c, f"k={k} {key}: Fortran observed {obs_f} != C-contiguous {obs_c}"

    res_f = perm_f["simlr_result"]
    res_c = perm_c["simlr_result"]
    assert res_f["learning_rate"] == res_c["learning_rate"]
    assert torch.equal(res_f["u"], res_c["u"])
    for vf, vc in zip(res_f["v"], res_c["v"]):
        assert torch.equal(vf, vc)

    # Also verify manual worker scoring on m[ident] matches simlr_perm(..., n_perms=0) bit-for-bit
    sliced_tensors = [torch.as_tensor(m[ident]).float() for m in f_views]
    fit_worker = simlr(sliced_tensors, k=k, verbose=False, **kw)
    scored = [
        preprocess_data(m, fit_worker["scale_list"], provenance=pv)
        for m, pv in zip(sliced_tensors, fit_worker["provenance_list"])
    ]
    worker_sims = pairwise_matrix_similarity(scored, [l1_normalize_features(v) for v in fit_worker["v"]])
    for key, val in worker_sims.items():
        assert float(val) == perm_f["stats"][key]["observed"]


@pytest.mark.parametrize("method", ["simlr", "lend", "ned", "pca"])
def test_fit_and_predict_bit_identical_for_fortran_dataframes_and_c_arrays(method: str):
    c_views, f_views, df_views = _make_fortran_and_c_views(seed=42, n=64, widths=(48, 36, 24), k=2)
    rng = np.random.default_rng(99)
    conf_c = np.ascontiguousarray(rng.standard_normal((64, 2)), dtype=np.float64)
    conf_f = np.asfortranarray(conf_c)

    kw = {"random_state": 0, "confounds": conf_c}
    kw_f = {"random_state": 0, "confounds": conf_f}
    if method != "pca":
        kw["iterations"] = 8
        kw_f["iterations"] = 8

    res_c = pysimlr.fit(method, c_views, k=2, **kw)
    res_f = pysimlr.fit(method, f_views, k=2, **kw_f)
    res_df = pysimlr.fit(method, df_views, k=2, **kw_f)

    assert torch.equal(res_c["u"], res_f["u"])
    assert torch.equal(res_c["u"], res_df["u"])
    for vc, vf, vdf in zip(res_c["v"], res_f["v"], res_df["v"]):
        assert torch.equal(vc, vf)
        assert torch.equal(vc, vdf)

    pred_c = pysimlr.predict(res_c, c_views[:])
    pred_f = pysimlr.predict(res_f, f_views[:])
    pred_df = pysimlr.predict(res_df, df_views[:])
    assert torch.equal(pred_c["u"], pred_f["u"])
    assert torch.equal(pred_c["u"], pred_df["u"])
    assert pred_c["r2"] == pred_f["r2"] == pred_df["r2"]


def test_low_level_entrypoints_normalize_fortran_tensors_and_arrays():
    c_views, f_views, _ = _make_fortran_and_c_views(seed=7, n=50, widths=(40, 30), k=2)
    f_tensors = [torch.from_numpy(m).float() for m in f_views]
    assert not f_tensors[0].is_contiguous()

    # preprocess_data returns C-contiguous output even when given a Fortran-strided tensor
    p_f, prov_f = preprocess_data(f_tensors[0], ["centerAndScale", "np"])
    p_c, prov_c = preprocess_data(torch.from_numpy(c_views[0]).float(), ["centerAndScale", "np"])
    assert p_f.is_contiguous()
    assert torch.equal(p_f, p_c)
    assert set(prov_f) == set(prov_c)
    for k_prov in prov_f:
        vf_p, vc_p = prov_f[k_prov], prov_c[k_prov]
        if isinstance(vf_p, torch.Tensor):
            assert torch.equal(vf_p, vc_p)
        else:
            assert vf_p == vc_p

    # initialize_simlr across deterministic & seeded strategies (both float32 tensors & float64 arrays)
    c_tensors = [torch.from_numpy(m).float() for m in c_views]
    for itype in ("pca", "gcca", "nndsvd", "varimax", "perturbed_pca"):
        g1 = torch.Generator().manual_seed(3)
        g2 = torch.Generator().manual_seed(3)
        vf = initialize_simlr(f_tensors, k=2, initialization_type=itype, positivity="positive", generator=g1)
        vc = initialize_simlr(c_tensors, k=2, initialization_type=itype, positivity="positive", generator=g2)
        for a, b in zip(vf, vc):
            assert a.is_contiguous()
            assert torch.equal(a, b)
        g3 = torch.Generator().manual_seed(3)
        g4 = torch.Generator().manual_seed(3)
        vf64 = initialize_simlr(f_views, k=2, initialization_type=itype, positivity="positive", generator=g3)
        vc64 = initialize_simlr(c_views, k=2, initialization_type=itype, positivity="positive", generator=g4)
        for a, b in zip(vf64, vc64):
            assert a.is_contiguous()
            assert torch.equal(a, b)

    # Direct simlr / predict_simlr / predict_shared_latent / decompose_energy / estimate_rank / simlr_path
    rf = simlr(f_views, k=2, iterations=8, random_state=0)
    rc = simlr(c_views, k=2, iterations=8, random_state=0)
    assert torch.equal(rf["u"], rc["u"])
    assert torch.equal(predict_shared_latent(f_views, rf), predict_shared_latent(c_views, rc))
    pf = predict_simlr(f_views, rf)
    pc = predict_simlr(c_views, rc)
    assert torch.equal(pf["u"], pc["u"])
    assert pf["errors"] == pc["errors"]

    df_e = decompose_energy(f_views, rf)
    dc_e = decompose_energy(c_views, rc)
    assert df_e["modality_energies"] == dc_e["modality_energies"]

    torch.manual_seed(0)
    kf = estimate_rank(f_views, n_permutations=3)
    torch.manual_seed(0)
    kc = estimate_rank(c_views, n_permutations=3)
    assert kf == kc

    pf_path = simlr_path(f_views, path_model=[[0, 1]], k=2, iterations=6, random_state=0)
    pc_path = simlr_path(c_views, path_model=[[0, 1]], k=2, iterations=6, random_state=0)
    assert pf_path["consensus_correlations"] == pc_path["consensus_correlations"]

    # SVD / PCA / whitening / varimax / feature_relevance
    uf, sf, v_f = ba_svd(f_views[0], nu=2, nv=2)
    uc, sc, v_c = ba_svd(c_views[0], nu=2, nv=2)
    assert torch.equal(uf, uc) and torch.equal(sf, sc) and torch.equal(v_f, v_c)
    assert torch.equal(safe_pca(f_views[0], nc=2)["u"], safe_pca(c_views[0], nc=2)["u"])
    assert torch.equal(whiten_matrix(f_views[0], nc=2)["whitened_matrix"],
                       whiten_matrix(c_views[0], nc=2)["whitened_matrix"])
    assert torch.equal(varimax(np.asfortranarray(c_views[0][:20, :3])),
                       varimax(c_views[0][:20, :3]))

    fit_r = pysimlr.fit("simlr", c_views, k=2, iterations=6, random_state=0)
    rel_f = feature_relevance(
        fit_r, f_views, component=0, n_refits=2, n_null_refits=0,
        n_permutations=5, n_jobs=1, random_state=0,
    )
    rel_c = feature_relevance(
        fit_r, c_views, component=0, n_refits=2, n_null_refits=0,
        n_permutations=5, n_jobs=1, random_state=0,
    )
    assert np.array_equal(
        rel_f["features"]["share_train"].to_numpy(),
        rel_c["features"]["share_train"].to_numpy(),
    )
    assert np.array_equal(
        rel_f["features"]["p_train"].to_numpy(),
        rel_c["features"]["p_train"].to_numpy(),
    )
