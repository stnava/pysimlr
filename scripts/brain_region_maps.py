#!/usr/bin/env python3
"""Paint per-region values (DKT cortical parcels) on the fsaverage cortex.

pysimlr itself does not depend on ANTsPy, pyvista, syntx or sulceye (sulceye
depends on pysimlr), so brain rendering lives in this script.

Surface: FreeSurfer fsaverage (41k vertices per hemisphere, TemplateFlow
``tpl-fsaverage``), shown inflated (the white surface smoothed with a Taubin
filter, which keeps vertex correspondence) and shaded by sulcal depth; ``pial``
is also available. GIFTI geometry is read with a small XML parser (no nibabel);
TemplateFlow's RAS coordinates are converted to ANTs' LPS (x, y negated) --
verified: 100% of mid-thickness vertices fall inside the fsaverage brain mask.

Labels: the CIT168 template's DKT parcellation (~/.antspyt1w), registered to the
fsaverage T1 with `syntx.registration` (SyN) and warped with genericLabel; each
vertex takes the nearest cortical label within 3 mm of its mid-thickness point
(84% of vertices; the rest is mostly the medial wall). Region identity comes from
the label (id mod 1000) and the hemisphere from the surface file: CIT168's
"left" labels (1xxx) lie at physical -x, which in LPS is anatomical right, so a
label's own hemisphere code is not trusted here (reported upstream, not fixed).
The registration and vertex labels are cached in ~/.cache/pysimlr.

Library use:
    meshes = fsaverage_meshes()
    fig = region_map_figure(pysimlr.feature_map_values(...), meshes, view, title)

Script use (ADNI-MCI, the Figure 3.9 fit; DiReCT thickness, predictive component):
    python scripts/brain_region_maps.py --out docs/figures/adni_dkt_feature_maps.png
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.expanduser("~/.antspyt1w/CIT168_T1w_700um_pad_adni.nii.gz")
DKT = os.path.expanduser("~/.antspyt1w/CIT168_T1w_700um_pad_adni_dkt.nii.gz")
VIEWS = ("lateral", "medial", "ventral")


def dkt_id(name: str) -> int:
    """ADNI-style region name ('left_entorhinal') -> FreeSurfer DKT id (1006)."""
    from sulceye.reporting import DKT_BASE_MAP
    side = 1000 if name.startswith("left_") else (2000 if name.startswith("right_") else None)
    base = re.sub(r"^(left|right)_", "", name)
    if side is None or base not in DKT_BASE_MAP:
        raise KeyError(f"not a DKT cortical region name: {name!r}")
    return DKT_BASE_MAP[base] - 1000 + side


FSAVG = os.path.expanduser("~/.cache/templateflow/tpl-fsaverage/")
CACHE = os.path.expanduser("~/.cache/pysimlr/")


def read_gifti_surface(path: str):
    """(vertices (V,3) float, faces (F,3) int) from a GIFTI surface, RAS as stored."""
    import base64
    import zlib
    s = open(path, "rb").read().decode("utf-8")
    if not s:
        raise FileNotFoundError(f"{path} is empty (a TemplateFlow placeholder); download it first.")
    arrs = []
    for m in re.finditer(r"<DataArray(.*?)>(.*?)</DataArray>", s, re.S):
        attrs, body = m.groups()
        enc = re.search(r'Encoding="([^"]+)"', attrs).group(1)
        dt = {"NIFTI_TYPE_FLOAT32": "f4", "NIFTI_TYPE_INT32": "i4"}[re.search(r'DataType="([^"]+)"', attrs).group(1)]
        nd = int(re.search(r'Dimensionality="(\d+)"', attrs).group(1))
        dims = [int(re.search(fr'Dim{i}="(\d+)"', attrs).group(1)) for i in range(nd)]
        big = re.search(r'Endian="BigEndian"', attrs) is not None
        raw = base64.b64decode(re.search(r"<Data>(.*?)</Data>", body, re.S).group(1).strip())
        if enc == "GZipBase64Binary":
            raw = zlib.decompress(raw)
        elif enc != "Base64Binary":
            raise ValueError(f"unsupported GIFTI encoding {enc}")
        arrs.append(np.frombuffer(raw, dtype=(">" if big else "<") + dt).reshape(dims))
    return arrs[0].astype(float), arrs[1].astype(int)


def _fsaverage_labels():
    """DKT region id (1xxx left / 2xxx right by surface) per fsaverage-41k vertex, cached."""
    path = CACHE + "fsaverage41k_dkt_from_cit168.npz"
    if os.path.exists(path):
        z = np.load(path)
        return {"left": z["left"], "right": z["right"]}
    import ants
    import syntx
    from scipy.ndimage import distance_transform_edt
    os.makedirs(CACHE, exist_ok=True)
    fix = ants.image_read(FSAVG + "tpl-fsaverage_res-01_T1w.nii.gz")
    fix = ants.iMath(fix * ants.image_read(FSAVG + "tpl-fsaverage_res-01_desc-brain_mask.nii.gz"), "Normalize")
    mov = ants.iMath(ants.resample_image(ants.image_read(TEMPLATE), (1, 1, 1), use_voxels=False, interp_type=0),
                     "Normalize")
    reg = syntx.registration(fix, mov, type_of_transform="SyN")
    lab = ants.apply_transforms(fix, ants.image_read(DKT), reg["fwdtransforms"], interpolator="genericLabel")
    A = lab.numpy().astype(int)
    o, sp, dm = np.array(lab.origin), np.array(lab.spacing), np.array(lab.direction)
    ctx = (A > 1000) & (A < 2100) & ((A % 1000) < 100)
    dist, near = distance_transform_edt(~ctx, return_indices=True)
    out = {}
    for hemi, h, lo in (("left", "L", 1000), ("right", "R", 2000)):
        pial, _ = read_gifti_surface(FSAVG + f"tpl-fsaverage_hemi-{h}_den-41k_pial.surf.gii")
        white, _ = read_gifti_surface(FSAVG + f"tpl-fsaverage_hemi-{h}_den-41k_white.surf.gii")
        mid = 0.5 * (pial + white) * np.array([-1.0, -1.0, 1.0])            # RAS -> LPS
        iv = tuple(np.clip(np.round(((mid - o) @ np.linalg.inv(dm).T) / sp).astype(int), 0,
                           np.array(A.shape) - 1).T)
        ids = A[tuple(near[(ax,) + iv] for ax in range(3))] % 1000 + lo
        ids[dist[iv] > 3] = 0
        out[hemi] = ids
    np.savez(path, **out)
    return out


def _clean_surface_labels(ids, faces, rounds: int = 3, max_hole: int = 400, min_island: int = 30):
    """Tidy labels transferred from a volume onto a mesh. Fill unlabelled
    components smaller than ``max_hole`` vertices from their labelled neighbours
    (the medial wall, one large component, stays unlabelled), keep only the
    largest connected piece of each region (DKT regions are contiguous patches)
    and refill the rest from neighbours, then a few rounds of majority vote."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(ids)
    i = np.r_[faces[:, 0], faces[:, 1], faces[:, 2], faces[:, 1], faces[:, 2], faces[:, 0]]
    j = np.r_[faces[:, 1], faces[:, 2], faces[:, 0], faces[:, 0], faces[:, 1], faces[:, 2]]
    adj = coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)).tocsr()
    ids = ids.copy()
    un = ids == 0
    sub = adj[un][:, un]
    ncomp, comp = connected_components(sub, directed=False)
    sizes = np.bincount(comp, minlength=ncomp)
    fill = np.zeros(n, bool)
    fill[np.where(un)[0][sizes[comp] < max_hole]] = True
    for _ in range(50):                                   # grow labels into the small holes
        todo = np.where(fill & (ids == 0))[0]
        if not len(todo):
            break
        for v in todo:
            nb = ids[adj.indices[adj.indptr[v]:adj.indptr[v + 1]]]
            nb = nb[nb > 0]
            if len(nb):
                ids[v] = np.bincount(nb).argmax()
    # Islands: a region's connected piece smaller than ``min_island`` vertices is
    # registration noise; clear it and let the neighbours grow back over it.
    for lab_ in np.unique(ids[ids > 0]):
        on = ids == lab_
        nc, cc = connected_components(adj[on][:, on], directed=False)
        sz = np.bincount(cc, minlength=nc)
        # Each DKT region is one contiguous cortical patch: keep its largest piece.
        small = np.where(on)[0][(sz[cc] < min_island) | (cc != sz.argmax())]
        ids[small] = 0
        fill[small] = True
    for _ in range(50):
        todo = np.where(fill & (ids == 0))[0]
        if not len(todo):
            break
        for v in todo:
            nb = ids[adj.indices[adj.indptr[v]:adj.indptr[v + 1]]]
            nb = nb[nb > 0]
            if len(nb):
                ids[v] = np.bincount(nb).argmax()
    for _ in range(rounds):                               # majority vote over each vertex's neighbours
        new = ids.copy()
        for v in np.where(ids > 0)[0]:
            nb = ids[adj.indices[adj.indptr[v]:adj.indptr[v + 1]]]
            nb = nb[nb > 0]
            if len(nb):
                vals, cnt = np.unique(np.r_[nb, ids[v]], return_counts=True)
                new[v] = vals[cnt.argmax()]
        ids = new
    return ids


def fsaverage_meshes(surface: str = "inflated", smooth_iter: int = 400):
    """{'left'|'right': (verts LPS, faces, vertex DKT ids, sulcal shade in [0, 1])}."""
    import pyvista as pv
    labels = _fsaverage_labels()
    out = {}
    for hemi, h in (("left", "L"), ("right", "R")):
        white, faces = read_gifti_surface(FSAVG + f"tpl-fsaverage_hemi-{h}_den-41k_white.surf.gii")
        pial, _ = read_gifti_surface(FSAVG + f"tpl-fsaverage_hemi-{h}_den-41k_pial.surf.gii")
        cells = np.c_[np.full(len(faces), 3), faces].ravel()
        w = pv.PolyData(white * np.array([-1.0, -1.0, 1.0]), cells)
        infl = w.smooth_taubin(n_iter=smooth_iter, pass_band=0.0005, normalize_coordinates=True)
        infl = infl.compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True)
        # Sulcal depth: how far the folded surface lies inside its inflated version.
        depth = -((w.points - infl.points) * infl.point_data["Normals"]).sum(1)
        shade = np.clip((depth - np.percentile(depth, 5)) / np.ptp(np.percentile(depth, [5, 95])), 0, 1)
        verts = np.asarray(infl.points) if surface == "inflated" else pial * np.array([-1.0, -1.0, 1.0])
        out[hemi] = (verts, faces, _clean_surface_labels(labels[hemi], faces), shade)
    return out


def _cameras(verts, hemi):
    """Camera (position, focal point, view-up) per view in LPS (+x left, +y
    posterior, +z superior)."""
    c = verts.mean(0)
    d = 2.6 * np.ptp(verts, axis=0).max()
    out_x = 1.0 if hemi == "left" else -1.0          # lateral side of this hemisphere
    return {
        "lateral": (c + np.array([out_x * d, 0, 0]), c, (0, 0, 1)),
        "medial": (c - np.array([out_x * d, 0, 0]), c, (0, 0, 1)),
        "ventral": (c - np.array([0, 0, d]), c, (0, -1, 0)),
    }


def render(values_by_id: dict, mesh, hemi: str, view: str, cmap, clim, selected=(), size=(760, 600)):
    """One off-screen render of one hemisphere as an RGBA array. Vertex colour is
    the map colour (grey where a vertex has no value), darkened slightly in sulci
    so the folding stays legible; regions in ``selected`` get a dark outline."""
    import pyvista as pv
    verts, faces, ids, shade = mesh
    vals = np.array([values_by_id.get(int(i), np.nan) for i in ids], float)
    t = np.clip((vals - clim[0]) / max(clim[1] - clim[0], 1e-12), 0, 1)
    rgb = np.asarray(cmap(np.nan_to_num(t)))[:, :3]
    base = np.array([0.90, 0.895, 0.875])
    rgb[~np.isfinite(vals)] = base
    rgb *= (1.0 - 0.28 * shade)[:, None]                  # sulci darker, gyri lighter
    poly = pv.PolyData(verts, np.c_[np.full(len(faces), 3), faces].ravel())
    poly.point_data["rgb"] = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    pl = pv.Plotter(off_screen=True, window_size=size)
    pl.set_background("white")
    pl.add_mesh(poly, scalars="rgb", rgb=True, smooth_shading=True, ambient=0.3, diffuse=0.7, specular=0.05)
    for rid in selected:
        inside = np.where(np.all(ids[faces] == int(rid), axis=1))[0]
        if len(inside):
            edge = poly.extract_cells(inside).extract_surface(algorithm="dataset_surface").extract_feature_edges(
                boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=False)
            if edge.n_points:
                pl.add_mesh(edge, color="#0b0b0b", line_width=4, render_lines_as_tubes=True)
    pos, foc, up = _cameras(verts, hemi)[view]
    pl.camera_position = [tuple(pos), tuple(foc), up]
    pl.camera.zoom(1.1 if view == "ventral" else 1.4)    # seen from below, the long axis is vertical
    img = pl.screenshot(return_img=True, transparent_background=True)
    pl.close()
    return img


def _crop(img, pad=6):
    a = img[..., 3] > 0
    ys, xs = np.where(a)
    if not len(ys):
        return img
    return img[max(ys.min() - pad, 0):ys.max() + pad, max(xs.min() - pad, 0):xs.max() + pad]


DKT_LOBES = {
    "frontal": ["superior_frontal", "rostral_middle_frontal", "caudal_middle_frontal", "pars_opercularis",
                "pars_triangularis", "pars_orbitalis", "lateral_orbitofrontal", "medial_orbitofrontal",
                "precentral", "paracentral"],
    "parietal": ["postcentral", "superior_parietal", "inferior_parietal", "supramarginal", "precuneus"],
    "temporal": ["superior_temporal", "middle_temporal", "inferior_temporal", "transverse_temporal", "fusiform",
                 "entorhinal", "parahippocampal"],
    "occipital": ["lateral_occipital", "lingual", "cuneus", "pericalcarine"],
    "cingulate": ["rostral_anterior_cingulate", "caudal_anterior_cingulate", "posterior_cingulate",
                  "isthmus_cingulate"],
    "insula": ["insula"],
}
REGION_LOBE = {r.replace("_", " "): lobe for lobe, rs in DKT_LOBES.items() for r in rs}


def split_hemisphere(name: str):
    """'left_entorhinal' -> ('entorhinal', 'L'): the grid's row and column."""
    side = "L" if name.startswith("left_") else ("R" if name.startswith("right_") else "")
    return re.sub(r"^(left|right)_", "", name).replace("_", " "), side


def region_map_figure(enc: dict, meshes, view: int, title: str, notes: dict | None = None):
    """Brain twin of `pysimlr.plot_feature_map_grid`: ``enc`` is
    `pysimlr.feature_map_values(...)` (names must be ADNI-style DKT region names).
    Rows = maps, each on its own 0..max scale of `pysimlr.map_colormap(view)`;
    columns = L lateral, L medial, ventral, R medial, R lateral; selected regions
    (>= 80% of refits) outlined."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    sys.path.insert(0, os.path.join(HERE, "..", "src"))
    from pysimlr.plotting import theme, _new, _INK, _INK2, _MUTED, _stability_color, map_colormap
    cmap = map_colormap(view)
    cols = [("left", "lateral", "L lateral"), ("left", "medial", "L medial"), ("both", "ventral", "ventral (R | L)"),
            ("right", "medial", "R medial"), ("right", "lateral", "R lateral")]
    imgs = {}
    for kind, e in enc.items():
        by_id = {dkt_id(n): float(v) for n, v in zip(e["names"], e["values"]) if np.isfinite(v)}
        sel = [dkt_id(n) for n, s_ in zip(e["names"], e["selected"])] if e["selected"] is not None else []
        sel = [i for i, s_ in zip(sel, e["selected"]) if s_] if sel else []
        clim = (0.0, e["vmax"])
        row = []
        for hemi, view_, _ in cols:
            if hemi == "both":
                a_ = _crop(render(by_id, meshes["left"], "left", "ventral", cmap, clim, sel))
                b_ = _crop(render(by_id, meshes["right"], "right", "ventral", cmap, clim, sel))
                h = max(a_.shape[0], b_.shape[0])
                pad = lambda im: np.pad(im, ((0, h - im.shape[0]), (0, 0), (0, 0)))
                # Seen from below (anterior up), the left hemisphere is on the viewer's right.
                row.append(np.concatenate([pad(b_), pad(a_)], axis=1))
            else:
                row.append(_crop(render(by_id, meshes[hemi], hemi, view_, cmap, clim, sel)))
        imgs[kind] = row
    aspect = [max(imgs[k][c].shape[1] / imgs[k][c].shape[0] for k in imgs) for c in range(len(cols))]
    any_sel = any(e["selected"] is not None for e in enc.values())
    with theme():
        fig = _new((sum(aspect) * 2.3 + 1.2, 2.55 * len(enc) + 1.2), title,
                   "Colour: each map on its own scale from 0 to its maximum"
                   + ("; dark outline: selected in ≥ 80% of refits" if any_sel else "") + ".")
        gs = fig.add_gridspec(len(enc), len(cols) + 1, width_ratios=aspect + [0.07])
        for r, (kind, e) in enumerate(enc.items()):
            for c, (_, _, head) in enumerate(cols):
                ax = fig.add_subplot(gs[r, c]); ax.axis("off")
                ax.imshow(imgs[kind][c])
                if r == 0:
                    ax.set_title(head, fontsize=9.5, color=_INK2)
                if c == 0:
                    st = e["stability"]
                    ax.text(-0.03, 0.5, e["label"] + (f"\nstability {st:.2f}" if st is not None else ""),
                            transform=ax.transAxes, ha="right", va="center", fontsize=9.6, weight="semibold",
                            color=_stability_color(st) if st is not None else _INK)
            cax = fig.add_subplot(gs[r, -1])
            cb = fig.colorbar(ScalarMappable(Normalize(0, e["vmax"]), cmap), cax=cax)
            cb.outline.set_visible(False); cb.set_label(e["units"], fontsize=8, color=_INK2)
            if notes and kind in notes:
                t = fig.axes[-2].text(1.0, -0.04, notes[kind], transform=fig.axes[-2].transAxes, ha="right",
                                      va="top", fontsize=8, color=_MUTED)
                t.set_in_layout(False)            # a long note must not shrink its panel
    return fig


def adni_figure(out: str):
    """DiReCT thickness, the predictive component of the Figure 3.9 fit: structure
    vs relative weight, as a brain (``out``) and its tabular twin (``*_grid.png``)."""
    sys.path[:0] = [os.path.join(HERE, "..", "src"), HERE]
    import torch
    import pysimlr
    import paper_adni_ablations as paa
    torch.set_num_threads(1)
    d = paa.protocol_b(42)
    views = paa.fit_views(d)
    inf = pysimlr.loading_inference("simlr", views, 3, n_subsamples=20, random_state=42, n_jobs=8,
                                    positivity="positive", sparseness=0.85, topology="graph",
                                    path_graph=paa.PATH_GRAPH, maps=("relative_weight",))
    res = inf["result"]
    u = res["u"].double().numpy() @ res["loading_rotation"].double().numpy()
    j = int(np.argmax([abs(np.corrcoef(u[:, c], d["pc1_tr"])[0, 1]) for c in range(u.shape[1])]))
    fnames = [[f"v{m}.{i}" for i in range(x.shape[1])] for m, x in enumerate(views)]
    fnames[2] = list(d["names"][2])
    vn = ["Volume", "Vol/area", "DiReCT thickness", "Cognition"]
    enc = pysimlr.feature_map_values(res, inf=inf, component=j, view=2, feature_names=fnames, view_names=vn)
    rw = enc["relative_weight"]
    pretty = lambda n: " ".join(split_hemisphere(n)[::-1])
    top = [rw["names"][i] for i in np.argsort(-rw["values"])[:4]]
    notes = {"relative_weight": "largest: " + ", ".join(pretty(n) for n in top)}
    title = f"ADNI-MCI {vn[2]}, component {j + 1}: association (structure) vs unique share (relative weight)"
    fig = region_map_figure(enc, fsaverage_meshes(), 2, title, notes)
    fig.savefig(out, dpi=170)
    grid = pysimlr.plot_feature_map_grid(res, inf=inf, component=j, view=2, feature_names=fnames, view_names=vn,
                                         split=split_hemisphere, groups=REGION_LOBE, value_fmt="{:.2f}")
    gout = out.replace(".png", "_grid.png")
    grid.savefig(gout, dpi=170)
    print("wrote", out, gout)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(HERE, "..", "docs", "figures", "adni_dkt_feature_maps.png"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    adni_figure(a.out)


if __name__ == "__main__":
    main()
