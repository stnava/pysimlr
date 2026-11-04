#!/usr/bin/env python3
"""Paint per-region values (DKT cortical parcels) on a template brain.

pysimlr itself does not depend on ANTsPy, pyvista or sulceye (sulceye depends on
pysimlr), so brain rendering lives in this script. It uses:
  * the CIT168 T1 template and its DKT labels (~/.antspyt1w, same grid);
  * `sulceye.voxel.visualizer.extract_surface_mesh` (marching cubes, physical
    coordinates) for one mesh per hemisphere;
  * `sulceye.reporting.DKT_BASE_MAP` (region name -> left DKT id; right = +1000);
  * pyvista off-screen rendering, composed in matplotlib with `pysimlr.plotting.theme`.

Views per hemisphere: lateral, medial and ventral (entorhinal and parahippocampal
cortex are only visible medially and ventrally).

Library use:
    meshes = hemisphere_meshes()
    fig = region_map_figure({"structure": vals_a, "relative weight": vals_b}, meshes, ...)
where each ``vals`` maps an ADNI-style region name ("left_entorhinal") or a DKT
id to a value. Regions without a value are drawn light grey.

Script use (ADNI-MCI, the Figure 3.9 fit; DiReCT thickness component 2):
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


def hemisphere_meshes(spacing: float = 1.0, sigma: float = 1.0):
    """{'left'|'right': (verts (V,3) physical, faces (F,3), vertex DKT ids (V,))}."""
    import ants
    from sulceye.voxel.visualizer import extract_surface_mesh, sample_3d_image_at_physical_coords
    lab = ants.image_read(DKT)
    lab = ants.resample_image(lab, (spacing,) * 3, use_voxels=False, interp_type=1)
    arr = lab.numpy()
    # Physical x of every voxel; the midline is midway between the hemispheres' cortex.
    idx = np.indices(arr.shape).reshape(3, -1).T
    xs = (np.array(lab.origin) + (idx * np.array(lab.spacing)) @ np.array(lab.direction).T)[:, 0].reshape(arr.shape)
    mid = 0.5 * (xs[(arr > 1000) & (arr < 1100)].mean() + xs[(arr > 2000) & (arr < 2100)].mean())
    out = {}
    for hemi, lo in (("left", 1000), ("right", 2000)):
        # The cortical labels form an open shell (no labels on the medial wall), so a
        # medial or ventral camera looks through the gap at the inside of the lateral
        # cortex. Add this side's deep grey nuclei and ventricles (not cerebellum
        # 6/7/8/45/46/47/630-632, brainstem 16, ventral DC 28/60, 4th ventricle 15
        # or CSF 24), close and fill into a solid hemisphere; medial-wall vertices
        # then carry no cortical label and render grey.
        side = (xs < mid) if hemi == "left" else (xs >= mid)
        drop = [6, 7, 8, 15, 16, 24, 28, 45, 46, 47, 60, 630, 631, 632]
        sub = (arr > 0) & (arr < 1000) & ~np.isin(arr, drop) & side
        solid = ((arr > lo) & (arr < lo + 100)) | sub
        mask = lab.new_image_like(solid.astype(np.float32))
        mask = ants.iMath(ants.iMath(mask, "MC", 5), "FillHoles")
        m = ants.smooth_image(mask, sigma, sigma_in_physical_coordinates=False)
        verts, faces, normals = extract_surface_mesh(m, step_size=1)
        verts, faces, normals = _largest_component(verts, faces, normals)
        # Each vertex takes the label of the nearest cortical voxel of this hemisphere
        # (within 4 mm); medial-wall and subcortical vertices stay unlabelled (0).
        from scipy.ndimage import distance_transform_edt
        ctx = (arr > lo) & (arr < lo + 100)
        dist, nearest = distance_transform_edt(~ctx, return_indices=True)
        vox = np.round(((verts - np.array(lab.origin)) @ np.linalg.inv(np.array(lab.direction)).T)
                       / np.array(lab.spacing)).astype(int)
        vox = np.clip(vox, 0, np.array(arr.shape) - 1)
        n_idx = tuple(nearest[(axis,) + tuple(vox.T)] for axis in range(3))
        ids = arr[n_idx].astype(int)
        ids[dist[tuple(vox.T)] * spacing > 4.0] = 0
        out[hemi] = (verts, faces, ids)
    return out


def _largest_component(verts, faces, normals):
    """Drop the small disconnected islands marching cubes leaves near the surface."""
    import pyvista as pv
    poly = pv.PolyData(verts, np.c_[np.full(len(faces), 3), faces].ravel())
    poly.point_data["n"] = normals
    big = poly.connectivity("largest").extract_surface(algorithm="dataset_surface")
    return np.asarray(big.points), big.faces.reshape(-1, 4)[:, 1:], np.asarray(big.point_data["n"])


def _cameras(verts, hemi):
    """Camera (position, focal point, view-up) per view. In this template's physical
    space the left hemisphere lies at -x, +y is posterior and +z superior (checked:
    left superior frontal mean x -12.7, lateral occipital y +83, entorhinal z -29)."""
    c = verts.mean(0)
    span = np.ptp(verts, axis=0).max()
    d = 2.6 * span
    out_x = -1.0 if hemi == "left" else 1.0          # lateral side of this hemisphere
    return {
        "lateral": (c + np.array([out_x * d, 0, 0]), c, (0, 0, 1)),
        "medial": (c - np.array([out_x * d, 0, 0]), c, (0, 0, 1)),
        "ventral": (c - np.array([0, 0, d]), c, (0, -1, 0)),
    }


def render(values_by_id: dict, mesh, hemi: str, view: str, cmap, clim, selected=(), size=(700, 560)):
    """One off-screen render of one hemisphere, returned as an RGBA array.
    Regions in ``selected`` (DKT ids) get a dark outline."""
    import pyvista as pv
    verts, faces, ids = mesh
    # One value per triangle (its majority vertex label), as cell data: region
    # boundaries stay crisp, and no colour is interpolated between a valued region
    # and an unvalued neighbour (vertex interpolation swept through the colormap).
    fl = ids[faces]
    face_id = np.where(fl[:, 1] == fl[:, 2], fl[:, 1], fl[:, 0])
    vals = np.array([values_by_id.get(int(i), np.nan) for i in face_id], float)
    poly = pv.PolyData(verts, np.c_[np.full(len(faces), 3), faces].ravel())
    poly.cell_data["v"] = vals
    pl = pv.Plotter(off_screen=True, window_size=size)
    pl.set_background("white")
    pl.add_mesh(poly, scalars="v", cmap=cmap, clim=clim, nan_color="#ecebe6", smooth_shading=True,
                show_scalar_bar=False, ambient=0.25, diffuse=0.75, specular=0.08)
    for rid in selected:
        cells = np.where(face_id == int(rid))[0]
        if len(cells):
            edge = poly.extract_cells(cells).extract_surface(algorithm="dataset_surface").extract_feature_edges(
                boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=False)
            if edge.n_points:
                pl.add_mesh(edge, color="#0b0b0b", line_width=3.5, render_lines_as_tubes=True)
    pos, foc, up = _cameras(verts, hemi)[view]
    pl.camera_position = [tuple(pos), tuple(foc), up]
    pl.camera.zoom(1.35)
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
    cols = [("left", "lateral", "L lateral"), ("left", "medial", "L medial"), ("both", "ventral", "ventral"),
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
                row.append(np.concatenate([pad(a_), pad(b_)], axis=1))   # left hemisphere on the left
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
    fig = region_map_figure(enc, hemisphere_meshes(), 2, title, notes)
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
