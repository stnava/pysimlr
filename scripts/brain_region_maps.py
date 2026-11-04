#!/usr/bin/env python3
"""ADNI-MCI DiReCT thickness feature maps, as a brain and its tabular twin.

Rendering is antsxviz's (the ANTsX house renderer: fsaverage cortex, inflated,
sulcal shading, DKT labels; see antsxviz docs/API.md). pysimlr supplies the
encoding (`feature_map_values`) shared by both figures, so values, colour scale,
selection outlines and stability are identical in the brain and the table.

The previous ~370 lines of GIFTI parsing, CIT168 -> fsaverage label transfer,
meshing and rendering in this script now live in antsxviz (consolidated there;
see pysimlr docs/tracking.md D23).

Usage:
    python scripts/brain_region_maps.py --out docs/figures/adni_dkt_feature_maps.png
Requires: pip install 'pysimlr[viz]'.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "src"), HERE]


def adni_figure(out: str):
    """DiReCT thickness, the predictive component of the Figure 3.9 fit: structure
    vs relative weight, as a brain (``out``) and its tabular twin (``*_grid.png``)."""
    import torch
    import antsxviz
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
    top = [rw["names"][i] for i in np.argsort(-rw["values"])[:4]]
    pretty = lambda n: " ".join(antsxviz.split_hemisphere(n)[::-1])
    title = f"ADNI-MCI {vn[2]}, component {j + 1}: association (structure) vs unique share (relative weight)"
    fig = pysimlr.plot_brain_map(res, inf=inf, component=j, view=2, feature_names=fnames, view_names=vn,
                                 title=title, notes={"relative_weight": "largest: " + ", ".join(map(pretty, top))})
    fig.savefig(out, dpi=170)
    grid = pysimlr.plot_feature_map_grid(res, inf=inf, component=j, view=2, feature_names=fnames, view_names=vn,
                                         split=antsxviz.split_hemisphere, groups=antsxviz.REGION_LOBE)
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
