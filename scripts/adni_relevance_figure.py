"""Tiered relevance figure (run scripts/adni_relevance_test.py first) for one view: brain (DKT cortex), forest plot of shares, lobe panel."""
import os, re, sys, numpy as np, pandas as pd
W = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path[:0] = [W + "/src", W + "/scripts", W + "/paper"]
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, ListedColormap
from matplotlib.patches import Patch
import brain_region_maps as brm


def fp(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"
from pysimlr.plotting import theme, _new, _clean, _dots, _INK, _INK2, _MUTED, _AXIS, _GRID, VIEW_COLORS
from paper_numbers import region_label

R = os.path.join(W, "docs", "audit", "evidence", "relevance") + "/"
fd, gd = pd.read_csv(R + "features.csv"), pd.read_csv(R + "groups.csv")
VIEW, COLOR = "vol_area", VIEW_COLORS[0]
f = fd[fd.view == VIEW].copy(); g = gd[gd.view == VIEW].copy()
f["tier"] = np.where((f.p_train < .05) & (f.p_test < .05), 2, np.where(f.p_train < .05, 1, 0))
dark = COLOR; light = tuple(0.45 * np.array(to_rgb(COLOR)) + 0.55)
TIER_COL = {2: dark, 1: light, 0: "#d9d8d2"}
fair = 1 / len(f)


def pretty(n):
    return region_label(n).replace(" (DKT cortex label)", "").replace(" (DKT parcellation)", " (parc.)")


# ---- brain: DKT cortex features only (one value per DKT region: the "dktcortex" variant)
vals = {}
for _, r in f.iterrows():
    m = re.match(r"^(left|right)_(.+)dktcortex$", r.feature)
    if m and r.tier > 0:
        vals[brm.dkt_id(f"{m.group(1)}_{m.group(2)}")] = float(r.tier)
cmap = ListedColormap([light, dark])
meshes = brm.fsaverage_meshes()
cols = [("left", "lateral", "L lateral"), ("left", "medial", "L medial"), ("both", "ventral", "ventral (R | L)"),
        ("right", "medial", "R medial"), ("right", "lateral", "R lateral")]
imgs = []
for hemi, view, _ in cols:
    if hemi == "both":
        a = brm._crop(brm.render(vals, meshes["left"], "left", "ventral", cmap, (1, 2)))
        b = brm._crop(brm.render(vals, meshes["right"], "right", "ventral", cmap, (1, 2)))
        h = max(a.shape[0], b.shape[0]); pad = lambda im: np.pad(im, ((0, h - im.shape[0]), (0, 0), (0, 0)))
        imgs.append(np.concatenate([pad(b), pad(a)], axis=1))
    else:
        imgs.append(brm._crop(brm.render(vals, meshes[hemi], hemi, view, cmap, (1, 2))))
aspect = [im.shape[1] / im.shape[0] for im in imgs]

with theme():
    fig = _new((15.5, 11.2), "ADNI-MCI volume / surface area, predictive component 2: where its variance concentrates",
               "Share of the variance this view explains in the component (relative weights). Fair share = 1/160. "
               "Rule (set before plotting): dark = permutation p < 0.05 in training AND in 326 held-out subjects; "
               "pale = training only; grey = neither.\nSame test on DiReCT cortical thickness: 0 of 62 regions at p < 0.05 "
               "in training, 1 held-out (chance 3.1), no lobe significant: its signal is the global thickness factor, not a "
               "region. L/R labels provisional (CIT168 DKT left/right check pending).")
    gs = fig.add_gridspec(2, len(cols), height_ratios=[1, 2.3], width_ratios=aspect)
    for c, (im, (_, _, head)) in enumerate(zip(imgs, cols)):
        ax = fig.add_subplot(gs[0, c]); ax.imshow(im); ax.axis("off"); ax.set_title(head, fontsize=9.5, color=_INK2)
    sub = gs[1, :].subgridspec(1, 2, width_ratios=[1.55, 1], wspace=0.08)

    # ---- forest: every feature passing in training, plus grey context rows
    ax = fig.add_subplot(sub[0, 0])
    show = f[f.tier > 0].sort_values("share_train", ascending=False)
    y = np.arange(len(show))
    ax.axvline(100 * fair, color=_INK2, lw=1, ls="--", zorder=1)
    ax.text(100 * fair, -1.0, " fair share", fontsize=8, color=_INK2, va="bottom")
    for yi, (_, r) in zip(y, show.iterrows()):
        c = TIER_COL[r.tier]
        ax.plot([100 * r.share_train, 100 * r.share_test], [yi, yi], color=_AXIS, lw=1.2, zorder=1)
        ax.scatter(100 * r.share_train, yi, s=58, color=c, edgecolors="#fcfcfb", linewidths=1.2, zorder=3)
        ax.scatter(100 * r.share_test, yi, s=46, facecolor="#fcfcfb", edgecolors=c if r.tier else _MUTED,
                   linewidths=1.6, zorder=3)
        ax.text(100 * max(r.share_train, r.share_test) + 0.12, yi,
                f"p {fp(r.p_train)} / {fp(r.p_test)}" + (f"   refits {r.refit_p05_freq:.0%}" if r.tier == 2 else ""),
                va="center", fontsize=7.6, color=_INK if r.tier == 2 else _INK2)
    ax.set_yticks(y); ax.set_yticklabels([pretty(n) for n in show.feature], fontsize=8.3)
    for lbl, t in zip(ax.get_yticklabels(), show.tier):
        lbl.set_color(_INK if t == 2 else _INK2); lbl.set_fontweight("semibold" if t == 2 else "normal")
    ax.set_ylim(len(show) - 0.4, -1.4); ax.set_xlim(0, 100 * show[["share_train", "share_test"]].max().max() + 2.2)
    ax.set_xlabel("% of explained variance (filled: training fit; ring: held-out subjects)")
    _clean(ax, grid="x")
    ax.set_title(f"Features with training p < 0.05: {int((f.tier > 0).sum())} of {len(f)} "
                 f"(chance: {0.05 * len(f):.0f}); replicated held-out: {int((f.tier == 2).sum())}", fontsize=10)
    ax.legend(handles=[Patch(color=dark, label="replicated (train and held-out p < 0.05)"),
                       Patch(color=light, label="training only")], loc="lower right", frameon=False, fontsize=8)

    # ---- lobes
    ax = fig.add_subplot(sub[0, 1])
    lb = g[g.group.str.startswith("lobe")].copy(); lb["name"] = lb.group.str.replace("lobe ", "")
    lb = lb.sort_values("ratio_fair", ascending=True)
    yy = np.arange(len(lb))
    ax.axvline(1.0, color=_INK2, lw=1, ls="--")
    for yi, (_, r) in zip(yy, lb.iterrows()):
        rep = (r.p_train < .05) and (r.p_test < .05)
        c = dark if rep else (light if r.p_train < .05 else "#c9c8c1")
        ax.hlines(yi, 1.0, r.ratio_fair, color=c, lw=6, alpha=0.9)
        ax.scatter(r.share_test / (r.n * fair), yi, s=46, facecolor="#fcfcfb", edgecolors=c if rep else _MUTED,
                   linewidths=1.6, zorder=3)
        ax.text(max(r.ratio_fair, r.share_test / (r.n * fair)) + 0.05, yi,
                f"{r.ratio_fair:.2f}x   p {fp(r.p_train)} / {fp(r.p_test)} / {fp(r.p_fullrefit_null)}",
                va="center", fontsize=7.8, color=_INK if rep else _INK2)
    ax.set_yticks(yy); ax.set_yticklabels([f"{n} ({k})" for n, k in zip(lb.name, lb.n)], fontsize=9)
    ax.set_xlim(0.4, 3.3); ax.set_xlabel("share ÷ fair share (bar: training; ring: held-out)")
    ax.set_title("DKT lobes  (p: training / held-out / full-refit null)", fontsize=10)
    _clean(ax, grid="x")
    out = os.path.join(W, "docs", "figures", "adni_volarea_relevance.png")
    fig.savefig(out, dpi=160); print("wrote", out)
