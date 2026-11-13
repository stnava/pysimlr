import argparse, warnings, numpy as np, torch, pysimlr
from pysimlr.utils import align_column_signs
ap = argparse.ArgumentParser(); ap.add_argument("--n-draws", type=int, default=500); ap.add_argument("--n-fits", type=int, default=6); ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
if a.smoke: a.n_draws = 2; a.n_fits = 2
g = torch.Generator().manual_seed(0)
n_eq = n_sign = 0; worst_rel = 0.0; worst_cols = 0.0
for _ in range(a.n_draws):
    x = torch.randn(40, 4, generator=g, dtype=torch.float64); y = 0.5 * x * torch.randint(0, 2, (1, 4), generator=g) .double().mul(2).sub(1) + torch.randn(40, 4, generator=g, dtype=torch.float64)
    gr = align_column_signs([x, y], method="greedy"); sp = align_column_signs([x, y], method="spectral")
    # relative sign between the two aligned views is identical
    rel_g = torch.sign((gr[0] * gr[1]).sum(0)); rel_s = torch.sign((sp[0] * sp[1]).sum(0))
    worst_rel = max(worst_rel, float((rel_g - rel_s).abs().max()))
    n_eq += int(all(torch.equal(p, q) for p, q in zip(gr, sp)))
    # up to a per-column global sign
    gsign = torch.sign((gr[0] * sp[0]).sum(0))
    worst_cols = max(worst_cols, float(max((gr[0] - sp[0] * gsign).abs().max(), (gr[1] - sp[1] * gsign).abs().max())))
print(f"align_column_signs, 2 views, {a.n_draws} draws (k=4): bitwise equal {n_eq}; relative sign identical (max diff {worst_rel}); equal up to per-column global sign, max |diff| {worst_cols:.1e}")
for k in (1, 2):
    for seed in range(a.n_fits):
        rng = np.random.default_rng(seed); z = rng.standard_normal((120, k))
        v = [z @ rng.standard_normal((k, p)) * 0.5 + rng.standard_normal((120, p)) for p in (20, 30)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = {sa: pysimlr.fit("simlr", v, k, sign_alignment=sa, random_state=0) for sa in ("greedy", "spectral")}
        ug, us = (r[s]["u"].detach().double().numpy() for s in ("greedy", "spectral"))
        sg = np.sign((ug * us).sum(0))
        print(f"fit k={k} seed {seed}: |u_greedy - u_spectral| up to column sign {np.abs(ug - us * sg).max():.2e}; equal without sign flip: {np.abs(ug - us).max():.2e}; latents",
              max(float(np.abs(a_.detach().double().numpy() * (np.sign((a_.detach().double().numpy() * b_.detach().double().numpy()).sum(0))) - b_.detach().double().numpy()).max()) for a_, b_ in zip(r['greedy']['latents'], r['spectral']['latents'])))
