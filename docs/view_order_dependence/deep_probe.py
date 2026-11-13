import sys, warnings, numpy as np, torch, pysimlr
from pysimlr import deep, flows
sys.path.insert(0,'tests')
from test_view_order_sign_alignment import _noise_views

def view_key(model, i):
    return model.input_dims[i] if hasattr(model,'input_dims') else None

def patch(cls):
    class Fixed(cls):
        def __init__(self, input_dims, *a, **kw):
            super().__init__(input_dims, *a, **kw)
            n = len(input_dims)
            for name, mod in self.named_children():
                if isinstance(mod, torch.nn.ModuleList) and len(mod) == n:
                    for i, sub in enumerate(mod):
                        g = torch.Generator().manual_seed(1000 + int(input_dims[i]))
                        for m in sub.modules():
                            if isinstance(m, torch.nn.Linear):
                                bound = 1.0 / np.sqrt(m.in_features)
                                with torch.no_grad():
                                    m.weight.copy_((torch.rand(m.weight.shape, generator=g) * 2 - 1) * bound)
                                    m.bias.copy_((torch.rand(m.bias.shape, generator=g) * 2 - 1) * bound)
    return Fixed

def run(fn, patch_target, order, views, iters, sa, fixed=True, **kw):
    import contextlib
    mod, name = patch_target
    orig = getattr(mod, name)
    if fixed: setattr(mod, name, patch(orig))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fn([views[n] for n in order], k=1, iterations=iters, random_state=3, sign_alignment=sa, **kw)
    finally:
        setattr(mod, name, orig)

def ud(a, b):
    a=np.asarray(a.detach().double()).ravel(); b=np.asarray(b.detach().double()).ravel()
    return min(np.abs(a-b).max(), np.abs(a+b).max())

o1=[f"V{i}" for i in range(5)]+["Root"]; o2=["V0","V1","V2","V4","V3","Root"]
methods=[("lend",deep.lend_simr,(deep,"LENDSiMRModel"),dict(dropout=0.0,batch_size=174)),
 ("ned",deep.ned_simr,(deep,"NEDSiMRModel"),dict(dropout=0.0,batch_size=174)),
 ("ned_sp",deep.ned_simr_shared_private,(deep,"NEDSharedPrivateSiMRModel"),dict(dropout=0.0,batch_size=174)),
 ("flow_v",flows.flow_simr_v,(flows,"FlowSiMRVModel"),dict(batch_size=174)),
 ("flow",flows.flow_simr,(flows,"FlowSiMRModel"),dict(batch_size=174))]
import argparse
ap=argparse.ArgumentParser()
ap.add_argument("methods",nargs="*"); ap.add_argument("--n-seeds",type=int,default=2); ap.add_argument("--iters",type=int,default=15)
ap.add_argument("--smoke",action="store_true")
args=ap.parse_args()
if args.smoke: args.n_seeds=2; args.iters=2
which=args.methods or [m[0] for m in methods]
for nm,fn,tgt,kw in methods:
    if nm not in which: continue
    for seed in range(3,3+args.n_seeds):
        views=_noise_views(seed)
        for iters in (1,args.iters):
            row=[]
            for fixed in (True,False):
                for sa in ("greedy","spectral"):
                    try:
                        r1=run(fn,tgt,o1,views,iters,sa,fixed,**kw); r2=run(fn,tgt,o2,views,iters,sa,fixed,**kw)
                        # un-permute latents: o2 view j -> o1 index
                        idx=[o2.index(n) for n in o1]
                        l2=[r2["latents"][j] for j in idx]
                        lat=max(ud(a,b) for a,b in zip(r1["latents"],l2))
                        row.append(f"{'fixedinit' if fixed else 'freeinit'}/{sa}: u {ud(r1['u'],r2['u']):.2e} lat {lat:.2e}")
                    except Exception as e:
                        row.append(f"{sa} ERR {type(e).__name__}: {str(e)[:80]}")
            print(nm,"seed",seed,"iters",iters,"|"," | ".join(row),flush=True)
