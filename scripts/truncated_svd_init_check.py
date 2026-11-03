"""Equivalence check for the truncated init SVD (svd.top_right_singular_vectors): run once on
the base tree and once on the new tree, then compare latent R2 per cell. Rule: mean dR2 >= -0.002,
every cell >= -0.02, converged count not lower. Result: docs/audit/evidence/truncated_svd_init_check.csv."""
import sys, time, warnings, itertools, numpy as np, torch, json
warnings.simplefilter("ignore")
from pysimlr import fit
from pysimlr.utils import procrustes_r2
rows=[]
cells=list(itertools.product((1200,3000),((1500,1200,1000),(4000,2000,1200)),(3,6),(0.5,2.0),("either","positive")))
for s,(n,ps,k,noise,pos) in enumerate(cells):
    g=np.random.default_rng(s); z=g.normal(size=(n,k))
    vs=[(z@g.normal(size=(k,p))+noise*g.normal(size=(n,p))).astype(np.float32) for p in ps]
    t=time.perf_counter(); r=fit("simlr", vs, k, positivity=pos, random_state=s); dt=time.perf_counter()-t
    u=r["u"]; u=u[0] if isinstance(u,list) else u
    rows.append(dict(cell=s,n=n,p=ps[0],k=k,noise=noise,pos=pos,r2=procrustes_r2(torch.tensor(z,dtype=torch.float32),torch.as_tensor(u).float()),
                     energy=float(r["best_energy"]),conv=r["convergence"]["converged"],steps=r["convergence"]["n_steps"],time=dt))
    print(json.dumps(rows[-1]),flush=True)
json.dump(rows,open(sys.argv[1],"w"))
