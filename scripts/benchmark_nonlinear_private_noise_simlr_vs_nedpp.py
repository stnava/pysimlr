import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from pysimlr.simlr import simlr
from pysimlr.deep import ned_simr, ned_simr_shared_private
from pysimlr.benchmarks.protocol import BenchmarkProtocol, run_repeated_benchmark

def generate_shared_private_data(n_samples=1000, noise=0.1):
    k_shared = 3
    k_private = 3
    d1, d2 = 50, 40
    
    u_shared = torch.randn(n_samples, k_shared)
    p1 = torch.randn(n_samples, k_private)
    p2 = torch.randn(n_samples, k_private)
    
    # Target only depends on shared
    y = u_shared[:, 0:1] * 2.0 + u_shared[:, 1:2] * 1.5 + 0.01 * torch.randn(n_samples, 1)
    
    def nonlin_map(z_shared, z_private, out_dim):
        # Combined nonlinearly: shared + private
        z = torch.cat([z_shared, z_private], dim=1)
        w = torch.randn(z.shape[1], out_dim)
        return torch.tanh(z @ w)

    x1 = nonlin_map(u_shared, p1, d1) + noise * torch.randn(n_samples, d1)
    x2 = nonlin_map(u_shared, p2, d2) + noise * torch.randn(n_samples, d2)
    
    # Normalize
    def norm(x): return (x - x.mean(0)) / (x.std(0) + 1e-6)
    return norm(x1), norm(x2), u_shared, y.numpy()

def fit_simlr(train_data, **kwargs):
    return simlr(train_data, iterations=100, energy_type="acc", **kwargs)

def fit_ned(train_data, **kwargs):
    return ned_simr(train_data, epochs=150, warmup_epochs=20, device="cpu", verbose=False, **kwargs)

def fit_nedpp(train_data, **kwargs):
    return ned_simr_shared_private(train_data, epochs=200, device="cpu", verbose=False, **kwargs)

def run_shared_private_benchmark(n_samples=1200, n_seeds=3, noise_level=0.1):
    protocol = BenchmarkProtocol(n_samples=n_samples)
    
    common_args = {"n_seeds": n_seeds, "generator_name": "shared_private_nuisance", "noise_level": noise_level, "k": 3}
    
    print("Benchmarking SIMLR...")
    df_simlr = run_repeated_benchmark(protocol, generate_shared_private_data, fit_simlr, "linear", **common_args)
    
    print("Benchmarking NED...")
    df_ned = run_repeated_benchmark(protocol, generate_shared_private_data, fit_ned, "ned", **common_args)
    
    print("Benchmarking NEDPP (Shared/Private)...")
    df_nedpp = run_repeated_benchmark(protocol, generate_shared_private_data, fit_nedpp, "nedpp", private_k=3, **common_args)
    
    df_all = pd.concat([df_simlr, df_ned, df_nedpp], ignore_index=True)
    
    summary = df_all.groupby("model")[[
        "latent_recovery_corr", "heldout_outcome_r2", "reconstruction_mse"
    ]].agg(["median", "std"])
    
    print("\nSummary Results:")
    print(summary)
    
    return df_all

if __name__ == "__main__":
    run_shared_private_benchmark()
