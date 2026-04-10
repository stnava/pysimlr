import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from pysimlr.simlr import simlr
from pysimlr.deep import ned_simr
from pysimlr.benchmarks.protocol import BenchmarkProtocol, run_repeated_benchmark

def generate_nonlinear_data(n_samples=1000, noise=0.1):
    k = 5
    d1, d2 = 50, 40
    u = torch.randn(n_samples, k)
    
    # Nonlinear forward maps: Inverse must genuinely require nonlinearity
    def nonlin_map(z, out_dim):
        w1 = torch.randn(z.shape[1], out_dim)
        w2 = torch.randn(z.shape[1], out_dim)
        # Use mixture of sin and cubic to make inversion hard for linear models
        res = torch.sin(z @ w1) + 0.5 * (z @ w2)**3
        return res

    x1 = nonlin_map(u, d1) + noise * torch.randn(n_samples, d1)
    x2 = nonlin_map(u, d2) + noise * torch.randn(n_samples, d2)
    
    # Target depends on shared signal
    y = u[:, 0:1] * 2.0 + u[:, 1:2] * 1.5 + 0.01 * torch.randn(n_samples, 1)
    
    # Normalize
    def norm(x): return (x - x.mean(0)) / (x.std(0) + 1e-6)
    return norm(x1), norm(x2), u, y.numpy()

def fit_simlr(train_data, **kwargs):
    return simlr(train_data, iterations=100, energy_type="acc", **kwargs)

def fit_ned(train_data, **kwargs):
    # Ensure sufficient epochs for nonlinear learning
    return ned_simr(train_data, epochs=200, warmup_epochs=20, device="cpu", verbose=False, **kwargs)

def run_ned_vs_simlr_benchmark(n_samples=1200, n_seeds=3, noise_level=0.1):
    protocol = BenchmarkProtocol(n_samples=n_samples)
    
    common_args = {"n_seeds": n_seeds, "generator_name": "strongly_nonlinear", "noise_level": noise_level, "k": 5}
    
    print("Benchmarking SIMLR (Linear)...")
    df_simlr = run_repeated_benchmark(protocol, generate_nonlinear_data, fit_simlr, "linear", **common_args)
    
    print("Benchmarking NED (Nonlinear)...")
    df_ned = run_repeated_benchmark(protocol, generate_nonlinear_data, fit_ned, "ned", **common_args)
    
    df_all = pd.concat([df_simlr, df_ned], ignore_index=True)
    
    summary = df_all.groupby("model")[[
        "latent_recovery_corr", "heldout_outcome_r2", "reconstruction_mse"
    ]].agg(["median", "std"])
    
    print("\nSummary Results:")
    print(summary)
    
    return df_all

if __name__ == "__main__":
    run_ned_vs_simlr_benchmark()
