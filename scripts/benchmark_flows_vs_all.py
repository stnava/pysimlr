import torch
import numpy as np
import pandas as pd
import sys
from pysimlr import simlr, ned_simr, flow_simr
from pysimlr.benchmarks.protocol import BenchmarkProtocol, run_repeated_benchmark

def generate_nonlinear_data(n_samples=1000, noise=0.1):
    k = 3
    d1, d2 = 30, 20
    u = torch.randn(n_samples, k)
    
    # Strongly non-linear mappings
    def nonlin_map(z, out_dim):
        w1 = torch.randn(z.shape[1], out_dim)
        w2 = torch.randn(z.shape[1], out_dim)
        res = torch.sin(z @ w1) + 0.5 * (z @ w2)**3
        return res

    x1 = nonlin_map(u, d1) + noise * torch.randn(n_samples, d1)
    x2 = nonlin_map(u, d2) + noise * torch.randn(n_samples, d2)
    
    # Outcome variable dependent on shared latents
    y = u[:, 0:1] * 2.0 - u[:, 1:2] * 1.0 + 0.1 * torch.randn(n_samples, 1)
    
    def norm(x): return (x - x.mean(0)) / (x.std(0) + 1e-6)
    return norm(x1), norm(x2), u, y.numpy()

def fit_simlr(train_data, **kwargs):
    k = kwargs.get("k", 3)
    return simlr(train_data, k=k, iterations=80, energy_type="acc", verbose=False)

def fit_ned(train_data, **kwargs):
    k = kwargs.get("k", 3)
    return ned_simr(train_data, k=k, epochs=100, warmup_epochs=15, device="cpu", verbose=False)

def fit_flow(train_data, **kwargs):
    k = kwargs.get("k", 3)
    return flow_simr(train_data, k=k, epochs=100, warmup_epochs=15, num_layers=3, hidden_dim=32, device="cpu", verbose=False)

def main():
    n_samples = 800
    n_seeds = 3
    noise_level = 0.1
    k = 3
    
    protocol = BenchmarkProtocol(n_samples=n_samples)
    common_args = {"n_seeds": n_seeds, "generator_name": "nonlinear_benchmark", "noise_level": noise_level, "k": k}
    
    print(f"Starting benchmark on {n_samples} samples across {n_seeds} seeds...")
    
    print("\n--- Running Linear SIMLR ---")
    df_simlr = run_repeated_benchmark(protocol, generate_nonlinear_data, fit_simlr, "linear", **common_args)
    
    print("\n--- Running Deep NED ---")
    df_ned = run_repeated_benchmark(protocol, generate_nonlinear_data, fit_ned, "ned_deep", **common_args)
    
    print("\n--- Running Flow-SiMLR ---")
    df_flow = run_repeated_benchmark(protocol, generate_nonlinear_data, fit_flow, "flow_simr", **common_args)
    
    df_all = pd.concat([df_simlr, df_ned, df_flow], ignore_index=True)
    
    summary = df_all.groupby("model")[[
        "latent_recovery_corr", "heldout_outcome_r2", "reconstruction_mse", "fit_time"
    ]].agg(["median", "std"])
    
    print("\n=== BENCHMARK COMPARISON SUMMARY ===")
    print(summary.to_string())
    
    # Save results to CSV
    df_all.to_csv("benchmark_flows_vs_all_results.csv", index=False)
    print("\nSaved detailed benchmark results to benchmark_flows_vs_all_results.csv")

if __name__ == "__main__":
    main()
