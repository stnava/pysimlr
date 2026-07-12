import os
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

# Ensure src is in the import path
import sys
sys.path.append(os.path.join(os.getcwd(), "src"))

from pysimlr import simlr, predict_simlr
from pysimlr.deep import ned_simr, ned_simr_shared_private, predict_deep
from pysimlr.flows import flow_simr, FlowConditionalInference

# ==========================================
# EXPERIMENT 1: Strongly Nonlinear Transformations
# ==========================================
def generate_nonlinear_data(n_samples=1000, noise=0.1, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    k = 3
    d1, d2 = 30, 20
    u = torch.randn(n_samples, k)
    
    def nonlin_map(z, out_dim):
        w = torch.randn(z.shape[1], out_dim)
        return torch.sin(2.0 * z @ w)

    x1 = nonlin_map(u, d1) + noise * torch.randn(n_samples, d1)
    x2 = nonlin_map(u, d2) + noise * torch.randn(n_samples, d2)
    y = u[:, 0:1] * 2.0 - u[:, 1:2] * 1.5 + 0.05 * torch.randn(n_samples)
    
    def norm(x): return (x - x.mean(0)) / (x.std(0) + 1e-6)
    return norm(x1), norm(x2), u, y.numpy()

def run_experiment_1(n_samples=1000, n_seeds=3):
    print("\n==================================================")
    print("EXPERIMENT 1: STRONGLY NONLINEAR TRANSFORMATIONS (SINUSOIDAL)")
    print("==================================================")
    
    k = 3
    results = []
    
    for i in range(n_seeds):
        seed = 42 + i
        x1, x2, u, y = generate_nonlinear_data(n_samples=n_samples, noise=0.1, seed=seed)
        
        # Split data 60% train, 40% test
        train_n = int(n_samples * 0.6)
        train_x = [x1[:train_n], x2[:train_n]]
        test_x = [x1[train_n:], x2[train_n:]]
        train_u, test_u = u[:train_n], u[train_n:]
        train_y, test_y = y[:train_n], y[train_n:]
        
        # 1. Linear SIMLR
        t0 = time.time()
        res_linear = simlr(train_x, k=k, iterations=100, energy_type="acc", verbose=False)
        t_linear = time.time() - t0
        pred_linear = predict_simlr(test_x, res_linear)
        u_linear = pred_linear["u"]
        reg_linear = Ridge().fit(res_linear["u"].numpy(), train_y)
        r2_linear = r2_score(test_y, reg_linear.predict(u_linear.numpy()))
        recon_linear = np.mean([
            torch.norm(x.cpu() - r.cpu(), p="fro").item() / (torch.norm(x.cpu(), p="fro").item() + 1e-10) 
            for x, r in zip(test_x, pred_linear["reconstructions"])
        ])
        
        # 2. Deep NED
        t0 = time.time()
        res_ned = ned_simr(train_x, k=k, epochs=100, warmup_epochs=15, device="cpu", verbose=False)
        t_ned = time.time() - t0
        pred_ned = predict_deep(test_x, res_ned, device="cpu")
        u_ned = pred_ned["u"]
        reg_ned = Ridge().fit(res_ned["u"].numpy(), train_y)
        r2_ned = r2_score(test_y, reg_ned.predict(u_ned.numpy()))
        recon_ned = np.mean([
            torch.norm(x.cpu() - r.cpu(), p="fro").item() / (torch.norm(x.cpu(), p="fro").item() + 1e-10) 
            for x, r in zip(test_x, pred_ned["reconstructions"])
        ])
        
        # 3. Flow SiMLR
        t0 = time.time()
        res_flow = flow_simr(train_x, k=k, epochs=100, warmup_epochs=15, device="cpu", verbose=False)
        t_flow = time.time() - t0
        pred_flow = predict_deep(test_x, res_flow, device="cpu")
        u_flow = pred_flow["u"]
        reg_flow = Ridge().fit(res_flow["u"].numpy(), train_y)
        r2_flow = r2_score(test_y, reg_flow.predict(u_flow.numpy()))
        recon_flow = np.mean([
            torch.norm(x.cpu() - r.cpu(), p="fro").item() / (torch.norm(x.cpu(), p="fro").item() + 1e-10) 
            for x, r in zip(test_x, pred_flow["reconstructions"])
        ])
        
        results.append({"Seed": seed, "Model": "Linear SIMLR", "R2": r2_linear, "Recon": recon_linear, "Time": t_linear})
        results.append({"Seed": seed, "Model": "Deep NED", "R2": r2_ned, "Recon": recon_ned, "Time": t_ned})
        results.append({"Seed": seed, "Model": "Flow-SiMLR", "R2": r2_flow, "Recon": recon_flow, "Time": t_flow})
        
    df = pd.DataFrame(results)
    summary = df.groupby("Model")[["R2", "Recon", "Time"]].agg(["mean", "std"])
    print(summary.to_string())
    return df

# ==========================================
# EXPERIMENT 2: Structured Modality-Specific Noise
# ==========================================
def generate_private_noise_data(n_samples=1000, noise=0.1, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    k_shared = 3
    k_private = 3
    d1, d2 = 30, 20
    u_shared = torch.randn(n_samples, k_shared)
    p1 = torch.randn(n_samples, k_private)
    p2 = torch.randn(n_samples, k_private)
    
    def mix_map(u, p, out_dim):
        z = torch.cat([u, p], dim=1)
        w = torch.randn(z.shape[1], out_dim)
        return (z @ w) + torch.sin(z @ w)

    x1 = mix_map(u_shared, p1, d1) + noise * torch.randn(n_samples, d1)
    x2 = mix_map(u_shared, p2, d2) + noise * torch.randn(n_samples, d2)
    y = u_shared[:, 0:1] * 2.0 - u_shared[:, 1:2] * 1.5 + 0.05 * torch.randn(n_samples)
    
    def norm(x): return (x - x.mean(0)) / (x.std(0) + 1e-6)
    return norm(x1), norm(x2), u_shared, y.numpy()

def run_experiment_2(n_samples=1000, n_seeds=3):
    print("\n==================================================")
    print("EXPERIMENT 2: STRUCTURED PRIVATE NOISE (SHARED/PRIVATE)")
    print("==================================================")
    
    k = 3
    results = []
    
    for i in range(n_seeds):
        seed = 42 + i
        x1, x2, u, y = generate_private_noise_data(n_samples=n_samples, noise=0.1, seed=seed)
        
        train_n = int(n_samples * 0.6)
        train_x = [x1[:train_n], x2[:train_n]]
        test_x = [x1[train_n:], x2[train_n:]]
        train_y, test_y = y[:train_n], y[train_n:]
        
        # 1. Standard Deep NED
        t0 = time.time()
        res_ned = ned_simr(train_x, k=k, epochs=100, warmup_epochs=15, device="cpu", verbose=False)
        t_ned = time.time() - t0
        pred_ned = predict_deep(test_x, res_ned, device="cpu")
        reg_ned = Ridge().fit(res_ned["u"].numpy(), train_y)
        r2_ned = r2_score(test_y, reg_ned.predict(pred_ned["u"].numpy()))
        recon_ned = np.mean([
            torch.norm(x.cpu() - r.cpu(), p="fro").item() / (torch.norm(x.cpu(), p="fro").item() + 1e-10) 
            for x, r in zip(test_x, pred_ned["reconstructions"])
        ])
        
        # 2. Deep NEDPP (Shared-Private)
        t0 = time.time()
        res_nedpp = ned_simr_shared_private(train_x, k=k, private_k=3, epochs=100, warmup_epochs=15, device="cpu", verbose=False)
        t_nedpp = time.time() - t0
        pred_nedpp = predict_deep(test_x, res_nedpp, device="cpu")
        reg_nedpp = Ridge().fit(res_nedpp["u"].numpy(), train_y)
        r2_nedpp = r2_score(test_y, reg_nedpp.predict(pred_nedpp["u"].numpy()))
        recon_nedpp = np.mean([
            torch.norm(x.cpu() - r.cpu(), p="fro").item() / (torch.norm(x.cpu(), p="fro").item() + 1e-10) 
            for x, r in zip(test_x, pred_nedpp["reconstructions"])
        ])
        
        # 3. Flow SiMLR (Shared-Private by default due to bijective mapping)
        t0 = time.time()
        res_flow = flow_simr(train_x, k=k, epochs=100, warmup_epochs=15, device="cpu", verbose=False)
        t_flow = time.time() - t0
        pred_flow = predict_deep(test_x, res_flow, device="cpu")
        reg_flow = Ridge().fit(res_flow["u"].numpy(), train_y)
        r2_flow = r2_score(test_y, reg_flow.predict(pred_flow["u"].numpy()))
        recon_flow = np.mean([
            torch.norm(x.cpu() - r.cpu(), p="fro").item() / (torch.norm(x.cpu(), p="fro").item() + 1e-10) 
            for x, r in zip(test_x, pred_flow["reconstructions"])
        ])
        
        results.append({"Seed": seed, "Model": "Standard NED", "R2": r2_ned, "Recon": recon_ned, "Time": t_ned})
        results.append({"Seed": seed, "Model": "NEDPP (Shared/Private)", "R2": r2_nedpp, "Recon": recon_nedpp, "Time": t_nedpp})
        results.append({"Seed": seed, "Model": "Flow-SiMLR", "R2": r2_flow, "Recon": recon_flow, "Time": t_flow})
        
    df = pd.DataFrame(results)
    summary = df.groupby("Model")[["R2", "Recon", "Time"]].agg(["mean", "std"])
    print(summary.to_string())
    return df

# ==========================================
# EXPERIMENT 3: Diabetes Progression Clinical Regression
# ==========================================
def run_experiment_3(n_seeds=3):
    print("\n==================================================")
    print("EXPERIMENT 3: DIABETES CLINICAL PROGRESSION & IMPUTATION")
    print("==================================================")
    
    # Load and scale diabetes dataset
    data = load_diabetes()
    X = StandardScaler().fit_transform(data.data)
    y = data.target
    k = 2
    
    # Partition into vitals (5) and blood serums (5)
    mats = [X[:, :5], X[:, 5:]]
    
    results = []
    
    for i in range(n_seeds):
        seed = 42 + i
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Train-Test Split 60/40
        n_samples = X.shape[0]
        perm = np.random.permutation(n_samples)
        train_idx = perm[:int(n_samples * 0.6)]
        test_idx = perm[int(n_samples * 0.6):]
        
        train_x = [torch.tensor(m[train_idx]).float() for m in mats]
        test_x = [torch.tensor(m[test_idx]).float() for m in mats]
        
        train_y = y[train_idx]
        test_y = y[test_idx]
        
        # 1. Linear SIMLR
        t0 = time.time()
        res_linear = simlr(train_x, k=k, iterations=80, energy_type="acc", verbose=False)
        t_linear = time.time() - t0
        pred_linear = predict_simlr(test_x, res_linear)
        reg_linear = Ridge().fit(res_linear["u"].numpy(), train_y)
        r2_linear = r2_score(test_y, reg_linear.predict(pred_linear["u"].numpy()))
        recon_linear = np.mean([
            torch.norm(x.cpu() - r.cpu(), p="fro").item() / (torch.norm(x.cpu(), p="fro").item() + 1e-10) 
            for x, r in zip(test_x, pred_linear["reconstructions"])
        ])
        
        # 2. Deep NED
        t0 = time.time()
        res_ned = ned_simr(train_x, k=k, epochs=100, warmup_epochs=15, device="cpu", verbose=False)
        t_ned = time.time() - t0
        pred_ned = predict_deep(test_x, res_ned, device="cpu")
        reg_ned = Ridge().fit(res_ned["u"].numpy(), train_y)
        r2_ned = r2_score(test_y, reg_ned.predict(pred_ned["u"].numpy()))
        recon_ned = np.mean([
            torch.norm(x.cpu() - r.cpu(), p="fro").item() / (torch.norm(x.cpu(), p="fro").item() + 1e-10) 
            for x, r in zip(test_x, pred_ned["reconstructions"])
        ])
        
        # 3. Flow SiMLR
        t0 = time.time()
        res_flow = flow_simr(train_x, k=k, epochs=100, warmup_epochs=15, device="cpu", verbose=False)
        t_flow = time.time() - t0
        pred_flow = predict_deep(test_x, res_flow, device="cpu")
        reg_flow = Ridge().fit(res_flow["u"].numpy(), train_y)
        r2_flow = r2_score(test_y, reg_flow.predict(pred_flow["u"].numpy()))
        recon_flow = np.mean([
            torch.norm(x.cpu() - r.cpu(), p="fro").item() / (torch.norm(x.cpu(), p="fro").item() + 1e-10) 
            for x, r in zip(test_x, pred_flow["reconstructions"])
        ])
        
        # 4. Cross-View Imputation (Predict view 2 from view 1 using Flow conditional inference)
        cond = res_flow["cond_inference"]
        with torch.no_grad():
            # Get observed shared latent representation of view 1 on test set
            obs_z1 = res_flow["model"].encoders[0](test_x[0])
            # Conditionally predict target shared latent of view 2
            pred_z2 = cond.predict_conditional(observed_idx=0, target_idx=1, observed_z=obs_z1)
            # Decode to construct synthesized view 2 features
            imputed_x2 = res_flow["model"].decoders[1](pred_z2)
            
            # Imputation error (relative Frobenius norm)
            imputation_err = (torch.norm(test_x[1] - imputed_x2, p="fro").item() / 
                              (torch.norm(test_x[1], p="fro").item() + 1e-10))
        
        results.append({
            "Seed": seed, 
            "Model": "Linear SIMLR", 
            "R2": r2_linear, 
            "Recon": recon_linear, 
            "ImputeErr": np.nan, 
            "Time": t_linear
        })
        results.append({
            "Seed": seed, 
            "Model": "Deep NED", 
            "R2": r2_ned, 
            "Recon": recon_ned, 
            "ImputeErr": np.nan, 
            "Time": t_ned
        })
        results.append({
            "Seed": seed, 
            "Model": "Flow-SiMLR", 
            "R2": r2_flow, 
            "Recon": recon_flow, 
            "ImputeErr": imputation_err, 
            "Time": t_flow
        })
        
    df = pd.DataFrame(results)
    summary = df.groupby("Model")[["R2", "Recon", "ImputeErr", "Time"]].agg(["mean", "std"])
    print(summary.to_string())
    return df

def main():
    print("==================================================")
    print("STARTING PAPER COMPARATIVE BENCHMARK RUNNER")
    print("==================================================")
    
    run_experiment_1(n_samples=800, n_seeds=3)
    run_experiment_2(n_samples=800, n_seeds=3)
    run_experiment_3(n_seeds=3)

if __name__ == "__main__":
    main()
