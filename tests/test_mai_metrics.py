import torch
import numpy as np
import pytest
from pysimlr.deep import LENDSiMRModel, lend_simr, ned_simr

def generate_controlled_latents(n=500, k=4):
    """
    Generate latents with a structured (low-rank) signal.
    """
    torch.manual_seed(42)
    # Signal is primarily in the first 2 dimensions
    u_base = torch.randn(n, 2)
    U_true = torch.zeros(n, k)
    U_true[:, :2] = u_base
    U_true = U_true / U_true.norm()
    
    def add_noise(signal, level):
        noise = torch.randn(n, k)
        noise = noise / noise.norm()
        return (1 - level) * signal + level * noise

    L1 = add_noise(U_true, 0.05)
    L2 = add_noise(U_true, 0.05)
    L3 = add_noise(U_true, 0.20)
    # Rogue is isotropic noise
    L4 = torch.randn(n, k)
    L4 = L4 / L4.norm()
    
    return [L1, L2, L3, L4]

def test_mai_metric_ranking():
    """
    Verify that procrustes_r2_sharp correctly ranks modalities.
    """
    n, k = 500, 4
    latents = generate_controlled_latents(n, k)
    input_dims = [k] * 4
    
    metrics = ['procrustes_r2', 'procrustes_r2_sharp', 'cca', 'rvcoef']
    results = {}
    
    for metric in metrics:
        model = LENDSiMRModel(input_dims, k, dynamic_weights=True, mai_metric=metric)
        model.mai.fill_(0.25)
        for _ in range(50):
            model.update_mai(latents, epoch=40, total_epochs=100)
        results[metric] = model.mai.clone().numpy()
        
    print("\nMAI Ranking Results (after 30 steps):")
    for m, vals in results.items():
        print(f"{m}: {vals}")
        
    r2_sharp = results['procrustes_r2_sharp']
    r2_base = results['procrustes_r2']
    
    # Golden > Partial > Rogue
    assert r2_sharp[0] > r2_sharp[2]
    assert r2_sharp[2] > r2_sharp[3]
    assert np.argmin(r2_sharp) == 3
    
    # Sharpness penalty check
    # Sharpness should penalize Rogue (isotropic) more than Golden (structured)
    # Ratio of Rogue/Golden
    ratio_sharp = r2_sharp[3] / r2_sharp[0]
    ratio_r2 = r2_base[3] / r2_base[0]
    # In the structured case, sharpness of Golden should be ~0.5 (2/4), 
    # sharpness of Rogue should be ~0.25 (1/4).
    # So ratio_sharp should be roughly 0.5 * ratio_r2.
    # If R2 is already near zero (perfect rejection), we accept it.
    if r2_base[3] > 0.02:
        assert ratio_sharp < ratio_r2, f"Sharpness should penalize Rogue more. Sharp ratio: {ratio_sharp:.4f}, R2 ratio: {ratio_r2:.4f}"

def test_spectral_sharpness_isotropic_noise():
    """
    Directly verify sharpness penalty for isotropic noise.
    """
    n, k = 500, 10
    torch.manual_seed(42)
    
    # Target U is rank 1
    u = torch.randn(n, 1)
    U = torch.zeros(n, k)
    U[:, 0] = u[:, 0]
    U = U / U.norm()
    
    # Z_struct: also rank 1, aligns with U
    Z_struct = U.clone()
    
    # Z_iso: rank K, but still contains the signal
    Z_iso = U + torch.randn(n, k) * 0.1
    Z_iso = Z_iso / Z_iso.norm()
    
    model = LENDSiMRModel([k, k], k, dynamic_weights=True, mai_metric='procrustes_r2_sharp')
    
    model.mai.fill_(0.5)
    model.update_mai([Z_struct, U], epoch=40, total_epochs=100)
    mai_struct = model.mai[0].item()
    
    model.mai.fill_(0.5)
    model.update_mai([Z_iso, U], epoch=40, total_epochs=100)
    mai_iso = model.mai[0].item()
    
    print(f"Sharpness check - Struct: {mai_struct:.4f}, Iso: {mai_iso:.4f}")
    assert mai_struct > mai_iso, "Structured alignment must be preferred over isotropic alignment"

def test_mai_metric_propagation():
    n, d, k = 50, 10, 2
    xs = [torch.randn(n, d) for _ in range(3)]
    res_lend = lend_simr(xs, k=k, epochs=1, dynamic_weights=True, mai_metric='cca', warmup_epochs=0)
    assert res_lend['model'].mai_metric == 'cca'
    res_ned = ned_simr(xs, k=k, epochs=1, dynamic_weights=True, mai_metric='rvcoef', warmup_epochs=0)
    assert res_ned['model'].mai_metric == 'rvcoef'

def test_update_mai_edge_cases():
    n, k = 50, 5
    model = LENDSiMRModel([k, k], k, dynamic_weights=True)
    latents_zero = [torch.zeros(n, k), torch.randn(n, k)]
    model.update_mai(latents_zero, 40, 100)
    assert model.mai[0] < 0.5
    latents_nan = [torch.full((n, k), float('nan')), torch.randn(n, k)]
    model.update_mai(latents_nan, 40, 100)
    assert not torch.isnan(model.mai).any()
