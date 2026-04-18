import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
import numpy as np
from .deep import (LENDNSAEncoder, ModalityDecoder, _train_loop, 
                   _resolve_stabilization_schedule, 
                   calculate_sim_loss, _update_first_layer_schedule)
from .consensus import compute_shared_consensus

class OptimizedModalityDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[128, 64], dropout=0.1):
        super().__init__()
        layers = []
        curr_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(nn.BatchNorm1d(h)) 
            layers.append(nn.LeakyReLU(0.01))
            layers.append(nn.Dropout(dropout))
            curr_dim = h
        layers.append(nn.Linear(curr_dim, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class LENDOptimizedSiMRModel(nn.Module):
    def __init__(self, input_dims, latent_dim, hidden_dims=[128, 64], dropout=0.1, nsa_w=0.1, positivity="positive", sparseness_quantile=0.0, mixing_algorithm="newton", use_nsa=True, first_layer_mode="scheduled", nsa_iterations=3):
        super().__init__()
        if isinstance(sparseness_quantile, (float, int)):
            sparseness_quantile = [float(sparseness_quantile)] * len(input_dims)
        self.encoders = nn.ModuleList([LENDNSAEncoder(dim, latent_dim, nsa_w, positivity, sq, use_nsa=use_nsa, first_layer_mode=first_layer_mode, nsa_iterations=nsa_iterations) for dim, sq in zip(input_dims, sparseness_quantile)])
        self.decoders = nn.ModuleList([OptimizedModalityDecoder(latent_dim, dim, hidden_dims, dropout) for dim in input_dims])
        self.mixing_algorithm, self.latent_dim = mixing_algorithm, latent_dim
        self.register_buffer("consensus_anchor", torch.zeros(len(input_dims) * latent_dim, latent_dim))

    def encode_first_layer(self, x_list, use_projected=None):
        return [enc.encode_first_layer(x, use_projected=use_projected) for enc, x in zip(self.encoders, x_list)]

    def forward(self, x_list: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        latents = self.encode_first_layer(x_list)
        res_u = compute_shared_consensus(latents, mixing_algorithm=self.mixing_algorithm, k=self.latent_dim, training=self.training, anchor=self.consensus_anchor)
        if self.training:
            u_shared, new_anchor = res_u
            self.consensus_anchor.copy_(0.9 * self.consensus_anchor + 0.1 * new_anchor)
        else:
            u_shared = res_u
        return latents, [dec(u_shared) for dec in self.decoders], u_shared
    
    def initialize_v(self, data_matrices, k):
        from .svd import ba_svd
        with torch.no_grad():
            for i, x in enumerate(data_matrices):
                u, s, v = ba_svd(x, nu=0, nv=k)
                if v.shape[1] < k: v = torch.cat([v, torch.randn(v.shape[0], k-v.shape[1], device=v.device)*1e-4], dim=1)
                if self.encoders[i].positivity in {"positive", "hard", "softplus"}:
                    for j in range(v.shape[1]):
                        if v[:, j].sum() < 0: v[:, j] *= -1
                self.encoders[i].v_raw.copy_(v.to(x.dtype))

    def first_layer_diagnostics(self):
        drifts = [float(enc.basis_drift().cpu()) for enc in self.encoders]
        alphas = [float(enc.projection_alpha) for enc in self.encoders]
        return {
            "basis_drift": float(sum(drifts) / max(1, len(drifts))),
            "projection_alpha": float(sum(alphas) / max(1, len(alphas))),
        }

def lend_simr_optimized(data_matrices, k, epochs=150, batch_size=64, learning_rate=5e-4, **kwargs):
    device = "cpu"
    mats = [torch.as_tensor(m).float() for m in data_matrices]
    input_dims = [m.shape[1] for m in mats]
    
    model = LENDOptimizedSiMRModel(input_dims, k, **kwargs).to(device)
    model.initialize_v(mats, k)
    
    dataset = TensorDataset(*mats)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    _train_loop(model, dataloader, optimizer, scheduler, nn.MSELoss(), epochs, 1.0, "regression", 20, False, device)
    
    model.eval()
    with torch.no_grad():
        _, _, u_shared = model(mats)
    return {"u": u_shared, "v": [enc.v for enc in model.encoders], "model": model}
