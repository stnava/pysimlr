import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from typing import List, Optional, Union, Dict, Any, Tuple

class ModalityEncoder(nn.Module):
    """
    A flexible MLP encoder for a single modality.
    """
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: List[int] = [128, 64], dropout: float = 0.1):
        super().__init__()
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, latent_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class ModalityDecoder(nn.Module):
    """
    A flexible MLP decoder for a single modality.
    """
    def __init__(self, latent_dim: int, output_dim: int, hidden_dims: List[int] = [64, 128], dropout: float = 0.1):
        super().__init__()
        layers = []
        curr_dim = latent_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, z):
        return self.network(z)

class DeepSiMLRModel(nn.Module):
    """
    Multi-view Autoencoder for Deep SiMLR.
    """
    def __init__(self, input_dims: List[int], latent_dim: int, hidden_dims: List[int] = [128, 64], dropout: float = 0.1):
        super().__init__()
        self.encoders = nn.ModuleList([
            ModalityEncoder(dim, latent_dim, hidden_dims, dropout) for dim in input_dims
        ])
        self.decoders = nn.ModuleList([
            ModalityDecoder(latent_dim, dim, hidden_dims[::-1], dropout) for dim in input_dims
        ])

    def forward(self, x_list: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        latents = [enc(x) for enc, x in zip(self.encoders, x_list)]
        reconstructions = [dec(z) for dec, z in zip(self.decoders, latents)]
        return latents, reconstructions

def deep_simlr(data_matrices: List[Union[torch.Tensor, np.ndarray]],
                k: int,
                epochs: int = 100,
                batch_size: int = 32,
                learning_rate: float = 1e-3,
                weight_decay: float = 1e-5,
                sim_weight: float = 0.1,
                hidden_dims: List[int] = [128, 64],
                dropout: float = 0.1,
                device: Optional[str] = None,
                verbose: bool = False) -> Dict[str, Any]:
    """
    Deep SiMLR: Non-linear Multi-modal Integration using PyTorch Autoencoders.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(device)
    
    torch_mats = [m if isinstance(m, torch.Tensor) else torch.from_numpy(m).float() for m in data_matrices]
    input_dims = [m.shape[1] for m in torch_mats]
    
    model = DeepSiMLRModel(input_dims, k, hidden_dims, dropout=dropout).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    mse_loss = nn.MSELoss()
    
    dataset = TensorDataset(*torch_mats)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    loss_history = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch in dataloader:
            batch_mats = [b.to(device) for b in batch]
            optimizer.zero_grad()
            
            latents, reconstructions = model(batch_mats)
            
            recon_loss = sum(mse_loss(r, x) for r, x in zip(reconstructions, batch_mats))
            u_shared = torch.mean(torch.stack(latents), dim=0)
            sim_loss = sum(mse_loss(z, u_shared) for z in latents)
            
            total_loss = recon_loss + sim_weight * sim_loss
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            
        epoch_loss /= len(dataloader)
        loss_history.append(epoch_loss)
        
        if verbose and epoch % max(1, epochs // 10) == 0:
            print(f"Epoch {epoch}: Average Loss {epoch_loss:.4f}")
            
    model.eval()
    with torch.no_grad():
        all_torch_mats = [m.to(device) for m in torch_mats]
        final_latents, _ = model(all_torch_mats)
        u_final = torch.mean(torch.stack(final_latents), dim=0)
        
    return {
        "model": model.cpu(),
        "u": u_final.cpu(),
        "latents": [l.cpu() for l in final_latents],
        "loss_history": loss_history
    }
