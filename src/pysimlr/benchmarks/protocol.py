import torch
import numpy as np
import pandas as pd
import time
from typing import List, Dict, Any, Optional, Union, Tuple
from .metrics import calculate_all_metrics
from ..simlr import predict_simlr
from ..deep import predict_deep

class BenchmarkProtocol:
    def __init__(self, n_samples: int, train_prop: float = 0.6, val_prop: float = 0.2):
        self.n_samples = n_samples
        self.train_prop = train_prop
        self.val_prop = val_prop
        
        self.train_n = int(n_samples * train_prop)
        self.val_n = int(n_samples * val_prop)
        self.test_n = n_samples - self.train_n - self.val_n

    def split_data(self, data: List[torch.Tensor], u: torch.Tensor, y: np.ndarray) -> Dict[str, Any]:
        """Split data into train, validation, and test sets."""
        res = {
            "train": {"data": [m[:self.train_n] for m in data], "u": u[:self.train_n], "y": y[:self.train_n]},
            "val": {"data": [m[self.train_n:self.train_n+self.val_n] for m in data], "u": u[self.train_n:self.train_n+self.val_n], "y": y[self.train_n:self.train_n+self.val_n]},
            "test": {"data": [m[self.train_n+self.val_n:] for m in data], "u": u[self.train_n+self.val_n:], "y": y[self.train_n+self.val_n:]}
        }
        return res

    def evaluate_model(self, model_res: Dict[str, Any], split_data: Dict[str, Any], model_type: str, seed: int, generator_name: str, noise_level: float) -> Dict[str, Any]:
        """Evaluate a trained model on the test split and return a standardized result."""
        test_data = split_data["test"]
        train_data = split_data["train"]
        
        # We always use the prediction path for both train and test to ensure 
        # the coordinate systems (sign, rotation) are identical for metric calculation.
        if "model" in model_res: # Deep model
            pred_test = predict_deep(test_data["data"], model_res, device="cpu")
            pred_train = predict_deep(train_data["data"], model_res, device="cpu")
        else: # Linear SIMLR
            pred_test = predict_simlr(test_data["data"], model_res)
            pred_train = predict_simlr(train_data["data"], model_res)

        # Calculate metrics
        shared_l = pred_test.get("latents")
        private_l = pred_test.get("private_latents")
        
        metrics = calculate_all_metrics(
            pred_test['u'], test_data["u"], test_data["y"], 
            test_data["data"], pred_test['reconstructions'],
            shared_latents=shared_l,
            private_latents=private_l,
            v_mats=model_res.get("v"),
            u_train=pred_train['u'],
            y_train=train_data["y"]
        )
        
        # Standardized schema
        result = {
            "seed": seed,
            "model": model_type,
            "train_n": self.train_n,
            "val_n": self.val_n,
            "test_n": self.test_n,
            "latent_recovery_corr": metrics['recovery'],
            "reconstruction_mse": metrics['recon_error'],
            "heldout_outcome_r2": metrics['test_r2'],
            "heldout_outcome_mse": metrics.get('test_mse', 0.0),
            "generator_name": generator_name,
            "noise_level": noise_level,
            # Diagnostics
            "u_std_mean": metrics.get("u_std_mean", 0.0),
            "u_off_diag_cov": metrics.get("u_off_diag_cov", 0.0)
        }
        
        # Add shared/private diagnostics if available
        for key in metrics:
            if "shared_var" in key or "private_var" in key or "cross_cov" in key:
                result[key] = metrics[key]
                
        return result

def run_repeated_benchmark(protocol: BenchmarkProtocol, 
                           data_generator: Any, 
                           model_fitter: Any, 
                           model_type: str,
                           n_seeds: int = 3,
                           generator_name: str = "unknown",
                           noise_level: float = 0.1,
                           **hparams) -> pd.DataFrame:
    """Run a benchmark across multiple seeds using the shared protocol."""
    results = []
    for i in range(n_seeds):
        seed = 42 + i
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Generate and split
        x1, x2, u, y = data_generator(n_samples=protocol.n_samples, noise=noise_level)
        splits = protocol.split_data([x1, x2], u, y)
        
        # Fit (using train and potentially val)
        start_time = time.time()
        model_res = model_fitter(splits["train"]["data"], **hparams)
        fit_time = time.time() - start_time
        
        # Evaluate
        res = protocol.evaluate_model(model_res, splits, model_type, seed, generator_name, noise_level)
        res["fit_time"] = fit_time
        results.append(res)
        
    return pd.DataFrame(results)
