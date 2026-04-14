import torch
import numpy as np
import pandas as pd
import time
from typing import List, Dict, Any, Optional, Union, Tuple
from .metrics import calculate_all_metrics
from ..simlr import predict_simlr
from ..deep import predict_deep

class BenchmarkProtocol:
    """
    Standardized protocol for SiMLR benchmarking.

    Handles data splitting (train/val/test) and systematic evaluation of 
    models across multiple metrics.

    Parameters
    ----------
    n_samples : int
        Total number of samples in the dataset.
    train_prop : float, default=0.6
        Proportion of data to use for training.
    val_prop : float, default=0.2
        Proportion of data to use for validation.
    """
    def __init__(self, n_samples: int, train_prop: float = 0.6, val_prop: float = 0.2):
        self.n_samples = n_samples
        self.train_prop = train_prop
        self.val_prop = val_prop
        
        self.train_n = int(n_samples * train_prop)
        self.val_n = int(n_samples * val_prop)
        self.test_n = n_samples - self.train_n - self.val_n

    def split_data(self, data: List[torch.Tensor], u: torch.Tensor, y: np.ndarray) -> Dict[str, Any]:
        """
        Split dataset into training, validation, and testing sets.

        Parameters
        ----------
        data : List[torch.Tensor]
            Input data modalities.
        u : torch.Tensor
            Ground truth shared latent space (if available).
        y : np.ndarray
            Ground truth outcomes (if available).

        Returns
        -------
        Dict[str, Any]
            A nested dictionary containing `train`, `val`, and `test` splits, 
            each with its own `data`, `u`, and `y` subsets.
        """
        res = {
            "train": {"data": [m[:self.train_n] for m in data], "u": u[:self.train_n], "y": y[:self.train_n]},
            "val": {"data": [m[self.train_n:self.train_n+self.val_n] for m in data], "u": u[self.train_n:self.train_n+self.val_n], "y": y[self.train_n:self.train_n+self.val_n]},
            "test": {"data": [m[self.train_n+self.val_n:] for m in data], "u": u[self.train_n+self.val_n:], "y": y[self.train_n+self.val_n:]}
        }
        return res

    def evaluate_model(self, model_res: Dict[str, Any], split_data: Dict[str, Any], model_type: str, seed: int, generator_name: str, noise_level: float) -> Dict[str, Any]:
        """
        Evaluate a trained model on the test split and return a standardized result.

        Computes performance metrics (recovery, R2, MSE) and collects model 
        diagnostics into a single result dictionary.

        Parameters
        ----------
        model_res : Dict[str, Any]
            Results from a model training run.
        split_data : Dict[str, Any]
            The split data dictionary from `split_data`.
        model_type : str
            Name/type of the model (e.g., `LEND`, `NED`).
        seed : int
            Random seed used for the experiment.
        generator_name : str
            Name of the data generator used.
        noise_level : float
            Noise level applied to the data.

        Returns
        -------
        Dict[str, Any]
            A comprehensive results dictionary for one experimental run.
        """
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
        
        # Standardized schema - Includes all diagnostic fields from metrics
        result = {
            "seed": seed,
            "model": model_type,
            "generator_name": generator_name,
            "noise_level": noise_level,
            "train_n": self.train_n,
            "val_n": self.val_n,
            "test_n": self.test_n,
            "latent_recovery_corr": metrics['recovery'],
            "heldout_outcome_r2": metrics['test_r2'],
            "heldout_outcome_mse": metrics.get('test_mse', 0.0),
            "reconstruction_mse": metrics['recon_error'],
            "reconstruction_mse_std": metrics.get('recon_error_std', 0.0)
        }
        
        # Capture all other diagnostic keys (variance, orthogonality, shared/private separation)
        diagnostic_prefixes = ["u_", "collapsed_", "mod", "orthogonality_"]
        for key, value in metrics.items():
            if any(key.startswith(p) for p in diagnostic_prefixes):
                result[key] = value
                
        return result

def run_repeated_benchmark(protocol: BenchmarkProtocol, 
                           data_generator: Any, 
                           model_fitter: Any, 
                           model_type: str,
                           n_seeds: int = 3,
                           generator_name: str = "unknown",
                           noise_level: float = 0.1,
                           **hparams) -> pd.DataFrame:
    """
    Run a benchmark across multiple random seeds using a standardized protocol.

    Automates data generation, splitting, model fitting, and evaluation over 
    multiple iterations to ensure stable performance estimates.

    Parameters
    ----------
    protocol : BenchmarkProtocol
        An instance of `BenchmarkProtocol` defining the split strategy.
    data_generator : Callable
        A function that generates synthetic data. Expected signature:
        `data_generator(n_samples, noise) -> x1, x2, u, y`.
    model_fitter : Callable
        A function that fits a SiMLR or deep model. Expected signature:
        `model_fitter(train_data, **hparams) -> model_res`.
    model_type : str
        The name of the model being tested.
    n_seeds : int, default=3
        Number of independent runs with different seeds.
    generator_name : str, default="unknown"
        Label for the data generator (for reporting).
    noise_level : float, default=0.1
        The amount of noise to apply during data generation.
    **hparams : Dict[str, Any]
        Hyperparameters passed to the `model_fitter`.

    Returns
    -------
    pd.DataFrame
        A DataFrame where each row corresponds to one experimental run, 
        containing all computed metrics and experimental conditions.
    """
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
