import torch
import numpy as np
import pandas as pd
import pytest
from pysimlr.benchmarks.shared_private_sweep import tune_shared_private
from unittest.mock import patch

def test_tune_shared_private():
    # Patch ortho_weights and var_weights to make test faster
    with patch('pysimlr.benchmarks.shared_private_sweep.build_case') as mock_build_case:
        # Mock case
        mock_case = {
            "data": [torch.randn(100, 10), torch.randn(100, 10)],
            "true_u": torch.randn(100, 2),
            "outcome": np.random.randn(100),
            "shared_k": 2
        }
        mock_build_case.return_value = mock_case
        
        # We need to patch the weights inside the function or just accept it will run 16 times
        # Let's just run it with very small epochs
        with patch('pysimlr.benchmarks.shared_private_sweep.run_single_experiment') as mock_run:
            mock_run.return_value = {"metrics": {"recovery": 0.8}, "result": {}}
            
            df = tune_shared_private(n_samples=50, n_seeds=1)
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 16 # 4 * 4
            assert "ortho_weight" in df.columns
            assert "var_weight" in df.columns
