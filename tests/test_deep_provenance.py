import torch
import numpy as np
from pysimlr import deep_simr, predict_simlr

def test_deep_preprocessing_provenance():
    # 1. Create train data with a specific offset
    train_x1 = torch.randn(100, 10) + 5.0
    train_x2 = torch.randn(100, 8) - 5.0
    
    # 2. Run deep_simr - it now captures provenance for ["centerAndScale"]
    # Reducing epochs for speed
    res = deep_simr([train_x1, train_x2], k=2, epochs=5, warmup_epochs=0)
    
    assert "provenance_list" in res
    assert "cas_mean" in res["provenance_list"][0]
    
    # 3. Create test data with all zeros
    test_x1 = torch.zeros(10, 10)
    test_x2 = torch.zeros(10, 8)
    
    # 4. Predict - should apply train statistics to test data
    # If it works, it shouldn't crash and error should be reasonable
    pred = predict_simlr([test_x1, test_x2], res)
    
    assert "errors" in pred
    assert len(pred["errors"]) == 2

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
