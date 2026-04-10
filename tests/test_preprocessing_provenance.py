import torch
import numpy as np
from pysimlr import simlr, predict_simlr

def test_preprocessing_provenance():
    # 1. Create train data with a specific mean
    train_x = torch.randn(100, 10) + 10.0
    
    # 2. Run simlr - it will capture the mean=10.0 in provenance
    res = simlr([train_x], k=2, iterations=2, scale_list=["center"])
    
    # 3. Create test data with all zeros
    test_x = torch.zeros(10, 10)
    
    # 4. Predict - should apply train mean (10.0) to test data
    # If it works, the internally preprocessed test data will be (0.0 - 10.0) = -10.0
    pred = predict_simlr([test_x], res)
    
    # Let's verify the provenance was captured
    assert "provenance_list" in res
    assert "center_mean" in res["provenance_list"][0]
    
    train_mean = res["provenance_list"][0]["center_mean"]
    assert torch.allclose(train_mean.mean(), torch.tensor(10.0), atol=0.5)

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
