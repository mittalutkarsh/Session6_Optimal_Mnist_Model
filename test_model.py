import pytest
import torch
import torch.nn as nn
from Model import Net

def count_parameters(model):
    """Count the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_parameter_count():
    """Test if model has less than 20k parameters."""
    model = Net()
    param_count = count_parameters(model)
    print(f"Total parameters: {param_count}")
    assert param_count < 20000, (
        f"Model has {param_count} parameters, which exceeds the limit of 20,000"
    )

def test_batch_normalization():
    """Test if model uses batch normalization."""
    model = Net()
    has_batchnorm = any(isinstance(m, nn.BatchNorm2d) for m in model.modules())
    assert has_batchnorm, "Model does not use Batch Normalization"
    
    # Count number of BatchNorm layers
    batchnorm_count = sum(
        1 for m in model.modules() if isinstance(m, nn.BatchNorm2d)
    )
    print(f"Number of BatchNorm layers: {batchnorm_count}")

def test_dropout():
    """Test if model uses dropout."""
    model = Net()
    has_dropout = any(isinstance(m, nn.Dropout) for m in model.modules())
    assert has_dropout, "Model does not use Dropout"
    
    # Check dropout value
    dropout_layers = [m for m in model.modules() if isinstance(m, nn.Dropout)]
    for layer in dropout_layers:
        assert 0 < layer.p <= 0.5, (
            f"Dropout value {layer.p} is not in reasonable range (0-0.5)"
        )
    print(f"Number of Dropout layers: {len(dropout_layers)}")

def test_gap_no_fc():
    """Test if model uses Global Average Pooling instead of Fully Connected layers."""
    model = Net()
    
    # Check for absence of Linear layers
    has_fc = any(isinstance(m, nn.Linear) for m in model.modules())
    assert not has_fc, "Model should not use Fully Connected (Linear) layers"
    
    # Check for presence of Global Average Pooling
    has_gap = any(
        isinstance(m, (nn.AdaptiveAvgPool2d, nn.AvgPool2d)) 
        for m in model.modules()
    )
    assert has_gap, "Model should use Global Average Pooling"

def test_model_output_shape():
    """Test if model produces correct output shape for MNIST."""
    model = Net()
    batch_size = 4
    input_tensor = torch.randn(batch_size, 1, 28, 28)  # MNIST image shape
    output = model(input_tensor)
    
    expected_shape = (batch_size, 10)  # 10 classes for MNIST
    assert output.shape == expected_shape, (
        f"Expected output shape {expected_shape}, but got {output.shape}"
    )

def test_model_forward_pass():
    """Test if model can perform a forward pass without errors."""
    model = Net()
    batch_size = 4
    input_tensor = torch.randn(batch_size, 1, 28, 28)
    
    try:
        output = model(input_tensor)
        loss = torch.nn.functional.nll_loss(
            output, torch.randint(0, 10, (batch_size,))
        )
        loss.backward()
    except Exception as e:
        pytest.fail(f"Forward/backward pass failed with error: {str(e)}")

if __name__ == "__main__":
    pytest.main([__file__]) 