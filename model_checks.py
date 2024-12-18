import torch
import torch.nn as nn
import sys
import os

# Add the repository root to Python path
sys.path.append(os.getcwd())
from Model import Net

def check_model():
    model = Net()
    
    # 1. Check Total Parameter Count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    if total_params > 20000:
        print("❌ Model has more than 20,000 parameters")
        sys.exit(1)
    else:
        print("✅ Parameter count check passed")

    # 2. Check for Batch Normalization
    has_batchnorm = any(isinstance(m, nn.BatchNorm2d) for m in model.modules())
    if not has_batchnorm:
        print("❌ Model does not use Batch Normalization")
        sys.exit(1)
    print("✅ Batch Normalization check passed")

    # 3. Check for Dropout
    has_dropout = any(isinstance(m, nn.Dropout) for m in model.modules())
    if not has_dropout:
        print("❌ Model does not use Dropout")
        sys.exit(1)
    print("✅ Dropout check passed")

    # 4. Check for GAP (Global Average Pooling) vs FC Layer
    has_gap = any(isinstance(m, (nn.AdaptiveAvgPool2d, nn.AvgPool2d)) 
                    for m in model.modules())
    has_fc = any(isinstance(m, nn.Linear) for m in model.modules())
    
    if has_fc:
        print("❌ Model uses Fully Connected layer instead of GAP")
        sys.exit(1)
    if not has_gap:
        print("❌ Model does not use Global Average Pooling")
        sys.exit(1)
    print("✅ GAP check passed")

if __name__ == "__main__":
    check_model() 