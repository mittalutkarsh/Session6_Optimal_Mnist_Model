# MNIST Classification Model

## Target
- Reduce the number of parameters while maintaining the model performance
- Use a smaller number of channels in the initial convolutional layers
- Implement architectural best practices (BatchNorm, GAP, Dropout)

## Results
- Parameters: 13,808
- Best Train Accuracy: 99.21%
- Best Test Accuracy: 99.40%
- No of Epochs: 20

## Model Architecture
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 26, 26]             144
              ReLU-2           [-1, 16, 26, 26]               0
       BatchNorm2d-3           [-1, 16, 26, 26]              32
           Dropout-4           [-1, 16, 26, 26]               0
            Conv2d-5           [-1, 16, 24, 24]           2,304
              ReLU-6           [-1, 16, 24, 24]               0
       BatchNorm2d-7           [-1, 16, 24, 24]              32
           Dropout-8           [-1, 16, 24, 24]               0
            Conv2d-9           [-1, 10, 24, 24]             160
        MaxPool2d-10           [-1, 10, 12, 12]               0
           Conv2d-11           [-1, 16, 10, 10]           1,440
             ReLU-12           [-1, 16, 10, 10]               0
      BatchNorm2d-13           [-1, 16, 10, 10]              32
          Dropout-14           [-1, 16, 10, 10]               0
           Conv2d-15             [-1, 16, 8, 8]           2,304
             ReLU-16             [-1, 16, 8, 8]               0
      BatchNorm2d-17             [-1, 16, 8, 8]              32
          Dropout-18             [-1, 16, 8, 8]               0
           Conv2d-19             [-1, 16, 6, 6]           2,304
             ReLU-20             [-1, 16, 6, 6]               0
      BatchNorm2d-21             [-1, 16, 6, 6]              32
          Dropout-22             [-1, 16, 6, 6]               0
           Conv2d-23             [-1, 16, 6, 6]           2,304
             ReLU-24             [-1, 16, 6, 6]               0
      BatchNorm2d-25             [-1, 16, 6, 6]              32
          Dropout-26             [-1, 16, 6, 6]               0
        AvgPool2d-27             [-1, 16, 1, 1]               0
           Conv2d-28             [-1, 10, 1, 1]             160
================================================================
Total params: 13,808
Trainable params: 13,808
Non-trainable params: 0
----------------------------------------------------------------
```

## Receptive Field Calculation
| Layer | Operation | Kernel Size | Stride | Padding | Input Size | Output Size | RF In | RF Out | Jump In | Jump Out |
|-------|-----------|-------------|---------|----------|------------|-------------|--------|---------|----------|-----------|
| 1 | Conv2d | 3x3 | 1 | 0 | 28x28 | 26x26 | 1 | 3 | 1 | 1 |
| 2 | Conv2d | 3x3 | 1 | 0 | 26x26 | 24x24 | 3 | 5 | 1 | 1 |
| 3 | Conv2d | 1x1 | 1 | 0 | 24x24 | 24x24 | 5 | 5 | 1 | 1 |
| 4 | MaxPool | 2x2 | 2 | 0 | 24x24 | 12x12 | 5 | 6 | 1 | 2 |
| 5 | Conv2d | 3x3 | 1 | 0 | 12x12 | 10x10 | 6 | 10 | 2 | 2 |
| 6 | Conv2d | 3x3 | 1 | 0 | 10x10 | 8x8 | 10 | 14 | 2 | 2 |
| 7 | Conv2d | 3x3 | 1 | 0 | 8x8 | 6x6 | 14 | 18 | 2 | 2 |
| 8 | Conv2d | 3x3 | 1 | 1 | 6x6 | 6x6 | 18 | 20 | 2 | 2 |
| 9 | AvgPool | 6x6 | 1 | 0 | 6x6 | 1x1 | 20 | 32 | 2 | 2 |
| 10 | Conv2d | 1x1 | 1 | 0 | 1x1 | 1x1 | 32 | 32 | 2 | 2 |

## Model Features
1. **Architecture Highlights**:
   - Uses Batch Normalization after every convolution layer
   - Implements Dropout (5%) for regularization
   - Uses Global Average Pooling instead of Fully Connected layers
   - Optimized channel sizes for parameter efficiency

2. **Training Details**:
   - Optimizer: SGD with momentum (0.9)
   - Learning Rate: 0.01
   - Batch Size: 128
   - Epochs: 20

3. **Data Augmentation**:
   - Normalization with mean=0.1307, std=0.3081

## Analysis
The model architecture is optimized by:
1. Using fewer channels in initial layers (16) to reduce parameters
2. Implementing 1x1 convolutions for channel reduction
3. Using Global Average Pooling to eliminate fully connected layers
4. Maintaining consistent dropout (5%) throughout the network

The model shows excellent generalization with test accuracy (99.40%) slightly higher than training accuracy (99.21%), indicating no overfitting. The architecture successfully balances model size and performance, achieving state-of-the-art accuracy with just 13.8K parameters.

## Training Logs
```
Epoch: 1
Training: Average loss: 0.13, Accuracy: 58642/60000 (97.74%)
Test set: Average loss: 0.05, Accuracy: 9892/10000 (98.92%)

...

Epoch: 20
Training: Average loss: 0.03, Accuracy: 59526/60000 (99.21%)
Test set: Average loss: 0.02, Accuracy: 9940/10000 (99.40%)
``` 