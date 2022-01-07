SharpNet is an Open-source Deep Learning library written in C# 8.0.

It supports:
 - Residual Networks [v1](https://arxiv.org/pdf/1512.03385.pdf), [v2](https://arxiv.org/pdf/1603.05027.pdf), [WideResNet](https://arxiv.org/pdf/1605.07146.pdf) and [EfficientNet](https://arxiv.org/pdf/1905.11946.pdf)
 - [DenseNet](https://arxiv.org/pdf/1608.06993.pdf)
 - BatchNorm / Conv1D / Conv2D / Dense / Dropout / Embedding / GRU / LSTM / Pooling / RNN / Shortcut / SimpleRNN layers
 - Elu / Relu / Leaky Relu / Sigmoid / Softmax / Swish / Tanh activations
 - SGD & Adam optimizers
 - Image Data Augmentation (with Cutout/CutMix/Mixup)
 - Ensemble Learning
 
It can be run both on GPU (using NVIDIA cuDNN) and on the CPU (using MKL Blas).

It is targeted to make a good use of the GPU (even if it is not currently as fast as MxNet) :
 - on ResNet18 v1, it is between 1.5x (batch size = 128) and 3x time (batch size = 32) faster then TensorFlow 1.x

It requires:
- [Visual Studio 2019](https://visualstudio.microsoft.com/downloads/)
- [DotNetCore 5.0](https://dotnet.microsoft.com/download/dotnet-core/5.0)
- [CUDA Toolkit 10.1,  10.2 or 11.0](https://developer.nvidia.com/cuda-downloads)
- [CuDNN 8.0](https://developer.nvidia.com/rdp/cudnn-download)
- [Intel MKL](https://software.intel.com/en-us/mkl)
