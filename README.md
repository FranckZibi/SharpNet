SharpNet is an Open-source Deep Learning library written in C# 7.0.

It supports:
 - Residual Networks [v1](https://arxiv.org/pdf/1512.03385.pdf) and [v2](https://arxiv.org/pdf/1603.05027.pdf)
 - [DenseNet](https://arxiv.org/pdf/1608.06993.pdf)
 - Dropout / BatchNorm / Conv / Pooling / Dense / Concatenate / Shortcut layers
 - Elu / Relu / Sigmoid / Softmax activations
 - SGD & Adam optimizers
 - Image Data Augmentation
 
It can be run both on GPU (using NVIDIA cuDNN) and on the CPU (using MKL Blas).

It is targeted to make a good use of the GPU (even if it is not currently as fast as MxNet) :
 - on ResNet18 v1, it is between 1.5x (batch size = 128) and 3x time (batch size = 32) faster then TensorFlow

It requires:
- Visual Studio 2017 or 2019 (https://visualstudio.microsoft.com/downloads/)
- CUDA Toolkit 10.0 or 10.1 (https://developer.nvidia.com/cuda-downloads)
- CuDNN 7.5 (https://developer.nvidia.com/rdp/cudnn-download)
- Intel MKL (https://software.intel.com/en-us/mkl)

Next Targets:
- ~~Add ResNet v2 support~~ => DONE
- ~~Add Dense Network support~~ => DONE
- Add multi GPU support
 - Add RNN / LSTM support
 - Improve memory efficiency for gradients
 - Add Wide ResNet / Wide DenseNet support
 - Improve Image Data Augmentation (with rotation)
