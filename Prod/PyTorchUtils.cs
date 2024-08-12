using SharpNet.Data;
using SharpNet.GPU;

namespace SharpNet;

public static class PyTorchUtils
{

    public static string ToPytorch(cudnnActivationMode_t activationFunction, Tensor activationParameter = null)
    {
        switch (activationFunction)
        {
            case cudnnActivationMode_t.CUDNN_ACTIVATION_RELU:
                return "torch.nn.ReLU()";
            case cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID:
                return "torch.nn.Sigmoid()";
            case cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX:
                return "torch.nn.Softmax()";
            case cudnnActivationMode_t.CUDNN_ACTIVATION_LEAKY_RELU:
                return "torch.nn.LeakyReLU(negative_slope=" + activationParameter.ContentAsFloatArray()[0] + ")";
            case cudnnActivationMode_t.CUDNN_ACTIVATION_ELU:
                return "torch.nn.ELU()";
            case cudnnActivationMode_t.CUDNN_ACTIVATION_SWISH:
                return "torch.nn.SiLU()";
            case cudnnActivationMode_t.CUDNN_ACTIVATION_TANH:
                return "torch.nn.Tanh()";
            case cudnnActivationMode_t.CUDNN_ACTIVATION_IDENTITY:
                return "torch.nn.Identity()";
            default:
                return "TODO: torch.nn." + activationFunction + "()";
        }
    }


}