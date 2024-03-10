using System;
using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.Data;
using SharpNet.GPU;
using SharpNet.Networks;

namespace SharpNet.Layers
{
    public class ActivationLayer : Layer
    {
        #region private fields
        private readonly Tensor _activationParameter;
        #endregion

        #region public fields and properties
        public cudnnActivationMode_t ActivationFunction { get; }
        #endregion

        //No need to configure the number of channels by filter: it is always the same as in previous layer
        public ActivationLayer(cudnnActivationMode_t activationFunctionType, Tensor activationParameter, Network network, string layerName) : base(network, layerName)
        {
            if (activationParameter != null && activationParameter.UseGPU != Network.UseGPU)
            {
                activationParameter = Network.UseGPU ? activationParameter.ToGPU<float>(Network.GpuWrapper) : activationParameter.ToCpuFloat();
            }
            _activationParameter = activationParameter;
            ActivationFunction = activationFunctionType;
        }

        #region forward and backward propagation
        public override void ForwardPropagation(List<Tensor> allX, Tensor y, bool isTraining)
        {
            Debug.Assert(allX.Count == 1);
            StartForwardTimer(LayerType()+">"+ToString(ActivationFunction), isTraining);
            allX[0].ActivationForward(ActivationFunction, _activationParameter, y);
            StopForwardTimer(LayerType()+">"+ToString(ActivationFunction), isTraining);
        }
        public override void BackwardPropagation(List<Tensor> allX, Tensor y, Tensor dy, List<Tensor> dx)
        {
            Debug.Assert(dx.Count == 1);

            if (!InputNeededForBackwardPropagation)
            {
                Debug.Assert(allX.Count == 0);
                allX.Add(y);
            }
            else
            {
                Debug.Assert(allX.Count == 1);
            }

            if (PrevLayer.IsInputLayer)
            {
                //no need to compute dy if previous Layer is the input layer
                return;
            }
            StartBackwardTimer(LayerType() + ">" + ToString(ActivationFunction));
            //we compute dx
            if (IsOutputLayer && Network.Sample.GetLoss() != EvaluationMetricEnum.Huber)
            {
                dy.CopyTo(dx[0]);
            }
            else
            {
                dx[0].ActivationBackward(ActivationFunction, _activationParameter, dy, allX[0], y);
            }
            StopBackwardTimer(LayerType() + ">" + ToString(ActivationFunction));
        }
        /// <summary>
        /// true if the input feature map 'x' is needed to compute the backward propagation of current layer
        /// </summary>
        public override bool InputNeededForBackwardPropagation => ActivationFunction==cudnnActivationMode_t.CUDNN_ACTIVATION_SWISH||ActivationFunction == cudnnActivationMode_t.CUDNN_ACTIVATION_ELU;

        #endregion

        #region serialization
        public override string Serialize()
        {
            return RootSerializer()
                .Add(nameof(ActivationFunction), (int)ActivationFunction)
                .Add(nameof(_activationParameter), _activationParameter)
                .ToString();
        }
        public static ActivationLayer Deserialize(IDictionary<string, object> serialized, Network network)
        {
            return new ActivationLayer(
                (cudnnActivationMode_t)serialized[nameof(ActivationFunction)],
                serialized.ContainsKey(nameof(_activationParameter)) ?(Tensor)serialized[nameof(_activationParameter)]:null,
                network,
                (string)serialized[nameof(LayerName)]);
        }
        public override void AddToOtherNetwork(Network otherNetwork) { AddToOtherNetwork(otherNetwork, Deserialize); }
        #endregion

        public override string LayerType()
        {
            switch (ActivationFunction)
            {
                case cudnnActivationMode_t.CUDNN_ACTIVATION_LEAKY_RELU: return "LeakyReLU";
                case cudnnActivationMode_t.CUDNN_ACTIVATION_RELU: return "ReLU";
                case cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX: return "Softmax";
                case cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX_WITH_HIERARCHY: return "Softmax_Hierarchy";
                case cudnnActivationMode_t.CUDNN_ACTIVATION_SWISH: return "Swish";
                default: return ToString(ActivationFunction);
            }
        }
        #region PyTorch support
        public override void ToPytorchModule(List<string> constructorLines, List<string> forwardLines)
        {
            //special case : for cross entropy loss, we do need to use the softmax function in PyTorch torch.nn.Module
            if (Network.Sample.GetLoss() == EvaluationMetricEnum.CategoricalCrossentropy && LayerIndex == (Layers.Count-1))
            {
                return;
            }
            constructorLines.Add("self." + LayerName + " = "+ ToPytorchConstructor());
            UpdateForwardLines(forwardLines);
        }
        private string ToPytorchConstructor()
        {
            switch (ActivationFunction)
            {
                case cudnnActivationMode_t.CUDNN_ACTIVATION_RELU:
                    return "torch.nn.ReLU()";
                default:
                    throw new NotImplementedException(ActivationFunction.ToString());
            }
        }

        #endregion

        private static string ToString(cudnnActivationMode_t activationFunction)
        {
            return activationFunction.ToString().Replace("CUDNN_ACTIVATION_", "");
        }
    }
}
