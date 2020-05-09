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
        private readonly double _alphaActivation;
        #endregion

        #region public fields and properties
        public cudnnActivationMode_t ActivationFunction { get; }
        #endregion

        //No need to configure the number of channels by filter: it is always the same as in previous layer
        public ActivationLayer(cudnnActivationMode_t activationFunctionType, double alphaActivation, Network network, string layerName) : base(network, layerName)
        {
            _alphaActivation = alphaActivation;
            ActivationFunction = activationFunctionType;
        }

        #region forward and backward propagation
        public override void ForwardPropagation(List<Tensor> allX, Tensor y, bool isTraining)
        {
            Debug.Assert(allX.Count == 1);
            StartForwardTimer(Type()+">"+ToString(ActivationFunction), isTraining);
            allX[0].ActivationForward(ActivationFunction, _alphaActivation, y);
            StopForwardTimer(Type()+">"+ToString(ActivationFunction), isTraining);
        }
        public override void BackwardPropagation(List<Tensor> allX, Tensor y, Tensor dy, List<Tensor> dx)
        {
            Debug.Assert(allX.Count == 1);
            Debug.Assert(dx.Count == 1);

            if (PrevLayer.IsInputLayer)
            {
                //no need to compute dy if previous Layer is the input layer
                return;
            }
            StartBackwardTimer(Type() + ">" + ToString(ActivationFunction));
            //we compute dx
            if (IsOutputLayer)
            {
                dy.CopyTo(dx[0]);
            }
            else
            {
                y.ActivationBackward(dy, allX[0], ActivationFunction, _alphaActivation, dx[0]);
            }
            StopBackwardTimer(Type() + ">" + ToString(ActivationFunction));
        }
        #endregion

        #region serialization
        public override string Serialize()
        {
            return RootSerializer()
                .Add(nameof(ActivationFunction), (int)ActivationFunction)
                .Add(nameof(_alphaActivation), _alphaActivation)
                .ToString();
        }
        public static ActivationLayer Deserialize(IDictionary<string, object> serialized, Network network)
        {
            return new ActivationLayer(
                (cudnnActivationMode_t)serialized[nameof(ActivationFunction)],
                (double)serialized[nameof(_alphaActivation)],
                network,
                (string)serialized[nameof(LayerName)]);
        }
        public override void AddToOtherNetwork(Network otherNetwork) { AddToOtherNetwork(otherNetwork, Deserialize); }
        #endregion
 
        private static string ToString(cudnnActivationMode_t activationFunction)
        {
            return activationFunction.ToString().Replace("CUDNN_ACTIVATION_", "");
        }
    }
}
