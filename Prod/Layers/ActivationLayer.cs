using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.Data;
using SharpNet.GPU;
using SharpNet.Networks;

namespace SharpNet.Layers
{
    public class ActivationLayer : Layer
    {
        public cudnnActivationMode_t ActivationFunction { get; }

        //No need to configure the number of channels by filter: it is always the same as in previous layer
        public ActivationLayer(cudnnActivationMode_t activationFunctionType, Network network, string layerName) : base(network, layerName)
        {
            ActivationFunction = activationFunctionType;
        }

        public override Layer Clone(Network newNetwork) { return new ActivationLayer(this, newNetwork); }
        private ActivationLayer(ActivationLayer toClone, Network newNetwork) : base(toClone, newNetwork)
        {
            ActivationFunction = toClone.ActivationFunction;
        }

        public override void ForwardPropagation(List<Tensor> allX, Tensor y, bool isTraining)
        {
            Debug.Assert(allX.Count == 1);
            Network.StartTimer(Type()+">"+ToString(ActivationFunction), isTraining ? Network.ForwardPropagationTrainingTime : Network.ForwardPropagationInferenceTime);
            allX[0].ActivationForward(ActivationFunction, y);
            Network.StopTimer(Type()+">"+ToString(ActivationFunction), isTraining ? Network.ForwardPropagationTrainingTime : Network.ForwardPropagationInferenceTime);
        }
        public override bool Equals(Layer b, double epsilon, string id, ref string errors)
        {
            if (!base.Equals(b, epsilon, id, ref errors))
            {
                return false;
            }
            var other = (ActivationLayer)b;
            var equals = true;
            equals &= Utils.Equals(ActivationFunction, other.ActivationFunction, id + ":ActivationFunction", ref errors);
            return equals;
        }

        #region serialization
        public override string Serialize()
        {
            return RootSerializer().Add(nameof(ActivationFunction), (int)ActivationFunction).ToString();
        }
        public ActivationLayer(IDictionary<string, object> serialized, Network network) : base(serialized, network)
        {
            ActivationFunction = (cudnnActivationMode_t)serialized[nameof(ActivationFunction)];
        }
        #endregion
        public override void BackwardPropagation(List<Tensor> allX, Tensor y, Tensor dy, List<Tensor> dx)
        {
            Debug.Assert(allX.Count == 1);
            Debug.Assert(dx.Count == 1);
            
            if (PrevLayer.IsInputLayer)
            {
                //no need to compute dy if previous Layer is the input layer
                return;  
            }

            Network.StartTimer(Type()+">"+ToString(ActivationFunction), Network.BackwardPropagationTime);
            //we compute dx
            if (IsOutputLayer)
            {
                dy.CopyTo(dx[0]);
            }
            else
            {
                y.ActivationBackward(dy, allX[0], ActivationFunction, dx[0]);
            }
            Network.StopTimer(Type()+">"+ToString(ActivationFunction), Network.BackwardPropagationTime);
        }
        public override void Dispose()
        {
            EmbeddedTensors.ForEach(x => x?.Dispose());
        }
        private static string ToString(cudnnActivationMode_t activationFunction)
        {
            return activationFunction.ToString().Replace("CUDNN_ACTIVATION_", "");
        }
    }
}
