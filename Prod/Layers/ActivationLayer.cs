using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.Data;
using SharpNet.GPU;

namespace SharpNet
{
    public class ActivationLayer : Layer
    {
        #region Private fields
        public override Tensor y { get; protected set; }    // (batchSize, C, H, W)
        public override Tensor dy { get; protected set; }   //same as 'y'
        #endregion

        public cudnnActivationMode_t ActivationFunction { get; }

        //No need to configure the number of channels by filter: it is always the same as in previous layer
        public ActivationLayer(cudnnActivationMode_t activationFunctionType, Network network) : base(network)
        {
            ActivationFunction = activationFunctionType;
        }
        public override void ForwardPropagation(bool isTraining)
        {
            Allocate_y_dy_if_necessary();
            var x = PrevLayer.y;
            x.ActivationForward(ActivationFunction, y);
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
        public override void BackwardPropagation()
        {
            //At this stage, we already know dy. We want to compute dx by backward propagation
            Debug.Assert(y.SameShape(dy));

            //we update dy if necessary (shortcut connection to a future layer)
            Update_dy_With_GradientFromShortcutIdentityConnection();

            //no need to compute dy if previous Layer it is the input layer
            if (PrevLayer.IsInputLayer)
            {
                return; 
            }

            //we compute dx
            var x = PrevLayer.y;
            var dx = PrevLayer.dy;
            if (IsOutputLayer)
            {
                dy.CopyTo(dx);
            }
            else
            {
                y.ActivationBackward(dy, x, ActivationFunction, dx);
            }
        }
        public override void Dispose()
        {
            EmbeddedTensors.ForEach(x => x?.Dispose());
        }

        public override string SummaryName() { return "activation_" + (1 + NbLayerOfSameTypeBefore()); }
        public override string Type() { return ToString(ActivationFunction); }
        private static string ToString(cudnnActivationMode_t activationFunction)
        {
            return activationFunction.ToString().Replace("CUDNN_ACTIVATION_", "");
        }
    }
}
