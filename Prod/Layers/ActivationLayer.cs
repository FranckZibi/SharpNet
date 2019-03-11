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
        public override string Serialize()
        {
            return RootSerializer().Add(nameof(ActivationFunction), (int)ActivationFunction).ToString();
        }
        public static ActivationLayer Deserialize(IDictionary<string, object> serialized, Network network)
        {
            return new ActivationLayer((cudnnActivationMode_t)serialized[nameof(ActivationFunction)], network);
        }
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
        public override string SummaryName()
        {
            return "Activation("+ ToString(ActivationFunction)+ ")";
        }
        private static string ToString(cudnnActivationMode_t activationFunction)
        {
            return activationFunction.ToString().Replace("CUDNN_ACTIVATION_", "");
        }
    }
}
