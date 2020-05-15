using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.Data;
using SharpNet.Networks;

namespace SharpNet.Layers
{
    public class FlattenLayer : Layer
    {
        public FlattenLayer(Network network, string layerName = "") : base(network, layerName)
        {
        }

        #region forward and backward propagation
        public override void ForwardPropagation(List<Tensor> allX, Tensor y, bool isTraining)
        {
            Debug.Assert(allX.Count == 1);
            allX[0].CopyTo(y);
        }
        public override void BackwardPropagation(List<Tensor> allX_NotUsed, Tensor y_NotUsed, Tensor dy, List<Tensor> dx)
        {
            Debug.Assert(allX_NotUsed.Count == 0);
            Debug.Assert(y_NotUsed == null);
            Debug.Assert(dx.Count == 1);
            if (PrevLayer.IsInputLayer)
            {
                return;
            }
            dy.CopyTo(dx[0]);
        }
        public override bool OutputNeededForBackwardPropagation => false;
        public override bool InputNeededForBackwardPropagation => false;

        #endregion

        #region serialization
        public static FlattenLayer Deserialize(IDictionary<string, object> serialized, Network network)
        {
            return new FlattenLayer(network, (string)serialized[nameof(LayerName)]);
        }
        public override void AddToOtherNetwork(Network otherNetwork) { AddToOtherNetwork(otherNetwork, Deserialize); }
        #endregion

        public override int[] OutputShape(int batchSize) {return new []{batchSize, PrevLayer.n_x};}
    }
}
