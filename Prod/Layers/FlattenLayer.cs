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
        public override void BackwardPropagation(List<Tensor> allX, Tensor y, Tensor dy, List<Tensor> dx)
        {
            Debug.Assert(dx.Count == 1);
            if (PrevLayer.IsInputLayer)
            {
                return;
            }
            dy.CopyTo(dx[0]);
        }
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
