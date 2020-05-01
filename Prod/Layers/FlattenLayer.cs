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
        public FlattenLayer(IDictionary<string, object> serialized, Network network) : base(serialized, network)
        {
        }
        #endregion

        #region clone layer

        public override void AddToOtherNetwork(Network otherNetwork)
        {
            otherNetwork.Layers.Add(new FlattenLayer(otherNetwork, LayerName));
        }
        #endregion

        public override int[] OutputShape(int batchSize) {return new []{batchSize, PrevLayer.n_x};}
    }
}
