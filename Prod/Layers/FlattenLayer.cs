using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.Data;
using SharpNet.Networks;

namespace SharpNet.Layers
{
    public class FlattenLayer : Layer
    {
        public override Tensor y { get; protected set; }
        public FlattenLayer(Network network, string layerName = "") : base(network, layerName)
        {
        }

        public override Layer Clone(Network newNetwork) { return new FlattenLayer(this, newNetwork); }
        private FlattenLayer(FlattenLayer toClone, Network newNetwork) : base(toClone, newNetwork) {}

        #region serialization
        public FlattenLayer(IDictionary<string, object> serialized, Network network) : base(serialized, network)
        {
        }
        #endregion
        public override void ForwardPropagation(bool isTraining)
        {
            Allocate_y_if_necessary();
            var x = PrevLayer.y;
            x.CopyTo(y);
        }

        public override void BackwardPropagation(Tensor dy, List<Tensor> dx)
        {
            Debug.Assert(dx.Count == 1);
            if (PrevLayer.IsInputLayer)
            {
                return;
            }
            dy.CopyTo(dx[0]);
        }
        public override int[] OutputShape(int batchSize) {return new []{batchSize, PrevLayer.n_x};}
    }
}
