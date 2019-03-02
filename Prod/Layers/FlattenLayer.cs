using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.Data;

namespace SharpNet
{
    public class FlattenLayer : Layer
    {
        public override Tensor y { get; protected set; }
        public override Tensor dy { get; protected set; }

        public FlattenLayer(Network network) : base(network)
        {
        }
        public override string Serialize() {return RootSerializer().ToString();}
        public static FlattenLayer Deserialize(Network network)
        {
            return new FlattenLayer(network);
        }
        public override void ForwardPropagation(bool isTraining)
        {
            Allocate_y_dy_if_necessary();
            var x = PrevLayer.y;
            x.CopyTo(y);
        }
        public override void BackwardPropagation()
        {
            Debug.Assert(y.SameShape(dy));
            if (PrevLayer.IsInputLayer)
            {
                return;
            }
            var dx = PrevLayer.dy;
            dy.CopyTo(dx);
        }
        public override int[] OutputShape(int batchSize) {return new []{batchSize, PrevLayer.n_x};}
        public override List<Tensor> TensorsIndependantOfBatchSize => new List<Tensor>();
    }
}
