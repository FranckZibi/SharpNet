using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.Data;
using SharpNet.Networks;

namespace SharpNet.Layers
{
    /// <summary>
    /// Layer that computes: Beta * PrevLayer + Alpha
    /// </summary>
    public class LinearFunctionLayer : Layer
    {
        private readonly float Beta;
        private readonly float Alpha;

        public LinearFunctionLayer(float beta, float alpha, Network network, string layerName = "") : base(network, layerName)
        {
            Beta = beta;
            Alpha = alpha;
        }
        #region forward and backward propagation
        public override void ForwardPropagation(List<Tensor> allX, Tensor y, bool isTraining)
        {
            Debug.Assert(allX.Count == 1);
            y.LinearFunction(Beta, allX[0], Alpha);
        }
        public override void BackwardPropagation(List<Tensor> allX_NotUsed, Tensor y_NotUsed, Tensor dy, List<Tensor> allDx)
        {
            Debug.Assert(allX_NotUsed.Count == 0);
            Debug.Assert(y_NotUsed == null);
            Debug.Assert(allDx.Count == 1);
            Debug.Assert(allDx[0].SameShape(dy));
            allDx[0].LinearFunction(Beta , dy, 0);
        }
        public override bool OutputNeededForBackwardPropagation => false;
        public override bool InputNeededForBackwardPropagation => false;
        #endregion

        #region serialization
        public override string Serialize()
        {
            return RootSerializer()
                .Add(nameof(Beta), Beta)
                .Add(nameof(Alpha), Alpha)
                .ToString();
        }
        public static LinearFunctionLayer Deserialize(IDictionary<string, object> serialized, Network network)
        {
            return new LinearFunctionLayer(
                (float)serialized[nameof(Beta)], 
                (float)serialized[nameof(Alpha)], 
                network, 
                (string)serialized[nameof(LayerName)]);
        }
        public override void AddToOtherNetwork(Network otherNetwork) { AddToOtherNetwork(otherNetwork, Deserialize); }
        #endregion
    }
}