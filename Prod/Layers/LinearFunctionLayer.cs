using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.Data;
using SharpNet.Networks;

namespace SharpNet.Layers
{
    /// <summary>
    /// Layer that computes: a * PrevLayer + b
    /// </summary>
    public class LinearFunctionLayer : Layer
    {
        private readonly float _a;
        private readonly float _b;

        public LinearFunctionLayer(float a, float b, Network network, string layerName = "") : base(network, layerName)
        {
            _a = a;
            _b = b;
        }
        #region forward and backward propagation
        public override void ForwardPropagation(List<Tensor> allX, Tensor y, bool isTraining)
        {
            Debug.Assert(allX.Count == 1);
            y.LinearFunction(_a, allX[0], _b);
        }
        public override void BackwardPropagation(List<Tensor> allX_NotUsed, Tensor y_NotUsed, Tensor dy, List<Tensor> allDx)
        {
            Debug.Assert(allX_NotUsed.Count == 0);
            Debug.Assert(y_NotUsed == null);
            Debug.Assert(allDx.Count == 1);
            Debug.Assert(allDx[0].SameShape(dy));
            allDx[0].LinearFunction(_a , dy, 0);
        }
        public override bool OutputNeededForBackwardPropagation => false;
        public override bool InputNeededForBackwardPropagation => false;
        #endregion

        #region serialization
        public override string Serialize()
        {
            return RootSerializer()
                .Add(nameof(_a), _a)
                .Add(nameof(_b), _b)
                .ToString();
        }
        public static LinearFunctionLayer Deserialize(IDictionary<string, object> serialized, Network network)
        {
            return new LinearFunctionLayer(
                (float)serialized[nameof(_a)], 
                (float)serialized[nameof(_b)], 
                network, 
                (string)serialized[nameof(LayerName)]);
        }
        public override void AddToOtherNetwork(Network otherNetwork) { AddToOtherNetwork(otherNetwork, Deserialize); }
        #endregion
    }
}